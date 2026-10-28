"""Gatheral raw SVI, one slice per expiry: the benchmark for the arbitrage-free models."""

from __future__ import annotations

import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.errors import ValidationError
from torch_pricer.market.surface.base import SliceQuotes, VolSurface, durrleman_g
from torch_pricer.market.surface.ssvi import _inverse_softplus, _logit

#: Lee's moment bound on a wing's slope in total variance: ``b (1 + |rho|) <= 2``.
_LEE = 2.0
#: Where the butterfly penalty looks, in log-moneyness, on every slice.
_PENALTY_GRID = torch.linspace(-1.5, 1.5, 61, dtype=torch.float64)
#: Weight of the butterfly penalty against the loss, which is ~1e-4 at a vol
#: point of error: a density of -0.01 costs as much as that whole error.
_PENALTY_WEIGHT = 1.0
#: Fewer quotes than this and a slice starts from an SSVI shape, not a fit:
#: five parameters need at least five quotes.
_MIN_QUOTES_TO_FIT = 5


def _svi(k: Tensor, a, b, rho, m, sigma) -> Tensor:
    x = k - m
    return a + b * (rho * x + torch.sqrt(x**2 + sigma**2))


def _quasi_explicit(q: SliceQuotes) -> tuple[float, float, float, float, float]:
    """Raw SVI ``(a, b, rho, m, sigma)`` for one slice, from a grid over ``(m, sigma)``.

    Zeliade's quasi-explicit idea: for fixed ``m`` and ``sigma``, ``w`` is
    linear in ``(a, b rho sigma, b sigma)``, a weighted least-squares solve.
    Each candidate is pushed inside the bounds, then judged by its implied-vol
    error; only a starting point, so the grid is coarse.
    """
    t, k, iv, weight = q.expiry_time, q.k, q.iv, q.weight
    w_mkt = iv**2 * t
    lsq = np.sqrt(weight) / (2 * iv * t)  # an iv error is a w error over 2 sigma T
    best, best_err = None, np.inf
    for m in np.linspace(-0.5, 0.5, 11) * np.sqrt(t):
        for sigma in np.geomspace(0.02, 2.0, 12) * np.sqrt(t):
            y = (k - m) / sigma
            design = np.column_stack([np.ones_like(y), y, np.sqrt(y**2 + 1)])
            (a, d, c), *_ = np.linalg.lstsq(design * lsq[:, None], w_mkt * lsq, rcond=None)
            b = max(c / sigma, 1e-6)
            rho = float(np.clip(d / (b * sigma), -0.99, 0.99))
            b = min(b, 0.99 * _LEE / (1 + abs(rho)))
            floor = 1e-3 * q.atm_total_variance - b * sigma * np.sqrt(1 - rho**2)
            a = max(a, floor)  # minimum variance positive
            w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
            err = float(np.sum(weight * (np.sqrt(np.maximum(w, 1e-12) / t) - iv) ** 2))
            if err < best_err:
                best, best_err = (a, b, rho, m, sigma), err
    return best


def _ssvi_shaped(theta: float, rho: float = -0.5) -> tuple[float, float, float, float, float]:
    """The SSVI slice at ATM variance ``theta`` with ``psi = sqrt(theta)``, as raw SVI."""
    phi = 1 / np.sqrt(theta)
    return theta / 2 * (1 - rho**2), theta * phi / 2, rho, -rho / phi, np.sqrt(1 - rho**2) / phi


class SVISurface(VolSurface):
    """Raw SVI per expiry: ``w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))``.

    The parametrisation keeps each slice admissible -- ``|rho| < 1``,
    ``sigma > 0``, ``0 < b (1 + |rho|) < 2`` (Lee's bound) and a positive
    minimum ``a + b sigma sqrt(1 - rho^2)`` -- but does not rule out
    butterfly arbitrage, which is a :meth:`penalty` on Durrleman's ``g``, nor
    calendar arbitrage between slices, which is only checked and reported.

    Between expiries ``w`` is linear in ``T`` at fixed ``k``: free of calendar
    arbitrage wherever the two slices do not cross. Before the first,
    ``w_1(k) T / T_1``, which keeps the butterfly condition (``g`` is concave
    in the scale, and non-negative at both ends). After the last,
    ``w_n(k) + w_n(0) (T - T_n) / T_n``: constant ATM vol, and the wings'
    slopes held, so Lee's bound still holds; the butterfly condition there is
    checked, not guaranteed.
    """

    def __init__(self, expiry_times, a, b, rho, m, sigma, as_of: dt.date):
        super().__init__()
        times = torch.as_tensor(expiry_times, dtype=torch.float64).flatten()
        a, b, rho, m, sigma = (torch.as_tensor(x, dtype=torch.float64).flatten() for x in (a, b, rho, m, sigma))
        n = times.numel()
        if n == 0 or not all(x.numel() == n for x in (a, b, rho, m, sigma)):
            raise ValidationError("need one (a, b, rho, m, sigma) per expiry, and at least one")
        if not bool((times[1:] > times[:-1]).all()) or not bool((times > 0).all()):
            raise ValidationError("expiry times must be positive and strictly increasing")
        lee = b * (1 + rho.abs()) / _LEE
        minimum = a + b * sigma * torch.sqrt(1 - rho**2)
        if not (bool((rho.abs() < 1).all()) and bool((sigma > 0).all())
                and bool(((lee > 0) & (lee < 1)).all()) and bool((minimum > 0).all())):
            raise ValidationError("SVI slices violate |rho| < 1, sigma > 0, Lee's bound or w > 0")

        self.register_buffer("expiry_times", times)
        self.raw_rho = nn.Parameter(torch.atanh(rho))
        self.raw_b = nn.Parameter(_logit(lee))
        self.raw_sigma = nn.Parameter(_inverse_softplus(sigma))
        self.m = nn.Parameter(m.clone())
        self.raw_minimum = nn.Parameter(_inverse_softplus(minimum))
        self._ref = as_of

    @classmethod
    def initial(cls, slices: tuple[SliceQuotes, ...], as_of: dt.date) -> SVISurface:
        """Each slice calibrated alone, or an SSVI shape if it has few quotes.

        The fit's loss -- weighted squared iv error plus the butterfly
        penalty -- is a sum over slices, so fitting them one at a time reaches
        its minimum and leaves the global fit only a polish; one badly scaled
        L-BFGS over every slice at once does not. Each slice tries two starts,
        the quasi-explicit fit and the arbitrage-free SSVI shape, and keeps the
        better: from a start deep in butterfly arbitrage, escaping the penalty
        can strand a slice far from its quotes.
        """
        fitted = [len(q.k) >= _MIN_QUOTES_TO_FIT for q in slices]
        starts = [_quasi_explicit(q) if fit else _ssvi_shaped(q.atm_total_variance)
                  for q, fit in zip(slices, fitted)]
        surface = cls([q.expiry_time for q in slices], *np.array(starts).T, as_of)
        for i, (q, start, fit) in enumerate(zip(slices, starts, fitted)):
            if fit:
                raw = cls._fit_slice(q, start, as_of)
                # Copied raw, since a fitted value can sit within rounding of its bound.
                with torch.no_grad():
                    for mine, theirs in zip(surface.parameters(), raw):
                        mine[i] = theirs[0]
        return surface

    @classmethod
    def _fit_slice(cls, q: SliceQuotes, start, as_of: dt.date) -> list[Tensor]:
        """One slice's raw parameters, minimising its share of the fit's loss."""
        best, best_loss = None, np.inf
        for guess in (start, _ssvi_shaped(q.atm_total_variance)):
            raw, loss = cls._fit_slice_from(q, guess, as_of)
            if loss < best_loss:
                best, best_loss = raw, loss
        return best if best is not None else list(cls([q.expiry_time], *([x] for x in start), as_of).parameters())

    @classmethod
    def _fit_slice_from(cls, q: SliceQuotes, start, as_of: dt.date) -> tuple[list[Tensor], float]:
        one = cls([q.expiry_time], *([x] for x in start), as_of)
        k, iv = torch.tensor(q.k), torch.tensor(q.iv)
        share = float(q.weight.sum())
        weight = torch.tensor(q.weight) / share  # and the penalty over share: the same minimum
        optimizer = torch.optim.LBFGS(one.parameters(), max_iter=200, line_search_fn="strong_wolfe",
                                      tolerance_grad=0.0, tolerance_change=1e-14, history_size=50)

        def closure():
            optimizer.zero_grad()
            loss = (weight * (one.implied_vol(k, q.expiry_time) - iv) ** 2).sum() + one.penalty() / share
            loss.backward()
            return loss

        optimizer.step(closure)
        raw = [x.detach() for x in one.parameters()]
        loss = float(closure().detach())
        return raw, loss if np.isfinite(loss) and all(bool(torch.isfinite(x).all()) for x in raw) else np.inf

    # -- parameters, constrained ------------------------------------------
    @property
    def rho(self) -> Tensor:
        return torch.tanh(self.raw_rho)

    @property
    def b(self) -> Tensor:
        return _LEE / (1 + self.rho.abs()) * torch.sigmoid(self.raw_b)

    @property
    def sigma(self) -> Tensor:
        return nn.functional.softplus(self.raw_sigma)

    @property
    def a(self) -> Tensor:
        """From the slice's minimum variance, which is kept positive."""
        return nn.functional.softplus(self.raw_minimum) - self.b * self.sigma * torch.sqrt(1 - self.rho**2)

    def slices(self) -> tuple[Tensor, ...]:
        """``(a, b, rho, m, sigma)``, one entry per expiry."""
        return self.a, self.b, self.rho, self.m, self.sigma

    def constraints(self) -> dict[str, dict[str, float]]:
        a, b, rho, m, sigma = (x.detach() for x in self.slices())
        return {
            "|rho| < 1": {"value": float(rho.abs().max()), "bound": 1.0},
            "b (1 + |rho|) <= 2": {"value": float((b * (1 + rho.abs())).max()), "bound": _LEE},
            "min w > 0": {"value": float((a + b * sigma * torch.sqrt(1 - rho**2)).min()), "bound": 0.0},
        }

    def describe(self) -> dict[str, float | list[float]]:
        return {name: x.detach().tolist() for name, x in zip(("a", "b", "rho", "m", "sigma"), self.slices())}

    def penalty(self) -> Tensor:
        """Squared butterfly violations on every slice, from ``g`` in closed form."""
        a, b, rho, m, sigma = (x[:, None] for x in self.slices())
        k = _PENALTY_GRID[None, :]
        x = k - m
        root = torch.sqrt(x**2 + sigma**2)
        w = a + b * (rho * x + root)
        g = durrleman_g(k, w, b * (rho + x / root), b * sigma**2 / root**3)
        return _PENALTY_WEIGHT * torch.relu(-g).pow(2).mean(dim=1).sum()

    # -- the surface ----------------------------------------------------------
    def total_variance(self, k, expiry) -> Tensor:
        t, k = self._as_tensor(expiry), self._as_tensor(k)
        times, params = self.expiry_times, self.slices()
        n = times.numel()

        def at(i):
            return _svi(k, *(p[i] for p in params))

        first, last = at(0), at(n - 1)
        before = first * t / times[0]
        after = last + _svi(torch.zeros_like(k), *(p[n - 1] for p in params)) * (t - times[-1]) / times[-1]
        if n == 1:
            return torch.where(t <= times[0], before, after)
        idx = torch.searchsorted(times, t.detach().contiguous()).clamp(1, n - 1)
        lam = ((t - times[idx - 1]) / (times[idx] - times[idx - 1])).clamp(0, 1)
        inside = (1 - lam) * at(idx - 1) + lam * at(idx)
        return torch.where(t < times[0], before, torch.where(t > times[-1], after, inside))

    @property
    def reference_date(self) -> dt.date:
        return self._ref

    def __repr__(self) -> str:  # pragma: no cover
        return f"SVISurface({self.expiry_times.numel()} expiries)"
