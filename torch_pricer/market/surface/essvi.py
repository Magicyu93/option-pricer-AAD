"""Extended SSVI (Hendriks & Martini, 2019): an SSVI slice per expiry, its own ``rho``."""

from __future__ import annotations

import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.errors import ValidationError
from torch_pricer.market.surface.base import SliceQuotes, VolSurface
from torch_pricer.market.surface.ssvi import _inverse_softplus, _logit, atm_term_structure

#: A wing slope ``psi (1 + |rho|)`` stays under this: Gatheral & Jacquier's first butterfly bound.
_MAX_SLOPE = 4.0
#: Starting fractions are kept this far inside (0, 1), so their logits are finite.
_FRACTION_FLOOR = 1e-6
#: A slice with fewer quotes than this does not get its own starting fit: it
#: takes its neighbours' shape, through the isotonic regressions in ``initial``.
_MIN_QUOTES_TO_FIT = 3


def _isotonic(y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-decreasing fit to ``y``: pool adjacent violators."""
    blocks: list[list[float]] = []  # [mean, weight, count]
    for yi, wi in zip(y, weight):
        blocks.append([float(yi), float(wi), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m2, w2, n2), (m1, w1, n1) = blocks.pop(), blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(n, m) for m, _, n in blocks])


def _fit_slice(q: SliceQuotes) -> tuple[float, float, float]:
    """``(theta, psi, rho)`` of one SSVI slice fitted to its quotes alone, unconstrained.

    Only a starting point: the constraints between slices are imposed afterwards.
    """
    t = q.expiry_time
    k, iv = torch.tensor(q.k), torch.tensor(q.iv)
    weight = torch.tensor(q.weight / q.weight.sum())
    raw = torch.tensor(
        [np.log(q.atm_total_variance), np.arctanh(-0.5), 0.5 * np.log(q.atm_total_variance)],
        dtype=torch.float64, requires_grad=True,
    )
    optimizer = torch.optim.LBFGS([raw], max_iter=200, line_search_fn="strong_wolfe",
                                  tolerance_grad=0.0, tolerance_change=1e-14)

    def closure():
        optimizer.zero_grad()
        theta, rho, psi = raw[0].exp(), torch.tanh(raw[1]), raw[2].exp()
        w = 0.5 * (theta + rho * psi * k + torch.sqrt((psi * k + rho * theta) ** 2 + theta**2 * (1 - rho**2)))
        loss = (weight * ((w / t).sqrt() - iv) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    theta, rho, psi = raw.detach()
    if not bool(torch.isfinite(raw.detach()).all()):
        return q.atm_total_variance, np.sqrt(q.atm_total_variance), -0.5
    return float(theta.exp()), float(psi.exp()), float(torch.tanh(rho))


def _cap(other: Tensor, theta: Tensor) -> Tensor:
    """The largest wing slope ``x`` that ``(x, other)`` admits at ATM variance ``theta``.

    The butterfly bounds in wing slopes: ``max(x, other) < 4`` and
    ``psi max(x, other) <= 4 theta`` with ``psi = (x + other) / 2``. When
    ``x >= other`` the second is ``(x + other) x <= 8 theta``, a quadratic in
    ``x``; otherwise it is linear. The two meet at ``other^2 = 4 theta``.
    """
    above = other**2 <= 4 * theta
    safe = torch.where(above, torch.ones_like(other), other)  # keep 8 theta / 0 out of the graph
    root = 0.5 * (torch.sqrt(other**2 + 32 * theta) - other)
    return torch.minimum(torch.where(above, root, 8 * theta / safe - other), other.new_tensor(_MAX_SLOPE))


def _room(other: Tensor, theta: Tensor, total: Tensor | None) -> Tensor:
    """The cap on one wing slope given the other: butterfly, and ``A + B <= total`` if given."""
    cap = _cap(other, theta)
    return cap if total is None else torch.minimum(cap, total - other)


class ESSVISurface(VolSurface):
    """Total variance at log-moneyness ``k``, slice by slice::

        w(k) = 1/2 (theta + rho psi k + sqrt((psi k + rho theta)^2 + theta^2 (1 - rho^2)))

    the SSVI slice with ``psi = theta phi``, and ``(theta_i, rho_i, psi_i)``
    one set per fitted expiry. Written in the wing slopes ``A = psi (1 + rho)``
    and ``B = psi (1 - rho)``, the asymptotic slopes of ``w`` on the right and
    the left.

    Free of static arbitrage by construction, not by checking:

    * butterfly, per slice (Gatheral & Jacquier, Thm 4.2):
      ``psi (1 + |rho|) < 4`` and ``psi^2 (1 + |rho|) <= 4 theta``;
    * calendar, between slices (Hendriks & Martini, Prop. 3.5, as restated by
      Mingone, 2022): ``theta``, ``A`` and ``B`` increasing in ``T`` -- the
      necessary part -- and ``phi = psi / theta`` non-increasing, which makes
      it sufficient. Without the last, two slices can cross just right of
      the money.

    ``theta`` is cumulative softplus increments, as in SSVI. ``A_i`` then
    ``B_i`` are each the previous slice's plus a sigmoid fraction of the room
    left under the butterfly caps (:func:`_cap`) and under
    ``A + B <= 2 phi_{i-1} theta_i``. The room is never empty: the previous
    slice's pair satisfies every bound at the larger ``theta_i``.

    Between expiries ``theta(T)`` is SSVI's smooth term structure, and ``A``
    and ``B`` are linear in ``theta`` -- the interpolation of Corbetta et al.,
    under which every intermediate slice is admissible too: ``psi max(A, B)``
    is convex in ``theta``, so it stays under ``4 theta`` between two slices
    that are. So ``w`` is only ``C^0`` in ``T`` at the expiries, and a Dupire
    local vol steps there. Before the first expiry ``A`` and ``B`` scale with
    ``theta`` (a frozen smile in implied vol, constant ATM vol); after the last
    they are held, while ``theta`` grows at constant ATM vol.
    """

    def __init__(self, expiry_times, theta, psi, rho, as_of: dt.date):
        super().__init__()
        times = torch.as_tensor(expiry_times, dtype=torch.float64).flatten()
        theta, psi, rho = (torch.as_tensor(x, dtype=torch.float64).flatten() for x in (theta, psi, rho))
        n = times.numel()
        if n == 0 or not all(x.numel() == n for x in (theta, psi, rho)):
            raise ValidationError("need one (theta, psi, rho) per expiry, and at least one")
        if not bool((times[1:] > times[:-1]).all()) or not bool((times > 0).all()):
            raise ValidationError("expiry times must be positive and strictly increasing")
        if not bool((theta[1:] > theta[:-1]).all()) or not bool((theta > 0).all()):
            raise ValidationError("ATM total variance must be positive and strictly increasing")
        if not (bool((psi > 0).all()) and bool((rho.abs() < 1).all())):
            raise ValidationError("need psi > 0 and |rho| < 1")
        target_a, target_b = psi * (1 + rho), psi * (1 - rho)

        self.register_buffer("expiry_times", times)
        self.raw_theta = nn.Parameter(_inverse_softplus(torch.diff(theta, prepend=theta.new_zeros(1))))
        # Fractions of the room left, slice by slice: the inverse of wing_slopes.
        frac_a, frac_b = torch.empty(n, dtype=torch.float64), torch.empty(n, dtype=torch.float64)
        prev_a = prev_b = theta.new_zeros(())
        total = None  # A + B under this from the second slice on: phi does not increase
        for i in range(n):
            frac_a[i], prev_a = self._fraction(target_a[i], prev_a, _room(prev_b, theta[i], total))
            frac_b[i], prev_b = self._fraction(target_b[i], prev_b, _room(prev_a, theta[i], total))
            if i + 1 < n:
                total = (prev_a + prev_b) * theta[i + 1] / theta[i]
        self.raw_a = nn.Parameter(_logit(frac_a))
        self.raw_b = nn.Parameter(_logit(frac_b))
        self._ref = as_of

    @staticmethod
    def _fraction(target: Tensor, prev: Tensor, cap: Tensor) -> tuple[Tensor, Tensor]:
        """The fraction of ``[prev, cap]`` nearest ``target``, and the slope it gives."""
        f = ((target - prev) / (cap - prev)).clamp(_FRACTION_FLOOR, 1 - _FRACTION_FLOOR)
        return f, prev + f * (cap - prev)

    @classmethod
    def initial(cls, slices: tuple[SliceQuotes, ...], as_of: dt.date) -> ESSVISurface:
        """Each slice fitted alone, then made consistent across expiries.

        ``theta``, ``A`` and ``B`` are each replaced by their weighted
        isotonic (increasing) regression, weighted by the slice's share of the
        fit, so that a sparse slice that lands off its neighbours does not pin
        every later one; the constructor then pushes what is left inside the
        bounds, slice by slice. Slices with too few quotes start from
        ``psi = sqrt(theta)``, ``rho = -1/2``, with negligible weight.
        """
        fits = np.array([
            _fit_slice(q) if len(q.k) >= _MIN_QUOTES_TO_FIT
            else (q.atm_total_variance, np.sqrt(q.atm_total_variance), -0.5)
            for q in slices
        ])
        weight = np.array([q.weight.sum() if len(q.k) >= _MIN_QUOTES_TO_FIT else 1e-9 for q in slices])
        theta, psi, rho = fits.T
        theta = _isotonic(theta, weight) * (1 + 1e-6 * np.arange(len(theta)))  # strictly increasing
        a, b = _isotonic(psi * (1 + rho), weight), _isotonic(psi * (1 - rho), weight)
        return cls([q.expiry_time for q in slices], theta, 0.5 * (a + b), (a - b) / (a + b), as_of)

    # -- parameters, constrained ------------------------------------------
    @property
    def theta(self) -> Tensor:
        """ATM total variance at each expiry: cumulative, so increasing."""
        return torch.cumsum(nn.functional.softplus(self.raw_theta), 0)

    def wing_slopes(self) -> tuple[Tensor, Tensor]:
        """``(A, B)`` at each expiry: non-decreasing, and under the butterfly caps."""
        theta = self.theta
        frac_a, frac_b = torch.sigmoid(self.raw_a), torch.sigmoid(self.raw_b)
        a, b = [], []
        prev_a = prev_b = theta.new_zeros(())
        total = None
        n = theta.numel()
        for i in range(n):
            prev_a = prev_a + frac_a[i] * (_room(prev_b, theta[i], total) - prev_a)
            prev_b = prev_b + frac_b[i] * (_room(prev_a, theta[i], total) - prev_b)
            if i + 1 < n:
                total = (prev_a + prev_b) * theta[i + 1] / theta[i]
            a.append(prev_a)
            b.append(prev_b)
        return torch.stack(a), torch.stack(b)

    @property
    def psi(self) -> Tensor:
        a, b = self.wing_slopes()
        return 0.5 * (a + b)

    @property
    def rho(self) -> Tensor:
        a, b = self.wing_slopes()
        return (a - b) / (a + b)

    def constraints(self) -> dict[str, dict[str, float]]:
        theta = self.theta.detach()
        a, b = (x.detach() for x in self.wing_slopes())
        psi, slope = 0.5 * (a + b), torch.maximum(a, b)

        def smallest_step(x: Tensor) -> float:
            return float(torch.diff(x).min()) if x.numel() > 1 else 0.0

        return {
            "psi (1 + |rho|) < 4": {"value": float(slope.max()), "bound": _MAX_SLOPE},
            "psi^2 (1 + |rho|) / theta <= 4": {"value": float((psi * slope / theta).max()), "bound": 4.0},
            # Smallest steps between expiries (decreases, for phi); each must stay at or above 0.
            "theta increasing": {"value": smallest_step(theta), "bound": 0.0},
            "psi (1 + rho) non-decreasing": {"value": smallest_step(a), "bound": 0.0},
            "psi (1 - rho) non-decreasing": {"value": smallest_step(b), "bound": 0.0},
            "phi = psi / theta non-increasing": {"value": smallest_step(-psi / theta), "bound": 0.0},
        }

    def describe(self) -> dict[str, float | list[float]]:
        a, b = (x.detach() for x in self.wing_slopes())
        return {
            "theta": self.theta.detach().tolist(),
            "rho": ((a - b) / (a + b)).tolist(),
            "psi": (0.5 * (a + b)).tolist(),
        }

    # -- the surface ----------------------------------------------------------
    def atm_variance(self, expiry) -> Tensor:
        """``theta(T)``: smooth between expiries, constant ATM vol outside them."""
        return atm_term_structure(self._as_tensor(expiry), self.expiry_times, self.theta)

    def slice_parameters(self, expiry) -> tuple[Tensor, Tensor, Tensor]:
        """``(theta, A, B)`` at ``expiry``: ``A`` and ``B`` linear in ``theta`` between expiries."""
        t = self._as_tensor(expiry)
        times, pillars = self.expiry_times, self.theta
        theta = atm_term_structure(t, times, pillars)
        a, b = self.wing_slopes()
        n = times.numel()
        if n > 1:
            idx = torch.searchsorted(times, t.detach().contiguous()).clamp(1, n - 1)
            # 0 before the first expiry and 1 after the last: those ends are patched below.
            lam = ((theta - pillars[idx - 1]) / (pillars[idx] - pillars[idx - 1])).clamp(0, 1)
            a_t = a[idx - 1] + lam * (a[idx] - a[idx - 1])
            b_t = b[idx - 1] + lam * (b[idx] - b[idx - 1])
        else:
            a_t, b_t = a[0].expand(t.shape), b[0].expand(t.shape)
        early, scale = t < times[0], theta / pillars[0]
        return theta, torch.where(early, a[0] * scale, a_t), torch.where(early, b[0] * scale, b_t)

    def total_variance(self, k, expiry) -> Tensor:
        theta, a, b = self.slice_parameters(expiry)
        k = self._as_tensor(k)
        psi, rho = 0.5 * (a + b), (a - b) / (a + b)
        return 0.5 * (theta + rho * psi * k + torch.sqrt((psi * k + rho * theta) ** 2 + theta**2 * (1 - rho**2)))

    @property
    def reference_date(self) -> dt.date:
        return self._ref

    def __repr__(self) -> str:  # pragma: no cover
        rho = self.rho.detach()
        return (
            f"ESSVISurface({self.expiry_times.numel()} expiries, "
            f"rho {float(rho.min()):.3f} to {float(rho.max()):.3f})"
        )
