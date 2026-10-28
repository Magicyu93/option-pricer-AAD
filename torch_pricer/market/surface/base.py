"""The implied-vol surface interface every surface model implements.

A surface is a *description of the market's quotes*. It is not the dynamics --
that is a :class:`~torch_pricer.models.base.Model`, which is fitted to a surface
and knows how to produce an SDE. Local vol is exactly the map between the two,
which is why it deserves both a surface and a model.

Everything works in **total implied variance** ``w(k, T) = sigma(k, T)^2 T``
over log-moneyness ``k = log(K / F(T))``, and not in implied vol over strike.
Three reasons, all of which matter downstream:

* Dupire's formula is a ratio of derivatives of ``w`` in exactly these
  coordinates, and writing it any other way buys a page of chain rule;
* the no-arbitrage conditions are clean here -- calendar arbitrage is
  ``dw/dT >= 0`` at fixed ``k``, butterfly arbitrage is one inequality in ``w``
  and its two ``k``-derivatives;
* interpolating linearly in ``w`` between expiries preserves the first of those
  automatically, while interpolating in vol does not.

So a surface knows nothing about forwards: the strike-to-``k`` conversion is
the snapshot's (:meth:`~torch_pricer.market.snapshot.MarketSnapshot.vol`), with
the forward its curves imply. One forward, one source.

A new surface model implements :meth:`VolSurface.total_variance`, and
:meth:`VolSurface.initial` if it is to be fitted by
:func:`~torch_pricer.market.surface.calibration.fit_surface`; it gets the
butterfly density and Dupire's local variance from the base class, by autograd.
``total_variance`` must broadcast ``k`` against a tensor of expiries, since
the checks evaluate whole ``(T, k)`` grids at once. Its parameters are
``nn.Parameter`` s -- never coerced with ``float()``, since bucketed vega is a
derivative with respect to them -- and any no-arbitrage constraint lives in how
the raw parameters map to the model's, so that an optimizer can move them
anywhere.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.tensors import as_tensor


@dataclass(frozen=True)
class SliceQuotes:
    """One expiry's quotes, as a fit's starting values see them.

    ``k``, ``iv`` and ``weight`` are sorted by ``k``; ``weight`` is the fit's.
    ``atm_total_variance`` is the observed one, already made increasing across
    the slices, so a model can take it as its ATM term structure as it stands.
    """

    expiry_time: float
    k: np.ndarray
    iv: np.ndarray
    weight: np.ndarray
    atm_total_variance: float


def durrleman_g(k: Tensor, w: Tensor, dw: Tensor, d2w: Tensor) -> Tensor:
    """Durrleman's ``g(k)`` from ``w`` and its first two ``k``-derivatives.

    The risk-neutral density up to a positive factor, so negative means
    butterfly arbitrage; it is also the denominator of Dupire's formula.
    """
    return (1 - k * dw / (2 * w)) ** 2 - dw**2 / 4 * (1 / w + 0.25) + d2w / 2


def _differentiable(x: Tensor) -> Tensor:
    """A fresh node to differentiate against, still connected to ``x``'s graph."""
    x = x.clone()
    return x if x.requires_grad else x.requires_grad_(True)


def _grad(y: Tensor, x: Tensor) -> Tensor:
    # A model flat in x leaves it out of the graph: its derivative is zero, not an error.
    if not y.requires_grad:
        return torch.zeros_like(x)
    (dy,) = torch.autograd.grad(y.sum(), x, create_graph=True, allow_unused=True)
    return torch.zeros_like(x) if dy is None else dy


class VolSurface(nn.Module, ABC):
    """Implied total variance as a function of log-moneyness and expiry."""

    @abstractmethod
    def total_variance(self, k, expiry) -> Tensor:
        """``w(k, T)`` at log-moneyness ``k`` for ``expiry`` years, broadcast over ``k``."""

    @property
    @abstractmethod
    def reference_date(self) -> dt.date:
        """The date the quotes were observed."""

    @classmethod
    def initial(cls, slices: tuple[SliceQuotes, ...], as_of: dt.date) -> VolSurface:
        """A starting surface for a fit, from each expiry's quotes.

        ``slices`` are in increasing expiry. Surfaces that cannot be fitted
        leave this unimplemented.
        """
        raise NotImplementedError(f"{cls.__name__} does not support fitting")

    def penalty(self) -> Tensor:
        """A soft constraint the fit adds to its loss: zero unless overridden.

        For conditions a model cannot put in its parametrisation, such as raw
        SVI's butterfly condition.
        """
        return torch.zeros((), dtype=torch.float64)

    def _derivatives(self, k, expiry, wrt_time: bool) -> tuple[Tensor, ...]:
        """``k, w, dw/dk, d2w/dk2`` and, if asked, ``dw/dT``, broadcast together.

        Built with ``create_graph``, so they stay differentiable in the
        surface's parameters -- vega through a local vol needs exactly that.
        """
        k, t = torch.broadcast_tensors(self._as_tensor(k), self._as_tensor(expiry))
        k = _differentiable(k)
        t = _differentiable(t) if wrt_time else t
        w = self.total_variance(k, t)
        dw = _grad(w, k)
        d2w = _grad(dw, k)
        if not wrt_time:
            return k, w, dw, d2w
        return k, w, dw, d2w, _grad(w, t)

    def butterfly_density(self, k, expiry) -> Tensor:
        """Durrleman's ``g(k)`` at ``(k, T)``, differentiable in the parameters."""
        with torch.enable_grad():
            return durrleman_g(*self._derivatives(k, expiry, wrt_time=False))

    def local_variance(self, k, expiry) -> Tensor:
        """Dupire's local variance at log-moneyness ``k`` and time ``expiry``.

        In total variance, ``sigma_LV^2 = (dw/dT) / g(k)`` with ``g``
        Durrleman's: non-negative wherever the surface is free of both
        arbitrages. ``dw/dT`` is at fixed ``k = log(K / F(T))``. At an expiry
        where ``w`` is only ``C^0`` in ``T`` it is the derivative from the side
        ``searchsorted`` picks. Differentiable in the parameters.
        """
        with torch.enable_grad():
            k, w, dw, d2w, dw_dt = self._derivatives(k, expiry, wrt_time=True)
            return dw_dt / durrleman_g(k, w, dw, d2w)

    def implied_vol(self, k, expiry) -> Tensor:
        """``sqrt(w(k, T) / T)``."""
        t = self._as_tensor(expiry)
        return (self.total_variance(k, t) / t).sqrt()

    def implied_vol_grid(self, k, expiries) -> np.ndarray:
        """Implied vols on a grid, for display: shape ``(len(expiries), len(k))``.

        Evaluated without gradients and returned as numpy, one row per expiry.
        """
        k = self._as_tensor(np.asarray(k, dtype=float))
        with torch.no_grad():
            return np.stack([self.implied_vol(k, float(t)).cpu().numpy() for t in np.asarray(expiries)])

    def constraints(self) -> dict[str, dict[str, float]]:
        """Each no-arbitrage bound the parametrisation enforces: ``{name: {value, bound}}``.

        Holding by construction, so a value can approach its bound but not
        cross it; how close it is says where the data pushes the model.
        """
        return {}

    def describe(self) -> dict[str, float | list[float]]:
        """The model's parameters, for reports. Raw parameters unless overridden."""
        return {name: p.detach().tolist() for name, p in self.named_parameters()}

    def _as_tensor(self, x) -> Tensor:
        """``x`` on this surface's dtype and device."""
        ref = next(chain(self.parameters(), self.buffers()), None)
        if ref is None:
            return as_tensor(x)
        return as_tensor(x, dtype=ref.dtype, device=ref.device)
