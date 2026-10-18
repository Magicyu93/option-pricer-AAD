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
:func:`~torch_pricer.market.surface.calibration.fit_surface`. Its parameters are
``nn.Parameter`` s -- never coerced with ``float()``, since bucketed vega is a
derivative with respect to them -- and any no-arbitrage constraint lives in how
the raw parameters map to the model's, so that an optimizer can move them
anywhere.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.tensors import as_tensor


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
    def initial(cls, expiry_times, atm_total_variance, as_of: dt.date) -> VolSurface:
        """A starting surface for a fit, from each expiry's ATM total variance.

        ``atm_total_variance`` is increasing in ``expiry_times``. Surfaces that
        cannot be fitted leave this unimplemented.
        """
        raise NotImplementedError(f"{cls.__name__} does not support fitting")

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
