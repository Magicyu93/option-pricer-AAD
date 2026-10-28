"""One vol everywhere: the Black-Scholes textbook surface."""

from __future__ import annotations

import datetime as dt

import torch
import torch.nn as nn
from torch import Tensor

from torch_pricer.market.surface.base import SliceQuotes, VolSurface
from torch_pricer.tensors import as_tensor


class FlatVolSurface(VolSurface):
    """``w(k, T) = sigma^2 T`` for every ``k``."""

    def __init__(self, vol: float, as_of: dt.date):
        super().__init__()
        self.level = nn.Parameter(as_tensor(float(vol)))
        self._ref = as_of

    @classmethod
    def initial(cls, slices: tuple[SliceQuotes, ...], as_of: dt.date) -> FlatVolSurface:
        """The median ATM vol across expiries."""
        t = torch.tensor([q.expiry_time for q in slices], dtype=torch.float64)
        w = torch.tensor([q.atm_total_variance for q in slices], dtype=torch.float64)
        surface = cls(float((w / t).sqrt().median()), as_of)
        return surface.to(torch.float64)

    def total_variance(self, k, expiry) -> Tensor:
        # Broadcast against k so callers get the shape they passed in, without
        # letting k's *value* into the result.
        k = self._as_tensor(k)
        t = self._as_tensor(expiry)
        return (self.level**2 * t).expand(torch.broadcast_shapes(k.shape, t.shape))

    @property
    def reference_date(self) -> dt.date:
        return self._ref

    def __repr__(self) -> str:  # pragma: no cover
        return f"FlatVolSurface({float(self.level.detach()):.4f}, {self._ref})"
