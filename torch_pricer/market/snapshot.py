"""The state of the world at one instant, as the pricer sees it.

Torch conversion boundary.

A snapshot holds *calibrated* objects, not quotes: curves that have been fitted
and a vol surface that has been calibrated. Raw observations live in
:mod:`torch_pricer.market.market_data`, upstream of here.

``as_of`` is an ordinary date and the only date in the pricing path. Everything
below this layer measures time as a year fraction from it, so there is no global
evaluation date for a concurrent request to move underneath a valuation.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
from dataclasses import dataclass

import torch
from torch import Tensor

from torch_pricer.conventions import DEFAULT_DAY_COUNT, to_date, year_fraction
from torch_pricer.market.curve.base import Curve
from torch_pricer.market.curve.curves import RateCurve
from torch_pricer.market.surface.base import VolSurface
from torch_pricer.market.surface.flat import FlatVolSurface
from torch_pricer.tensors import as_tensor


@dataclass(frozen=True)
class MarketSnapshot:
    """Spot, funding, dividends and dynamics, all dated ``as_of``.

    Build one with :meth:`flat` for a quick mark, or assemble the pieces
    yourself for a real one.
    """

    ticker: str  # option spec id
    as_of: dt.date
    spot: Tensor  # underlying stock spot price

    # market condition
    discount: Curve
    dividend: Curve  # continuous dividend yield curve: flat, or a term structure
    vol_surface: VolSurface  # implied vol surface for options on this underlying

    # misc
    day_count: str = DEFAULT_DAY_COUNT

    # -- constructors ---------------------------------------------------
    @classmethod
    def flat(
        cls,
        as_of: dt.date | str,
        spot: float,
        ticker: str = "",
        flat_rate: float = 0.0,
        flat_dividend: float = 0.0,
        flat_vol: float = 0.2,
        day_count: str = DEFAULT_DAY_COUNT,
    ) -> MarketSnapshot:
        """Flat curves and a constant vol -- the Black-Scholes textbook market."""
        as_of = to_date(as_of)
        return cls(
            as_of=as_of,
            spot=as_tensor(float(spot)),
            ticker=ticker,
            discount=RateCurve.flat(flat_rate, "flat_r"),
            dividend=RateCurve.flat(flat_dividend, "flat_d"),
            vol_surface=FlatVolSurface(flat_vol, as_of),
            day_count=day_count,
        )

    # -- queries --------------------------------------------------------
    def time_to(self, d: dt.date | str) -> float:
        """Year fraction from ``as_of`` to ``d`` under this snapshot's day count."""
        return year_fraction(self.as_of, to_date(d), self.day_count)

    def forward(self, t) -> Tensor:
        """Forward level to ``t`` years: ``S D_q(t) / D_r(t)``."""
        return self.spot * self.dividend.discount(t) / self.discount.discount(t)

    def vol(self, strike, t) -> Tensor:
        """Implied vol at ``strike`` for ``t`` years.

        The surface lives in log-moneyness; the strike is placed on it against
        this snapshot's own forward, so the forward has one source -- the curves
        -- and a derivative of the vol reaches spot and the curve pillars as well
        as the surface.
        """
        t = as_tensor(t, dtype=self.spot.dtype, device=self.spot.device)
        k = torch.log(as_tensor(strike, dtype=t.dtype, device=t.device) / self.forward(t))
        return self.vol_surface.implied_vol(k, t)

    def to(self, device=None, dtype: torch.dtype | None = None) -> MarketSnapshot:
        """A copy of this snapshot on a different device or dtype.

        The curves and surface are deep-copied first. ``nn.Module.to`` moves a
        module *in place* and returns it, so without the copy this would quietly
        recast the caller's curves as a side effect of pricing.
        """
        if device is None and dtype is None:
            return self
        return dataclasses.replace(
            self,
            spot=self.spot.to(device=device, dtype=dtype),
            discount=copy.deepcopy(self.discount).to(device=device, dtype=dtype),
            dividend=copy.deepcopy(self.dividend).to(device=device, dtype=dtype),
            vol_surface=copy.deepcopy(self.vol_surface).to(device=device, dtype=dtype),
        )

    def with_spot(self, spot: Tensor) -> MarketSnapshot:
        """A copy marked at a different spot, sharing the curves and surface.

        The spot passes through :func:`~torch_pricer.tensors.as_tensor` by
        identity, so a leaf the caller intends to differentiate against stays
        attached to the graph.
        """
        return dataclasses.replace(self, spot=as_tensor(spot))

    def __repr__(self) -> str:  # pragma: no cover
        tag = f"{self.ticker} " if self.ticker else ""
        return (
            f"MarketSnapshot({tag}{self.as_of}, S={float(self.spot):.4f}, "
            f"{self.vol_surface!r})"
        )
