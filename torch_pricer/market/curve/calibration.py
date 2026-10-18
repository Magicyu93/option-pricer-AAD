"""Curves calibrated to market data: the discount curve, and the carry.

One function per curve model, each returning a :class:`~torch_pricer.market.curve.base.Curve`.
Curve models differ too much in what they consume -- par yields here, option
forwards below -- for one generic fitter to pay off; a new model (a spread over
Treasuries, a parametric curve) is a new function next to these.

**Discount curve.** :func:`treasury_curve` bootstraps the quote set's Treasury
par yields (:func:`~torch_pricer.market.curve.treasury.par_to_zero`).

**Carry.** :func:`implied_carry_curve` turns the put-call parity forwards of
:mod:`torch_pricer.market.parity` into a dividend curve, ``q(T) = r(T) - ln(F/S) / T``,
so the snapshot reproduces each forward without a dividend forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from torch_pricer.conventions import tenor_label, tenor_years
from torch_pricer.errors import MarketDataError
from torch_pricer.market.curve.base import Curve
from torch_pricer.market.curve.curves import RateCurve
from torch_pricer.market.curve.treasury import par_to_zero
from torch_pricer.market.market_data import QuoteSet, RateQuote

# -- discount curve ----------------------------------------------------------------

def par_yield_quotes(tenors, yields_pct, as_of=None) -> tuple[RateQuote, ...]:
    """Par-yield quotes from nominal tenors in years and yields in percent.

    The shape the data pipeline's ``ParYieldSnapshot`` carries; the caller
    passes its fields, so nothing here imports the pipeline.
    """
    return tuple(
        RateQuote(tenor_label(float(t)), float(y) / 100.0, kind="par", as_of=as_of)
        for t, y in zip(tenors, yields_pct)
    )


def treasury_curve(quotes: QuoteSet, label: str = "UST") -> RateCurve:
    """The zero curve bootstrapped from the quote set's par yields.

    Raises :class:`MarketDataError` when the quote set carries none, rather than
    falling back to a flat rate nobody chose.
    """
    par = [q for q in quotes.rates if q.kind == "par"]
    if not par:
        raise MarketDataError(f"quote set for {quotes.as_of} has no par yields")
    tenors, zeros = par_to_zero([tenor_years(q.tenor) for q in par], [q.rate for q in par])
    return RateCurve.from_zeros(tenors, zeros, label=label)


# -- carry -------------------------------------------------------------------------

def implied_carry_curve(forwards: pd.DataFrame, curve: Curve) -> RateCurve:
    """The dividend curve under which ``S D_q(T) / D_r(T)`` is each expiry's forward.

    A zero curve with one pillar per expiry: ``q(T) = r(T) - ln(F/S) / T``.
    """
    t = forwards["T"].to_numpy(dtype=float)
    r = curve.zero_rate(t).detach().numpy()
    q = r - np.log(forwards["ratio"].to_numpy(dtype=float)) / t
    return RateCurve.from_zeros(t, q, label="implied_carry")


#: Curve models a caller can choose by name: each takes the quote set and
#: returns the discount :class:`~torch_pricer.market.curve.base.Curve`.
CURVE_MODELS = {
    "Treasury bootstrap": treasury_curve,
}


# -- views for inspection ----------------------------------------------------------

def rates_comparison(forwards: pd.DataFrame, curve: Curve, carry: Curve) -> pd.DataFrame:
    """Per expiry: the discount curve's rate, and what the option forwards imply.

    ``implied_growth = ln(F/S) / T`` is the ``r - q`` the forwards price in;
    ``carry = r - implied_growth`` is the implied carry curve's zero rate; the
    forward columns are each curve's forward rate from the previous expiry.
    All rates continuously compounded, as decimals.
    """
    t = forwards["T"].to_numpy(dtype=float)
    prev = np.concatenate([[0.0], t[:-1]])
    out = forwards[["expiry", "T", "source", "n_pairs"]].copy()
    out["treasury_zero"] = curve.zero_rate(t).detach().numpy()
    out["implied_growth"] = np.log(forwards["ratio"].to_numpy(dtype=float)) / t
    out["carry_zero"] = carry.zero_rate(t).detach().numpy()
    out["treasury_forward"] = curve.forward_rate(prev, t).detach().numpy()
    out["carry_forward"] = carry.forward_rate(prev, t).detach().numpy()
    return out
