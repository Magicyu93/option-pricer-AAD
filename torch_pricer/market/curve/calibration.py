"""Curves from market data: the discount curve, and the forwards two curves imply.

**Discount curve.** One function per curve model, each returning a
:class:`~torch_pricer.market.curve.base.Curve`. :func:`treasury_curve` bootstraps the
quote set's Treasury par yields (:func:`~torch_pricer.market.curve.treasury.par_to_zero`);
a new model (a spread over Treasuries, a parametric curve) is a new function next to it.

**Forwards.** :func:`curve_forwards` gives each expiry's forward from the discount curve
and a dividend curve, ``F = S D_q(T) / D_r(T)`` -- the snapshot's own
:meth:`~torch_pricer.market.snapshot.MarketSnapshot.forward`, so the forward that places
a quote on the surface is the one the pricer uses. The dividend curve is an input
(flat, or a term structure from a dividend forecast), not backed out of option prices:
the forwards put-call parity implies on trade prints are too noisy to build on.
"""

from __future__ import annotations

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


# -- forwards ----------------------------------------------------------------------

#: :func:`curve_forwards` output: one row per expiry, at the close.
FORWARD_COLUMNS = ["expiry", "T", "discount", "ratio", "forward"]


def curve_forwards(bars: pd.DataFrame, discount: Curve, dividend: Curve, spot: float) -> pd.DataFrame:
    """One forward per expiry in ``bars``, at the close and the reference ``spot``.

    ``bars`` is :func:`~torch_pricer.market.market_data.window_bars` output. Each
    expiry's ``T`` is its ``T_close``; ``discount`` is ``D_r(T)``, ``ratio = F / S =
    D_q(T) / D_r(T)`` and ``forward = spot * ratio``. Returns ``FORWARD_COLUMNS``,
    sorted by ``T``.
    """
    out = (bars.groupby("expiry", as_index=False)["T_close"].first()
           .rename(columns={"T_close": "T"}).sort_values("T"))
    t = out["T"].to_numpy(dtype=float)
    out["discount"] = discount.discount(t).detach().numpy()
    out["ratio"] = dividend.discount(t).detach().numpy() / out["discount"]
    out["forward"] = spot * out["ratio"]
    return out[FORWARD_COLUMNS].reset_index(drop=True)


#: Curve models a caller can choose by name: each takes the quote set and
#: returns the discount :class:`~torch_pricer.market.curve.base.Curve`.
CURVE_MODELS = {
    "Treasury bootstrap": treasury_curve,
}

