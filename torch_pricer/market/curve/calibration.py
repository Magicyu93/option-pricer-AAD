"""Curves from market data: the discount curve, and the forwards two curves imply.

**Discount curve.** One function per curve model, each returning a
:class:`~torch_pricer.market.curve.base.Curve`. :func:`treasury_curve` bootstraps the
quote set's Treasury par yields (:func:`~torch_pricer.market.curve.treasury.par_to_zero`);
a new model (a spread over Treasuries, a parametric curve) is a new function next to it.

**Dividend curve.** :func:`implied_dividend_curve` turns the per-expiry parity forwards
of :func:`~torch_pricer.market.parity.parity_forwards` into one curve. With
``F = S D_q / D_r = (S - PV) / D_r``, the dividend discount factor is ``D_q(T) = 1 - PV(T) / S``,
``PV`` the present value of dividends to ``T``: implied dividends never fall with ``T``, so
the cumulative carry ``y(T) = -ln D_q(T)`` never falls either. A weighted isotonic
regression imposes that, and the curve interpolates ``y`` linearly between expiries -- a
constant forward dividend rate -- which is :class:`RateCurve`'s own rule. The carry also
absorbs the gap between Treasury rates and the market's funding rate, so it may start
below zero; the forward is what the market trades at.

**Forwards.** :func:`curve_forwards` gives each expiry's forward from the discount curve
and the dividend curve, ``F = S D_q(T) / D_r(T)`` -- the snapshot's own
:meth:`~torch_pricer.market.snapshot.MarketSnapshot.forward`, so the forward that places
a quote on the surface is the one the pricer uses.
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


# -- dividend curve ----------------------------------------------------------------

def isotonic(y, weights) -> np.ndarray:
    """The non-decreasing sequence closest to ``y`` in weighted least squares.

    Pool-adjacent-violators: a value below its predecessor is pooled with it into their
    weighted mean, repeatedly, so each pool sits between the points it replaces.
    """
    pools: list[list[float]] = []  # [value, weight, count]
    for value, weight in zip(np.asarray(y, dtype=float), np.asarray(weights, dtype=float)):
        pools.append([value, weight, 1])
        while len(pools) > 1 and pools[-2][0] > pools[-1][0]:
            v2, w2, n2 = pools.pop()
            v1, w1, n1 = pools.pop()
            pools.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.repeat([p[0] for p in pools], [p[2] for p in pools])


def implied_dividend_curve(parity_fwds: pd.DataFrame, label: str = "parity") -> RateCurve:
    """The dividend curve through the parity forwards, cumulative carry made non-decreasing.

    ``parity_fwds`` is :func:`~torch_pricer.market.parity.parity_forwards` output. Each
    expiry is a pillar at its ``T``; ``y`` is fitted by :func:`isotonic` with weights
    ``1 / y_se**2``, so a noisy forward gives way to its neighbours. Beyond the first and
    last pillar the zero rate is flat. Raises :class:`MarketDataError` with no forwards:
    there is no flat rate to fall back on that anyone chose.
    """
    fw = parity_fwds[parity_fwds["T"] > 0].sort_values("T")
    if fw.empty:
        raise MarketDataError("no put-call parity forwards to build a dividend curve from")
    t = fw["T"].to_numpy(dtype=float, copy=True)
    y = isotonic(fw["y"], 1.0 / fw["y_se"].to_numpy(dtype=float) ** 2)
    return RateCurve.from_zeros(t, y / t, label=label)


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

