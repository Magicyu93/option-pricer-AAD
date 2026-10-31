"""The forwards behind every iv: put-call parity pairs, what filters them, and the curve.

The library does the work (``torch_pricer.market.parity.forward_pairs`` and
``parity_forwards``, ``torch_pricer.market.curve.calibration.implied_dividend_curve``);
what is here turns its frames into what the pages draw: implied dividends against the
curve, how many pairs each filter drops, and one expiry's forward through the day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from torch_pricer.market.market_data import TIMEZONE
from torch_pricer.market.parity import DROP_REASONS


def _curve_ratio(T, curve, dividend) -> np.ndarray:
    """``F / S = D_q(T) / D_r(T)`` from the two curves."""
    T = np.maximum(np.asarray(T, dtype=float), 0.0)
    return dividend.discount(T.copy()).detach().numpy() / curve.discount(T.copy()).detach().numpy()


def implied_dividends(parity_fwds: pd.DataFrame, dividend, expiries: pd.DataFrame,
                      n: int = 200) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Implied dividends ``PV(T) = S (1 - D_q(T))`` in index points: the expiries' own, and the curve's.

    ``expiries`` is ``expiry_table`` output: every expiry the page shows. Returns
    ``(own, line, interpolated)``:

    * ``own``: ``parity_fwds`` with ``days``, ``PV`` from its ``y`` at its own spot, ``PV_se``
      and ``PV_curve``, the curve's at the same spot and ``T``;
    * ``line``: the curve on a grid of ``T`` at the median spot;
    * ``interpolated``: expiries with no forward of their own, on the curve.
    """
    own = parity_fwds.assign(days=parity_fwds["T"] * 365)
    own["PV"] = own.spot * -np.expm1(-own.y)
    own["PV_se"] = own.spot * np.exp(-own.y) * own.y_se
    own["PV_curve"] = own.spot * (1 - dividend.discount(own["T"].to_numpy(copy=True)).detach().numpy())

    shown = expiries[(expiries["T"] > 0) & expiries["T"].notna()]
    spot = float(own.spot.median()) if len(own) else float(shown.spot.median())
    t_all = pd.concat([own["T"], shown["T"]])
    t = np.geomspace(t_all.min(), t_all.max(), n) if len(t_all) else np.array([])
    line = pd.DataFrame({"T": t, "days": 365 * t, "PV": spot * (1 - dividend.discount(t.copy()).detach().numpy())})

    rest = shown[shown.tier == "curve"]
    interpolated = pd.DataFrame({
        "expiry": rest.expiry, "option_root": rest.option_root, "T": rest["T"], "days": rest.days,
        "PV": rest.spot * (1 - dividend.discount(rest["T"].to_numpy(copy=True)).detach().numpy()),
    })
    return own, line, interpolated


def filter_funnel(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pairs dropped by each filter, in the order they apply, then those kept, by tier.

    ``pairs`` is ``forward_pairs`` output. Columns ``step, close, day, total``.
    """
    steps = [*DROP_REASONS, "kept"]
    label = pairs.dropped_by.replace("", "kept")
    counts = pd.crosstab(label, pairs.tier).reindex(index=steps, columns=["close", "day"], fill_value=0)
    counts["total"] = counts.sum(axis=1)
    return counts.rename_axis("step").reset_index().rename_axis(None, axis=1)


def intraday_forward(pairs: pd.DataFrame, expiry, curve, dividend) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One expiry's pair forwards through the day, against the curve's forward at each minute.

    Returns ``(pairs, minutes)``: every pair of the expiry with ``time`` (ET) and
    ``gap = forward - S D_q(T) / D_r(T)`` in index points, at its own minute's spot and ``T``;
    and the median ``gap`` of the kept pairs per minute.
    """
    p = pairs[pairs.expiry == expiry].copy()
    p["gap"] = p.forward - p.spot * _curve_ratio(p["T"], curve, dividend)
    p["time"] = p.timestamp.dt.tz_convert(TIMEZONE).dt.tz_localize(None)  # wall clock, for the axis
    kept = p[p.dropped_by == ""]
    minutes = kept.groupby("time").agg(gap=("gap", "median"), n=("gap", "size"), tier=("tier", "first")).reset_index()
    return p, minutes
