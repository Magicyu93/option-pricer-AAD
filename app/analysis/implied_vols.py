"""Black implied vols of traded minute bars, and the call-put parity checks between them.

Rows are :func:`torch_pricer.market.market_data.window_bars` output: one contract's trade
bar in one minute, with ``T`` from that minute to the contract's settlement (the library's
rule: SPXW at the 16:00 close, SPX monthlies at the open). For each row:

* ``r``  the curve's zero rate to ``T``; ``D = exp(-r T)``
* ``F``  the forward grown from that minute's spot, ``F = S exp((r - q) T)``, ``q`` a flat
  dividend yield
* ``iv`` Black-76 inversion of the bar's price on ``(F, K, T, D)``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from torch_pricer.black_formula import implied_vol
from torch_pricer.market.market_data import TIMEZONE

DIVIDEND_YIELD = 0.011  # flat continuous SPX dividend yield
KEYS = ["option_root", "expiry", "strike", "option_type"]  # one contract


def curve_forward(spot, T, curve, dividend_yield: float = DIVIDEND_YIELD):
    """Zero rate r(T), discount D = exp(-r T) and forward F = S exp((r - q) T) from ``curve``."""
    T = np.asarray(T, dtype=float)
    r = curve.zero_rate(np.maximum(T, 0.0)).detach().numpy()
    discount = np.exp(-r * T)
    forward = np.asarray(spot, dtype=float) * np.exp((r - dividend_yield) * T)
    return r, discount, forward


def contract_implied_vols(bars: pd.DataFrame, curve, dividend_yield: float = DIVIDEND_YIELD) -> pd.DataFrame:
    """``bars`` with ``option_type``, r, discount, forward, k = ln(K/F) and iv.

    iv is nan where no vol reproduces the price: below intrinsic, or no time value left
    to resolve.
    """
    out = bars.copy()
    out["option_type"] = np.where(out.right > 0, "call", "put")
    out["r"], out["discount"], out["forward"] = curve_forward(out.spot, out["T"], curve, dividend_yield)
    out["k"] = np.log(out.strike / out.forward)
    out["iv"] = implied_vol(out.price, out.forward, out.strike, out["T"], out.discount, out.right)
    return out


def contract_vol_summary(df_iv: pd.DataFrame, combine: bool = True) -> pd.DataFrame:
    """Contracts with an implied vol, either combined across minutes or one row per bar.

    combine=True: one row per contract (root, expiry, strike, call/put), each column the
        volume-weighted average over the contract's bars in the window; ``volume`` is the
        total traded, ``n_bars`` the number of minutes it traded in.
    combine=False: one row per contract per minute, nothing averaged: ``price`` is the bar's
        close, ``spot`` the underlying's close in the same minute, ``volume`` the
        contracts traded in that minute, and ``timestamp`` says which minute.

    ``otm`` marks puts struck below the forward and calls at or above it: the side whose
    price is mostly time value, and so the reliable vol at that strike.
    """
    keys = KEYS
    values = ["T", "spot", "forward", "discount", "k", "iv", "price"]
    bars = df_iv[df_iv.iv.notna()]

    if combine:
        weighted = bars[values].mul(bars.volume, axis=0)
        weighted[keys] = bars[keys]
        total = bars.groupby(keys).volume.sum()
        summary = weighted.groupby(keys)[values].sum().div(total, axis=0)
        summary["volume"] = total
        summary["n_bars"] = bars.groupby(keys).size()
        summary = summary.reset_index()
        order = ["expiry", "strike", "option_type"]
    else:
        summary = bars[["timestamp", *keys, *values, "volume"]].assign(n_bars=1).reset_index(drop=True)
        order = ["expiry", "strike", "option_type", "timestamp"]

    summary["otm"] = np.where(summary.option_type == "call", summary.k >= 0, summary.k < 0)
    summary["total_variance"] = summary.iv**2 * summary["T"]
    return summary.sort_values(order, ignore_index=True)


def is_per_minute(contract_vols: pd.DataFrame) -> bool:
    """True for contract_vol_summary(combine=False) output: one row per contract per minute."""
    return "timestamp" in contract_vols.columns


def select_points(contract_vols: pd.DataFrame, side: str = "otm", min_days: float = 1.0,
                  max_days: float | None = None, min_price: float = 0.05,
                  max_abs_k: float = 1.0) -> pd.DataFrame:
    """Rows on ``side`` ("otm", "itm" or "both") passing the filters, with ``days`` (and
    ``time``, the bar's minute in ET, for per-minute rows).

    The moneyness band ``|k| <= max_abs_k * sqrt(T)`` widens with maturity.
    """
    if side not in ("otm", "itm", "both"):
        raise ValueError(f"side must be 'otm', 'itm' or 'both', got {side!r}")
    v = contract_vols
    days = v["T"] * 365
    keep = (days >= min_days) & (v.price >= min_price) & (v.k.abs() <= max_abs_k * np.sqrt(v["T"]))
    if max_days is not None:
        keep &= days <= max_days
    if side != "both":
        keep &= v.otm if side == "otm" else ~v.otm
    points = v[keep].assign(days=days[keep])
    if is_per_minute(points):
        points["time"] = points.timestamp.dt.tz_convert(TIMEZONE).dt.strftime("%H:%M")
    return points.sort_values(["expiry", "k"])


def near_the_money(points: pd.DataFrame, band: float = 0.05) -> pd.Series:
    """Mask of ``points`` with ``|k| <= band * sqrt(T)``."""
    return points.k.abs() <= band * np.sqrt(points["T"])


def call_put_pairs(contract_vols: pd.DataFrame, parity_discount: pd.Series, **filters) -> pd.DataFrame:
    """A call and a put at the same root, expiry, strike and minute, both passing the filters.

    ``iv_gap = iv_call - iv_put`` in vol points. Two sides, each with its own discount:

    * Treasury + flat q, the F and D the ivs were computed with:
      ``parity_residual = (C - P) - D (F - K)``;
    * put-call parity, with ``parity_discount`` (the regression's D, indexed by expiry):
      ``F_pair = K + (C - P) / D_parity``, NaN for an expiry with no regression.

    ``F_error = F - F_pair`` compares the two forwards in index points. Per-minute rows
    only: a combined row averages a call and a put over different minutes, so their gap
    would mix timing into parity.
    """
    if not is_per_minute(contract_vols):
        raise ValueError("call_put_pairs needs per-minute rows: contract_vol_summary(..., combine=False)")
    points = select_points(contract_vols, side="both", **filters)
    keys = ["option_root", "expiry", "strike", "timestamp"]
    shared = ["T", "days", "time", "spot", "forward", "discount", "k"]
    legs = ["price", "iv", "volume"]
    calls = points[points.option_type == "call"].set_index(keys)
    puts = points[points.option_type == "put"].set_index(keys)
    pairs = calls[shared + legs].join(puts[legs], lsuffix="_call", rsuffix="_put", how="inner").reset_index()
    pairs["iv_gap"] = 100 * (pairs.iv_call - pairs.iv_put)
    pairs["parity"] = pairs.price_call - pairs.price_put
    pairs["parity_residual"] = pairs.parity - pairs.discount * (pairs.forward - pairs.strike)
    pairs["parity_discount"] = pairs.expiry.map(parity_discount)
    pairs["F_pair"] = pairs.strike + pairs.parity / pairs.parity_discount
    pairs["F_error"] = pairs.forward - pairs.F_pair
    return pairs.sort_values(["expiry", "strike", "timestamp"], ignore_index=True)


def parity_by_expiry(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per expiry of :func:`call_put_pairs`: the pairs with an ``F_pair``, mean days, the
    average ``discount`` and ``forward`` (the line ``D (F - K)``), and the median and middle
    half (25%-75%) of ``F_error`` and ``parity_residual``. Expiries with no parity regression
    have no ``F_error``; the caller drops them."""
    def q25(s):
        return s.quantile(0.25)

    def q75(s):
        return s.quantile(0.75)

    return pairs.groupby(["option_root", "expiry"]).agg(
        n=("F_error", "count"), days=("days", "mean"),
        discount=("discount", "mean"), forward=("forward", "mean"),
        F_error_median=("F_error", "median"), F_error_q25=("F_error", q25), F_error_q75=("F_error", q75),
        residual_median=("parity_residual", "median"), residual_q25=("parity_residual", q25),
        residual_q75=("parity_residual", q75),
    ).reset_index().sort_values("days", ignore_index=True)
