"""Put-call parity on traded minute bars: the pairs, the regression, and the forwards.

For a European call and put on the same strike and expiry,
``C - P = D (F - K) = A_t - D K`` with ``D = exp(-r T)`` and ``A_t = S_t exp(-q T)``.
Every strike where both traded gives a forward ``F = K + (C - P) / D`` -- with no model
and no dividend forecast. That is how a snapshot leaves dividends out without getting
them wrong: whatever the market expects the index to pay is already in ``F``, and the
*implied carry* ``q(T) = r(T) - ln(F/S) / T`` stands in for the dividend curve.

Trades are not simultaneous, and the index moves, so a call and a put only make a pair
when they traded in the same minute, and each minute has its own intercept ``A_t``.
Pairs are kept within a moneyness band of that minute's spot -- far-wing prints are
stale -- and only in minutes with two or more strikes, since one strike cannot separate
the intercept from the slope.

Two uses of the pairs:

* :func:`regress_parity` -- ``D`` estimated from the trades alone: OLS of ``C - P`` on
  ``K`` with one intercept per minute, by demeaning within each minute,
  ``-D = sum(K_dm * P_dm) / sum(K_dm^2)``. Then ``r = -ln(D) / T`` and
  ``q = -ln(A_t / S_t) / T``. A diagnostic of the discount curve and the carry; noisy
  for short expiries, where ``D`` is close to 1 and the slope says little about ``r``.
* :func:`parity_forwards` -- the forward per expiry with ``D`` from the discount curve,
  the same ``D`` the snapshot prices with: each minute's average ``(C - P + D K) / D``
  against that minute's spot. Expiries with too few pairs -- most long-dated ones, where
  calls barely trade -- take their carry from the expiries that have them.

Every pair uses its bar's own ``T`` (time from the minute to settlement); the forwards
are reported at the close, ``T_close``, by holding each expiry's growth rate
``ln(F/S) / T`` fixed.

Box-spread prints -- two strikes in a minute, all four legs with the same volume -- are
flagged ``is_box``: traded as one package, they price the box financing rate, which runs
above T-bills.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from torch_pricer.errors import MarketDataError
from torch_pricer.market.curve.base import Curve

#: Keep pairs with ``|K / S - 1|`` below this: far-wing trade prints are stale.
MONEYNESS_BAND = 0.10
FORWARD_COLUMNS = ["expiry", "T", "discount", "ratio", "forward", "n_pairs", "source"]
PAIR_COLUMNS = [
    "expiry", "timestamp", "option_root", "strike", "call", "put", "call_volume", "put_volume",
    "spot", "days", "T", "T_close", "parity", "is_box",
]
_BY_MINUTE = ["expiry", "timestamp"]


def parity_pairs(bars: pd.DataFrame, moneyness_band: float = MONEYNESS_BAND) -> pd.DataFrame:
    """Every same-minute call/put pair within ``moneyness_band`` of that minute's spot.

    ``bars`` is :func:`~torch_pricer.market.market_data.window_bars` output. One row
    per pair, ``parity = C - P``, keeping only minutes of an expiry with two or more
    strikes. Columns are ``PAIR_COLUMNS``.
    """
    keys = ["expiry", "timestamp", "strike"]
    shared = ["option_root", "spot", "days", "T", "T_close"]
    calls = bars.loc[bars["right"] > 0, [*keys, *shared, "price", "volume"]]
    puts = bars.loc[bars["right"] < 0, [*keys, "price", "volume"]]
    pairs = calls.merge(puts, on=keys, suffixes=("_call", "_put")).rename(columns={
        "price_call": "call", "price_put": "put", "volume_call": "call_volume", "volume_put": "put_volume",
    })
    pairs["parity"] = pairs["call"] - pairs["put"]
    pairs = pairs[(pairs["strike"] / pairs["spot"] - 1).abs() < moneyness_band]
    pairs = pairs[pairs.groupby(_BY_MINUTE)["strike"].transform("size") >= 2].copy()

    grouped = pairs.groupby(_BY_MINUTE)
    one_volume = (
        (pairs["call_volume"] == pairs["put_volume"])
        & (grouped["call_volume"].transform("nunique") == 1)
        & (grouped["put_volume"].transform("nunique") == 1)
    )
    pairs["is_box"] = (grouped["strike"].transform("size") == 2) & one_volume
    return pairs[PAIR_COLUMNS].sort_values(["expiry", "timestamp", "strike"]).reset_index(drop=True)


def regress_parity(pairs: pd.DataFrame, exclude_boxes: bool = False) -> dict:
    """The minute-demeaned parity regression on one expiry of :func:`parity_pairs`.

    Returns ``discount`` D and ``div_discount`` exp(-q T) with minute-clustered standard
    errors, the continuous ``r`` and ``q``, the mean ``T``, the counts, and ``intercept``:
    the average ``C - P + D K``, the level of the line ``intercept - D K`` through the
    raw pairs.
    """
    p = pairs[~pairs["is_box"]] if exclude_boxes else pairs
    p = p[p.groupby(_BY_MINUTE)["strike"].transform("size") >= 2]
    empty = {"discount": np.nan, "discount_se": np.nan, "div_discount": np.nan,
             "div_discount_se": np.nan, "r": np.nan, "r_se": np.nan, "q": np.nan, "q_se": np.nan,
             "T": np.nan, "n_pairs": 0, "n_minutes": 0, "intercept": np.nan}
    if p.empty:
        return empty
    grouped = p.groupby(_BY_MINUTE)
    k_dm = (p["strike"] - grouped["strike"].transform("mean")).to_numpy()
    p_dm = (p["parity"] - grouped["parity"].transform("mean")).to_numpy()

    discount = -(k_dm @ p_dm) / (k_dm @ k_dm)
    resid = p_dm + discount * k_dm

    # Minute-clustered standard error: trades in one minute share spot and timing noise.
    n_minutes = grouped.ngroups
    cluster_score = pd.Series(k_dm * resid).groupby(grouped.ngroup().to_numpy()).sum()
    discount_se = np.sqrt((cluster_score**2).sum()) / (k_dm @ k_dm)
    if n_minutes < 2 or len(p) - n_minutes - 1 <= 0:
        discount_se = np.nan  # an exact fit (e.g. one box print) leaves no residual to measure noise

    # Per-minute intercepts A_t, scaled by that minute's spot.
    minute = grouped.agg(parity=("parity", "mean"), strike=("strike", "mean"), spot=("spot", "first"))
    div_discount_t = (minute["parity"] + discount * minute["strike"]) / minute["spot"]
    # Spread across minutes, plus the error in D carried through D * mean(K) / S_t.
    div_discount_se = np.sqrt(
        div_discount_t.var() / n_minutes + (discount_se * (minute["strike"] / minute["spot"]).mean()) ** 2
    )
    div_discount = div_discount_t.mean()

    # Continuous rate and dividend yield; se via d(-ln x / T) = dx / (x T).
    T = p["T"].mean()
    return {"discount": discount, "discount_se": discount_se,
            "div_discount": div_discount, "div_discount_se": div_discount_se,
            "r": -np.log(discount) / T, "r_se": discount_se / (discount * T),
            "q": -np.log(div_discount) / T, "q_se": div_discount_se / (div_discount * T),
            "T": T, "n_pairs": len(p), "n_minutes": n_minutes,
            "intercept": (p["parity"] + discount * p["strike"]).mean()}


def parity_fit(pairs: pd.DataFrame) -> pd.DataFrame:
    """:func:`regress_parity` per expiry, shortest first, indexed by ``expiry``.

    Adds the expiry's ``option_root``, ``discount_ex_boxes`` (the fit without box
    prints) and ``days`` (= 365 T).
    """
    columns = [*regress_parity(pairs.iloc[:0]), "discount_ex_boxes", "option_root", "days"]
    rows = {}
    for expiry, g in pairs.groupby("expiry"):
        fit = regress_parity(g)
        fit["discount_ex_boxes"] = regress_parity(g, exclude_boxes=True)["discount"]
        rows[expiry] = fit
    if not rows:
        return pd.DataFrame(columns=columns).rename_axis("expiry")
    fit = pd.DataFrame(rows).T.rename_axis("expiry").astype(float)
    fit[["n_pairs", "n_minutes"]] = fit[["n_pairs", "n_minutes"]].astype(int)
    fit["option_root"] = pairs.groupby("expiry")["option_root"].first()
    fit["days"] = fit["T"] * 365
    return fit.sort_values("T")


def parity_forwards(
    bars: pd.DataFrame,
    curve: Curve,
    spot: float,
    *,
    moneyness_band: float = MONEYNESS_BAND,
    min_pairs: int = 2,
    pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One forward per expiry in ``bars``, at the close and the reference ``spot``.

    ``bars`` is :func:`~torch_pricer.market.market_data.window_bars` output; ``pairs``,
    if given, is :func:`parity_pairs` output for the same bars, so the pairs a caller
    displays are exactly the ones used here. Each pair's forward uses the curve's ``D``
    at the pair's own ``T``; each minute's forwards are averaged into a growth rate
    ``ln(F/S) / T``, and the expiry's rate is the mean over its minutes.

    Returns ``FORWARD_COLUMNS``, with ``T`` the expiry's ``T_close``, ``ratio = F/S``
    and ``source`` either ``"parity"`` or ``"interpolated"``. Raises
    :class:`MarketDataError` if no expiry has ``min_pairs`` pairs, since then there is
    nothing to interpolate from.
    """
    if pairs is None:
        pairs = parity_pairs(bars, moneyness_band)
    expiries = (
        bars.groupby("expiry", as_index=False)["T_close"].first()
        .rename(columns={"T_close": "T"}).sort_values("T")
    )
    expiries["discount"] = curve.discount(expiries["T"].to_numpy()).detach().numpy()

    discount = curve.discount(pairs["T"].to_numpy()).detach().numpy()
    forward = pairs["strike"] + pairs["parity"] / discount
    by_minute = pairs.assign(forward=forward).groupby(_BY_MINUTE).agg(
        forward=("forward", "mean"), spot=("spot", "first"), T=("T", "first"), n=("strike", "size"),
    )
    by_minute["growth"] = np.log(by_minute["forward"] / by_minute["spot"]) / by_minute["T"]
    parity = by_minute.groupby("expiry").agg(growth=("growth", "mean"), n_pairs=("n", "sum"))
    parity = parity[parity["n_pairs"] >= min_pairs].reset_index()
    if parity.empty:
        raise MarketDataError(f"no expiry has {min_pairs} same-minute call/put pairs")

    out = expiries.merge(parity, on="expiry", how="left")
    out["source"] = np.where(out["growth"].notna(), "parity", "interpolated")
    out["n_pairs"] = out["n_pairs"].fillna(0).astype(int)

    # ln(F/S) interpolated linearly in T is a carry rate held piecewise constant;
    # beyond the parity expiries, hold the nearest one's rate flat.
    known = out[out["source"] == "parity"]
    log_ratio = np.interp(out["T"], known["T"], known["growth"] * known["T"])
    log_ratio = np.where(out["T"] < known["T"].iloc[0], known["growth"].iloc[0] * out["T"], log_ratio)
    log_ratio = np.where(out["T"] > known["T"].iloc[-1], known["growth"].iloc[-1] * out["T"], log_ratio)
    out["ratio"] = np.where(out["growth"].notna(), np.exp(out["growth"] * out["T"]), np.exp(log_ratio))
    out["forward"] = spot * out["ratio"]
    return out[FORWARD_COLUMNS].reset_index(drop=True)


def parity_residuals(pairs: pd.DataFrame, forwards: pd.DataFrame, curve: Curve) -> pd.DataFrame:
    """Each pair's ``C - P`` against the line its expiry's forward implies.

    At the pair's minute the forward is that minute's spot grown at the expiry's rate,
    ``F = S exp(T ln(ratio) / T_close)``, and the line is ``D(T) (F - K)``;
    ``residual`` is how far the traded ``C - P`` sits from it, in premium.
    """
    fwd = forwards.set_index("expiry")
    growth = np.log(fwd.loc[pairs["expiry"], "ratio"].to_numpy()) / fwd.loc[pairs["expiry"], "T"].to_numpy()
    out = pairs.copy()
    out["discount"] = curve.discount(out["T"].to_numpy()).detach().numpy()
    out["expiry_forward"] = out["spot"].to_numpy() * np.exp(growth * out["T"].to_numpy())
    out["fitted"] = out["discount"] * (out["expiry_forward"] - out["strike"])
    out["residual"] = out["parity"] - out["fitted"]
    return out
