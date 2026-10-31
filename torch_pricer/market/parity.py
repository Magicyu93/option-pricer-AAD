"""Put-call parity on traded minute bars: the forwards, and a regression kept as a diagnostic.

For a European call and put on the same strike and expiry,
``C - P = D (F - K)`` with ``D = exp(-r T)``. Given ``D``, every same-minute pair is a
forward, ``F = K + (C - P) / D``.

**The forwards** (:func:`forward_pairs`, :func:`parity_forwards`) take ``D`` from the
discount curve -- Treasuries -- never from the trades: backing ``r`` out of parity is far
too noisy at short expiries. Near the money the choice of ``D`` barely matters, since
an error in it moves the forward by ``(F - K) dD / D``. Which pairs count is set by
:class:`ParityConfig`, every filter a named field; :func:`forward_pairs` labels each pair
with the filter that drops it. Each expiry's forward is then one robust number with a
standard error, in cumulative carry ``y = -ln(F D / S)``, ready for
:func:`~torch_pricer.market.curve.calibration.implied_dividend_curve`.

Trades are not simultaneous, and the index moves, so a call and a put only make a pair
when they traded in the same minute, each marked against that minute's spot and time to
settlement. The forward drifts during the day relative to spot, so late pairs set an
expiry's forward (the *close* tier) and earlier ones only fill expiries without late
pairs (the *day* tier), with a wider standard error.

**The regression** (:func:`parity_pairs`, :func:`regress_parity`, :func:`parity_fit`)
estimates ``D`` and the intercept from the trades alone: OLS of ``C - P`` on ``K`` with one
intercept per minute, by demeaning within each minute,
``-D = sum(K_dm * P_dm) / sum(K_dm^2)``. Then ``r = -ln(D) / T`` and
``q = -ln(A_t / S_t) / T``. A diagnostic only: it shows how far the Treasury curve sits
from what the trades imply, and is never an input.

Box-spread prints -- two strikes in a minute, all four legs with the same volume -- are
flagged ``is_box``: traded as one package, they price the box financing rate, which runs
above T-bills.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from torch_pricer.market.curve.base import Curve
from torch_pricer.market.market_data import TIMEZONE

#: Keep pairs with ``|K / S - 1|`` below this: far-wing trade prints are stale.
MONEYNESS_BAND = 0.10
PAIR_COLUMNS = [
    "expiry", "timestamp", "option_root", "strike", "call", "put", "call_volume", "put_volume",
    "spot", "days", "T", "T_close", "parity", "is_box",
]
_BY_MINUTE = ["expiry", "timestamp"]


def _same_minute_pairs(bars: pd.DataFrame) -> pd.DataFrame:
    """Every call and put at the same expiry, strike and minute, with ``parity = C - P``."""
    keys = ["expiry", "timestamp", "strike"]
    shared = ["option_root", "spot", "days", "T", "T_close"]
    calls = bars.loc[bars["right"] > 0, [*keys, *shared, "price", "volume"]]
    puts = bars.loc[bars["right"] < 0, [*keys, "price", "volume"]]
    pairs = calls.merge(puts, on=keys, suffixes=("_call", "_put")).rename(columns={
        "price_call": "call", "price_put": "put", "volume_call": "call_volume", "volume_put": "put_volume",
    })
    pairs["parity"] = pairs["call"] - pairs["put"]
    return pairs


def _is_box(pairs: pd.DataFrame) -> pd.Series:
    """Two strikes in a minute, all four legs with one volume: a box traded as a package."""
    grouped = pairs.groupby(_BY_MINUTE)
    one_volume = (
        (pairs["call_volume"] == pairs["put_volume"])
        & (grouped["call_volume"].transform("nunique") == 1)
        & (grouped["put_volume"].transform("nunique") == 1)
    )
    return (grouped["strike"].transform("size") == 2) & one_volume


def parity_pairs(bars: pd.DataFrame, moneyness_band: float = MONEYNESS_BAND) -> pd.DataFrame:
    """Every same-minute call/put pair within ``moneyness_band`` of that minute's spot.

    ``bars`` is :func:`~torch_pricer.market.market_data.window_bars` output. One row
    per pair, ``parity = C - P``, keeping only minutes of an expiry with two or more
    strikes. Columns are ``PAIR_COLUMNS``.
    """
    pairs = _same_minute_pairs(bars)
    pairs = pairs[(pairs["strike"] / pairs["spot"] - 1).abs() < moneyness_band]
    pairs = pairs[pairs.groupby(_BY_MINUTE)["strike"].transform("size") >= 2].copy()

    pairs["is_box"] = _is_box(pairs)
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


# -- forwards: Treasury discount, one robust forward per expiry ----------------------------

#: The filters :func:`forward_pairs` applies, in order: a pair is labelled with the first
#: one that drops it, so each pair is counted once.
DROP_REASONS = (
    "before the window",
    "box print",
    "leg volume under the minimum",
    "leg price under the minimum",
    "outside the moneyness band",
    "too few strikes in its minute",
)
ESTIMATORS = ("median", "mean")


@dataclass(frozen=True)
class ParityConfig:
    """Which same-minute call/put pairs set each expiry's forward, and how.

    Every filter :func:`forward_pairs` applies is a field here, checked in the order of
    ``DROP_REASONS``; :meth:`describe` says them in words.
    """

    # -- when: ET clock times, "HH:MM"
    #: End of the window, exclusive, as in ``QuoteConfig``.
    window_end: str = "16:00"
    #: Pairs from here to the end set an expiry's forward: the *close* tier.
    close_start: str = "15:00"
    #: Earlier pairs, back to here, fill expiries with too few close minutes: the *day*
    #: tier. ``None`` turns the fill off, leaving those expiries to the curve.
    fill_start: str | None = "09:30"

    # -- which pairs
    #: Drop box-spread prints, which price the box financing rate, not the forward.
    exclude_boxes: bool = True
    #: Contracts traded in the minute, on each leg.
    min_volume: int = 1
    #: Price of each leg; a leg below one tick is noise.
    min_price: float = 0.0
    #: Keep ``|ln(K / S)| <= max(band_sd * sqrt(T), band_floor)``, with the pair's own
    #: minute's spot: a band in standard deviations (at 100% vol), widening with maturity.
    band_sd: float = 0.10
    #: The band's floor, so very short expiries keep more than one strike.
    band_floor: float = 0.01
    #: Strikes left in the pair's minute after the filters above.
    min_strikes: int = 2

    # -- one forward per expiry
    #: ``"median"`` or ``"mean"`` of the pairs' cumulative carry.
    estimator: str = "median"
    #: Close-tier minutes an expiry needs for a close forward; fewer, and it is filled
    #: from the day (or left to the curve when ``fill_start`` is ``None``).
    min_minutes: int = 2
    #: Smallest standard error, in index points: one minute's spread says little.
    se_floor_pts: float = 0.5
    #: Added in quadrature to a day-tier forward's standard error, in index points: the
    #: forward drifts relative to spot during the day.
    drift_pts: float = 1.0

    def __post_init__(self):
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}, got {self.estimator!r}")
        if self.close_start >= self.window_end:
            raise ValueError(f"close_start {self.close_start} is not before window_end {self.window_end}")
        if self.fill_start is not None and self.fill_start > self.close_start:
            raise ValueError(f"fill_start {self.fill_start} is after close_start {self.close_start}")

    @property
    def data_start(self) -> str:
        """The earliest minute any pair can come from."""
        return self.fill_start or self.close_start

    def describe(self) -> list[str]:
        """The filters and rules, one line each, in the order they apply."""
        fill = (f"expiries with fewer than {self.min_minutes} close minutes are filled from pairs since "
                f"{self.fill_start} ET, standard error widened by {self.drift_pts:g} pts for the intraday drift"
                if self.fill_start else
                f"expiries with fewer than {self.min_minutes} close minutes get no forward of their own")
        return [
            f"close tier: pairs from {self.close_start} to before {self.window_end} ET",
            fill,
            "box prints dropped" if self.exclude_boxes else "box prints kept",
            f"each leg: volume ≥ {self.min_volume}, price ≥ {self.min_price:g}",
            f"moneyness: |ln(K/S)| ≤ max({self.band_sd:g}·√T, {self.band_floor:.1%})",
            f"at least {self.min_strikes} strikes left in the pair's minute",
            f"forward per expiry: {self.estimator} over its pairs, standard error ≥ {self.se_floor_pts:g} pts",
        ]


#: :func:`parity_forwards` output: one row per expiry.
PARITY_FORWARD_COLUMNS = [
    "expiry", "option_root", "T", "discount", "y", "y_se", "ratio", "forward", "forward_se", "spot",
    "n_pairs", "n_minutes", "tier",
]


def forward_pairs(bars: pd.DataFrame, discount: Curve, config: ParityConfig = ParityConfig()) -> pd.DataFrame:
    """Every same-minute call/put pair in ``bars``, its forward, and what, if anything, drops it.

    ``bars`` is :func:`~torch_pricer.market.market_data.window_bars` output covering
    ``config.data_start`` to ``config.window_end``, with no price or expiry floor. Each
    pair gets:

    * ``D = discount(T)``, ``forward = K + (C - P) / D`` and ``ratio = forward / spot``,
      all at the pair's own minute;
    * ``y``: the cumulative carry to the close, ``-ln(ratio D) * T_close / T`` -- the carry
      to the pair's ``T``, at a flat rate over the minutes left until the close;
    * ``D_close = discount(T_close)``, ``band`` (the moneyness band at its ``T``) and
      ``tier``, ``"close"`` or ``"day"`` by its minute;
    * ``dropped_by``: the first of ``DROP_REASONS`` that drops it, ``""`` if kept.
    """
    pairs = _same_minute_pairs(bars)
    pairs["is_box"] = _is_box(pairs) if len(pairs) else pd.Series(dtype=bool)
    t = pairs["T"].to_numpy(dtype=float, copy=True)
    t_close = pairs["T_close"].to_numpy(dtype=float, copy=True)
    pairs["D"] = discount.discount(t).detach().numpy()
    pairs["D_close"] = discount.discount(t_close).detach().numpy()
    pairs["forward"] = pairs["strike"] + pairs["parity"] / pairs["D"]
    pairs["ratio"] = pairs["forward"] / pairs["spot"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pairs["y"] = -np.log(pairs["ratio"] * pairs["D"]) * t_close / t
    pairs["band"] = np.maximum(config.band_sd * np.sqrt(t), config.band_floor)
    clock = pairs["timestamp"].dt.tz_convert(TIMEZONE).dt.strftime("%H:%M")
    pairs["tier"] = np.where(clock >= config.close_start, "close", "day")

    dropped = pd.Series("", index=pairs.index, dtype=object)

    def drop(mask, reason):
        dropped[(dropped == "") & mask] = reason

    drop((clock < config.data_start) | (clock >= config.window_end), DROP_REASONS[0])
    drop(pairs["is_box"] & config.exclude_boxes, DROP_REASONS[1])
    drop(pairs[["call_volume", "put_volume"]].min(axis=1) < config.min_volume, DROP_REASONS[2])
    drop(pairs[["call", "put"]].min(axis=1) < config.min_price, DROP_REASONS[3])
    drop(np.log(pairs["strike"] / pairs["spot"]).abs() > pairs["band"], DROP_REASONS[4])
    alive = dropped == ""
    strikes = pairs["strike"].where(alive).groupby([pairs["expiry"], pairs["timestamp"]]).transform("count")
    drop(strikes < config.min_strikes, DROP_REASONS[5])
    pairs["dropped_by"] = dropped
    return pairs.sort_values(["expiry", "timestamp", "strike"]).reset_index(drop=True)


def parity_forwards(pairs: pd.DataFrame, config: ParityConfig = ParityConfig()) -> pd.DataFrame:
    """One forward per expiry from the kept pairs of :func:`forward_pairs`.

    An expiry with ``config.min_minutes`` or more close-tier minutes gets the ``estimator``
    of its close pairs' ``y`` (tier ``"close"``); otherwise, if ``config.fill_start`` is set,
    that of all its kept pairs (tier ``"day"``). ``y_se`` is the spread of the per-minute
    estimates over sqrt(minutes), floored at ``se_floor_pts``, and for the day tier
    widened by ``drift_pts`` in quadrature. ``forward = spot * ratio`` and ``forward_se`` are
    in index points at one ``spot`` for every expiry, the last kept minute's, so forwards
    estimated from different parts of the day compare. Expiries settling at the close
    (``T_close = 0``) have no carry left and are skipped. Returns
    ``PARITY_FORWARD_COLUMNS``, sorted by ``T``.
    """
    kept = pairs[(pairs["dropped_by"] == "") & (pairs["T_close"] > 0)]
    if kept.empty:
        return pd.DataFrame(columns=PARITY_FORWARD_COLUMNS)
    spot = float(kept.loc[kept["timestamp"].idxmax(), "spot"])
    floor = config.se_floor_pts / spot
    rows = []
    for expiry, g in kept.groupby("expiry"):
        close = g[g["tier"] == "close"]
        if close["timestamp"].nunique() >= config.min_minutes:
            tier, used = "close", close
        elif config.fill_start is not None:
            tier, used = "day", g
        else:
            continue
        by_minute = used.groupby("timestamp")["y"].agg(config.estimator)
        se = by_minute.std() / np.sqrt(len(by_minute)) if len(by_minute) > 1 else np.nan
        se = max(se, floor) if np.isfinite(se) else floor
        if tier == "day":
            se = float(np.hypot(se, config.drift_pts / spot))
        y = float(used["y"].agg(config.estimator))
        d_close = float(used["D_close"].iloc[0])
        ratio = np.exp(-y) / d_close
        rows.append({
            "expiry": expiry, "option_root": used["option_root"].iloc[0], "T": float(used["T_close"].iloc[0]),
            "discount": d_close, "y": y, "y_se": se, "ratio": ratio, "forward": spot * ratio,
            "forward_se": spot * ratio * se, "spot": spot, "n_pairs": len(used),
            "n_minutes": len(by_minute), "tier": tier,
        })
    return pd.DataFrame(rows, columns=PARITY_FORWARD_COLUMNS).sort_values("T").reset_index(drop=True)
