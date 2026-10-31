"""Put-call parity forwards: each filter, the two tiers, and the monotone dividend curve."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tests.test_market_quotes import AS_OF, CARRY, CURVE, RATE, SPX, _market
from torch_pricer.errors import MarketDataError
from torch_pricer.market.curve.calibration import curve_forwards, implied_dividend_curve, isotonic
from torch_pricer.market.market_data import QuoteConfig, window_bars
from torch_pricer.market.parity import (
    DROP_REASONS,
    PARITY_FORWARD_COLUMNS,
    ParityConfig,
    forward_pairs,
    parity_forwards,
)

WIDE = QuoteConfig(window_start="09:30", window_end="16:00", min_days=0, min_price=0.0)


def _pairs(config=ParityConfig(), **market):
    bars, _ = window_bars(_market(**{"minutes": (48, 52, 56), **market}), AS_OF, SPX, WIDE)
    return forward_pairs(bars, CURVE, config)


def _reasons(pairs) -> dict:
    return pairs["dropped_by"].value_counts().to_dict()


# -- forward_pairs: one label per pair, the first filter that drops it --------------------

def test_every_pair_is_a_forward_at_the_treasury_discount():
    pairs = _pairs()
    expected = pairs.spot * np.exp((RATE - CARRY) * pairs["T"])
    np.testing.assert_allclose(pairs.forward, expected, rtol=1e-12)
    np.testing.assert_allclose(pairs.y, CARRY * pairs.T_close, rtol=1e-9)
    assert set(pairs.dropped_by) <= {"", *DROP_REASONS}


def test_the_band_widens_with_sqrt_t_and_has_a_floor():
    pairs = _pairs()
    x = np.log(pairs.strike / pairs.spot).abs()
    kept = pairs.dropped_by == ""
    assert (x[kept] <= pairs.band[kept]).all()
    assert (pairs.dropped_by[~kept] == "outside the moneyness band").all() and (~kept).any()
    np.testing.assert_allclose(pairs.band, np.maximum(0.1 * np.sqrt(pairs["T"]), 0.01))
    floor_only = _pairs(ParityConfig(band_sd=0.0, band_floor=0.004))
    assert (floor_only.band == 0.004).all()


def test_volume_price_and_strike_count_filters_are_labelled():
    every = len(_pairs())  # 13 strikes, 2 expiries, 3 minutes
    assert _reasons(_pairs(ParityConfig(min_volume=11))) == {"leg volume under the minimum": every}
    cheap = _pairs(ParityConfig(min_price=60.0, band_sd=1.0))
    assert cheap.dropped_by.eq("leg price under the minimum").any()
    assert (cheap[cheap.dropped_by == ""][["call", "put"]].min(axis=1) >= 60).all()
    # A band so narrow only the strike nearest spot survives: one strike is too few.
    lonely = _pairs(ParityConfig(band_sd=0.0, band_floor=0.0035))
    assert set(lonely.dropped_by) == {"outside the moneyness band", "too few strikes in its minute"}
    assert _pairs(ParityConfig(band_sd=0.0, band_floor=0.0035, min_strikes=1)).dropped_by.eq("").any()


def test_box_prints_are_dropped_unless_kept_on_purpose():
    box = dict(minutes=(50,), strikes=[7550.0, 7650.0])  # two strikes, every leg the same volume
    assert set(_pairs(**box).dropped_by) == {"box print"}
    assert set(_pairs(ParityConfig(exclude_boxes=False), **box).dropped_by) == {""}


def test_pairs_outside_the_configured_window_are_dropped():
    late = ParityConfig(close_start="15:52", fill_start=None)
    pairs = _pairs(late)
    clock = pairs.timestamp.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    assert (pairs.dropped_by[clock < "15:52"] == "before the window").all()
    assert (pairs.dropped_by[clock >= "15:52"] != "before the window").all()


# -- parity_forwards: close tier, day tier, standard errors -------------------------------

def test_close_forwards_recover_the_carry_with_a_floored_se():
    fw = parity_forwards(_pairs())
    assert list(fw.columns) == PARITY_FORWARD_COLUMNS
    assert list(fw.tier) == ["close", "close"] and (fw.n_minutes == 3).all()
    np.testing.assert_allclose(fw.y, CARRY * fw["T"], rtol=1e-9)
    np.testing.assert_allclose(fw.forward_se / fw.forward, 0.5 / fw.spot, rtol=1e-9)  # noiseless: the floor


def test_expiries_short_of_close_minutes_are_filled_from_the_day_or_left_out():
    config = ParityConfig(close_start="15:55")  # only the 15:56 minute is close
    fw = parity_forwards(_pairs(config), config)
    assert list(fw.tier) == ["day", "day"] and (fw.n_minutes == 3).all()
    np.testing.assert_allclose(fw.forward_se / fw.forward, np.hypot(0.5, 1.0) / fw.spot, rtol=1e-9)
    no_fill = dataclasses.replace(config, fill_start=None)
    assert parity_forwards(_pairs(no_fill), no_fill).empty


def test_morning_pairs_only_reach_the_day_tier():
    morning = _market(minutes=(10, 20), hour=11)
    late = _market(expiries=("2026-10-16",), minutes=(48, 52, 56))
    bars, _ = window_bars(pd.concat([morning, late]), AS_OF, SPX, WIDE)
    fw = parity_forwards(forward_pairs(bars, CURVE)).set_index("expiry")
    assert fw.tier.to_dict() == {pd.Timestamp("2026-10-16").date(): "close", pd.Timestamp("2026-12-18").date(): "day"}
    assert fw.n_minutes.tolist() == [3, 2]


def test_the_median_shrugs_off_a_stale_leg():
    pairs = _pairs()
    first = pairs.index[pairs.dropped_by == ""][0]
    pairs.loc[first, "y"] += 0.05  # one pair 5% of spot off
    fw = parity_forwards(pairs)
    np.testing.assert_allclose(fw.y, CARRY * fw["T"], rtol=1e-9)
    mean = parity_forwards(pairs, ParityConfig(estimator="mean"))
    assert not np.allclose(mean.y, CARRY * mean["T"], rtol=1e-6)


def test_noisy_prices_still_give_the_forward_within_its_se():
    fw = parity_forwards(_pairs(noise=0.002, minutes=tuple(range(40, 60))))
    err = (fw.y - CARRY * fw["T"]) * fw.spot
    assert (err.abs() < 3 * fw.forward_se).all()


def test_config_is_checked_and_described():
    with pytest.raises(ValueError, match="estimator"):
        ParityConfig(estimator="mode")
    with pytest.raises(ValueError, match="after close_start"):
        ParityConfig(fill_start="15:30", close_start="15:00")
    lines = ParityConfig().describe()
    assert any("√T" in line for line in lines) and any("09:30" in line for line in lines)


# -- the curve ------------------------------------------------------------------------------

def test_isotonic_keeps_a_rising_sequence_and_pools_a_drop():
    np.testing.assert_array_equal(isotonic([1, 2, 3], [1, 1, 1]), [1, 2, 3])
    np.testing.assert_allclose(isotonic([1, 3, 2, 4], [1, 1, 3, 1]), [1, 2.25, 2.25, 4])
    out = isotonic([0.5, -1, 2, 1.5, 1, 3], np.ones(6))
    assert (np.diff(out) >= 0).all()


def _fw(t, y, se):
    return pd.DataFrame({"T": t, "y": y, "y_se": se})


def test_the_curve_runs_through_monotone_forwards_and_pools_the_rest():
    curve = implied_dividend_curve(_fw([0.1, 0.5, 1.0], [0.001, 0.005, 0.012], [1e-4] * 3))
    t = np.array([0.1, 0.5, 1.0])
    np.testing.assert_allclose(curve.zero_rate(t).detach().numpy() * t, [0.001, 0.005, 0.012], rtol=1e-12)
    # The middle pillar dips; the precise outer ones hold, the noisy one gives way.
    pooled = implied_dividend_curve(_fw([0.1, 0.5, 1.0], [0.004, 0.003, 0.012], [1e-4, 1e-3, 1e-4]))
    y = pooled.zero_rate(np.array([0.1, 0.5])).detach().numpy() * np.array([0.1, 0.5])
    assert y[0] == pytest.approx(y[1]) and 0.003 < y[0] < 0.004 and y[0] == pytest.approx(0.004, abs=1e-4)


def test_negative_short_carry_is_allowed_and_no_forwards_is_refused():
    curve = implied_dividend_curve(_fw([0.01, 0.5], [-0.0001, 0.004], [1e-4, 1e-4]))
    assert float(curve.zero_rate(0.01).detach()) < 0
    with pytest.raises(MarketDataError):
        implied_dividend_curve(_fw([], [], []))


def test_the_curve_reproduces_each_close_forward_in_curve_forwards():
    pairs = _pairs(carry={"2026-10-16": 0.002, "2026-12-18": 0.015})
    fw = parity_forwards(pairs)
    bars, _ = window_bars(_market(minutes=(56,)), AS_OF, SPX, WIDE)
    out = curve_forwards(bars, CURVE, implied_dividend_curve(fw), spot=7606.0)
    np.testing.assert_allclose(out.ratio, fw.ratio, rtol=1e-12)
