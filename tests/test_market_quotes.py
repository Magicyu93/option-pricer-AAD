"""Minute bars to quotes and forwards, on synthetic markets with known answers."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import torch

from torch_pricer.black_formula import black_price
from torch_pricer.instruments.listed import listed_product
from torch_pricer.market.curve import RateCurve
from torch_pricer.market.curve.calibration import FORWARD_COLUMNS, curve_forwards
from torch_pricer.market.market_data import reference_spot, time_to_settlement, window_bars
from torch_pricer.market.parity import parity_fit, parity_pairs
from torch_pricer.market.market_data import QuoteConfig
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.market.surface import FlatVolSurface
from torch_pricer.market.surface.calibration import option_quotes

AS_OF = dt.date(2026, 9, 8)
TZ = "America/New_York"
RATE, CARRY, VOL = 0.04, 0.012, 0.18
SPX = listed_product("SPX")
CURVE = RateCurve.from_zeros(
    torch.tensor([0.1, 1.0], dtype=torch.float64), torch.tensor([RATE, RATE], dtype=torch.float64)
)


def _years(minute: int, expiry: str) -> float:
    """ACT/365 from 15:mm on AS_OF to a PM settlement at 16:00 on ``expiry``."""
    return ((dt.date.fromisoformat(expiry) - AS_OF).days + (60 - minute) / (60 * 24)) / 365.0


def _row(minute, root, expiry, right, strike, price, spot, volume=10, hour=15):
    return {
        "session_date": AS_OF,
        "option_timestamp": pd.Timestamp(f"{AS_OF} {hour}:{minute:02d}", tz=TZ),
        "option_root": root,
        "expiration": pd.Timestamp(expiry),
        "option_type": "call" if right > 0 else "put",
        "strike": strike,
        "option_close": price,
        "option_volume": volume,
        "spot": spot,
        "matched_underlying": True,
    }


def _market(expiries=("2026-10-16", "2026-12-18"), minutes=(50, 55), strikes=None, noise=0.0, seed=0):
    """Black prices for both rights at every strike and minute, spot drifting by minute.

    Each minute is priced at its own time to settlement, as a real trade is.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for minute in minutes:
        spot = 7600.0 + (minute - 50)  # one point a minute
        for expiry in expiries:
            t = _years(minute, expiry)
            forward = spot * np.exp((RATE - CARRY) * t)
            for strike in strikes or np.arange(7300.0, 7950.0, 50.0):
                for right in (1, -1):
                    price = black_price(forward, strike, t, VOL, np.exp(-RATE * t), right)
                    price = float(price) * (1 + noise * rng.standard_normal())
                    rows.append(_row(minute, "SPXW", expiry, right, strike, price, spot))
    return pd.DataFrame(rows)


# -- window_bars ----------------------------------------------------------------

def test_window_minimum_price_and_short_expiries_are_dropped_and_counted():
    frame = pd.DataFrame([
        _row(50, "SPXW", "2026-10-16", 1, 7600, 100.0, 7600),
        _row(30, "SPXW", "2026-10-16", 1, 7600, 100.0, 7600),  # 15:30, before the window
        _row(50, "SPXW", "2026-10-16", 1, 9000, 0.03, 7600),  # under a tick
        _row(50, "SPXW", "2026-09-10", 1, 7600, 20.0, 7600),  # two days out
        _row(0, "SPXW", "2026-10-16", 1, 7600, 100.0, 7600, hour=16),  # 16:00, after the close
    ])
    bars, drops = window_bars(frame, AS_OF, SPX)
    assert len(bars) == 1
    assert drops["outside window"] == 2
    assert drops["below minimum price"] == 1
    assert drops["under 7 days to expiry"] == 1


def test_am_series_is_dropped_where_a_pm_weekly_shares_the_date_and_shortened_elsewhere():
    frame = pd.DataFrame([
        _row(50, "SPX", "2026-10-16", 1, 7600, 100.0, 7600),  # AM, PM sibling exists
        _row(50, "SPXW", "2026-10-16", 1, 7600, 101.0, 7600),
        _row(50, "SPX", "2026-12-17", 1, 7600, 200.0, 7600),  # AM only
    ])
    bars, drops = window_bars(frame, AS_OF, SPX)
    assert drops["AM series shadowed by PM weekly"] == 1
    by_root = bars.set_index("option_root")
    ten_minutes = 10 / (60 * 24 * 365)
    assert by_root.loc["SPXW", "T_close"] == pytest.approx(38 / 365)
    assert by_root.loc["SPX", "T_close"] == pytest.approx((100 - 6.5 / 24) / 365)
    # A bar at 15:50 has ten more minutes to run than the close.
    assert by_root.loc["SPXW", "T"] == pytest.approx(38 / 365 + ten_minutes)
    assert by_root.loc["SPX", "T"] == pytest.approx((100 - 6.5 / 24) / 365 + ten_minutes)
    assert by_root.loc["SPX", "settlement"] == "am"


def test_time_to_settlement_counts_the_clock_and_the_day_count():
    observed = pd.Series(pd.to_datetime(["2026-09-08 15:30", "2026-09-08 09:30"]).tz_localize(TZ))
    expiry = pd.Series([dt.date(2026, 9, 9)] * 2)
    pm = time_to_settlement(observed, expiry, pd.Series(["pm", "pm"]))
    np.testing.assert_allclose(pm, [(1 + 0.5 / 24) / 365, (1 + 6.5 / 24) / 365])
    am = time_to_settlement(observed, expiry, pd.Series(["am", "am"]), day_count="ACT/360")
    np.testing.assert_allclose(am, [(1 - 6 / 24) / 360, 1 / 360])


def test_a_bar_at_or_past_settlement_is_dropped():
    frame = pd.DataFrame([
        _row(50, "SPX", "2026-09-08", 1, 7600, 1.0, 7600),  # AM, settled this morning
        _row(50, "SPXW", "2026-09-09", 1, 7600, 1.0, 7600),  # PM, a day and ten minutes left
    ])
    bars, drops = window_bars(frame, AS_OF, SPX, QuoteConfig(min_days=0))
    assert bars["option_root"].tolist() == ["SPXW"]
    assert drops["at or past settlement"] == 1


def test_reference_spot_is_the_last_minute():
    bars, _ = window_bars(_market(minutes=(50, 58)), AS_OF, SPX)
    assert reference_spot(bars) == 7608.0


# -- parity -----------------------------------------------------------------------

def test_pairs_stay_in_the_band_with_two_strikes_a_minute():
    bars, _ = window_bars(_market(strikes=[6500.0, 7550.0, 7600.0, 8000.0]), AS_OF, SPX)
    pairs = parity_pairs(bars, moneyness_band=0.05)
    assert set(pairs["strike"]) == {7550.0, 7600.0}  # 6500 and 8000 are over 5% away
    assert (pairs.groupby(["expiry", "timestamp"]).size() >= 2).all()
    # Two strikes a minute, every leg ten lots: exactly what a box-spread print looks like.
    assert pairs["is_box"].all()
    assert not parity_pairs(bars, moneyness_band=0.10)["is_box"].any()  # three strikes a minute


def test_the_regression_recovers_the_rate_and_the_carry():
    bars, _ = window_bars(_market(minutes=(46, 50, 58)), AS_OF, SPX)
    fit = parity_fit(parity_pairs(bars))
    np.testing.assert_allclose(fit["r"], RATE, atol=1e-4)
    np.testing.assert_allclose(fit["q"], CARRY, atol=1e-4)
    assert (fit["n_minutes"] == 3).all()

# -- curve_forwards -------------------------------------------------------------

DIVIDEND = RateCurve.from_zeros([1.0], [CARRY], "flat_q")  # flat, in float64


def test_curve_forwards_grow_at_r_minus_q_to_the_close():
    bars, _ = window_bars(_market(minutes=(46, 50, 58)), AS_OF, SPX)
    spot = reference_spot(bars)
    fwd = curve_forwards(bars, CURVE, DIVIDEND, spot)
    assert list(fwd.columns) == FORWARD_COLUMNS
    assert list(fwd["expiry"]) == sorted(bars["expiry"].unique())
    np.testing.assert_allclose(fwd["T"], bars.groupby("expiry")["T_close"].first().sort_index())
    np.testing.assert_allclose(fwd["forward"], spot * np.exp((RATE - CARRY) * fwd["T"]), rtol=1e-12)
    np.testing.assert_allclose(fwd["discount"], np.exp(-RATE * fwd["T"]), rtol=1e-12)


def test_curve_forwards_are_the_snapshots_own():
    bars, _ = window_bars(_market(), AS_OF, SPX)
    spot = reference_spot(bars)
    dividend = RateCurve.from_zeros([0.05, 0.5], [0.0, 0.03], label="rising")
    fwd = curve_forwards(bars, CURVE, dividend, spot)
    snapshot = MarketSnapshot(
        ticker="SPX", as_of=AS_OF, spot=torch.tensor(spot, dtype=torch.float64),
        discount=CURVE, dividend=dividend, vol_surface=FlatVolSurface(VOL, AS_OF),
    )
    for row in fwd.itertuples():
        assert float(snapshot.forward(row.T).detach()) == pytest.approx(row.forward, rel=1e-12)


# -- option_quotes --------------------------------------------------------------

def test_quotes_recover_the_vol_keep_only_out_of_the_money_and_sum_volume():
    bars, _ = window_bars(_market(minutes=(50, 55)), AS_OF, SPX)
    fwd = curve_forwards(bars, CURVE, DIVIDEND, reference_spot(bars))
    quotes, used, drops = option_quotes(bars, fwd)

    assert drops["in the money"] > 0
    assert all((q.right.sign > 0) == (np.log(q.strike / f) >= 0)
               for q in quotes
               for f in [fwd.set_index("expiry").loc[q.expiry, "forward"]])
    assert [q.implied_vol for q in quotes] == pytest.approx([VOL] * len(quotes), abs=1e-6)
    assert all(q.volume == 20 for q in quotes)  # two minutes of ten contracts
