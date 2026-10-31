"""The app's analysis, on the library's window and parity, reproduces scripts/vol_surface_visual.ipynb.

The day and yields are the ones the notebook ran on, because the expected counts are the
notebook's output. The notebook kept the 16:00 bar (``<= 16:00``); the library's window
ends before it, since that bar's trades come after the index's closing print. With the
end moved to 16:01 the library reproduces the notebook exactly. Skipped unless that day's
SPX options are cached.
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.coverage import cached_days

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
DAY = "2026-09-10"
NOTEBOOK_Q = 0.011  # the notebook's flat dividend yield: its counts depend on it
pytestmark = pytest.mark.skipif(DAY not in cached_days("options", "SPX"), reason=f"no SPX options cached for {DAY}")


@pytest.fixture(scope="module")
def day():
    from data_pipeline import MassiveHistoricalDataLoader, RatesDataLoader
    from torch_pricer.market.curve import treasury_zero_curve

    market = MassiveHistoricalDataLoader(verbose=False).load_market([DAY], ["SPX"])
    par = RatesDataLoader("massive").yields_as_of("2026-09-08", lag_days=0)
    return market, treasury_zero_curve(par.tenors, par.yields_pct)


def _run(day, window_end: str):
    import analysis
    from torch_pricer.instruments.listed import listed_product
    from torch_pricer.market.market_data import QuoteConfig, window_bars
    from torch_pricer.market.parity import parity_pairs

    market, curve = day
    config = QuoteConfig(window_start="15:30", window_end=window_end, min_days=0, min_price=0.0)
    bars, _ = window_bars(market, dt.date.fromisoformat(DAY), listed_product("SPX"), config)
    bars = bars[bars.option_root == "SPXW"]
    ivs = analysis.contract_implied_vols(bars, curve, _flat_q())
    return parity_pairs(bars), ivs, analysis.contract_vol_summary(ivs, combine=False)


def _flat_q():
    from torch_pricer.market.curve import RateCurve

    return RateCurve.from_zeros([1.0], [NOTEBOOK_Q], "flat_q")


def test_with_the_16_00_bar_the_notebook_is_reproduced(day):
    pairs, ivs, vols = _run(day, "16:01")
    assert (len(pairs), int(pairs.is_box.sum()), pairs.expiry.nunique()) == (1357, 0, 9)
    assert (int(ivs.iv.notna().sum()), len(ivs)) == (12483, 12857)
    assert (len(vols), int(vols.otm.sum())) == (12483, 10371)


def test_the_window_ends_before_the_16_00_bar(day):
    pairs, ivs, vols = _run(day, "16:00")
    assert (len(pairs), int(pairs.is_box.sum()), pairs.expiry.nunique()) == (1340, 0, 9)
    assert (int(ivs.iv.notna().sum()), len(ivs)) == (12143, 12514)
    assert int(vols.otm.sum()) == 10071
    assert ivs.timestamp.dt.tz_convert("America/New_York").max().strftime("%H:%M") == "15:59"


@pytest.fixture(scope="module")
def parity_day(day):
    """The library's parity forwards and dividend curve on the whole day, SPXW only."""
    from torch_pricer.instruments.listed import listed_product
    from torch_pricer.market.curve.calibration import implied_dividend_curve
    from torch_pricer.market.market_data import QuoteConfig, window_bars
    from torch_pricer.market.parity import ParityConfig, forward_pairs, parity_forwards

    market, curve = day
    config = ParityConfig()
    wide = QuoteConfig(window_start=config.data_start, window_end=config.window_end, min_days=0, min_price=0.0)
    bars, _ = window_bars(market, dt.date.fromisoformat(DAY), listed_product("SPX"), wide)
    pairs = forward_pairs(bars[bars.option_root == "SPXW"], curve, config)
    fw = parity_forwards(pairs, config)
    return pairs, fw, implied_dividend_curve(fw)


def test_expiry_table_has_one_row_per_expiry(day, parity_day):
    import analysis
    from torch_pricer.market.parity import parity_fit

    pairs, _, vols = _run(day, "16:00")
    _, fw, dividend = parity_day
    fit = parity_fit(pairs)
    points = analysis.select_points(vols, "both")
    iv_pairs = analysis.call_put_pairs(vols)
    table = analysis.expiry_table(pairs, fit, iv_pairs, points, day[1], dividend, fw)
    assert len(table) == len(set(fit.index) | set(points.expiry) | set(fw.expiry)) == table.expiry.nunique()
    assert table.forward_gap.notna().sum() == len(fit)
    assert table.option_root.eq("SPXW").all()
    assert set(table.tier) <= {"close", "day", "curve"} and (table.tier != "curve").sum() == len(fw)
    # F_pair is the pair's forward at the same Treasury D the ivs use.
    np.testing.assert_allclose(iv_pairs.F_pair, iv_pairs.strike + iv_pairs.parity / iv_pairs.discount)


def test_the_parity_curve_centres_the_call_put_gap(day, parity_day):
    import analysis
    from torch_pricer.instruments.listed import listed_product
    from torch_pricer.market.market_data import QuoteConfig, window_bars

    market, curve = day
    dividend = parity_day[2]
    config = QuoteConfig(window_start="15:30", min_days=0, min_price=0.0)
    window, _ = window_bars(market, dt.date.fromisoformat(DAY), listed_product("SPX"), config)
    window = window[window.option_root == "SPXW"]
    vols = analysis.contract_vol_summary(analysis.contract_implied_vols(window, curve, dividend), combine=False)
    gap = analysis.call_put_pairs(vols, min_days=1.0)
    near = gap[analysis.near_the_money(gap, 0.05)]
    assert len(near) > 50
    assert abs(near.iv_gap.median()) < 0.1  # vol points


def test_the_forward_views_account_for_every_pair(day, parity_day):
    import analysis

    pairs, fw, dividend = parity_day
    funnel = analysis.filter_funnel(pairs)
    assert funnel.total.sum() == len(pairs)
    assert funnel.set_index("step").total["kept"] == (pairs.dropped_by == "").sum()
    expiry = fw.expiry.iloc[len(fw) // 2]
    one, minutes = analysis.intraday_forward(pairs, expiry, day[1], dividend)
    assert len(one) == (pairs.expiry == expiry).sum() and minutes.n.sum() == (one.dropped_by == "").sum()


@pytest.fixture(scope="module")
def surface_fits(day):
    from analysis import surface_models as sm
    from torch_pricer.instruments.listed import listed_product
    from torch_pricer.market.market_data import QuoteConfig, window_bars

    market, curve = day
    wide = QuoteConfig(window_start="15:30", min_days=0, min_price=0.0)
    bars, _ = window_bars(market, dt.date.fromisoformat(DAY), listed_product("SPX"), wide)
    bars = bars[bars.option_root == "SPXW"]
    inputs = sm.fit_inputs(bars, curve, QuoteConfig(window_start="15:30"), _flat_q())
    fits = [sm.fit_model(m, inputs, dt.date.fromisoformat(DAY)) for m in ("SSVI", "eSSVI")]
    return bars, curve, inputs, fits, sm.residuals(fits)


def test_surface_models_are_fitted_to_the_same_quotes(surface_fits):
    from analysis import surface_models as sm

    _, _, inputs, fits, resid = surface_fits
    assert inputs.dropped["under min days"] > 0  # the fit's floor is applied to the wide window
    assert resid.days.min() >= 7
    summary = sm.model_summary(fits).set_index("model")
    assert list(summary.index) == ["SSVI", "eSSVI"]
    assert (summary.quotes == len(inputs.quotes)).all()
    assert summary.loc["eSSVI", "rmse"] < summary.loc["SSVI", "rmse"]
    assert summary.arbitrage_free.all()
    per_expiry = sm.per_expiry_errors(resid)
    assert len(per_expiry) == 2 * resid.expiry.nunique()


def test_surface_model_grids(surface_fits):
    from analysis import surface_models as sm

    _, _, _, fits, resid = surface_fits
    expiries = sorted(resid.expiry.unique())[:3]
    curves = sm.smile_curves(fits, resid, expiries, n=11)
    assert len(curves) == 2 * 3 * 11 and curves.iv.notna().all()
    mesh = sm.surface_mesh(fits[1], resid, n_x=5, n_t=7)
    assert mesh["iv"].shape == mesh["k"].shape and (mesh["iv"] > 0).all()
    grid = sm.local_vol_grid(fits[1], resid, n_x=7, n_t=9)
    assert (grid["variance"] >= 0).all()  # eSSVI is arbitrage free, so Dupire's variance is too
    lines, atm = sm.atm_term_structure(fits, resid)
    assert set(lines.model) == {"SSVI", "eSSVI"} and atm.iv.between(0.05, 1.0).all()
    table, scalars = sm.parameter_table(fits[1])
    assert len(table) == resid.expiry.nunique() and {"theta", "rho", "psi", "atm_vol"} <= set(table)
    assert set(sm.parameter_table(fits[0])[1]) == {"rho", "eta", "gamma"}


def test_the_fits_forwards_come_from_the_dividend_curve_given(surface_fits):
    from torch_pricer.market.curve.calibration import FORWARD_COLUMNS

    _, curve, inputs, _, _ = surface_fits
    fw = inputs.forwards
    assert list(fw.columns) == FORWARD_COLUMNS
    fw = fw[fw["T"] > 0]  # the same-day expiry has no time left at the close
    r = -np.log(fw.discount) / fw["T"]
    np.testing.assert_allclose(fw.forward, inputs.spot * np.exp((r - NOTEBOOK_Q) * fw["T"]), rtol=1e-12)
