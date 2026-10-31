"""build_snapshot: the steps chain, the trace records them, and refusals are explicit."""

import dataclasses
import datetime as dt
import operator

import numpy as np
import pytest

from tests.test_market_quotes import AS_OF, CARRY, RATE, VOL, _market
from torch_pricer.errors import MarketDataError, ValidationError
from torch_pricer.market.build import SnapshotSettings, build_snapshot, drop_counts
from torch_pricer.market.curve.calibration import par_yield_quotes
from torch_pricer.instruments.listed import listed_product

# A flat 4% bill curve, as simple-interest par yields (what a 1M/3M bill quotes).
TENORS = [1 / 12, 0.25]
RATE_QUOTES = par_yield_quotes(
    TENORS, [100 * np.expm1(RATE * t) / t for t in TENORS], as_of=AS_OF - dt.timedelta(days=1)
)
STEPS = [
    "Observation window", "Reference spot", "Discount curve", "Dividend curve",
    "Forwards", "Option quotes", "Surface fit", "Market snapshot",
]


@pytest.fixture(scope="module")
def run():
    return build_snapshot(_market(minutes=(48, 52, 56)), RATE_QUOTES, listed_product("SPX"), SnapshotSettings(AS_OF))


def test_every_step_is_recorded_in_order_with_its_function_and_artifact(run):
    assert [s.name for s in run.trace] == STEPS
    for step in run.trace:
        assert step.function.startswith("torch_pricer."), step
        assert operator.attrgetter(step.artifact)(run) is not None, step
        assert step.seconds >= 0


def test_the_snapshot_reproduces_the_market_it_was_built_from(run):
    spot = run.observations.spot.value
    for row in run.forwards.itertuples():
        expected = spot * np.exp((RATE - CARRY) * row.T)
        assert float(run.snapshot.forward(row.T).detach()) == pytest.approx(expected, rel=1e-9)
    assert float(run.snapshot.discount.zero_rate(0.2).detach()) == pytest.approx(RATE, abs=1e-12)
    # The synthetic market is flat at VOL, so the surface is too.
    t = float(run.forwards["T"].iloc[0])
    assert float(run.snapshot.vol(spot, t).detach()) == pytest.approx(VOL, abs=1e-4)
    assert run.result.converged


def test_the_trace_counts_agree_with_the_artifacts(run):
    assert run.step("Observation window").outputs["bars kept"] == len(run.bars)
    assert run.step("Forwards").outputs["forwards"] == len(run.forwards) == run.bars["expiry"].nunique()
    assert run.step("Dividend curve").outputs["label"] == "parity"
    assert run.step("Dividend curve").outputs["close-tier expiries"] == len(run.parity_forwards) == 2
    assert run.step("Option quotes").outputs["quotes"] == len(run.observations.options)
    assert run.observations.rates == RATE_QUOTES
    assert run.step("Discount curve").inputs["published"] == str(AS_OF - dt.timedelta(days=1))
    assert drop_counts(run)["in the money"] > 0
    assert len(run.result.diagnostics["loss_history"]) > run.result.diagnostics["probe_start"] > 0


def test_the_dividend_curve_is_implied_by_parity_and_sets_the_forwards(run):
    # The synthetic market carries CARRY at every expiry, so parity recovers it at each pillar.
    t = run.parity_forwards["T"].to_numpy()
    np.testing.assert_allclose(run.snapshot.dividend.zero_rate(t).detach().numpy(), CARRY, rtol=1e-9)
    spot = run.observations.spot.value
    for row in run.forwards.itertuples():
        assert row.forward == pytest.approx(float(run.snapshot.forward(row.T).detach()), rel=1e-12)
        assert row.discount == pytest.approx(float(run.snapshot.discount.discount(row.T).detach()), rel=1e-12)
        assert row.forward == pytest.approx(spot * row.ratio, rel=1e-12)


def test_a_term_structure_of_dividends_is_recovered_expiry_by_expiry():
    carry = {"2026-10-16": 0.002, "2026-12-18": 0.015}
    run = build_snapshot(_market(minutes=(48, 52, 56), carry=carry), RATE_QUOTES, listed_product("SPX"),
                         SnapshotSettings(AS_OF))
    spot = run.observations.spot.value
    t = run.forwards["T"].to_numpy()
    q = np.array(list(carry.values()))
    r = run.snapshot.discount.zero_rate(t).detach().numpy()
    np.testing.assert_allclose(run.forwards["forward"], spot * np.exp((r - q) * t), rtol=1e-9)


def test_the_run_hands_model_calibration_its_inputs(run):
    inputs = run.inputs
    assert inputs.market is run.snapshot and inputs.quotes is run.observations
    # One copy of each fitted object: on the snapshot.
    assert not any(hasattr(run, name) for name in ("curve", "carry", "surface", "spot", "quotes", "pairs"))


def test_one_day_count_for_the_bars_and_the_snapshot():
    run = build_snapshot(_market(), RATE_QUOTES, listed_product("SPX"), SnapshotSettings(AS_OF, day_count="ACT/360"))
    assert run.snapshot.day_count == "ACT/360"
    expiry = run.bars["expiry"].iloc[0]
    assert run.bars["T_close"].iloc[0] == pytest.approx(run.snapshot.time_to(expiry))


@pytest.mark.parametrize("change", [{"surface_model": "NOPE"}, {"curve_model": "NOPE"}])
def test_unknown_models_are_refused(change):
    settings = dataclasses.replace(SnapshotSettings(AS_OF), **change)
    with pytest.raises(ValidationError):
        build_snapshot(_market(), RATE_QUOTES, listed_product("SPX"), settings)


def test_american_options_are_refused_with_the_reason():
    with pytest.raises(ValidationError, match="early-exercise premium"):
        build_snapshot(_market(), RATE_QUOTES, listed_product("SPY"), SnapshotSettings(AS_OF))


def test_the_fitted_surface_stays_inside_its_bounds(run):
    bounds = run.snapshot.vol_surface.constraints()
    assert bounds["gamma <= 1/2"]["value"] <= 0.5
    assert bounds["eta (1 + |rho|) < 2"]["value"] < 2


def test_a_day_with_no_pairs_is_refused_rather_than_given_a_flat_carry():
    calls_only = _market(minutes=(48, 52, 56))
    calls_only = calls_only[calls_only.option_type == "call"]
    with pytest.raises(MarketDataError, match="no put-call parity forwards"):
        build_snapshot(calls_only, RATE_QUOTES, listed_product("SPX"), SnapshotSettings(AS_OF))
