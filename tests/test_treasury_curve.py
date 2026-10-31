"""Par-yield bootstrap: exact on the cases with a closed form, and it reprices
every quoted par bond at par."""

import datetime as dt
import math

import numpy as np
import pytest
import torch

from torch_pricer.errors import MarketDataError, ValidationError
from torch_pricer.market.curve.calibration import par_yield_quotes, treasury_curve
from torch_pricer.market.market_data import QuoteSet, SpotQuote
from torch_pricer.market.curve.treasury import (
    bootstrap_discount_factors,
    par_to_zero,
    treasury_zero_curve,
)

TENORS = [1 / 12, 0.25, 1, 2, 5, 10, 30]
# The CMT curve published for 2026-09-21, in percent.
PAR_PCT = [3.96, 4.17, 4.45, 4.76, 4.83, 4.96, 5.29]


def test_flat_par_curve_gives_flat_semiannual_zeros():
    t, z = par_to_zero([0.5, 1, 2, 10, 30], [0.04] * 5)
    assert z == pytest.approx([2 * math.log(1.02)] * 5, abs=1e-14)


def test_bills_are_simple_interest():
    t, z = par_to_zero([1 / 12, 0.25], [0.04, 0.05])
    assert z == pytest.approx([12 * math.log1p(0.04 / 12), 4 * math.log1p(0.05 / 4)])


def test_six_month_bill_and_bond_conventions_agree():
    # At 0.5y a bill and a one-coupon bond are the same cash flow.
    _, dfs = bootstrap_discount_factors([0.5, 1], [0.04, 0.04])
    _, (z,) = par_to_zero([0.5], [0.04])
    assert dfs[0] == pytest.approx(math.exp(-0.5 * z))


def test_every_quoted_par_bond_reprices_at_par():
    par = np.array(PAR_PCT) / 100
    grid, dfs = bootstrap_discount_factors(TENORS, par)
    for tenor, c in zip(TENORS, par):
        if tenor <= 0.5:
            continue
        n = round(tenor / 0.5)
        price = c / 2 * dfs[:n].sum() + dfs[n - 1]
        assert price == pytest.approx(1.0, abs=1e-13), tenor


def test_upward_slope_puts_zeros_above_par_yields():
    # Compare on the same compounding so only the bootstrap effect remains.
    par = np.array(PAR_PCT) / 100
    t, z = par_to_zero(TENORS, par)
    coupon = t > 0.5
    semiannual_zero = 2 * np.expm1(z[coupon] / 2)
    assert np.all(semiannual_zero >= par[coupon] - 1e-12)
    assert semiannual_zero[-1] - par[-1] > 1e-3  # 30y: over 10bp


def test_known_zeros_for_2026_09_21():
    # Regression values; correctness is pinned by the repricing test above.
    t, z = par_to_zero(TENORS, np.array(PAR_PCT) / 100)
    assert z * 100 == pytest.approx(
        [3.95348, 4.14841, 4.40325, 4.71356, 4.78089, 4.92462, 5.43111], abs=1e-5
    )


def test_missing_quotes_are_dropped_and_input_is_sorted():
    t, z = par_to_zero([2, 1, 0.5, 30], [0.045, 0.044, float("nan"), 0.05])
    assert t.tolist() == [1, 2, 30]


def test_builds_a_rate_curve_through_the_zeros():
    curve = treasury_zero_curve(TENORS, PAR_PCT)
    _, z = par_to_zero(TENORS, np.array(PAR_PCT) / 100)
    assert curve.pillar_zeros.detach().numpy() == pytest.approx(z)
    assert float(curve.zero_rate(2.0)) == pytest.approx(z[3])
    # Inside 1M the curve holds the 1M rate flat.
    assert float(curve.zero_rate(1 / 365)) == pytest.approx(z[0])


@pytest.mark.parametrize(
    "tenors, yields",
    [
        ([1, 2], [0.04]),
        ([], []),
        ([1, 1], [0.04, 0.04]),
        ([0.75, 1.3], [0.04, 0.04]),
        ([-1, 1], [0.04, 0.04]),
    ],
)
def test_rejects_bad_input(tenors, yields):
    with pytest.raises(ValidationError):
        par_to_zero(tenors, yields)


# -- from a quote set -------------------------------------------------------------

def _quotes(rates=()):
    return QuoteSet(as_of=dt.date(2026, 9, 22), spot=SpotQuote("SPX", 6500.0), rates=rates)


def test_quotes_carry_decimal_rates_and_publication_date():
    quotes = par_yield_quotes(TENORS, PAR_PCT, as_of=dt.date(2026, 9, 21))
    assert [q.tenor for q in quotes] == ["1M", "3M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    assert quotes[0].rate == pytest.approx(0.0396)
    assert all(q.kind == "par" and q.as_of == dt.date(2026, 9, 21) for q in quotes)
    assert all(type(q.rate) is float for q in quotes)


def test_curve_from_quotes_matches_the_bootstrap_in_float64():
    curve = treasury_curve(_quotes(par_yield_quotes(TENORS, PAR_PCT)))
    reference = treasury_zero_curve(TENORS, PAR_PCT)
    assert curve.pillar_zeros.dtype == torch.float64
    assert reference.pillar_zeros.dtype == torch.float64  # no float32 default any more
    for t in (0.1, 1.0, 7.0, 30.0):
        assert float(curve.zero_rate(t).detach()) == pytest.approx(
            float(reference.zero_rate(t).detach()), rel=1e-14
        )


def test_no_par_yields_is_an_error_not_a_flat_curve():
    with pytest.raises(MarketDataError):
        treasury_curve(_quotes())
