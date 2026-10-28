"""Black implied vol: inverts black_price, and says nan instead of guessing."""

import numpy as np
import pytest

from torch_pricer.black_formula import black_price, implied_vol


@pytest.mark.parametrize("right", [1, -1])
def test_round_trip_across_moneyness_maturity_and_vol(right):
    k = np.linspace(-0.6, 0.4, 41)[:, None, None]
    t = np.array([7 / 365, 0.1, 1.0, 5.0])[None, :, None]
    vol = np.array([0.08, 0.2, 0.6, 1.5])[None, None, :]
    forward, discount = 7600.0, 0.97
    strike = forward * np.exp(k * np.sqrt(t))
    price = black_price(forward, strike, t, vol, discount, right)

    iv = implied_vol(price, forward, strike, t, discount, right)
    # Where the price carries time value worth resolving, the vol comes back.
    resolvable = price / discount - np.maximum(right * (forward - strike), 0) > 1e-9 * forward
    assert np.isfinite(iv[resolvable]).all()
    # To the documented tolerance: the bracket stops once narrower than 1e-5.
    np.testing.assert_allclose(
        iv[resolvable], np.broadcast_to(vol, iv.shape)[resolvable], rtol=0, atol=1e-5
    )


def test_deep_in_the_money_resolves_the_time_value_not_just_the_premium():
    # Premium ~2920, time value ~0.001: a premium-relative stop was off by 0.13 vol.
    forward, strike, t, vol = 7600.0, 7600.0 * np.exp(-0.437), 0.081, 0.296
    price = black_price(forward, strike, t, vol)
    assert implied_vol(price, forward, strike, t) == pytest.approx(vol, abs=1e-5)


def test_put_and_call_at_the_same_strike_agree_through_parity():
    forward, strike, t, discount, vol = 100.0, 90.0, 0.5, 0.98, 0.3
    call = black_price(forward, strike, t, vol, discount, 1)
    put = call - discount * (forward - strike)
    assert implied_vol(put, forward, strike, t, discount, -1) == pytest.approx(vol, abs=1e-5)


@pytest.mark.parametrize(
    "price,label",
    [(9.0, "below intrinsic"), (10.0, "at intrinsic"), (100.0, "at the forward"), (-1.0, "negative")],
)
def test_prices_outside_the_no_arbitrage_range_are_nan(price, label):
    # Call struck at 90 on a forward of 100, undiscounted: intrinsic is 10.
    assert np.isnan(implied_vol(price, 100.0, 90.0, 1.0)), label


def test_price_beyond_the_vol_bounds_is_nan():
    price = black_price(100.0, 100.0, 1.0, 6.0)
    assert np.isnan(implied_vol(price, 100.0, 100.0, 1.0, vol_bounds=(1e-4, 5.0)))


def test_shapes_broadcast_and_scalars_stay_scalar_like():
    iv = implied_vol([5.0, 6.0], 100.0, [100.0, 100.0], 1.0)
    assert iv.shape == (2,)
    assert np.ndim(implied_vol(5.0, 100.0, 100.0, 1.0)) == 0
