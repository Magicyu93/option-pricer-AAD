"""SSVI: the surface is what its formula says, arbitrage-free for any raw
parameters, priced against the snapshot's forward, and the fit recovers a
surface it was generated from."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import torch

from torch_pricer.errors import CalibrationError, ValidationError
from torch_pricer.instruments.spec import Right
from torch_pricer.market.curve import RateCurve
from torch_pricer.market.market_data import OptionQuote
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.market.surface import SSVISurface
from torch_pricer.market.surface.calibration import butterfly_density, fit_ssvi

AS_OF = dt.date(2026, 9, 8)
TIMES = [0.05, 0.25, 0.5, 1.0]
FORWARDS = [7610.0, 7650.0, 7700.0, 7800.0]
THETA = [0.0012, 0.0055, 0.0115, 0.0250]


def _surface(**kw) -> SSVISurface:
    return SSVISurface(TIMES, THETA, AS_OF, **kw)


def _snapshot(surface, spot=7600.0, rate=0.04, carry=0.012) -> MarketSnapshot:
    curve = lambda r, label: RateCurve.from_zeros([0.5, 1.0], [r, r], label)  # noqa: E731
    return MarketSnapshot(
        ticker="SPX", as_of=AS_OF, spot=torch.tensor(spot, dtype=torch.float64),
        discount=curve(rate, "r"), dividend=curve(carry, "q"), vol_surface=surface,
    )


# -- the surface ------------------------------------------------------------------

def test_atm_vol_at_each_expiry_is_sqrt_theta_over_t():
    s = _surface()
    for t, theta in zip(TIMES, THETA):
        assert float(s.implied_vol(0.0, t).detach()) == pytest.approx(np.sqrt(theta / t), rel=1e-12)


def test_implied_vol_is_total_variance_over_time():
    s = _surface(rho=-0.6, eta=1.1, gamma=0.3)
    k = torch.tensor([-0.25, -0.09, 0.0, 0.1], dtype=torch.float64)
    torch.testing.assert_close(s.implied_vol(k, 0.25), torch.sqrt(s.total_variance(k, 0.25) / 0.25))


def test_atm_variance_is_smooth_and_increasing_between_expiries_and_constant_vol_outside():
    s = _surface()
    for t, theta in zip(TIMES, THETA):
        assert float(s.atm_variance(t).detach()) == pytest.approx(theta, rel=1e-12)
    assert float(s.atm_variance(0.01).detach()) == pytest.approx(THETA[0] * 0.01 / TIMES[0])
    assert float(s.atm_variance(3.0).detach()) == pytest.approx(THETA[-1] * 3.0 / TIMES[-1])

    t = torch.linspace(0.001, 2.0, 2001, dtype=torch.float64)
    assert bool((torch.diff(s.atm_variance(t).detach()) > 0).all())
    # C^1: the slope just before each expiry matches the slope just after it,
    # including where the cubic meets the constant-vol extrapolation.
    h = 1e-7
    for knot in TIMES:
        left = (s.atm_variance(knot) - s.atm_variance(knot - h)) / h
        right = (s.atm_variance(knot + h) - s.atm_variance(knot)) / h
        assert float(left.detach()) == pytest.approx(float(right.detach()), rel=1e-4)


def test_initial_parameters_round_trip_through_the_constraints():
    s = _surface(rho=-0.6, eta=1.1, gamma=0.3)
    assert float(s.rho.detach()) == pytest.approx(-0.6)
    assert float(s.eta.detach()) == pytest.approx(1.1)
    assert float(s.gamma.detach()) == pytest.approx(0.3)
    np.testing.assert_allclose(s.theta.detach().numpy(), THETA, rtol=1e-12)
    assert s.describe()["rho"] == pytest.approx(-0.6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theta": [0.002, 0.001, 0.02, 0.03]},  # decreasing: calendar arbitrage
        {"rho": -0.5, "eta": 1.5},  # eta (1 + |rho|) = 2.25 > 2
        {"gamma": 0.7},
    ],
)
def test_arbitrageable_inputs_are_rejected(kwargs):
    theta = kwargs.pop("theta", THETA)
    with pytest.raises(ValidationError):
        SSVISurface(TIMES, theta, AS_OF, **kwargs)


@pytest.mark.parametrize("seed", range(20))
def test_any_raw_parameters_give_an_arbitrage_free_surface(seed):
    s = _surface()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in s.parameters():
            p.copy_(3 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    assert float((s.eta * (1 + s.rho.abs())).detach()) < 2
    assert 0 < float(s.gamma.detach()) <= 0.5
    assert bool((torch.diff(s.theta) > 0).all())

    k = torch.linspace(-1.5, 1.5, 301, dtype=torch.float64)
    for t in TIMES:  # butterfly: non-negative density
        assert float(butterfly_density(s, t, k).min()) >= -1e-10
    times = torch.linspace(0.01, 2.0, 60, dtype=torch.float64)
    with torch.no_grad():  # calendar: total variance non-decreasing at fixed k
        w = torch.stack([s.total_variance(k, t) for t in times])
    assert float(torch.diff(w, dim=0).min()) >= -1e-12


def test_implied_vol_carries_gradients_to_every_parameter():
    s = _surface()
    s.implied_vol(torch.tensor([-0.08, 0.01], dtype=torch.float64), 0.4).sum().backward()
    for name, p in s.named_parameters():
        assert p.grad is not None and bool(p.grad.abs().sum() > 0), name


# -- through the snapshot ---------------------------------------------------------

def test_snapshot_places_strikes_against_its_own_forward():
    market = _snapshot(_surface(rho=-0.6, eta=1.1, gamma=0.3))
    for t, theta in zip(TIMES, THETA):
        forward = market.forward(t)
        assert float(market.vol(forward, t).detach()) == pytest.approx(np.sqrt(theta / t), rel=1e-12)
    strike, t = 7000.0, 0.25
    k = torch.log(torch.tensor(strike, dtype=torch.float64) / market.forward(t))
    torch.testing.assert_close(market.vol(strike, t), market.vol_surface.implied_vol(k, t))


def test_snapshot_vol_reaches_spot_curves_and_surface():
    spot = torch.tensor(7600.0, dtype=torch.float64, requires_grad=True)
    market = _snapshot(_surface()).with_spot(spot)
    market.vol(7000.0, 0.3).backward()
    assert spot.grad is not None and float(spot.grad) != 0  # the smile moves with spot
    assert bool(market.discount.pillar_zeros.grad.abs().sum() > 0)
    assert bool(market.dividend.pillar_zeros.grad.abs().sum() > 0)
    assert all(p.grad is not None for p in market.vol_surface.parameters())


def test_snapshot_with_a_surface_survives_a_dtype_move():
    market = _snapshot(_surface())
    moved = market.to(dtype=torch.float64)
    assert moved.vol_surface is not market.vol_surface
    assert float(moved.vol(7500.0, 0.3).detach()) == pytest.approx(
        float(market.vol(7500.0, 0.3).detach())
    )


# -- the fit ----------------------------------------------------------------------

def _synthetic_quotes(truth: SSVISurface, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    quotes, rows = [], []
    for t, f in zip(TIMES, FORWARDS):
        expiry = AS_OF + dt.timedelta(days=round(t * 365))
        rows.append({"expiry": expiry, "T": t, "forward": f, "ratio": f / 7600.0,
                     "discount": 1.0, "n_pairs": 10, "source": "parity"})
        width = 0.35 * np.sqrt(t)
        for k in np.linspace(-width, 0.6 * width, 15):
            vol = float(truth.implied_vol(k, t).detach()) + noise * rng.standard_normal()
            quotes.append(OptionQuote(expiry, f * np.exp(k), Right.CALL if k >= 0 else Right.PUT,
                                      implied_vol=vol, volume=10.0))
    return tuple(quotes), pd.DataFrame(rows)


def test_fit_recovers_the_surface_it_was_generated_from():
    truth = _surface(rho=-0.65, eta=1.05, gamma=0.35)
    quotes, forwards = _synthetic_quotes(truth)
    fitted, result = fit_ssvi(quotes, forwards, AS_OF)

    assert isinstance(fitted, SSVISurface)
    assert result.converged
    assert result.rmse < 1e-5
    assert result.parameters["model"] == "SSVISurface"
    assert result.parameters["rho"] == pytest.approx(-0.65, abs=1e-3)
    assert result.parameters["eta"] == pytest.approx(1.05, abs=1e-2)
    assert result.parameters["gamma"] == pytest.approx(0.35, abs=1e-2)
    np.testing.assert_allclose(result.parameters["theta"], THETA, rtol=1e-3)
    assert result.diagnostics["min_butterfly_density"] > 0
    assert result.diagnostics["min_calendar_increment"] >= 0


def test_fit_with_noise_stays_close_and_reports_per_expiry_error():
    truth = _surface(rho=-0.65, eta=1.05, gamma=0.35)
    quotes, forwards = _synthetic_quotes(truth, noise=0.005)
    _, result = fit_ssvi(quotes, forwards, AS_OF)
    assert result.rmse < 0.008
    assert set(result.diagnostics["per_expiry"].columns) >= {"rmse", "max_abs", "quotes"}
    assert len(result.residuals) == len(quotes)


def test_no_usable_quotes_is_a_calibration_error():
    _, forwards = _synthetic_quotes(_surface())
    with pytest.raises(CalibrationError):
        fit_ssvi((), forwards, AS_OF)


def test_implied_vol_grid_is_one_row_per_expiry():
    s = _surface()
    k = np.linspace(-0.2, 0.1, 7)
    grid = s.implied_vol_grid(k, TIMES)
    assert grid.shape == (len(TIMES), len(k))
    for row, t in zip(grid, TIMES):
        np.testing.assert_allclose(row, s.implied_vol(torch.tensor(k), t).detach().numpy(), rtol=1e-14)
