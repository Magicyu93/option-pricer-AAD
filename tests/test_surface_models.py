"""eSSVI and raw SVI, and what every surface gets from the base class: the smooth
term structure, Dupire's local variance, and the dense arbitrage checks."""

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import torch

from tests.test_build import RATE_QUOTES
from tests.test_market_quotes import AS_OF as MARKET_AS_OF
from tests.test_market_quotes import VOL, _market
from torch_pricer.errors import ValidationError
from torch_pricer.instruments.listed import listed_product
from torch_pricer.instruments.spec import Right
from torch_pricer.market.build import SnapshotSettings, build_snapshot
from torch_pricer.market.market_data import OptionQuote
from torch_pricer.market.surface import ESSVISurface, FlatVolSurface, SSVISurface, SVISurface
from torch_pricer.market.surface.calibration import (
    SURFACE_MODELS,
    arbitrage_checks,
    butterfly_density,
    check_times,
    fit_surface,
)
from torch_pricer.tensors import hermite_slopes, interp_hermite

AS_OF = dt.date(2026, 9, 8)
TIMES = [0.05, 0.25, 0.5, 1.0]
FORWARDS = [7610.0, 7650.0, 7700.0, 7800.0]
THETA = [0.0012, 0.0055, 0.0115, 0.0250]
K = torch.linspace(-1.5, 1.5, 301, dtype=torch.float64)


def _essvi(psi=(0.035, 0.08, 0.11, 0.16), rho=(-0.6, -0.55, -0.5, -0.45)) -> ESSVISurface:
    return ESSVISurface(TIMES, THETA, psi, rho, AS_OF)


def _svi() -> SVISurface:
    # SSVI-shaped slices: admissible, and free of butterfly arbitrage.
    theta, rho = np.array(THETA), -0.5
    phi = 1 / np.sqrt(theta)
    return SVISurface(TIMES, theta / 2 * (1 - rho**2), theta * phi / 2, np.full(4, rho), -rho / phi,
                      np.sqrt(1 - rho**2) / phi, AS_OF)


def _quotes(truth, noise=0.0, seed=0):
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


def _randomise(surface, seed: int, scale: float = 3.0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in surface.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    return surface


# -- the smooth term structure ------------------------------------------------------

def test_hermite_interpolation_is_monotone_through_the_knots_and_honours_end_slopes():
    xp = torch.tensor([0.1, 0.3, 0.35, 1.0, 2.0], dtype=torch.float64)
    fp = torch.tensor([0.01, 0.012, 0.03, 0.031, 0.08], dtype=torch.float64)  # uneven steps
    slopes = hermite_slopes(xp, fp, torch.tensor(0.025), torch.tensor(0.04))
    assert float(slopes[0]) == pytest.approx(0.025) and float(slopes[-1]) == pytest.approx(0.04)
    torch.testing.assert_close(interp_hermite(xp, xp, fp, slopes), fp)
    x = torch.linspace(0.1, 2.0, 4001, dtype=torch.float64)
    assert bool((torch.diff(interp_hermite(x, xp, fp, slopes)) >= 0).all())


def test_an_end_slope_too_steep_for_a_monotone_cubic_is_capped():
    xp = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    fp = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
    slopes = hermite_slopes(xp, fp, torch.tensor(5.0), torch.tensor(0.1))
    assert float(slopes[0]) == pytest.approx(0.3)  # three times the secant


# -- local variance and the density, from the base class --------------------------

def test_a_flat_surface_has_its_vol_as_local_vol_and_a_unit_density():
    s = FlatVolSurface(0.2, AS_OF).to(torch.float64)
    k = torch.tensor([-0.3, 0.0, 0.2], dtype=torch.float64)
    torch.testing.assert_close(s.local_variance(k, 0.5), torch.full((3,), 0.04, dtype=torch.float64))
    torch.testing.assert_close(s.butterfly_density(k, 0.5), torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize("surface", [SSVISurface(TIMES, THETA, AS_OF, rho=-0.6, eta=1.1, gamma=0.3), _essvi()])
def test_local_variance_is_dupire_by_finite_differences(surface):
    k = torch.tensor([-0.2, -0.05, 0.0, 0.1], dtype=torch.float64)
    t, hk, ht = 0.37, 1e-4, 1e-6

    def w(kk, tt):
        return surface.total_variance(kk, tt).detach()

    w0 = w(k, t)
    dw = (w(k + hk, t) - w(k - hk, t)) / (2 * hk)
    d2w = (w(k + hk, t) - 2 * w0 + w(k - hk, t)) / hk**2
    dt_ = (w(k, t + ht) - w(k, t - ht)) / (2 * ht)
    g = (1 - k * dw / (2 * w0)) ** 2 - dw**2 / 4 * (1 / w0 + 0.25) + d2w / 2
    torch.testing.assert_close(surface.local_variance(k, t).detach(), dt_ / g, rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(butterfly_density(surface, t, k), g, rtol=1e-5, atol=1e-7)


def test_local_variance_carries_gradients_to_every_parameter():
    s = _essvi()
    s.local_variance(torch.tensor([-0.1, 0.05], dtype=torch.float64), 0.4).sum().backward()
    for name, p in s.named_parameters():
        assert p.grad is not None and bool(p.grad.abs().sum() > 0), name


def test_the_checks_look_between_expiries_and_beyond_them_but_not_on_them_for_dw_dt():
    every, off = check_times(TIMES)
    assert set(TIMES) <= set(every) and not set(TIMES) & set(off)
    assert off.min() < TIMES[0] and off.max() > TIMES[-1]
    assert ((off > TIMES[1]) & (off < TIMES[2])).sum() == 3


def test_ssvi_with_smooth_theta_passes_every_dense_check():
    checks = arbitrage_checks(SSVISurface(TIMES, THETA, AS_OF), TIMES)
    assert checks["min_butterfly_density"] > 0
    assert checks["min_calendar_increment"] >= 0
    assert checks["min_local_variance"] >= 0


# -- eSSVI ---------------------------------------------------------------------------

def test_essvi_round_trips_its_slices_and_has_its_own_rho_per_expiry():
    psi, rho = (0.035, 0.08, 0.11, 0.16), (-0.6, -0.55, -0.5, -0.45)
    s = _essvi(psi, rho)
    np.testing.assert_allclose(s.theta.detach(), THETA, rtol=1e-12)
    np.testing.assert_allclose(s.psi.detach(), psi, rtol=1e-9)
    np.testing.assert_allclose(s.rho.detach(), rho, rtol=1e-9)
    for t, theta in zip(TIMES, THETA):  # ATM total variance is theta at each expiry
        assert float(s.total_variance(0.0, t).detach()) == pytest.approx(theta, rel=1e-12)


def test_essvi_before_the_first_expiry_is_a_frozen_smile():
    s = _essvi()
    k = torch.tensor([-0.3, 0.0, 0.2], dtype=torch.float64)
    torch.testing.assert_close(s.implied_vol(k, 0.01), s.implied_vol(k, TIMES[0]))


def test_essvi_projects_slices_that_would_cross_onto_ones_that_do_not():
    # Wing slopes increasing but phi = psi / theta increasing too: these two
    # slices cross just right of the money, and the parametrisation refuses them.
    theta, a, b = (0.00622391, 0.00628198), (0.0436944, 0.04538158), (0.13952347, 0.14913606)
    psi = [0.5 * (x + y) for x, y in zip(a, b)]
    rho = [(x - y) / (x + y) for x, y in zip(a, b)]
    s = ESSVISurface([0.30, 0.31], theta, psi, rho, AS_OF)
    with torch.no_grad():
        increment = s.total_variance(K, 0.31) - s.total_variance(K, 0.30)
    assert float(increment.min()) >= -1e-15
    phi = (s.psi / s.theta).detach()
    assert float(phi[1]) <= float(phi[0])


@pytest.mark.parametrize("seed", range(20))
def test_any_raw_parameters_give_an_arbitrage_free_essvi_surface(seed):
    s = _randomise(_essvi(), seed)
    for name, bound in s.constraints().items():
        if "<" in name:
            assert bound["value"] <= bound["bound"] + 1e-12, name
        else:
            assert bound["value"] >= -1e-15, name
    checks = arbitrage_checks(s, TIMES)
    assert checks["min_butterfly_density"] >= -1e-10
    assert checks["min_calendar_increment"] >= -1e-12
    assert checks["min_local_variance"] >= -1e-10


def test_inadmissible_essvi_inputs_are_rejected():
    with pytest.raises(ValidationError):
        ESSVISurface(TIMES, THETA[::-1], [0.1] * 4, [-0.5] * 4, AS_OF)
    with pytest.raises(ValidationError):
        ESSVISurface(TIMES, THETA, [0.1] * 4, [-1.0] * 4, AS_OF)


def test_essvi_fit_recovers_the_surface_it_was_generated_from():
    truth = _essvi()
    quotes, forwards = _quotes(truth)
    fitted, result = fit_surface(ESSVISurface, quotes, forwards, AS_OF)
    assert isinstance(fitted, ESSVISurface)
    assert result.rmse < 1e-4
    np.testing.assert_allclose(result.parameters["rho"], truth.rho.detach(), atol=2e-2)
    np.testing.assert_allclose(result.parameters["theta"], THETA, rtol=1e-2)
    assert result.diagnostics["min_calendar_increment"] >= -1e-12
    assert result.diagnostics["min_local_variance"] >= 0


def test_essvi_fits_an_ssvi_market_at_least_as_well_as_ssvi():
    truth = SSVISurface(TIMES, THETA, AS_OF, rho=-0.65, eta=1.05, gamma=0.35)
    quotes, forwards = _quotes(truth, noise=0.003)
    _, ssvi = fit_surface(SSVISurface, quotes, forwards, AS_OF)
    _, essvi = fit_surface(ESSVISurface, quotes, forwards, AS_OF)
    assert essvi.rmse <= ssvi.rmse + 1e-4


# -- raw SVI -------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_any_raw_parameters_give_admissible_svi_slices(seed):
    s = _randomise(_svi(), seed)
    bounds = s.constraints()
    assert bounds["|rho| < 1"]["value"] < 1
    assert bounds["b (1 + |rho|) <= 2"]["value"] <= 2
    assert bounds["min w > 0"]["value"] > 0
    assert bool((s.total_variance(K[None, :], torch.tensor(TIMES)[:, None]) > 0).all())


def test_svi_joins_slices_linearly_in_w_and_extrapolates_by_the_rules():
    s = _svi()
    k = torch.tensor([-0.4, 0.0, 0.3], dtype=torch.float64)
    w1, w2 = s.total_variance(k, TIMES[1]), s.total_variance(k, TIMES[2])
    mid = 0.5 * (TIMES[1] + TIMES[2])
    torch.testing.assert_close(s.total_variance(k, mid), 0.5 * (w1 + w2))
    torch.testing.assert_close(s.implied_vol(k, 0.01), s.implied_vol(k, TIMES[0]))  # frozen smile
    # After the last: constant ATM vol, wings shifted, not steepened.
    t_far = 2 * TIMES[-1]
    assert float(s.implied_vol(0.0, t_far).detach()) == pytest.approx(float(s.implied_vol(0.0, TIMES[-1]).detach()))
    shift = s.total_variance(k, t_far) - s.total_variance(k, TIMES[-1])
    torch.testing.assert_close(shift, shift[1].expand(3))


def test_the_svi_penalty_is_zero_for_an_arbitrage_free_slice_and_bites_otherwise():
    s = _svi()
    assert float(s.penalty()) == 0.0
    bad = SVISurface([0.5], [0.001], [1.3], [-0.5], [0.0], [0.01], AS_OF)  # sharp, steep: g < 0
    assert float(bad.penalty()) > 0
    assert float(butterfly_density(bad, 0.5, K).min()) < 0


def test_svi_fit_reproduces_its_own_quotes_and_reports_the_checks():
    quotes, forwards = _quotes(_svi(), noise=0.0)
    fitted, result = fit_surface(SVISurface, quotes, forwards, AS_OF)
    assert isinstance(fitted, SVISurface)
    assert result.rmse < 1e-4
    assert {"min_butterfly_density", "min_calendar_increment", "min_local_variance",
            "penalty"} <= set(result.diagnostics)


# -- wiring ----------------------------------------------------------------------

def test_the_new_models_are_selectable_by_name():
    assert SURFACE_MODELS["eSSVI"] is ESSVISurface
    assert SURFACE_MODELS["SVI"] is SVISurface


def test_a_snapshot_builds_with_an_essvi_surface():
    settings = dataclasses.replace(SnapshotSettings(MARKET_AS_OF), surface_model="eSSVI")
    run = build_snapshot(_market(minutes=(48, 52, 56)), RATE_QUOTES, listed_product("SPX"), settings)
    assert isinstance(run.snapshot.vol_surface, ESSVISurface)
    spot = run.observations.spot.value
    t = float(run.forwards["T"].iloc[0])
    assert float(run.snapshot.vol(spot, t).detach()) == pytest.approx(VOL, abs=1e-3)
