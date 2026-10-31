"""Guards for bugs that were live in this engine, one test each.

Every test here failed before its fix. They are cheap and specific on purpose:
the shape bugs in particular were silent, and a price that is merely plausible
is exactly what they produced.
"""

import datetime as dt
import math

import pytest
import torch

from torch_pricer.errors import ValidationError
from torch_pricer.instruments.payoff import payoff_for
from torch_pricer.instruments.spec import (
    AsianOption,
    AverageKind,
    Right,
    Stock,
    Style,
    VanillaOption,
)
from torch_pricer.market.curve.curves import RateCurve
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.black import BlackScholesModel
from torch_pricer.pricer.engine import MCConfig, price
from torch_pricer.simulator.rng import NormalDraws

from .conftest import AS_OF, DIV, EXPIRY, RATE, SPOT, VOL


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -- shapes and wiring ---------------------------------------------------


def test_draws_are_indexed_by_step_not_path():
    """``draws[t_idx]`` selected a *path*; only n_steps paths were ever touched."""
    d = NormalDraws(n_paths=8, n_factors=1, seed=0, antithetic=False).draw(n_steps=3)
    assert d.shape == (8, 3, 1)
    assert d[:, 0].shape == (8, 1)  # what the simulator must slice


def test_antithetic_paths_mirror():
    d = NormalDraws(n_paths=8, n_factors=1, seed=0, antithetic=True).draw(n_steps=3)
    assert torch.allclose(d[:4], -d[4:])


def test_simulation_uses_every_path(market, call):
    """A wrong ``x0`` shape simulated n_steps paths instead of n_paths, which
    shows up as a standard error far larger than the sample size implies."""
    res = price(call, market, BlackScholesModel(VOL), MCConfig(n_paths=100_000, n_steps=50))
    assert res.stderr < 0.05


def test_initial_state_is_log_spot(market):
    """GBM integrates log-spot; seeding it with raw spot exponentiates to e^100."""
    state = BlackScholesModel(VOL).initial_state(market)
    assert state.shape == (1,)
    assert float(state) == pytest.approx(math.log(SPOT))


def test_config_defaults_when_omitted(market, call):
    """``price`` dereferenced ``config.device`` before defaulting it."""
    assert price(call, market, BlackScholesModel(VOL)).price > 0


def test_diffusion_is_matrix_valued(market):
    """The Euler step uses ``bmm``, which needs (n_paths, dim, n_factors)."""
    sde = BlackScholesModel(VOL).to_sde(market)
    xt = torch.zeros(7, 1, dtype=torch.get_default_dtype())
    assert sde.diffusion_coefficient(xt, torch.tensor(0.5)).shape == (7, 1, 1)
    assert sde.drift_coefficient(xt, torch.tensor(0.5)).shape == (7, 1)


# -- greeks --------------------------------------------------------------


def test_vol_is_a_parameter_so_vega_exists():
    """``vol`` was a Python float, which made vega structurally impossible."""
    model = BlackScholesModel(VOL)
    assert isinstance(model.vol, torch.nn.Parameter)
    assert model.vol.requires_grad


def test_bucketed_rho_sums_to_parallel_rho():
    """Rho is a vector, one entry per pillar, and pillars past maturity are dead."""
    market = MarketSnapshot(
        ticker="",
        as_of=AS_OF,
        spot=torch.tensor(SPOT),
        discount=RateCurve.from_zeros([0.25, 0.5, 1.0, 2.0], [RATE] * 4, "r"),
        dividend=RateCurve.from_zeros([0.25, 0.5, 1.0, 2.0], [DIV] * 4, "q"),
        vol_surface=MarketSnapshot.flat(AS_OF, SPOT).vol_surface,
    )
    call = VanillaOption(strike=100.0, maturity=EXPIRY, right=Right.CALL)
    res = price(call, market, BlackScholesModel(VOL),
                MCConfig(n_paths=200_000, n_steps=20, seed=7), greeks=("rho",))
    rho = res.greeks["rho"]
    assert rho.shape == (4,)

    t = market.time_to(EXPIRY)
    fwd = SPOT * math.exp((RATE - DIV) * t)
    d2 = (math.log(fwd / 100.0) - 0.5 * VOL**2 * t) / (VOL * math.sqrt(t))
    analytic = 100.0 * t * math.exp(-RATE * t) * _norm_cdf(d2)
    assert float(rho.sum()) == pytest.approx(analytic, rel=1e-2)
    assert float(rho[-1]) == pytest.approx(0.0, abs=1e-9)  # pillar past maturity


def test_unknown_greek_is_rejected(market, call):
    with pytest.raises(ValidationError, match="unknown greeks"):
        price(call, market, BlackScholesModel(VOL), greeks=("charm",))


def test_stderr_is_computed_over_antithetic_pair_means():
    """Paths i and i + n/2 are one draw, not two.

    These payoffs are perfectly anti-correlated within each pair, so every pair
    mean is identical and the true standard error is zero. Treating the six
    paths as independent reports a large error instead.
    """
    from torch_pricer.pricer.engine import _stderr

    pv = torch.tensor([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
    assert _stderr(pv, antithetic=True) == pytest.approx(0.0, abs=1e-12)
    assert _stderr(pv, antithetic=False) > 0.3


def test_antithetic_reduces_variance(market, call):
    anti = price(call, market, BlackScholesModel(VOL),
                 MCConfig(n_paths=100_000, n_steps=20, seed=3, antithetic=True))
    plain = price(call, market, BlackScholesModel(VOL),
                  MCConfig(n_paths=100_000, n_steps=20, seed=3, antithetic=False))
    assert anti.stderr < plain.stderr


# -- contracts -----------------------------------------------------------


@pytest.mark.parametrize("style", [Style.AMERICAN, Style.BERMUDAN])
def test_early_exercise_is_rejected_not_silently_europeanised(style):
    """The engine prices the terminal payoff; treating a Bermudan as European
    understates it with no warning at all."""
    kwargs = {"exercise_dates": (dt.date(2025, 7, 1), EXPIRY)} if style is Style.BERMUDAN else {}
    spec = VanillaOption(strike=100.0, maturity=EXPIRY, right=Right.CALL, style=style, **kwargs)
    with pytest.raises(ValidationError, match="not supported"):
        payoff_for(spec)


def test_instrument_without_expiry_is_rejected(market):
    with pytest.raises(ValidationError, match="no expiry"):
        price(Stock(underlying="X"), market, BlackScholesModel(VOL))


def test_snapshot_to_does_not_mutate_the_original(market):
    """``nn.Module.to`` is in place, so pricing silently recast the caller's curves."""
    before = market.discount.pillar_zeros.dtype
    moved = market.to(dtype=torch.float64)
    assert moved.discount.pillar_zeros.dtype == torch.float64
    assert market.discount.pillar_zeros.dtype == before
    assert moved.discount is not market.discount


def test_snapshot_repr_works(market):
    assert "MarketSnapshot" in repr(market)  # referenced a field that did not exist


def test_unimplemented_things_raise_rather_than_return():
    # implied_vol was here until it was implemented; see tests/test_implied_vol.py.
    from torch_pricer.simulator.brownian_bridge import BrownianBridge

    with pytest.raises(NotImplementedError):
        BrownianBridge(0.0, 0.2).drift_coefficient(torch.zeros(2, 1), torch.tensor(0.0))
    with pytest.raises(NotImplementedError):
        BlackScholesModel(VOL).calibrate(None)


# -- path-dependent ------------------------------------------------------


def test_geometric_asian_matches_closed_form(market):
    """Exercises the trajectory branch against an exact discrete-average formula.

    The payoff averages the n grid points after t=0, and the geometric average
    of lognormals is lognormal, so this has a closed form with no approximation
    to hide a shape bug behind.
    """
    n = 25
    spec = AsianOption(strike=100.0, maturity=EXPIRY, right=Right.CALL,
                       average=AverageKind.GEOMETRIC)
    res = price(spec, market, BlackScholesModel(VOL),
                MCConfig(n_paths=200_000, n_steps=n, seed=7))

    t = market.time_to(EXPIRY)
    mu = RATE - DIV
    m = math.log(SPOT) + (mu - 0.5 * VOL**2) * t * (n + 1) / (2 * n)
    v = VOL**2 * t * (n + 1) * (2 * n + 1) / (6 * n**2)
    sv = math.sqrt(v)
    d1 = (m - math.log(100.0) + v) / sv
    expected = math.exp(-RATE * t) * (
        math.exp(m + 0.5 * v) * _norm_cdf(d1) - 100.0 * _norm_cdf(d1 - sv)
    )
    assert abs(res.price - expected) < 3 * res.stderr
