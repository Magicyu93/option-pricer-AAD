"""New curve and surface models plug in without touching the code that uses them.

Each test defines the smallest possible model and runs it through the shared
machinery -- the snapshot, the engine, the generic surface fit -- unchanged.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from torch_pricer.black_formula import black_price
from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.market.curve import Curve
from torch_pricer.market.market_data import OptionQuote
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.market.surface import FlatVolSurface, VolSurface
from torch_pricer.market.surface.calibration import fit_surface
from torch_pricer.models.black import BlackScholesModel
from torch_pricer.pricer.engine import MCConfig, price

AS_OF = dt.date(2026, 9, 8)


class ConstantCurve(Curve):
    """The smallest curve: one rate, everywhere."""

    def __init__(self, rate: float):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float64))

    def zero_rate(self, t):
        t = self._time(t)
        return self.rate.expand(t.shape) if t.dim() else self.rate

    @property
    def risk_factors(self):
        return self.rate


class SkewOnlySurface(VolSurface):
    """The smallest fittable surface: one vol level and a linear skew in k."""

    def __init__(self, level: float, as_of: dt.date):
        super().__init__()
        self.level = nn.Parameter(torch.tensor(level, dtype=torch.float64))
        self.skew = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self._ref = as_of

    @classmethod
    def initial(cls, slices, as_of):
        return cls(float(np.sqrt(np.median([q.atm_total_variance / q.expiry_time for q in slices]))), as_of)

    def total_variance(self, k, expiry):
        k, t = self._as_tensor(k), self._as_tensor(expiry)
        return (self.level + self.skew * k) ** 2 * t

    @property
    def reference_date(self):
        return self._ref


def test_a_new_curve_works_in_the_snapshot_and_the_engine():
    rate, carry, vol = 0.04, 0.01, 0.2
    market = MarketSnapshot(
        ticker="X", as_of=AS_OF, spot=torch.tensor(100.0, dtype=torch.float64),
        discount=ConstantCurve(rate), dividend=ConstantCurve(carry),
        vol_surface=FlatVolSurface(vol, AS_OF),
    )
    assert float(market.forward(1.0).detach()) == pytest.approx(100 * np.exp(rate - carry))

    spec = VanillaOption(
        underlying="X", strike=100.0, maturity=dt.date(2027, 9, 8), right=Right.CALL,
        style=Style.EUROPEAN,
    )
    result = price(spec, market, BlackScholesModel(vol),
                   MCConfig(n_paths=20_000, n_steps=10, seed=1), greeks=("rho",))
    t = market.time_to(spec.expiry)
    expected = black_price(100 * np.exp((rate - carry) * t), 100.0, t, vol, np.exp(-rate * t))
    assert result.price == pytest.approx(float(expected), abs=4 * result.stderr)
    assert np.ndim(result.greeks["rho"]) == 0  # rho against the curve's one risk factor


def test_the_generic_fit_fits_a_new_surface_model_unchanged():
    truth = SkewOnlySurface(0.2, AS_OF)
    with torch.no_grad():
        truth.skew.fill_(-0.15)
    quotes, rows = [], []
    for t in (0.1, 0.5, 1.0):
        expiry = AS_OF + dt.timedelta(days=round(t * 365))
        rows.append({"expiry": expiry, "T": t, "forward": 100.0, "ratio": 1.0,
                     "discount": 1.0, "n_pairs": 10, "source": "parity"})
        for k in np.linspace(-0.2, 0.15, 9):
            quotes.append(OptionQuote(expiry, 100 * np.exp(k), Right.CALL if k >= 0 else Right.PUT,
                                      implied_vol=float(truth.implied_vol(k, t).detach()),
                                      volume=5.0))

    fitted, result = fit_surface(SkewOnlySurface, tuple(quotes), pd.DataFrame(rows), AS_OF)
    assert isinstance(fitted, SkewOnlySurface)
    assert result.parameters["model"] == "SkewOnlySurface"
    assert result.parameters["skew"] == pytest.approx(-0.15, abs=1e-6)
    assert result.rmse < 1e-7
    assert "min_butterfly_density" in result.diagnostics


def test_the_flat_surface_is_fittable_too():
    rows = [{"expiry": AS_OF + dt.timedelta(days=90), "T": 90 / 365, "forward": 100.0,
             "ratio": 1.0, "discount": 1.0, "n_pairs": 10, "source": "parity"}]
    quotes = tuple(
        OptionQuote(rows[0]["expiry"], strike, Right.CALL, implied_vol=0.25, volume=1.0)
        for strike in (95.0, 100.0, 105.0)
    )
    fitted, result = fit_surface(FlatVolSurface, quotes, pd.DataFrame(rows), AS_OF)
    assert float(fitted.level.detach()) == pytest.approx(0.25, abs=1e-8)
