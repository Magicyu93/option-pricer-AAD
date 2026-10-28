"""The sidebar's settings, and every computation behind Streamlit's cache.

Each function makes one call into ``data_pipeline``, ``torch_pricer`` or ``app/analysis``
and caches the result, so switching pages or changing a plot option does not reload or
recompute the day. The window, the time to settlement and the parity regression are the
library's, so the app sees the bars a snapshot is built from.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for path in (APP_DIR.parent, APP_DIR):  # `streamlit run app/streamlit_app.py`, no install needed
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import analysis  # noqa: E402
from analysis import surface_models  # noqa: E402
from data_pipeline import MassiveHistoricalDataLoader, RatesDataLoader, cached_days  # noqa: E402
from torch_pricer.instruments.listed import LISTED_PRODUCTS, listed_product  # noqa: E402
from torch_pricer.market import parity  # noqa: E402
from torch_pricer.market.curve import treasury_zero_curve  # noqa: E402
from torch_pricer.market.market_data import QuoteConfig, window_bars  # noqa: E402

SETTINGS_KEY, FILTERS_KEY, EXPIRY_KEY = "settings", "filters", "drill_expiry"
FIT_KEY = "fit_settings"


@dataclass(frozen=True)
class Settings:
    """What the sidebar chose: one day, one window, one rate curve and dividend yield."""

    ticker: str
    day: str
    roots: tuple[str, ...]
    window_start: time
    window_end: time
    rates_provider: str
    rates_lag_days: int
    dividend_yield: float
    moneyness_band: float


@dataclass(frozen=True)
class Filters:
    """The filters ``analysis.select_points`` applies to the iv points."""

    min_days: float
    max_days: float | None
    min_price: float
    max_abs_k: float

    def kwargs(self) -> dict:
        return {"min_days": self.min_days, "max_days": self.max_days,
                "min_price": self.min_price, "max_abs_k": self.max_abs_k}


def settings() -> Settings:
    return st.session_state[SETTINGS_KEY]


def filters() -> Filters:
    return st.session_state[FILTERS_KEY]


def caption(s: Settings) -> None:
    st.caption(
        f"{s.ticker} {', '.join(s.roots)} on {s.day} · bars {s.window_start:%H:%M} to before {s.window_end:%H:%M} ET · "
        f"Treasury ({s.rates_provider}, lag {s.rates_lag_days}d) · flat q = {s.dividend_yield:.2%}"
    )


# -- what is cached on disk ------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=60)
def tickers_with_options() -> list[str]:
    return [t for t in LISTED_PRODUCTS if cached_days("options", t)]


@st.cache_data(show_spinner=False, ttl=60)
def days_with_options(ticker: str) -> list[str]:
    return cached_days("options", ticker)


def product_roots(ticker: str) -> list[str]:
    return sorted(listed_product(ticker).root_names)


# -- inputs ------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading minute bars from the cache...", max_entries=4)
def market(ticker: str, day: str) -> pd.DataFrame:
    return MassiveHistoricalDataLoader(verbose=False).load_market([day], [ticker])


@st.cache_data(show_spinner="Loading Treasury yields...")
def par_yields(day: str, provider: str, lag_days: int):
    """The ``ParYieldSnapshot`` in force on ``day``."""
    return RatesDataLoader(provider).yields_as_of(day, lag_days=lag_days)


@st.cache_resource(show_spinner=False, max_entries=8)
def rate_curve(day: str, provider: str, lag_days: int):
    snap = par_yields(day, provider, lag_days)
    return treasury_zero_curve(snap.tenors, snap.yields_pct)


def curve(s: Settings):
    return rate_curve(s.day, s.rates_provider, s.rates_lag_days)


@st.cache_data(show_spinner=False)
def rate_curve_table(s: Settings, t_min: float) -> dict[str, pd.DataFrame]:
    return analysis.rate_curve_table(curve(s), t_min=t_min)


@st.cache_data(show_spinner=False, max_entries=8)
def window(s: Settings) -> pd.DataFrame:
    """The library's ``window_bars`` for the chosen roots, with no price or expiry floor:
    the plot filters apply those later, so they can change without recomputing."""
    config = QuoteConfig(window_start=f"{s.window_start:%H:%M}", window_end=f"{s.window_end:%H:%M}",
                         min_days=0, min_price=0.0)
    bars, _ = window_bars(market(s.ticker, s.day), date.fromisoformat(s.day), listed_product(s.ticker), config)
    return bars[bars.option_root.isin(s.roots)].reset_index(drop=True)


# -- put-call parity regression ----------------------------------------------------------

@st.cache_data(show_spinner="Matching call/put pairs...", max_entries=8)
def parity_pairs(s: Settings) -> pd.DataFrame:
    return parity.parity_pairs(window(s), s.moneyness_band)


@st.cache_data(show_spinner=False, max_entries=8)
def parity_fit(s: Settings) -> pd.DataFrame:
    return parity.parity_fit(parity_pairs(s))


# -- implied vols ------------------------------------------------------------------------

@st.cache_data(show_spinner="Inverting implied vols...", max_entries=8)
def implied_vols(s: Settings) -> pd.DataFrame:
    return analysis.contract_implied_vols(window(s), curve(s), s.dividend_yield)


@st.cache_data(show_spinner=False, max_entries=8)
def contract_vols(s: Settings, combine: bool) -> pd.DataFrame:
    return analysis.contract_vol_summary(implied_vols(s), combine=combine)


@st.cache_data(show_spinner=False, max_entries=16)
def points(s: Settings, f: Filters, side: str = "both", combine: bool = False) -> pd.DataFrame:
    return analysis.select_points(contract_vols(s, combine), side, **f.kwargs())


@st.cache_data(show_spinner=False, max_entries=16)
def iv_pairs(s: Settings, f: Filters) -> pd.DataFrame:
    return analysis.call_put_pairs(contract_vols(s, False), parity_fit(s)["discount"], **f.kwargs())


@st.cache_data(show_spinner=False, max_entries=16)
def iv_parity_by_expiry(s: Settings, f: Filters) -> pd.DataFrame:
    by_expiry = analysis.parity_by_expiry(iv_pairs(s, f))
    return by_expiry[by_expiry.n > 0].reset_index(drop=True)


def near_the_money(pts: pd.DataFrame, band: float) -> pd.Series:
    return analysis.near_the_money(pts, band)


# -- cross-checks ------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=16)
def expiry_table(s: Settings, f: Filters) -> pd.DataFrame:
    return analysis.expiry_table(parity_pairs(s), parity_fit(s), iv_pairs(s, f), points(s, f), curve(s),
                                 s.dividend_yield)


@st.cache_data(show_spinner=False, max_entries=16)
def expiry_parity_lines(s: Settings, expiry: date):
    pairs, fit = parity_pairs(s), parity_fit(s)
    return analysis.expiry_parity_lines(pairs[pairs.expiry == expiry], fit.loc[expiry], curve(s), s.dividend_yield)


# -- surface models ----------------------------------------------------------------------

@dataclass(frozen=True)
class FitSettings:
    """What the surface-models page chose: which quotes go into a fit.

    The forwards are always the sidebar's Treasury curve and flat q.
    """

    min_days: float = 7.0
    min_price: float = 0.05
    max_abs_k: float = 1.0
    max_iter: int = 500

    def quote_config(self, s: Settings) -> QuoteConfig:
        return QuoteConfig(window_start=f"{s.window_start:%H:%M}", window_end=f"{s.window_end:%H:%M}",
                           min_days=self.min_days, min_price=self.min_price, max_abs_k=self.max_abs_k)


@st.cache_data(show_spinner="Building quotes and forwards for the fits...", max_entries=8)
def fit_inputs(s: Settings, fs: FitSettings):
    return surface_models.fit_inputs(window(s), curve(s), fs.quote_config(s), s.dividend_yield)


@st.cache_resource(show_spinner=False, max_entries=24)
def model_fit(s: Settings, fs: FitSettings, model: str):
    """One model's fit. A resource, not data: the surface is a torch module, kept as is."""
    return surface_models.fit_model(model, fit_inputs(s, fs), date.fromisoformat(s.day), fs.max_iter)


def model_fits(s: Settings, fs: FitSettings, models: tuple[str, ...]):
    return [model_fit(s, fs, m) for m in models]


@st.cache_data(show_spinner=False, max_entries=16)
def model_residuals(s: Settings, fs: FitSettings, models: tuple[str, ...]) -> pd.DataFrame:
    return surface_models.residuals(model_fits(s, fs, models))


@st.cache_data(show_spinner=False, max_entries=16)
def smile_curves(s: Settings, fs: FitSettings, models: tuple[str, ...], expiries: tuple[date, ...]) -> pd.DataFrame:
    return surface_models.smile_curves(model_fits(s, fs, models), model_residuals(s, fs, models), expiries)


@st.cache_data(show_spinner=False, max_entries=16)
def surface_mesh(s: Settings, fs: FitSettings, model: str, models: tuple[str, ...]) -> dict:
    return surface_models.surface_mesh(model_fit(s, fs, model), model_residuals(s, fs, models))


@st.cache_data(show_spinner=False, max_entries=16)
def atm_term_structure(s: Settings, fs: FitSettings, models: tuple[str, ...]):
    return surface_models.atm_term_structure(model_fits(s, fs, models), model_residuals(s, fs, models))


@st.cache_data(show_spinner="Computing Dupire's local vol...", max_entries=16)
def local_vol_grid(s: Settings, fs: FitSettings, model: str, models: tuple[str, ...]) -> dict:
    return surface_models.local_vol_grid(model_fit(s, fs, model), model_residuals(s, fs, models))


@st.cache_data(show_spinner=False, max_entries=16)
def model_summary(s: Settings, fs: FitSettings, models: tuple[str, ...]) -> pd.DataFrame:
    return surface_models.model_summary(model_fits(s, fs, models))


@st.cache_data(show_spinner=False, max_entries=16)
def per_expiry_errors(s: Settings, fs: FitSettings, models: tuple[str, ...]) -> pd.DataFrame:
    return surface_models.per_expiry_errors(model_residuals(s, fs, models))


@st.cache_data(show_spinner=False, max_entries=24)
def parameter_table(s: Settings, fs: FitSettings, model: str):
    fit = model_fit(s, fs, model)
    return (*surface_models.parameter_table(fit), fit.surface.constraints())
