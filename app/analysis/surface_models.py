"""Surface models side by side: the library's fits on one day, and what to draw of them.

The fits are the library's own (``option_quotes`` and ``fit_surface``, as in
``build_snapshot``); what is here turns them into frames for the page: a summary per
model, errors per expiry and per quote, smiles and the surface on grids, the ATM term
structure and Dupire's local vol. The forwards are the Treasury curve and a flat q, as on
the other pages: the forwards backed out of put-call parity are too noisy to fit to.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from torch_pricer.calibration import CalibrationResult
from torch_pricer.market.curve import RateCurve
from torch_pricer.market.curve.calibration import curve_forwards
from torch_pricer.market.market_data import QuoteConfig, reference_spot
from torch_pricer.market.surface.base import VolSurface
from torch_pricer.market.surface.calibration import SURFACE_MODELS, fit_surface, option_quotes

#: The models the page compares, in the order it shows them.
MODELS = ["SSVI", "eSSVI", "SVI"]


@dataclass(frozen=True)
class FitInputs:
    """What every model is fitted to: one quote per contract, and a forward per expiry."""

    quotes: tuple  # OptionQuote per contract
    quote_bars: pd.DataFrame  # the per-bar ivs behind the quotes
    forwards: pd.DataFrame  # curve_forwards output
    spot: float
    dropped: dict[str, int]


@dataclass
class ModelFit:
    name: str
    surface: VolSurface
    result: CalibrationResult
    seconds: float


def flat_q_forwards(bars: pd.DataFrame, curve, spot: float, dividend_yield: float) -> pd.DataFrame:
    """The library's forwards from the curve and a flat dividend yield: ``F = S e^{-qT} / D(T)``.

    The flat curve is built with ``from_zeros``, in float64 like the discount curve:
    ``RateCurve.flat`` is float32 and would round the forward at the seventh digit.
    """
    return curve_forwards(bars, curve, RateCurve.from_zeros([1.0], [dividend_yield], "flat_q"), spot)


def fit_inputs(bars: pd.DataFrame, curve, config: QuoteConfig, dividend_yield: float) -> FitInputs:
    """The quotes and forwards a fit takes, from ``window_bars`` output.

    ``bars`` may hold short expiries and cheap prints: ``config.min_days`` and
    ``config.min_price`` are applied here, as ``window_bars`` would in a snapshot.
    """
    long_enough = bars["days"] >= config.min_days
    keep = long_enough & (bars["price"] >= config.min_price)
    dropped = {"under min days": int((~long_enough).sum()), "under min price": int((long_enough & ~keep).sum())}
    bars = bars[keep].reset_index(drop=True)
    spot = reference_spot(bars)
    forwards = flat_q_forwards(bars, curve, spot, dividend_yield)
    quotes, quote_bars, drops = option_quotes(bars, forwards, config)
    return FitInputs(quotes, quote_bars, forwards, spot, {**dropped, **drops})


def fit_model(name: str, inputs: FitInputs, as_of: dt.date, max_iter: int = 500) -> ModelFit:
    started = time.perf_counter()
    surface, result = fit_surface(SURFACE_MODELS[name], inputs.quotes, inputs.forwards, as_of, max_iter=max_iter)
    return ModelFit(name, surface, result, time.perf_counter() - started)


# -- fit quality -------------------------------------------------------------------------

def model_summary(fits: list[ModelFit]) -> pd.DataFrame:
    """One row per model: the fit's error, convergence, and the three arbitrage checks."""
    rows = []
    for fit in fits:
        r, d = fit.result, fit.result.diagnostics
        rows.append({
            "model": fit.name,
            "rmse": 100 * r.rmse,
            "max_abs_error": 100 * float(r.residuals["error"].abs().max()),
            "quotes": len(r.residuals),
            "expiries": int(r.residuals["expiry"].nunique()),
            "converged": r.converged,
            "iterations": r.iterations,
            "seconds": fit.seconds,
            "min_butterfly": d["min_butterfly_density"],
            "min_calendar": d["min_calendar_increment"],
            "min_local_variance": d["min_local_variance"],
            "arbitrage_free": min(d["min_butterfly_density"], d["min_calendar_increment"],
                                  d["min_local_variance"]) >= 0,
        })
    return pd.DataFrame(rows)


def residuals(fits: list[ModelFit]) -> pd.DataFrame:
    """Every quote's error under every model, with days, ``k / sqrt(T)`` and the side."""
    frames = []
    for fit in fits:
        r = fit.result.residuals
        frames.append(r.assign(
            model=fit.name, days=365.0 * r["T"], x=r["k"] / np.sqrt(r["T"]),
            option_type=r["right"].astype(str),  # "call" or "put"
        ))
    return pd.concat(frames, ignore_index=True)


def per_expiry_errors(resid: pd.DataFrame) -> pd.DataFrame:
    """Weighted RMSE and mean error (the bias) per model and expiry, in vol points."""
    def stats(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "T": g["T"].iloc[0], "days": g["days"].iloc[0], "quotes": len(g),
            "rmse": 100 * float(np.sqrt(np.average(g["error"] ** 2, weights=g["weight"]))),
            "bias": 100 * float(np.average(g["error"], weights=g["weight"])),
            "max_abs": 100 * float(g["error"].abs().max()),
        })
    return resid.groupby(["model", "expiry"], sort=False).apply(stats, include_groups=False).reset_index()


# -- the surface on grids ----------------------------------------------------------------

def smile_curves(fits: list[ModelFit], resid: pd.DataFrame, expiries, n: int = 121) -> pd.DataFrame:
    """Each model's smile at each of ``expiries``, over the quotes' k range plus a margin."""
    frames = []
    for expiry in expiries:
        quotes = resid[resid["expiry"] == expiry]
        T = float(quotes["T"].iloc[0])
        lo, hi = quotes["k"].min(), quotes["k"].max()
        pad = 0.15 * (hi - lo) + 0.02 * np.sqrt(T)
        k = np.linspace(lo - pad, hi + pad, n)
        for fit in fits:
            iv = fit.surface.implied_vol_grid(k, [T])[0]
            frames.append(pd.DataFrame({"model": fit.name, "expiry": expiry, "T": T, "k": k, "iv": iv}))
    return pd.concat(frames, ignore_index=True)


def time_grid(resid: pd.DataFrame, n: int = 60) -> np.ndarray:
    """Times from the first to the last expiry, log-spaced, with the expiries themselves."""
    t = np.sort(resid["T"].unique())
    return np.unique(np.concatenate([np.geomspace(t[0], t[-1], n), t]))


def surface_mesh(fit: ModelFit, resid: pd.DataFrame, n_x: int = 41, n_t: int = 60) -> dict[str, np.ndarray]:
    """The fitted iv on a mesh in ``(k / sqrt(T), T)``, so it covers the band the quotes fill.

    Returns ``k``, ``days`` and ``iv``, each of shape ``(n_times, n_x)``.
    """
    x = np.linspace(resid["x"].min(), resid["x"].max(), n_x)
    t = time_grid(resid, n_t)
    k = x[None, :] * np.sqrt(t)[:, None]
    iv = np.stack([fit.surface.implied_vol_grid(row, [ti])[0] for row, ti in zip(k, t)])
    return {"k": k, "days": np.broadcast_to(365.0 * t[:, None], k.shape), "iv": iv, "x": x, "t": t}


def atm_term_structure(fits: list[ModelFit], resid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each model's ATM vol on a fine time grid, and the quotes' ATM vol per expiry.

    The quotes' ATM vol is their iv interpolated to ``k = 0``, for expiries whose quotes
    straddle it; the model curves run between the expiries too, so they show how each
    model interpolates.
    """
    t = time_grid(resid, 120)
    lines = pd.concat([
        pd.DataFrame({"model": fit.name, "T": t, "days": 365.0 * t,
                      "iv": [float(fit.surface.implied_vol_grid([0.0], [ti])[0, 0]) for ti in t]})
        for fit in fits
    ], ignore_index=True)
    first = resid[resid["model"] == resid["model"].iloc[0]]
    straddles = first.groupby("expiry")["k"].transform(lambda k: (k.min() < 0) & (k.max() > 0))
    quotes = first[straddles.astype(bool)].sort_values("k").groupby("expiry").apply(
        lambda g: pd.Series({"T": g["T"].iloc[0], "days": g["days"].iloc[0],
                             "iv": float(np.interp(0.0, g["k"], g["iv"]))}),
        include_groups=False,
    ).reset_index()
    return lines, quotes


def local_vol_grid(fit: ModelFit, resid: pd.DataFrame, n_x: int = 61, n_t: int = 80) -> dict[str, np.ndarray]:
    """Dupire's local variance on a ``(k / sqrt(T), T)`` grid, off the expiries.

    The expiries are left out because an interpolation that is only ``C^0`` in ``T``
    has a one-sided ``dw/dT`` there. Returns ``x``, ``days``, the local ``variance``
    and the local ``vol`` (NaN where the variance is negative).
    """
    x = np.linspace(resid["x"].min(), resid["x"].max(), n_x)
    expiries = np.sort(resid["T"].unique())
    t = np.geomspace(expiries[0], expiries[-1], n_t)
    t = t[np.abs(t[:, None] - expiries[None, :]).min(axis=1) > 1e-6]
    k = torch.tensor(x[None, :] * np.sqrt(t)[:, None], dtype=torch.float64)
    T = torch.tensor(t[:, None], dtype=torch.float64)
    variance = fit.surface.local_variance(k, T).detach().numpy()
    with np.errstate(invalid="ignore"):
        vol = np.where(variance >= 0, np.sqrt(variance), np.nan)
    return {"x": x, "days": 365.0 * t, "variance": variance, "vol": vol}


def parameter_table(fit: ModelFit) -> tuple[pd.DataFrame, dict[str, float]]:
    """The fitted parameters: per expiry as a table, and the global ones as a dict."""
    params = fit.surface.describe()
    resid = fit.result.residuals
    expiries = resid.groupby("expiry")["T"].first().sort_values()
    per_expiry = {k: v for k, v in params.items() if isinstance(v, list) and len(v) == len(expiries)}
    table = pd.DataFrame({"expiry": expiries.index, "days": 365.0 * expiries.to_numpy(), **per_expiry})
    if "theta" in table:
        table["atm_vol"] = np.sqrt(table["theta"] / expiries.to_numpy())
    return table, {k: v for k, v in params.items() if k not in per_expiry}
