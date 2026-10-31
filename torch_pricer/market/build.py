"""From one day of market data to a calibrated snapshot, recording every step.

:func:`build_snapshot` chains the calibration steps -- observation window,
discount curve, dividend curve, forwards, implied vols, surface fit -- into a
:class:`~torch_pricer.market.snapshot.MarketSnapshot`, and returns it inside a
:class:`SnapshotRun` with what produced it:

* the observations, as one :class:`~torch_pricer.market.market_data.QuoteSet`
  (spot, option quotes, par yields) -- what the market published;
* the intermediate frames a reader needs to follow the computation (bars,
  forwards, per-bar vols) and the surface fit's result;
* a :class:`StepRecord` per step: which library function ran, with what
  settings, on how much data, what it dropped, how long it took.

The curves and the surface live on the snapshot only, so there is one copy of each.

Input is data, not loaders: the minute-bar frame from the data pipeline's
``load_market`` and the par-yield quotes, so this module never imports the pipeline.
The dividend curve is not an input: it is implied by put-call parity on the day's
trades, each pair's forward taken with the Treasury discount factor
(:func:`~torch_pricer.market.parity.forward_pairs`), one forward per expiry, and a
non-decreasing curve of implied dividends through them
(:func:`~torch_pricer.market.curve.calibration.implied_dividend_curve`). With the
discount curve it sets every forward, ``F = S D_q(T) / D_r(T)``. Parity's own ``r`` and
``q`` are never used: on trade prints they are far too noisy. What differs between
underlyings comes from the listed product (:mod:`torch_pricer.instruments.listed`); what
counts as a usable observation from :class:`~torch_pricer.market.market_data.QuoteConfig`
(the surface's quotes) and :class:`~torch_pricer.market.parity.ParityConfig` (the
pairs behind the forwards).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch

from torch_pricer.calibration import CalibrationInputs, CalibrationResult
from torch_pricer.conventions import DEFAULT_DAY_COUNT
from torch_pricer.errors import ValidationError
from torch_pricer.instruments.listed import OptionProduct
from torch_pricer.market.curve.calibration import CURVE_MODELS, curve_forwards, implied_dividend_curve
from torch_pricer.market.market_data import (
    QuoteConfig,
    QuoteSet,
    RateQuote,
    SpotQuote,
    reference_spot,
    window_bars,
)
from torch_pricer.market.parity import DROP_REASONS, ParityConfig, forward_pairs, parity_forwards
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.market.surface.calibration import SURFACE_MODELS, fit_surface, option_quotes


@dataclass(frozen=True)
class SnapshotSettings:
    """The calibration choices: everything that decides the result beyond the data."""

    as_of: dt.date
    quote_config: QuoteConfig = field(default_factory=QuoteConfig)
    parity_config: ParityConfig = field(default_factory=ParityConfig)
    curve_model: str = "Treasury bootstrap"
    surface_model: str = "SSVI"
    max_iter: int = 500
    day_count: str = DEFAULT_DAY_COUNT  # the snapshot's, and so every T in the run


@dataclass(frozen=True)
class StepRecord:
    """One step of a calibration: what ran, on what, producing what."""

    name: str
    function: str  # dotted path of the library function that did the work
    settings: dict[str, Any]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    dropped: dict[str, int]
    seconds: float
    artifact: str  # where the step's result is on the run, e.g. "snapshot.discount"


@dataclass
class SnapshotRun:
    """A calibrated snapshot, what it was built from, and the trace of how."""

    settings: SnapshotSettings
    product: OptionProduct
    observations: QuoteSet  # spot, option quotes (with implied vols), par yields
    bars: pd.DataFrame  # the observation window's minute bars
    parity_forwards: pd.DataFrame  # each expiry's own forward from put-call parity, behind the dividend curve
    forwards: pd.DataFrame  # one forward per expiry, from the discount and dividend curves
    quote_bars: pd.DataFrame  # per-bar forwards and implied vols behind the quotes
    result: CalibrationResult  # the surface fit
    snapshot: MarketSnapshot  # spot, discount, dividend (implied by parity), vol surface
    trace: list[StepRecord] = field(default_factory=list)

    @property
    def inputs(self) -> CalibrationInputs:
        """What a model's ``calibrate`` takes: the snapshot and the quotes it came from."""
        return CalibrationInputs(market=self.snapshot, quotes=self.observations)

    def step(self, name: str) -> StepRecord:
        return next(s for s in self.trace if s.name == name)


def _path(fn) -> str:
    return f"{fn.__module__}.{fn.__qualname__}"


def build_snapshot(
    frame: pd.DataFrame,
    rate_quotes: tuple[RateQuote, ...],
    product: OptionProduct,
    settings: SnapshotSettings,
) -> SnapshotRun:
    """Calibrate ``product``'s options on ``settings.as_of`` from one day of data.

    ``frame`` is the data pipeline's ``load_market`` output for that day and
    underlying; ``rate_quotes`` are par yields (see
    :func:`~torch_pricer.market.curve.calibration.par_yield_quotes`). The dividend
    curve is implied by put-call parity, from pairs chosen by ``settings.parity_config``
    over their own window. Raises :class:`ValidationError` for a product with non-European exercise (its
    implied vols would come from a European formula), or an unknown model name.
    """
    if not product.is_european:
        raise ValidationError(
            f"cannot calibrate {product.underlying}: {product.style.value} exercise, so a Black "
            "implied vol would leave out the early-exercise premium"
        )
    if settings.curve_model not in CURVE_MODELS:
        raise ValidationError(f"unknown curve model {settings.curve_model!r}")
    if settings.surface_model not in SURFACE_MODELS:
        raise ValidationError(f"unknown surface model {settings.surface_model!r}")

    config, as_of, ticker = settings.quote_config, settings.as_of, product.underlying
    trace: list[StepRecord] = []

    def record(name, fn, step_settings, inputs, outputs, dropped, started, artifact):
        trace.append(StepRecord(
            name=name, function=_path(fn), settings=step_settings, inputs=inputs,
            outputs=outputs, dropped=dict(dropped), seconds=time.perf_counter() - started,
            artifact=artifact,
        ))

    # 1. The observation window.
    started = time.perf_counter()
    bars, dropped = window_bars(frame, as_of, product, config, settings.day_count)
    record(
        "Observation window", window_bars,
        {"window": f"{config.window_start}-{config.window_end} ET", "min_days": config.min_days,
         "min_price": config.min_price, "AM-settled roots": sorted(product.am_settled_roots()),
         "day_count": settings.day_count},
        {"minute bars": len(frame)},
        {"bars kept": len(bars), "expiries": bars["expiry"].nunique(),
         "contracts": bars.groupby(["expiry", "strike", "right"]).ngroups},
        dropped, started, "bars",
    )

    started = time.perf_counter()
    spot = reference_spot(bars)
    record("Reference spot", reference_spot, {}, {"bars kept": len(bars)},
           {"spot": spot, "at": str(bars["timestamp"].max())}, {}, started, "observations.spot")

    # 2. The discount curve, from the published rates.
    started = time.perf_counter()
    curve_fn = CURVE_MODELS[settings.curve_model]
    observations = QuoteSet(as_of=as_of, spot=SpotQuote(ticker, spot, as_of), rates=rate_quotes)
    curve = curve_fn(observations)
    published = sorted({str(q.as_of) for q in rate_quotes if q.as_of is not None})
    record(
        "Discount curve", curve_fn, {"model": settings.curve_model},
        {"par yields": len(rate_quotes), "published": ", ".join(published) or "unknown"},
        {"pillars": int(curve.risk_factors.numel())}, {}, started, "snapshot.discount",
    )

    # 3. The dividend curve, implied by put-call parity on pairs from their own window.
    started = time.perf_counter()
    pc = settings.parity_config
    parity_window = QuoteConfig(window_start=pc.data_start, window_end=pc.window_end, min_days=0, min_price=0.0)
    parity_bars, _ = window_bars(frame, as_of, product, parity_window, settings.day_count)
    pairs = forward_pairs(parity_bars, curve, pc)
    parity_fw = parity_forwards(pairs, pc)
    dividend = implied_dividend_curve(parity_fw)
    tiers = parity_fw["tier"].value_counts()
    record(
        "Dividend curve", implied_dividend_curve, {**dataclasses.asdict(pc), "discount": curve.label},
        {"pairs": len(pairs)},
        {"pairs kept": int((pairs["dropped_by"] == "").sum()), "close-tier expiries": int(tiers.get("close", 0)),
         "day-tier expiries": int(tiers.get("day", 0)), "pillars": int(dividend.risk_factors.numel()),
         "label": dividend.label},
        {r: int(n) for r, n in pairs["dropped_by"].value_counts().items() if r in DROP_REASONS},
        started, "snapshot.dividend",
    )

    started = time.perf_counter()
    forwards = curve_forwards(bars, curve, dividend, spot)
    record(
        "Forwards", curve_forwards, {"discount": curve.label, "dividend": dividend.label},
        {"expiries": int(bars["expiry"].nunique())}, {"forwards": len(forwards)},
        {}, started, "forwards",
    )

    # 4. Implied vols, one quote per contract; the observations are now complete.
    started = time.perf_counter()
    quotes, quote_bars, quote_drops = option_quotes(bars, forwards, config)
    observations = QuoteSet(as_of=as_of, spot=observations.spot, options=quotes, rates=rate_quotes)
    record(
        "Option quotes", option_quotes, {"max_abs_k": config.max_abs_k},
        {"bars kept": len(bars)}, {"bars used": len(quote_bars), "quotes": len(quotes)},
        quote_drops, started, "observations.options",
    )

    # 5. The surface.
    started = time.perf_counter()
    surface_cls = SURFACE_MODELS[settings.surface_model]
    surface, result = fit_surface(surface_cls, quotes, forwards, as_of, max_iter=settings.max_iter)
    record(
        "Surface fit", fit_surface,
        {"model": settings.surface_model, "max_iter": settings.max_iter},
        {"quotes": len(quotes), "expiries": int(result.residuals["expiry"].nunique())},
        {"rmse (vol pts)": round(100 * result.rmse, 4), "converged": result.converged,
         "iterations": result.iterations},
        {}, started, "snapshot.vol_surface",
    )

    # 6. The snapshot.
    started = time.perf_counter()
    snapshot = MarketSnapshot(
        ticker=ticker, as_of=as_of, spot=torch.tensor(spot, dtype=torch.float64),
        discount=curve, dividend=dividend, vol_surface=surface, day_count=settings.day_count,
    )
    record("Market snapshot", MarketSnapshot, {"day_count": settings.day_count},
           {"discount": curve.label, "dividend": dividend.label, "vol_surface": surface_cls.__name__},
           {"spot": spot}, {}, started, "snapshot")

    return SnapshotRun(
        settings=settings, product=product, observations=observations, bars=bars,
        parity_forwards=parity_fw, forwards=forwards, quote_bars=quote_bars, result=result, snapshot=snapshot,
        trace=trace,
    )


def drop_counts(run: SnapshotRun) -> Counter:
    """Every observation dropped along the way, by reason."""
    total: Counter = Counter()
    for step in run.trace:
        total.update(step.dropped)
    return total
