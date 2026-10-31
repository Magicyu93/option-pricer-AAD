"""Surfaces calibrated to option prices: implied vols, and a model-agnostic fit.

Two steps:

1. :func:`option_quotes` -- an implied vol per minute bar, against *that bar's*
   forward, averaged by volume into one :class:`OptionQuote` per contract. Per
   bar, not a vol of averaged prices: the index moves while the window is open
   (twelve points on 2026-09-08), and an averaged premium set against one spot
   mis-states the vol most for the short expiries where the averaging is worst.
2. :func:`fit_surface` -- any :class:`~torch_pricer.market.surface.base.VolSurface`
   subclass with an ``initial`` constructor, fitted by weighted least squares in
   implied vol with L-BFGS on its raw parameters. Each model keeps its
   no-arbitrage constraints in its parametrisation, so the fitter needs to know
   nothing about the model, and every surface it visits is admissible. A new
   model is a new file in this package, fitted here unchanged.

Weights: ``sqrt(volume)`` -- with trades and no bid/ask, volume is the only
measure of how much a price is worth -- times the Black vega's shape
``exp(-d^2 / 2)``, which damps the wings, where a trade price moves the vol most
and means least.

The checks after a fit -- butterfly (Durrleman's ``g(k) >= 0``), calendar
(``w`` non-decreasing in ``T``) and the sign of Dupire's local variance -- need
only ``total_variance``, so they hold any model to the same standard. They run
on a dense grid in ``T``: at the expiries, between them and beyond both ends,
since that is where an interpolation can fail and a local vol will look.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import torch

from torch_pricer.black_formula import implied_vol
from torch_pricer.calibration import CalibrationResult
from torch_pricer.errors import CalibrationError
from torch_pricer.instruments.spec import Right
from torch_pricer.market.market_data import OptionQuote, QuoteConfig
from torch_pricer.market.surface.base import SliceQuotes, VolSurface
from torch_pricer.market.surface.essvi import ESSVISurface
from torch_pricer.market.surface.flat import FlatVolSurface
from torch_pricer.market.surface.ssvi import SSVISurface
from torch_pricer.market.surface.svi import SVISurface

#: Log-moneyness grid the no-arbitrage checks run on.
_CHECK_GRID = torch.linspace(-1.5, 1.5, 301, dtype=torch.float64)
#: Where the checks look between consecutive expiries, as fractions of the gap.
_BETWEEN = (0.25, 0.5, 0.75)
#: ... and beyond the ends, as multiples of the first and last expiry.
_BEFORE, _AFTER = (0.25, 0.5, 0.75), (1.5, 2.0)


# -- quotes ------------------------------------------------------------------------

def option_quotes(
    bars: pd.DataFrame,
    forwards: pd.DataFrame,
    config: QuoteConfig = QuoteConfig(),
) -> tuple[tuple[OptionQuote, ...], pd.DataFrame, Counter]:
    """One quote per contract, its vol averaged over the window's bars.

    ``bars`` is :func:`~torch_pricer.market.market_data.window_bars` output and
    ``forwards`` is :func:`~torch_pricer.market.curve.calibration.curve_forwards` output.
    Each bar is priced at its own ``T``: its forward is the bar's spot grown at
    the expiry's rate ``ln(ratio) / T_close``, and its discount the expiry's
    rate held to ``T``. Only out-of-the-money options inside the moneyness band
    are kept.

    Returns the quotes, the per-bar frame they came from (with ``forward``,
    ``k`` and ``iv``), and the bars dropped, by reason.
    """
    drops: Counter = Counter()

    def keep(mask, reason: str, frame: pd.DataFrame) -> pd.DataFrame:
        drops[reason] += int((~mask).sum())
        return frame[mask]

    fwd = forwards.set_index("expiry")
    bars = keep(bars["expiry"].isin(fwd.index), "no forward for expiry", bars)
    expiry = fwd.loc[bars["expiry"]]
    held = bars["T"].to_numpy() / expiry["T"].to_numpy()  # T / T_close
    bars = bars.assign(
        forward=bars["spot"].to_numpy() * expiry["ratio"].to_numpy() ** held,
        discount=expiry["discount"].to_numpy() ** held,
    )
    bars = bars.assign(k=np.log(bars["strike"] / bars["forward"]))
    otm = np.where(bars["right"] > 0, bars["k"] >= 0, bars["k"] < 0)
    bars = keep(otm, "in the money", bars)
    bars = keep(
        bars["k"].abs() <= config.max_abs_k * np.sqrt(bars["T"]), "outside moneyness band", bars
    )
    iv = implied_vol(
        bars["price"], bars["forward"], bars["strike"], bars["T"], bars["discount"], bars["right"]
    )
    bars = keep(np.isfinite(iv), "no implied vol", bars.assign(iv=iv))

    weighted = bars.assign(iv_w=bars["iv"] * bars["volume"], px_w=bars["price"] * bars["volume"])
    contracts = weighted.groupby(["expiry", "strike", "right"], as_index=False).agg(
        iv_w=("iv_w", "sum"), px_w=("px_w", "sum"), volume=("volume", "sum")
    )
    quotes = tuple(
        OptionQuote(
            expiry=row.expiry,
            strike=float(row.strike),
            right=Right.CALL if row.right > 0 else Right.PUT,
            last=float(row.px_w / row.volume),
            implied_vol=float(row.iv_w / row.volume),
            volume=float(row.volume),
        )
        for row in contracts.itertuples(index=False)
    )
    return quotes, bars.reset_index(drop=True), drops


# -- no-arbitrage checks -----------------------------------------------------------

def butterfly_density(surface: VolSurface, expiry, k: torch.Tensor) -> torch.Tensor:
    """Durrleman's ``g(k)``: the risk-neutral density up to a positive factor.

    Negative anywhere means a butterfly spread with negative cost. Detached:
    for a check; :meth:`VolSurface.butterfly_density` keeps the graph.
    """
    return surface.butterfly_density(k, expiry).detach()


def calendar_increment(surface: VolSurface, times, k: torch.Tensor) -> float:
    """The smallest step in ``w`` between consecutive ``times`` at fixed ``k``.

    Negative means a calendar spread with negative cost.
    """
    with torch.no_grad():
        w = torch.stack([surface.total_variance(k, float(t)) for t in times])
    return float(torch.diff(w, dim=0).min()) if len(times) > 1 else 0.0


def check_times(expiry_times) -> tuple[np.ndarray, np.ndarray]:
    """The times the checks run at: ``(all, off the expiries)``, both sorted.

    The second leaves out the expiries themselves, where an interpolation
    that is only ``C^0`` in ``T`` has a one-sided ``dw/dT``.
    """
    times = np.asarray(expiry_times, dtype=float)
    between = [a + f * (b - a) for a, b in zip(times[:-1], times[1:]) for f in _BETWEEN]
    outside = [m * times[0] for m in _BEFORE] + [m * times[-1] for m in _AFTER]
    off = np.unique(np.concatenate([between, outside]))
    return np.unique(np.concatenate([times, off])), off


def arbitrage_checks(surface: VolSurface, expiry_times, k: torch.Tensor = _CHECK_GRID) -> dict:
    """The smallest butterfly density, calendar step and local variance on a dense grid.

    All three are non-negative for a surface free of static arbitrage. The
    local variance is checked off the expiries (see :func:`check_times`).
    """
    every, off = check_times(expiry_times)
    t_all = torch.tensor(every, dtype=torch.float64)[:, None]
    t_off = torch.tensor(off, dtype=torch.float64)[:, None]
    return {
        "min_butterfly_density": float(surface.butterfly_density(k[None, :], t_all).detach().min()),
        "min_calendar_increment": calendar_increment(surface, every, k),
        "min_local_variance": float(surface.local_variance(k[None, :], t_off).detach().min()),
    }


# -- the fit -----------------------------------------------------------------------

def _quote_frame(quotes, forwards: pd.DataFrame) -> pd.DataFrame:
    fwd = forwards.set_index("expiry")
    frame = pd.DataFrame(
        {
            "expiry": [q.expiry for q in quotes],
            "strike": [q.strike for q in quotes],
            "right": [q.right.value for q in quotes],
            "iv": [q.implied_vol for q in quotes],
            "volume": [q.volume or 1.0 for q in quotes],
        }
    )
    frame = frame[frame["iv"].notna() & frame["expiry"].isin(fwd.index)]
    frame["T"] = fwd.loc[frame["expiry"], "T"].to_numpy()
    frame["forward"] = fwd.loc[frame["expiry"], "forward"].to_numpy()
    frame["k"] = np.log(frame["strike"] / frame["forward"])
    return frame.sort_values(["T", "k"]).reset_index(drop=True)


def _slices(frame: pd.DataFrame) -> tuple[SliceQuotes, ...]:
    """Each expiry's quotes, with an ATM total variance from those nearest ``k = 0``.

    The ATM variances are made increasing, as every model's term structure must be.
    """
    groups = sorted(frame.groupby("expiry"), key=lambda item: item[1]["T"].iloc[0])
    t = np.array([g["T"].iloc[0] for _, g in groups])
    theta = np.array([float(np.interp(0.0, g["k"], g["iv"])) ** 2 for _, g in groups]) * t
    # Strictly increasing: nudge ties up by a hair of variance.
    theta = np.maximum.accumulate(theta) + 1e-8 * np.arange(len(theta))
    return tuple(
        SliceQuotes(
            expiry_time=float(ti), k=g["k"].to_numpy(), iv=g["iv"].to_numpy(),
            weight=g["weight"].to_numpy(), atm_total_variance=float(th),
        )
        for (_, g), ti, th in zip(groups, t, theta)
    )


def fit_surface(
    surface_cls: type[VolSurface],
    quotes: tuple[OptionQuote, ...],
    forwards: pd.DataFrame,
    as_of: dt.date,
    *,
    max_iter: int = 500,
) -> tuple[VolSurface, CalibrationResult]:
    """Fit a ``surface_cls`` surface to ``quotes`` with implied vols.

    ``forwards`` supplies each expiry's ``T`` and the forward that sets ``k``.
    The surface starts from ``surface_cls.initial`` at each expiry's quotes;
    expiries with quotes are its pillars. The loss adds the surface's own
    :meth:`~VolSurface.penalty`, zero for models whose constraints are all in
    their parametrisation.
    """
    frame = _quote_frame(quotes, forwards)
    if frame.empty:
        raise CalibrationError("no quotes with an implied vol and a forward")

    T = torch.tensor(frame["T"].to_numpy(), dtype=torch.float64)
    k = torch.tensor(frame["k"].to_numpy(), dtype=torch.float64)
    iv = torch.tensor(frame["iv"].to_numpy(), dtype=torch.float64)
    d = k / (iv * T.sqrt()) + 0.5 * iv * T.sqrt()  # Black d1 at the market vol
    weights = torch.tensor(np.sqrt(frame["volume"].to_numpy()), dtype=torch.float64)
    weights = weights * torch.exp(-0.5 * d**2)
    weights = weights / weights.sum()
    frame["weight"] = weights.numpy()

    slices = _slices(frame)
    surface = surface_cls.initial(slices, as_of)

    history: list[float] = []  # every loss evaluation, line searches included

    def loss_fn() -> torch.Tensor:
        value = (weights * (surface.implied_vol(k, T) - iv) ** 2).sum() + surface.penalty()
        history.append(float(value.detach()))
        return value

    def run(max_steps: int, tolerance_grad: float, tolerance_change: float) -> float:
        optimizer = torch.optim.LBFGS(
            surface.parameters(), lr=1.0, max_iter=max_steps, line_search_fn="strong_wolfe",
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=50,
        )

        def closure():
            optimizer.zero_grad()
            value = loss_fn()
            value.backward()
            return value

        optimizer.step(closure)
        run.iterations = int(optimizer.state[optimizer._params[0]].get("n_iter", 0))
        return float(closure().detach())

    # The loss is ~1e-4 (vol errors of a point or so, squared): 1e-12 is a
    # relative change of 1e-8, well past anything a trade price can resolve.
    loss = run(max_iter, 1e-10, 1e-12)
    iterations = run.iterations
    probe_start = len(history)

    # Converged means more iterations would not help -- not that L-BFGS met its
    # own tolerances. When a constraint binds (SSVI's gamma pressed against 1/2
    # is typical), the raw parameter drifts towards infinity with a vanishing
    # gradient and L-BFGS creeps on without improving anything.
    probed = run(50, 0.0, 0.0)
    if not (np.isfinite(loss) and np.isfinite(probed)):
        raise CalibrationError(f"{surface_cls.__name__} fit diverged")
    # 1e-4 of the loss is ~5e-5 of the RMSE: hundredths of a basis point of vol.
    # The absolute floor is vol errors of 1e-6 squared, for a fit that is exact.
    converged = loss - probed <= 1e-4 * loss + 1e-12
    loss = min(loss, probed)

    with torch.no_grad():
        fitted = surface.implied_vol(k, T)
    frame["model_iv"] = fitted.numpy()
    frame["error"] = frame["model_iv"] - frame["iv"]
    per_expiry = frame.groupby("expiry").apply(
        lambda g: pd.Series({
            "T": g["T"].iloc[0],
            "quotes": len(g),
            "rmse": float(np.sqrt(np.average(g["error"] ** 2, weights=g["weight"]))),
            "max_abs": float(g["error"].abs().max()),
        }),
        include_groups=False,
    )

    result = CalibrationResult(
        parameters={"model": surface_cls.__name__, **surface.describe()},
        loss=loss,
        rmse=float(np.sqrt(np.average(frame["error"] ** 2, weights=frame["weight"]))),
        converged=bool(converged),
        iterations=iterations,
        residuals=frame,
        diagnostics={
            "per_expiry": per_expiry,
            **arbitrage_checks(surface, [q.expiry_time for q in slices]),
            "penalty": float(surface.penalty().detach()),
            # The optimizer's path: one entry per loss evaluation. Entries from
            # ``probe_start`` on are the 50-iteration convergence probe.
            "loss_history": history,
            "probe_start": probe_start,
        },
    )
    return surface, result


#: The SSVI fit, the default surface model.
fit_ssvi = partial(fit_surface, SSVISurface)

#: Surface models a caller can choose by name: each is fittable by
#: :func:`fit_surface`. A new model is one more entry once it implements ``initial``.
SURFACE_MODELS: dict[str, type[VolSurface]] = {
    "SSVI": SSVISurface,
    "eSSVI": ESSVISurface,
    "SVI": SVISurface,
    "Flat": FlatVolSurface,
}
