"""Monte Carlo convergence rates for the European option example.

The price in :mod:`examples.european_options` carries two independent errors,
and they behave nothing like each other:

* **Sampling error**, from averaging finitely many paths. It falls as
  ``O(N^-1/2)`` -- the central limit theorem, and nothing in the engine changes
  the exponent. Antithetic sampling and common random numbers move the
  *constant*; only more paths move the rate.
* **Discretization error**, from stepping an SDE on a finite time grid. Here it
  is *absent*: ``GeometricBrownianMotion`` integrates log-spot, whose diffusion
  is state-independent, so Euler-Maruyama is exact and ``n_steps`` changes only
  which Brownian path a seed produces.

That second point is a property of this *setup*, not of the engine, and both
halves of it matter. A state-dependent diffusion -- local vol -- reintroduces a
genuine slope. So does a non-flat rate curve, more quietly: the drift
``mu(t) - sigma^2/2`` is integrated by a left-endpoint rule, exact only while
``mu`` is constant, which it is under ``MarketSnapshot.flat`` and is not on a
pillared curve. Rerun this sweep before trusting a coarse grid anywhere else.

Errors are measured against the closed-form Black values, as RMSE over
independent seeds, so what is plotted is the true error and not a proxy for it.

Run with ``python -m examples.convergence``. Writes ``convergence.png``.
"""

from __future__ import annotations

import datetime as dt
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from torch_pricer.black_formula import black_delta, black_gamma, black_price
from torch_pricer.instruments.spec import Right, Style, VanillaOption
from torch_pricer.market.snapshot import MarketSnapshot
from torch_pricer.models.black import BlackScholesModel
from torch_pricer.pricer.engine import MCConfig, price

AS_OF, EXPIRY = dt.date(2025, 1, 2), dt.date(2026, 1, 2)
SPOT, RATE, DIV, VOL, STRIKE = 100.0, 0.03, 0.01, 0.20, 100.0

#: Path counts to sweep. Powers of two so the log-log fit is evenly spaced.
N_GRID = [2**k for k in range(10, 21)]
#: Seeds per point. An RMSE estimated from m seeds is itself uncertain by
#: ~1/sqrt(2(m-1)) in relative terms -- 6.3% at 128 -- which sets how tightly
#: panel 2 can pin the reported standard error. The whole sweep runs in ~1 min.
PRICE_SEEDS = 128
GREEK_SEEDS = 32
#: Step counts for the discretization sweep, at a fixed path count.
STEP_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
STEP_PATHS = 131_072
#: Steps used everywhere else. Irrelevant to the answer here -- that is the point.
BASE_STEPS = 20

# Categorical slots 1-3 of the validated palette, plus its ink and surface.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"


def exact() -> dict[str, float]:
    """Closed-form Black price and spot greeks for the example's contract."""
    t = MarketSnapshot.flat(
        AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=DIV, flat_vol=VOL
    ).time_to(EXPIRY)
    fwd, disc = SPOT * math.exp((RATE - DIV) * t), math.exp(-RATE * t)
    # black_delta and black_gamma are taken in the forward; convert to spot.
    dfwd_dspot = math.exp((RATE - DIV) * t)
    return {
        "price": float(black_price(fwd, STRIKE, t, VOL, disc, 1.0)),
        "delta": float(black_delta(fwd, STRIKE, t, VOL, disc, 1.0)) * dfwd_dspot,
        "gamma": float(black_gamma(fwd, STRIKE, t, VOL, disc)) * dfwd_dspot**2,
    }


def _price(n_paths: int, seed: int, antithetic: bool, n_steps: int, greeks=()):
    market = MarketSnapshot.flat(
        AS_OF, spot=SPOT, flat_rate=RATE, flat_dividend=DIV, flat_vol=VOL
    )
    spec = VanillaOption(
        strike=STRIKE, maturity=EXPIRY, right=Right.CALL, style=Style.EUROPEAN
    )
    config = MCConfig(
        n_paths=n_paths, n_steps=n_steps, seed=seed, antithetic=antithetic
    )
    return price(spec, market, BlackScholesModel(VOL), config, greeks=greeks)


def _rate(x, y) -> float:
    """Slope of log y on log x -- the convergence exponent."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def sampling_study(ref: float) -> dict[str, dict[str, np.ndarray]]:
    """Price RMSE and the engine's own reported standard error, against N."""
    out = {}
    for label, antithetic in (("plain", False), ("antithetic", True)):
        rmse, reported = [], []
        for n in N_GRID:
            runs = [_price(n, s, antithetic, BASE_STEPS) for s in range(PRICE_SEEDS)]
            err = np.array([r.price - ref for r in runs])
            rmse.append(math.sqrt((err**2).mean()))
            reported.append(float(np.mean([r.stderr for r in runs])))
        out[label] = {"rmse": np.array(rmse), "reported": np.array(reported)}
    return out


def greek_study(ref: dict[str, float]) -> dict[str, np.ndarray]:
    """Relative RMSE of delta and gamma against N.

    Delta is pathwise -- one backward pass, unbiased. Gamma is a central
    difference of that delta under common random numbers. Both converge at
    ``N^-1/2``; the constants are what differ.
    """
    delta, gamma = [], []
    for n in N_GRID:
        runs = [
            _price(n, s, True, BASE_STEPS, greeks=("delta", "gamma"))
            for s in range(GREEK_SEEDS)
        ]
        d = np.array([r.greeks["delta"] - ref["delta"] for r in runs])
        g = np.array([r.greeks["gamma"] - ref["gamma"] for r in runs])
        delta.append(math.sqrt((d**2).mean()) / ref["delta"])
        gamma.append(math.sqrt((g**2).mean()) / ref["gamma"])
    return {"delta": np.array(delta), "gamma": np.array(gamma)}


def step_study(ref: float) -> np.ndarray:
    """Price RMSE against step count, at fixed N. Flat means no Euler bias."""
    rmse = []
    for steps in STEP_GRID:
        err = np.array(
            [_price(STEP_PATHS, s, True, steps).price - ref for s in range(GREEK_SEEDS)]
        )
        rmse.append(math.sqrt((err**2).mean()))
    return np.array(rmse)


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=9, fontweight="600")
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, colors=INK_2, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot(ref, sampling, greeks, steps, path="european_bs_convergence.png") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), facecolor=SURFACE)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
    n = np.array(N_GRID, dtype=float)

    # -- 1. price sampling error -------------------------------------------
    ax = axes[0, 0]
    guide = sampling["plain"]["rmse"][0] * (n / n[0]) ** -0.5
    ax.plot(n, guide, color=INK_2, lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.annotate(
        r"$N^{-1/2}$", (n[-3], guide[-3]), textcoords="offset points",
        xytext=(4, 9), fontsize=9, color=INK_2,
    )
    for label, color in (("plain", BLUE), ("antithetic", ORANGE)):
        y = sampling[label]["rmse"]
        ax.plot(n, y, color=color, lw=1.8, marker="o", ms=5.5, zorder=3,
                mec=SURFACE, mew=1.2,
                label=f"{label}  (slope {_rate(n, y):+.3f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    _style(ax, "Price error falls as $N^{-1/2}$",
           "paths $N$", "RMSE vs Black  (price units)")

    # -- 2. is the reported standard error honest? -------------------------
    ax = axes[0, 1]
    # An RMSE over m seeds carries ~1/sqrt(2(m-1)) relative noise of its own, so
    # the ratio cannot sit on 1.0 exactly. Draw the band it should scatter in;
    # without it the wobble reads as a defect rather than as sample size.
    noise = 1.0 / math.sqrt(2 * (PRICE_SEEDS - 1))
    ax.axhspan(1 - noise, 1 + noise, color=INK_2, alpha=0.09, lw=0, zorder=0)
    ax.axhline(1.0, color=INK_2, lw=1.1, ls=(0, (5, 3)), zorder=1)
    ratios = []
    for label, color in (("plain", BLUE), ("antithetic", ORANGE)):
        r = sampling[label]["reported"] / sampling[label]["rmse"]
        ratios.append(r)
        ax.plot(n, r, color=color, lw=1.8, marker="o",
                ms=5.5, mec=SURFACE, mew=1.2, zorder=3, label=label)
    ax.set_xscale("log")
    lo, hi = min(r.min() for r in ratios), max(r.max() for r in ratios)
    pad = 0.12 * (hi - lo)
    ax.set_ylim(min(lo - pad, 1 - 1.6 * noise), max(hi + pad, 1 + 1.6 * noise))
    ax.annotate(f"+/-{noise:.0%}: sampling noise on the RMSE itself",
                (n[-1], 1 - noise), textcoords="offset points", xytext=(-4, -12),
                fontsize=8, color=INK_2, ha="right", va="top")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    _style(ax, "Reported standard error tracks the true error",
           "paths $N$", "reported stderr / realised RMSE")

    # -- 3. greeks ----------------------------------------------------------
    ax = axes[1, 0]
    for key, color in (("delta", BLUE), ("gamma", ORANGE)):
        y = greeks[key]
        ax.plot(n, y, color=color, lw=1.8, marker="o", ms=5.5, mec=SURFACE,
                mew=1.2, zorder=3, label=f"{key}  (slope {_rate(n, y):+.3f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        lambda v, _: f"{v * 100:g}%" if v >= 1e-4 else f"{v * 100:.2g}%"
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    _style(ax, "Same rate for both greeks, ~40x the constant for gamma",
           "paths $N$", "relative RMSE")

    # -- 4. discretization --------------------------------------------------
    ax = axes[1, 1]
    st = np.array(STEP_GRID, dtype=float)
    ax.axhline(steps.mean(), color=INK_2, lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.plot(st, steps, color=AQUA, lw=1.8, marker="o", ms=5.5, mec=SURFACE,
            mew=1.2, zorder=3, label=f"price RMSE  (slope {_rate(st, steps):+.3f})")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, max(steps) * 1.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    _style(ax, f"No Euler bias: steps buy nothing  (N = {STEP_PATHS:,})",
           "time steps", "RMSE vs Black  (price units)")

    fig.suptitle(
        "Monte Carlo convergence  ·  1y ATM European call, Black-Scholes"
        f"  ·  S={SPOT:g} K={STRIKE:g} r={RATE:g} q={DIV:g} σ={VOL:g}",
        fontsize=11.5, color=INK, x=0.008, ha="left", y=0.991, fontweight="600",
    )
    fig.text(
        0.008, 0.958,
        f"RMSE over {PRICE_SEEDS} seeds (price) / {GREEK_SEEDS} seeds (greeks), "
        "against closed-form Black. Slopes fitted on log-log.",
        fontsize=9, color=INK_2, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=2.6, w_pad=3.0)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print(f"wrote {path}")


def main() -> None:
    ref = exact()
    print(f"Black reference: price={ref['price']:.6f} delta={ref['delta']:.6f} "
          f"gamma={ref['gamma']:.6f}")

    sampling = sampling_study(ref["price"])
    greeks = greek_study(ref)
    steps = step_study(ref["price"])

    n = np.array(N_GRID, dtype=float)
    print(f"\n{'N':>10} {'plain':>10} {'antithetic':>12} {'ratio':>7} "
          f"{'delta':>10} {'gamma':>10}")
    for i, npaths in enumerate(N_GRID):
        p, a = sampling["plain"]["rmse"][i], sampling["antithetic"]["rmse"][i]
        print(f"{npaths:>10,} {p:>10.5f} {a:>12.5f} {p / a:>7.2f} "
              f"{greeks['delta'][i]:>9.3%} {greeks['gamma'][i]:>9.3%}")

    print(f"\nfitted rates (expect -0.5):"
          f"  price {_rate(n, sampling['plain']['rmse']):+.3f} plain,"
          f" {_rate(n, sampling['antithetic']['rmse']):+.3f} antithetic,"
          f"  delta {_rate(n, greeks['delta']):+.3f},"
          f"  gamma {_rate(n, greeks['gamma']):+.3f}")
    print(f"steps 1 -> 128 at N={STEP_PATHS:,}: rate "
          f"{_rate(np.array(STEP_GRID, dtype=float), steps):+.3f} (expect 0)")

    plot(ref, sampling, greeks, steps)


if __name__ == "__main__":
    main()
