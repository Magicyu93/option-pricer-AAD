"""Per-expiry cross-checks: what the traded call-put pairs imply against what the ivs assume.

The ivs are computed with the Treasury zero rate ``r(T)`` and a flat dividend yield ``q``,
so each bar's forward is ``F = S exp((r - q) T)``. The parity regression estimates the same
two numbers from the trades alone: ``D = exp(-r T)`` from the slope, and the forward
``F = A / D`` from the intercept ``A = S exp(-q T)``. Where the two disagree, every iv at
that expiry is computed off the wrong forward, which shows up as an iv gap between a call
and a put at the same strike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .implied_vols import curve_forward

_KEY = "expiry"


def expiry_table(parity_pairs: pd.DataFrame, parity_fit: pd.DataFrame, iv_pairs: pd.DataFrame,
                 points: pd.DataFrame, curve, dividend_yield: float) -> pd.DataFrame:
    """One row per expiry in any of the inputs, with every number about it.

    ``parity_pairs`` / ``parity_fit``: the library's ``torch_pricer.market.parity``
    ``parity_pairs`` and ``parity_fit``. ``iv_pairs``: ``call_put_pairs``; ``points``:
    ``select_points(..., side="both")``.

    Columns: ``expiry, option_root, days, T, spot``; the regression's ``reg_discount, reg_r, reg_r_se, reg_q,
    reg_q_se, reg_forward``; the iv inputs' ``tsy_r`` (Treasury zero rate at ``T``),
    ``flat_q`` and ``curve_forward``; ``forward_gap = reg_forward - curve_forward`` in index
    points; ``n_pairs, n_box`` (regression pairs), ``n_iv_pairs, median_iv_gap,
    median_F_error`` (iv pairs) and ``n_otm, n_itm`` (iv points).
    """
    reg = pd.DataFrame(index=parity_fit.index)
    if not parity_fit.empty:
        reg = parity_fit.rename(columns={"discount": "reg_discount", "r": "reg_r", "r_se": "reg_r_se",
                                         "q": "reg_q", "q_se": "reg_q_se"})
        reg["reg_forward"] = parity_fit.intercept / parity_fit.discount
        reg = reg[["T", "reg_discount", "reg_r", "reg_r_se", "reg_q", "reg_q_se", "reg_forward", "n_pairs"]]
        by_pairs = parity_pairs.groupby(_KEY).agg(spot=("spot", "mean"), n_box=("is_box", "sum"))
        reg = reg.join(by_pairs)

    ivs = iv_pairs.groupby(_KEY).agg(n_iv_pairs=("iv_gap", "size"), median_iv_gap=("iv_gap", "median"),
                                      median_F_error=("F_error", "median"))
    pts = points.groupby(_KEY).agg(T_points=("T", "mean"), spot_points=("spot", "mean"),
                                    n_otm=("otm", "sum"), n_itm=("otm", lambda s: int((~s).sum())))

    out = reg.join([ivs, pts], how="outer")
    roots = pd.concat([parity_pairs[[_KEY, "option_root"]], points[[_KEY, "option_root"]]])
    out["option_root"] = roots.drop_duplicates(_KEY).set_index(_KEY)["option_root"]
    if "T" not in out:
        out["T"], out["spot"] = np.nan, np.nan
    out["T"] = out["T"].fillna(out.T_points)
    out["spot"] = out["spot"].fillna(out.spot_points)
    out = out.drop(columns=["T_points", "spot_points"])
    out["days"] = out["T"] * 365
    out["tsy_r"], _, out["curve_forward"] = curve_forward(out.spot, out["T"], curve, dividend_yield)
    out["flat_q"] = dividend_yield
    if "reg_forward" in out:
        out["forward_gap"] = out.reg_forward - out.curve_forward
    for col in ["n_pairs", "n_box", "n_iv_pairs", "n_otm", "n_itm"]:
        out[col] = out[col].fillna(0).astype(int) if col in out else 0

    order = ["option_root", "days", "T", "spot", "reg_discount", "reg_r", "reg_r_se", "tsy_r", "reg_q", "reg_q_se", "flat_q",
             "reg_forward", "curve_forward", "forward_gap", "n_pairs", "n_box", "n_iv_pairs",
             "median_iv_gap", "median_F_error", "n_otm", "n_itm"]
    return out.reindex(columns=order).rename_axis(_KEY).sort_values("T").reset_index()


def expiry_parity_lines(pairs: pd.DataFrame, fit: pd.Series, curve, dividend_yield: float):
    """One expiry's raw ``C - P`` against the regression's line and the iv inputs' line.

    ``pairs`` is the library's ``parity_pairs`` for one expiry; ``fit`` its row of
    ``parity_fit``. Returns ``(pairs, lines)``:

    * ``pairs`` gains ``resid_regression = C - P - (A_t - D K)``, with ``A_t`` that minute's
      intercept, and ``resid_curve = C - P - D_c (F_c - K)`` with the curve's discount and
      forward at that minute's spot; per minute both, so spot moves between minutes cancel.
    * ``lines``: ``strike`` (the end points) and the two lines through them at the average
      minute: ``regression = intercept - D K`` and ``curve = D_c (F_c - K)``.
    """
    out = pairs.copy()
    D = fit.discount
    a_t = (out.parity + D * out.strike).groupby(out.timestamp).transform("mean")
    out["resid_regression"] = out.parity - (a_t - D * out.strike)
    _, d_c, f_c = curve_forward(out.spot, out["T"], curve, dividend_yield)
    out["curve_discount"], out["curve_forward"] = d_c, f_c
    out["resid_curve"] = out.parity - d_c * (f_c - out.strike)

    strike = np.array([out.strike.min(), out.strike.max()])
    lines = pd.DataFrame({
        "strike": strike,
        "regression": fit.intercept - D * strike,
        "curve": d_c.mean() * (f_c.mean() - strike),
    })
    return out, lines
