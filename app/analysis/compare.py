"""Per-expiry cross-checks: what the traded call-put pairs imply against what the ivs assume.

The ivs are computed with the Treasury zero rate ``r(T)`` and the library's dividend curve
implied by put-call parity, so each bar's forward is ``F = S D_q(T) / D_r(T)``. Two checks
on it, per expiry:

* the expiry's own parity forward (``parity_forwards``, Treasury D) against the curve: where
  the curve's monotone fit moved it, and by how much;
* the parity regression, which estimates ``D`` and the forward from the trades alone: ``D``
  from the slope, ``F = A / D`` from the intercept. A diagnostic, never an input.

Where a forward is off, every iv at that expiry is too, which shows up as an iv gap between
a call and a put at the same strike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .implied_vols import curve_forward

_KEY = "expiry"


def expiry_table(parity_pairs: pd.DataFrame, parity_fit: pd.DataFrame, iv_pairs: pd.DataFrame,
                 points: pd.DataFrame, curve, dividend, parity_fwds: pd.DataFrame) -> pd.DataFrame:
    """One row per expiry in any of the inputs, with every number about it.

    ``parity_pairs`` / ``parity_fit``: the library's ``torch_pricer.market.parity``
    ``parity_pairs`` and ``parity_fit`` (the regression); ``parity_fwds``: its
    ``parity_forwards``. ``iv_pairs``: ``call_put_pairs``; ``points``:
    ``select_points(..., side="both")``.

    Columns: ``expiry, option_root, days, T, spot``; the regression's ``reg_discount, reg_r,
    reg_r_se, reg_q, reg_q_se, reg_forward``; the iv inputs' ``tsy_r`` (Treasury zero rate at
    ``T``), ``curve_q`` (the dividend curve's zero rate) and ``curve_forward``;
    ``forward_gap = reg_forward - curve_forward`` in index points; the expiry's own parity
    forward, ``tier`` (close, day, or ``curve`` with none: the curve interpolates),
    ``parity_forward, parity_forward_se`` and ``own_gap = parity_forward - curve`` at the
    same spot and ``T``; ``n_pairs, n_box`` (regression pairs), ``n_iv_pairs,
    median_iv_gap, median_F_error`` (iv pairs) and ``n_otm, n_itm`` (iv points).
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

    own = parity_fwds.set_index(_KEY)
    own_curve = own.spot * dividend.discount(own["T"].to_numpy(copy=True)).detach().numpy() / own.discount
    own = pd.DataFrame({"tier": own.tier, "parity_forward": own.forward, "parity_forward_se": own.forward_se,
                        "own_gap": own.forward - own_curve, "T_own": own["T"], "spot_own": own.spot})

    out = reg.join([ivs, pts], how="outer").join(own, how="outer")
    roots = pd.concat([parity_pairs[[_KEY, "option_root"]], points[[_KEY, "option_root"]],
                       parity_fwds[[_KEY, "option_root"]]])
    out["option_root"] = roots.drop_duplicates(_KEY).set_index(_KEY)["option_root"]
    if "T" not in out:
        out["T"], out["spot"] = np.nan, np.nan
    out["T"] = out["T"].fillna(out.T_points).fillna(out.T_own)
    out["spot"] = out["spot"].fillna(out.spot_points).fillna(out.spot_own)
    out = out.drop(columns=["T_points", "spot_points", "T_own", "spot_own"])
    out["days"] = out["T"] * 365
    out["tsy_r"], _, out["curve_forward"] = curve_forward(out.spot, out["T"], curve, dividend)
    out["curve_q"] = dividend.zero_rate(np.maximum(out["T"].to_numpy(dtype=float, copy=True), 0.0)).detach().numpy()
    out["tier"] = out["tier"].fillna("curve")
    if "reg_forward" in out:
        out["forward_gap"] = out.reg_forward - out.curve_forward
    for col in ["n_pairs", "n_box", "n_iv_pairs", "n_otm", "n_itm"]:
        out[col] = out[col].fillna(0).astype(int) if col in out else 0

    order = ["option_root", "days", "T", "spot", "tier", "parity_forward", "parity_forward_se", "curve_forward",
             "own_gap", "curve_q", "tsy_r", "reg_discount", "reg_r", "reg_r_se", "reg_q", "reg_q_se",
             "reg_forward", "forward_gap", "n_pairs", "n_box", "n_iv_pairs", "median_iv_gap", "median_F_error",
             "n_otm", "n_itm"]
    return out.reindex(columns=order).rename_axis(_KEY).sort_values("T").reset_index()


def expiry_parity_lines(pairs: pd.DataFrame, fit: pd.Series, curve, dividend):
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
    _, d_c, f_c = curve_forward(out.spot, out["T"], curve, dividend)
    out["curve_discount"], out["curve_forward"] = d_c, f_c
    out["resid_curve"] = out.parity - d_c * (f_c - out.strike)

    strike = np.array([out.strike.min(), out.strike.max()])
    lines = pd.DataFrame({
        "strike": strike,
        "regression": fit.intercept - D * strike,
        "curve": d_c.mean() * (f_c.mean() - strike),
    })
    return out, lines
