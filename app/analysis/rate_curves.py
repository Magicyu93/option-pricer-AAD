"""A rate curve's par, zero and instantaneous forward rates, as tables to plot.

All three are views of one discount function D(T), so the curve alone defines them:

* zero     ``r(T) = -ln D(T) / T``
* forward  ``f(T) = -d ln D / dT`` (over a one-basis-point-of-a-year window)
* par      up to 6M a bill yield ``(1 / D(T) - 1) / T``; beyond, the semiannual coupon that
  prices a bond at par, ``2 (1 - D(T)) / sum_i D(t_i)`` over ``t_i = 0.5, 1.0, ..., T``.

The curve's par yields are not the published ones: it keeps zeros only at the quoted
tenors, so coupon dates in between are interpolated (up to ~8bp apart at 30Y).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COUPON_PERIOD = 0.5


def par_rates(curve, max_tenor: float) -> tuple[np.ndarray, np.ndarray]:
    """The curve's par yields: bills on a fine grid to 6M, then every half-year coupon tenor."""
    bills = np.geomspace(1 / 52, COUPON_PERIOD, 60)
    bill_par = (1.0 / curve.discount(bills).detach().numpy() - 1.0) / bills
    coupon_dates = COUPON_PERIOD * np.arange(1, int(round(max_tenor / COUPON_PERIOD)) + 1)
    dfs = curve.discount(coupon_dates).detach().numpy()
    bond_par = (1.0 - dfs) / (COUPON_PERIOD * np.cumsum(dfs))  # tenor t_n uses coupons t_1..t_n
    return np.concatenate([bills, coupon_dates[1:]]), np.concatenate([bill_par, bond_par[1:]])


def rate_curve_table(curve, max_tenor: float | None = None, t_min: float = 1 / 52,
                     n: int = 600) -> dict[str, pd.DataFrame]:
    """``curve``'s rates in decimals, as three tables keyed by name.

    ``"curve"``: ``t, zero, forward`` on a log grid from ``t_min`` (1W) to ``max_tenor`` (the last pillar
    when None), stopping one forward bump short of the end: past the last pillar the curve is
    flat by construction. ``"par"``: ``t, par`` from :func:`par_rates`. ``"pillars"``:
    ``t, zero`` at the curve's pillars.
    """
    pillars = curve.pillar_times.numpy()
    max_tenor = float(max_tenor or pillars.max())
    t = np.geomspace(t_min, max_tenor - 2e-4, n)
    par_t, par = par_rates(curve, max_tenor)
    return {
        "curve": pd.DataFrame({"t": t, "zero": curve.zero_rate(t).detach().numpy(),
                               "forward": curve.instantaneous_forward(t).detach().numpy()}),
        "par": pd.DataFrame({"t": par_t, "par": par}),
        "pillars": pd.DataFrame({"t": pillars, "zero": curve.pillar_zeros.detach().numpy()}),
    }
