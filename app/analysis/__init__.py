"""Ad hoc analysis behind the app, ported from ``scripts/vol_surface_visual.ipynb``.

Model-free diagnostics on one day's option bars: Black implied vols off the Treasury curve
and the dividend curve put-call parity implies, the pairs and filters behind that curve,
and the per-expiry comparison between it and the parity regression. The window, the time to settlement and the parity regression are the
library's (``torch_pricer.market.market_data`` and ``torch_pricer.market.parity``); what is
here are research views, and the app pages only plot what these functions return.
``surface_models`` sets the library's surface fits side by side.
"""

from .compare import expiry_parity_lines, expiry_table
from .forwards import filter_funnel, implied_dividends, intraday_forward
from .implied_vols import (
    call_put_pairs,
    contract_implied_vols,
    contract_vol_summary,
    curve_forward,
    near_the_money,
    parity_by_expiry,
    select_points,
)
from .rate_curves import par_rates, rate_curve_table
from . import surface_models

__all__ = [
    "call_put_pairs",
    "contract_implied_vols",
    "contract_vol_summary",
    "curve_forward",
    "expiry_parity_lines",
    "expiry_table",
    "filter_funnel",
    "implied_dividends",
    "intraday_forward",
    "near_the_money",
    "par_rates",
    "parity_by_expiry",
    "rate_curve_table",
    "select_points",
    "surface_models",
]
