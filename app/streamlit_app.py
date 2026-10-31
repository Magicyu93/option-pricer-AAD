"""Vol surface research app: one day of cached option bars, from the notebook's analysis.

Run from the repo root: ``streamlit run app/streamlit_app.py``
(``pip install -e .[app]`` for the dependencies).

Two pages put the rates, the forwards and the call vs put vols side by side: a day
overview across expiries, and a drill-down into one expiry. A third sets the library's
surface models against each other and against the quotes. The sidebar's choices apply
to all three. Computation lives in ``app/analysis`` and the library; pages only plot.
"""

from __future__ import annotations

from datetime import date, time

import streamlit as st

import cache
from torch_pricer.market.parity import ESTIMATORS, ParityConfig

st.set_page_config(page_title="Vol surface research", layout="wide")

PAGES = [
    st.Page("pages/day_overview.py", title="Day overview", default=True),
    st.Page("pages/expiry_drilldown.py", title="Expiry drill-down"),
    st.Page("pages/surface_models.py", title="Surface models"),
]


def sidebar() -> None:
    sb = st.sidebar
    tickers = cache.tickers_with_options()
    if not tickers:
        sb.error("No option days are cached. Download some with `scripts/download_market_data.py`.")
        st.stop()

    sb.header("Data")
    ticker = sb.selectbox("Underlying", tickers, key="ticker")
    days = [date.fromisoformat(d) for d in cache.days_with_options(ticker)]
    picked = sb.date_input("Day", days[-1], min_value=days[0], max_value=days[-1], key=f"day_{ticker}",
                           help=f"{len(days)} days of {ticker} options are cached, {days[0]} to {days[-1]}.")
    if picked not in days:
        before, after = [d for d in days if d < picked][-1:], [d for d in days if d > picked][:1]
        sb.error(f"No {ticker} options cached for {picked}. Nearest cached: "
                 f"{', '.join(str(d) for d in before + after)}.")
        st.stop()
    day = picked.isoformat()
    roots = cache.product_roots(ticker)
    chosen = sb.multiselect("Option roots", roots, default=[r for r in roots if r.endswith("W")] or roots,
                            key=f"roots_{ticker}")
    c1, c2 = sb.columns(2)
    start = c1.time_input("Window from (ET)", time(15, 30), step=300, key="window_start")
    end = c2.time_input("to (exclusive)", time(16, 0), step=300, key="window_end",
                        help="The 16:00 bar holds trades after the closing print, so it is left out.")

    sb.header("Rates")
    provider = sb.selectbox("Treasury yields from", ["massive", "fred"], key="rates_provider")
    lag = sb.number_input("Yield lag (days)", 0, 10, 0, help="Use the yields published on or before "
                          "day - lag; 1 avoids using the day's own end-of-day print.", key="rates_lag")

    parity_config = parity_sidebar(sb, end)
    band = sb.number_input("Parity regression (diagnostic): max |K/S - 1|", 0.01, 0.5, 0.10, 0.01,
                           key="moneyness_band", help="Only for the regression's r and q, shown for comparison; "
                           "never used for the forwards.")

    sb.header("Plot filters")
    min_days = sb.number_input("Min days to expiry", 0.0, 100.0, 1.0, 0.5, key="min_days")
    cap_days = sb.checkbox("Cap days to expiry", key="cap_days")
    max_days = sb.number_input("Max days to expiry", 1.0, 2000.0, 365.0, 1.0, key="max_days",
                               disabled=not cap_days)
    min_price = sb.number_input("Min option price", 0.0, 10.0, 0.05, 0.05, key="min_price")
    max_abs_k = sb.number_input("Max |k| / sqrt(T)", 0.05, 5.0, 1.0, 0.05, key="max_abs_k")

    if not chosen:
        sb.warning("Pick at least one option root.")
        st.stop()
    st.session_state[cache.SETTINGS_KEY] = cache.Settings(
        ticker=ticker, day=day, roots=tuple(chosen), window_start=start, window_end=end,
        rates_provider=provider, rates_lag_days=int(lag), parity=parity_config, moneyness_band=band,
    )
    st.session_state[cache.FILTERS_KEY] = cache.Filters(
        min_days=min_days, max_days=max_days if cap_days else None, min_price=min_price, max_abs_k=max_abs_k,
    )


def parity_sidebar(sb, window_end: time) -> ParityConfig:
    """Every filter on the pairs behind the forwards; the defaults are ``ParityConfig``'s."""
    d = ParityConfig()
    sb.header("Forwards: put-call parity")
    sb.caption("Each same-minute call/put pair gives a forward K + (C − P) / D, D from the Treasury curve. One "
               "forward per expiry, then a curve of implied dividends that never falls with T.")
    fill = sb.toggle("Fill expiries without close pairs from earlier in the day", value=d.fill_start is not None,
                     key="parity_fill", help="The forward drifts during the day, so earlier pairs only fill "
                     "expiries with too few close minutes, with a wider standard error.")
    c1, c2 = sb.columns(2)
    close = c1.time_input("Close pairs from (ET)", time.fromisoformat(d.close_start), step=300, key="parity_close",
                          help="Pairs from here to the window's end set an expiry's forward: the close tier.")
    fill_start = c2.time_input("Fill from (ET)", time.fromisoformat(d.fill_start), step=300, key="parity_fill_start",
                               disabled=not fill)
    c1, c2 = sb.columns(2)
    band_sd = c1.number_input("Band: × √T", 0.0, 2.0, d.band_sd, 0.01, key="parity_band_sd",
                              help="Keep |ln(K/S)| ≤ max(this × √T, floor), S the pair's own minute's spot.")
    floor = c2.number_input("Band floor (%)", 0.0, 20.0, 100 * d.band_floor, 0.1, key="parity_band_floor")
    estimator = sb.radio("Forward per expiry", ESTIMATORS, index=ESTIMATORS.index(d.estimator), horizontal=True,
                         key="parity_estimator")
    with sb.expander("More pair filters"):
        c1, c2 = st.columns(2)
        min_volume = c1.number_input("Min volume, each leg", 1, 1000, d.min_volume, key="parity_min_volume")
        min_price = c2.number_input("Min price, each leg", 0.0, 50.0, d.min_price, 0.05, key="parity_min_price")
        min_strikes = c1.number_input("Min strikes per minute", 1, 20, d.min_strikes, key="parity_min_strikes")
        min_minutes = c2.number_input("Min close minutes", 1, 60, d.min_minutes, key="parity_min_minutes",
                                      help="An expiry with fewer close minutes is filled from the day, or left to "
                                      "the curve when the fill is off.")
        se_floor = c1.number_input("se floor (pts)", 0.0, 10.0, d.se_floor_pts, 0.1, key="parity_se_floor")
        drift = c2.number_input("Day-tier drift (pts)", 0.0, 10.0, d.drift_pts, 0.1, key="parity_drift",
                                help="Added in quadrature to a filled forward's standard error.")
        boxes = st.toggle("Drop box-spread prints", value=d.exclude_boxes, key="parity_boxes")
    try:
        return ParityConfig(
            window_end=f"{window_end:%H:%M}", close_start=f"{close:%H:%M}",
            fill_start=f"{fill_start:%H:%M}" if fill else None, exclude_boxes=boxes, min_volume=int(min_volume),
            min_price=float(min_price), band_sd=float(band_sd), band_floor=floor / 100.0,
            min_strikes=int(min_strikes), estimator=estimator, min_minutes=int(min_minutes),
            se_floor_pts=float(se_floor), drift_pts=float(drift),
        )
    except ValueError as exc:
        sb.error(str(exc))
        st.stop()


page = st.navigation(PAGES)
sidebar()
page.run()
