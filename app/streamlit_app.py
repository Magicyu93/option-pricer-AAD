"""Vol surface research app: one day of cached option bars, from the notebook's analysis.

Run from the repo root: ``streamlit run app/streamlit_app.py``
(``pip install -e .[app]`` for the dependencies).

Two pages, each putting the rates, the forwards and the call vs put vols side by side:
a day overview across expiries, and a drill-down into one expiry. The sidebar's choices
apply to both. Computation lives in ``app/analysis`` and the library; pages only plot.
"""

from __future__ import annotations

from datetime import date, time

import streamlit as st

import cache

st.set_page_config(page_title="Vol surface research", layout="wide")

PAGES = [
    st.Page("pages/day_overview.py", title="Day overview", default=True),
    st.Page("pages/expiry_drilldown.py", title="Expiry drill-down"),
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

    sb.header("Rates and carry")
    provider = sb.selectbox("Treasury yields from", ["massive", "fred"], key="rates_provider")
    lag = sb.number_input("Yield lag (days)", 0, 10, 0, help="Use the yields published on or before "
                          "day - lag; 1 avoids using the day's own end-of-day print.", key="rates_lag")
    q = sb.number_input("Flat dividend yield (%)", 0.0, 10.0, 1.1, 0.05, format="%.2f", key="dividend_yield")
    band = sb.number_input("Parity regression: max |K/S - 1|", 0.01, 0.5, 0.10, 0.01, key="moneyness_band")

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
        rates_provider=provider, rates_lag_days=int(lag), dividend_yield=q / 100.0, moneyness_band=band,
    )
    st.session_state[cache.FILTERS_KEY] = cache.Filters(
        min_days=min_days, max_days=max_days if cap_days else None, min_price=min_price, max_abs_k=max_abs_k,
    )


page = st.navigation(PAGES)
sidebar()
page.run()
