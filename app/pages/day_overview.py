"""All expiries at once: the forwards and the pairs behind them, rates and carry, whether
parity holds, and the surface."""

import streamlit as st

import cache
import plots

s, f = cache.settings(), cache.filters()
st.title("Day overview")
cache.caption(s)

if cache.window(s).empty:
    st.warning("No bars in this window for the chosen roots.")
    st.stop()
try:
    snap = cache.par_yields(s.day, s.rates_provider, s.rates_lag_days)
except (LookupError, RuntimeError, ValueError) as exc:
    st.error(f"No Treasury yields for {s.day}: {exc}")
    st.stop()
cache.require_forwards(s)

expiries = cache.expiry_table(s, f)
shown = expiries[expiries.days >= f.min_days]  # same day filter as the iv points

# -- the forwards ------------------------------------------------------------------------
st.subheader("Forwards from put-call parity")
st.markdown("**Pair filters, in order** (sidebar → *Forwards: put-call parity*): "
            + " · ".join(s.parity.describe()))
st.caption("Each kept pair's forward is K + (C − P) / D with the Treasury D at its own minute. One forward per "
           "expiry (filled: close tier; hollow: filled from the day), then the curve of implied dividends "
           "S − D·F, fitted so it never falls with T; expiries without a forward of their own (×) are read off it. "
           "Every iv on every page uses this curve's forwards.")
own, line, interpolated = cache.implied_dividends(s, f)
c1, c2, c3 = st.columns(3)
c1.plotly_chart(plots.implied_dividends(own, line, interpolated), width="stretch")
c2.plotly_chart(plots.own_vs_curve(expiries), width="stretch")
c3.plotly_chart(plots.filter_funnel(cache.filter_funnel(s)), width="stretch")
fw = cache.parity_forwards(s)
st.caption(f"{len(fw)} expiries with a forward of their own ({int((fw.tier == 'close').sum())} close, "
           f"{int((fw.tier == 'day').sum())} filled from the day); {len(interpolated)} more on this page read off "
           "the curve. The middle panel shows how far the monotone fit moved each expiry's own forward.")

# -- rates and carry ---------------------------------------------------------------------
st.subheader("Rates and carry")
st.caption("The ivs use the Treasury zero rate and the parity dividend curve's zero rate q(T). The parity "
           "regression estimates both from the trades alone, per expiry, as a diagnostic; error bars are ±1 "
           "standard error, clustered by minute.")
zoom = st.toggle("Zoom to the curves (noisy short expiries go off the scale)", value=True)
c1, c2 = st.columns(2)
t_min = min(1 / 52, float(shown["T"].min())) if len(shown) else 1 / 52
c1.plotly_chart(plots.rates_vs_parity(cache.rate_curve_table(s, t_min), snap, shown, zoom), width="stretch")
c2.plotly_chart(plots.carry_vs_parity(shown, zoom), width="stretch")

# -- does parity hold --------------------------------------------------------------------
st.subheader("Does parity hold?")
st.caption("Forward gap: the curve's forward minus each iv pair's own K + (C − P) / D, both with the Treasury D. "
           "Centred on zero where the curve fits the trades.")
pairs = cache.iv_pairs(s, f)
c1, c2, c3 = st.columns(3)
if pairs.empty:
    c1.info("No call and put traded at the same strike and minute with these filters.")
else:
    c1.plotly_chart(plots.forward_error_by_expiry(cache.iv_parity_by_expiry(s, f)), width="stretch")
    c2.plotly_chart(plots.iv_gap_vs_k(pairs), width="stretch")
c3.plotly_chart(plots.pairs_per_expiry(shown), width="stretch")

# -- the surface -------------------------------------------------------------------------
st.subheader("The surface")
SIDE_LABELS = {"both": "both", "otm": "out of the money", "itm": "in the money"}


def side_picker(container, key: str, default: str) -> str:
    return container.radio("Side", list(SIDE_LABELS), index=list(SIDE_LABELS).index(default),
                           format_func=SIDE_LABELS.get, horizontal=True, key=key)


smile_tab, iv_tab, both_tab = st.tabs(["Smiles", "Implied vol 3D", "Price and iv 3D"])
with smile_tab:
    c = st.columns(3)
    side = side_picker(c[0], "smile_side", "both")
    standardized = c[1].toggle("x = k / sqrt(T)", value=True)
    band = c[2].number_input("Near-the-money band (× sqrt(T))", 0.01, 1.0, 0.05, 0.01)
    pts = cache.points(s, f, side)
    if pts.empty:
        st.info("No rows pass the filters.")
    else:
        st.plotly_chart(plots.smiles(pts, cache.near_the_money(pts, band), standardized, band), width="stretch")
with iv_tab:
    c = st.columns(3)
    side = side_picker(c[0], "iv_side", "otm")
    overlay = c[1].toggle("Overlay call-put pairs", value=True)
    log_days = c[2].toggle("Log days axis", value=True, key="iv_log")
    pts = cache.points(s, f, side)
    if pts.empty:
        st.info("No rows pass the filters.")
    else:
        st.plotly_chart(plots.iv_3d(pts, pairs if overlay and len(pairs) else None, log_days), width="stretch")
with both_tab:
    c = st.columns(3)
    price_side = side_picker(c[0], "price_side", "both")
    iv_side = side_picker(c[1], "both_iv_side", "otm")
    log_days = c[2].toggle("Log days axis", value=True, key="both_log")
    price_pts, iv_pts = cache.points(s, f, price_side), cache.points(s, f, iv_side)
    if price_pts.empty or iv_pts.empty:
        st.info("No rows pass the filters.")
    else:
        st.caption("The panels rotate independently; a legend click hides a layer in both.")
        st.plotly_chart(plots.price_and_iv_3d(price_pts, iv_pts, log_days,
                                              (f"Option price ({SIDE_LABELS[price_side]})",
                                               f"Implied vol ({SIDE_LABELS[iv_side]})")), width="stretch")

# -- per expiry --------------------------------------------------------------------------
st.subheader("Per expiry")
st.caption("Every number about each expiry in one row. `tier`: where its forward comes from (close, day, or "
           "curve: none of its own). `own_gap` = its own parity forward − the curve's; `forward_gap` = the "
           "regression's forward − the curve's, in index points. Select a row to open it in the drill-down.")
PCT_COLUMNS = ["reg_r", "reg_r_se", "tsy_r", "reg_q", "reg_q_se", "curve_q"]
pct = st.column_config.NumberColumn(format="%.2f%%")
view = expiries.assign(**{c: 100 * expiries[c] for c in PCT_COLUMNS})
event = st.dataframe(
    view, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="expiry_table",
    column_config={
        "expiry": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "days": st.column_config.NumberColumn(format="%.2f"),
        "T": None,
        "spot": st.column_config.NumberColumn(format="%.2f"),
        "reg_discount": st.column_config.NumberColumn(format="%.6f"),
        **{c: pct for c in PCT_COLUMNS},
        **{c: st.column_config.NumberColumn(format="%.2f")
           for c in ["parity_forward", "parity_forward_se", "own_gap", "reg_forward", "curve_forward",
                     "forward_gap", "median_iv_gap", "median_F_error"]},
    },
)
if event.selection.rows:
    row = expiries.iloc[event.selection.rows[0]]
    st.session_state[cache.EXPIRY_KEY] = row.expiry
    st.switch_page("pages/expiry_drilldown.py")
