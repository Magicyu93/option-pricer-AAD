"""One expiry: its parity forward, the curve's, and the regression's, on the trades themselves."""

import streamlit as st

import cache
import plots

s, f = cache.settings(), cache.filters()
st.title("Expiry drill-down")
cache.caption(s)

if cache.window(s).empty:
    st.warning("No bars in this window for the chosen roots.")
    st.stop()
cache.require_forwards(s)
expiries = cache.expiry_table(s, f)
keys = list(expiries.expiry)
if not keys:
    st.warning("No expiries with pairs or iv points.")
    st.stop()

wanted = st.session_state.get(cache.EXPIRY_KEY)
fitted = expiries[expiries.reg_r.notna() & (expiries.days >= f.min_days)]
default = wanted if wanted in keys else fitted.expiry.iloc[0] if len(fitted) else keys[0]
labels = {k: f"{r} {k:%Y-%m-%d} ({d:.1f}d)" for k, r, d in zip(keys, expiries.option_root, expiries.days)}
expiry = st.selectbox("Expiry", keys, index=keys.index(default), format_func=labels.get)
st.session_state[cache.EXPIRY_KEY] = expiry
row = expiries.iloc[keys.index(expiry)]

# -- the numbers, side by side -----------------------------------------------------------
c = st.columns(5)
c[0].metric("Days to expiry", f"{row.days:.2f}")
c[1].metric("r: regression vs Treasury", f"{row.reg_r:.2%}" if row.reg_r == row.reg_r else "n/a",
            f"Treasury {row.tsy_r:.2%}", delta_color="off")
c[2].metric("q: regression vs curve", f"{row.reg_q:.2%}" if row.reg_q == row.reg_q else "n/a",
            f"curve {row.curve_q:.2%}", delta_color="off")
own = row.parity_forward == row.parity_forward
c[3].metric(f"Forward: own ({row.tier} tier) vs curve" if own else "Forward: none of its own",
            f"{row.parity_forward:,.2f} ± {row.parity_forward_se:.2f}" if own else "read off the curve",
            f"curve {row.curve_forward:,.2f}", delta_color="off")
c[4].metric("Regression forward − curve", f"{row.forward_gap:+.2f}" if row.forward_gap == row.forward_gap
            else "n/a", f"median iv gap {row.median_iv_gap:+.2f} vol pts" if row.median_iv_gap == row.median_iv_gap
            else None, delta_color="off")

# -- the forward through the day ---------------------------------------------------------
st.subheader("The forward through the day")
day_pairs, minutes = cache.intraday_forward(s, expiry)
if day_pairs.empty:
    st.info("No call and put traded in the same minute for this expiry in the parity window.")
else:
    c1, c2 = st.columns([3, 1])
    c1.plotly_chart(plots.intraday_forward(day_pairs, minutes, s.parity.close_start), width="stretch")
    reasons = day_pairs.dropped_by.replace("", "kept").value_counts().rename_axis("pair").rename("count")
    c2.markdown("**This expiry's pairs**")
    c2.dataframe(reasons, width="stretch")
    st.caption("Every same-minute pair's forward minus the curve's at that minute's spot and T. Kept pairs after "
               "the dotted line set the close-tier forward; earlier ones only fill an expiry with too few close "
               "minutes. A drift before the line is why.")

# -- parity on the raw prices ------------------------------------------------------------
st.subheader("Parity on the trades")
if row.n_pairs:
    pairs, lines = cache.expiry_parity_lines(s, expiry)
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.expiry_parity(pairs, lines), width="stretch")
    c2.plotly_chart(plots.expiry_residuals(pairs), width="stretch")
    st.caption("A residual offset from zero against the forward curve, but centred against the regression, means "
               "the curve's forward is off here. The regression and its band are a diagnostic only.")
else:
    pairs = None
    st.info(f"No minute has a call and a put at two or more strikes within ±{s.moneyness_band:.0%} of spot, "
            "so there is no regression for this expiry.")

# -- the vols ----------------------------------------------------------------------------
st.subheader("The vols")
pts = cache.points(s, f)
pts = pts[pts.expiry == expiry]
ivp = cache.iv_pairs(s, f)
ivp = ivp[ivp.expiry == expiry]
c1, c2 = st.columns(2)
if pts.empty:
    c1.info("No iv points pass the filters for this expiry.")
else:
    c1.plotly_chart(plots.expiry_smile(pts, ivp), width="stretch")
if ivp.empty:
    c2.info("No call and put at the same strike and minute for this expiry.")
else:
    c2.plotly_chart(plots.iv_gap_by_minute(ivp), width="stretch")

with st.expander("Tables: regression pairs, iv pairs, iv points"):
    if pairs is not None:
        st.markdown("**Regression pairs**")
        st.dataframe(pairs, hide_index=True, width="stretch")
    st.markdown("**iv pairs**")
    st.dataframe(ivp, hide_index=True, width="stretch")
    st.markdown("**iv points**")
    st.dataframe(pts, hide_index=True, width="stretch")
