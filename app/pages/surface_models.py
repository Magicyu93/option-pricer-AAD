"""The surface models side by side: fit quality, the surface against its quotes, smiles,
the ATM term structure and Dupire's local vol."""

import streamlit as st

import cache
import plots
from analysis.surface_models import MODELS

s = cache.settings()
st.title("Surface models")
cache.caption(s)

if cache.window(s).empty:
    st.warning("No bars in this window for the chosen roots.")
    st.stop()
cache.require_forwards(s)

# -- what to fit -------------------------------------------------------------------------
with st.form("fit_settings"):
    c = st.columns([3, 1, 1, 1])
    models = c[0].multiselect("Models", MODELS, default=MODELS, key="fit_models")
    min_days = c[1].number_input("Min days to expiry", 0.0, 100.0, 7.0, 1.0, key="fit_min_days")
    min_price = c[2].number_input("Min price", 0.0, 10.0, 0.05, 0.05, key="fit_min_price")
    max_abs_k = c[3].number_input("Max |k| / sqrt(T)", 0.05, 5.0, 1.0, 0.05, key="fit_max_abs_k")
    st.form_submit_button("Fit")
st.caption("The library's fit, as in a snapshot: one quote per contract, out of the money, its iv averaged by "
           "volume over the window's bars, weighted by sqrt(volume) × the vega shape. Forwards are the Treasury "
           "curve and the put-call parity dividend curve, as on the other pages. The window, rates and parity "
           "filters come from the sidebar; the sidebar's plot filters don't apply here. A fit is cached once it has run "
           "(SVI takes about 15 s).")
if not models:
    st.info("Pick at least one model.")
    st.stop()

fs = cache.FitSettings(min_days=min_days, min_price=min_price, max_abs_k=max_abs_k)
try:
    inputs = cache.fit_inputs(s, fs)
except Exception as exc:  # noqa: BLE001 -- a data problem on this day, shown rather than raised
    st.error(f"Could not build the fit's quotes and forwards: {exc}")
    st.stop()
if not inputs.quotes:
    st.warning("No quotes pass these settings.")
    st.stop()

fitted, failed = [], {}
for model in models:
    with st.spinner(f"Fitting {model}..."):
        try:
            cache.model_fit(s, fs, model)
            fitted.append(model)
        except Exception as exc:  # noqa: BLE001
            failed[model] = exc
for model, exc in failed.items():
    st.error(f"{model} fit failed: {exc}")
if not fitted:
    st.stop()
models = tuple(fitted)
resid = cache.model_residuals(s, fs, models)

# -- fit quality -------------------------------------------------------------------------
st.subheader("Fit quality")
st.caption("Errors in vol points, weighted as in the fit. The three checks are the smallest value on a dense "
           "(k, T) grid, between and beyond the expiries too: all must be ≥ 0 for a surface free of static "
           "arbitrage, which local vol and LSV need.")
summary = cache.model_summary(s, fs, models)
c1, c2 = st.columns([5, 4])
c1.dataframe(summary, hide_index=True, width="stretch", column_config={
    "rmse": st.column_config.NumberColumn("rmse (vol pts)", format="%.3f"),
    "max_abs_error": st.column_config.NumberColumn("max |error|", format="%.2f"),
    "seconds": st.column_config.NumberColumn(format="%.1f"),
    "min_butterfly": st.column_config.NumberColumn("min butterfly g", format="%.3g"),
    "min_calendar": st.column_config.NumberColumn("min calendar step", format="%.3g"),
    "min_local_variance": st.column_config.NumberColumn("min local var", format="%.3g"),
    "arbitrage_free": st.column_config.CheckboxColumn("arbitrage free"),
})
fw = inputs.forwards
c1.caption(f"{len(inputs.quotes):,} quotes over {resid.expiry.nunique()} expiries, from "
           f"{len(inputs.quote_bars):,} bars; reference spot {inputs.spot:,.2f}; forwards from the Treasury "
           "curve and the parity dividend curve.")
per_expiry = cache.per_expiry_errors(s, fs, models)
c2.plotly_chart(plots.model_errors_by_expiry(per_expiry), width="stretch")
st.plotly_chart(plots.residuals_by_model(resid, list(models)), width="stretch")

# -- the surface against the data --------------------------------------------------------
st.subheader("The surface against the quotes")
c = st.columns(4)
shown = c[0].selectbox("Model", models, index=models.index("eSSVI") if "eSSVI" in models else 0,
                       key="surface_model_3d")
sticks = c[1].toggle("Join each quote to the surface", value=True)
by_error = c[2].toggle("Colour quotes by error", value=True)
log_days = c[3].toggle("Log days axis", value=True, key="surface_log")
st.plotly_chart(plots.surface_vs_quotes_3d(cache.surface_mesh(s, fs, shown, models), resid[resid.model == shown],
                                           shown, log_days, sticks, by_error), width="stretch")
st.caption("The mesh runs over the quotes' band in k / sqrt(T), at the expiries and between them; white "
           "lines are constant days. A segment's length is the quote's fit error.")

# -- smiles ------------------------------------------------------------------------------
st.subheader("Smiles")
expiries = sorted(resid.expiry.unique())
step = max(1, len(expiries) // 8)
picked = st.multiselect("Expiries", expiries, default=expiries[::step][:8],
                        format_func=lambda e: f"{e:%Y-%m-%d}", key="smile_expiries")
if picked:
    picked = sorted(picked)
    curves = cache.smile_curves(s, fs, models, tuple(picked))
    st.plotly_chart(plots.smile_grid(resid, curves, picked, list(models)), width="stretch")

# -- term structure and local vol --------------------------------------------------------
st.subheader("Term structure and local vol")
lines, atm = cache.atm_term_structure(s, fs, models)
c1, c2 = st.columns(2)
c1.plotly_chart(plots.atm_term_structure(lines, atm), width="stretch")
lv_model = c2.selectbox("Local vol from", models, index=models.index(shown), key="local_vol_model")
c2.plotly_chart(plots.local_vol_heatmap(cache.local_vol_grid(s, fs, lv_model, models), lv_model), width="stretch")
st.caption("Local vol is Dupire's formula on the fitted surface, by autograd, off the expiries themselves; "
           "the colour scale stops at the 98th percentile. Horizontal bands are local vol stepping at each "
           "expiry, where the interpolation in T is only C^0 (open question 6 of the surface plan). The grid "
           "stays between the first and last expiry, where each model extrapolates. Only expiries whose quotes "
           "straddle k = 0 get an ATM quote point.")

# -- details -----------------------------------------------------------------------------
with st.expander("Parameters, forwards, quotes"):
    tabs = st.tabs([*models, "Forwards", "Quotes dropped"])
    for tab, model in zip(tabs, models):
        table, scalars, constraints = cache.parameter_table(s, fs, model)
        if scalars:
            tab.write({k: round(v, 6) if isinstance(v, float) else v for k, v in scalars.items()})
        tab.dataframe(table, hide_index=True, width="stretch")
        if constraints:
            tab.markdown("**Constraints** (held by construction; how close each value is to its bound)")
            tab.dataframe(constraints, width="stretch")
    tabs[-2].dataframe(fw, hide_index=True, width="stretch")
    tabs[-1].write(inputs.dropped)
