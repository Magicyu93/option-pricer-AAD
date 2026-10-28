"""Plotly figures for the pages: each draws the frames it is given, nothing more.

Encoding, the same everywhere: colour = days to expiry (light short, dark long);
filled = out of the money, hollow = in the money; call = diamond, put = square, a call
and a put of one pair joined by a grey segment whose length is their iv gap. Click
legend entries to hide or show a layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MUTED, GRID = "#898781", "rgba(137, 135, 129, 0.25)"
POINT_COLOR, ACCENT = "#2a78d6", "#eb6834"
PAR_COLOR, ZERO_COLOR, FORWARD_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
EXPIRY_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # short expiries light, long dark
EXPIRY_COLORSCALE = [[i / (len(EXPIRY_RAMP) - 1), c] for i, c in enumerate(EXPIRY_RAMP)]
DAY_TICKS = {1: "1d", 7: "1w", 14: "2w", 30: "1m", 61: "2m", 91: "3m", 182: "6m", 365: "1y",
             730: "2y", 1461: "4y"}
TENOR_TICKS = {1 / 365: "1D", 1 / 52: "1W", 1 / 12: "1M", 0.25: "3M", 0.5: "6M", 1: "1Y", 2: "2Y", 3: "3Y",
               5: "5Y", 7: "7Y", 10: "10Y", 20: "20Y", 30: "30Y"}
SIDES = {"otm": "out of the money", "itm": "in the money"}
Z_AXES = {"iv": ("implied vol (%)", 100.0), "price": ("option price (index points)", 1.0)}
CAMERA = dict(eye=dict(x=1.6, y=-1.6, z=0.9))


def describe(points: pd.DataFrame) -> str:
    unit = "bars" if "timestamp" in points.columns else "contracts"
    n = {"otm": int(points.otm.sum()), "itm": int((~points.otm).sum())}
    return " and ".join(f"{v:,} {SIDES[k]}" for k, v in n.items() if v) + f" {unit}"


# -- shared encoding -------------------------------------------------------------------


def _day_ticks(days: pd.Series) -> dict:
    """Readable ticks for a log days axis, only those inside the data's range."""
    shown = {d: t for d, t in DAY_TICKS.items() if days.min() * 0.9 <= d <= days.max() * 1.1}
    return dict(tickvals=list(shown), ticktext=list(shown.values()))


def _day_marker(days: pd.Series, day_range: pd.Series, showscale: bool, **marker) -> dict:
    """Marker coloured by log days on the expiry ramp, scaled to ``day_range`` so traces agree."""
    ticks = _day_ticks(day_range)
    return dict(
        color=np.log10(days), colorscale=EXPIRY_COLORSCALE,
        cmin=np.log10(day_range.min()), cmax=np.log10(day_range.max()), showscale=showscale,
        colorbar=dict(title="days", len=0.7, tickvals=np.log10(ticks["tickvals"]), ticktext=ticks["ticktext"]),
        **marker,
    )


def _point_hover(points: pd.DataFrame) -> tuple[np.ndarray, str]:
    """customdata and hovertemplate for one marker per row, with every raw value."""
    per_minute = "timestamp" in points.columns
    data = pd.DataFrame({
        "expiry": points.expiry.astype(str), "right": points.option_type,
        "strike": points.strike, "side": np.where(points.otm, SIDES["otm"], SIDES["itm"]),
        "when": points.time if per_minute else points.n_bars, "spot": points.spot,
        "k": points.k, "days": points.days, "price": points.price, "iv": 100 * points.iv,
        "volume": points.volume,
    })
    when = ("traded %{customdata[4]} ET, spot %{customdata[5]:.2f}" if per_minute
            else "%{customdata[4]} minutes, average spot %{customdata[5]:.2f}")
    template = (
        "%{customdata[0]} %{customdata[1]} K=%{customdata[2]:.0f}, %{customdata[3]}<br>" + when + "<br>"
        "k=%{customdata[6]:.4f}, %{customdata[7]:.2f} days<br>"
        "price %{customdata[8]:.2f}, iv %{customdata[9]:.2f}%<br>"
        + ("volume this minute" if per_minute else "volume in the window") + " %{customdata[10]:.0f}"
        "<extra></extra>"
    )
    return data.to_numpy(), template


def _pair_hover(pairs: pd.DataFrame) -> tuple[np.ndarray, str]:
    """customdata and hovertemplate for one call-put pair: both legs and the parity checks."""
    data = pd.DataFrame({
        "expiry": pairs.expiry.astype(str), "strike": pairs.strike, "time": pairs.time,
        "spot": pairs.spot, "k": pairs.k, "days": pairs.days,
        "call": pairs.price_call, "put": pairs.price_put, "iv_call": 100 * pairs.iv_call,
        "iv_put": 100 * pairs.iv_put, "gap": pairs.iv_gap, "residual": pairs.parity_residual,
        "F": pairs.forward, "F_pair": pairs.F_pair,
    })
    template = (
        "%{customdata[0]} K=%{customdata[1]:.0f}, traded %{customdata[2]} ET, spot %{customdata[3]:.2f}<br>"
        "k=%{customdata[4]:.4f}, %{customdata[5]:.2f} days<br>"
        "call %{customdata[6]:.2f} (iv %{customdata[8]:.2f}%), put %{customdata[7]:.2f} (iv %{customdata[9]:.2f}%)<br>"
        "iv gap %{customdata[10]:+.2f} vol pts, residual vs Treasury + flat q %{customdata[11]:+.2f}<br>"
        "F: Treasury + flat q %{customdata[12]:.2f}, parity %{customdata[13]:.2f}<extra></extra>"
    )
    return data.to_numpy(), template


def _style(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font_size=14), height=height,
        legend=dict(itemsizing="constant", orientation="h", y=-0.22),
        margin=dict(l=55, r=15, t=55, b=50),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED)
    return fig


def _log_days_axis(days: pd.Series) -> dict:
    return dict(type="log", title="days to expiry", **_day_ticks(days))


def _tenor_axis(t_min: float, t_max: float) -> dict:
    ticks = {t: label for t, label in TENOR_TICKS.items() if t_min * 0.9 <= t <= t_max * 1.1}
    return dict(type="log", title="maturity", tickvals=list(ticks), ticktext=list(ticks.values()))


def _zoom(fig: go.Figure, values: np.ndarray, pad: float) -> None:
    """Fix the y range to ``values`` plus ``pad`` either side."""
    fig.update_yaxes(range=[np.nanmin(values) - pad, np.nanmax(values) + pad])


# -- page 1: rates and carry -------------------------------------------------------------


def rates_vs_parity(tables: dict[str, pd.DataFrame], published, expiries: pd.DataFrame,
                    zoom: bool) -> go.Figure:
    """Treasury par / zero / forward curves, and each expiry's parity-regression r with ±1 se."""
    curve, par, pillars = tables["curve"], tables["par"], tables["pillars"]
    reg = expiries.dropna(subset=["reg_r"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.t, y=100 * curve.forward, mode="lines", name="Treasury forward f(T)",
                             line=dict(color=FORWARD_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=curve.t, y=100 * curve.zero, mode="lines", name="Treasury zero r(T)",
                             line=dict(color=ZERO_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=par.t, y=100 * par.par, mode="lines", name="Treasury par (curve)",
                             line=dict(color=PAR_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=published.tenors, y=published.yields_pct, mode="markers",
                             name="Treasury par (published)",
                             marker=dict(symbol="diamond-open", color=PAR_COLOR, size=9, line=dict(width=2))))
    fig.add_trace(go.Scatter(
        x=reg["T"], y=100 * reg.reg_r, mode="markers", name="parity regression r ± se",
        marker=dict(color=MUTED, size=8, line=dict(color="white", width=1)),
        error_y=dict(type="data", array=100 * reg.reg_r_se, color=MUTED, thickness=1),
        customdata=np.column_stack([reg.expiry.astype(str), reg.days, 100 * reg.tsy_r, reg.n_pairs]),
        hovertemplate=("%{customdata[0]} (%{customdata[1]:.1f}d, %{customdata[3]} pairs)<br>parity r %{y:.2f}%"
                       " vs Treasury %{customdata[2]:.2f}%<extra></extra>")))
    fig.update_xaxes(**_tenor_axis(float(curve.t.min()), float(pillars.t.max())))
    fig.update_yaxes(title_text="rate (%)", ticksuffix="%")
    if zoom:
        _zoom(fig, 100 * np.concatenate([curve.zero, curve.forward, par.par]), 1.5)
    title = f"Discount rate: Treasury ({published.observation_date}) vs put-call parity"
    return _style(fig, title, 430)


def carry_vs_parity(expiries: pd.DataFrame, flat_q: float, zoom: bool) -> go.Figure:
    """Each expiry's parity-regression q with ±1 se, against the flat q the ivs use."""
    reg = expiries.dropna(subset=["reg_q"])
    fig = go.Figure()
    t = expiries["T"]
    fig.add_trace(go.Scatter(x=[t.min(), t.max()], y=[100 * flat_q] * 2, mode="lines", name="flat q (used for ivs)",
                             line=dict(color=ZERO_COLOR, width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=reg["T"], y=100 * reg.reg_q, mode="markers", name="parity regression q ± se",
        marker=dict(color=MUTED, size=8, line=dict(color="white", width=1)),
        error_y=dict(type="data", array=100 * reg.reg_q_se, color=MUTED, thickness=1),
        customdata=np.column_stack([reg.expiry.astype(str), reg.days, reg.n_pairs]),
        hovertemplate="%{customdata[0]} (%{customdata[1]:.1f}d, %{customdata[2]} pairs)<br>parity q %{y:.2f}%"
                      "<extra></extra>"))
    fig.update_xaxes(**_tenor_axis(float(t.min()), float(t.max())))
    fig.update_yaxes(title_text="dividend yield (%)", ticksuffix="%")
    if zoom:
        _zoom(fig, np.array([100 * flat_q]), 3.0)
    return _style(fig, "Carry: flat q vs put-call parity", 430)


# -- page 1: does parity hold ------------------------------------------------------------


def forward_error_by_expiry(by_expiry: pd.DataFrame) -> go.Figure:
    """Median ``F - F_pair`` per expiry (Treasury + flat q forward minus the pair's forward at
    the parity regression's D), with the middle half of the pairs as an error bar."""
    fig = go.Figure(go.Scatter(
        x=by_expiry.days, y=by_expiry.F_error_median, mode="markers",
        marker=_day_marker(by_expiry.days, by_expiry.days, showscale=False, size=9, line=dict(color="white", width=1)),
        error_y=dict(type="data", symmetric=False, array=by_expiry.F_error_q75 - by_expiry.F_error_median,
                     arrayminus=by_expiry.F_error_median - by_expiry.F_error_q25, color=MUTED, thickness=1),
        customdata=np.column_stack([by_expiry.expiry.astype(str), by_expiry.n,
                                    by_expiry.F_error_q25, by_expiry.F_error_q75]),
        hovertemplate=("%{customdata[0]}, %{customdata[1]} pairs<br>median F error %{y:+.2f}<br>"
                       "middle half %{customdata[2]:+.2f} to %{customdata[3]:+.2f}<extra></extra>")))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(**_log_days_axis(by_expiry.days))
    fig.update_yaxes(title_text="index points")
    return _style(fig, "Forward: Treasury + flat q − parity", 380)


def iv_gap_vs_k(pairs: pd.DataFrame) -> go.Figure:
    """Call - put implied vol of every pair, against log-moneyness."""
    customdata, template = _pair_hover(pairs)
    fig = go.Figure(go.Scattergl(x=pairs.k, y=pairs.iv_gap, mode="markers",
                                 marker=_day_marker(pairs.days, pairs.days, showscale=True, size=5),
                                 customdata=customdata, hovertemplate=template))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title_text="k = ln(K / F)")
    fig.update_yaxes(title_text="vol points")
    return _style(fig, f"Call − put iv, {len(pairs):,} pairs", 380)


def pairs_per_expiry(expiries: pd.DataFrame) -> go.Figure:
    """How many pairs each check rests on: the regression's (|K/S - 1| band) and the ivs'."""
    labels = [f"{e:%m-%d} {r}" for r, e in zip(expiries.option_root, expiries.expiry)]
    fig = go.Figure([
        go.Bar(x=labels, y=expiries.n_pairs, name="regression pairs", marker_color=POINT_COLOR),
        go.Bar(x=labels, y=expiries.n_iv_pairs, name="iv pairs", marker_color=ACCENT),
    ])
    fig.update_layout(barmode="group")
    fig.update_yaxes(type="log", title_text="pairs (log)")
    fig.update_xaxes(tickangle=-90)
    return _style(fig, "Pairs per expiry", 380)


# -- page 1: the surface -------------------------------------------------------------------


def smiles(points: pd.DataFrame, near: pd.Series, standardized: bool, atm_band: float) -> go.Figure:
    """Implied vol against k (or k / sqrt(T)) by expiry, and near-the-money vol against days."""
    x = points.k / np.sqrt(points["T"]) if standardized else points.k
    customdata, template = _point_hover(points)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.08,
                        subplot_titles=["Smile by expiry", f"Near the money, |k| <= {atm_band:g} sqrt(T)"])
    shown_scale = False
    for otm, label in ((True, SIDES["otm"]), (False, SIDES["itm"])):
        part = points.otm == otm
        if not part.any():
            continue
        symbol = "circle" if otm else "circle-open"
        fig.add_trace(go.Scattergl(
            x=x[part], y=100 * points.iv[part], mode="markers", name=label, legendgroup=label,
            marker=_day_marker(points.days[part], points.days, showscale=not shown_scale, size=5, symbol=symbol),
            customdata=customdata[part.to_numpy()], hovertemplate=template), row=1, col=1)
        shown_scale = True
        sel = part & near
        fig.add_trace(go.Scattergl(
            x=points.days[sel], y=100 * points.iv[sel], mode="markers", name=label, legendgroup=label,
            showlegend=False, marker=dict(color=POINT_COLOR, size=5, symbol=symbol, opacity=0.6),
            customdata=customdata[sel.to_numpy()], hovertemplate=template), row=1, col=2)
    fig.add_vline(x=0, line=dict(color=MUTED, dash="dot", width=1), row=1, col=1)
    fig.update_xaxes(title_text="k / sqrt(T)" if standardized else "k = ln(K / F)", row=1, col=1)
    fig.update_xaxes(**_log_days_axis(points.days), row=1, col=2)
    fig.update_yaxes(title_text="implied vol (%)", row=1, col=1)
    return _style(fig, f"Implied vols: {describe(points)}, {points.expiry.nunique()} expiries", 540)


def _scene(ranges: pd.DataFrame, z: str, log_days: bool) -> dict:
    """Axis titles, the k / days ranges of ``ranges`` and the starting camera for one panel."""
    axis = dict(gridcolor=GRID, showbackground=False)
    days_axis = dict(axis, title="days to expiry")
    if log_days:
        days_axis.update(type="log", **_day_ticks(ranges.days))
    else:
        days_axis.update(range=[ranges.days.min(), ranges.days.max()])
    return dict(xaxis=dict(axis, title="k = ln(K / F)", range=[ranges.k.min(), ranges.k.max()]),
                yaxis=days_axis, zaxis=dict(axis, title=Z_AXES[z][0]), camera=CAMERA, aspectmode="cube")


def _point_traces_3d(points: pd.DataFrame, z: str, day_range: pd.Series, showscale: bool,
                     legend: set) -> list[go.Scatter3d]:
    """Out-of-the-money (filled) and in-the-money (hollow) markers; ``legend`` tracks shown entries."""
    customdata, template = _point_hover(points)
    traces = []
    for otm, label in ((True, SIDES["otm"]), (False, SIDES["itm"])):
        part = (points.otm == otm).to_numpy()
        if not part.any():
            continue
        traces.append(go.Scatter3d(
            x=points.k[part], y=points.days[part], z=Z_AXES[z][1] * points[z][part], mode="markers",
            name=label, legendgroup=label, showlegend=label not in legend,
            marker=_day_marker(points.days[part], day_range, showscale=showscale and not traces,
                               size=2.5 if otm else 3, symbol="circle" if otm else "circle-open"),
            customdata=customdata[part], hovertemplate=template))
        legend.add(label)
    return traces


def _parity_traces_3d(pairs: pd.DataFrame, day_range: pd.Series, legend: set) -> list[go.Scatter3d]:
    """Each pair's call (diamond) and put (square) iv, joined by a grey segment."""
    customdata, template = _pair_hover(pairs)
    nan = np.full(len(pairs), np.nan)
    segments = go.Scatter3d(
        x=np.column_stack([pairs.k, pairs.k, nan]).ravel(),
        y=np.column_stack([pairs.days, pairs.days, nan]).ravel(),
        z=np.column_stack([100 * pairs.iv_call, 100 * pairs.iv_put, nan]).ravel(),
        mode="lines", line=dict(color=MUTED, width=2), name="call − put gap",
        legendgroup="call − put gap", showlegend="call − put gap" not in legend, hoverinfo="skip")
    traces = [segments]
    for leg, symbol in (("call", "diamond"), ("put", "square")):
        name = f"pair: {leg}"
        traces.append(go.Scatter3d(
            x=pairs.k, y=pairs.days, z=100 * pairs[f"iv_{leg}"], mode="markers", name=name, legendgroup=name,
            showlegend=name not in legend,
            marker=_day_marker(pairs.days, day_range, showscale=False, size=3, symbol=symbol),
            customdata=customdata, hovertemplate=template))
    legend.update(["call − put gap", "pair: call", "pair: put"])
    return traces


def _style_3d(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.01, font_size=14), height=height,
                      margin=dict(l=0, r=0, t=55, b=0), legend=dict(itemsizing="constant", x=0.01, y=0.92))
    return fig


def iv_3d(points: pd.DataFrame, pairs: pd.DataFrame | None, log_days: bool) -> go.Figure:
    """Implied vol over log-moneyness and days to expiry, optionally with the call-put pairs."""
    frames = [points[["k", "days"]]] + ([pairs[["k", "days"]]] if pairs is not None else [])
    ranges = pd.concat(frames)
    legend: set = set()
    fig = go.Figure(_point_traces_3d(points, "iv", ranges.days, showscale=True, legend=legend))
    if pairs is not None:
        fig.add_traces(_parity_traces_3d(pairs, ranges.days, legend))
    fig.update_layout(scene=_scene(ranges, "iv", log_days))
    extra = f", {len(pairs):,} call-put pairs" if pairs is not None else ""
    return _style_3d(fig, f"Implied vols: {describe(points)}{extra}", 680)


def price_and_iv_3d(price_points: pd.DataFrame, iv_points: pd.DataFrame, log_days: bool,
                    subtitles: tuple[str, str]) -> go.Figure:
    """Price (left) and implied vol (right) on shared k / days axes and starting camera."""
    ranges = pd.concat([price_points[["k", "days"]], iv_points[["k", "days"]]])
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=list(subtitles), horizontal_spacing=0.02)
    legend: set = set()
    for trace in _point_traces_3d(price_points, "price", ranges.days, showscale=False, legend=legend):
        fig.add_trace(trace, row=1, col=1)
    for trace in _point_traces_3d(iv_points, "iv", ranges.days, showscale=True, legend=legend):
        fig.add_trace(trace, row=1, col=2)
    fig.update_layout(scene=_scene(ranges, "price", log_days), scene2=_scene(ranges, "iv", log_days))
    return _style_3d(fig, f"Price: {describe(price_points)}; iv: {describe(iv_points)}", 680)


# -- page 2: one expiry ------------------------------------------------------------------


def expiry_parity(pairs: pd.DataFrame, lines: pd.DataFrame) -> go.Figure:
    """Raw C - P against K, with the regression's line and the iv inputs' line."""
    fig = go.Figure()
    for box, name, color, size in ((False, "pair", POINT_COLOR, 6), (True, "box print", ACCENT, 9)):
        part = pairs[pairs.is_box == box]
        if len(part):
            fig.add_trace(go.Scattergl(
                x=part.strike, y=part.parity, mode="markers", name=name,
                marker=dict(color=color, size=size, opacity=0.6 if not box else 1),
                customdata=part.timestamp.dt.strftime("%H:%M"),
                hovertemplate="K=%{x:.0f}, C − P = %{y:.2f}, %{customdata} ET<extra></extra>"))
    fig.add_trace(go.Scatter(x=lines.strike, y=lines.regression, mode="lines", name="regression: A − D·K",
                             line=dict(color=MUTED, width=2)))
    fig.add_trace(go.Scatter(x=lines.strike, y=lines.curve, mode="lines",
                             name="Treasury + flat q: D(F − K)", line=dict(color=ZERO_COLOR, width=2, dash="dash")))
    fig.update_xaxes(title_text="strike K")
    fig.update_yaxes(title_text="C − P (index points)")
    return _style(fig, "C − P against strike", 420)


def expiry_residuals(pairs: pd.DataFrame) -> go.Figure:
    """Each pair's distance from both lines, at its own minute's spot."""
    time = pairs.timestamp.dt.strftime("%H:%M")
    fig = go.Figure([
        go.Scattergl(x=pairs.strike, y=pairs.resid_regression, mode="markers", name="vs regression",
                     marker=dict(color=MUTED, size=6, opacity=0.7), customdata=time,
                     hovertemplate="K=%{x:.0f}, %{customdata} ET: %{y:+.2f}<extra></extra>"),
        go.Scattergl(x=pairs.strike, y=pairs.resid_curve, mode="markers", name="vs Treasury + flat q",
                     marker=dict(color=ZERO_COLOR, size=6, opacity=0.7, symbol="x"), customdata=time,
                     hovertemplate="K=%{x:.0f}, %{customdata} ET: %{y:+.2f}<extra></extra>"),
    ])
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title_text="strike K")
    fig.update_yaxes(title_text="residual (index points)")
    return _style(fig, "Residual from each line, per minute", 420)


def expiry_smile(points: pd.DataFrame, pairs: pd.DataFrame) -> go.Figure:
    """One expiry's ivs: calls diamonds, puts squares, OTM filled, ITM hollow; pairs joined."""
    customdata, template = _point_hover(points)
    fig = go.Figure()
    if len(pairs):
        nan = np.full(len(pairs), np.nan)
        fig.add_trace(go.Scatter(
            x=np.column_stack([pairs.k, pairs.k, nan]).ravel(),
            y=np.column_stack([100 * pairs.iv_call, 100 * pairs.iv_put, nan]).ravel(),
            mode="lines", line=dict(color=MUTED, width=1), name="call − put pair", hoverinfo="skip"))
    for right, symbol, color in (("call", "diamond", POINT_COLOR), ("put", "square", ACCENT)):
        for otm in (True, False):
            part = ((points.option_type == right) & (points.otm == otm)).to_numpy()
            if part.any():
                fig.add_trace(go.Scattergl(
                    x=points.k[part], y=100 * points.iv[part], mode="markers",
                    name=f"{right}, {SIDES['otm' if otm else 'itm']}",
                    marker=dict(color=color, size=6, symbol=symbol if otm else f"{symbol}-open", opacity=0.7),
                    customdata=customdata[part], hovertemplate=template))
    fig.add_vline(x=0, line=dict(color=MUTED, dash="dot", width=1))
    fig.update_xaxes(title_text="k = ln(K / F)")
    fig.update_yaxes(title_text="implied vol (%)")
    return _style(fig, f"Smile: {describe(points)}", 460)


def iv_gap_by_minute(pairs: pd.DataFrame) -> go.Figure:
    """Call - put iv against strike, coloured by the minute the pair traded."""
    customdata, template = _pair_hover(pairs)
    minute = pairs.timestamp.dt.hour * 60 + pairs.timestamp.dt.minute
    fig = go.Figure(go.Scattergl(
        x=pairs.strike, y=pairs.iv_gap, mode="markers",
        marker=dict(color=minute, colorscale="Viridis", size=7, showscale=True,
                    colorbar=dict(title="ET", tickvals=sorted(set(minute))[::5],
                                  ticktext=[f"{m // 60:02d}:{m % 60:02d}" for m in sorted(set(minute))[::5]])),
        customdata=customdata, hovertemplate=template))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title_text="strike K")
    fig.update_yaxes(title_text="call − put iv (vol points)")
    return _style(fig, f"Call − put iv by minute, {len(pairs)} pairs", 460)


# -- page 3: surface models ----------------------------------------------------------------

MODEL_COLORS = {"SSVI": ACCENT, "eSSVI": FORWARD_COLOR, "SVI": "#8a5cd1", "Flat": MUTED}
ERROR_SCALE = "RdBu_r"  # model above the quote red, below blue


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, MUTED)


def _quote_hover(resid: pd.DataFrame) -> tuple[np.ndarray, str]:
    """customdata and hovertemplate for one fitted quote: the market's iv and the model's."""
    data = pd.DataFrame({
        "model": resid.model, "expiry": resid.expiry.astype(str), "right": resid.option_type,
        "strike": resid.strike, "k": resid.k, "days": resid.days, "iv": 100 * resid.iv,
        "model_iv": 100 * resid.model_iv, "error": 100 * resid.error, "volume": resid.volume,
    })
    template = (
        "%{customdata[1]} %{customdata[2]} K=%{customdata[3]:.0f}<br>"
        "k=%{customdata[4]:.4f}, %{customdata[5]:.2f} days<br>"
        "quote iv %{customdata[6]:.2f}%, %{customdata[0]} %{customdata[7]:.2f}% "
        "(error %{customdata[8]:+.2f} vol pts)<br>volume in the window %{customdata[9]:.0f}<extra></extra>"
    )
    return data.to_numpy(), template


def model_errors_by_expiry(per_expiry: pd.DataFrame) -> go.Figure:
    """Each model's weighted RMSE (top) and mean error (bottom) per expiry."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["RMSE per expiry", "Mean error (model − quote) per expiry"])
    for model, part in per_expiry.groupby("model", sort=False):
        part = part.sort_values("days")
        custom = np.column_stack([part.expiry.astype(str), part.quotes, part.max_abs])
        for row, col in ((1, "rmse"), (2, "bias")):
            fig.add_trace(go.Scatter(
                x=part.days, y=part[col], mode="lines+markers", name=model, legendgroup=model,
                showlegend=row == 1, line=dict(color=_model_color(model), width=1.5), marker=dict(size=6),
                customdata=custom,
                hovertemplate=(f"{model} %{{customdata[0]}}, %{{customdata[1]:.0f}} quotes<br>{col} %{{y:.3f}} "
                               "vol pts<br>largest error %{customdata[2]:.2f}<extra></extra>")), row=row, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=2, col=1)
    fig.update_xaxes(**_log_days_axis(per_expiry.days), row=2, col=1)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(title_text="vol points", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="vol points", row=2, col=1)
    return _style(fig, "Fit error by expiry (weighted as in the fit)", 520)


def surface_vs_quotes_3d(mesh: dict, resid: pd.DataFrame, model: str, log_days: bool, sticks: bool,
                         color_by_error: bool) -> go.Figure:
    """The fitted surface as a mesh, the quotes it was fitted to as points, and optionally
    a segment from each quote to the surface whose length is the fit error."""
    customdata, template = _quote_hover(resid)
    fig = go.Figure(go.Surface(
        x=mesh["k"], y=mesh["days"], z=100 * mesh["iv"], colorscale=EXPIRY_COLORSCALE, opacity=0.55,
        showscale=False, name=f"{model} surface", showlegend=True,
        hovertemplate=f"{model}<br>k=%{{x:.3f}}, %{{y:.1f}} days<br>iv %{{z:.2f}}%<extra></extra>",
        contours=dict(x=dict(show=False), y=dict(show=True, color="white", width=1, highlight=False))))
    if sticks:
        nan = np.full(len(resid), np.nan)
        fig.add_trace(go.Scatter3d(
            x=np.column_stack([resid.k, resid.k, nan]).ravel(),
            y=np.column_stack([resid.days, resid.days, nan]).ravel(),
            z=np.column_stack([100 * resid.iv, 100 * resid.model_iv, nan]).ravel(),
            mode="lines", line=dict(color=MUTED, width=2), name="quote − surface", hoverinfo="skip"))
    if color_by_error:
        bound = float(np.nanpercentile(np.abs(100 * resid.error), 98)) or 1.0
        marker = dict(color=100 * resid.error, colorscale=ERROR_SCALE, cmin=-bound, cmax=bound, size=3,
                      colorbar=dict(title="error<br>(vol pts)", len=0.6))
    else:
        marker = _day_marker(resid.days, resid.days, showscale=True, size=3)
    fig.add_trace(go.Scatter3d(x=resid.k, y=resid.days, z=100 * resid.iv, mode="markers", name="quotes",
                               marker=marker, customdata=customdata, hovertemplate=template))
    ranges = pd.DataFrame({"k": np.concatenate([resid.k, mesh["k"].ravel()]),
                           "days": np.concatenate([resid.days, mesh["days"].ravel()])})
    fig.update_layout(scene=_scene(ranges, "iv", log_days))
    return _style_3d(fig, f"{model} surface against the {len(resid):,} quotes it was fitted to", 700)


def residuals_by_model(resid: pd.DataFrame, models: list[str]) -> go.Figure:
    """Each quote's error against ``k / sqrt(T)``, one panel per model, coloured by days."""
    fig = make_subplots(rows=1, cols=len(models), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=models)
    for i, model in enumerate(models, start=1):
        part = resid[resid.model == model]
        customdata, template = _quote_hover(part)
        fig.add_trace(go.Scattergl(
            x=part.x, y=100 * part.error, mode="markers", name=model, showlegend=False,
            marker=_day_marker(part.days, resid.days, showscale=i == len(models), size=5,
                               symbol=np.where(part.option_type == "call", "diamond", "square")),
            customdata=customdata, hovertemplate=template), row=1, col=i)
        fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=1, col=i)
        fig.update_xaxes(title_text="k / sqrt(T)", row=1, col=i)
    fig.update_yaxes(title_text="model − quote (vol points)", row=1, col=1)
    return _style(fig, "Error per quote (call diamonds, put squares)", 420)


def smile_grid(resid: pd.DataFrame, curves: pd.DataFrame, expiries: list, models: list[str],
               cols: int = 4) -> go.Figure:
    """One panel per expiry: the quotes as points, and each model's smile as a line."""
    rows = -(-len(expiries) // cols)
    quotes = resid[resid.model == models[0]]
    titles = [f"{e:%Y-%m-%d} ({365 * quotes[quotes.expiry == e]['T'].iloc[0]:.0f}d)" for e in expiries]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.05,
                        vertical_spacing=0.32 / rows)
    for i, expiry in enumerate(expiries):
        r, c = i // cols + 1, i % cols + 1
        part = quotes[quotes.expiry == expiry]
        customdata, template = _quote_hover(part)
        for right, symbol in (("call", "diamond"), ("put", "square")):
            sel = (part.option_type == right).to_numpy()
            fig.add_trace(go.Scattergl(
                x=part.k[sel], y=100 * part.iv[sel], mode="markers", name=f"quote: {right}",
                legendgroup=right, showlegend=i == 0,
                marker=dict(color=POINT_COLOR, size=5, symbol=symbol, opacity=0.6),
                customdata=customdata[sel], hovertemplate=template), row=r, col=c)
        for model in models:
            line = curves[(curves.expiry == expiry) & (curves.model == model)]
            fig.add_trace(go.Scatter(
                x=line.k, y=100 * line.iv, mode="lines", name=model, legendgroup=model, showlegend=i == 0,
                line=dict(color=_model_color(model), width=2),
                hovertemplate=f"{model}: k=%{{x:.3f}}, iv %{{y:.2f}}%<extra></extra>"), row=r, col=c)
    fig.update_annotations(font_size=11)
    fig.update_xaxes(title_text="k", title_standoff=2)
    _style(fig, "Smiles: quotes and every model", 260 * rows + 90)
    fig.update_layout(legend=dict(y=-0.25 / rows))
    return fig


def atm_term_structure(lines: pd.DataFrame, quotes: pd.DataFrame) -> go.Figure:
    """ATM vol against days: the quotes' per expiry, and each model's between the expiries."""
    fig = go.Figure()
    for model, part in lines.groupby("model", sort=False):
        fig.add_trace(go.Scatter(x=part.days, y=100 * part.iv, mode="lines", name=model,
                                 line=dict(color=_model_color(model), width=2),
                                 hovertemplate=f"{model}: %{{x:.1f}} days, ATM iv %{{y:.2f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=quotes.days, y=100 * quotes.iv, mode="markers", name="quotes, at k = 0",
        marker=dict(color=POINT_COLOR, size=7, line=dict(color="white", width=1)),
        customdata=quotes.expiry.astype(str),
        hovertemplate="%{customdata}: %{x:.1f} days, ATM iv %{y:.2f}%<extra></extra>"))
    fig.update_xaxes(**_log_days_axis(lines.days))
    fig.update_yaxes(title_text="ATM implied vol (%)")
    return _style(fig, "ATM term structure, between the expiries too", 460)


def local_vol_heatmap(grid: dict, model: str) -> go.Figure:
    """Dupire's local vol over ``k / sqrt(T)`` and days; cells with negative local variance marked."""
    vol = 100 * grid["vol"]
    top = float(np.nanpercentile(vol, 98)) if np.isfinite(vol).any() else None  # spikes would wash out the rest
    fig = go.Figure(go.Heatmap(
        x=grid["x"], y=grid["days"], z=vol, colorscale="Viridis", zmin=0, zmax=top,
        colorbar=dict(title="local vol (%)"), zsmooth=False,
        hovertemplate="k/sqrt(T)=%{x:.2f}, %{y:.1f} days<br>local vol %{z:.2f}%<extra></extra>"))
    neg_t, neg_x = np.nonzero(grid["variance"] < 0)
    if len(neg_t):
        fig.add_trace(go.Scatter(
            x=grid["x"][neg_x], y=grid["days"][neg_t], mode="markers", name="negative local variance",
            marker=dict(symbol="x", color=ACCENT, size=6),
            customdata=grid["variance"][neg_t, neg_x],
            hovertemplate="k/sqrt(T)=%{x:.2f}, %{y:.1f} days<br>local variance %{customdata:.3g}<extra></extra>"))
    fig.update_xaxes(title_text="k / sqrt(T)")
    fig.update_yaxes(**_log_days_axis(pd.Series(grid["days"])))
    share = len(neg_t) / grid["variance"].size
    note = f", negative on {share:.0%} of the grid" if len(neg_t) else ", non-negative everywhere"
    return _style(fig, f"{model}: Dupire local vol{note}", 460)
