"""Plotly figures for the dashboard.

House rules applied throughout: one y-axis per chart, 2px lines, recessive
grid and axes, a legend whenever two or more series share a chart plus direct
end-of-line labels so identity never rests on colour alone, and a crosshair
tooltip on every time series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import apply_layout, tokens

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _end_point(series: pd.Series):
    s = series.dropna()
    return (s.index[-1], float(s.iloc[-1])) if not s.empty else None


def _place_labels(fig, specs: list[tuple], y_span: float, size: int = 12,
                  min_frac: float = 0.09):
    """Direct labels at the end of each line, nudged apart when they'd overlap.

    Lines that converge -- two moving averages at a crossover, a strategy that
    tracks its benchmark -- would otherwise stack their labels on top of each
    other and become unreadable.
    """
    if not specs:
        return
    gap = abs(y_span) * min_frac
    last = None
    for x, y, text, color in sorted(specs, key=lambda s: s[1]):
        if last is not None and y - last < gap:
            y = last + gap
        last = y
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left",
                           xshift=8, yanchor="middle", font=dict(size=size, color=color))


def equity_chart(result, mode: str = "light", height: int = 400) -> go.Figure:
    t = tokens(mode)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.benchmark.index, y=result.benchmark.values, name="Buy & hold",
        mode="lines", line=dict(color=t["series_2"], width=2, dash="dash"),
        hovertemplate="Buy & hold  $%{y:,.0f}<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=result.equity.index, y=result.equity.values, name="Strategy",
        mode="lines", line=dict(color=t["series_1"], width=2),
        hovertemplate="Strategy  $%{y:,.0f}<extra></extra>"))

    both = pd.concat([result.equity, result.benchmark])
    specs = [(*pt, label, color) for pt, label, color in (
        (_end_point(result.benchmark), "Buy & hold", t["series_2"]),
        (_end_point(result.equity), "Strategy", t["series_1"]),
    ) if pt]
    _place_labels(fig, specs, float(both.max() - both.min()))

    apply_layout(fig, mode, "Equity curve", height, legend=True)
    fig.update_layout(margin=dict(l=8, r=96, t=72, b=8))
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def drawdown_chart(result, mode: str = "light", height: int = 240) -> go.Figure:
    """One series -- the title names it, so no legend box."""
    t = tokens(mode)
    dd = result.drawdown * 100

    r, g, b = (int(t["drawdown"][i:i + 2], 16) for i in (1, 3, 5))
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", fill="tozeroy",
        line=dict(color=t["drawdown"], width=2), fillcolor=f"rgba({r},{g},{b},0.16)",
        hovertemplate="Drawdown  %{y:.2f}%<extra></extra>", showlegend=False))

    worst = dd.idxmin() if not dd.empty else None
    if worst is not None and dd.min() < -0.01:
        fig.add_annotation(x=worst, y=dd.min(), text=f"{dd.min():.1f}%", showarrow=False,
                           yshift=-14, font=dict(size=12, color=t["text_secondary"]))

    apply_layout(fig, mode, "Drawdown from peak", height)
    fig.update_yaxes(ticksuffix="%")
    return fig


def price_chart(result, indicators: dict[str, pd.Series] | None = None,
                mode: str = "light", height: int = 420) -> go.Figure:
    """Price with indicator overlays and entry/exit markers.

    Marker shape carries the meaning (up = entry, down = exit), so colour is a
    reinforcement rather than the only channel.
    """
    t = tokens(mode)
    price = result.price
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=price.index, y=price.values, name="Close", mode="lines",
        line=dict(color=t["text_secondary"], width=2),
        hovertemplate="Close  $%{y:,.2f}<extra></extra>"))

    slots = [t["series_1"], t["series_2"], t["series_3"]]
    specs, span_lo, span_hi = [], float(price.min()), float(price.max())
    for i, (label, series) in enumerate((indicators or {}).items()):
        color = slots[i % len(slots)]
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=label, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"{label}  $%{{y:,.2f}}<extra></extra>"))
        point = _end_point(series)
        if point:
            specs.append((*point, label, color))
            span_lo, span_hi = min(span_lo, point[1]), max(span_hi, point[1])
    _place_labels(fig, specs, span_hi - span_lo, size=11)

    trades = result.trades
    if trades is not None and not trades.empty:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]

        fig.add_trace(go.Scatter(
            x=trades["entry_time"], y=trades["entry_price"], name="Entry", mode="markers",
            marker=dict(symbol="triangle-up", size=11, color=t["series_1"],
                        line=dict(color=t["surface"], width=2)),
            hovertemplate="Entry  $%{y:,.2f}<extra></extra>"))

        for name, subset, symbol, color in (
            ("Exit -- profit", wins, "triangle-down", t["good"]),
            ("Exit -- loss", losses, "x", t["critical"]),
        ):
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset["exit_time"], y=subset["exit_price"], name=name, mode="markers",
                marker=dict(symbol=symbol, size=11, color=color,
                            line=dict(color=t["surface"], width=2)),
                customdata=subset[["pnl", "return_pct"]].to_numpy(),
                hovertemplate=(f"{name}  $%{{y:,.2f}}<br>"
                               "P&L $%{customdata[0]:,.2f} (%{customdata[1]:.2f}%)<extra></extra>")))

    apply_layout(fig, mode, "Price, signals and fills", height, legend=True)
    fig.update_layout(margin=dict(l=8, r=110, t=72, b=8), hovermode="x unified")
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def exposure_chart(result, mode: str = "light", height: int = 150) -> go.Figure:
    t = tokens(mode)
    exp = result.exposure
    fig = go.Figure(go.Scatter(
        x=exp.index, y=exp.values, mode="lines", line=dict(color=t["series_1"], width=2,
                                                           shape="hv"),
        fill="tozeroy", hovertemplate="Position  %{y:.0%}<extra></extra>", showlegend=False))
    r, g, b = (int(t["series_1"][i:i + 2], 16) for i in (1, 3, 5))
    fig.update_traces(fillcolor=f"rgba({r},{g},{b},0.16)")

    apply_layout(fig, mode, "Position held", height)
    fig.update_yaxes(tickformat=".0%", range=[min(-0.1, exp.min() * 1.1), max(1.1, exp.max() * 1.1)])
    return fig


def monthly_heatmap(grid: pd.DataFrame, mode: str = "light", height: int = 300) -> go.Figure:
    """Diverging blue/red around a neutral grey -- losses and gains are opposites."""
    t = tokens(mode)
    if grid.empty:
        return apply_layout(go.Figure(), mode, "Monthly returns", height)

    z = grid.to_numpy(dtype=float) * 100
    limit = float(np.nanmax(np.abs(z))) or 1.0

    fig = go.Figure(go.Heatmap(
        z=z, x=[MONTHS[m - 1] for m in grid.columns], y=[str(y) for y in grid.index],
        colorscale=[[0.0, t["div_neg"]], [0.5, t["div_mid"]], [1.0, t["div_pos"]]],
        zmid=0, zmin=-limit, zmax=limit, xgap=2, ygap=2,
        colorbar=dict(ticksuffix="%", outlinewidth=0, thickness=10,
                      tickfont=dict(color=t["muted"], size=11)),
        hovertemplate="%{y} %{x}  %{z:.2f}%<extra></extra>"))

    apply_layout(fig, mode, "Monthly returns", height)
    # month labels sit above the grid, so the title needs room of its own
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=8, t=82, b=8))
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def trade_return_hist(trades: pd.DataFrame, mode: str = "light", height: int = 260) -> go.Figure:
    t = tokens(mode)
    fig = go.Figure()
    if trades is None or trades.empty:
        return apply_layout(fig, mode, "Trade returns", height)

    fig.add_trace(go.Histogram(
        x=trades["return_pct"], nbinsx=min(40, max(8, len(trades) // 2)),
        marker=dict(color=t["series_1"], line=dict(color=t["surface"], width=2)),
        hovertemplate="%{x:.1f}%  ·  %{y} trades<extra></extra>", showlegend=False))

    fig.add_vline(x=0, line=dict(color=t["axis"], width=1, dash="dot"))
    apply_layout(fig, mode, "Distribution of trade returns", height)
    fig.update_layout(hovermode="closest", bargap=0.04)
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(title=None)
    return fig
