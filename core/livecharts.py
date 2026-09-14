"""Charts for watching a session happen, rather than reviewing one.

Different job from `core.charts`: those summarise a finished backtest, these
show a day in progress. So the emphasis is on *where you are right now* --
the last price large and legible, your entry marked, the stop and target as
lines you can see the price approaching, and the part of the session you are
allowed to trade shaded in.

Everything is drawn on the Eastern clock, matching the bar index.
"""
from __future__ import annotations

from dataclasses import dataclass

from datetime import datetime, time

import pandas as pd
import plotly.graph_objects as go

from core.theme import apply_layout, tokens


def session_chart(bars: pd.DataFrame, mode: str = "light",
                  entry: float | None = None, stop: float | None = None,
                  target: float | None = None, side: str = "buy",
                  window: tuple[time, time] | None = None,
                  fills: pd.DataFrame | None = None,
                  cycles: list[dict] | None = None,
                  height: int = 420) -> go.Figure:
    """Today's price, with your position drawn on it."""
    t = tokens(mode)
    fig = go.Figure()

    if bars is not None and not bars.empty:
        # Regular hours only. Pre-market bars stretch the axis across hours
        # with almost no trading in them and squeeze the actual session into
        # a corner.
        bars = bars.between_time("09:30", "15:59")
    if bars is None or bars.empty:
        apply_layout(fig, mode, "Waiting for bars", height)
        return fig

    # The tradeable window, shaded. Outside it the rule stands down, and
    # seeing that as a band explains a quiet chart better than a caption.
    if window:
        lo, hi = window
        day = bars.index[0].normalize()
        fig.add_vrect(
            x0=day + pd.Timedelta(hours=lo.hour, minutes=lo.minute),
            x1=day + pd.Timedelta(hours=hi.hour, minutes=hi.minute),
            fillcolor=t["good"], opacity=0.05, layer="below", line_width=0)

    fig.add_trace(go.Candlestick(
        x=bars.index, open=bars["open"], high=bars["high"],
        low=bars["low"], close=bars["close"], name="price",
        increasing_line_color=t["good"], decreasing_line_color=t["critical"],
        increasing_fillcolor=t["good"], decreasing_fillcolor=t["critical"],
        line_width=1, showlegend=False))

    last = float(bars["close"].iloc[-1])
    fig.add_hline(y=last, line_color=t["muted"], line_width=1,
                  line_dash="dot",
                  annotation_text=f"  {last:,.2f}",
                  annotation_position="right",
                  annotation_font_size=13)

    for value, label, colour in [
            (entry, "entry", t["text_primary"]),
            (stop, "stop", t["critical"]),
            (target, "target", t["good"])]:
        if value:
            fig.add_hline(y=value, line_color=colour, line_width=1.5,
                          line_dash="dash" if label != "entry" else "solid",
                          annotation_text=f"{label} {value:,.2f}",
                          annotation_position="left",
                          annotation_font_size=11,
                          annotation_font_color=colour)

    if fills is not None and not fills.empty and "filled_price" in fills:
        got = fills[fills["filled_price"].notna()]
        for direction, symbol_, colour in [("buy", "triangle-up", t["good"]),
                                           ("sell", "triangle-down", t["critical"])]:
            side_rows = got[got["side"] == direction]
            if side_rows.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(side_rows["ts"], errors="coerce", utc=True)
                  .dt.tz_convert("America/New_York").dt.tz_localize(None),
                y=side_rows["filled_price"], mode="markers",
                marker=dict(symbol=symbol_, size=13, color=colour,
                            line=dict(width=1, color=t["surface"])),
                name=direction, showlegend=False,
                hovertemplate=f"{direction} %{{y:.2f}}<extra></extra>"))

    # Each cycle as a tick along the bottom, so a quiet chart still shows the
    # loop was awake. Without these, "nothing happened" and "nothing ran" look
    # identical, and they are very different problems.
    if cycles:
        low = float(bars["low"].min())
        span = float(bars["high"].max()) - low
        for c in cycles:
            when = c.get("at")
            if when is None:
                continue
            colour = {"order": t["good"], "flatten": t["critical"],
                      "blocked": t["muted"]}.get(c.get("kind", ""), t["muted"])
            fig.add_trace(go.Scatter(
                x=[when], y=[low - span * 0.04], mode="markers",
                marker=dict(symbol="line-ns-open", size=9, color=colour,
                            line=dict(width=2, color=colour)),
                showlegend=False, hoverinfo="text",
                hovertext=f"{c.get('label', 'cycle')}"))

    apply_layout(fig, mode, "", height)
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="closest",
                      margin=dict(l=8, r=8, t=8, b=8))
    return fig


@dataclass(frozen=True)
class CycleKind:
    """One outcome a cycle can have, and how it is drawn and described.

    The strip's colours and the page's legend both read this. They used to be
    written out separately and had drifted: "watched, no setup" was labelled
    blue and drawn orange, and "told to wait" was labelled orange and drawn
    the same green as "placed an order" -- so the one block that means a trade
    happened looked like the one that means nothing did.
    """
    token: str        #: key into the theme palette
    swatch: str       #: the emoji in the legend, which has to match that colour
    words: str        #: what it means, in the legend and in the summary line


#: In the order they are listed to the reader.
CYCLE_KINDS = {
    "order":     CycleKind("good",     "🟩", "placed an order"),
    "flatten":   CycleKind("critical", "🟥", "closed out"),
    "watching":  CycleKind("series_1", "🟦", "watched, no setup"),
    "blocked":   CycleKind("muted",    "⬜",     "stood down (a rail blocked it)"),
    "throttled": CycleKind("pending",  "🟨", "told to wait (you clicked too soon)"),
}


def cycle_legend() -> str:
    """The caption under the strip, built from the same mapping it is drawn from."""
    return "Each block is one cycle:  " + "  ·  ".join(
        f"{v.swatch} {v.words}" for v in CYCLE_KINDS.values())


def cycle_strip(cycles: list[dict], mode: str = "light",
                height: int = 96) -> go.Figure:
    """Every cycle as a block on a timeline, newest on the right.

    Answers "has it been running, and what did it decide each time" at a
    glance. A list of identical log lines cannot: the eye reads a row of
    grey blocks with two green ones as a shape, and reads twenty sentences
    as a wall.
    """
    t = tokens(mode)
    fig = go.Figure()
    if not cycles:
        apply_layout(fig, mode, "No cycles yet", height)
        fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
        return fig

    palette = {k: t[v.token] for k, v in CYCLE_KINDS.items()}
    fig.add_trace(go.Bar(
        x=list(range(len(cycles))),
        y=[1] * len(cycles),
        marker_color=[palette.get(c.get("kind", "blocked"), t["muted"])
                      for c in cycles],
        marker_line_width=0, width=0.82,
        hovertext=[f"{c.get('time', '')}  {c.get('label', '')}"
                   for c in cycles],
        hoverinfo="text", showlegend=False))

    apply_layout(fig, mode, "", height)
    fig.update_layout(
        margin=dict(l=8, r=8, t=4, b=4), bargap=0.18, hovermode="closest",
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 1.1]))
    return fig


def pnl_chart(fills: pd.DataFrame, mode: str = "light",
              height: int = 160) -> go.Figure:
    """Cumulative realised money across the fills recorded today."""
    t = tokens(mode)
    fig = go.Figure()
    if fills is None or fills.empty:
        apply_layout(fig, mode, "No fills yet", height)
        return fig

    f = fills.copy()
    f["ts"] = pd.to_datetime(f["ts"], errors="coerce", utc=True)
    f = f.dropna(subset=["ts"]).sort_values("ts")
    if "slippage_bps" not in f or f["filled_qty"].fillna(0).sum() == 0:
        apply_layout(fig, mode, "No fills yet", height)
        return fig

    # Cost paid, cumulative. Not profit -- profit needs round trips matched
    # up, and showing a cost curve labelled as profit would be worse than
    # showing nothing.
    cost = (f["slippage_bps"].fillna(0) / 10_000.0 * f["filled_qty"].fillna(0)
            * f["filled_price"].fillna(0)).cumsum()
    fig.add_trace(go.Scatter(
        x=f["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None),
        y=-cost, mode="lines", line=dict(color=t["critical"], width=2),
        fill="tozeroy", name="cost", showlegend=False,
        hovertemplate="cost so far $%{y:.2f}<extra></extra>"))
    apply_layout(fig, mode, "", height)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), hovermode="closest",
                      yaxis_title=None, xaxis_title=None)
    return fig


def candidate_bars(candidates, mode: str = "light",
                   height: int = 300, top: int = 12) -> go.Figure:
    """Every tested rule as a bar, coloured by whether it is usable.

    The point of drawing this rather than tabulating it: the rules with huge
    averages and tiny samples stand out as obviously different, instead of
    sitting at the top of a table looking like winners.
    """
    t = tokens(mode)
    fig = go.Figure()
    if not candidates:
        apply_layout(fig, mode, "Nothing tested yet", height)
        return fig

    rows = sorted(candidates, key=lambda c: c.expectancy_pct)[-top:]
    fig.add_trace(go.Bar(
        x=[c.expectancy_pct for c in rows],
        y=[f"{c.strategy} · {c.symbol}" for c in rows],
        orientation="h",
        marker_color=[t["good"] if c.credible and c.expectancy_pct > 0
                      else (t["muted"] if c.credible else t["series_3"])
                      for c in rows],
        customdata=[[int(c.trades), c.t_stat] for c in rows],
        hovertemplate=("%{y}<br>%{x:.2f} bps a trade"
                       "<br>%{customdata[0]} trades, t=%{customdata[1]:.2f}"
                       "<extra></extra>"),
        showlegend=False))
    fig.add_vline(x=0, line_color=t["muted"], line_width=1)
    apply_layout(fig, mode, "", height)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), hovermode="closest",
                      xaxis_title="basis points per trade")
    return fig
