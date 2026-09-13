"""Event studies in the browser.

One company's history of a rare event is an anecdote. The same event pooled
across a hundred companies is evidence. This page is where that conversion
happens -- and where it tells you when the pool is still too small.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data import load_bars
from core.env import load_env
from core.eventstudy import (EventWindow, events_from_condition,
                             events_from_news, interpret, run_event_study)
from core.news import NewsError, load_news, to_frame
from core.sentiment import EVENT_PATTERNS, get_scorer, score_frame
from core.theme import apply_layout, tokens
from core.ui import page_header, pct, tile

load_env()

st.set_page_config(page_title="Event study", layout="wide")

UNIVERSES = {
    "megacap": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
                "TSLA", "JPM", "V", "WMT", "XOM"],
    "sp30": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
             "JPM", "V", "WMT", "XOM", "JNJ", "PG", "MA", "HD", "CVX", "MRK",
             "ABBV", "KO", "PEP", "COST", "ADBE", "CRM", "NFLX", "AMD", "INTC",
             "CSCO", "QCOM", "TXN"],
    "banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "SCHW"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HAL"],
}

CONDITIONS = {
    "gap down 3%": lambda df: df["open"] / df["close"].shift(1) - 1 < -0.03,
    "gap up 3%": lambda df: df["open"] / df["close"].shift(1) - 1 > 0.03,
    "new 52-week high": lambda df: df["close"] >= df["close"].rolling(252).max(),
    "new 52-week low": lambda df: df["close"] <= df["close"].rolling(252).min(),
    "single-day drop 5%": lambda df: df["close"].pct_change() < -0.05,
    "volume spike 3x": lambda df: df["volume"] > 3 * df["volume"].rolling(50).mean(),
}


@st.cache_data(show_spinner="Fetching prices...")
def cached_bars(symbols: tuple, start: str, end: str, source: str) -> dict:
    out = {}
    for symbol in symbols:
        try:
            df = load_bars(symbol, start, end, "1Day", source)
            if len(df) > 60:
                out[symbol] = df
        except Exception:  # noqa: BLE001
            continue
    return out


@st.cache_data(show_spinner="Fetching and scoring news...")
def cached_scored_news(symbols: tuple, start: str, end: str, scorer: str):
    items = load_news(list(symbols), start, end)
    if not items:
        return pd.DataFrame()
    return score_frame(to_frame(items), get_scorer(scorer))


def car_chart(daily: pd.DataFrame, mode: str) -> go.Figure:
    t = tokens(mode)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily["day"], y=daily["mean_car_%"], mode="lines",
        line=dict(color=t["series_1"], width=2), name="Mean CAR",
        hovertemplate="day %{x}  ·  CAR %{y:.3f}%<extra></extra>"))

    fig.add_vline(x=0, line=dict(color=t["critical"], width=2, dash="dash"))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    fig.add_annotation(x=0, y=daily["mean_car_%"].max(), text="event",
                       showarrow=False, xshift=26,
                       font=dict(size=12, color=t["critical"]))

    apply_layout(fig, mode, "Cumulative abnormal return around the event", 400)
    fig.update_xaxes(title="trading days relative to the event")
    fig.update_yaxes(ticksuffix="%")
    return fig


def sidebar():
    cfg = {}
    with st.sidebar:
        st.markdown("### Event study")
        preset = st.selectbox("Universe", list(UNIVERSES) + ["custom"], index=1)
        default = ",".join(UNIVERSES.get(preset, UNIVERSES["sp30"]))
        raw = st.text_area("Symbols", default, height=90)
        cfg["symbols"] = tuple(s.strip().upper() for s in raw.split(",") if s.strip())

        today = date.today()
        c1, c2 = st.columns(2)
        cfg["start"] = c1.date_input("From", today - timedelta(days=365 * 4))
        cfg["end"] = c2.date_input("To", today, max_value=today)
        cfg["source"] = st.selectbox("Price source", ["yfinance", "alpaca"])

        st.divider()
        st.markdown("**What counts as an event**")
        cfg["kind"] = st.radio("Trigger", ["Price condition", "News"],
                               horizontal=True)
        if cfg["kind"] == "Price condition":
            cfg["condition"] = st.selectbox("Condition", list(CONDITIONS))
        else:
            cfg["event_type"] = st.selectbox(
                "Event category", ["(any)"] + sorted(EVENT_PATTERNS))
            cfg["keyword"] = st.text_input(
                "Keyword", "", help="Literal phrase, e.g. ad campaign")
            cfg["min_sentiment"] = st.slider("Minimum sentiment", -1.0, 1.0, -1.0, 0.05)
            cfg["scorer"] = st.selectbox("Scorer", ["lexicon", "finbert"])

        st.divider()
        st.markdown("**Window**")
        cfg["pre"] = st.slider("Bars before", 0, 20, 5)
        cfg["post"] = st.slider("Bars after", 1, 60, 20)
        cfg["estimation"] = st.slider("Estimation window", 60, 400, 200, 10)
        cfg["gap"] = st.slider("Gap before event", 0, 30, 10,
                               help="Keeps the estimation window clear of the "
                                    "event, so the model is not fitted on data "
                                    "the event already moved.")
        cfg["model"] = st.selectbox("Model", ["market", "market_adjusted", "raw"])
        cfg["benchmark"] = st.text_input("Benchmark", "SPY")
    return cfg


def main():
    mode = page_header(
        "Event study",
        "What happens after an event, pooled across companies rather than "
        "read off one.")
    cfg = sidebar()

    if len(cfg["symbols"]) < 2:
        st.error("Pool across at least two symbols — more is the whole point.")
        return

    bars = cached_bars(cfg["symbols"], str(cfg["start"]), str(cfg["end"]),
                       cfg["source"])
    if len(bars) < 2:
        st.error("Not enough symbols returned price history.")
        return

    benchmark = None
    try:
        benchmark = load_bars(cfg["benchmark"], str(cfg["start"]),
                              str(cfg["end"]), "1Day", cfg["source"])["close"]
    except Exception:  # noqa: BLE001
        st.warning(f"{cfg['benchmark']} unavailable — returns will not be "
                   "benchmark-adjusted.")

    if cfg["kind"] == "Price condition":
        events = events_from_condition(bars, CONDITIONS[cfg["condition"]])
        label = cfg["condition"]
    else:
        try:
            scored = cached_scored_news(cfg["symbols"], str(cfg["start"]),
                                        str(cfg["end"]), cfg["scorer"])
        except NewsError as exc:
            st.error(f"News unavailable: {exc}")
            return
        if scored.empty:
            st.warning("No news returned for that universe and window.")
            return
        events = events_from_news(
            scored,
            event_type=None if cfg["event_type"] == "(any)" else cfg["event_type"],
            keyword=cfg["keyword"] or None,
            min_sentiment=None if cfg["min_sentiment"] <= -1.0 else cfg["min_sentiment"])
        label = cfg["keyword"] or cfg["event_type"]

    if events.empty:
        st.warning("No events matched. Loosen the filter or widen the window.")
        return

    st.caption(f"**{label}** · {len(events):,} candidate events across "
               f"{events['symbol'].nunique()} symbols")

    try:
        result = run_event_study(
            events, bars, benchmark,
            EventWindow(pre=cfg["pre"], post=cfg["post"],
                        estimation=cfg["estimation"], gap=cfg["gap"]),
            cfg["model"])
    except ValueError as exc:
        st.error(f"Could not run the study: {exc}")
        return

    s = result.summary
    cols = st.columns(6)
    tile(cols[0], "Usable events", f"{s['events']:,}",
         f"{s['symbols']} symbols",
         "up" if s["events"] >= 30 else "flat")
    tile(cols[1], "Event day", f"{s['event_day_%']:+.2f}%",
         f"t = {s['t_event_day']:+.2f}", money_kind="once")
    for i, day in enumerate((5, 10, 20)):
        key = f"car_{day}d_%"
        if key in s:
            t_val = s[f"t_{day}d"]
            tile(cols[2 + i], f"{day}-day drift", f"{s[key]:+.2f}%",
                 f"t = {t_val:+.2f}",
                 "up" if t_val >= 2 else ("down" if t_val <= -2 else "flat"),
                 money_kind="once")
    tile(cols[5], "Clustering", f"{s['clustering']:.0%}",
         "share on one date",
         "down" if s["clustering"] > 0.2 else "flat")

    if cfg["kind"] == "Price condition":
        st.info("The event-day figure restates the filter — a gap-down screen "
                "selects on exactly that move. Only the drift **after** it is "
                "tradeable, which is why the two are reported separately.")

    st.write("")
    st.plotly_chart(car_chart(result.daily, mode), use_container_width=True)

    for note in interpret(result):
        st.warning(note) if ("Only" in note or "calendar" in note) else st.info(note)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Day by day**")
        st.dataframe(result.daily, hide_index=True, use_container_width=True,
                     height=320,
                     column_config={
                         "day": st.column_config.NumberColumn("Day"),
                         "mean_ar_%": st.column_config.NumberColumn("Mean AR", format="%.3f%%"),
                         "mean_car_%": st.column_config.NumberColumn("Mean CAR", format="%.3f%%"),
                         "positive_share": st.column_config.NumberColumn("Positive", format="%.0f%%"),
                         "t_ar": st.column_config.NumberColumn("t (AR)", format="%.2f"),
                         "t_car": st.column_config.NumberColumn("t (CAR)", format="%.2f"),
                     })
    with right:
        st.markdown("**Events used**")
        st.dataframe(result.events[["symbol", "event_date", "car", "beta"]],
                     hide_index=True, use_container_width=True, height=320,
                     column_config={
                         "event_date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                         "car": st.column_config.NumberColumn("CAR", format="%.4f"),
                         "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
                     })
        st.download_button("Download events (CSV)",
                           result.events.to_csv(index=False).encode(),
                           file_name="events.csv", mime="text/csv")

    st.caption(
        "Abnormal returns are measured against the benchmark, so a rising "
        "market is already accounted for. Trading costs are not — an edge under "
        "roughly 0.3% does not survive crossing the spread twice.")


if __name__ == "__main__":
    main()
