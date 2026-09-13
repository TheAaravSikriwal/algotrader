"""The reasoning loop: brief out, overlay back.

You are the middle of this loop on purpose. Generate a briefing here, paste it
into Claude Code, paste the response back, review what it would do to the book,
and record it. Being the rate limiter is a feature -- running five hundred
variations by hand is hard, and running five hundred variations is the main way
people destroy themselves at this.

Every overlay is journaled the moment it is recorded, before any outcome
exists. That timestamp is the whole basis for scoring the reasoning layer
later: its contribution is exactly `overlaid minus base`, measured on bars that
did not exist when the reasoning was written.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.briefing import build_briefing
from core.data import load_bars
from core.env import load_env
from core.gdelt import GdeltError, day_features
from core.news import NewsError, load_news, to_frame
from core.overlay import (DEFAULT_MAX_TILT, Overlay, OverlayError,
                          apply_overlay, ledger_frame, parse_overlay, record)
from core.panel import Panel, PanelConfig, run_panel_backtest
from core.precedent import (find_precedents, load_outcomes, score_pending,
                            summarise_precedents)
from core.taxonomy import classify
from core.ui import page_header, tile
from strategies.cross_sectional import available_xs, get_xs_strategy

load_env()

try:
    st.set_page_config(page_title="Reasoning", layout="wide")
except Exception:
    pass  # the host page already configured it

BUCKETS = {
    "macro": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLK", "XLF", "XLE",
              "XLV", "XLI", "XLY", "XLP", "XLU"],
    "company": ["NVDA", "GOOGL", "TSLA", "MSFT", "AMZN", "META", "AAPL", "MU",
                "AMD", "INTC", "AVGO", "CRM", "JPM", "NFLX"],
}


@st.cache_data(show_spinner="Fetching prices...")
def cached_bars(symbols: tuple, start: str, end: str) -> dict:
    out = {}
    for symbol in symbols:
        try:
            df = load_bars(symbol, start, end, "1Day", "yfinance")
            if len(df) > 60:
                out[symbol] = df
        except Exception:  # noqa: BLE001
            continue
    return out


@st.cache_data(show_spinner="Fetching recent news...")
def cached_news(symbols: tuple, days: int) -> pd.DataFrame:
    end = date.today()
    items = load_news(list(symbols), end - timedelta(days=days), end)
    return to_frame(items) if items else pd.DataFrame()


@st.cache_data(show_spinner="Reading world events...")
def cached_gdelt(day: str) -> dict:
    return day_features(pd.Timestamp(day).date(), countries=("USA", "RUS", "UKR",
                                                             "ISR", "CHN"))


def base_weights(bucket: str, symbols: tuple, strategy: str) -> pd.Series:
    """Today's target book from the deterministic base strategy."""
    end = date.today()
    bars = cached_bars(symbols, str(end - timedelta(days=365 * 3)), str(end))
    if len(bars) < 3:
        return pd.Series(dtype=float)
    panel = Panel.from_bars(bars)
    weights = get_xs_strategy(strategy)().generate_weights(panel)
    return weights.iloc[-1]


def main():
    mode = page_header(
        "Reasoning layer",
        "Brief out, overlay back. The base strategy stays fixed; the overlay "
        "only tilts it, so its contribution stays measurable.")

    with st.sidebar:
        st.markdown("### Reasoning")
        bucket = st.radio("Bucket", ["macro", "company"], horizontal=True)
        raw = st.text_area("Universe", ",".join(BUCKETS[bucket]), height=90)
        symbols = tuple(s.strip().upper() for s in raw.split(",") if s.strip())

        names = available_xs()
        default = "Equal weight all"
        strategy = st.selectbox("Base strategy", names,
                                index=names.index(default) if default in names else 0)
        since_days = st.slider("Look back (days)", 1, 30, 7)
        max_tilt = st.slider("Max total tilt", 0.05, 0.60, DEFAULT_MAX_TILT, 0.05,
                             help="Caps what one overlay can move. A confident "
                                  "wrong call should cost a slice, not the book.")
        include_gdelt = st.checkbox("Include world-event context", True)

    if len(symbols) < 3:
        st.error("Need at least three symbols to form a book.")
        return

    book = base_weights(bucket, symbols, strategy)
    if book.empty:
        st.error("Could not build a base book from those symbols.")
        return

    tab_brief, tab_apply, tab_ledger = st.tabs(
        ["1 · Generate briefing", "2 · Apply overlay", "3 · Ledger"])

    # -- 1. the briefing --------------------------------------------------
    with tab_brief:
        try:
            articles = cached_news(symbols, since_days)
        except NewsError as exc:
            st.error(f"News unavailable: {exc}")
            articles = pd.DataFrame()

        context = {}
        if include_gdelt:
            try:
                g = cached_gdelt(str(date.today() - timedelta(days=2)))
                context = {k: v for k, v in g.items()
                           if k in ("date", "gdelt_tone", "gdelt_goldstein",
                                    "gdelt_material_conflict", "gdelt_fight",
                                    "gdelt_conflict_rus", "gdelt_conflict_ukr",
                                    "gdelt_conflict_isr")}
            except GdeltError as exc:
                st.caption(f"World-event context unavailable: {exc}")

        # Positions expire; what they taught does not. Pull scored outcomes
        # from events of this shape so the reasoning sees its own track record.
        categories = sorted({c for text_ in
                             articles.get("text", pd.Series(dtype=str)).astype(str)
                             for c in (classify(text_)["macro"]
                                       + classify(text_)["company"])})
        precedents = find_precedents(categories, bucket=bucket) if categories \
            else pd.DataFrame()
        stats = summarise_precedents(precedents)
        if not precedents.empty:
            context["_precedents"] = precedents

        briefing = build_briefing(articles, book, bucket=bucket,
                                  symbols=list(symbols), since_days=since_days,
                                  context=context)
        text = briefing.to_markdown()

        cols = st.columns(5)
        tile(cols[0], "Events", f"{len(briefing.articles)}",
             f"last {since_days} days")
        tile(cols[1], "Book", f"{int((book.abs() > 1e-6).sum())}", "positions held")
        tile(cols[2], "Precedents", f"{stats['n']}", "scored outcomes on record",
             "up" if stats["n"] >= 5 else "flat")
        tile(cols[3], "Hit rate",
             f"{stats.get('hit_rate', 0):.0%}" if stats["n"] else "—",
             "past calls that went the right way")
        tile(cols[4], "Packet", f"{len(text):,}", "characters")

        if stats["n"]:
            st.caption(stats["note"])

        if briefing.articles.empty:
            st.info("No qualifying events. Issuing no adjustments is the "
                    "correct response to a quiet week — and the most common one.")

        st.caption("Copy this into Claude Code. Market-reaction headlines are "
                   "filtered out: pasting *'energy stocks rallied 4%'* and asking "
                   "what happens next is asking a model to read, not forecast.")
        st.code(text, language="markdown")
        st.download_button("Download briefing", text.encode(),
                           file_name=f"briefing_{bucket}_{date.today()}.md")

    # -- 2. apply what came back ------------------------------------------
    with tab_apply:
        st.markdown("Paste the json response below.")
        payload = st.text_area("Overlay json", height=240,
                               placeholder='{"schema": "algotrader.overlay.v1", ...}')

        if not payload.strip():
            st.caption("Nothing pasted yet.")
        else:
            try:
                overlay = parse_overlay(payload)
            except OverlayError as exc:
                st.error(f"Could not read that overlay: {exc}")
                overlay = None

            if overlay is not None:
                adjusted, report = apply_overlay(book, overlay, max_tilt=max_tilt)
                st.success(f"Overlay {overlay.id} — {len(report['applied'])} "
                           f"adjustment(s) applied")
                st.caption(overlay.rationale)

                if report["unknown_symbols"]:
                    st.warning("Not in the book, so ignored: "
                               + ", ".join(report["unknown_symbols"]))
                if report["clipped"]:
                    st.warning("Clipped to the per-name limit: "
                               + ", ".join(report["clipped"]))
                if report["scaled_by"] < 1.0:
                    st.warning(f"Total tilt exceeded the cap and was scaled to "
                               f"{report['scaled_by']:.0%}.")
                if report["expired"]:
                    st.info(f"{report['expired']} adjustment(s) already expired.")

                comparison = pd.DataFrame({
                    "base": book, "overlaid": adjusted,
                    "change": adjusted - book}).sort_values(
                    "change", key=abs, ascending=False)
                comparison = comparison[comparison.abs().sum(axis=1) > 1e-9]

                left, right = st.columns([2, 3])
                with left:
                    st.metric("Turnover this implies", f"{report['turnover']:.1%}")
                    st.metric("Gross exposure",
                              f"{report['gross_after']:.1%}",
                              delta=f"{report['gross_after'] - report['gross_before']:+.2%}")
                with right:
                    st.dataframe(
                        comparison, use_container_width=True, height=260,
                        column_config={
                            "base": st.column_config.NumberColumn("Base", format="%.2f%%"),
                            "overlaid": st.column_config.NumberColumn("Overlaid", format="%.2f%%"),
                            "change": st.column_config.NumberColumn("Change", format="%.2f%%"),
                        })

                st.markdown("**Reasons given**")
                st.dataframe(pd.DataFrame(report["applied"]), hide_index=True,
                             use_container_width=True)

                if st.button("Record this overlay", type="primary"):
                    record(overlay)
                    st.success(f"Recorded {overlay.id}. It is now timestamped "
                               "before any outcome exists, which is what makes "
                               "it scoreable later.")

    # -- 3. what has been issued ------------------------------------------
    with tab_ledger:
        frame = ledger_frame()
        if frame.empty:
            st.info("No overlays recorded yet.")
        else:
            active = int(frame["active"].sum())
            cols = st.columns(3)
            tile(cols[0], "Adjustments", f"{len(frame)}", "all time")
            tile(cols[1], "Active", f"{active}", "not yet expired",
                 "up" if active else "flat")
            tile(cols[2], "Overlays", f"{frame['overlay'].nunique()}", "issued")

            st.write("")
            st.dataframe(
                frame.sort_values("issued", ascending=False), hide_index=True,
                use_container_width=True, height=420,
                column_config={
                    "weight": st.column_config.NumberColumn("Weight", format="%.3f"),
                    "confidence": st.column_config.NumberColumn("Conf", format="%.2f"),
                    "effective": st.column_config.NumberColumn("Effective", format="%.3f"),
                    "active": st.column_config.CheckboxColumn("Active"),
                    "reason": st.column_config.TextColumn("Reason", width="large"),
                })
            st.download_button("Download ledger (CSV)",
                               frame.to_csv(index=False).encode(),
                               file_name="overlay_ledger.csv", mime="text/csv")

        st.divider()
        st.markdown("#### Outcomes")
        st.caption(
            "Positions expire; what they taught does not. Once an adjustment "
            "passes its expiry, the market's actual move is measured against "
            "the benchmark and kept as a precedent — available to every future "
            "briefing on an event of the same shape.")

        outcomes = load_outcomes()
        if st.button("Score expired adjustments"):
            end = date.today()
            from core.overlay import load_overlays
            issued = {a.symbol for o in load_overlays() for a in o.adjustments}
            all_symbols = tuple(sorted(issued | set(symbols)))
            bars = cached_bars(all_symbols, str(end - timedelta(days=365 * 2)),
                               str(end))
            bench_bars = cached_bars(("SPY",), str(end - timedelta(days=365 * 2)),
                                     str(end))
            benchmark = bench_bars["SPY"]["close"] if "SPY" in bench_bars else None
            fresh = score_pending(bars, benchmark)
            if fresh:
                st.success(f"Scored {len(fresh)} newly expired adjustment(s).")
            else:
                st.info("Nothing new to score — every expired adjustment already "
                        "has an outcome, or none have expired yet.")
            outcomes = load_outcomes()

        if outcomes:
            frame = find_precedents()
            summary = summarise_precedents(frame)
            cols = st.columns(4)
            tile(cols[0], "Scored", f"{summary['n']}", "outcomes on record")
            tile(cols[1], "Hit rate", f"{summary.get('hit_rate', 0):.0%}",
                 "right direction")
            tile(cols[2], "Mean contribution",
                 f"{summary.get('mean_contribution_%', 0):+.2f}%", "per call",
                 money_kind="once")
            tile(cols[3], "t", f"{summary.get('tstat', 0):+.2f}",
                 "vs zero",
                 "up" if abs(summary.get("tstat", 0)) >= 2 else "flat")
            st.caption(summary["note"])
            st.dataframe(frame, hide_index=True, use_container_width=True,
                         height=300)
        else:
            st.info("No outcomes scored yet. They appear once issued "
                    "adjustments pass their expiry.")


if __name__ == "__main__":
    main()
