"""Backtesting dashboard.

Run it with:   streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import charts
from core.bundle import BundleError, build_bundle, describe, load_bundle
from core.data import DataError, bars_per_year, clear_cache, load_bars
from core.engine import BacktestConfig, run_backtest
from core.metrics import monthly_returns, summarise
from core.strategy import available, get_strategy
from core.ui import (active_mode, inject_css, param_control, pct, tile,
                     tone_of)

from core.env import load_env

load_env()
import strategies  # noqa: F401  -- registers the built-ins

st.set_page_config(page_title="Backtester", page_icon="~", layout="wide")

TIMEFRAMES = ["1Day", "1Hour", "15Min", "5Min", "1Min"]


@st.cache_data(show_spinner="Fetching bars...")
def cached_bars(symbol, start, end, timeframe, source):
    return load_bars(symbol, start, end, timeframe, source)


@st.cache_data(show_spinner="Fetching and scoring news...")
def cached_news_features(symbol, start, end, scorer_name, _bars):
    """News features aligned to `_bars`. Underscore keeps the frame out of the key."""
    from core.news import load_news, to_frame
    from core.newsfeatures import build_news_features
    from core.sentiment import get_scorer, score_frame

    items = load_news([symbol], start, end)
    if not items:
        return None
    scored = score_frame(to_frame(items), get_scorer(scorer_name))
    return build_news_features(scored, _bars, symbol)


def apply_bundle(bundle: dict):
    """Push a loaded bundle into widget state before the widgets are built."""
    run, strat = bundle["run"], bundle["strategy"]
    ss = st.session_state
    ss["src"] = run.get("source", "yfinance")
    ss["sym"] = run["symbol"]
    ss["tf"] = run.get("timeframe", "1Day")
    ss["start"] = pd.to_datetime(run["start"]).date()
    ss["end"] = pd.to_datetime(run["end"]).date()
    ss["rule"] = strat["name"]
    for name, value in strat.get("params", {}).items():
        ss[f"p_{strat['name']}_{name}"] = value

    ex = bundle.get("execution", {})
    ss["cash"] = int(ex.get("initial_cash", 10_000))
    ss["psize"] = float(ex.get("position_size", 1.0))
    ss["slip"] = float(ex.get("slippage_bps", 5.0))
    ss["comm"] = float(ex.get("commission_pct", 0.0)) * 100
    ss["short"] = bool(ex.get("allow_short", False))
    ss["stop"] = float(ex.get("stop_loss_pct", 0.0))
    ss["tp"] = float(ex.get("take_profit_pct", 0.0))


def sidebar():
    cfg = {}
    with st.sidebar:
        st.markdown("### Backtester")

        with st.expander("Import a saved run"):
            upload = st.file_uploader("Bundle (.json)", type="json",
                                      label_visibility="collapsed")
            if upload is not None and st.button("Load it", use_container_width=True):
                try:
                    bundle = load_bundle(upload)
                    apply_bundle(bundle)
                    st.session_state["imported"] = describe(bundle)
                    st.rerun()
                except BundleError as exc:
                    st.error(str(exc))
            if st.session_state.get("imported"):
                st.caption(f"Loaded: {st.session_state['imported']}")

        st.markdown("**Market data**")
        cfg["source"] = st.selectbox(
            "Source", ["yfinance", "alpaca"], key="src",
            help="yfinance needs no key. alpaca reads ALPACA_API_KEY_ID / "
                 "ALPACA_API_SECRET_KEY from the environment.")
        cfg["symbol"] = st.text_input("Symbol", "SPY", key="sym").strip().upper()
        cfg["timeframe"] = st.selectbox("Timeframe", TIMEFRAMES, index=0, key="tf")

        today = date.today()
        default_start = today - timedelta(days=365 * 5)
        c1, c2 = st.columns(2)
        cfg["start"] = c1.date_input("From", default_start, max_value=today, key="start")
        cfg["end"] = c2.date_input("To", today, max_value=today, key="end")

        st.divider()
        st.markdown("**Strategy**")
        names = available()
        cfg["strategy"] = st.selectbox(
            "Rule", names, key="rule",
            index=names.index("SMA crossover") if "SMA crossover" in names else 0)
        cls = get_strategy(cfg["strategy"])
        if cls.description:
            st.caption(cls.description)
        cfg["scorer"] = "lexicon"
        if cls.requires_features:
            cfg["scorer"] = st.selectbox(
                "News scorer", ["lexicon", "finbert"], key="scorer",
                help="lexicon needs nothing. finbert is more accurate and needs "
                     "transformers + torch installed.")
            st.caption("Needs Alpaca keys — news is fetched and scored on demand.")
        cfg["params"] = {p.name: param_control(p, f"p_{cfg['strategy']}_{p.name}")
                         for p in cls.params}

        st.divider()
        st.markdown("**Execution**")
        cfg["cash"] = st.number_input("Starting cash ($)", 100, 10_000_000, 10_000, 500,
                                      key="cash")
        cfg["position_size"] = st.slider("Equity per trade", 0.05, 1.0, 1.0, 0.05, key="psize",
                                         help="Fraction of account equity committed to a position")
        cfg["slippage"] = st.slider("Slippage (bps per side)", 0.0, 50.0, 5.0, 0.5, key="slip",
                                    help="1 bp = 0.01%. Alpaca charges no commission, but you "
                                         "still pay the spread.")
        cfg["commission_pct"] = st.number_input("Commission (% of notional)", 0.0, 1.0, 0.0, 0.01,
                                                format="%.3f", key="comm") / 100.0
        cfg["allow_short"] = st.checkbox("Allow short positions", False, key="short")

        st.markdown("**Risk exits** — 0 disables")
        c3, c4 = st.columns(2)
        cfg["stop_loss"] = c3.number_input("Stop loss %", 0.0, 90.0, 0.0, 0.5, key="stop")
        cfg["take_profit"] = c4.number_input("Take profit %", 0.0, 500.0, 0.0, 0.5, key="tp")

        st.divider()
        if st.button("Clear data cache", use_container_width=True):
            n = clear_cache()
            cached_bars.clear()
            st.success(f"Removed {n} cached file(s)")
    return cfg


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    mode = active_mode()
    inject_css(mode)
    cfg = sidebar()

    st.markdown(f"## {cfg['symbol']} — {cfg['strategy']}")

    if cfg["start"] >= cfg["end"]:
        st.error("The start date has to come before the end date.")
        return

    try:
        df = cached_bars(cfg["symbol"], str(cfg["start"]), str(cfg["end"]),
                         cfg["timeframe"], cfg["source"])
    except DataError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 -- surface anything the data layer throws
        st.error(f"Could not load data: {exc}")
        return

    if len(df) < 30:
        st.warning(f"Only {len(df)} bars returned — too few to say anything meaningful.")
        if df.empty:
            return

    strategy = get_strategy(cfg["strategy"])(**cfg["params"])

    # news strategies need extra columns; fetch and join them on demand
    if not strategy.can_run_on(df):
        try:
            features = cached_news_features(cfg["symbol"], str(cfg["start"]),
                                            str(cfg["end"]), cfg["scorer"], df)
        except Exception as exc:  # noqa: BLE001 -- surface the reason, don't crash
            st.error(f"**{cfg['strategy']}** needs news data, which could not be "
                     f"loaded: {exc}")
            st.info("News comes from Alpaca. Add your keys to `.env`, or pick a "
                    "price-only strategy.")
            return
        if features is None:
            st.warning(f"No news found for {cfg['symbol']} in this window, so "
                       f"**{cfg['strategy']}** has nothing to act on.")
            return
        df = df.join(features)
        covered = int((df["news_count"] > 0).sum())
        st.caption(f"News features joined — {covered:,} of {len(df):,} bars have "
                   f"coverage ({cfg['scorer']} scorer).")

    signals = strategy.generate_signals(df)

    result = run_backtest(df, signals, BacktestConfig(
        initial_cash=float(cfg["cash"]),
        position_size=float(cfg["position_size"]),
        slippage_bps=float(cfg["slippage"]),
        commission_pct=float(cfg["commission_pct"]),
        allow_short=bool(cfg["allow_short"]),
        stop_loss_pct=float(cfg["stop_loss"]),
        take_profit_pct=float(cfg["take_profit"]),
    ))

    ppy = bars_per_year(df.index)
    stats = summarise(result, ppy)

    # ---- headline tiles ---------------------------------------------------
    cols = st.columns(6)
    tile(cols[0], "Total return", pct(stats["Total return"]),
         f"buy & hold {pct(stats.get('Buy & hold return', 0.0))}",
         tone_of(stats["Total return"]), money_kind="once")
    tile(cols[1], "CAGR", pct(stats["CAGR"]), "annualised", tone_of(stats["CAGR"]),
         money_kind="annual")
    tile(cols[2], "Sharpe", f"{stats['Sharpe']:.2f}",
         f"buy & hold {stats.get('Buy & hold Sharpe', 0.0):.2f}")
    tile(cols[3], "Max drawdown", pct(stats["Max drawdown"]),
         f"buy & hold {pct(stats.get('Buy & hold max DD', 0.0))}", "down",
         money_kind="once")
    tile(cols[4], "Win rate", pct(stats["Win rate"], 0), f"{stats['Trades']} trades")
    tile(cols[5], "Time in market", pct(stats["Time in market"], 0),
         f"final ${stats['Final equity']:,.0f}")

    bundle = build_bundle(symbol=cfg["symbol"], start=cfg["start"], end=cfg["end"],
                          timeframe=cfg["timeframe"], source=cfg["source"],
                          strategy=strategy, result=result, stats=stats)
    slug = f"{cfg['symbol']}_{cfg['strategy'].replace(' ', '_')}_{cfg['start']}_{cfg['end']}"
    st.download_button(
        "Export this run (strategy + settings + results)",
        json.dumps(bundle, indent=2, default=str).encode(),
        file_name=f"{slug}.json", mime="application/json",
        help="Reload it later from the sidebar to restore every setting and "
             "re-run the backtest against the saved result.")

    st.write("")
    tab_perf, tab_trades, tab_season, tab_stats, tab_data = st.tabs(
        ["Performance", "Trades", "Seasonality", "All statistics", "Bars"])

    with tab_perf:
        st.plotly_chart(charts.equity_chart(result, mode), use_container_width=True)
        st.plotly_chart(charts.drawdown_chart(result, mode), use_container_width=True)
        st.plotly_chart(charts.exposure_chart(result, mode), use_container_width=True)

    with tab_trades:
        st.plotly_chart(
            charts.price_chart(result, strategy.indicators(df), mode), use_container_width=True)

        left, right = st.columns([3, 2])
        with left:
            trades = result.trades.copy()
            if trades.empty:
                st.info("This strategy never opened a position over the selected window.")
            else:
                st.dataframe(
                    trades, use_container_width=True, hide_index=True, height=340,
                    column_config={
                        "entry_time": st.column_config.DatetimeColumn("Entry", format="YYYY-MM-DD"),
                        "exit_time": st.column_config.DatetimeColumn("Exit", format="YYYY-MM-DD"),
                        "direction": st.column_config.TextColumn("Side"),
                        "qty": st.column_config.NumberColumn("Qty", format="%.2f"),
                        "entry_price": st.column_config.NumberColumn("In", format="$%.2f"),
                        "exit_price": st.column_config.NumberColumn("Out", format="$%.2f"),
                        "fees": st.column_config.NumberColumn("Fees", format="$%.2f"),
                        "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
                        "return_pct": st.column_config.NumberColumn("Return", format="%.2f%%"),
                        "bars_held": st.column_config.NumberColumn("Bars"),
                        "exit_reason": st.column_config.TextColumn("Exit reason"),
                    })
                st.download_button(
                    "Download trades (CSV)", trades.to_csv(index=False).encode(),
                    file_name=f"{cfg['symbol']}_{cfg['strategy'].replace(' ', '_')}_trades.csv",
                    mime="text/csv")
        with right:
            st.plotly_chart(charts.trade_return_hist(result.trades, mode),
                            use_container_width=True)

    with tab_season:
        st.plotly_chart(charts.monthly_heatmap(monthly_returns(result.equity), mode),
                        use_container_width=True)

    with tab_stats:
        rows = []
        for k, v in stats.items():
            if isinstance(v, float) and any(w in k for w in
                                            ("return", "CAGR", "drawdown", "rate",
                                             "Volatility", "Time in market", "DD")):
                shown = pct(v, 2)
            elif isinstance(v, float):
                shown = f"{v:,.2f}"
            else:
                shown = f"{v:,}"
            rows.append({"Metric": k, "Value": shown})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     height=560)
        st.caption(f"Annualised with {ppy:,.0f} bars per year. "
                   "Sharpe and Sortino assume a 0% risk-free rate.")

    with tab_data:
        st.dataframe(df.tail(500), use_container_width=True, height=520)
        st.caption(f"{len(df):,} bars from {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}. "
                   "Showing the most recent 500.")

    st.caption(
        "Backtested results are hypothetical and assume every fill happens at the next "
        "bar's open. Slippage is applied to the fill price itself, so it shows up in the "
        "entry and exit columns rather than in commissions. These figures ignore gaps, "
        "partial fills, borrow costs and taxes, and are not a prediction of future "
        "performance.")


if __name__ == "__main__":
    main()
