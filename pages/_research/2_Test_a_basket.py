"""Cross-sectional backtests: rank a universe, hold the best.

A separate page because the shape of the question is different. The main page
asks "should I hold SPY today?"; this one asks "of these twelve names, which
three?" -- and the signal is the comparison between them, which a single-symbol
view cannot express.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.data import bars_per_year, load_bars
from core.env import load_env
from core.metrics import equity_stats
from core.panel import Panel, PanelConfig, run_panel_backtest
from core.theme import apply_layout, tokens
from core.ui import page_header, param_form, pct, tile, tone_of
from strategies.cross_sectional import available_xs, get_xs_strategy

load_env()

try:
    st.set_page_config(page_title="Cross-sectional", layout="wide")
except Exception:
    pass  # the host page already configured it

UNIVERSES = {
    "megacap": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
                "TSLA", "JPM", "V", "WMT", "XOM"],
    "sectors": ["XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB",
                "XLU", "XLRE", "XLC"],
    "dow": ["AAPL", "MSFT", "JNJ", "KO", "XOM", "JPM", "PG", "WMT", "CVX",
            "MRK", "HD", "CAT", "IBM", "CSCO", "VZ", "MCD", "NKE", "AXP"],
    "factor": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD",
               "DBC", "VNQ"],
}


@st.cache_data(show_spinner="Fetching the universe...")
def load_panel_bars(symbols: tuple, start: str, end: str, source: str) -> dict:
    bars, failed = {}, []
    for symbol in symbols:
        try:
            df = load_bars(symbol, start, end, "1Day", source)
            if len(df) > 30:
                bars[symbol] = df
        except Exception:  # noqa: BLE001
            failed.append(symbol)
    return {"bars": bars, "failed": failed}


def paired_tstat(a_equity: pd.Series, b_equity: pd.Series) -> float:
    a = a_equity.pct_change().dropna()
    b = b_equity.reindex(a.index).pct_change().dropna()
    diff = (a - b).dropna()
    if len(diff) < 30:
        return 0.0
    sd = diff.std(ddof=1)
    return 0.0 if sd <= 0 else float(diff.mean() / (sd / len(diff) ** 0.5))


def own_tstat(equity: pd.Series) -> float:
    returns = equity.pct_change().dropna()
    if len(returns) < 30:
        return 0.0
    sd = returns.std(ddof=1)
    return 0.0 if sd <= 0 else float(returns.mean() / (sd / len(returns) ** 0.5))


def equity_chart(curves: dict[str, pd.Series], mode: str) -> go.Figure:
    t = tokens(mode)
    slots = [t["series_1"], t["series_2"], t["series_3"], t["drawdown"]]
    fig = go.Figure()
    for i, (label, series) in enumerate(curves.items()):
        colour = t["muted"] if label.startswith("Equal weight") else slots[i % len(slots)]
        dash = "dash" if label.startswith("Equal weight") else "solid"
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, name=label, mode="lines",
            line=dict(color=colour, width=2, dash=dash),
            hovertemplate=f"{label}  $%{{y:,.0f}}<extra></extra>"))
        clean = series.dropna()
        if not clean.empty:
            fig.add_annotation(x=clean.index[-1], y=clean.iloc[-1], text=label,
                               showarrow=False, xanchor="left", xshift=8,
                               font=dict(size=11, color=colour))
    apply_layout(fig, mode, "Equity curve", 420)
    fig.update_layout(margin=dict(l=8, r=150, t=44, b=8))
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def sidebar():
    cfg = {}
    with st.sidebar:
        st.markdown("### Cross-sectional")
        preset = st.selectbox("Universe", ["megacap", "sectors", "dow", "factor",
                                           "custom"])
        default = ",".join(UNIVERSES.get(preset, UNIVERSES["megacap"]))
        raw = st.text_area("Symbols", default, height=90,
                           help="Comma separated. Three minimum for a cross-section.")
        cfg["symbols"] = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
        cfg["source"] = st.selectbox("Price source", ["yfinance", "alpaca"])

        today = date.today()
        c1, c2 = st.columns(2)
        cfg["start"] = c1.date_input("From", today - timedelta(days=365 * 10),
                                     max_value=today)
        cfg["end"] = c2.date_input("To", today, max_value=today)

        st.divider()
        st.markdown("**Strategy**")
        names = available_xs()
        default_idx = names.index("Cross-sectional momentum") if \
            "Cross-sectional momentum" in names else 0
        cfg["strategy"] = st.selectbox("Rule", names, index=default_idx)
        cls = get_xs_strategy(cfg["strategy"])
        if cls.description:
            st.caption(cls.description)
        cfg["params"] = param_form(cls.params, f"xs_{cfg['strategy']}")

        st.divider()
        st.markdown("**Execution**")
        cfg["cash"] = st.number_input("Starting cash ($)", 1_000, 10_000_000,
                                      100_000, 5_000)
        cfg["slippage"] = st.slider("Slippage (bps per side)", 0.0, 50.0, 5.0, 0.5)
        cfg["allow_short"] = st.checkbox(
            "Allow short positions", False,
            help="Required by the dollar-neutral rules. Without it their short "
                 "leg is clipped and you are measuring a different strategy.")
        cfg["split"] = st.slider(
            "In-sample fraction", 0.0, 0.9, 0.6, 0.05,
            help="0 runs the whole window. Otherwise the remainder is held out "
                 "and reported separately.")
    return cfg


def evaluate(panel: Panel, name: str, cfg: PanelConfig, params: dict | None = None):
    strategy = get_xs_strategy(name)(**(params or {}))
    weights = strategy.generate_weights(panel)
    result = run_panel_backtest(panel, weights, cfg)
    return result, equity_stats(result.equity, bars_per_year(result.equity.index))


def slice_panel(panel: Panel, start: int, end: int) -> Panel:
    return Panel(opens=panel.opens.iloc[start:end], highs=panel.highs.iloc[start:end],
                 lows=panel.lows.iloc[start:end], closes=panel.closes.iloc[start:end],
                 volumes=panel.volumes.iloc[start:end])


def show_window(label: str, window: Panel, cfg: dict, pcfg: PanelConfig, mode: str):
    st.markdown(f"#### {label}")
    st.caption(f"{window.index[0]:%Y-%m-%d} to {window.index[-1]:%Y-%m-%d} "
               f"· {len(window):,} bars")

    try:
        result, stats = evaluate(window, cfg["strategy"], pcfg, cfg["params"])
        ew_result, ew_stats = evaluate(window, "Equal weight all", pcfg)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not run this strategy: {exc}")
        return

    t_vs_ew = paired_tstat(result.equity, ew_result.equity)
    t_vs_zero = own_tstat(result.equity)

    cols = st.columns(6)
    tile(cols[0], "Total return", pct(stats.get("Total return", 0)),
         f"equal weight {pct(ew_stats.get('Total return', 0))}",
         tone_of(stats.get("Total return", 0)), money_kind="once")
    tile(cols[1], "CAGR", pct(stats.get("CAGR", 0)), "annualised",
         tone_of(stats.get("CAGR", 0)), money_kind="annual")
    tile(cols[2], "Sharpe", f"{stats.get('Sharpe', 0):.2f}",
         f"equal weight {ew_stats.get('Sharpe', 0):.2f}")
    tile(cols[3], "Max drawdown", pct(stats.get("Max drawdown", 0)),
         f"equal weight {pct(ew_stats.get('Max drawdown', 0))}", "down",
         money_kind="once")
    tile(cols[4], "t vs equal weight", f"{t_vs_ew:+.2f}",
         "did the ranking add anything",
         "up" if t_vs_ew >= 2 else ("down" if t_vs_ew <= -2 else "flat"))
    tile(cols[5], "t vs zero", f"{t_vs_zero:+.2f}", "is the return real at all",
         "up" if t_vs_zero >= 2 else ("down" if t_vs_zero <= -2 else "flat"))

    st.write("")
    st.plotly_chart(equity_chart({cfg["strategy"]: result.equity,
                                  "Equal weight": ew_result.equity}, mode),
                    use_container_width=True,
                    key=f"eq_{label}_{cfg['strategy']}")

    left, right = st.columns([3, 2])
    with left:
        holdings = result.weights[result.weights.abs().sum(axis=1) > 1e-9]
        if holdings.empty:
            st.info("This strategy never took a position over the window.")
        else:
            latest = holdings.iloc[-1]
            held = latest[latest.abs() > 1e-6].sort_values(ascending=False)
            st.markdown("**Holdings on the final bar**")
            st.dataframe(
                pd.DataFrame({"symbol": held.index, "weight": held.to_numpy()}),
                hide_index=True, use_container_width=True, height=240,
                column_config={"weight": st.column_config.NumberColumn(
                    "Weight", format="%.1f%%")})
    with right:
        st.markdown("**Cost and churn**")
        st.metric("Annual turnover", f"{result.annual_turnover:,.2f}x")
        st.metric("Fills", f"{len(result.trades):,}")
        if result.annual_turnover > 20:
            st.warning(
                f"At {result.annual_turnover:,.0f}x turnover, slippage alone "
                f"costs roughly {result.annual_turnover * pcfg.slippage_bps / 100:,.1f}% "
                "a year. Compare against a zero-slippage run to see whether "
                "the signal or the cost is doing the work.")


def main():
    mode = page_header(
        "Cross-sectional",
        "Rank a universe and hold the best. The signal is the comparison "
        "between names, which a single-symbol backtest cannot express.")
    cfg = sidebar()

    if len(cfg["symbols"]) < 3:
        st.error("A cross-section needs at least three symbols.")
        return
    if cfg["start"] >= cfg["end"]:
        st.error("The start date has to come before the end date.")
        return

    loaded = load_panel_bars(cfg["symbols"], str(cfg["start"]), str(cfg["end"]),
                             cfg["source"])
    if loaded["failed"]:
        st.warning(f"No data for: {', '.join(loaded['failed'])}")
    if len(loaded["bars"]) < 3:
        st.error("Fewer than three symbols returned data.")
        return

    panel = Panel.from_bars(loaded["bars"])
    st.caption(f"{len(panel.symbols)} symbols · {len(panel):,} bars · "
               f"{panel.index[0]:%Y-%m-%d} to {panel.index[-1]:%Y-%m-%d}")

    for note in panel.survivorship_warning():
        st.warning(f"**Survivorship:** {note}")

    pcfg = PanelConfig(initial_cash=float(cfg["cash"]),
                       slippage_bps=float(cfg["slippage"]),
                       allow_short=bool(cfg["allow_short"]))

    windows = [("Full period", 0, len(panel))]
    if cfg["split"] > 0:
        cut = int(len(panel) * cfg["split"])
        windows = [("In-sample", 0, cut), ("Held out", cut, len(panel))]

    for label, start, end in windows:
        window = slice_panel(panel, start, end)
        if len(window) < 60:
            st.info(f"{label} is too short to evaluate.")
            continue
        show_window(label, window, cfg, pcfg, mode)
        st.divider()

    st.caption(
        "Equal weight is the benchmark that matters here: beating SPY may only "
        "mean the universe outperformed, while beating equal weight means the "
        "ranking added something. For a dollar-neutral rule, read *t vs zero* "
        "instead — comparing a market-neutral book to a long-only basket "
        "penalises it for exposure it deliberately does not take.")


if __name__ == "__main__":
    main()
