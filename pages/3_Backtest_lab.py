"""Backtest lab: test a rule as a day trade, and let both modes use it.

Everything here runs under day-trading rules -- flat by the close, confined to
the trading window, costs charged both sides of every turn. That is not the
same as running a strategy on five-minute bars: a plain backtest carries
positions overnight, trades the expensive first minutes, and charges a
daily-bar cost model to something that turns over sixty times a week. All
three flatter the result.

What comes out is per-trade economics, because that is what decides whether a
day-trading rule is worth running: expectancy, payoff, how many trades it
rests on, and the bet size the edge actually supports.
"""
from __future__ import annotations

import sys
from datetime import date, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import money
from core.data import DataError, load_bars, session as rth
from core.env import load_env
from core.expectancy import risk_of_ruin, summarise as expectancy_of
from core.intraday import IntradayConfig, combine, run
from core.marketclock import CalendarError, MarketCalendar
from core.strategy import REGISTRY
from core.ui import active_mode, inject_css, page_header, param_form, plain, tile
import strategies  # noqa: F401  -- registers everything

load_env()
st.set_page_config(page_title="Backtest lab", layout="wide")
mode = active_mode()
inject_css(mode)

page_header("Backtest lab", "Test a rule the way you would actually trade it.")

RESULTS = Path(__file__).resolve().parent.parent / "research" / "results"
INTRADAY = RESULTS / "evaluate_intraday.csv"
SPREAD_BPS = {"SPY": 1.59, "QQQ": 2.12, "IWM": 2.10}

try:
    calendar = MarketCalendar.load()
except CalendarError as exc:
    st.error(f"No market calendar. {exc}")
    st.stop()

names = sorted(n for n, c in REGISTRY.items()
               if not c.requires_features and not n.startswith("ZZ"))

st.markdown("### 1. The rule")
left, right = st.columns([2, 3])
with left:
    chosen = st.selectbox("Rule", names)
    cls = REGISTRY[chosen]
    plain(cls.description)
with right:
    params = param_form(cls.params, prefix="lab") if cls.params else {}
    if not cls.params:
        st.caption("This rule has no settings.")

st.markdown("### 2. What and when")
c = st.columns(4)
symbols = c[0].multiselect("Symbols",
                           ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "AMD",
                            "TSLA", "MSFT"], default=["SPY", "QQQ"])
months = c[1].slider("Months of history", 1, 24, 6)
timeframe = c[2].selectbox("Bar size", ["5Min", "15Min", "1Min"])
allow_short = c[3].checkbox("Allow shorting", value=True)

w = st.columns(3)
start_t = w[0].time_input("Trade from", time(10, 30))
end_t = w[1].time_input("Trade until", time(15, 30))
default_cost = w[2].slider("Cost for symbols not measured (bps)",
                           0.0, 20.0, 4.0, 0.5,
                           help="Round trip. SPY, QQQ and IWM use their own "
                                "measured mid-day spread instead.")

st.caption(
    "The window excludes the open by default because that is where trading is "
    "most expensive: SPY's spread is 3.78 bps in the first fifteen minutes "
    "against 1.59 mid-day, which is wider than most intraday edges.")


def load(symbol: str) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=31 * months)
    frames, cursor = [], start
    while cursor < end:
        stop = min(date(cursor.year + 1, 1, 1), end)
        try:
            frames.append(load_bars(symbol, cursor, stop, timeframe, "alpaca"))
        except (DataError, Exception):
            pass
        cursor = stop
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return rth(df[~df.index.duplicated(keep="last")].sort_index())


if st.button("Run the backtest", type="primary", disabled=not symbols):
    results, prog = [], st.progress(0.0, "Loading...")
    for i, sym in enumerate(symbols, 1):
        df = load(sym)
        if len(df) < 200:
            prog.progress(i / len(symbols), f"{sym}: not enough data")
            continue
        cfg = IntradayConfig(
            timeframe=timeframe, window_start=start_t, window_end=end_t,
            cost_bps=SPREAD_BPS.get(sym, default_cost), allow_short=allow_short)
        results.append(run(df, chosen, sym, params=params, cfg=cfg,
                           calendar=calendar))
        prog.progress(i / len(symbols), f"Tested {sym}")
    prog.empty()
    st.session_state["lab_results"] = results
    st.session_state["lab_meta"] = {"strategy": chosen, "params": params,
                                    "timeframe": timeframe, "months": months}

results = st.session_state.get("lab_results")
if not results:
    st.stop()

both = combine(results)
if both.empty:
    st.warning("That rule produced no trades under these settings. A window "
               "this narrow, or a rule that rarely fires, will do that.")
    st.stop()

e = expectancy_of(both["return_pct"].tolist())
gross = expectancy_of(both["gross_pct"].tolist())
sessions = sum(r.sessions for r in results)

st.divider()
st.markdown("### 3. What it did")

t1 = st.columns(4)
tile(t1[0], "Per trade, after costs", f"{e.expectancy_pct * 100:+.2f} bps",
     f"gross {gross.expectancy_pct * 100:+.2f} bps",
     "good" if e.expectancy_pct > 0 else "bad", money_kind="bps")
tile(t1[1], "Trades", f"{e.trades:,}",
     f"{e.trades / max(sessions, 1):.1f} a session, {sessions} sessions",
     "good" if e.trades >= 100 else "bad")
tile(t1[2], "Win rate", f"{e.win_rate:.0%}",
     f"needs {e.breakeven_win_rate:.0%} at this payoff",
     "good" if e.win_rate > e.breakeven_win_rate else "bad")
tile(t1[3], "Confidence", f"t = {e.t_stat:+.2f}" if e.t_stat == e.t_stat else "—",
     "about 2.0 to mean anything",
     "good" if abs(e.t_stat or 0) >= 1.96 and e.expectancy_pct > 0 else "bad")

t2 = st.columns(4)
tile(t2[0], "Payoff", f"{e.payoff_ratio:.2f}x",
     f"+{e.avg_win:.2f}% win, -{e.avg_loss:.2f}% loss")
tile(t2[1], "Profit factor",
     f"{e.profit_factor:.3f}" if e.profit_factor == e.profit_factor else "—",
     "above 1.0 makes money",
     "good" if (e.profit_factor or 0) > 1 else "bad")
tile(t2[2], "Bet size the maths supports", f"{e.suggested_fraction:.2%}",
     f"quarter of Kelly ({e.kelly_fraction:.1%})",
     "good" if e.suggested_fraction > 0 else "bad")
need = e.trades_needed()
tile(t2[3], "Trades to confirm it", f"{need:,}" if need else "—",
     "at 80% power" if need else "no sample confirms a losing edge",
     "good" if need and need <= e.trades else "bad")

# ------------------------------------------------------------- the verdict
if e.expectancy_pct <= 0:
    st.error(money.md(
        f"**Loses {money.fmt(abs(money.amount(e.expectancy_pct, 1_000)))} per "
        f"$1,000 traded.** No position size fixes a negative expectancy — "
        f"larger bets only reach zero sooner, which is why the suggested size "
        f"is 0%. Gross it made {gross.expectancy_pct * 100:+.2f} bps, so the "
        f"cost of trading is what turned it negative."))
elif e.trades < 100:
    st.warning(
        f"**Only {e.trades} trades.** Too few to separate an edge from luck. "
        f"The modular mode will show this but never recommend it.")
elif abs(e.t_stat or 0) < 1.96:
    st.warning(money.md(
        f"**Positive but inside the noise** (t={e.t_stat:.2f}). Worth "
        f"{money.fmt(money.amount(e.expectancy_pct, 1_000))} per $1,000 "
        f"traded if real. Confirming it would take about "
        f"{need:,} trades against the {e.trades:,} you have."))
else:
    st.success(money.md(
        f"**Positive and significant** (t={e.t_stat:.2f}) on {e.trades:,} "
        f"trades — worth {money.fmt(money.amount(e.expectancy_pct, 1_000))} "
        f"per $1,000 traded. Rare enough to be worth double-checking before "
        f"trusting: try a different window and a different set of symbols."))

# ----------------------------------------------------------------- the risk
with st.expander("What could this do to the account?"):
    size = st.slider("Risk per trade (%)", 0.1, 5.0,
                     max(e.suggested_fraction * 100, 0.5), 0.1, key="lab_size")
    ruin = risk_of_ruin(e, size / 100.0, trades=250, runs=1500)
    r = st.columns(3)
    tile(r[0], "Chance of losing half", f"{ruin['risk_of_ruin']:.1%}",
         "within 250 trades",
         "bad" if ruin["risk_of_ruin"] > 0.1 else "good")
    tile(r[1], "Typical worst drawdown", f"{ruin['worst_drawdown']:.1%}",
         "median across simulations")
    tile(r[2], "Median outcome", f"{(ruin['median_end'] - 1) * 100:+.1f}%",
         "of starting equity",
         "good" if ruin["median_end"] > 1 else "bad")

# ------------------------------------------------------------- the breakdown
per_symbol = pd.DataFrame([{
    "Symbol": r.symbol, "Sessions": r.sessions, "Trades": len(r.trades),
    "Per trade": f"{expectancy_of(r.trade_returns).expectancy_pct * 100:+.2f} bps",
    "Win rate": f"{expectancy_of(r.trade_returns).win_rate:.0%}",
    "Cost charged": f"{SPREAD_BPS.get(r.symbol, default_cost):.2f} bps",
} for r in results if not r.trades.empty])
if not per_symbol.empty:
    st.markdown("#### Per symbol")
    st.dataframe(per_symbol, width="stretch", hide_index=True)
    st.caption("A rule that only works on one symbol has usually found that "
               "symbol's recent history rather than a repeatable effect.")

with st.expander(f"Every trade ({len(both):,})"):
    show = both[["entry_ts", "exit_ts", "symbol", "direction", "entry_px",
                 "exit_px", "bars_held", "gross_pct", "return_pct", "reason"]]
    st.dataframe(show.tail(400).round(4), width="stretch", hide_index=True)

# ------------------------------------------------------------ make it usable
st.divider()
st.markdown("### 4. Make it available to the trading modes")
plain("Saving appends this result to the same file both modes read. They will "
      "only ever recommend it if it has positive expectancy on at least a "
      "hundred trades — saving a losing rule records the evidence, it does "
      "not promote it.")

if st.button("Save this result"):
    rows = []
    for r in results:
        if r.trades.empty:
            continue
        se = expectancy_of(r.trade_returns)
        sg = expectancy_of(r.trades["gross_pct"].tolist())
        rows.append({
            "strategy": chosen, "symbol": r.symbol, "sessions": r.sessions,
            "trades": se.trades,
            "trades_per_session": se.trades / max(r.sessions, 1),
            "win_rate": se.win_rate,
            "expectancy_pct": se.expectancy_pct,
            "expectancy_bps": se.expectancy_pct * 100.0,
            "gross_bps": sg.expectancy_pct * 100.0,
            "cost_bps": SPREAD_BPS.get(r.symbol, default_cost),
            "payoff_ratio": se.payoff_ratio,
            "profit_factor": se.profit_factor,
            "t_stat": se.t_stat, "kelly": se.kelly_fraction,
            "suggested_size": se.suggested_fraction,
            "stdev_pct": se.stdev_pct, "trades_needed": se.trades_needed(),
            "avg_bars_held": float(r.trades["bars_held"].mean()),
        })
    if rows:
        new = pd.DataFrame(rows)
        if INTRADAY.exists():
            old = pd.read_csv(INTRADAY)
            # Replace any earlier run of the same rule on the same symbol
            # rather than letting duplicates accumulate and split the evidence.
            keep = ~(old["strategy"].eq(chosen)
                     & old["symbol"].isin(new["symbol"]))
            new = pd.concat([old[keep], new], ignore_index=True)
        INTRADAY.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(INTRADAY, index=False)
        st.success(f"Saved {len(rows)} result(s). Both trading modes can see "
                   f"it now.")
        if st.button("Open Modular"):
            st.switch_page("pages/2_Modular.py")
