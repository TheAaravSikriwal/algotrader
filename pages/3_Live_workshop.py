"""Live workshop: which tested rule should be running right now.

This is the everchanging side. It does three things, in order:

  1. **Reviews what you are running.** Is it still the best-evidenced choice
     for this horizon, given everything measured so far? Hold, switch, or
     stand aside.
  2. **Shows the news** that would make you want to override that.
  3. **Places the orders** on the practice account, once you approve them.

The review is deliberately biased toward "hold" and toward "stand aside".
Switching on a small difference converts noise into turnover and turnover
into spread, and the evidence in this repo is that no short-horizon rule
tested here has a positive edge after costs. A recommender that named a
winner every time would be lying in a way that costs money.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import money
from core.algobook import QUADRANTS, SHORT_LIVE, AlgoBook
from core.broker import BrokerError
from core.daytrade import DayTradeConfig
from core.daytrader import DayTrader, DayTraderConfig, NotPaper
from core.env import load_env
from core.expectancy import risk_of_ruin, summarise as expectancy_of
from core.fills import FillLog, summarise as fill_summary, verdict
from core.marketclock import CalendarError, MarketCalendar
from core.recommend import best_for, load_candidates, rank, review
from core.ui import active_mode, inject_css, page_header, plain, step, tile

load_env()
st.set_page_config(page_title="Live workshop", layout="wide")
mode = active_mode()
inject_css(mode)

page_header(
    "Live workshop",
    "Which tested rule should be running right now, and what it would do.")

book = AlgoBook()

horizon = st.radio(
    "Horizon", ["short", "long"], horizontal=True,
    format_func=lambda h: "Short term (day trading)" if h == "short"
    else "Long term", label_visibility="collapsed")

# ---------------------------------------------------------------- 1. review
step(1, "What should be running", active=True)

candidates = load_candidates(horizon=horizon)
running = st.selectbox(
    "What are you running now?",
    ["nothing yet"] + sorted({c.strategy for c in candidates}))

rec = (best_for(horizon=horizon, candidates=candidates)
       if running == "nothing yet"
       else review(running, horizon=horizon, candidates=candidates))

if rec.action == "stand_aside":
    st.error(f"**{rec.headline}.** {rec.reason}")
elif rec.action == "hold":
    st.success(f"**{rec.headline}.** {rec.reason}")
else:
    box = st.info if rec.confident else st.warning
    box(f"**{rec.headline}.** {rec.reason}")

if not rec.confident and rec.action != "stand_aside":
    st.caption(
        "Not statistically confident. Across 348 measured strategy-symbol "
        "pairs, none cleared the corrected significance bar — so this is the "
        "best-evidenced guess, not a proven rule.")

if candidates:
    with st.expander(f"The whole field ({len(candidates)} tested pairs)"):
        rows = [{
            "Rule": c.strategy, "Symbol": c.symbol,
            "vs holding": f"{c.expectancy_pct:+.2f}%/yr",
            "On $1,000": money.fmt(money.amount(c.expectancy_pct, 1_000)),
            "t": f"{c.t_stat:+.2f}", "Trades": int(c.trades),
            "Holds": f"{c.avg_hold_bars:.1f}d",
            "Usable": "yes" if c.credible else "too few trades",
        } for c in rank(candidates)[:25]]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Ranked on evidence, not on return: a large average on forty "
            "trades does not outrank a small one on nine hundred.")
else:
    st.warning("Nothing evaluated for this horizon yet. Run "
               "`python research/evaluate_all.py` to populate it.")

# ------------------------------------------------------------------ 2. news
st.divider()
step(2, "What the news says")
plain("Context for overriding the review above. The app gathers it; you and "
      "I decide what it means — that judgement is deliberately not automated.")

with st.expander("This week's news and how to feed it in"):
    st.markdown(
        "1. Open **Research → News** to gather the week's headlines.\n"
        "2. Paste the block into Claude along with what you are running.\n"
        "3. Paste the answer back there; it becomes weight adjustments.\n\n"
        "The base rule sets the positions. The news adjusts how much of each, "
        "and never silently — every adjustment is recorded with its reason "
        "and an expiry date.")
    if st.button("Open the news tools"):
        st.switch_page("pages/4_Research.py")

# ------------------------------------------------------------- 3. the orders
st.divider()
step(3, "Place it on the practice account")

if horizon != "short":
    st.info("Order placement here covers the short-term loop. For long-term "
            "books use **Research → Practice book**.")
    st.stop()

try:
    calendar = MarketCalendar.load()
except CalendarError as exc:
    st.warning(f"No market calendar yet. {exc}")
    if st.button("Download it"):
        try:
            MarketCalendar.fetch("2020-01-01", "2027-12-31")
            st.rerun()
        except CalendarError as e:
            st.error(str(e))
    st.stop()

with st.sidebar:
    st.subheader("Trading settings")
    symbols = st.multiselect("What to trade",
                             ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA"],
                             default=["SPY", "QQQ"])
    risk_pct = st.slider("Risk per trade (% of account)", 0.1, 2.0, 0.5, 0.1)
    max_loss = st.slider("Stop for the day after losing (%)", 0.5, 5.0, 2.0, 0.5)
    st.caption("Practice account only. The code refuses a live one.")

if not symbols:
    st.info("Pick at least one symbol in the sidebar.")
    st.stop()

rule = DayTradeConfig(symbols=tuple(symbols), risk_frac=risk_pct / 100.0)
cfg = DayTraderConfig(symbols=tuple(symbols), rule=rule,
                      max_daily_loss_frac=max_loss / 100.0,
                      max_open_positions=max(len(symbols), 1))


@st.cache_resource(show_spinner=False)
def _broker():
    from brokers.alpaca import AlpacaBroker
    return AlpacaBroker(paper=True)


try:
    broker = _broker()
    trader = DayTrader(broker, cfg, calendar)
except Exception as exc:
    st.error(f"Cannot reach the practice account: {exc}")
    st.stop()

now = datetime.now()
try:
    session = calendar.session(now)
    is_open = session.contains(now)
    when = (f"{session.open:%H:%M}–{session.close:%H:%M} Eastern"
            + ("  (short day)" if session.is_half_day else ""))
except CalendarError:
    is_open = False
    upcoming = [s for s in calendar.sessions() if s.day > now.date()]
    when = (f"Next open {upcoming[0].day:%A %d %B}" if upcoming else "Closed")

m = st.columns(3)
tile(m[0], "Market", "Open" if is_open else "Closed", when,
     "good" if is_open else "")
try:
    acct = broker.get_account()
    tile(m[1], "Practice account", f"${acct.equity:,.0f}", "fake money")
    tile(m[2], "Open positions", str(len(broker.get_positions())),
         "should be 0 overnight")
except BrokerError as exc:
    tile(m[1], "Practice account", "unreachable", str(exc)[:50])

if st.button("Show me the plan", type="primary"):
    with st.spinner("Reading the market..."):
        try:
            st.session_state["lw_plan"] = trader.plan(now=now)
        except Exception as exc:
            st.error(f"Could not build a plan: {exc}")

plan = st.session_state.get("lw_plan")
if plan:
    if plan["blocks"]:
        st.warning("**Not trading right now, because:**\n\n"
                   + "\n".join(f"- {b}" for b in plan["blocks"]))
    if plan["intents"]:
        rows = []
        for it in plan["intents"]:
            s = it.setup
            rows.append({
                "Buy or sell": "Buy" if s.direction > 0 else "Sell short",
                "What": s.symbol, "How many": it.qty,
                "At this price or better": round(s.entry_px, 2),
                "Give up at": round(s.stop_px, 2),
                "Take profit at": round(s.target_px, 2),
                "Most you can lose": money.fmt(it.qty * s.risk_per_share),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if st.button(f"Place these {len(plan['intents'])} orders"):
            try:
                sent = trader.execute(plan["intents"])
                st.success(f"Sent {len(sent)} order(s).")
                st.dataframe(pd.DataFrame(sent), width="stretch",
                             hide_index=True)
            except NotPaper as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
    elif not plan["blocks"]:
        st.info("No setup right now. Most checks find nothing — the rule "
                "wants a specific shape of price move.")

    if plan.get("positions") and st.button("Close everything now"):
        st.dataframe(pd.DataFrame(trader.flatten()), width="stretch",
                     hide_index=True)

# ------------------------------------------------------------- 4. fill quality
st.divider()
step(4, "Did the orders actually fill?")
plain("**This is the measurement that matters.** Fifty trades cannot tell a "
      "3 basis point edge from zero, so the profit column is noise. Whether "
      "your limit orders fill the way the backtest assumed is answerable, and "
      "it is the assumption every result here rests on.")

fills = FillLog(trader.fills.path).frame()
if fills.empty:
    st.info("No orders recorded yet.")
else:
    s = fill_summary(fills)
    v = verdict(fills, assumed_cost_bps=1.59)
    f = st.columns(4)
    tile(f[0], "Orders placed", f"{s['orders']}")
    tile(f[1], "Filled", f"{s['fill_rate']:.0%}",
         "below 60% and the backtest was wrong",
         "good" if s["fill_rate"] >= 0.6 else "bad")
    if "slippage_bps_mean" in s:
        tile(f[2], "Average slippage", f"{s['slippage_bps_mean']:.2f} bps",
             "positive means it cost you", money_kind="bps",
             tone="bad" if s["slippage_bps_mean"] > 1 else "good")
        tile(f[3], "Real round-trip cost",
             f"{s['effective_spread_bps']:.2f} bps",
             "the backtest assumed 1.59", money_kind="bps",
             tone="bad" if s["effective_spread_bps"] > 1.59 else "good")

    if v["verdict"] == "insufficient data":
        st.info(f"**Too early to say.** {v['note']}")
    elif v["verdict"] == "fails":
        st.error("**The backtest's assumptions are not holding:**\n\n"
                 + "\n".join(f"- {r}" for r in v["reasons"]))
    else:
        st.success("Real fills match what the backtest assumed, so far.")

    with st.expander("Every order"):
        cols = [c for c in ["ts", "symbol", "side", "qty", "limit_price",
                            "filled_qty", "filled_price", "status",
                            "slippage_bps"] if c in fills.columns]
        st.dataframe(fills[cols].tail(200), width="stretch", hide_index=True)

st.caption("Nothing here touches real money. Create a file called HALT in the "
           "project folder to stop the loop opening anything.")
