"""Day trading on the practice account.

Written to be read by someone who has not done this before. The headline is
deliberately the honest one: this rule is not expected to make money, and the
page says so before it offers a single button.

Three steps, in a fixed order, same as the practice page: see what it wants to
do, send it, check what actually happened. The third step is the one that
matters here -- the whole reason to run this is to find out whether limit
orders fill in life the way the backtest assumed.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.broker import BrokerError
from core.daytrade import DayTradeConfig
from core.daytrader import DayTrader, DayTraderConfig, NotPaper
from core.env import load_env
from core.fills import FillLog, summarise, verdict
from core.marketclock import CalendarError, MarketCalendar
from core.ui import active_mode, inject_css, page_header, plain, step, tile

load_env()

st.set_page_config(page_title="Day trading", layout="wide")
mode = active_mode()
inject_css(mode)

st.markdown(page_header(
    "Day trading practice",
    "Small, fast trades on the practice account. Read the box below first."),
    unsafe_allow_html=True)

# ---------------------------------------------------------------- the warning
st.error(
    "**This is not expected to make money, and that is not a guess.**\n\n"
    "The rule was tested on about 141,000 trades over five years. It does have "
    "a real edge before costs. But the edge sits almost entirely in the first "
    "few minutes after the market opens, which is exactly when trading is most "
    "expensive. In the calmer hours where trading is cheap, the edge is smaller "
    "than the cost of placing the trade.\n\n"
    "**So why run it at all?** Because there is one thing no amount of testing "
    "on old data can answer: when you place an order to buy at a set price, "
    "does it actually get filled? The practice run measures that. Watch the "
    "fill rate in step 3, not the profit.")

with st.expander("Show me the numbers behind that"):
    plain("Each row is a slice of the trading day. **Gross** is what the rule "
        "earned before costs. <b>Spread</b> is what it costs to trade in that "
        "slice. <b>Net</b> is what is left. A basis point (bps) is one hundredth "
        "of one percent.")
    st.dataframe(pd.DataFrame([
        {"When": "First 15 minutes", "Trades": 880, "Gross (bps)": 5.38,
         "Cost (bps)": 3.78, "Net (bps)": 1.60},
        {"When": "First hour", "Trades": 12029, "Gross (bps)": 3.41,
         "Cost (bps)": 2.34, "Net (bps)": 1.07},
        {"When": "Whole day", "Trades": 140879, "Gross (bps)": 0.91,
         "Cost (bps)": 2.10, "Net (bps)": -1.19},
        {"When": "10:30-15:30 (this page)", "Trades": 121631, "Gross (bps)": 0.67,
         "Cost (bps)": 1.59, "Net (bps)": -0.92},
        {"When": "Last hour", "Trades": 19328, "Gross (bps)": 1.31,
         "Cost (bps)": 1.79, "Net (bps)": -0.48},
    ]), width="stretch", hide_index=True)
    st.caption(
        "Even the two positive rows are thin. On the worst tenth of days the "
        "spread in the first fifteen minutes is 10.6 bps, which turns that "
        "+1.60 into roughly -5. A positive average is not the same as a "
        "positive trade.")

# ---------------------------------------------------------------- the setup
try:
    calendar = MarketCalendar.load()
except CalendarError as exc:
    st.warning(f"No market calendar yet. {exc}")
    if st.button("Download the market calendar"):
        try:
            MarketCalendar.fetch("2020-01-01", "2027-12-31")
            st.rerun()
        except CalendarError as e:
            st.error(str(e))
    st.stop()

with st.sidebar:
    st.subheader("Settings")
    symbols = st.multiselect(
        "What to trade", ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA"],
        default=["SPY", "QQQ"],
        help="SPY and QQQ are the cheapest to trade, so they lose the least.")
    risk_pct = st.slider("Risk per trade (% of account)", 0.1, 2.0, 0.5, 0.1,
                         help="If the trade goes wrong, this is roughly what "
                              "you lose. 0.5% is the tested setting.")
    max_loss = st.slider("Stop for the day after losing (%)", 0.5, 5.0, 2.0, 0.5)
    st.caption("Practice account only. The code refuses to run on a live one.")

rule = DayTradeConfig(symbols=tuple(symbols), risk_frac=risk_pct / 100.0)
cfg = DayTraderConfig(symbols=tuple(symbols), rule=rule,
                      max_daily_loss_frac=max_loss / 100.0,
                      max_open_positions=max(len(symbols), 1))


@st.cache_resource(show_spinner=False)
def _broker():
    from brokers.alpaca import AlpacaBroker
    return AlpacaBroker(paper=True)


if not symbols:
    st.info("Pick at least one thing to trade in the sidebar.")
    st.stop()

try:
    broker = _broker()
    trader = DayTrader(broker, cfg, calendar)
except Exception as exc:
    st.error(f"Cannot reach the practice account: {exc}")
    st.info("Check that ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are in .env")
    st.stop()

now = datetime.now()

# ---------------------------------------------------------------- is it open
try:
    session = calendar.session(now)
    is_open = session.contains(now)
    when = (f"Open today {session.open:%H:%M} to {session.close:%H:%M} Eastern"
            + ("  (short day)" if session.is_half_day else ""))
except CalendarError:
    session, is_open = None, False
    upcoming = [s for s in calendar.sessions() if s.day > now.date()]
    when = (f"Closed today. Next open {upcoming[0].day:%A %d %B}"
            if upcoming else "Closed today.")

c1, c2, c3 = st.columns(3)
tile(c1, "Market", "Open" if is_open else "Closed", when,
     "good" if is_open else "")
try:
    acct = broker.get_account()
    tile(c2, "Practice account", f"${acct.equity:,.0f}", "play money")
    tile(c3, "Open positions", str(len(broker.get_positions())),
         "should be 0 overnight")
except BrokerError as exc:
    tile(c2, "Practice account", "unreachable", str(exc)[:60])

st.divider()

# ---------------------------------------------------------------- step 1
step(1, "See what it wants to do", active=True)
plain("Nothing is sent yet. This just reads the market and shows you the orders "
    "it would place.")

if st.button("Show me the plan", type="primary"):
    with st.spinner("Reading the market..."):
        try:
            st.session_state["dt_plan"] = trader.plan(now=now)
        except Exception as exc:
            st.error(f"Could not build a plan: {exc}")

plan = st.session_state.get("dt_plan")

if plan:
    if plan["blocks"]:
        st.warning("**Not trading right now, because:**\n\n"
                   + "\n".join(f"- {b}" for b in plan["blocks"]))
    if plan.get("closing"):
        st.info("Past the daily cut-off. From here the loop only closes "
                "positions -- it never opens new ones.")

    if plan["intents"]:
        rows = []
        for it in plan["intents"]:
            s = it.setup
            rows.append({
                "Buy or sell": "Buy" if s.direction > 0 else "Sell short",
                "What": s.symbol,
                "How many": it.qty,
                "Only at this price or better": round(s.entry_px, 2),
                "Give up if it falls to": round(s.stop_px, 2),
                "Take profit at": round(s.target_px, 2),
                "Most you can lose": f"${it.qty * s.risk_per_share:,.0f}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "These are limit orders: they only fill at your price or better, "
            "and may not fill at all. That is the point of the experiment.")
    elif not plan["blocks"]:
        st.info("No setup right now. That is normal and most checks find "
                "nothing -- the rule wants a specific shape of price move.")

    # ------------------------------------------------------------ step 2
    st.divider()
    step(2, "Send it to the practice account",
                     active=bool(plan["intents"]))

    if plan["intents"]:
        plain("This places the orders above on your Alpaca **paper** account. "
            "No real money is involved. The code refuses to run against a "
            "live account.")
        if st.button(f"Place these {len(plan['intents'])} orders"):
            try:
                sent = trader.execute(plan["intents"])
                st.success(f"Sent {len(sent)} order(s).")
                st.dataframe(pd.DataFrame(sent), width="stretch", hide_index=True)
            except NotPaper as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Something went wrong sending orders: {exc}")
    else:
        st.caption("Nothing to send.")

    if plan.get("positions"):
        if st.button("Close everything now"):
            st.dataframe(pd.DataFrame(trader.flatten()), width="stretch",
                         hide_index=True)

# ------------------------------------------------- dry run on a closed market
st.divider()
step(0, "Not sure what this does? Watch it on a past day")
plain("Runs the exact same rule over a day that has already happened, so you "
      "can see the trades it would have taken without sending anything. "
      "Useful when the market is shut.")

past = [x for x in calendar.sessions() if x.day < now.date()]
if past:
    pick = st.selectbox(
        "Which day", past[-20:][::-1], index=0,
        format_func=lambda x: (f"{x.day:%A %d %B %Y}"
                               + ("  (short day)" if x.is_half_day else "")))

    if st.button("Replay that day"):
        from core.data import DataError, load_bars
        from core.data import session as rth
        from core.daytrade import backtest, summarise

        bars, problems = {}, []
        with st.spinner("Fetching that day's bars..."):
            for sym in symbols:
                try:
                    df = load_bars(sym, pick.day, pick.day + pd.Timedelta(days=1),
                                   rule.timeframe, "alpaca")
                    bars[sym] = rth(df)
                except (DataError, Exception) as exc:
                    problems.append(f"{sym}: {exc}")

        if problems:
            st.warning("\n".join(f"- {p}" for p in problems))

        if bars:
            trades = backtest(bars, rule, calendar, cost_bps=0.0)
            gross = summarise(trades)
            costed = summarise(backtest(bars, rule, calendar, cost_bps=2.0),
                               cost_bps=2.0)

            if trades.empty:
                st.info("No trades that day. That is a common outcome -- the "
                        "rule waits for a specific shape and most days do not "
                        "produce one inside the trading window.")
            else:
                a, b, c = st.columns(3)
                tile(a, "Trades it would have taken", str(gross["trades"]))
                tile(b, "Won", f"{gross['win_rate']:.0%}",
                     "a 2R target means under half can still work")
                net = costed["mean_ret_bps"]
                tile(c, "Average, after costs", f"{net:+.2f} bps",
                     "per trade, at a 2 bps spread",
                     "good" if net > 0 else "bad")

                show = trades.assign(
                    Direction=trades["direction"].map({1: "Long", -1: "Short"}),
                    Result=trades["reason"].map({
                        "target": "Hit the target", "stop": "Stopped out",
                        "flatten": "Closed at the bell",
                        "timeout": "Ran out of time",
                        "eod_rollover": "Closed at the bell"}),
                )[["symbol", "Direction", "entry_ts", "entry_px", "exit_px",
                   "Result", "r", "ret_pct"]].rename(columns={
                    "symbol": "What", "entry_ts": "Entered",
                    "entry_px": "In at", "exit_px": "Out at",
                    "r": "Risk multiples", "ret_pct": "Return %"})
                st.dataframe(show.round(3), width="stretch", hide_index=True)
                st.caption(
                    "**Risk multiples** is the honest scorecard: +2 means it "
                    "made twice what it was risking, -1 means it lost exactly "
                    "what it put at risk. One day is far too small to judge "
                    "anything -- this is here to show you the mechanics.")
else:
    st.caption("No past sessions in the calendar yet.")

# ---------------------------------------------------------------- step 3
st.divider()
step(3, "Check what actually happened")
plain("**This is the part that matters.** The test is not whether you made "
    "money over a few dozen trades -- that tells you nothing, the numbers are "
    "far too small to read. The test is whether your orders filled, and at "
    "what price.")

st.info(
    "**One caveat about your account size.** With a $5,000 practice account "
    "and SPY near $765, a single order works out at one or two shares. That "
    "is fine for learning the mechanics, but it flatters the very thing "
    "being measured: a one-share limit order fills far more easily than a "
    "realistic one, because it can slot into a gap in the queue that a "
    "hundred shares could not. So read a good fill rate here as an upper "
    "bound, not a result.")

log = FillLog(trader.fills.path)
fills = log.frame()

if fills.empty:
    st.info("No orders recorded yet. Come back after step 2.")
else:
    s = summarise(fills)
    v = verdict(fills, assumed_cost_bps=1.59)

    a, b, c, d = st.columns(4)
    tile(a, "Orders placed", f"{s['orders']}")
    tile(b, "How many filled", f"{s['fill_rate']:.0%}",
         "below 60% and the backtest was wrong",
         "good" if s["fill_rate"] >= 0.6 else "bad")
    if "slippage_bps_mean" in s:
        tile(c, "Average slippage", f"{s['slippage_bps_mean']:.2f} bps",
             "positive means it cost you",
             "bad" if s["slippage_bps_mean"] > 1 else "good")
        tile(d, "Real cost per round trip", f"{s['effective_spread_bps']:.2f} bps",
             "the backtest assumed 1.59",
             "bad" if s["effective_spread_bps"] > 1.59 else "good")

    if v["verdict"] == "insufficient data":
        st.info(f"**Too early to say.** {v['note']}\n\n"
                "Thirty orders is the minimum before these numbers mean "
                "anything, and even then they are rough.")
    elif v["verdict"] == "fails":
        st.error("**The backtest's assumptions are not holding up:**\n\n"
                 + "\n".join(f"- {r}" for r in v["reasons"]))
    else:
        st.success("So far the real fills match what the backtest assumed. "
                   "Keep going -- this needs many more orders to be solid.")

    with st.expander("Every order, in detail"):
        cols = [c for c in ["ts", "symbol", "side", "qty", "order_type",
                            "limit_price", "filled_qty", "filled_price",
                            "status", "slippage_bps"] if c in fills.columns]
        st.dataframe(fills[cols].tail(200), width="stretch", hide_index=True)

st.divider()
st.caption(
    "Stuck on? Create a file called HALT in the project folder and the loop "
    "refuses to open anything until you delete it.")
