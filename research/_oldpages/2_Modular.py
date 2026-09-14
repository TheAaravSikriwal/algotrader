"""Modular: the app names the rule for this moment, and re-checks it.

Every candidate here was backtested in this app as an actual day trade --
flat by the close, inside the trading window, costs charged both sides. The
page ranks them on evidence rather than on the largest number, which matters
more than it sounds: the two highest averages in the current data come from
two trades and seven trades respectively.

It is allowed to say no, and on the present evidence it usually should. A
recommender that names a winner every session would be laundering "least bad
of a losing set" into "best choice", and that costs money in a way no
disclaimer undoes.
"""
from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import money
from core.broker import BrokerError
from core.daytrade import DayTradeConfig
from core.daytrader import DayTrader, DayTraderConfig, NotPaper
from core.env import load_env
from core.expectancy import risk_of_ruin, summarise as expectancy_of
from core.marketclock import CalendarError, MarketCalendar
from core.recommend import (
    MIN_TRADES,
    best_intraday,
    load_intraday,
    rank,
    review,
)
from core.ui import active_mode, inject_css, page_header, plain, step, tile

load_env()
st.set_page_config(page_title="Modular", layout="wide")
mode = active_mode()
inject_css(mode)

page_header("Modular",
            "The app picks the rule for right now, from what it has tested.")

pool = load_intraday()
if not pool:
    st.error(
        "**Nothing has been backtested as a day trade yet.** The modular mode "
        "will not fall back on the daily-bar results — those answer a "
        "different question. Run `python research/evaluate_intraday.py`, or "
        "build a rule in the backtest lab.")
    if st.button("Open the backtest lab", type="primary"):
        st.switch_page("pages/3_Backtest_lab.py")
    st.stop()

# ------------------------------------------------------------------ filters
with st.sidebar:
    st.subheader("What to consider")
    syms = sorted({c.symbol for c in pool})
    only = st.multiselect("Symbols", syms, default=syms)
    floor = st.slider("Minimum trades tested", 0, 500, MIN_TRADES, 25,
                      help="Below about a hundred, an average is noise "
                           "wearing a decimal point.")
    st.caption(f"{len(pool)} rule-symbol pairs on record.")

considered = [c for c in pool if c.symbol in only]
running = st.selectbox("What are you running now?",
                       ["nothing yet"] + sorted({c.strategy for c in considered}))

rec = (best_intraday(candidates=considered) if running == "nothing yet"
       else review(running, candidates=considered))

# ---------------------------------------------------------------- 1. verdict
step(1, "What to run", active=True)

if rec.action == "stand_aside":
    st.error(f"**Stand aside.** {rec.reason}")
elif rec.action == "hold":
    st.success(f"**Keep {running}.** {rec.reason}")
else:
    box = st.info if rec.confident else st.warning
    box(f"**{rec.headline}.** {rec.reason}")

if rec.best:
    b = rec.best
    e = st.columns(5)
    tile(e[0], "Rule", b.strategy, f"on {b.symbol}")
    tile(e[1], "Per trade, after costs", f"{b.expectancy_pct:+.2f} bps",
         b.note, "good" if b.expectancy_pct > 0 else "bad", money_kind="bps")
    tile(e[2], "Confidence", f"t = {b.t_stat:+.2f}",
         "about 2.0 to mean anything",
         "good" if b.significant else "bad")
    tile(e[3], "Tested on", f"{b.trades:.0f}", "trades",
         "good" if b.credible else "bad")
    tile(e[4], "Trades a session", f"{b.exposure:.1f}",
         f"holds ~{b.avg_hold_bars:.0f} bars")

    # What it is worth, in money, at a sane size.
    per_1k = money.amount(b.expectancy_pct / 100.0, 1_000)
    a_year = money.grow(b.expectancy_pct / 100.0, int(b.exposure * 252) or 1, 1_000)
    st.caption(money.md(
        f"On a $1,000 trade that is {money.fmt(per_1k)} a trade. At "
        f"{b.exposure:.1f} trades a session it compounds to "
        f"{money.fmt(a_year - 1_000)} over a year — before tax, and assuming "
        f"the edge is real, which at t={b.t_stat:.2f} is not established."))

st.divider()

# ------------------------------------------------------------ 2. the field
step(2, "Why that one")
plain("Ranked on evidence, not on the biggest number. A large average on a "
      "handful of trades does not outrank a small one on hundreds.")

rows = []
for c in rank(considered)[:30]:
    rows.append({
        "Rule": c.strategy, "Symbol": c.symbol,
        "Per trade": f"{c.expectancy_pct:+.2f} bps",
        "On $1,000": money.fmt(money.amount(c.expectancy_pct / 100.0, 1_000)),
        "t": f"{c.t_stat:+.2f}",
        "Trades": int(c.trades),
        "Verdict": ("usable" if c.credible
                    else ("too few trades" if c.expectancy_pct > 0
                          else "loses money")),
    })
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

thin = [c for c in considered if c.expectancy_pct > 0 and not c.credible]
if thin:
    worst = max(thin, key=lambda c: c.expectancy_pct)
    st.caption(
        f"{len(thin)} rule(s) show a positive average on fewer than {floor} "
        f"trades. The largest is {worst.strategy} at "
        f"{worst.expectancy_pct:+.1f} bps on {worst.trades:.0f} trades — "
        f"shown, never recommended.")

st.divider()

# -------------------------------------------------------------- 3. the size
step(3, "How much to bet")
if rec.best and rec.best.expectancy_pct > 0:
    st.caption(
        "The bet size is set by the size of the edge, not by how confident "
        "you feel. Past the Kelly fraction more risk means less money, and "
        "below zero expectancy no size is profitable.")
    size = st.slider("Risk per trade (% of account)", 0.1, 3.0, 0.5, 0.1)
    equity = st.number_input("Account size ($)", 500, 1_000_000, 5_000, 500)
    per_trade_risk = equity * size / 100.0
    st.markdown(money.md(
        f"Risking **{money.fmt(per_trade_risk)}** a trade on a "
        f"{money.fmt(equity)} account. At {rec.best.exposure:.1f} trades a "
        f"session that is **{money.fmt(per_trade_risk * rec.best.exposure)}** "
        f"of risk put up per day."))
else:
    st.info("No size to recommend while the evidenced answer is to stand "
            "aside. That is not caution, it is arithmetic: a negative "
            "expectancy loses faster the more you bet.")

st.divider()

# ------------------------------------------------------------ 4. the orders
step(4, "Place it on the practice account")

try:
    calendar = MarketCalendar.load()
except CalendarError as exc:
    st.warning(f"No market calendar. {exc}")
    st.stop()

if rec.action == "stand_aside":
    st.info("Nothing to place while the answer is to stand aside. The Trade "
            "page will still let you place an order yourself if you disagree.")
    if st.button("Go to Trade"):
        st.switch_page("pages/1_Trade.py")
    st.stop()

with st.sidebar:
    st.subheader("Trading settings")
    risk_pct = st.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1, key="mod_risk")
    max_loss = st.slider("Daily loss limit (%)", 0.5, 5.0, 2.0, 0.5, key="mod_loss")

trade_symbols = (rec.best.symbol,) if rec.best else tuple(only[:2])
rule_cfg = DayTradeConfig(symbols=trade_symbols, risk_frac=risk_pct / 100.0)
cfg = DayTraderConfig(symbols=trade_symbols, rule=rule_cfg,
                      max_daily_loss_frac=max_loss / 100.0,
                      max_open_positions=len(trade_symbols))


@st.cache_resource(show_spinner=False)
def _broker():
    from brokers.alpaca import AlpacaBroker
    return AlpacaBroker(paper=True)


try:
    broker = _broker()
    trader = DayTrader(broker, cfg, calendar)
    acct = broker.get_account()
except Exception as exc:
    st.error(f"Cannot reach the practice account: {exc}")
    st.stop()

now = datetime.now()
m = st.columns(3)
try:
    s = calendar.session(now)
    tile(m[0], "Market", "Open" if s.contains(now) else "Closed",
         f"{s.open:%H:%M}–{s.close:%H:%M} ET",
         "good" if s.contains(now) else "")
except CalendarError:
    tile(m[0], "Market", "Closed", "not a trading day")
tile(m[1], "Practice account", f"${acct.equity:,.0f}", "fake money")
tile(m[2], "Open positions", str(len(broker.get_positions())),
     "should be 0 overnight")

if st.button("Show me the plan", type="primary"):
    with st.spinner("Reading the market..."):
        try:
            st.session_state["mod_plan"] = trader.plan(now=now)
        except Exception as exc:
            st.error(f"Could not build a plan: {exc}")

plan = st.session_state.get("mod_plan")
if plan:
    if plan["blocks"]:
        st.warning("**Not trading right now, because:**\n\n"
                   + "\n".join(f"- {b}" for b in plan["blocks"]))
    if plan["intents"]:
        st.dataframe(pd.DataFrame([{
            "Buy or sell": "Buy" if i.setup.direction > 0 else "Sell short",
            "What": i.setup.symbol, "How many": i.qty,
            "At this price or better": round(i.setup.entry_px, 2),
            "Give up at": round(i.setup.stop_px, 2),
            "Take profit at": round(i.setup.target_px, 2),
            "Most you can lose": money.fmt(i.qty * i.setup.risk_per_share),
        } for i in plan["intents"]]), width="stretch", hide_index=True)
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
        st.info("No setup on the bar that just closed. Most checks find "
                "nothing — that is the rule being selective, not broken.")

    if plan.get("positions") and st.button("Close everything now"):
        st.dataframe(pd.DataFrame(trader.flatten()), width="stretch",
                     hide_index=True)

st.caption(
    "Every candidate here was backtested in this app under day-trading rules. "
    "Fills on a paper account are optimistic — see "
    "research/PAPER_TRADING_LIMITS.md before reading anything into them.")
