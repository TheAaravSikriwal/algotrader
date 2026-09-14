"""Trade: you choose the rule, or you place the orders yourself.

Two ways to use this page.

**Base algorithm.** Pick one of the rules backtested here, and the page shows
what it wants to do right now. Nothing is sent until you press the button.

**Manual.** Place your own order. The app does not second-guess the trade, but
it still holds every rail: session window, flatten deadline, daily loss limit,
position cap, whole shares, and a HALT file that stops everything. It also
records the fill so the numbers accumulate the same way.

The rails are the point of doing it here rather than in the broker's own app.
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
from core.expectancy import summarise as expectancy_of
from core.fills import FillLog, FillRecord, mid, summarise as fill_summary
from core.marketclock import CalendarError, MarketCalendar
from core.recommend import load_intraday
from core.ui import active_mode, inject_css, page_header, plain, step, tile

def _cost_of(candidate) -> float:
    """The cost charged in that candidate's backtest, from its note.

    The note reads "62% win, 17.99 bps before 4.00 bps of cost". Pulling the
    figure back out beats threading another field through the CSV for one
    sentence of copy, and falls back to zero rather than guessing.
    """
    import re

    m = re.search(r"before ([\d.]+) bps of cost", candidate.note or "")
    return float(m.group(1)) if m else 0.0


load_env()
st.set_page_config(page_title="Trade", layout="wide")
mode = active_mode()
inject_css(mode)

page_header("Trade", "You pick the rule, or you place the order yourself.")

try:
    calendar = MarketCalendar.load()
except CalendarError as exc:
    st.warning(f"No market calendar. {exc}")
    if st.button("Download it"):
        try:
            MarketCalendar.fetch("2020-01-01", "2027-12-31")
            st.rerun()
        except CalendarError as e:
            st.error(str(e))
    st.stop()

with st.sidebar:
    st.subheader("Settings")
    symbols = st.multiselect(
        "What to trade", ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "AMD", "TSLA",
                          "MSFT"], default=["SPY", "QQQ"])
    risk_pct = st.slider("Risk per trade (% of account)", 0.1, 2.0, 0.5, 0.1)
    max_loss = st.slider("Stop for the day after losing (%)", 0.5, 5.0, 2.0, 0.5)
    max_age = st.slider("Refuse bars older than (min)", 2.0, 20.0, 6.0, 1.0,
                        help="Alpaca's free feed is 15 minutes behind. Raising "
                             "this lets the loop trade a stale price.")
    st.caption("Practice account only. The code refuses a live one.")

if not symbols:
    st.info("Pick at least one symbol in the sidebar.")
    st.stop()

rule_cfg = DayTradeConfig(symbols=tuple(symbols), risk_frac=risk_pct / 100.0)
cfg = DayTraderConfig(symbols=tuple(symbols), rule=rule_cfg,
                      max_daily_loss_frac=max_loss / 100.0,
                      max_open_positions=max(len(symbols), 1),
                      max_bar_age_minutes=max_age)


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
    when = f"{session.open:%H:%M}–{session.close:%H:%M} ET"
    flat_at = session.flatten_deadline(5)
except CalendarError:
    session, is_open, flat_at = None, False, None
    when = "closed today"

m = st.columns(4)
tile(m[0], "Market", "Open" if is_open else "Closed", when,
     "good" if is_open else "")
try:
    acct = broker.get_account()
    positions = broker.get_positions()
    tile(m[1], "Practice account", f"${acct.equity:,.0f}", "fake money")
    tile(m[2], "Open positions", str(len(positions)),
         "flat by " + (f"{flat_at:%H:%M}" if flat_at else "the close"),
         "bad" if positions and not is_open else "")
    tile(m[3], "Buying power", f"${acct.buying_power:,.0f}", "intraday")
except BrokerError as exc:
    st.error(f"Broker unreachable: {exc}")
    st.stop()

st.divider()

how = st.radio("How do you want to trade?",
               ["Run a base algorithm", "Place it myself"], horizontal=True)

# ====================================================================== auto
if how == "Run a base algorithm":
    pool = load_intraday()
    tested = sorted({c.strategy for c in pool})

    step(1, "Pick the rule", active=True)
    if tested:
        by_name = {}
        for c in pool:
            by_name.setdefault(c.strategy, []).append(c)

        # Order by evidence, not alphabetically. The alphabetical default put
        # ADX trend first -- a rule whose gross edge of 1.60 bps is entirely
        # eaten by a 1.59 bps spread -- which is a poor thing to greet someone
        # with on the page where they choose what to run.
        def _rank_key(name: str):
            """Rules with a usable sample come first, best expectancy within.

            Sorting on expectancy alone put "Buy the dip, +205.87 bps" at the
            top -- on three trades. That is the same trap the recommender
            refuses by design, reintroduced in a sort key, and it would have
            been the first thing offered on the page where you choose what to
            run.
            """
            usable = [c for c in by_name[name] if c.credible]
            if usable:
                return (1, max(c.expectancy_pct for c in usable))
            return (0, max(c.expectancy_pct for c in by_name[name]))

        tested = sorted(tested, key=_rank_key, reverse=True)
        n_usable = sum(1 for n in tested if any(c.credible for c in by_name[n]))
        chosen = st.selectbox(
            "Rule", tested,
            help="Rules with at least 100 tested trades come first, ordered "
                 "by expectancy. Below those sit rules whose sample is too "
                 "small to judge, however large their average looks.")
        st.caption(f"{n_usable} of {len(tested)} rules have a usable sample. "
                   f"The rest are listed below them.")
        mine = sorted(by_name[chosen], key=lambda c: -c.expectancy_pct)
        best = mine[0]

        e = st.columns(4)
        tile(e[0], "Best symbol for it", best.symbol,
             f"{best.trades:.0f} trades tested")
        tile(e[1], "Per trade, after costs",
             f"{best.expectancy_pct:+.2f} bps", best.note,
             "good" if best.expectancy_pct > 0 else "bad", money_kind="bps")
        tile(e[2], "Confidence", f"t = {best.t_stat:+.2f}",
             "needs about 2.0 to mean anything",
             "good" if best.significant else "bad")
        tile(e[3], "Usable sample", "yes" if best.credible else "no",
             f"{best.trades:.0f} trades, 100 is the floor",
             "good" if best.credible else "bad")

        gross = best.expectancy_pct + _cost_of(best)
        if best.expectancy_pct <= 0:
            st.error(money.md(
                f"**Loses money per trade** on the tested data. No position "
                f"size fixes that — larger bets only lose it faster."))
        elif abs(best.expectancy_pct) < 0.5:
            # Below half a basis point the number is zero at any size a retail
            # account trades. Calling it "positive" is arithmetically true and
            # practically false.
            st.error(money.md(
                f"**Effectively zero.** {best.expectancy_pct:+.2f} bps a trade "
                f"is {money.fmt(money.amount(best.expectancy_pct / 100.0, 1_000))} "
                f"on a $1,000 trade — nothing, at any size you would trade. "
                f"It made {gross:.2f} bps before costs and the spread took "
                f"almost all of it."))
        elif not best.credible:
            st.warning("Too few trades to judge. Treat anything it does as "
                       "an experiment.")
        elif not best.significant:
            st.warning(money.md(
                f"**Positive but inside the noise.** Worth "
                f"{money.fmt(money.amount(best.expectancy_pct / 100.0, 1_000))} "
                f"a trade on $1,000 if real, but t={best.t_stat:.2f} cannot "
                f"tell it from luck."))
        else:
            st.success("Positive and statistically distinguishable from zero "
                       "— rare enough here to be worth double-checking.")
    else:
        st.warning("Nothing has been backtested as a day trade yet. Open the "
                   "backtest lab first.")
        if st.button("Open the backtest lab"):
            st.switch_page("pages/3_Backtest_lab.py")
        st.stop()

    st.divider()
    step(2, "See what it wants to do")
    plain("Nothing is sent. This reads the market and shows the orders it "
          "would place.")

    if st.button("Show me the plan", type="primary"):
        with st.spinner("Reading the market..."):
            try:
                st.session_state["trade_plan"] = trader.plan(now=now)
            except Exception as exc:
                st.error(f"Could not build a plan: {exc}")

    plan = st.session_state.get("trade_plan")
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

            st.divider()
            step(3, "Send it")
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
            st.info("No setup right now. Most checks find nothing.")

# ==================================================================== manual
else:
    step(1, "Your order", active=True)
    plain("The app will not argue with the trade. It still holds the rails: "
          "session window, flatten deadline, daily loss limit, whole shares, "
          "and the HALT file.")

    blocks = trader.check_rails(acct, session, now) if session else ["market closed"]
    if blocks:
        st.warning("**The rails are blocking new positions:**\n\n"
                   + "\n".join(f"- {b}" for b in blocks))

    c = st.columns(4)
    sym = c[0].selectbox("Symbol", symbols)
    side = c[1].selectbox("Side", ["buy", "sell"])
    kind = c[2].selectbox("Order", ["limit", "market"])
    qty = c[3].number_input("Shares", min_value=1, max_value=10_000, value=1)

    ref_px, bid, ask = None, None, None
    try:
        recent = broker.get_bars(sym, "5Min", 3)
        if len(recent):
            ref_px = float(recent["close"].iloc[-1])
    except BrokerError:
        pass

    p = st.columns(3)
    limit_px = p[0].number_input(
        "Limit price", min_value=0.01, value=float(round(ref_px or 100.0, 2)),
        step=0.01, disabled=(kind != "limit"))
    stop_px = p[1].number_input("Stop loss (0 = none)", min_value=0.0,
                                value=0.0, step=0.01)
    target_px = p[2].number_input("Take profit (0 = none)", min_value=0.0,
                                  value=0.0, step=0.01)

    if ref_px:
        st.caption(money.md(
            f"Last 5-minute close {money.fmt(ref_px)}. "
            f"{qty} shares is about {money.fmt(qty * ref_px)} of notional; "
            f"1% against you is {money.fmt(qty * ref_px * 0.01)}."))

    st.divider()
    step(2, "Send it")
    disabled = bool(blocks)
    if disabled:
        st.caption("Clear the blocks above first, or close a position.")

    if st.button("Place this order", type="primary", disabled=disabled):
        try:
            order = broker.submit_order(
                sym, int(qty), side, order_type=kind,
                limit_price=round(limit_px, 2) if kind == "limit" else None,
                stop_loss=round(stop_px, 2) if stop_px > 0 else None,
                take_profit=round(target_px, 2) if target_px > 0 else None)
            trader.fills.record(FillRecord(
                symbol=sym, side=side, qty=float(qty),
                reference_price=float(ref_px or limit_px),
                order_type=kind,
                limit_price=float(limit_px) if kind == "limit" else None,
                filled_qty=float(getattr(order, "filled_qty", 0.0) or 0.0),
                filled_price=getattr(order, "filled_price", None),
                status=getattr(order, "status", "submitted"),
                strategy="manual", order_id=getattr(order, "id", "")))
            st.success(f"Sent. Order {getattr(order, 'id', '')[:8]}")
        except BrokerError as exc:
            st.error(f"The venue rejected it: {exc}")
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")

# =================================================================== always
st.divider()
step(4, "Positions and fills")

pos = broker.get_positions()
if pos:
    st.dataframe(pd.DataFrame([{
        "Symbol": p.symbol, "Shares": p.qty, "Average price": round(p.avg_price, 2),
        "Worth": money.fmt(p.market_value),
        "Up or down": money.fmt(p.unrealized_pl),
    } for p in pos.values()]), width="stretch", hide_index=True)
    if st.button("Close everything now"):
        st.dataframe(pd.DataFrame(trader.flatten()), width="stretch",
                     hide_index=True)
else:
    st.caption("No open positions.")

fills = FillLog(trader.fills.path).frame()
if not fills.empty:
    s = fill_summary(fills)
    today = fills[pd.to_datetime(fills["ts"], errors="coerce", utc=True)
                  >= pd.Timestamp.now(tz="UTC").normalize()]
    f = st.columns(3)
    tile(f[0], "Orders today", f"{len(today)}", f"{s['orders']} all time")
    tile(f[1], "Filled", f"{s['fill_rate']:.0%}", "of all orders",
         "good" if s["fill_rate"] >= 0.6 else "bad")
    if "slippage_bps_mean" in s:
        tile(f[2], "Average slippage", f"{s['slippage_bps_mean']:.2f} bps",
             "positive means it cost you", money_kind="bps",
             tone="bad" if s["slippage_bps_mean"] > 1 else "good")

    filled = fills[fills["filled_qty"] > 0]
    if len(filled) >= 2 and "slippage_bps" in filled:
        e = expectancy_of((-filled["slippage_bps"] / 100.0).tolist())
        st.caption(
            f"Across {e.trades} fills the average cost was "
            f"{-e.expectancy_pct * 100:.2f} bps a side. On a paper account "
            f"read that as a floor, not a measurement — see "
            f"research/PAPER_TRADING_LIMITS.md.")

    with st.expander("Every order"):
        cols = [c for c in ["ts", "symbol", "side", "qty", "order_type",
                            "limit_price", "filled_qty", "filled_price",
                            "status", "slippage_bps", "strategy"]
                if c in fills.columns]
        st.dataframe(fills[cols].tail(200), width="stretch", hide_index=True)

st.caption("Create a file called HALT in the project folder to stop the loop "
           "opening anything.")
