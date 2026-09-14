"""Landing page: pick how you want to trade today.

Two ways in, and the difference between them is who chooses the rule:

  * **Trade** -- you pick a base algorithm, or you place the orders yourself
    and the app just keeps the rails on.
  * **Modular** -- the app names the best-evidenced rule for right now, from
    the ones backtested in this app as actual day trades, and re-checks it as
    the session moves.

Everything either mode can offer has been through the backtest lab under
day-trading rules: flat by the close, confined to the trading window, costs
charged both sides of every turn. Nothing reaches this page on daily-bar
evidence, because "which rule right now" is not a question daily bars answer.
"""
from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import money
from core.env import load_env
from core.marketclock import CalendarError, MarketCalendar
from core.recommend import best_intraday, load_intraday, rank
from core.ui import active_mode, inject_css, page_header, plain, tile
from core.version import current as current_version

load_env()

st.set_page_config(page_title="Day trader", layout="wide", page_icon="📈")
mode = active_mode()
inject_css(mode)

page_header("Day trader", "Pick how you want to trade today.")

# ------------------------------------------------------------- market state
now = datetime.now()
session = None
try:
    calendar = MarketCalendar.load()
    session = calendar.session(now)
    is_open = session.contains(now)
    lo, hi = session.window(time(10, 30), time(15, 30))
    in_window = lo <= now < hi
    when = f"{session.open:%H:%M}–{session.close:%H:%M} ET"
    if session.is_half_day:
        when += "  (short day)"
except CalendarError:
    calendar, is_open, in_window = None, False, False
    when = "closed today"
    try:
        nxt = [s for s in MarketCalendar.load().sessions() if s.day > now.date()]
        when = f"closed — next open {nxt[0].day:%A %d %b}" if nxt else "closed"
    except CalendarError:
        pass

pool = load_intraday()
rec = best_intraday(candidates=pool)

m = st.columns(4)
tile(m[0], "Market", "Open" if is_open else "Closed", when,
     "good" if is_open else "")
tile(m[1], "Trading window", "Open" if in_window else "Shut",
     "10:30–15:30, when the spread is narrowest",
     "good" if in_window else "")
tile(m[2], "Rules backtested", f"{len(pool)}",
     "as day trades, costs included")
usable = [c for c in pool if c.credible]
tile(m[3], "With a usable sample", f"{len(usable)}",
     f"at least 100 trades each", "good" if usable else "bad")

st.divider()

# ------------------------------------------------------------- the two modes
left, right = st.columns(2)

with left:
    st.markdown("### Trade")
    plain("You choose. Run one base algorithm all session, or place the "
          "orders yourself and let the app hold the rails — session window, "
          "flatten deadline, daily loss limit, position caps.")
    st.caption("Best when you have a view of your own, or you want to watch "
               "one rule behave before trusting it.")
    if st.button("Open Trade", type="primary", width="stretch"):
        st.switch_page("pages/1_Trade.py")

with right:
    st.markdown("### Modular")
    plain("The app chooses. It names the best-evidenced rule for this moment "
          "out of everything backtested here, re-checks it as the session "
          "moves, and tells you when the honest answer is to stand aside.")
    st.caption("It will refuse to name one when nothing has a real edge. "
               "That refusal is the feature.")
    if st.button("Open Modular", width="stretch"):
        st.switch_page("pages/2_Modular.py")

# --------------------------------------------------------- what it says now
st.divider()
st.markdown("### What the app would pick right now")

if rec.action == "stand_aside":
    st.error(f"**Stand aside.** {rec.reason}")
elif rec.best:
    box = st.info if rec.confident else st.warning
    box(money.md(
        f"**{rec.best.strategy} on {rec.best.symbol}** — "
        f"{rec.best.expectancy_pct:+.2f} bps a trade after costs, which is "
        f"{money.fmt(money.amount(rec.best.expectancy_pct / 100.0, 1_000))} "
        f"on a $1,000 trade, on {rec.best.trades:.0f} trades "
        f"(t={rec.best.t_stat:.2f})."
        + ("" if rec.confident else
           "  The margin is inside the noise — the best-evidenced guess, not "
           "a proven edge.")))

if pool:
    with st.expander(f"Every rule backtested here ({len(pool)})"):
        import pandas as pd
        st.dataframe(pd.DataFrame([{
            "Rule": c.strategy, "Symbol": c.symbol,
            "Per trade": f"{c.expectancy_pct:+.2f} bps",
            "On $1,000": money.fmt(money.amount(c.expectancy_pct / 100.0, 1_000)),
            "t": f"{c.t_stat:+.2f}", "Trades": int(c.trades),
            "Usable": "yes" if c.credible else "too few trades",
        } for c in rank(pool)[:40]]), width="stretch", hide_index=True)
        st.caption(
            "Ranked on evidence, not on the biggest number: a large average "
            "on a handful of trades does not outrank a small one on hundreds. "
            "Two rules in this table show more than +65 bps on fewer than ten "
            "trades, and neither is recommendable.")

st.divider()
b1, b2 = st.columns(2)
with b1:
    st.markdown("**Backtest lab**")
    plain("Build a rule and test it as a day trade. Anything that passes "
          "through here becomes available to both modes above.")
    if st.button("Open the lab", width="stretch"):
        st.switch_page("pages/3_Backtest_lab.py")
with b2:
    st.markdown("**Research**")
    plain("The longer-horizon tools: basket tests, event studies, news, and "
          "the scoreboard of everything ever tried.")
    if st.button("Open research", width="stretch"):
        st.switch_page("pages/4_Research.py")

st.caption(
    "Nothing here places a real order. Both modes trade a practice account "
    "with fake money and the code refuses a live one. Alpaca's free data is "
    "15 minutes delayed, and the loop stands down rather than trading a stale "
    "price — see research/PAPER_TRADING_LIMITS.md.")

_v = current_version()
if _v.stale:
    st.warning(
        f"**The code changed after this app started.** You are looking at "
        f"pages from `{_v.commit}` but the modules behind them were loaded "
        f"earlier. Close the window and reopen it — Streamlit caches imports, "
        f"so a reload is not enough.")
st.caption(f"Version {_v.label()}"
           + (f" — {_v.subject}" if _v.subject else ""))
