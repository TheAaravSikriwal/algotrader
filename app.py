"""Day trader — two tabs, and a picture of what is happening.

**Trade** is the whole job: Auto or Manual across the top, the session chart
underneath, your position drawn on it. Auto runs the buy-and-sell loop itself
and picks its own algorithm; Manual is the same workflow with you pressing the
buttons. Both obey the same rails and record into the same log.

**Workshop** tests and adds algorithms. Not the daily path — it feeds the
pool that Auto chooses from.

Auto publishes what it is doing to `live/state.json` every cycle, and reads
`live/instructions.json` before deciding anything. That is the seam Claude
works through: watch the state, write an instruction, the loop picks it up.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import money
from core.autotrader import AutoConfig, AutoTrader
from core.broker import BrokerError
from core.daytrade import DayTradeConfig
from core.daytrader import DayTrader, DayTraderConfig, NotPaper
from core.env import load_env
from core.fills import FillLog, FillRecord
from core.livecharts import (candidate_bars, cycle_strip, pnl_chart,
                             session_chart)
from core.livestate import Bridge
from core.marketclock import CalendarError, MarketCalendar
from core.recommend import load_intraday, rank
from core.ui import active_mode, inject_css
from core.version import current as current_version

load_env()
st.set_page_config(page_title="Day trader", layout="wide", page_icon="📈")
MODE = active_mode()
inject_css(MODE)

WINDOW = (time(10, 30), time(15, 30))
REFRESH_SECONDS = 30

st.markdown("""
<style>
  /* clamp() so the headline number shrinks rather than wrapping: "$4,999"
     breaking into "$4,99 / 9" is worse than a smaller font. */
  .big   { font-size: clamp(22px, 2.2vw, 36px); font-weight: 650;
           line-height: 1.1; white-space: nowrap; }
  .mid   { font-size: 22px; font-weight: 600; }
  .lab   { font-size: 12px; opacity: .6; letter-spacing: .03em;
           text-transform: uppercase; }
  .sub   { font-size: 13px; opacity: .7; }
  .pill  { display:inline-block; padding: 3px 10px; border-radius: 999px;
           font-size: 12px; font-weight: 600; }
  .on    { background: rgba(77,190,130,.18); color: #2e9c68; }
  .off   { background: rgba(140,140,150,.18); color: #7b7b85; }
  .warn  { background: rgba(230,160,60,.18);  color: #b97f22; }
  .stop  { background: rgba(220,90,90,.18);   color: #c0504d; }
</style>""", unsafe_allow_html=True)


def cycle_kind(row: dict) -> str:
    """One word for what a cycle did, for colouring the strip and the ticks."""
    head = str(row.get("headline") or row.get("result") or "").lower()
    if "placed" in head or row.get("event") == "orders":
        return "order"
    if "closed" in head or row.get("event") == "flatten":
        return "flatten"
    if "too soon" in head:
        return "throttled"
    if "watching" in head:
        return "watching"
    return "blocked"


def pill(text: str, kind: str = "off") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def stat(col, label: str, value: str, sub: str = "", tone: str = ""):
    """A big labelled number. Renders raw HTML, so it un-escapes money.

    Whether a caller passes money.md(...) or the plain string stops mattering
    here, which it should: the same helper output is correct in a markdown
    block and wrong in this one, and that distinction is not worth asking
    every call site to remember.
    """
    colour = {"good": "var(--good)", "bad": "var(--bad)"}.get(tone, "inherit")
    value, sub = money.unmd(value), money.unmd(sub)
    col.markdown(
        f'<div class="lab">{label}</div>'
        f'<div class="big" style="color:{colour}">{value}</div>'
        f'<div class="sub">{sub}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def _broker():
    from brokers.alpaca import AlpacaBroker
    return AlpacaBroker(paper=True)


@st.cache_resource(show_spinner=False)
def _calendar():
    return MarketCalendar.load()


bridge = Bridge()
try:
    calendar = _calendar()
except CalendarError as exc:
    st.error(f"No market calendar. {exc}")
    if st.button("Download it"):
        MarketCalendar.fetch("2020-01-01", "2027-12-31")
        st.rerun()
    st.stop()

try:
    broker = _broker()
except Exception as exc:                                  # noqa: BLE001
    st.error(f"Cannot reach the practice account: {exc}")
    st.stop()

now = datetime.now()
try:
    session = calendar.session(now)
    market_open = session.contains(now)
    lo, hi = session.window(*WINDOW)
    in_window = lo <= now < hi
    flat_at = session.flatten_deadline(5)
except CalendarError:
    session, market_open, in_window, flat_at = None, False, False, None

trade_tab, shop_tab = st.tabs(["**Trade**", "Workshop"])

# =========================================================== TRADE
with trade_tab:
    auto_on = st.session_state.get("auto_on", False)

    head = st.columns([2, 3, 2, 2])
    with head[0]:
        st.markdown("### Day trader")
        st.markdown(
            pill("MARKET OPEN" if market_open else "MARKET CLOSED",
                 "on" if market_open else "off")
            + " " + pill("WINDOW OPEN" if in_window else "OUT OF WINDOW",
                         "on" if in_window else "warn"),
            unsafe_allow_html=True)
        if flat_at:
            st.markdown(f'<div class="sub">flat by {flat_at:%H:%M}</div>',
                        unsafe_allow_html=True)

    with head[1]:
        how = st.radio("Mode", ["Auto", "Manual"], horizontal=True,
                       label_visibility="collapsed",
                       index=0 if auto_on else 1)

    try:
        account = broker.get_account()
        positions = broker.get_positions()
    except BrokerError as exc:
        st.error(f"Broker unreachable: {exc}")
        st.stop()

    held = next(iter(positions.values()), None)
    with head[2]:
        stat(st, "Account", f"${account.equity:,.0f}", "practice money")
    with head[3]:
        if held:
            stat(st, "Open", f"{held.qty:g} {held.symbol}",
                 f"{money.fmt(held.unrealized_pl)} unrealised",
                 "good" if held.unrealized_pl >= 0 else "bad")
        else:
            stat(st, "Open", "flat", "nothing held")

    st.divider()

    # ------------------------------------------------------------ settings
    with st.sidebar:
        st.markdown("### Settings")
        symbols = st.multiselect(
            "Trade", ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "AMD", "TSLA"],
            default=["SPY", "QQQ"])
        risk_pct = st.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1)
        max_loss = st.slider("Stop for the day at (%)", 0.5, 5.0, 2.0, 0.5)
        max_age = st.slider("Refuse bars older than (min)", 2.0, 20.0, 6.0, 1.0,
                            help="The free data feed is 15 minutes behind. "
                                 "Raising this trades a stale price.")
        st.caption("Practice account only. The code refuses a live one.")
        st.divider()
        st.caption(f"Version {current_version().label()}")

    if not symbols:
        st.info("Pick something to trade in the sidebar.")
        st.stop()

    auto = AutoTrader(broker, AutoConfig(
        symbols=tuple(symbols), risk_frac=risk_pct / 100.0,
        max_daily_loss_frac=max_loss / 100.0,
        max_open_positions=len(symbols),
        max_bar_age_minutes=max_age), calendar, bridge)
    auto._running = st.session_state.get("running", "")
    auto.cfg.paused = st.session_state.get("paused", False)

    # ============================================================= AUTO
    if how == "Auto":
        st.session_state["auto_on"] = True
        # The throttle lives on the AutoTrader, but that object is rebuilt on
        # every Streamlit rerun, so the last-cycle time is kept in session
        # state and handed back. Without this the cooldown resets on every
        # click and does nothing at all.
        auto._last_cycle_at = st.session_state.get("last_cycle_at")
        wait = auto.seconds_until_ready(now)

        c = st.columns([1, 1, 4])
        label = ("▶ Run one cycle" if wait <= 0
                 else f"▶ Ready in {wait:.0f}s")
        if c[0].button(label, type="primary", width="stretch",
                       disabled=wait > 0):
            cyc = auto.cycle(now=now, execute=True)
            if not cyc.throttled:
                st.session_state["last_cycle_at"] = auto._last_cycle_at
            st.session_state["running"] = auto._running
            st.session_state["paused"] = auto.cfg.paused
            st.rerun()
        if c[1].button("■ Close all", width="stretch"):
            auto._trader.flatten()
            st.rerun()
        if wait > 0:
            st.caption(
                f"Waiting {wait:.0f}s. The rule reads five-minute bars, so "
                f"running it again inside one bar cannot find anything new — "
                f"but it can place a second order against the same signal.")

        state = bridge.state()
        age = bridge.state_age_seconds()

        s1 = st.columns(4)
        running = state.get("running") or "nothing"
        because = state.get("chose_because", "") or "not chosen yet"
        if len(because) > 96:
            # Cut at a word, not mid-word: "...best tested rule. Switchin"
            # reads like the app broke rather than like a summary.
            because = because[:96].rsplit(" ", 1)[0] + "..."
        stat(s1[0], "Running", running, because,
             "good" if state.get("running") else "")
        cand = state.get("candidate") or {}
        if cand:
            stat(s1[1], "Its edge", f"{cand.get('expectancy_bps', 0):+.2f} bps",
                 money.md(money.brief(cand.get("expectancy_bps", 0) / 100.0,
                                      "once")),
                 "good" if cand.get("expectancy_bps", 0) > 0 else "bad")
            stat(s1[2], "Confidence", f"t = {cand.get('t_stat', 0):+.2f}",
                 f"{int(cand.get('trades', 0))} trades tested",
                 "good" if abs(cand.get("t_stat", 0)) >= 1.96 else "bad")
        stat(s1[3], "Last look",
             f"{age:.0f}s ago" if age is not None else "never",
             state.get("headline", ""), "" if (age or 0) < 120 else "bad")

        if state.get("blocks"):
            st.warning("**Standing down:**  " + "  ·  ".join(state["blocks"]))
        if state.get("instruction_applied"):
            st.info(f"**Instruction applied:** {state['instruction_applied']}")

        st.caption(
            "Auto writes what it is doing to `live/state.json` each cycle and "
            "reads `live/instructions.json` before deciding. That is how "
            "Claude steps in — watch the state, leave an instruction, the "
            "loop picks it up next cycle.")

    # =========================================================== MANUAL
    else:
        st.session_state["auto_on"] = False
        trader = DayTrader(broker, DayTraderConfig(
            symbols=tuple(symbols),
            rule=DayTradeConfig(symbols=tuple(symbols),
                                risk_frac=risk_pct / 100.0),
            max_daily_loss_frac=max_loss / 100.0,
            max_open_positions=len(symbols),
            max_bar_age_minutes=max_age), calendar)

        blocks = trader.check_rails(account, session, now) if session else \
            ["the market is closed"]
        if blocks:
            st.warning("**Rails blocking new positions:**  "
                       + "  ·  ".join(blocks))

        f = st.columns([1, 1, 1, 1, 1, 1])
        sym = f[0].selectbox("Symbol", symbols)
        side = f[1].selectbox("Side", ["buy", "sell"])
        qty = f[2].number_input("Shares", 1, 10_000, 1)

        ref = None
        try:
            recent = broker.get_bars(sym, "5Min", 3)
            if len(recent):
                ref = float(recent["close"].iloc[-1])
        except BrokerError:
            pass
        base = float(round(ref or 100.0, 2))

        limit_px = f[3].number_input("Limit", 0.01, value=base, step=0.01)
        stop_px = f[4].number_input("Stop (0=none)", 0.0, value=0.0, step=0.01)
        tgt_px = f[5].number_input("Target (0=none)", 0.0, value=0.0, step=0.01)

        if ref:
            st.caption(money.md(
                f"{qty} share(s) ≈ {money.fmt(qty * ref)} of notional. "
                f"1% against you is {money.fmt(qty * ref * 0.01)}."
                + ("  No stop set — nothing will close this for you."
                   if stop_px == 0 else "")))

        go_col, close_col = st.columns([1, 1])
        if go_col.button(f"{side.title()} {qty} {sym}", type="primary",
                         disabled=bool(blocks), width="stretch"):
            try:
                order = broker.submit_order(
                    sym, int(qty), side, order_type="limit",
                    limit_price=round(limit_px, 2),
                    stop_loss=round(stop_px, 2) if stop_px > 0 else None,
                    take_profit=round(tgt_px, 2) if tgt_px > 0 else None)
                trader.fills.record(FillRecord(
                    symbol=sym, side=side, qty=float(qty),
                    reference_price=float(ref or limit_px), order_type="limit",
                    limit_price=float(limit_px),
                    filled_qty=float(getattr(order, "filled_qty", 0) or 0),
                    filled_price=getattr(order, "filled_price", None),
                    status=getattr(order, "status", "submitted"),
                    strategy="manual", order_id=getattr(order, "id", "")))
                bridge.record("manual_order", symbol=sym, side=side, qty=int(qty),
                              limit=round(limit_px, 2))
                st.success(f"Sent — order {getattr(order, 'id', '')[:8]}")
            except BrokerError as exc:
                st.error(f"Rejected: {exc}")
        if close_col.button("Close everything now", width="stretch"):
            st.dataframe(pd.DataFrame(trader.flatten()), width="stretch",
                         hide_index=True)

    # -------------------------------------------------------- the picture
    # ------------------------------------------------- the cycle timeline
    st.divider()
    raw = bridge.decisions(limit=120)
    cycle_rows = [r for r in raw
                  if r.get("event") in {"cycle", "orders", "flatten"}]
    cycles = [{"kind": cycle_kind(r),
               "time": str(r.get("ts", ""))[11:16],
               "label": str(r.get("headline") or r.get("result") or "")[:90],
               "at": pd.to_datetime(r.get("ts"), errors="coerce", utc=True)}
              for r in cycle_rows]
    for c in cycles:
        if pd.notna(c["at"]):
            c["at"] = c["at"].tz_convert("America/New_York").tz_localize(None)
        else:
            c["at"] = None

    if cycles:
        st.markdown("**Every cycle today**")
        st.plotly_chart(cycle_strip(cycles[-60:], MODE), width="stretch",
                        config={"displayModeBar": False})

        counts: dict[str, int] = {}
        for c in cycles:
            counts[c["kind"]] = counts.get(c["kind"], 0) + 1
        ordered = [k for k in ("order", "flatten", "watching", "blocked",
                               "throttled") if k in counts]
        words = {"order": "placed an order", "flatten": "closed out",
                 "watching": "watched, no setup",
                 "blocked": "stood down", "throttled": "was told to wait"}
        summary = ", ".join(f"**{counts[k]}** {words[k]}" for k in ordered)
        last = cycles[-1]
        st.markdown(
            f"{len(cycles)} cycles so far — {summary}. "
            f"The most recent, at **{last['time']}**, {last['label'].lower()}. "
            f"Each tick under the price chart is one cycle, so a flat chart "
            f"still shows the loop was awake.")
    else:
        st.caption("No cycles run yet today. Press **Run one cycle** above — "
                   "each run reads the market once and decides, and every "
                   "decision appears here.")

    st.divider()
    watch = held.symbol if held else symbols[0]
    try:
        bars = broker.get_bars(watch, "5Min", 120)
        today = bars[bars.index.normalize() == pd.Timestamp(now.date())]
    except BrokerError:
        today = pd.DataFrame()

    fills_all = FillLog(Path("logs") / "daytrade_fills.jsonl").frame()
    fills_today = pd.DataFrame()
    if not fills_all.empty and "ts" in fills_all:
        ts = pd.to_datetime(fills_all["ts"], errors="coerce", utc=True)
        fills_today = fills_all[ts >= pd.Timestamp.now(tz="UTC").normalize()]

    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"**{watch} today**")
        st.plotly_chart(session_chart(
            today, MODE,
            entry=float(held.avg_price) if held else None,
            window=WINDOW, fills=fills_today, cycles=cycles),
            width="stretch",
            config={"displayModeBar": False})
    with right:
        st.markdown("**What happened**")
        rows = bridge.decisions(limit=60)

        # Collapse runs of the identical message. A loop that stands down for
        # the same reason every cycle otherwise fills the whole feed with one
        # sentence and buries the events that actually differ.
        feed: list[tuple[str, str, int, str]] = []
        for r in rows:
            when = str(r.get("ts", ""))[11:16]
            line = str(r.get("headline") or r.get("result") or r.get("event"))
            # Compare with the digits stripped out. "market data is 16
            # minutes behind" and "...17 minutes behind" are the same event
            # one minute apart, and treating them as different fills the feed
            # with a counter ticking up.
            shape = re.sub(r"\d+", "#", line)
            if feed and feed[-1][3] == shape:
                t0, _, n, sh = feed[-1]
                feed[-1] = (t0, line, n + 1, sh)   # keep the newest wording
            else:
                feed.append((when, line, 1, shape))

        if not feed:
            st.caption("Nothing yet today.")
        for when, line, n, _ in reversed(feed[-12:]):
            times = (f'<span style="opacity:.45"> x{n}</span>' if n > 1 else "")
            st.markdown(
                f'<div style="font-size:12px;padding:5px 0;'
                f'border-bottom:1px solid rgba(128,128,128,.15)">'
                f'<span style="opacity:.5">{when}</span> &nbsp;'
                f'{line[:150]}{times}</div>',
                unsafe_allow_html=True)

    traded_today = (not fills_today.empty
                    and fills_today["filled_qty"].fillna(0).sum() > 0)
    if traded_today:
        st.markdown("**Cost paid today**")
        st.plotly_chart(pnl_chart(fills_today, MODE), width="stretch",
                        config={"displayModeBar": False})

# ======================================================== WORKSHOP
with shop_tab:
    st.markdown("### Workshop")
    st.caption("Everything Auto can choose from. Tested as day trades — flat "
               "by the close, inside the window, costs charged both sides.")

    pool = load_intraday()
    usable = [c for c in pool if c.credible]
    w = st.columns(4)
    stat(w[0], "Rules tested", f"{len(pool)}", "rule-symbol pairs")
    stat(w[1], "Usable sample", f"{len(usable)}", "100+ trades each",
         "good" if usable else "bad")
    positive = [c for c in usable if c.expectancy_pct > 0]
    stat(w[2], "Positive", f"{len(positive)}", "after costs",
         "good" if positive else "bad")
    real = [c for c in positive if c.significant]
    stat(w[3], "Statistically real", f"{len(real)}", "t above 2",
         "good" if real else "bad")

    if not real:
        st.warning(
            "**Nothing here is distinguishable from luck yet.** Auto will "
            "still run the best-evidenced rule, and will tell you that is "
            "what it is doing. It stands aside entirely when nothing is even "
            "positive.")

    st.plotly_chart(candidate_bars(pool, MODE), width="stretch",
                    config={"displayModeBar": False})
    st.caption("Green is usable and positive, grey usable and negative, "
               "amber too few trades to judge however large the bar.")

    with st.expander("The full table"):
        st.dataframe(pd.DataFrame([{
            "Rule": c.strategy, "Symbol": c.symbol,
            "Per trade": f"{c.expectancy_pct:+.2f} bps",
            "On $1,000": money.fmt(money.amount(c.expectancy_pct / 100.0, 1_000)),
            "t": f"{c.t_stat:+.2f}", "Trades": int(c.trades),
            "Verdict": "usable" if c.credible else "too few trades",
        } for c in rank(pool)]), width="stretch", hide_index=True)

    st.caption("To add or retest a rule: `python research/evaluate_intraday.py`, "
               "or ask Claude. Results land here automatically.")
