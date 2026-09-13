"""Workshop: add an algorithm and have it tested the moment you save it.

Pick a rule, set its knobs, choose what it trades. On save it is backtested
over the full history and over the last month, filed into a quadrant by its
*measured* holding period, and scored on the arithmetic that decides whether
it is worth trading at all -- expectancy, payoff ratio, the bet size the edge
supports, and the chance of ruin at that size.

The test runs automatically because an untested algorithm in a list of tested
ones is an invitation to trade it. There is no save-without-testing path.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import money
from core.algobook import QUADRANTS, AlgoBook, SavedAlgo, suggest_quadrant
from core.data import DataError, load_bars
from core.engine import BacktestConfig, run_backtest
from core.env import load_env
from core.expectancy import risk_of_ruin, summarise as expectancy_of
from core.strategy import REGISTRY
from core.ui import active_mode, inject_css, page_header, param_form, plain, tile
import strategies  # noqa: F401  -- registers everything

load_env()
st.set_page_config(page_title="Workshop", layout="wide")
mode = active_mode()
inject_css(mode)

page_header(
    "Workshop",
    "Add an algorithm. It gets tested the moment you save it.")

book = AlgoBook()
PRICE_ONLY = sorted(n for n, c in REGISTRY.items() if not c.requires_features)

PRESETS = {
    "Fast movers (short term)": ["SPY", "QQQ", "NVDA", "AMD", "TSLA"],
    "Big and liquid": ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"],
    "Just SPY": ["SPY"],
}


def backtest_one(symbol: str, name: str, params: dict, start, end,
                 cost_bps: float, stop_pct: float, take_pct: float) -> dict | None:
    try:
        df = load_bars(symbol, start, end, "1Day", "yfinance")
    except (DataError, Exception):
        return None
    if len(df) < 60:
        return None
    try:
        sig = REGISTRY[name](**params).generate_signals(df)
        cfg = BacktestConfig(slippage_bps=cost_bps / 2.0, allow_short=True,
                             stop_loss_pct=stop_pct, take_profit_pct=take_pct)
        res = run_backtest(df, sig, cfg)
    except Exception:
        return None

    eq = res.equity
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
    hold = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100.0

    # "If you had been invested the past month" -- the question people
    # actually ask, answered on the same series rather than re-run.
    cutoff = eq.index.max() - pd.Timedelta(days=30)
    recent = eq[eq.index >= cutoff]
    last_month = (float(recent.iloc[-1] / recent.iloc[0] - 1.0) * 100.0
                  if len(recent) > 1 else None)

    trades = res.trades
    returns = (trades["return_pct"].tolist()
               if trades is not None and "return_pct" in trades else [])
    return {"symbol": symbol, "total_return_pct": total, "hold_return_pct": hold,
            "vs_hold_pct": total - hold, "last_month_pct": last_month,
            "trade_returns": returns, "bars": int(len(df)),
            "exposure": float(res.exposure.abs().mean())}


# ---------------------------------------------------------------- the form
st.markdown("### 1. Pick a rule")
left, right = st.columns([2, 3])

with left:
    chosen = st.selectbox("Rule", PRICE_ONLY,
                          index=PRICE_ONLY.index("Gap fade")
                          if "Gap fade" in PRICE_ONLY else 0)
    cls = REGISTRY[chosen]
    plain(cls.description)

    hint = suggest_quadrant(chosen)
    st.caption(f"Measured holding period files this under "
               f"**{QUADRANTS[hint].title}**.")

with right:
    params = param_form(cls.params, prefix="ws") if cls.params else {}
    if not cls.params:
        st.caption("This rule has no settings.")

st.markdown("### 2. Pick what it trades")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    preset = st.selectbox("Universe", list(PRESETS))
    symbols = st.multiselect("Symbols", sorted(
        {s for v in PRESETS.values() for s in v} | set(PRESETS[preset])),
        default=PRESETS[preset])
with c2:
    years = st.slider("Years of history", 1, 15, 5)
with c3:
    cost_bps = st.slider("Cost per trade (bps)", 0.0, 20.0, 5.0, 0.5,
                         help="Round trip. SPY mid-day measures about 1.6.")

st.markdown("### 3. Risk limits")
r1, r2, r3 = st.columns(3)
with r1:
    stop_pct = st.slider("Stop loss (%)", 0.0, 20.0, 0.0, 0.5,
                         help="0 means the rule exits on its own signal.")
with r2:
    take_pct = st.slider("Take profit (%)", 0.0, 40.0, 0.0, 0.5)
with r3:
    live = st.checkbox("Everchanging", value=False,
                       help="File under the live quadrant -- reviewed as news "
                            "and conditions change, rather than left alone.")

name = st.text_input("Name it", value=f"{chosen} — {preset}")
rationale = st.text_area(
    "Why should this work?", height=80,
    placeholder="One or two sentences. Worth writing now: when it stops "
                "working you will want to know what you originally believed.")

quadrant = suggest_quadrant(chosen, live=live)
st.caption(f"Will be filed under **{QUADRANTS[quadrant].title}**.")

if st.button("Save and test it", type="primary", disabled=not symbols):
    end = date.today()
    start = end - timedelta(days=365 * years)
    results, prog = [], st.progress(0.0, "Testing...")
    for i, sym in enumerate(symbols, 1):
        r = backtest_one(sym, chosen, params, start, end, cost_bps,
                         stop_pct, take_pct)
        if r:
            results.append(r)
        prog.progress(i / len(symbols), f"Tested {sym}")
    prog.empty()

    if not results:
        st.error("Could not test that on any of those symbols. Try a longer "
                 "history or different symbols.")
    else:
        all_trades = [x for r in results for x in r["trade_returns"]]
        exp = expectancy_of(all_trades)
        agg = {
            "total_return_pct": sum(r["total_return_pct"] for r in results) / len(results),
            "hold_return_pct": sum(r["hold_return_pct"] for r in results) / len(results),
            "vs_hold_pct": sum(r["vs_hold_pct"] for r in results) / len(results),
            "last_month_pct": (sum(r["last_month_pct"] for r in results
                                   if r["last_month_pct"] is not None)
                               / max(sum(1 for r in results
                                         if r["last_month_pct"] is not None), 1)),
            "symbols_tested": len(results),
            "years": years, "cost_bps": cost_bps,
            "trades": exp.trades,
            "win_rate": exp.win_rate,
            "expectancy_pct": exp.expectancy_pct,
            "payoff_ratio": exp.payoff_ratio,
            "profit_factor": exp.profit_factor,
            "t_stat": exp.t_stat,
            "kelly": exp.kelly_fraction,
            "suggested_size": exp.suggested_fraction,
            "stdev_pct": exp.stdev_pct,
            # Computed here because this is the only place the raw per-trade
            # returns exist; the saved record keeps aggregates only.
            "trades_needed": exp.trades_needed(),
            "breakeven_win_rate": exp.breakeven_win_rate,
            "per_symbol": [{k: v for k, v in r.items() if k != "trade_returns"}
                           for r in results],
        }
        saved = book.add(SavedAlgo(
            name=name.strip() or chosen, quadrant=quadrant, strategy=chosen,
            params=params, symbols=symbols, stop_loss_pct=stop_pct,
            take_profit_pct=take_pct, rationale=rationale.strip(),
            backtest=agg))
        st.session_state["just_saved"] = saved.id
        st.rerun()

# ------------------------------------------------------------- the result
saved_id = st.session_state.get("just_saved")
if saved_id and (algo := book.get(saved_id)):
    b = algo.backtest
    st.divider()
    st.success(f"Saved **{algo.name}** to {algo.quadrant_title}.")

    cols = st.columns(4)
    tile(cols[0], "Return over the test", f"{b['total_return_pct']:+.1f}%",
         f"buy & hold {b['hold_return_pct']:+.1f}%",
         "good" if b["vs_hold_pct"] > 0 else "bad", money_kind="once")
    tile(cols[1], "Versus holding", f"{b['vs_hold_pct']:+.1f}%",
         "the only comparison that matters",
         "good" if b["vs_hold_pct"] > 0 else "bad", money_kind="once")
    lm = b.get("last_month_pct")
    tile(cols[2], "If invested the past month",
         f"{lm:+.1f}%" if lm is not None else "—", "last 30 days",
         "good" if (lm or 0) > 0 else "bad", money_kind="once")
    tile(cols[3], "Trades", f"{b['trades']:,}",
         f"over {b['years']} years, {b['symbols_tested']} symbols")

    st.markdown("#### Is the edge real, and what size does it support?")
    e = st.columns(4)
    tile(e[0], "Expectancy per trade", f"{b['expectancy_pct']:+.3f}%",
         f"t = {b['t_stat']:.2f}" if b["t_stat"] == b["t_stat"] else "",
         "good" if b["expectancy_pct"] > 0 else "bad", money_kind="once")
    tile(e[1], "Win rate", f"{b['win_rate']:.0%}",
         f"payoff {b['payoff_ratio']:.2f}x")
    tile(e[2], "Profit factor", f"{b['profit_factor']:.3f}"
         if b["profit_factor"] == b["profit_factor"] else "—",
         "above 1.0 makes money")
    tile(e[3], "Bet size the maths supports",
         f"{b['suggested_size']:.2%}",
         f"quarter of Kelly ({b['kelly']:.1%})",
         "good" if b["suggested_size"] > 0 else "bad")

    if b["expectancy_pct"] <= 0:
        st.error(money.md(
            "**This rule loses money per trade.** No position size fixes "
            "that -- larger bets just reach zero sooner, which is why the "
            "suggested size is 0%. It is saved so you can see the evidence, "
            "not because it is worth running."))
    elif b["trades"] < 100:
        st.warning(
            f"**Only {b['trades']} trades.** That is too few to tell a real "
            f"edge from luck. Treat the numbers above as a hint, not a result.")
    else:
        t = b.get("t_stat")
        significant = t is not None and t == t and abs(t) >= 1.96
        lost_to_holding = b.get("vs_hold_pct", 0.0) <= 0

        lines = [
            f"Positive expectancy of {b['expectancy_pct']:+.3f}% a trade — "
            f"{money.fmt(money.amount(b['expectancy_pct'], 1000))} on $1,000."]
        if not significant:
            lines.append(
                (f"**But t = {t:.2f}, so it is not distinguishable from "
                 f"zero.** "
                 + (f"Confirming an edge this size would take about "
                    f"{b['trades_needed']:,} trades. "
                    if b.get("trades_needed") else "")
                 + "Treat it as a hint, not a result.")
                if t == t else "")
        if lost_to_holding:
            lines.append(
                f"**And it lost to simply holding by "
                f"{abs(b['vs_hold_pct']):.1f}%.** A positive expectancy per "
                f"trade still loses if buying and holding did better over the "
                f"same years.")
        lines.append("Whether it survives real fills is a separate question; "
                     "the live workshop measures that.")

        box = st.info if (significant and not lost_to_holding) else st.warning
        box(money.md("  ".join(x for x in lines if x)))

    if b.get("per_symbol"):
        with st.expander("Per symbol"):
            st.dataframe(pd.DataFrame(b["per_symbol"]).round(2),
                         width="stretch", hide_index=True)

    if st.button("See it in my algorithms"):
        st.session_state["quadrant"] = algo.quadrant
        st.switch_page("pages/1_My_algorithms.py")

# --------------------------------------------------------------- the list
st.divider()
existing = book.all()
st.markdown(f"### Saved algorithms ({len(existing)})")
if not existing:
    plain("Nothing saved yet.")
else:
    def cell(b, key, fmt, default="—"):
        """`.get(key, 0)` returns None when the key exists holding null.

        Seeded entries store None for anything the evaluation did not
        measure, so a plain default never fires and the format string blows
        up on NoneType. A dash also says "not measured" rather than "zero".
        """
        v = b.get(key)
        return fmt.format(v) if v is not None else default

    rows = []
    for a in existing:
        b = a.backtest or {}
        rows.append({
            "Name": a.name, "Quadrant": a.quadrant_title, "Rule": a.strategy,
            "Trades": b.get("trades") or 0,
            "Return": cell(b, "total_return_pct", "{:+.1f}%"),
            "vs holding": cell(b, "vs_hold_pct", "{:+.1f}%"),
            "Expectancy": cell(b, "expectancy_pct", "{:+.3f}%"),
            "Size": cell(b, "suggested_size", "{:.2%}"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    gone = st.selectbox("Remove one", ["—"] + [a.name for a in existing])
    if gone != "—" and st.button("Remove", type="secondary"):
        target = next(a for a in existing if a.name == gone)
        book.remove(target.id)
        st.rerun()
