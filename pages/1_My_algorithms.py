"""What you hold in the selected quadrant, and why.

One row per algorithm: what it trades, what it made, where it gives up, and
the reason you wrote down when you saved it. That last column matters more
than it looks -- when a rule stops working you will want to know what you
originally believed, and reconstructing it afterwards is how a thesis quietly
becomes whatever the chart is currently doing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import money
from core.algobook import QUADRANTS, SHORT_TERM, AlgoBook, quadrant_totals
from core.env import load_env
from core.ui import active_mode, inject_css, page_header, plain, tile

load_env()
st.set_page_config(page_title="My algorithms", layout="wide")
mode = active_mode()
inject_css(mode)

book = AlgoBook()
totals = quadrant_totals(book)

keys = list(QUADRANTS)
current = st.session_state.get("quadrant", SHORT_TERM)
if current not in keys:
    current = SHORT_TERM

picked = st.radio(
    "Quadrant", keys, index=keys.index(current), horizontal=True,
    format_func=lambda k: QUADRANTS[k].title, label_visibility="collapsed")
st.session_state["quadrant"] = picked
q = QUADRANTS[picked]

page_header(q.title, q.blurb)
plain(q.detail)

algos = book.by_quadrant(picked)
t = totals[picked]

# ------------------------------------------------------------ the headline
cols = st.columns(4)
if t["tested"]:
    tile(cols[0], "Average return", f"{t['total_return_pct']:+.1f}%",
         "across the algorithms here, backtested",
         "good" if t["total_return_pct"] > 0 else "bad", money_kind="once")
    tile(cols[1], "Versus holding", f"{t['vs_hold_pct']:+.1f}%",
         "the comparison that decides it",
         "good" if t["vs_hold_pct"] > 0 else "bad", money_kind="once")
    lm = t.get("last_month_pct")
    tile(cols[2], "Past month", f"{lm:+.1f}%" if lm is not None else "—",
         "if you had been invested",
         "good" if (lm or 0) > 0 else "bad", money_kind="once")
    tile(cols[3], "Beat buy-and-hold", f"{t['beat_hold']} of {t['tested']}",
         "algorithms in this quadrant",
         "good" if t["beat_hold"] else "bad")
else:
    tile(cols[0], "Algorithms here", f"{t['algos']}", "none tested yet")

if not algos:
    st.info("**Nothing filed here yet.** Add an algorithm in the Workshop and "
            "it lands in a quadrant automatically, chosen by how long it "
            "actually holds a position.")
    if st.button("Open the Workshop", type="primary"):
        st.switch_page("pages/2_Workshop.py")
    st.stop()

st.divider()

# ------------------------------------------------------------ the algorithms
for a in algos:
    b = a.backtest or {}
    exp = b.get("expectancy_pct", 0.0)
    good = b.get("vs_hold_pct", 0.0) > 0

    with st.container(border=True):
        head, stat = st.columns([3, 2])
        with head:
            st.markdown(f"**{a.name}**")
            st.caption(f"{a.strategy} · {', '.join(a.symbols) or 'no symbols'}")
            if a.rationale:
                plain(a.rationale)
            else:
                st.caption("_No reason recorded. Worth adding one._")
        with stat:
            if a.tested:
                mark = "good" if good else "bad"
                st.markdown(
                    f'<div style="font-size:26px;font-weight:600;'
                    f'color:{"var(--good)" if good else "var(--bad)"}">'
                    f'{b["total_return_pct"]:+.1f}%</div>'
                    f'<div style="opacity:.7;font-size:13px">'
                    f'{money.brief(b["total_return_pct"], "once")}'
                    f'</div>'
                    f'<div style="opacity:.6;font-size:12px">'
                    f'{b["vs_hold_pct"]:+.1f}% vs holding · '
                    f'{b.get("trades") or 0:,} trades</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown("_Not tested yet._")

        def shown(key, fmt, default="—"):
            """A missing measurement reads as a dash, never as zero.

            Seeded entries carry annualised excess but no per-trade outcomes,
            and rendering those as "0% win rate" states something false
            rather than something absent.
            """
            v = b.get(key)
            return fmt.format(v) if v is not None else default

        d = st.columns(5)
        d[0].metric("Expectancy", f"{exp:+.3f}%",
                    help="Average result of one trade. Negative means no "
                         "position size makes this profitable.")
        d[1].metric("Win rate", shown("win_rate", "{:.0%}"))
        d[2].metric("Payoff", shown("payoff_ratio", "{:.2f}x"),
                    help="Average win divided by average loss.")
        d[3].metric("Stop loss",
                    f"{a.stop_loss_pct:.1f}%" if a.stop_loss_pct else "on signal",
                    help="Where the position is given up. 'On signal' means "
                         "the rule itself decides.")
        d[4].metric("Suggested size", shown("suggested_size", "{:.2%}"),
                    help="A quarter of the Kelly fraction implied by this "
                         "rule's own win rate and payoff.")

        if a.tested and exp <= 0:
            st.error(
                "Loses money per trade. Larger bets would only lose it faster.")
        elif a.tested and b.get("trades", 0) < 100:
            st.warning(f"Only {b.get('trades', 0)} trades — too few to trust.")

        if b.get("per_symbol"):
            with st.expander("Per symbol"):
                st.dataframe(pd.DataFrame(b["per_symbol"]).round(2),
                             width="stretch", hide_index=True)

st.divider()
a1, a2 = st.columns(2)
with a1:
    if st.button("Add another algorithm", width="stretch"):
        st.switch_page("pages/2_Workshop.py")
with a2:
    if q.live and st.button("Open the live workshop", type="primary",
                            width="stretch"):
        st.switch_page("pages/3_Live_workshop.py")

st.caption("Returns here are backtested, not money made. " + money.caveat())
