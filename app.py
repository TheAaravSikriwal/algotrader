"""Landing page: pick a quadrant, see what you hold there.

Four quadrants, split on horizon and on whether the rule is reviewed as
conditions change:

                    | fixed rule   | everchanging
    ----------------|--------------|--------------
    short term      | Short term   | Short term, live
    long term       | Long term    | Long term, live

Each tile shows the holistic gain or loss of the algorithms filed under it.
That number is *backtested*, and the page says so on every tile, because a
landing page full of green percentages that were never traded is precisely
how someone talks themselves into funding a losing rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import money
from core.algobook import (
    LONG_LIVE,
    LONG_TERM,
    QUADRANTS,
    SHORT_LIVE,
    SHORT_TERM,
    AlgoBook,
    quadrant_totals,
)
from core.env import load_env
from core.ui import active_mode, inject_css, page_header, plain

load_env()

st.set_page_config(page_title="Trading workbench", layout="wide",
                   page_icon="📈")
mode = active_mode()
inject_css(mode)

page_header(
    "Trading workbench",
    "Pick the kind of algorithm you want to look at.")

book = AlgoBook()
totals = quadrant_totals(book)

ORDER = [SHORT_TERM, SHORT_LIVE, LONG_TERM, LONG_LIVE]


def render(container, key: str):
    q = QUADRANTS[key]
    t = totals[key]
    with container:
        st.markdown(
            f'<div style="font-size:19px;font-weight:600;margin-bottom:2px">'
            f'{q.title}</div>'
            f'<div style="opacity:.65;font-size:13px;margin-bottom:10px">'
            f'{q.blurb}</div>', unsafe_allow_html=True)

        if t["tested"]:
            ret = t["total_return_pct"]
            vs = t["vs_hold_pct"]
            tone = "var(--good)" if ret > 0 else "var(--bad)"
            st.markdown(
                f'<div style="font-size:30px;font-weight:600;color:{tone}">'
                f'{ret:+.1f}%</div>'
                f'<div style="opacity:.7;font-size:13px">'
                f'{money.brief(ret, "once")} &middot; '
                f'{vs:+.1f}% vs holding</div>'
                f'<div style="opacity:.5;font-size:12px;margin-top:4px">'
                f'{t["tested"]} of {t["algos"]} tested &middot; '
                f'{t["beat_hold"]} beat buy-and-hold</div>',
                unsafe_allow_html=True)
        elif t["algos"]:
            st.markdown(
                f'<div style="font-size:30px;font-weight:600;opacity:.4">—</div>'
                f'<div style="opacity:.6;font-size:13px">'
                f'{t["algos"]} saved, none backtested yet</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="font-size:30px;font-weight:600;opacity:.3">—</div>'
                '<div style="opacity:.6;font-size:13px">nothing here yet</div>',
                unsafe_allow_html=True)

        st.caption(q.detail)
        if st.button(f"Open {q.title}", key=f"open_{key}",
                     width="stretch"):
            st.session_state["quadrant"] = key
            st.switch_page("pages/1_My_algorithms.py")


row1 = st.columns(2)
row2 = st.columns(2)
render(row1[0], SHORT_TERM)
render(row1[1], SHORT_LIVE)
render(row2[0], LONG_TERM)
render(row2[1], LONG_LIVE)

st.divider()

total_algos = sum(t["algos"] for t in totals.values())
if not total_algos:
    st.info(
        "**No algorithms saved yet.** Go to the Workshop, pick a rule, and it "
        "gets backtested the moment you save it. Everything you save shows up "
        "in one of the four quadrants above.")
    if st.button("Open the Workshop", type="primary"):
        st.switch_page("pages/2_Workshop.py")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Workshop**")
    plain("Add an algorithm. It is backtested automatically, with its "
          "expectancy, its risk of ruin and the bet size the maths supports.")
    if st.button("Add or test an algorithm", width="stretch"):
        st.switch_page("pages/2_Workshop.py")
with c2:
    st.markdown("**Live workshop**")
    plain("The everchanging side. Which tested rule is the best one to run "
          "right now, given the news and where the position already is.")
    if st.button("Open the live workshop", width="stretch"):
        st.switch_page("pages/3_Live_workshop.py")
with c3:
    st.markdown("**Research**")
    plain("The raw tools underneath: basket tests, event studies, and the "
          "scoreboard of everything ever tried here.")
    if st.button("Open research", width="stretch"):
        st.switch_page("pages/4_Research.py")

st.divider()
st.caption(
    "Every percentage on this page is backtested, not money that was made. "
    "Nothing here places a real order; the live workshop trades a practice "
    "account with fake money and refuses to run against a live one.")
