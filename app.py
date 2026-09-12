"""Start here.

Deliberately not a dashboard. Someone new needs to know what this is for and
what to do next, in that order -- a wall of charts answers neither question.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.env import load_env
from core.ui import explained_tile, page_header, plain, step

load_env()

st.set_page_config(page_title="Trading workbench", page_icon="~", layout="wide")


def safe(fn, default=None):
    """Read state without letting one missing file break the whole page."""
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def read_state() -> dict:
    from core.journal import Journal
    from core.overlay import ledger_frame
    from core.precedent import load_outcomes

    summary = safe(lambda: Journal().summary(), {}) or {}
    overlays = safe(ledger_frame, pd.DataFrame())
    if overlays is None or not isinstance(overlays, pd.DataFrame):
        overlays = pd.DataFrame()

    return {
        "tests": summary.get("tests_run", 0),
        "bar": summary.get("corrected_bar", 0.0),
        "found": summary.get("significant_positive", 0),
        "overlays": 0 if overlays.empty else int(overlays["overlay"].nunique()),
        "active": 0 if overlays.empty else int(overlays["active"].sum()),
        "outcomes": len(safe(load_outcomes, []) or []),
        "keys": bool(os.getenv("ALPACA_API_KEY_ID")),
    }


def main():
    page_header("Trading workbench")
    state = read_state()

    st.markdown(
        "A workshop for **testing trading ideas honestly**, before any money is "
        "involved. Most trading ideas do not work — the point of this app is to "
        "find that out cheaply, rather than by losing money.")

    st.info(
        "**The one thing to understand.** Every number here is compared against "
        "*doing nothing* — just buying and holding. An idea that made 40% sounds "
        "great until you learn the market made 60% over the same years. That "
        "comparison sits on every page, and it is the only one that matters.",
        icon="💡")

    cols = st.columns(4)
    explained_tile(cols[0], "corrected_bar", f"{state['bar']:.2f}",
                   f"from {state['tests']} ideas tested")
    explained_tile(cols[1], "total_return", f"{state['found']}",
                   "ideas that actually held up",
                   "up" if state["found"] else "flat", label="Winners found")
    explained_tile(cols[2], "precedents", f"{state['outcomes']}",
                   f"{state['active']} news views live now")
    explained_tile(cols[3], "gross_exposure",
                   "connected" if state["keys"] else "not set up",
                   "practice account", "up" if state["keys"] else "down",
                   label="Alpaca")

    st.divider()
    st.markdown("### What to do, in order")
    plain("You do not need all of these. Start at the top and stop once you have "
          "your answer.")

    left, right = st.columns(2)

    with left:
        step(1, "Test a simple idea", active=True)
        plain("One rule, one thing to buy. *Buy when the 20-day average crosses "
              "above the 50-day* — did that work? Start here if this is new.")
        st.page_link("pages/1_Test_an_idea.py", label="Test an idea  →")
        st.write("")

        step(2, "Test a whole basket")
        plain("Rank a group of things and hold the best few. Most real "
              "strategies work this way rather than on one ticker.")
        st.page_link("pages/2_Test_a_basket.py", label="Test a basket  →")
        st.write("")

        step(3, "Check a hunch against history")
        plain("*When companies do X, what usually happens next?* Answered across "
              "hundreds of past cases, not the one you remember.")
        st.page_link("pages/3_Check_a_hunch.py", label="Check a hunch  →")

    with right:
        step(4, "Turn this week's news into positions",
             done=state["overlays"] > 0)
        plain("The app gathers the week's news and hands you a block of text. "
              "Paste it to Claude, paste the answer back, and it becomes small "
              "adjustments to what you hold.")
        st.page_link("pages/4_This_weeks_news.py", label="This week's news  →")
        st.write("")

        step(5, "Trade it with fake money")
        plain("Run the whole book against a practice account. Real prices, real "
              "timing, no real money at any point.")
        st.page_link("pages/5_Practice_trading.py", label="Practice trading  →")
        st.write("")

        step(6, "See what has actually worked", done=state["tests"] > 0)
        plain("The scoreboard: every idea ever tested here and whether it "
              "survived scrutiny.")
        st.page_link("pages/6_Scoreboard.py", label="Scoreboard  →")
        st.write("")

        step(7, "Day trading (optional, and honest about it)")
        plain("Fast in-and-out trades on the practice account. **This one is "
              "not expected to make money** — it is here to find out whether "
              "orders fill in real life the way the tests assumed. Worth doing "
              "for that answer alone, but read the warning on the page first.")
        st.page_link("pages/7_Day_trading.py", label="Day trading  →")

    st.divider()

    with st.expander("Why does this app keep telling me things did not work?"):
        st.markdown(
            "Because they mostly do not, and an app that hid that would be worse "
            "than useless.\n\n"
            "Markets are competitive. If a simple rule reliably made money, "
            "thousands of well-funded people would already be running it and the "
            "profit would be gone. So the honest default answer to *does this "
            "work* is **no**, and the burden of proof sits with the idea.\n\n"
            f"So far **{state['tests']} ideas have been tested and "
            f"{state['found']} have held up.** That is not a broken app — it is "
            "what testing honestly looks like, and it has already saved you from "
            "funding several strategies that looked convincing on the surface.")

    with st.expander("I have never traded. What are all these tickers?"):
        st.markdown(
            "They are **ETFs** — one ticker that holds a whole basket, so you "
            "are never betting on a single company.\n\n"
            "| Ticker | What it is |\n|---|---|\n"
            "| **SPY** | The 500 biggest US companies. The benchmark to beat. |\n"
            "| **QQQ** | The big technology names. |\n"
            "| **IWM** | Smaller US companies. |\n"
            "| **TLT** | Long-term government bonds. Moves opposite to interest rates. |\n"
            "| **GLD** | Gold. |\n"
            "| **XLE / XLK / XLF / XLV** | One slice each: energy, tech, finance, healthcare. |\n\n"
            "**SPY is the bar.** If an idea cannot beat it, buying SPY and "
            "waiting is the better plan — and that is a perfectly respectable "
            "answer to arrive at.")

    with st.expander("What do the sliders on the left of each page do?"):
        st.markdown(
            "They change the **question being asked**, and the page redraws "
            "immediately.\n\n"
            "- **Symbols / universe** — what the strategy is allowed to buy.\n"
            "- **Dates** — which slice of history is being tested. Longer is "
            "usually more trustworthy.\n"
            "- **Rule** — the strategy itself. The controls beneath it change "
            "to match whichever rule you picked.\n"
            "- **Slippage** — the small cost of every trade. Setting it to zero "
            "is the easiest way to fool yourself.\n\n"
            "Nothing you change can break anything or spend money. Move things "
            "and watch what happens.")

    st.caption(
        f"Nothing on any page places a real order. Practice trading uses a "
        f"simulated account with fake money. · {date.today():%d %b %Y}")


if __name__ == "__main__":
    main()
