"""Research: the raw tools underneath the workbench.

Four things that do not belong on the main path but should not be buried
either -- basket tests, event studies, the news pipeline, and the scoreboard
of everything ever tried here.

These are the pages that used to sit in the sidebar individually. They are
tabs now because the top-level choice should be *which quadrant*, not *which
of seven tools*, and because most of a session should never need to come
here at all.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.env import load_env
from core.ui import active_mode, inject_css, page_header, plain

load_env()
st.set_page_config(page_title="Research", layout="wide")
mode = active_mode()
inject_css(mode)

page_header(
    "Research",
    "The raw tools. Most sessions never need these.")

TOOLS = {
    "Basket test": (
        "_research/2_Test_a_basket.py",
        "Rank a group of symbols and hold the best few. Most real strategies "
        "work this way rather than on a single ticker."),
    "Check a hunch": (
        "_research/3_Check_a_hunch.py",
        "When companies do X, what usually happens next? Answered across "
        "hundreds of past cases rather than the one you remember."),
    "News": (
        "_research/4_This_weeks_news.py",
        "Gather the week's news into a block you can hand to Claude, and turn "
        "the answer into recorded, expiring weight adjustments."),
    "Practice book": (
        "_research/5_Practice_trading.py",
        "Run a whole long-term book against the practice account."),
    "Scoreboard": (
        "_research/6_Scoreboard.py",
        "Every idea ever tested here and whether it survived scrutiny."),
}

choice = st.radio("Tool", list(TOOLS), horizontal=True,
                  label_visibility="collapsed")
path, blurb = TOOLS[choice]
plain(blurb)
st.divider()

target = REPO / "pages" / path
if not target.exists():
    st.error(f"Missing tool file: {path}")
    st.stop()

# Each tool is a standalone Streamlit script. Running it in place keeps one
# copy of the code rather than a fork that slowly drifts from the original.
try:
    # run_name must be "__main__": these scripts guard their body with
    # `if __name__ == "__main__":`, so any other name executes the imports and
    # renders nothing at all -- a blank page with no error.
    runpy.run_path(str(target), run_name="__main__")
except SystemExit:
    pass
except Exception as exc:  # pragma: no cover - surfaced to the user
    st.error(f"{choice} failed to load: {type(exc).__name__}: {exc}")
    st.exception(exc)
