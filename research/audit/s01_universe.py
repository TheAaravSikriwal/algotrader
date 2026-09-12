"""Source 01 audit -- the tradable universe, screened as of the backtest start.

core.universe.screen documents the rule: screen on TRADABILITY, never on past
returns, and screen `as_of` the backtest start so realised volatility and daily
range are not future quantities relative to the test that follows.

The repo's `short_term` profile is the matched one -- its rationale is exactly
the cost argument the source itself raises ("trading fees can wipe out small
profits") -- but its top_n=15 is too narrow to rank a cross-section into
deciles. This builds a Profile with the same thresholds and a wider top_n, and
reports how many names clear the thresholds at all.
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402,F401  (puts the repo root on sys.path)

from core.universe import PROFILES, Profile, profile_frame, screen  # noqa: E402
from core.taxonomy import asset_type  # noqa: E402

BT_START = "2015-01-01"
AS_OF = "2014-12-31"

WIDE = Profile(
    name="s01_swing",
    rationale=("Source 01 says 'liquid stocks' held 2-5 days. Same thresholds "
               "as the repo short_term profile (costs bind on a 2-5 day hold), "
               "widened to top_n=80 so a cross-section can be ranked into "
               "deciles, and restricted to single names because the source "
               "says stocks."),
    min_price=10.0, min_dollar_volume=2e8, max_range_pct=0.035,
    min_history_days=500, min_completeness=0.95, top_n=80)


def build(bars: dict[str, pd.DataFrame], as_of: str = AS_OF,
          companies_only: bool = True, prof: Profile | None = None):
    frame = profile_frame(bars, as_of=as_of)
    if companies_only:
        frame = frame[frame["asset_type"] == "company"]
    picked = screen(frame, prof or WIDE)
    return frame, picked


if __name__ == "__main__":
    import s01_fetch

    bars = s01_fetch.load_all(verbose=False)
    frame, picked = build(bars)
    print(f"pool fetched: {len(bars)} symbols")
    print(f"single-name (non-ETF) candidates: {len(frame)}")
    print(f"pass s01_swing thresholds as of {AS_OF}: {len(picked)}")
    print(sorted(picked.index.tolist()))
    print()
    print("strict repo short_term profile (top_n=15):")
    strict = screen(profile_frame(bars, as_of=AS_OF), PROFILES["short_term"])
    print(sorted(strict.index.tolist()))
    print()
    print(picked[["price", "dollar_volume", "range_pct", "ann_vol",
                  "history_days", "completeness"]].round(4).to_string())
