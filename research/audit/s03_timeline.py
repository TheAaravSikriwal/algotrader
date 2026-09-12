"""Step 2: build the quadrant timeline, with and without publication lag."""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s03_core as S  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 400)


def main():
    macro = S.load_macro()
    px = S.load_prices()
    mep = S.month_end_prices(px)
    print("month-end obs:", len(mep), mep.index.min().date(), "..", mep.index.max().date())
    for c in mep.columns:
        s = mep[c].dropna()
        print(f"  {c:5s} n={len(s):3d}  {s.index.min().date()} .. {s.index.max().date()}")

    lagged = S.build_quadrants(macro, mep.index, walcl_lag_days=7, ff_lag_months=1)
    nolag = S.build_quadrants(macro, mep.index, walcl_lag_days=0, ff_lag_months=0)

    print("\n=== LAGGED (primary spec) quadrant counts ===")
    print(lagged["quad"].value_counts(dropna=False))
    print("\n=== NO-LAG quadrant counts ===")
    print(nolag["quad"].value_counts(dropna=False))

    both = pd.DataFrame({"lagged": lagged["quad"], "nolag": nolag["quad"]}).dropna()
    agree = (both["lagged"] == both["nolag"]).mean()
    print(f"\nlag vs no-lag label agreement: {agree:.4f} over {len(both)} months "
          f"({int((1-agree)*len(both))} months differ)")
    diff = both[both["lagged"] != both["nolag"]]
    print("months where they differ:")
    print(diff)

    ep = S.episodes(lagged["quad"])
    print("\n=== EPISODES (contiguous runs), lagged spec ===")
    for _, r in ep.iterrows():
        print(f"  {r['quad']}  {r['start'].date()} .. {r['end'].date()}  ({r['months']} months)")
    print("\nepisode count per quadrant:")
    print(ep.groupby("quad")["months"].agg(["count", "sum"]))

    lagged.to_csv(A.RESULTS / "s03_quadrants_lagged.csv")
    nolag.to_csv(A.RESULTS / "s03_quadrants_nolag.csv")
    ep.to_csv(A.RESULTS / "s03_episodes.csv", index=False)
    mep.to_csv(A.RESULTS / "s03_month_end_prices.csv")
    print("\nwrote quadrant + episode + price CSVs")


if __name__ == "__main__":
    main()
