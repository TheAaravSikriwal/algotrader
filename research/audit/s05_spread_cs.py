"""Source 05 audit -- STEP 4 fallback: spread estimate WITHOUT the quotes API.

Alpaca's historical quotes endpoint returned {"message":"backend request
timeout"} on every request during this audit (it answered once at the start of
the session, then stopped), so the empirical spread is estimated from bars:

  (a) Corwin-Schultz (2012) two-period high-low estimator on 1Min bars, run
      both in the first 30 minutes (where the strategy trades) and midday
      12:00-14:00 (calmer -- CS is biased UP by volatility, so midday is the
      more conservative read and 09:30-10:00 is the upper bound).
  (b) A hard mechanical floor: US equities tick in $0.01, so the spread on a
      $2-$20 name cannot be tighter than 1 cent = 10000/price bps.

Writes results/s05_spread_cs.csv, results/s05_spread_summary.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s05_core as C  # noqa: E402
from s05_spreads import corwin_schultz  # noqa: E402


def main():
    ev = pd.read_csv(A.RESULTS / "s05_gap_events.csv", keep_default_na=False)
    ev["date"] = ev["date"].astype(str)
    rows = []
    for date, grp in ev.groupby("date", sort=True):
        mins = C.load_minute(date)
        for _, e in grp.iterrows():
            m1 = mins.get(e["symbol"])
            if m1 is None or len(m1) == 0:
                continue
            w0 = m1[(m1.index.time >= C.dt.time(9, 30)) & (m1.index.time < C.dt.time(10, 0))]
            wm = m1[(m1.index.time >= C.dt.time(12, 0)) & (m1.index.time < C.dt.time(14, 0))]
            if len(w0) < 10:
                continue
            px = float(w0["close"].median())
            rows.append({
                "date": date, "symbol": e["symbol"], "gap": e["gap"],
                "prev_close": e["prev_close"], "px_open30": px,
                "cs_open30_bps": corwin_schultz(w0["high"].to_numpy(), w0["low"].to_numpy()),
                "cs_midday_bps": (corwin_schultz(wm["high"].to_numpy(), wm["low"].to_numpy())
                                  if len(wm) >= 10 else np.nan),
                "tick_floor_bps": 100.0 / px,   # $0.01 / px in bps
            })
    cd = pd.DataFrame(rows)
    A.save(cd, "s05_spread_cs.csv")
    print(f"events with usable 1Min data: {len(cd)}", flush=True)
    for col in ("cs_open30_bps", "cs_midday_bps", "tick_floor_bps"):
        print(f"\n{col}:")
        print(cd[col].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string(),
              flush=True)

    cd["pxbin"] = pd.cut(cd["px_open30"], [0, 3, 5, 10, 20, 1e9],
                         labels=["<$3", "$3-5", "$5-10", "$10-20", ">$20"])
    print("\nby price bucket (median bps):")
    print(cd.groupby("pxbin", observed=True)[
        ["cs_open30_bps", "cs_midday_bps", "tick_floor_bps"]]
        .median().round(1).to_string(), flush=True)

    summ = pd.DataFrame([{
        "n_events": len(cd),
        "cs_open30_median_bps": cd["cs_open30_bps"].median(),
        "cs_midday_median_bps": cd["cs_midday_bps"].median(),
        "tick_floor_median_bps": cd["tick_floor_bps"].median(),
        "chosen_half_spread_bps": cd["cs_midday_bps"].median() / 2,
        "chosen_round_trip_bps": cd["cs_midday_bps"].median(),
        "note": "quotes API unavailable (backend request timeout); CS fallback",
    }])
    A.save(summ, "s05_spread_summary.csv")
    print("\n", summ.round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
