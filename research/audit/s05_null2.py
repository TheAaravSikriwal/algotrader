"""Source 05 audit -- STEP 5b, stricter null.

The uniform 09:35-11:00 null in s05_null.py can enter EARLIER than the real
rule's trigger on a day the rule eventually traded, which is a small
informational advantage for the null. This removes it: random entry uniformly
among the 1-minute bars at or after the real rule's own trigger timestamp
(capped at 12:00), same stop rule, same exit. If the real rule still cannot
beat this, the "first candle makes a new high" trigger is worthless.

Writes results/s05_null_conditional.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s05_core as C  # noqa: E402

N_DRAWS = 200
SEED = 20260913
KEYS = ["ma", "vwap", "fill", "stop", "target"]
BK = {"ma": "ema", "vwap": "premkt", "fill": "nextopen",
      "stop": "pullback", "target": "eod"}


def main():
    td = pd.read_parquet(A.CACHE / "s05_trades.parquet")
    real = td[~td["inverted"]]
    for k, v in BK.items():
        real = real[real[k] == v]
    trig = {(str(r["date"])[:10], r["symbol"]): pd.Timestamp(r["entry_ts"])
            for _, r in real.iterrows()}
    real_mean = float(real["pnl_pct"].mean())
    print(f"real: n={len(real)} mean={real_mean:.4f}%", flush=True)

    ev = pd.read_csv(A.RESULTS / "s05_gap_events.csv", keep_default_na=False)
    ev["date"] = ev["date"].astype(str)
    paths = []
    for date, grp in ev.groupby("date", sort=True):
        need = [s for s in grp["symbol"] if (date, s) in trig]
        if not need:
            continue
        mins = C.load_minute(date)
        for sym in need:
            m1 = mins.get(sym)
            if m1 is None or len(m1) == 0:
                continue
            p = C.Path(m1)
            if p.n < 60:
                continue
            t0 = trig[(date, sym)]
            cand = np.where((p.ts >= t0)
                            & np.array([t.time() <= C.dt.time(12, 0) for t in p.ts]))[0]
            cand = cand[cand < p.eod_pos]
            if len(cand):
                paths.append((p, cand))
    print(f"usable paths: {len(paths)}", flush=True)

    rng = np.random.default_rng(SEED)
    draws = []
    for d in range(N_DRAWS):
        pn = []
        for p, cand in paths:
            i = int(rng.choice(cand))
            entry = float(p.o[i])
            spx = float(p.l[max(0, i - 15):i].min()) - C.TICK if i >= 1 else np.nan
            if np.isfinite(spx) and (entry - spx) / entry < 0.002:
                spx = entry * 0.998
            r = C.simulate(p, {"trig_ts": p.ts[i], "pb_ts": p.ts[i]},
                           entry, i, spx, "eod")
            if r:
                pn.append(r["pnl_pct"])
        if pn:
            draws.append({"draw": d, "n": len(pn), "mean_pct": float(np.mean(pn)),
                          "win_rate": float(np.mean(np.array(pn) > 0)),
                          "t": A.tstat(pn)})
    dd = pd.DataFrame(draws)
    A.save(dd, "s05_null_conditional.csv")
    pct = float((dd["mean_pct"] < real_mean).mean() * 100)
    print("\n=== CONDITIONAL NULL (random entry at-or-after the real trigger) ===")
    print(dd[["mean_pct", "win_rate", "t"]]
          .describe(percentiles=[.05, .5, .95]).round(4).to_string(), flush=True)
    print(f"\nreal mean = {real_mean:+.4f}%  -> percentile {pct:.1f} of this null")
    print(f"z vs null = "
          f"{(real_mean - dd['mean_pct'].mean()) / dd['mean_pct'].std(ddof=1):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
