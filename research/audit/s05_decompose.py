"""Source 05 audit -- addendum: WHERE does the gross number come from?

The trivial 09:30->15:55 benchmark is strongly NEGATIVE on these gap days, yet
random entry in the first 90 minutes with a stop is POSITIVE. So three effects
are tangled together. This isolates them on the identical matched event set:

  A. buy 09:30 open, hold to 15:55, no stop          -- the raw gapper
  B. buy at a random time in the first 90 min, hold to 15:55, NO stop
                                                     -- effect of waiting
  C. B plus the 15-minute-low stop                   -- effect of the stop
  D. the real first-pullback rule with the same stop -- effect of the RULE

Writes results/s05_decomposition.csv, results/s05_stopwidth.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s05_core as C  # noqa: E402

N_DRAWS = 100
SEED = 20260912
KEYS = ["ma", "vwap", "fill", "stop", "target"]
BK = {"ma": "ema", "vwap": "premkt", "fill": "nextopen",
      "stop": "pullback", "target": "eod"}


def main():
    td = pd.read_parquet(A.CACHE / "s05_trades.parquet")
    real = td[~td["inverted"]]
    for k, v in BK.items():
        real = real[real[k] == v]
    ev_set = set(zip(real["date"].astype(str), real["symbol"]))
    print(f"matched events: {len(ev_set)}", flush=True)

    # stop width of the real rule, as % of entry
    sw = (real["R_dollars"] / real["entry_px"] * 100).describe(
        percentiles=[.1, .25, .5, .75, .9])
    print("\nreal rule stop distance, % of entry price:")
    print(sw.round(3).to_string(), flush=True)
    A.save(sw.to_frame("stop_pct_of_entry").reset_index(), "s05_stopwidth.csv")
    print("\nhold time, minutes:")
    ht = (pd.to_datetime(real["exit_ts"]) - pd.to_datetime(real["entry_ts"])
          ).dt.total_seconds() / 60
    print(ht.describe(percentiles=[.25, .5, .75]).round(1).to_string(), flush=True)

    ev = pd.read_csv(A.RESULTS / "s05_gap_events.csv", keep_default_na=False)
    ev["date"] = ev["date"].astype(str)
    paths = []
    for date, grp in ev.groupby("date", sort=True):
        need = [s for s in grp["symbol"] if (date, s) in ev_set]
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
            cand = np.where(np.array([t.time() <= C.dt.time(11, 0) for t in p.ts])
                            & (np.arange(p.n) >= 5))[0]
            if len(cand):
                paths.append((p, cand))
    print(f"usable paths: {len(paths)}", flush=True)

    # A: open -> 15:55, no stop
    a = [(float(p.c[p.eod_pos]) / float(p.o[0]) - 1) * 100 for p, _ in paths]

    rng = np.random.default_rng(SEED)
    b_means, c_means = [], []
    for _ in range(N_DRAWS):
        b, c = [], []
        for p, cand in paths:
            i = int(rng.choice(cand))
            entry = float(p.o[i])
            b.append((float(p.c[p.eod_pos]) / entry - 1) * 100)
            spx = float(p.l[max(0, i - 15):i].min()) - C.TICK if i >= 1 else np.nan
            if np.isfinite(spx) and (entry - spx) / entry < 0.002:
                spx = entry * 0.998
            r = C.simulate(p, {"trig_ts": p.ts[i], "pb_ts": p.ts[i]},
                           entry, i, spx, "eod")
            if r:
                c.append(r["pnl_pct"])
        b_means.append(np.mean(b)); c_means.append(np.mean(c))

    d = real["pnl_pct"]
    rows = [
        {"leg": "A open->15:55 no stop", "n": len(a), "mean_pct": float(np.mean(a)),
         "median_pct": float(np.median(a)), "win_rate": float(np.mean(np.array(a) > 0)),
         "t": A.tstat(a)},
        {"leg": "B random entry ->15:55 no stop", "n": len(paths),
         "mean_pct": float(np.mean(b_means)), "median_pct": np.nan,
         "win_rate": np.nan, "t": np.nan,
         "draw_sd": float(np.std(b_means, ddof=1))},
        {"leg": "C random entry + 15min-low stop", "n": len(paths),
         "mean_pct": float(np.mean(c_means)), "median_pct": np.nan,
         "win_rate": np.nan, "t": np.nan,
         "draw_sd": float(np.std(c_means, ddof=1))},
        {"leg": "D REAL first-pullback + stop", "n": len(d),
         "mean_pct": float(d.mean()), "median_pct": float(d.median()),
         "win_rate": float((d > 0).mean()), "t": A.tstat(d)},
    ]
    out = pd.DataFrame(rows)
    A.save(out, "s05_decomposition.csv")
    print("\n=== DECOMPOSITION (identical matched events) ===")
    print(out.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
