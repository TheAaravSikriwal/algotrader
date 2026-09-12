"""Source 05 audit -- STEP 5b: the random-entry null.

The correct null for "first pullback to VWAP/9MA" is NOT zero. It is: be long
the same gapper, on the same day, with the same stop and target, entered at a
uniformly random time in the first 90 minutes. If the real rule cannot beat that
distribution, the pullback rule adds nothing beyond "own a gapper intraday".

200 draws. Each draw re-runs the entire trade population with a random entry bar
per event and the SAME exit machinery from s05_core.

Writes results/s05_null_random.csv, results/s05_null_summary.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s05_core as C  # noqa: E402

N_DRAWS = 200
SEED = 20260911
WINDOW_END = C.dt.time(11, 0)     # "first 90 minutes" 09:30 -> 11:00
KEYS = ["ma", "vwap", "fill", "stop", "target"]


def main():
    grid = pd.read_csv(A.RESULTS / "s05_grid_summary.csv")
    best = grid.sort_values("mean_pct", ascending=False).iloc[0]
    bk = {k: best[k] for k in KEYS}
    print("null is matched to the best real variant:", bk, flush=True)

    td = pd.read_parquet(A.CACHE / "s05_trades.parquet")
    real = td[(~td["inverted"])]
    for k, v in bk.items():
        real = real[real[k] == v]
    real_events = set(zip(real["date"].astype(str), real["symbol"]))
    real_mean = float(real["pnl_pct"].mean())
    real_R = float(real["R_mult"].mean())
    print(f"real: n={len(real)} mean={real_mean:.4f}% avgR={real_R:.4f}", flush=True)

    # collect every event's 1Min path once, restricted to the SAME events the
    # real rule actually traded (matched sample -- no day-selection advantage)
    ev = pd.read_csv(A.RESULTS / "s05_gap_events.csv", keep_default_na=False)
    ev["date"] = ev["date"].astype(str)
    paths = []
    t0 = time.time()
    for date, grp in ev.groupby("date", sort=True):
        need = [s for s in grp["symbol"] if (date, s) in real_events]
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
            cand = np.where(np.array([t.time() <= WINDOW_END for t in p.ts])
                            & (np.arange(p.n) >= 5))[0]
            if len(cand) == 0:
                continue
            m5 = C.to5(m1)
            d5 = C.indicators(m5, bk["vwap"] == "premkt", bk["ma"])
            paths.append((date, sym, p, cand, d5))
    print(f"matched events with usable paths: {len(paths)} ({time.time()-t0:.0f}s)",
          flush=True)

    rng = np.random.default_rng(SEED)
    stop_mode, tgt = bk["stop"], bk["target"]
    draws = []
    for d in range(N_DRAWS):
        pn, rm = [], []
        for date, sym, p, cand, d5 in paths:
            i = int(rng.choice(cand))
            entry = float(p.o[i])
            if stop_mode == "c20":
                spx = entry - 0.20
            elif stop_mode == "atr":
                a = C.atr5(d5, p.ts[i])
                spx = entry - a if np.isfinite(a) else np.nan
            else:   # "pullback low" has no meaning for a random entry. The
                    # matched analogue is the lowest low of the prior 15 minutes
                    # -- the same span a 5Min pullback low covers. (A 3-minute
                    # low makes R degenerate and blows up the R-multiple.)
                spx = float(p.l[max(0, i - 15):i].min()) - C.TICK if i >= 1 else np.nan
            if np.isfinite(spx) and (entry - spx) / entry < 0.002:
                spx = entry * 0.998   # floor R at 20bps so R-multiples stay finite
            sig = {"trig_ts": p.ts[i], "pb_ts": p.ts[i]}
            r = C.simulate(p, sig, entry, i, spx, tgt)
            if r is None:
                continue
            pn.append(r["pnl_pct"]); rm.append(r["R_mult"])
        if pn:
            draws.append({"draw": d, "n": len(pn), "mean_pct": float(np.mean(pn)),
                          "median_pct": float(np.median(pn)),
                          "win_rate": float(np.mean(np.array(pn) > 0)),
                          "avg_R": float(np.nanmean(rm)),
                          "t": A.tstat(pn)})
        if d % 25 == 0:
            print(f"  draw {d} ({time.time()-t0:.0f}s)", flush=True)
    dd = pd.DataFrame(draws)
    A.save(dd, "s05_null_random.csv")

    pct = float((dd["mean_pct"] < real_mean).mean() * 100)
    pctR = float((dd["avg_R"] < real_R).mean() * 100)
    print("\n=== RANDOM-ENTRY NULL, {} draws ===".format(len(dd)))
    print(dd[["mean_pct", "avg_R", "win_rate", "t"]]
          .describe(percentiles=[.05, .25, .5, .75, .95]).round(4).to_string(),
          flush=True)
    print(f"\nreal mean per trade  = {real_mean:+.4f}%   -> percentile {pct:.1f} "
          f"of the random-entry null")
    print(f"real avg R           = {real_R:+.4f}    -> percentile {pctR:.1f}",
          flush=True)
    summ = pd.DataFrame([{
        "variant": "|".join(str(bk[k]) for k in KEYS),
        "real_n": len(real), "real_mean_pct": real_mean, "real_avg_R": real_R,
        "null_draws": len(dd), "null_mean_of_means": dd["mean_pct"].mean(),
        "null_sd_of_means": dd["mean_pct"].std(ddof=1),
        "null_p05": dd["mean_pct"].quantile(.05),
        "null_p95": dd["mean_pct"].quantile(.95),
        "real_percentile_mean": pct, "real_percentile_R": pctR,
        "z_vs_null": (real_mean - dd["mean_pct"].mean()) / dd["mean_pct"].std(ddof=1),
    }])
    A.save(summ, "s05_null_summary.csv")
    print(summ.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
