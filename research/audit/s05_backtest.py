"""Source 05 audit -- STEP 2/3: run the full first-pullback parameter grid.

Grid (declared up front, 120 combinations):
    ma_kind  : sma9 | ema9                          (source says only "9 MA")
    vwap     : rth  | premkt                        (VWAP anchor)
    fill     : trigger | nextopen                   (stop-buy vs repo convention)
    stop     : pullback low | fixed 20c | 1x ATR14  (5Min ATR)
    target   : 1R | 2R | 3R | eod | trail
  2 * 2 * 2 * 3 * 5 = 120

Also computes, per event, the trivial 09:30-open -> 15:55-close benchmark and
the deliberately-inverted control entry.

Writes: cache/s05_trades.parquet, results/s05_benchmarks.csv,
        results/s05_coverage.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s05_core as C  # noqa: E402

MAS = ["sma", "ema"]
VWAPS = [("rth", False), ("premkt", True)]
FILLS = ["trigger", "nextopen"]
STOPS = ["pullback", "c20", "atr"]
TARGETS = ["1R", "2R", "3R", "eod", "trail"]
N_COMBOS = len(MAS) * len(VWAPS) * len(FILLS) * len(STOPS) * len(TARGETS)


def main():
    ev = pd.read_csv(A.RESULTS / "s05_gap_events.csv", keep_default_na=False)
    ev["date"] = ev["date"].astype(str)
    print(f"events={len(ev)}  grid combinations={N_COMBOS}", flush=True)

    trades, bench, cover = [], [], []
    t0 = time.time()
    for di, (date, grp) in enumerate(ev.groupby("date", sort=True)):
        mins = C.load_minute(date)
        for _, e in grp.iterrows():
            sym = e["symbol"]
            m1 = mins.get(sym)
            rec = {"date": date, "symbol": sym, "gap": e["gap"],
                   "prev_close": e["prev_close"], "rank": e["rank"]}
            if m1 is None or len(m1) == 0:
                cover.append({**rec, "status": "no_minute_data"})
                continue
            m1r = m1[(m1.index.time >= C.dt.time(9, 30))
                     & (m1.index.time < C.dt.time(16, 0))]
            if len(m1r) < 60 or m1r.index[0].time() != C.dt.time(9, 30):
                cover.append({**rec, "status": "thin_session"})
                continue
            cover.append({**rec, "status": "ok", "rth_minutes": len(m1r)})

            # trivial benchmark: buy the 09:30 open, sell the 15:55 close
            p = C.Path(m1)
            o0 = float(p.o[0]); cE = float(p.c[p.eod_pos])
            bench.append({**rec, "open_px": o0, "close1555": cE,
                          "pnl_pct": (cE / o0 - 1.0) * 100.0,
                          "hi_pct": (float(p.h.max()) / o0 - 1.0) * 100.0,
                          "lo_pct": (float(p.l.min()) / o0 - 1.0) * 100.0})

            m5full = C.to5(m1)
            for ma in MAS:
                for vtag, vpm in VWAPS:
                    d5 = C.indicators(m5full, vpm, ma)
                    for inv in (False, True):
                        sig = C.find_entry(d5, invert=inv)
                        if sig is None:
                            continue
                        a = C.atr5(d5, sig["prev_ts"])
                        for fill in FILLS:
                            ef = C.entry_fill(p, sig, fill)
                            if ef is None:
                                continue
                            entry, start = ef
                            stops = {"pullback": sig["pullback_low"] - C.TICK,
                                     "c20": entry - 0.20,
                                     "atr": entry - a if np.isfinite(a) else np.nan}
                            for st, spx in stops.items():
                                for tg in TARGETS:
                                    r = C.simulate(p, sig, entry, start, spx, tg)
                                    if r is None:
                                        continue
                                    trades.append({
                                        **rec, "ma": ma, "vwap": vtag, "fill": fill,
                                        "stop": st, "target": tg,
                                        "inverted": inv,
                                        "entry_ts": str(sig["trig_ts"]),
                                        "entry_px": r["entry_px"],
                                        "exit_px": r["exit_px"],
                                        "exit_ts": str(r["exit_ts"]),
                                        "exit_reason": r["exit_reason"],
                                        "R_dollars": r["R_dollars"],
                                        "pnl_pct": r["pnl_pct"],
                                        "R_mult": r["R_mult"]})
        if di % 100 == 0:
            print(f"  {di} dates, {len(trades):,} trade rows ({time.time() - t0:.0f}s)",
                  flush=True)

    td = pd.DataFrame(trades)
    td.to_parquet(A.CACHE / "s05_trades.parquet", index=False)
    A.save(pd.DataFrame(bench), "s05_benchmarks.csv")
    A.save(pd.DataFrame(cover), "s05_coverage.csv")
    print(f"\ntrade rows={len(td):,}  benchmark events={len(bench):,} "
          f"({time.time() - t0:.0f}s)", flush=True)
    cv = pd.DataFrame(cover)["status"].value_counts()
    print(cv.to_string(), flush=True)


if __name__ == "__main__":
    main()
