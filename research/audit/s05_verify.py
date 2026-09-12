"""Source 05 audit -- STEP 5d: mechanical verification of the test itself.

  1. 1Min -> 5Min resample vs Alpaca's NATIVE 5Min bars on a sampled set of
     event-days (the backtest resamples, so this has to agree).
  2. The 09:30 bar carries the volume spike and matches the daily open.
  3. A worked example: print one full signal with every input and the exact
     timestamp at which it was knowable.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s05_core as C  # noqa: E402

CHK = A.CACHE / "s05_min5_check"


def main():
    ev = pd.read_csv(A.RESULTS / "s05_gap_events.csv", keep_default_na=False)
    ev["date"] = ev["date"].astype(str)
    files = sorted(CHK.glob("*.parquet"))
    print(f"native-5Min audit files: {len(files)}", flush=True)

    diffs, opens, vols = [], [], []
    for f in files:
        date = f.stem
        nat = pd.read_parquet(f)
        if len(nat) == 0:
            continue
        mins = C.load_minute(date)
        for sym, sub in nat.groupby("symbol"):
            sym = str(sym)
            m1 = mins.get(sym)
            if m1 is None or len(m1) == 0:
                continue
            n5 = A.to_eastern(sub.set_index("timestamp")
                              [["open", "high", "low", "close", "volume"]].sort_index())
            r5 = C.to5(m1)
            j = n5.index.intersection(r5.index)
            j = [t for t in j if C.dt.time(9, 30) <= t.time() < C.dt.time(16, 0)]
            if not j:
                continue
            for col in ("open", "high", "low", "close"):
                d = (r5.loc[j, col] - n5.loc[j, col]).abs() / n5.loc[j, col]
                diffs.append(pd.DataFrame({"col": col, "reldiff": d.to_numpy()}))
            # daily open check
            drow = ev[(ev["date"] == date) & (ev["symbol"] == sym)]
            first = r5[r5.index.time == C.dt.time(9, 30)]
            if len(drow) and len(first):
                o930 = float(first["open"].iloc[0])
                opens.append(abs(o930 / float(drow["open"].iloc[0]) - 1.0))
            rth5 = r5[(r5.index.time >= C.dt.time(9, 30))
                      & (r5.index.time < C.dt.time(16, 0))]
            if len(rth5) > 10:
                vols.append(float(rth5["volume"].iloc[0]) / float(rth5["volume"].median()))

    dd = pd.concat(diffs, ignore_index=True)
    print("\n1. resampled-5Min vs native-5Min, |rel diff| by field:")
    print(dd.groupby("col")["reldiff"]
          .agg(["count", "mean", "median", "max"]).round(8).to_string(), flush=True)
    print(f"   share of bars agreeing to <1e-6: "
          f"{(dd['reldiff'] < 1e-6).mean():.4f}", flush=True)

    print(f"\n2a. 09:30 5Min open vs Alpaca DAILY open, n={len(opens)}: "
          f"median |rel diff| = {np.median(opens):.2e}, "
          f"share <1e-6 = {np.mean(np.array(opens) < 1e-6):.4f}", flush=True)
    v = np.array(vols)
    print(f"2b. 09:30 bar volume / median RTH 5Min bar volume, n={len(v)}: "
          f"median x{np.median(v):.2f}, share>2x = {np.mean(v > 2):.3f}", flush=True)

    # 3. worked example
    print("\n3. WORKED EXAMPLE (no-lookahead ledger)", flush=True)
    shown = 0
    for date, grp in ev.groupby("date", sort=True):
        mins = C.load_minute(date)
        for _, e in grp.iterrows():
            m1 = mins.get(e["symbol"])
            if m1 is None or len(m1) < 200:
                continue
            d5 = C.indicators(C.to5(m1), True, "ema")
            sig = C.find_entry(d5)
            if sig is None:
                continue
            p = C.Path(m1)
            ef = C.entry_fill(p, sig, "trigger")
            if ef is None:
                continue
            entry, start = ef
            r = C.simulate(p, sig, entry, start, sig["pullback_low"] - C.TICK, "2R")
            if r is None:
                continue
            print(f"\n  {e['symbol']} {date} gap={e['gap']*100:.1f}% "
                  f"prev_close=${e['prev_close']:.2f}")
            w = d5[(d5.index.time >= C.dt.time(9, 30)) & (d5.index.time <= C.dt.time(11, 0))]
            print(w[["open", "high", "low", "close", "vwap", "ma9"]]
                  .head(20).round(3).to_string())
            print(f"    push bar     : {w.index[sig['push_i']]}  (new session high)")
            print(f"    pullback bar : {sig['pb_ts']}  low<=max(vwap={sig['vwap_at_pb']:.3f},"
                  f" ma9={sig['ma9_at_pb']:.3f})*1.001 -- known at that bar's CLOSE")
            print(f"    prior bar    : {sig['prev_ts']}  high={d5.loc[sig['prev_ts'],'high']:.3f}"
                  f" -> stop-buy resting at {sig['trigger_px']:.3f} placed at its close")
            print(f"    trigger bar  : {sig['trig_ts']}  filled {entry:.3f}")
            print(f"    stop         : {sig['pullback_low']-C.TICK:.3f} "
                  f"(R=${r['R_dollars']:.3f})")
            print(f"    exit         : {r['exit_ts']} @ {r['exit_px']:.3f} "
                  f"({r['exit_reason']}) pnl={r['pnl_pct']:+.2f}% R={r['R_mult']:+.2f}",
                  flush=True)
            shown += 1
            break
        if shown >= 3:
            break


if __name__ == "__main__":
    main()
