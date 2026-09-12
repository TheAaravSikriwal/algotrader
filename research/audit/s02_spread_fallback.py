"""s02 STEP 1b -- spread estimates that need NO quotes endpoint.

Three independent, bar-only estimates, all on the cached 5Min consolidated bars
2021-01-01..2026-09-01. They exist because (a) the quotes endpoint may be down
and (b) an audit should not rest a cost assumption on one source.

  1. TICK FLOOR. US equities above $1 quote in $0.01 increments, so the quoted
     spread cannot be tighter than $0.01. One tick as a fraction of the measured
     median price is a HARD LOWER BOUND on the quoted spread in bps, computed
     from prices this audit actually observed.
  2. CORWIN-SCHULTZ (2012) two-period high-low estimator. Known to be biased
     high on short bars; use as an upper bound.
  3. ABDI-RANALDO (2017) close-high-low estimator, which is better behaved than
     CS when the mid drifts. S = sqrt(max(4*E[(c-eta_t)(c-eta_{t+1})], 0)).

Writes results/s02_spread_bar_estimators.csv
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMD", "MSFT", "AMZN", "META"]
BUCKETS = [("09:30-09:45", dt.time(9, 30), dt.time(9, 45)),
           ("09:45-10:30", dt.time(9, 45), dt.time(10, 30)),
           ("10:30-15:30", dt.time(10, 30), dt.time(15, 30)),
           ("15:30-16:00", dt.time(15, 30), dt.time(16, 0))]


def corwin_schultz(h, l) -> float:
    h = np.asarray(h, float); l = np.asarray(l, float)
    ok = (h > 0) & (l > 0)
    h, l = h[ok], l[ok]
    if len(h) < 3:
        return np.nan
    beta = np.log(h[:-1] / l[:-1]) ** 2 + np.log(h[1:] / l[1:]) ** 2
    gamma = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(l[:-1], l[1:])) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = np.clip(alpha, 0.0, None)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s[np.isfinite(s)]
    return float(np.median(s) * 1e4) if len(s) else np.nan


def abdi_ranaldo(c, h, l) -> float:
    c = np.log(np.asarray(c, float)); h = np.log(np.asarray(h, float))
    l = np.log(np.asarray(l, float))
    if len(c) < 3:
        return np.nan
    eta = (h + l) / 2.0
    x = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(np.sqrt(max(4.0 * x.mean(), 0.0)) * 1e4)


def main():
    rows = []
    for sym in SYMBOLS:
        bars = A.rth(A.bars_chunked(sym, "2021-01-01", "2026-09-01", "5Min"))
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        day = pd.DatetimeIndex(bars.index).normalize()
        tt = bars.index.time
        print(f"{sym}: {len(bars)} bars, {len(set(day))} sessions", flush=True)
        for label, a, b in BUCKETS:
            m = (tt >= a) & (tt < b)
            w, wd = bars[m], day[m]
            if not len(w):
                continue
            cs, ar = [], []
            for _, g in w.groupby(wd):
                if len(g) >= 3:
                    cs.append(corwin_schultz(g["high"], g["low"]))
                    ar.append(abdi_ranaldo(g["close"], g["high"], g["low"]))
            cs = np.array([v for v in cs if np.isfinite(v)])
            ar = np.array([v for v in ar if np.isfinite(v)])
            px = float(w["close"].median())
            rows.append({
                "symbol": sym, "bucket": label, "n_sessions": len(cs),
                "median_price": px,
                "tick_floor_bps": 0.01 / px * 1e4,
                "cs_median_bps": float(np.median(cs)) if len(cs) else np.nan,
                "cs_p90_bps": float(np.percentile(cs, 90)) if len(cs) else np.nan,
                "ar_median_bps": float(np.median(ar)) if len(ar) else np.nan,
                "ar_p90_bps": float(np.percentile(ar, 90)) if len(ar) else np.nan,
                "median_5min_range_bps": float(((w["high"] / w["low"] - 1) * 1e4).median()),
            })
        for yr, g in bars.groupby(pd.DatetimeIndex(bars.index).year):
            gd = pd.DatetimeIndex(g.index).normalize()
            cs = np.array([v for v in (corwin_schultz(x["high"], x["low"])
                                       for _, x in g.groupby(gd)) if np.isfinite(v)])
            ar = np.array([v for v in (abdi_ranaldo(x["close"], x["high"], x["low"])
                                       for _, x in g.groupby(gd)) if np.isfinite(v)])
            px = float(g["close"].median())
            rows.append({"symbol": sym, "bucket": f"YEAR_{yr}", "n_sessions": len(cs),
                         "median_price": px, "tick_floor_bps": 0.01 / px * 1e4,
                         "cs_median_bps": float(np.median(cs)) if len(cs) else np.nan,
                         "cs_p90_bps": float(np.percentile(cs, 90)) if len(cs) else np.nan,
                         "ar_median_bps": float(np.median(ar)) if len(ar) else np.nan,
                         "ar_p90_bps": float(np.percentile(ar, 90)) if len(ar) else np.nan,
                         "median_5min_range_bps": float(((g["high"] / g["low"] - 1) * 1e4).median())})
    df = pd.DataFrame(rows)
    A.save(df, "s02_spread_bar_estimators.csv")
    pd.set_option("display.width", 250)
    print("\nBY TIME OF DAY")
    print(df[~df["bucket"].str.startswith("YEAR")].round(3).to_string(index=False))
    print("\nBY YEAR (whole session)")
    print(df[df["bucket"].str.startswith("YEAR")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
