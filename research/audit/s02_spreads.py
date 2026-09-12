"""s02 STEP 1 -- measured bid/ask spreads. Everything downstream is costed with these.

Two independent estimates:

  (A) Alpaca historical NBBO quotes. 25 sessions spanning 2021-2026 (2022 bear,
      Aug-2024 vol spike, Apr-2025 tariff crash included), 7 instruments, 11
      sample windows per session of 30s each, bucketed by time of day. The
      relative quoted spread is (ask-bid)/mid in bps. Median and p90 reported.

  (B) Corwin-Schultz (2012) two-day high-low estimator on the cached 5Min bars.
      Needs no quotes endpoint, so it is a genuine cross-check rather than a
      re-derivation. It is known to be biased high on short bars; it is here to
      bound the quote estimate, not to replace it.

Keys are read from the environment and sent in headers only. Never logged.

Writes results/s02_spreads.csv (the headline table) plus
      results/s02_spread_samples.csv (every raw window) and
      results/s02_spread_corwin_schultz.csv
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMD"]

DATES = [
    "2021-03-10", "2021-06-16", "2021-11-08", "2021-12-01",
    "2022-01-24", "2022-03-08", "2022-05-09", "2022-06-13",
    "2022-09-13", "2022-10-13", "2022-12-14",
    "2023-03-13", "2023-06-14", "2023-11-14",
    "2024-02-13", "2024-04-15", "2024-08-05", "2024-11-06",
    "2025-01-27", "2025-04-04", "2025-04-08", "2025-07-15",
    "2026-02-11", "2026-06-10", "2026-08-12",
]

# (bucket label, sample time)
POINTS = [
    ("09:30-09:45", "09:30:05"), ("09:30-09:45", "09:33:00"), ("09:30-09:45", "09:40:00"),
    ("09:45-10:30", "09:50:00"), ("09:45-10:30", "10:10:00"),
    ("10:30-15:30", "11:15:00"), ("10:30-15:30", "12:45:00"), ("10:30-15:30", "14:30:00"),
    ("15:30-16:00", "15:35:00"), ("15:30-16:00", "15:50:00"), ("15:30-16:00", "15:58:00"),
]
WINDOW_S = 30
SAMPLE_CACHE = A.CACHE / "s02_quote_samples.parquet"


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY_ID"],
                                     os.environ["ALPACA_API_SECRET_KEY"])


def fetch_window(dc, date: str, tm: str, tries: int = 6):
    from alpaca.data.requests import StockQuotesRequest
    s = pd.Timestamp(f"{date} {tm}", tz="America/New_York")
    e = s + pd.Timedelta(seconds=WINDOW_S)
    for i in range(tries):
        try:
            r = dc.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=SYMBOLS, start=s.to_pydatetime(),
                end=e.to_pydatetime(), limit=10000))
            return r.df
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)[:80]
            if i == tries - 1:
                print(f"  FAIL {date} {tm}: {msg}", flush=True)
                return None
            time.sleep(min(5 * (i + 1), 30))
    return None


def summarise(df: pd.DataFrame, date: str, bucket: str, tm: str) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    d = df.reset_index()
    out = []
    for sym, g in d.groupby("symbol"):
        b = pd.to_numeric(g["bid_price"], errors="coerce")
        a = pd.to_numeric(g["ask_price"], errors="coerce")
        bs = pd.to_numeric(g.get("bid_size", pd.Series(index=g.index)), errors="coerce")
        asz = pd.to_numeric(g.get("ask_size", pd.Series(index=g.index)), errors="coerce")
        ok = (b > 0) & (a > 0) & (a >= b) & ((a / b - 1) < 0.10)
        b, a = b[ok], a[ok]
        if len(b) < 5:
            continue
        mid = (a + b) / 2.0
        rel = (a - b) / mid * 1e4
        out.append({
            "symbol": sym, "date": date, "bucket": bucket, "time": tm,
            "n_quotes": int(len(b)),
            "median_bps": float(rel.median()), "mean_bps": float(rel.mean()),
            "p90_bps": float(rel.quantile(0.90)), "p99_bps": float(rel.quantile(0.99)),
            "median_abs_spread": float((a - b).median()),
            "median_mid": float(mid.median()),
            "median_bid_size": float(bs[ok].median()) if bs.notna().any() else np.nan,
            "median_ask_size": float(asz[ok].median()) if asz.notna().any() else np.nan,
        })
    return out


def collect_quotes() -> pd.DataFrame:
    have = pd.read_parquet(SAMPLE_CACHE) if SAMPLE_CACHE.exists() else pd.DataFrame()
    done = set()
    if len(have):
        done = set(zip(have["date"], have["time"]))
    todo = [(d, b, t) for d in DATES for (b, t) in POINTS if (d, t) not in done]
    print(f"quote windows: {len(done)} cached, {len(todo)} to fetch", flush=True)
    if not todo:
        return have
    dc = _client()
    rows = []
    t0 = time.time()
    for k, (d, b, t) in enumerate(todo):
        df = fetch_window(dc, d, t)
        rows.extend(summarise(df, d, b, t))
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(todo)}  {time.time()-t0:.0f}s  rows={len(rows)}", flush=True)
            if rows:
                pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_parquet(SAMPLE_CACHE)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if rows else have
    if len(out):
        out.to_parquet(SAMPLE_CACHE)
    return out


# ---------------------------------------------------------------------------
# Corwin-Schultz fallback / cross-check on cached 5Min bars
# ---------------------------------------------------------------------------
def corwin_schultz(h: np.ndarray, l: np.ndarray) -> float:
    h = np.asarray(h, float); l = np.asarray(l, float)
    ok = (h > 0) & (l > 0)
    h, l = h[ok], l[ok]
    if len(h) < 3:
        return np.nan
    b1 = np.log(h[:-1] / l[:-1]) ** 2
    b2 = np.log(h[1:] / l[1:]) ** 2
    beta = b1 + b2
    hh = np.maximum(h[:-1], h[1:]); ll = np.minimum(l[:-1], l[1:])
    gamma = np.log(hh / ll) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    alpha = np.where(alpha < 0, 0.0, alpha)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s[np.isfinite(s)]
    return float(np.median(s) * 1e4) if len(s) else np.nan


BUCKET_EDGES = [("09:30-09:45", dt.time(9, 30), dt.time(9, 45)),
                ("09:45-10:30", dt.time(9, 45), dt.time(10, 30)),
                ("10:30-15:30", dt.time(10, 30), dt.time(15, 30)),
                ("15:30-16:00", dt.time(15, 30), dt.time(16, 0))]


def cs_table() -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        try:
            bars = A.rth(A.bars_chunked(sym, "2021-01-01", "2026-09-01", "5Min"))
        except Exception as exc:  # noqa: BLE001
            print(f"  CS {sym}: {exc}")
            continue
        day = pd.DatetimeIndex(bars.index).normalize()
        t = bars.index.time
        for label, a, b in BUCKET_EDGES:
            m = (t >= a) & (t < b)
            w = bars[m]
            wd = day[m]
            vals = []
            for _, g in w.groupby(wd):
                if len(g) >= 3:
                    v = corwin_schultz(g["high"].to_numpy(), g["low"].to_numpy())
                    if np.isfinite(v):
                        vals.append(v)
            if vals:
                vals = np.array(vals)
                rows.append({"symbol": sym, "bucket": label, "n_sessions": len(vals),
                             "cs_median_bps": float(np.median(vals)),
                             "cs_p90_bps": float(np.percentile(vals, 90))})
        # by year, whole session
        for yr, g in bars.groupby(pd.DatetimeIndex(bars.index).year):
            gd = pd.DatetimeIndex(g.index).normalize()
            vals = [corwin_schultz(x["high"].to_numpy(), x["low"].to_numpy())
                    for _, x in g.groupby(gd) if len(x) >= 3]
            vals = np.array([v for v in vals if np.isfinite(v)])
            if len(vals):
                rows.append({"symbol": sym, "bucket": f"YEAR_{yr}", "n_sessions": len(vals),
                             "cs_median_bps": float(np.median(vals)),
                             "cs_p90_bps": float(np.percentile(vals, 90))})
    return pd.DataFrame(rows)


def main():
    q = collect_quotes()
    if len(q):
        A.save(q, "s02_spread_samples.csv")
        print("\n" + "=" * 90)
        print("NBBO QUOTED SPREAD, bps of mid -- by instrument x time-of-day bucket")
        print("=" * 90)
        tab = q.groupby(["symbol", "bucket"]).agg(
            n_windows=("median_bps", "size"),
            n_quotes=("n_quotes", "sum"),
            median_bps=("median_bps", "median"),
            p90_bps=("p90_bps", "median"),
            worst_window_bps=("median_bps", "max"),
            median_abs_spread=("median_abs_spread", "median"),
            median_mid=("median_mid", "median"),
        ).reset_index()
        tab["half_spread_bps"] = tab["median_bps"] / 2.0
        print(tab.round(3).to_string(index=False))

        print("\nby YEAR (whole session):")
        q["year"] = pd.to_datetime(q["date"]).dt.year
        yr = q.groupby(["symbol", "year"])["median_bps"].median().unstack()
        print(yr.round(2).to_string())
        A.save(yr.reset_index(), "s02_spread_by_year.csv")
    else:
        tab = pd.DataFrame()
        print("NO QUOTE DATA -- Alpaca market-data API unavailable.")

    cs = cs_table()
    A.save(cs, "s02_spread_corwin_schultz.csv")
    print("\n" + "=" * 90)
    print("CORWIN-SCHULTZ high-low estimator on 5Min bars 2021-2026 (cross-check, biased HIGH)")
    print("=" * 90)
    print(cs[~cs["bucket"].str.startswith("YEAR")].round(2).to_string(index=False))
    print()
    print(cs[cs["bucket"].str.startswith("YEAR")].pivot_table(
        index="symbol", columns="bucket", values="cs_median_bps").round(2).to_string())

    if len(tab):
        A.save(tab, "s02_spreads.csv")
    else:
        A.save(cs, "s02_spreads.csv")


if __name__ == "__main__":
    main()
