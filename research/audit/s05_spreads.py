"""Source 05 audit -- STEP 4: what does it actually cost to trade these names?

Two independent estimates of the bid/ask spread on $2-$20 gappers:
  (1) Alpaca historical NBBO quotes, sampled in the exact minute of a random
      subset of the strategy's own entries and exits.
  (2) Corwin-Schultz (2012) high-low estimator on the 1Min bars of every event,
      as a cross-check that does not depend on the quotes endpoint.

Writes results/s05_spread_quotes.csv, results/s05_spread_cs.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402
import s05_core as C  # noqa: E402

KEY = os.getenv("ALPACA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY")
N_EVENTS = 60
SEED = 20260911


def sample_quotes(dc, sym: str, ts_et: pd.Timestamp, limit: int = 400):
    from alpaca.data.requests import StockQuotesRequest
    s = pd.Timestamp(ts_et, tz="America/New_York")
    e = s + pd.Timedelta(minutes=1)
    for attempt in range(3):
        try:
            r = dc.get_stock_quotes(StockQuotesRequest(
                symbol_or_symbols=sym, start=s.to_pydatetime(),
                end=e.to_pydatetime(), limit=limit))
            df = r.df
            break
        except Exception as exc:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
            df = None
    if df is None or len(df) == 0:
        return None
    df = df.reset_index()
    b = pd.to_numeric(df["bid_price"], errors="coerce")
    a = pd.to_numeric(df["ask_price"], errors="coerce")
    ok = (b > 0) & (a > 0) & (a >= b)
    b, a = b[ok], a[ok]
    if len(b) == 0:
        return None
    mid = (a + b) / 2.0
    rel = (a - b) / mid * 10000.0
    return {"symbol": sym, "ts": str(ts_et), "n_quotes": int(len(b)),
            "median_bps": float(rel.median()), "mean_bps": float(rel.mean()),
            "p25_bps": float(rel.quantile(.25)), "p75_bps": float(rel.quantile(.75)),
            "median_mid": float(mid.median()),
            "median_abs_spread": float((a - b).median())}


def corwin_schultz(h: np.ndarray, l: np.ndarray) -> float:
    """2-period high-low spread estimator, returned in bps of price."""
    if len(h) < 3:
        return np.nan
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
    if len(s) == 0:
        return np.nan
    return float(np.median(s) * 10000.0)


def main():
    td = pd.read_parquet(A.CACHE / "s05_trades.parquet")
    base = td[(~td["inverted"]) & (td["ma"] == "ema") & (td["vwap"] == "premkt")
              & (td["fill"] == "trigger") & (td["stop"] == "pullback")
              & (td["target"] == "2R")]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(base), size=min(N_EVENTS, len(base)), replace=False)
    sub = base.iloc[sorted(idx)]
    print(f"sampling quotes for {len(sub)} entries + exits", flush=True)

    from alpaca.data.historical import StockHistoricalDataClient
    dc = StockHistoricalDataClient(KEY, SEC)

    rows = []
    for _, r in sub.iterrows():
        for tag, ts in (("entry", r["entry_ts"]), ("exit", r["exit_ts"])):
            q = sample_quotes(dc, r["symbol"], pd.Timestamp(ts))
            if q:
                q.update({"leg": tag, "date": r["date"], "prev_close": r["prev_close"],
                          "gap": r["gap"]})
                rows.append(q)
    qd = pd.DataFrame(rows)
    if len(qd):
        print("\nNBBO relative spread, bps of mid:")
        print(qd["median_bps"].describe(percentiles=[.25, .5, .75, .9])
              .round(1).to_string(), flush=True)
        print("\nby leg:")
        print(qd.groupby("leg")["median_bps"].describe().round(1).to_string(), flush=True)
        print("\nmedian absolute spread ($):",
              round(float(qd["median_abs_spread"].median()), 4), flush=True)
    A.save(qd, "s05_spread_quotes.csv")

    # --- Corwin-Schultz cross-check on 1Min bars, first 30 min ------------
    ev = pd.read_csv(A.RESULTS / "s05_gap_events.csv", keep_default_na=False)
    ev["date"] = ev["date"].astype(str)
    cs = []
    for date, grp in ev.groupby("date", sort=True):
        mins = C.load_minute(date)
        for _, e in grp.iterrows():
            m1 = mins.get(e["symbol"])
            if m1 is None or len(m1) == 0:
                continue
            w = m1[(m1.index.time >= C.dt.time(9, 30)) & (m1.index.time < C.dt.time(10, 0))]
            if len(w) < 10:
                continue
            cs.append({"date": date, "symbol": e["symbol"], "gap": e["gap"],
                       "prev_close": e["prev_close"],
                       "cs_bps": corwin_schultz(w["high"].to_numpy(), w["low"].to_numpy())})
    cd = pd.DataFrame(cs)
    print(f"\nCorwin-Schultz on {len(cd)} events, first 30 min, bps:")
    print(cd["cs_bps"].describe(percentiles=[.25, .5, .75, .9]).round(1).to_string(),
          flush=True)
    A.save(cd, "s05_spread_cs.csv")


if __name__ == "__main__":
    main()
