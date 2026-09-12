"""s02 STEP 2 -- overnight vs intraday decomposition.

Daily bars from yfinance (long history; Alpaca daily only reaches 2016 and the
point of this test is the longest sample available).

Decomposition, per symbol per day t:
    overnight_t = Open_t  / Close_{t-1} - 1
    intraday_t  = Close_t / Open_t      - 1
    total_t     = Close_t / Close_{t-1} - 1 = (1+on)(1+id) - 1

All four OHLC series are split/dividend adjusted by the SAME daily factor
(AdjClose/Close), so every ratio above is a true total return and the dividend
drop lands in the overnight leg, where it belongs.

TIMESTAMP PROVENANCE: the only inputs are the official opening auction price and
the official closing auction price. To trade the overnight leg you must be filled
at the close of day t-1 (known at 16:00:00 ET on t-1) and at the open of day t
(known at 09:30:00 ET on t). Nothing here uses any datum before it exists.

Writes:
  results/s02_overnight_decomp.csv   per-symbol per-period legs
  results/s02_overnight_agg.csv      index/basket level summary
  results/s02_overnight_strategy.csv the buy-close/sell-open strategy after cost
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

warnings.filterwarnings("ignore")

# RSP (equal-weight S&P) and DIA are added as SURVIVORSHIP-FREE analogues of the
# single-name basket below: an index ETF holds the future losers too, so if the
# 40-name basket's intraday leg looks better than RSP's, the difference is bias.
ETFS = ["SPY", "QQQ", "IWM", "RSP", "DIA", "MDY", "XLK", "XLF"]
# 30 liquid US large caps. SURVIVORSHIP: this is today's liquid list applied
# backwards, so the LEVEL of every return here is biased upward. The
# decomposition (which leg earns it) is what the test is about, and the bias
# hits both legs the same way.
LARGECAPS = [
    "AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "KO", "PEP", "WMT", "HD",
    "DIS", "CSCO", "INTC", "VZ", "MRK", "PFE", "CVX", "BA", "CAT", "MMM",
    "MCD", "NKE", "ORCL", "IBM", "WFC", "BAC", "C", "UNH", "AMGN", "TXN",
]
MODERN = ["AMZN", "NVDA", "GOOGL", "META", "TSLA", "AMD", "NFLX", "CRM", "ADBE", "COST"]

PERIODS = {
    "full": ("1990-01-01", "2026-09-11"),
    "2021_2026": ("2021-01-01", "2026-09-11"),
    "last3y": ("2023-09-11", "2026-09-11"),
    "pre2015": ("1990-01-01", "2015-01-01"),
}

CACHE = A.CACHE / "s02_daily.parquet"


def fetch_daily(symbols) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    if CACHE.exists():
        big = pd.read_parquet(CACHE)
        have = set(big["symbol"].unique())
        missing = [s for s in symbols if s not in have]
    else:
        big = pd.DataFrame()
        missing = list(symbols)
    frames = [big] if len(big) else []
    for sym in missing:
        d = yf.download(sym, start="1990-01-01", end="2026-09-11", interval="1d",
                        auto_adjust=False, progress=False, actions=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.dropna(subset=["Open", "Close", "Adj Close"])
        if len(d) < 250:
            print(f"  {sym}: only {len(d)} rows, skipped")
            continue
        f = (d["Adj Close"] / d["Close"]).astype(float)
        out = pd.DataFrame({
            "date": pd.DatetimeIndex(d.index).tz_localize(None).normalize(),
            "symbol": sym,
            "open": d["Open"].astype(float) * f,
            "high": d["High"].astype(float) * f,
            "low": d["Low"].astype(float) * f,
            "close": d["Adj Close"].astype(float),
            "raw_close": d["Close"].astype(float),
            "volume": d["Volume"].astype(float),
        }).reset_index(drop=True)
        frames.append(out)
        print(f"  {sym}: {len(out)} rows {out['date'].iloc[0].date()}..{out['date'].iloc[-1].date()}",
              flush=True)
    big = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    big.to_parquet(CACHE)
    return {s: g.sort_values("date").reset_index(drop=True)
            for s, g in big.groupby("symbol")}


def legs(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    pc = d["close"].shift(1)
    d["overnight"] = d["open"] / pc - 1.0
    d["intraday"] = d["close"] / d["open"] - 1.0
    d["total"] = d["close"] / pc - 1.0
    # guard against bad ticks / unadjusted split residue
    bad = (d["overnight"].abs() > 0.5) | (d["intraday"].abs() > 0.5)
    d.loc[bad, ["overnight", "intraday", "total"]] = np.nan
    return d.dropna(subset=["overnight", "intraday", "total"])


def stats(r: pd.Series, label: str, leg: str) -> dict:
    r = pd.Series(r, dtype=float).dropna()
    n = len(r)
    if n < 60:
        return {}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    t = A.tstat(r)
    geo = float(np.expm1(np.log1p(r).sum() * 252.0 / n))
    return {
        "label": label, "leg": leg, "n": n,
        "mean_bps": mu * 1e4,
        "ann_ret_arith_pct": mu * 252 * 100,
        "ann_ret_geom_pct": geo * 100,
        "ann_vol_pct": sd * np.sqrt(252) * 100,
        "sharpe": mu / sd * np.sqrt(252) if sd else np.nan,
        "t": t, "p": A.two_sided_p(t),
        "win_rate": float((r > 0).mean()),
        "start": str(r.index[0].date()) if isinstance(r.index, pd.DatetimeIndex) else "",
        "end": str(r.index[-1].date()) if isinstance(r.index, pd.DatetimeIndex) else "",
    }


def main():
    syms = ETFS + LARGECAPS + MODERN
    print(f"fetching {len(syms)} symbols of daily bars", flush=True)
    data = fetch_daily(syms)
    print(f"have {len(data)} symbols", flush=True)

    per = {s: legs(df).set_index("date") for s, df in data.items()}

    rows = []
    for period, (a, b) in PERIODS.items():
        for sym, d in per.items():
            w = d.loc[(d.index >= a) & (d.index < b)]
            for leg in ("overnight", "intraday", "total"):
                st = stats(w[leg], f"{sym}|{period}", leg)
                if st:
                    st["symbol"] = sym
                    st["period"] = period
                    rows.append(st)
    decomp = pd.DataFrame(rows)
    A.save(decomp, "s02_overnight_decomp.csv")

    # ---- aggregates -----------------------------------------------------
    agg_rows = []
    baskets = {"LARGECAP30": LARGECAPS, "MODERN10": MODERN,
               "ALL40": LARGECAPS + MODERN}
    for period, (a, b) in PERIODS.items():
        for sym in ETFS:
            if sym not in per:
                continue
            d = per[sym]
            w = d.loc[(d.index >= a) & (d.index < b)]
            for leg in ("overnight", "intraday", "total"):
                st = stats(w[leg], sym, leg)
                if st:
                    st["period"] = period
                    st["universe"] = sym
                    agg_rows.append(st)
        for bname, members in baskets.items():
            for leg in ("overnight", "intraday", "total"):
                cols = {}
                for sym in members:
                    if sym in per:
                        d = per[sym]
                        cols[sym] = d.loc[(d.index >= a) & (d.index < b), leg]
                if not cols:
                    continue
                panel = pd.DataFrame(cols)
                ew = panel.mean(axis=1).dropna()      # equal weight, daily rebalanced
                st = stats(ew, bname, leg)
                if st:
                    st["period"] = period
                    st["universe"] = bname
                    st["n_names"] = panel.notna().sum(axis=1).mean()
                    agg_rows.append(st)
    agg = pd.DataFrame(agg_rows)
    A.save(agg, "s02_overnight_agg.csv")

    print("\n" + "=" * 100)
    print("OVERNIGHT vs INTRADAY -- aggregates")
    print("=" * 100)
    show = ["period", "universe", "leg", "n", "mean_bps", "ann_ret_geom_pct",
            "ann_vol_pct", "sharpe", "t", "p", "win_rate"]
    for period in PERIODS:
        sub = agg[agg["period"] == period]
        if len(sub) == 0:
            continue
        print(f"\n--- {period} ---")
        print(sub[show].round(4).to_string(index=False))

    # ---- cross-sectional: how many of the 40 names have on > id? --------
    print("\n" + "=" * 100)
    print("CROSS-SECTION of single names (share with overnight mean > intraday mean)")
    print("=" * 100)
    xs_rows = []
    for period in PERIODS:
        sub = decomp[(decomp["period"] == period) & (~decomp["symbol"].isin(ETFS))]
        piv = sub.pivot_table(index="symbol", columns="leg", values="mean_bps")
        if not len(piv):
            continue
        share = float((piv["overnight"] > piv["intraday"]).mean())
        neg_id = float((piv["intraday"] < 0).mean())
        xs_rows.append({"period": period, "n_names": len(piv),
                        "share_overnight_beats_intraday": share,
                        "share_intraday_mean_negative": neg_id,
                        "median_overnight_bps": float(piv["overnight"].median()),
                        "median_intraday_bps": float(piv["intraday"].median())})
    xs = pd.DataFrame(xs_rows)
    print(xs.round(4).to_string(index=False))
    A.save(xs, "s02_overnight_crosssection.csv")

    # ---- the strategy: buy the close, sell the open, every day ----------
    print("\n" + "=" * 100)
    print("STRATEGY: long SPY from close to next open, flat intraday. 2 legs/day.")
    print("=" * 100)
    strat_rows = []
    for sym in ETFS:
        if sym not in per:
            continue
        d = per[sym]
        for period, (a, b) in PERIODS.items():
            r = d.loc[(d.index >= a) & (d.index < b), "overnight"].dropna()
            if len(r) < 60:
                continue
            n = len(r)
            yrs = n / 252.0
            gross_bps = float(r.mean() * 1e4)
            sd = float(r.std(ddof=1))
            for rt_bps in (0, 1, 2, 3, 5, 8, 10, 20):
                net = r - rt_bps / 1e4
                mu = float(net.mean())
                geo = float(np.expm1(np.log1p(net.clip(lower=-0.99)).sum() / yrs))
                strat_rows.append({
                    "symbol": sym, "period": period, "n_days": n,
                    "round_trips_per_yr": 252,
                    "round_trip_cost_bps": rt_bps,
                    "gross_mean_bps": gross_bps,
                    "net_mean_bps": mu * 1e4,
                    "net_ann_ret_geom_pct": geo * 100,
                    "net_sharpe": mu / sd * np.sqrt(252) if sd else np.nan,
                    "net_t": A.tstat(net),
                    "net_p": A.two_sided_p(A.tstat(net)),
                })
    strat = pd.DataFrame(strat_rows)
    A.save(strat, "s02_overnight_strategy.csv")
    for period in PERIODS:
        sub = strat[(strat["period"] == period)]
        if len(sub) == 0:
            continue
        print(f"\n--- {period} ---")
        print(sub.drop(columns=["period"]).round(3).to_string(index=False))

    print("\nBREAK-EVEN round-trip cost (bps) = gross mean overnight return in bps:")
    be = strat[strat["round_trip_cost_bps"] == 0][
        ["symbol", "period", "n_days", "gross_mean_bps"]].rename(
        columns={"gross_mean_bps": "breakeven_rt_bps"})
    print(be.round(3).to_string(index=False))
    A.save(be, "s02_overnight_breakeven.csv")


if __name__ == "__main__":
    main()
