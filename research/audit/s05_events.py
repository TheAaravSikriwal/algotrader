"""Source 05 audit -- STEP 1b: turn daily bars into "top 5 gappers per day".

No lookahead in the ranking: gap% = today_open / yesterday_close - 1 is known at
09:30:00 ET, and the earliest possible entry in this strategy is 09:40 ET.

Writes research/audit/results/s05_gap_events.csv and s05_gap_distribution.csv.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

DAILY_ADJ = A.CACHE / "s05_daily_adj"   # split+div adjusted -> honest GAP %
DAILY_RAW = A.CACHE / "s05_daily"       # unadjusted -> the PRICE the trader saw

PRICE_LO, PRICE_HI = 2.0, 20.0
MIN_PREV_DOLLAR_VOL = 1_000_000.0   # prior-day $ volume floor: tradability, no lookahead
TOP_N = 5
GAP_THRESHOLDS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
GAP_MIN = 0.10


def load_daily(DAILY) -> pd.DataFrame:
    files = sorted(DAILY.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no daily parquet files in {DAILY}")
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        if len(d) == 0:
            continue
        frames.append(d[["symbol", "timestamp", "open", "high", "low", "close", "volume"]])
    df = pd.concat(frames, ignore_index=True)
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df = df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    return df


def main():
    adj = load_daily(DAILY_ADJ)
    raw = load_daily(DAILY_RAW)
    print(f"adjusted rows={len(adj):,}  raw rows={len(raw):,}  "
          f"symbols={adj['symbol'].nunique():,} dates={adj['date'].nunique():,} "
          f"{adj['date'].min().date()} .. {adj['date'].max().date()}", flush=True)

    g = adj.groupby("symbol", sort=False)
    adj["prev_close"] = g["close"].shift(1)
    adj["prev_date"] = g["date"].shift(1)
    adj["prev_vol"] = g["volume"].shift(1)
    adj["gap"] = adj["open"] / adj["prev_close"] - 1.0

    # RAW prices decide the $2-$20 screen and the dollar-volume floor: that is
    # the price a 2021 trader actually saw on the scanner. Adjusted prices would
    # retroactively divide by every later reverse split.
    gr = raw.groupby("symbol", sort=False)
    raw["raw_prev_close"] = gr["close"].shift(1)
    raw["raw_prev_vol"] = gr["volume"].shift(1)
    raw = raw.rename(columns={"open": "raw_open", "high": "raw_high",
                              "low": "raw_low", "close": "raw_close",
                              "volume": "raw_volume"})
    df = adj.merge(raw[["symbol", "date", "raw_open", "raw_high", "raw_low",
                        "raw_close", "raw_volume", "raw_prev_close",
                        "raw_prev_vol"]], on=["symbol", "date"], how="inner")
    df["prev_dollar_vol"] = df["raw_prev_close"] * df["raw_prev_vol"]

    # gap must be against the IMMEDIATELY preceding session (<=5 calendar days,
    # covers long weekends) -- otherwise a halted/illiquid name fakes a gap
    df["gap_days"] = (df["date"] - df["prev_date"]).dt.days
    df = df[df["gap_days"].between(1, 5)]

    # split-artefact guard: if the raw gap and the adjusted gap disagree by more
    # than 5 percentage points a corporate action straddles the boundary
    df["raw_gap"] = df["raw_open"] / df["raw_prev_close"] - 1.0
    df["split_flag"] = (df["gap"] - df["raw_gap"]).abs() > 0.05

    elig = df[
        df["raw_prev_close"].between(PRICE_LO, PRICE_HI)
        & df["raw_open"].between(PRICE_LO * 0.5, PRICE_HI * 3)
        & (df["prev_dollar_vol"] >= MIN_PREV_DOLLAR_VOL)
        & df["gap"].notna() & np.isfinite(df["gap"])
        & ~df["split_flag"]
    ].copy()
    print(f"eligible symbol-days (RAW prev close ${PRICE_LO}-${PRICE_HI}, "
          f"prev $vol>=${MIN_PREV_DOLLAR_VOL:,.0f}, no split artefact): "
          f"{len(elig):,}", flush=True)
    print(f"rows dropped as split artefacts: {int(df['split_flag'].sum()):,}",
          flush=True)

    # --- gap distribution over the eligible pool -------------------------
    q = elig["gap"].describe(percentiles=[.5, .9, .95, .99, .999])
    print("\ngap distribution over ALL eligible symbol-days (%):")
    print(pd.concat([q[["count"]], q.drop("count") * 100]).round(2).to_string(),
          flush=True)

    rows = []
    for th in GAP_THRESHOLDS:
        sub = elig[elig["gap"] >= th]
        rows.append({"threshold_pct": th * 100, "symbol_days": len(sub),
                     "distinct_symbols": sub["symbol"].nunique(),
                     "days_with_ge1": sub["date"].nunique(),
                     "mean_gap_pct": sub["gap"].mean() * 100 if len(sub) else np.nan,
                     "median_gap_pct": sub["gap"].median() * 100 if len(sub) else np.nan})
    dist = pd.DataFrame(rows)
    print("\n", dist.to_string(index=False), flush=True)

    # --- top-5 gappers per day ------------------------------------------
    elig["rank"] = elig.groupby("date")["gap"].rank(ascending=False, method="first")
    sel = elig[(elig["rank"] <= TOP_N) & (elig["gap"] >= GAP_MIN)].copy()
    sel = sel.sort_values(["date", "rank"])

    print(f"\nTOP {TOP_N}/day with gap >= {GAP_MIN:.0%}:  events={len(sel):,} "
          f"distinct_symbols={sel['symbol'].nunique():,} "
          f"trading_days_covered={sel['date'].nunique():,}", flush=True)
    print("selected gap size distribution (%):")
    sq = sel["gap"].describe(percentiles=[.25, .5, .75, .9, .99])
    print(pd.concat([sq[["count"]], sq.drop("count") * 100]).round(2).to_string(),
          flush=True)
    print("\nevents per year:")
    print(sel.groupby(sel["date"].dt.year).size().to_string(), flush=True)
    print("\nselected RAW prev_close distribution ($):")
    print(sel["raw_prev_close"].describe(percentiles=[.25, .5, .75]).round(2).to_string(),
          flush=True)
    print("\nmost frequent symbols:")
    print(sel["symbol"].value_counts().head(15).to_string(), flush=True)

    out = sel[["date", "symbol", "rank", "gap", "raw_gap", "raw_prev_close",
               "raw_open", "raw_high", "raw_low", "raw_close", "raw_volume",
               "prev_dollar_vol"]].copy()
    out = out.rename(columns={"raw_prev_close": "prev_close", "raw_open": "open",
                              "raw_high": "high", "raw_low": "low",
                              "raw_close": "close", "raw_volume": "volume"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    A.save(out, "s05_gap_events.csv")
    A.save(dist, "s05_gap_distribution.csv")
    print("\nwrote", A.RESULTS / "s05_gap_events.csv", flush=True)


if __name__ == "__main__":
    main()
