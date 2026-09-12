"""Source 05 audit -- STEP 1a: build the symbol pool and fetch daily bars.

Read-only market data. Writes ONLY:
  research/audit/cache/s05_assets.csv
  research/audit/cache/s05_daily/<year>_<batchno>.parquet

SURVIVORSHIP WARNING (restated in the report): Alpaca's assets endpoint returns
CURRENT active membership. Every ticker that delisted, was acquired, reverse
split into oblivion or went to zero between 2021 and today is ABSENT. For a
strategy that buys $2-$20 speculative gappers that is a large optimistic bias.
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

KEY = os.getenv("ALPACA_API_KEY_ID")
SEC = os.getenv("ALPACA_API_SECRET_KEY")

# Alpaca's bars default to adjustment="raw". In a universe of $2-$20 small caps
# reverse splits are routine, and a raw 1-for-20 reverse split shows up as a
# +1900% overnight "gap". Every result here uses SPLIT+DIVIDEND adjusted daily
# bars so the gap screen measures a real price move.
OUT = A.CACHE / "s05_daily_adj"
OUT.mkdir(exist_ok=True)

START = "2021-01-01"
END = "2026-09-02"

KEEP_EXCH = {"NASDAQ", "NYSE", "AMEX"}
# crude ETF / fund / trust exclusion by name -- Alpaca has no ETF flag
FUND_WORDS = (" ETF", " ETN", "ETF ", "INDEX FUND", " TRUST", "PROSHARES",
              "ISHARES", "SPDR", "DIREXION", "INVESCO", "VANGUARD", "WISDOMTREE",
              "GLOBAL X", "FIRST TRUST", "VANECK", "GRANITESHARES", "DEFIANCE ETF",
              "SIMPLIFY", "AMPLIFY", "ROUNDHILL", "YIELDMAX", " CLOSED END",
              "MUNICIPAL", "INCOME FUND", "EQUITY FUND", "GROWTH FUND")


def build_assets() -> pd.DataFrame:
    path = A.CACHE / "s05_assets.csv"
    if path.exists():
        # keep_default_na=False: the ticker "NA" would otherwise parse as NaN
        return pd.read_csv(path, keep_default_na=False)
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    tc = TradingClient(KEY, SEC, paper=True)
    assets = tc.get_all_assets(GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE))
    rows = []
    for a in assets:
        rows.append({
            "symbol": a.symbol,
            "name": (a.name or ""),
            "exchange": str(a.exchange).split(".")[-1],
            "tradable": bool(a.tradable),
            "shortable": bool(a.shortable),
            "fractionable": bool(a.fractionable),
            "marginable": bool(a.marginable),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def pool(df: pd.DataFrame) -> list[str]:
    d = df[df["tradable"] & df["exchange"].isin(KEEP_EXCH)].copy()
    up = d["name"].fillna("").str.upper()
    isfund = pd.Series(False, index=d.index)
    for w in FUND_WORDS:
        isfund |= up.str.contains(w, regex=False)
    d = d[~isfund]
    # drop obvious warrants / units / rights / preferreds: Alpaca uses SYM.WS etc
    bad = d["symbol"].str.contains(r"[./]", regex=True)
    d = d[~bad]
    syms = sorted(d["symbol"].unique().tolist())
    return syms


def fetch(symbols: list[str], year: int, batch_no: int):
    path = OUT / f"{year}_{batch_no:03d}.parquet"
    if path.exists():
        return path, True
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment

    dc = StockHistoricalDataClient(KEY, SEC)
    s = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(START))
    e = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(END))
    for attempt in range(4):
        try:
            r = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=s.to_pydatetime(),
                end=e.to_pydatetime(),
                adjustment=Adjustment.ALL,
            ))
            df = r.df
            break
        except Exception as exc:  # noqa: BLE001
            print(f"    retry {attempt} {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError("give up")
    if df is None or len(df) == 0:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.to_parquet(path)
        return path, False
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    keep = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    if "trade_count" in df.columns:
        keep.append("trade_count")
    if "vwap" in df.columns:
        keep.append("vwap")
    df = df[keep]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["symbol"] = df["symbol"].astype("category")
    df.to_parquet(path, index=False)
    return path, False


def main():
    assets = build_assets()
    syms = pool(assets)
    print(f"assets total={len(assets)}  pool after filters={len(syms)}", flush=True)
    batch = int(os.getenv("S05_BATCH", "400"))
    batches = [syms[i:i + batch] for i in range(0, len(syms), batch)]
    years = list(range(2021, 2027))
    t0 = time.time()
    for y in years:
        for bi, b in enumerate(batches):
            p, cached = fetch(b, y, bi)
            if not cached:
                print(f"  {y} batch {bi + 1}/{len(batches)} -> {p.name} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
