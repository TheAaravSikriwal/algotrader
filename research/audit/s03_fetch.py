"""Source 03 audit: fetch FRED macro series + ETF proxies. Writes s03_macro.csv."""
from __future__ import annotations

import io
import sys

import pandas as pd
import requests

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

START = "2002-01-01"
END = "2026-09-11"

FRED_SERIES = ["WALCL", "FEDFUNDS", "DFF", "DTB3"]


def fred(series_id: str) -> pd.Series:
    cache = A.CACHE / f"fred_{series_id}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        return s.astype(float)
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
           f"&cosd=1990-01-01&coed={END}")
    r = requests.get(url, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0 (audit research script)"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.set_index(date_col)[val_col].dropna()
    s.name = series_id
    s.to_frame().to_csv(cache)
    return s.astype(float)


PROXIES = {
    "IWO": "D_long (Russell 2000 Growth, small-cap speculative growth)",
    "XBI": "D_alt (biotech, loss-making growth)",
    "ARKK": "D_short (unprofitable high-growth)",
    "QQQ": "C (growth)",
    "IWF": "C_alt (Russell 1000 Growth)",
    "SPY": "B (quality/blend)",
    "IWD": "A (Russell 1000 Value)",
    "VTV": "A_alt (Vanguard Value)",
    "SPLV": "defensive (low volatility)",
    "USMV": "defensive_alt (min vol)",
}


def main():
    frames = {}
    for sid in FRED_SERIES:
        s = fred(sid)
        print(f"{sid}: {len(s)} obs, {s.index.min().date()} .. {s.index.max().date()}")
        frames[sid] = s
    macro = pd.concat(frames, axis=1)
    macro.index.name = "date"
    out = A.RESULTS / "s03_macro.csv"
    macro.to_csv(out)
    print("wrote", out, macro.shape)

    for sym in PROXIES:
        try:
            df = A.bars(sym, START, END, "1Day", source="yfinance", eastern=False)
            print(f"{sym}: {len(df)} bars, {df.index.min().date()} .. {df.index.max().date()}")
        except Exception as exc:  # noqa: BLE001
            print(f"{sym}: FAILED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
