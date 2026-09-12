"""Source 04/10/11 audit: fetch the long-history index series + T-bill rate.

Writes nothing into the repo's own data_cache/; everything lands in
research/audit/cache/ via auditlib.bars, plus a FRED CSV for DTB3.
"""
from __future__ import annotations

import io
import sys

import pandas as pd
import requests

sys.path.insert(0, r"D:\VisualStudioProjects\algotrader\research\audit")
import auditlib as A  # noqa: E402

END = "2026-09-11"

LONG = {
    "VFINX": "1980-01-01",   # Vanguard 500 Index Investor, total return (auto_adjust)
    "VFIAX": "2000-11-01",   # Vanguard 500 Index Admiral, total return
    "SPY": "1993-01-01",     # SPDR S&P 500 ETF, total return
    "^GSPC": "1927-12-30",   # S&P 500 PRICE ONLY -- no dividends
}


def fred(series_id: str) -> pd.Series:
    cache = A.CACHE / f"fred_{series_id}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        return s.astype(float)
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
           f"&cosd=1900-01-01&coed={END}")
    r = requests.get(url, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0 (audit research script)"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df[df.columns[1]] = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s = df.set_index(df.columns[0])[df.columns[1]].dropna()
    s.name = series_id
    s.to_frame().to_csv(cache)
    return s.astype(float)


def main():
    for sym, start in LONG.items():
        try:
            df = A.bars(sym, start, END, "1Day", source="yfinance", eastern=False)
            print(f"{sym}: {len(df)} bars {df.index.min().date()} .. {df.index.max().date()} "
                  f"first_close={df['close'].iloc[0]:.4f} last_close={df['close'].iloc[-1]:.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{sym}: FAILED {type(exc).__name__}: {exc}")

    for sid in ("DTB3", "TB3MS"):
        try:
            s = fred(sid)
            print(f"{sid}: {len(s)} obs {s.index.min().date()} .. {s.index.max().date()}")
        except Exception as exc:  # noqa: BLE001
            print(f"{sid}: FAILED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
