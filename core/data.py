"""Historical bar data.

Two sources, same output shape:
  * ``yfinance``  -- free, no key, good enough to develop against
  * ``alpaca``    -- your broker's own data, needs API keys in the environment

Output is always a DataFrame indexed by tz-naive timestamps with columns
``open, high, low, close, volume``.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

COLUMNS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    pass


def _cache_path(symbol: str, start: str, end: str, timeframe: str, source: str) -> Path:
    safe = f"{source}_{symbol.upper()}_{timeframe}_{start}_{end}".replace(":", "-")
    return CACHE_DIR / f"{safe}.csv"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"data source returned no {missing} column(s)")
    df = df[COLUMNS]
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.DatetimeIndex(df.index).rename("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.astype(float).dropna(subset=["open", "high", "low", "close"])
    return df


def _from_yfinance(symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataError("yfinance is not installed -- pip install yfinance") from exc

    interval = {"1Day": "1d", "1Hour": "1h", "15Min": "15m", "5Min": "5m", "1Min": "1m"}[timeframe]
    raw = yf.download(symbol, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise DataError(f"yfinance returned no rows for {symbol} between {start} and {end}")
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance returns a (field, ticker) MultiIndex even for a single symbol
        raw.columns = raw.columns.get_level_values(0)
    return _normalise(raw)


def _from_alpaca(symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
    key = os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise DataError(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set. "
            "Copy .env.example to .env and fill them in, or use the yfinance source."
        )
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise DataError("alpaca-py is not installed -- pip install alpaca-py") from exc

    tf = {
        "1Day": TimeFrame.Day,
        "1Hour": TimeFrame.Hour,
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "1Min": TimeFrame.Minute,
    }[timeframe]

    client = StockHistoricalDataClient(key, secret)
    bars = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=tf,
        start=pd.Timestamp(start).to_pydatetime(),
        end=pd.Timestamp(end).to_pydatetime(),
    ))
    df = bars.df
    if df is None or df.empty:
        raise DataError(f"Alpaca returned no rows for {symbol} between {start} and {end}")
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol.upper(), level="symbol")
    return _normalise(df)


def load_bars(symbol: str, start, end, timeframe: str = "1Day",
              source: str = "yfinance", use_cache: bool = True) -> pd.DataFrame:
    """Fetch bars, transparently caching them on disk as CSV."""
    start = str(start)[:10]
    end = str(end)[:10]
    path = _cache_path(symbol, start, end, timeframe, source)

    if use_cache and path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if not cached.empty:
            return _normalise(cached)

    fetch = _from_alpaca if source == "alpaca" else _from_yfinance
    df = fetch(symbol, start, end, timeframe)
    if use_cache:
        df.to_csv(path)
    return df


def clear_cache() -> int:
    files = list(CACHE_DIR.glob("*.csv"))
    for f in files:
        f.unlink()
    return len(files)


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Infer annualisation factor from the actual spacing of the bars."""
    if len(index) < 3:
        return 252.0
    median_gap = pd.Series(index).diff().median()
    if pd.isna(median_gap) or median_gap <= pd.Timedelta(0):
        return 252.0
    seconds = median_gap.total_seconds()
    if seconds >= 86400 * 25:
        return 12.0
    if seconds >= 86400 * 6:
        return 52.0
    if seconds >= 86400:
        return 252.0
    return 252.0 * (6.5 * 3600 / seconds)  # intraday: 6.5h US session
