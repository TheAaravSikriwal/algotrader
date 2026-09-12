"""Historical bar data.

Two sources, same output shape:
  * ``yfinance``  -- free, no key, good enough to develop against
  * ``alpaca``    -- your broker's own data, needs API keys in the environment

Output is always a DataFrame indexed by tz-naive timestamps with columns
``open, high, low, close, volume``.

**The index is US/Eastern wall-clock time, not UTC.** That is the clock every
trading rule is written in: "the open" means 09:30 on this index, and
``bars.between_time("09:30", "15:59")`` really is the regular session.

This used to be UTC, which was a trap. Both Alpaca and yfinance hand back
tz-aware UTC timestamps for intraday bars; dropping the zone without
converting left 09:30 ET sitting at 13:30 (or 14:30 in winter). A session
filter written the obvious way then selected pre-market plus the first two
hours -- about a third of the day's volume -- while looking perfectly
correct. Use `session()` below rather than hand-rolling the filter.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

COLUMNS = ["open", "high", "low", "close", "volume"]

EXCHANGE_TZ = "America/New_York"

# The regular US equity session. Bars are stamped at the time they *open*, so
# the last bar of the day opens at 15:59 and the 16:00 stub, when a feed emits
# one, belongs to the close auction rather than to continuous trading.
SESSION_OPEN = "09:30"
SESSION_LAST_BAR = "15:59"


class DataError(RuntimeError):
    pass


def _cache_path(symbol: str, start: str, end: str, timeframe: str, source: str) -> Path:
    safe = f"{source}_{symbol.upper()}_{timeframe}_{start}_{end}".replace(":", "-")
    return CACHE_DIR / f"{safe}.csv"


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Put any source's bars on the repo's contract: Eastern, tz-naive, OHLCV.

    Public because broker adapters need it too. Keeping a second copy in
    brokers/alpaca.py is what let the UTC bug survive being fixed here -- the
    live loop went through the copy.
    """

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"data source returned no {missing} column(s)")
    df = df[COLUMNS]
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        # Convert to Eastern *before* dropping the zone. tz_convert(None) alone
        # would silently leave the index in UTC. This also lands Alpaca's daily
        # bars -- stamped 05:00 UTC in winter, 04:00 in summer -- on midnight,
        # matching yfinance instead of sitting a few hours off it.
        df.index = df.index.tz_convert(EXCHANGE_TZ).tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).rename("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.astype(float).dropna(subset=["open", "high", "low", "close"])
    return df


#: Historic private name. Kept so existing call sites and tests keep working.
_normalise = normalise


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


def session(df: pd.DataFrame) -> pd.DataFrame:
    """Regular trading hours only -- 09:30 through 15:59 Eastern.

    Intraday feeds hand back pre- and post-market bars by default, which is
    rarely what a rule means by "the open" or "the close". On a normal day this
    keeps 78 five-minute or 390 one-minute bars.

    Daily bars are returned untouched: they are stamped at midnight, which is
    outside any session window, so filtering them would throw the lot away.
    """
    idx = pd.DatetimeIndex(df.index)
    if len(idx) == 0:
        return df
    if (idx.normalize() == idx).all():
        return df
    return df.between_time(SESSION_OPEN, SESSION_LAST_BAR)


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
