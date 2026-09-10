"""Small indicator library. Everything here is causal -- no forward-looking windows."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def donchian(df: pd.DataFrame, n: int = 20):
    """Prior-N-bar channel. Shifted so the current bar is excluded."""
    upper = df["high"].rolling(n, min_periods=n).max().shift(1)
    lower = df["low"].rolling(n, min_periods=n).min().shift(1)
    return lower, upper


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def adx(df: pd.DataFrame, n: int = 14):
    """Average Directional Index with its +DI / -DI components.

    ADX measures trend *strength* without regard to direction; the DI lines
    carry the direction. The pairing is the point -- a DI crossover in a
    trendless market (low ADX) is the classic whipsaw generator.

    Returns (adx, plus_di, minus_di).
    """
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()

    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)

    atr_n = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_n

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), plus_di, minus_di


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Volume-weighted average price over a rolling window.

    True VWAP is an intraday, session-anchored measure. On daily bars this is
    an approximation using the typical price -- useful as a volume-aware mean,
    not a substitute for the intraday quantity execution desks track.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].replace(0, np.nan)
    num = (typical * volume).rolling(n, min_periods=n).sum()
    den = volume.rolling(n, min_periods=n).sum()
    return num / den
