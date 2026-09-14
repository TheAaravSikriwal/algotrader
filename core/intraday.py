"""Backtest any registered strategy as a day trade.

`core.engine` will run a strategy on five-minute bars quite happily, but the
result is not a day trade: it carries positions across the close, ignores the
session window, and charges a daily-bar cost model to something that trades
sixty times a week. All three flatter the result, and the third by an order of
magnitude.

This runs a strategy **one session at a time** under day-trading rules:

  * no position survives the close -- every day starts and ends flat
  * signals only inside the trading window, so the open is excluded by default
    (SPY's spread is 3.78 bps in the first fifteen minutes against 1.59
    mid-day, and that difference is larger than most intraday edges)
  * costs charged per side on every change of position

The output is per-trade returns, which is what `core.expectancy` needs to say
anything useful about whether a rule is worth trading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd

from core.data import session as rth
from core.marketclock import CalendarError, MarketCalendar, Session
from core.strategy import REGISTRY

DEFAULT_WINDOW = (time(10, 30), time(15, 30))


@dataclass(frozen=True)
class IntradayConfig:
    timeframe: str = "5Min"
    window_start: time = DEFAULT_WINDOW[0]
    window_end: time = DEFAULT_WINDOW[1]
    cost_bps: float = 2.0            # round trip; SPY mid-day measures ~1.6
    allow_short: bool = True
    flatten_minutes_before_close: int = 5
    min_bars_per_session: int = 12


@dataclass
class IntradayResult:
    trades: pd.DataFrame            # one row per round trip
    sessions: int
    bars: int
    symbol: str
    strategy: str
    equity: pd.Series = field(default_factory=pd.Series)

    @property
    def trade_returns(self) -> list[float]:
        if self.trades.empty:
            return []
        return self.trades["return_pct"].tolist()


def _session_trades(day: pd.DataFrame, sig: pd.Series, cfg: IntradayConfig,
                    flat_by: time | None) -> list[dict]:
    """Turn a signal series for one session into completed round trips.

    A signal on bar t is acted on at the OPEN of bar t+1, matching
    `core.engine`. Acting at bar t's close would use a price the signal was
    derived from, which is the single most common way an intraday backtest
    invents an edge.
    """
    idx = day.index
    opens = day["open"].to_numpy(float)
    closes = day["close"].to_numpy(float)
    n = len(day)
    want = sig.reindex(idx).fillna(0.0).to_numpy(float)

    trades: list[dict] = []
    pos, entry_px, entry_i = 0.0, 0.0, -1
    half = cfg.cost_bps / 2.0 / 10_000.0

    for i in range(1, n):
        target = want[i - 1]
        if not cfg.allow_short:
            target = max(target, 0.0)
        target = float(np.sign(target))          # one unit, long or short

        # Force flat at the deadline regardless of what the rule wants.
        forced = flat_by is not None and idx[i].time() >= flat_by
        if forced:
            target = 0.0

        if target == pos:
            continue

        px = opens[i]
        if pos != 0.0:
            gross = (px - entry_px) / entry_px * pos
            trades.append({
                "entry_ts": idx[entry_i], "exit_ts": idx[i],
                "direction": int(pos), "entry_px": entry_px, "exit_px": px,
                "bars_held": i - entry_i,
                # Both sides charged: you paid to get in and to get out.
                "return_pct": (gross - 2 * half) * 100.0,
                "gross_pct": gross * 100.0,
                "reason": "flatten" if forced else "signal",
            })
            pos, entry_px, entry_i = 0.0, 0.0, -1

        if target != 0.0 and not forced:
            pos, entry_px, entry_i = target, px, i

    # Anything still open closes on the session's last bar.
    if pos != 0.0:
        px = closes[-1]
        gross = (px - entry_px) / entry_px * pos
        trades.append({
            "entry_ts": idx[entry_i], "exit_ts": idx[-1],
            "direction": int(pos), "entry_px": entry_px, "exit_px": px,
            "bars_held": n - 1 - entry_i,
            "return_pct": (gross - 2 * half) * 100.0,
            "gross_pct": gross * 100.0,
            "reason": "eod",
        })
    return trades


def run(df: pd.DataFrame, strategy: str, symbol: str = "",
        params: dict | None = None, cfg: IntradayConfig | None = None,
        calendar: MarketCalendar | None = None) -> IntradayResult:
    """Run one strategy over intraday bars, session by session.

    `df` must be intraday bars on the Eastern clock (see `core.data`).
    """
    cfg = cfg or IntradayConfig()
    params = params or {}
    if strategy not in REGISTRY:
        raise KeyError(f"unknown strategy {strategy!r}")

    df = rth(df)
    if df.empty:
        return IntradayResult(pd.DataFrame(), 0, 0, symbol, strategy)

    rows, sessions = [], 0
    for day, chunk in df.groupby(df.index.normalize()):
        if len(chunk) < cfg.min_bars_per_session:
            continue

        flat_by = None
        if calendar is not None:
            if not calendar.is_trading_day(day):
                continue
            try:
                s = calendar.session(day)
            except CalendarError:
                continue
            flat_by = s.flatten_deadline(cfg.flatten_minutes_before_close).time()
            try:
                lo, hi = s.window(cfg.window_start, cfg.window_end)
                lo_t, hi_t = lo.time(), hi.time()
            except CalendarError:
                continue
        else:
            lo_t, hi_t = cfg.window_start, cfg.window_end

        # The whole session feeds the indicator; only the window may trade.
        # Cropping first would restart every moving average at 10:30 with no
        # history, which is a different strategy.
        try:
            sig = REGISTRY[strategy](**params).generate_signals(chunk)
        except Exception:
            continue
        inside = pd.Series(
            [(lo_t <= t.time() < hi_t) for t in chunk.index], index=chunk.index)
        sig = sig.where(inside, 0.0)

        rows.extend(_session_trades(chunk, sig, cfg, flat_by))
        sessions += 1

    trades = pd.DataFrame(rows)
    equity = pd.Series(dtype=float)
    if not trades.empty:
        trades = trades.sort_values("entry_ts").reset_index(drop=True)
        equity = (1.0 + trades["return_pct"] / 100.0).cumprod()
        equity.index = trades["exit_ts"]
    return IntradayResult(trades, sessions, len(df), symbol, strategy, equity)


def combine(results: Iterable[IntradayResult]) -> pd.DataFrame:
    """One table of trades across symbols, for a single expectancy figure."""
    frames = []
    for r in results:
        if r.trades.empty:
            continue
        t = r.trades.copy()
        t["symbol"] = r.symbol
        t["strategy"] = r.strategy
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("entry_ts")
