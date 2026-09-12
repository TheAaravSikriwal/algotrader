"""The one day-trading rule the audit left standing.

Source 10 proposed: break of structure, then change of character, then a limit
at the midpoint of a fair value gap, targeting 4R. Testing kept the last part
and threw away the rest. On 5,917 trades across ten symbols the trend
machinery contributed nothing (dropping it entirely scored the same on 2.4x
the sample), and ~90% of the edge survived with the gap replaced by a limit at
a random level the same distance away. Entering the same signal at *market*
instead scored -0.009R.

So what is actually being traded is **passive execution**: a resting limit,
paid for providing liquidity rather than taking it. The fair value gap is kept
only because it is a concrete, non-discretionary trigger -- not because it is
proven better than the alternatives.

**Re-running it here did not reproduce the audit's recommendation, and the
reason matters.** The audit measured a ~5.4 bps gross edge over the whole
session, then separately recommended restricting to 10:30-15:30 because the
spread is cheapest there. Nothing checked that the edge survived the
restriction. It does not. On 2021-2026, ten symbols
(`research/audit/window_sensitivity.py`):

    window              trades   gross bps      t   spread     net
    09:30-09:45            880        5.38   5.89     3.78   +1.60
    09:30-10:30         12,029        3.41   6.19     2.34   +1.07
    full session       140,879        0.91   5.80     2.10   -1.19
    10:30-15:30        121,631        0.67   3.85     1.59   -0.92   <- recommended
    11:00-14:00         74,349        0.75   3.22     1.59   -0.84
    15:00-15:55         19,328        1.31   5.31     1.79   -0.48

The gross edge is real and strongly significant, but it is concentrated almost
entirely in the opening minutes -- and that is exactly where the spread is
widest. The audit's 5.43 bps headline matches the first-fifteen-minutes figure
almost exactly, which is the tell that its sample was dominated by opening
trades. Move to the cheap window and the edge goes with the cost.

So the recommendation is self-defeating, and the honest reading is that the
edge *is* the spread: you are paid for providing liquidity roughly what
providing liquidity is worth. Which is what an efficient market would predict.

Two further cautions on the only windows that price positively:

  * The margin at the open is +1.07 to +1.60 bps against a p90 spread of
    10.6 bps. At the ninetieth percentile the first-fifteen-minutes variant
    nets about -5 bps. A positive mean here is not a positive trade.
  * The backtest fills any limit the bar's range covers. Real fills are fewer
    and adversely selected -- you get filled when the other side wanted it,
    which correlates with being wrong. `core.fills` measures whether that
    assumption survives; until it has, every number above is an upper bound.

Kept in the repo because it is fully specified, honestly measured, and the
right vehicle for testing execution infrastructure on paper. Not because it
is expected to make money.

`Setup` is a *bracket intent*, not a position. The existing `Strategy` protocol
returns a target-position series, which is a market-order convention and cannot
express "rest a limit here, cancel it in twelve bars".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd

from core.marketclock import CalendarError, MarketCalendar, Session

# Defaults are the audit's recommendation, not tuned here.
DEFAULT_SYMBOLS = ("SPY", "QQQ")
WINDOW_START = time(10, 30)
WINDOW_END = time(15, 30)


@dataclass(frozen=True)
class DayTradeConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    timeframe: str = "5Min"
    window_start: time = WINDOW_START
    window_end: time = WINDOW_END
    target_r: float = 2.0            # 2R had the best hit rate at equal break-even
    expiry_bars: int = 12            # cancel an unfilled limit after this many
    max_hold_bars: int = 78          # one session of 5-minute bars
    flatten_minutes_before_close: int = 5
    risk_frac: float = 0.005         # 0.5% of equity per trade
    max_trades_per_symbol_per_day: int = 1
    swing_k: int = 3
    min_stop_frac: float = 0.0005    # reject setups whose stop is unrealistically tight


@dataclass(frozen=True)
class FVG:
    """A three-candle imbalance, exactly as source 10 defines it.

    Bullish when ``high[i-2] < low[i]``. Knowable at the close of candle 3 and
    not one bar earlier -- the definition needs all three.
    """
    idx: int
    direction: int                   # +1 bullish, -1 bearish
    lo: float
    hi: float

    @property
    def mid(self) -> float:
        """Source 10's 'consequent encroachment' -- the 50% level, and the entry."""
        return (self.lo + self.hi) / 2.0

    @property
    def width(self) -> float:
        return self.hi - self.lo


@dataclass
class Setup:
    """One bracket intent. Nothing here has been sent anywhere."""
    symbol: str
    signal_ts: pd.Timestamp          # close of the bar that produced it
    direction: int                   # +1 long, -1 short
    entry_px: float                  # the resting limit
    stop_px: float
    target_r: float
    expires_after_bars: int
    reason: str = "fvg_midpoint"

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_px - self.stop_px)

    @property
    def target_px(self) -> float:
        return self.entry_px + self.direction * self.target_r * self.risk_per_share

    @property
    def side(self) -> str:
        return "buy" if self.direction > 0 else "sell"

    def shares(self, equity: float, risk_frac: float) -> int:
        """Whole shares sized so a stop-out costs `risk_frac` of equity.

        Truncated, never rounded: rounding up means risking more than the
        number the rule promised.
        """
        if self.risk_per_share <= 0:
            return 0
        return int(np.trunc(equity * risk_frac / self.risk_per_share))


def fair_value_gaps(df: pd.DataFrame) -> list[FVG]:
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    out: list[FVG] = []
    for i in range(2, len(df)):
        if highs[i - 2] < lows[i]:
            out.append(FVG(i, +1, float(highs[i - 2]), float(lows[i])))
        elif lows[i - 2] > highs[i]:
            out.append(FVG(i, -1, float(highs[i]), float(lows[i - 2])))
    return out


def swing_levels(df: pd.DataFrame, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Most recent confirmed swing low / high as of each bar.

    A pivot at bar i needs k bars either side, so it is not confirmed until
    bar i+k. Publishing it at bar i would be lookahead -- the stop would be
    placed using bars that had not printed yet.
    """
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    n = len(df)
    last_low = np.full(n, np.nan)
    last_high = np.full(n, np.nan)
    cur_lo, cur_hi = np.nan, np.nan
    pending: list[tuple[int, float, str]] = []

    for i in range(k, n - k):
        wh, wl = highs[i - k:i + k + 1], lows[i - k:i + k + 1]
        if highs[i] == wh.max() and (wh[:k] < highs[i]).all() and (wh[k + 1:] < highs[i]).all():
            pending.append((i + k, float(highs[i]), "high"))
        if lows[i] == wl.min() and (wl[:k] > lows[i]).all() and (wl[k + 1:] > lows[i]).all():
            pending.append((i + k, float(lows[i]), "low"))

    by_bar: dict[int, list[tuple[float, str]]] = {}
    for known_at, px, kind in pending:
        by_bar.setdefault(known_at, []).append((px, kind))

    for i in range(n):
        for px, kind in by_bar.get(i, []):
            if kind == "low":
                cur_lo = px
            else:
                cur_hi = px
        last_low[i], last_high[i] = cur_lo, cur_hi
    return last_low, last_high


def find_setups(df: pd.DataFrame, symbol: str, cfg: DayTradeConfig,
                session: Session | None = None) -> list[Setup]:
    """Bracket intents for one symbol's session.

    `df` must be 5-minute bars on the Eastern clock (see `core.data`). Bars
    outside the trading window are still used for context -- swings and gaps
    need history -- but no setup is produced from them.
    """
    if len(df) < 3:
        return []

    lo_sw, hi_sw = swing_levels(df, cfg.swing_k)
    idx = df.index
    out: list[Setup] = []
    per_day: dict[pd.Timestamp, int] = {}

    lo_bound, hi_bound = cfg.window_start, cfg.window_end
    if session is not None:
        try:
            w0, w1 = session.window(cfg.window_start, cfg.window_end)
        except CalendarError:
            # The window does not overlap today's session at all -- a 15:00
            # start on a 13:00 half-day. That is a real answer, not an error:
            # there are no setups today. Raising here would abort a five-year
            # backtest because of one short Friday in November.
            return []
        lo_bound, hi_bound = w0.time(), w1.time()

    for gap in fair_value_gaps(df):
        i = gap.idx
        ts = idx[i]
        if not (lo_bound <= ts.time() < hi_bound):
            continue

        day = ts.normalize()
        if per_day.get(day, 0) >= cfg.max_trades_per_symbol_per_day:
            continue

        entry = gap.mid
        # Stop beyond the last confirmed swing, falling back to the gap's far
        # edge. The swing is the level price already failed to break; the gap
        # edge is merely where the imbalance ends.
        if gap.direction > 0:
            swing = lo_sw[i]
            stop = min(swing, gap.lo) if swing == swing else gap.lo
        else:
            swing = hi_sw[i]
            stop = max(swing, gap.hi) if swing == swing else gap.hi

        risk = abs(entry - stop)
        if risk <= 0 or risk < cfg.min_stop_frac * entry:
            continue

        out.append(Setup(symbol=symbol, signal_ts=ts, direction=gap.direction,
                         entry_px=float(entry), stop_px=float(stop),
                         target_r=cfg.target_r,
                         expires_after_bars=cfg.expiry_bars))
        per_day[day] = per_day.get(day, 0) + 1

    return out


def simulate(df: pd.DataFrame, setup: Setup, cfg: DayTradeConfig,
             session: Session | None = None) -> dict | None:
    """Resolve one bracket against the bars. Returns None if it never filled.

    Pessimistic within a bar, matching `core.engine`: when a bar's range covers
    both stop and target, the stop is taken, because a bar says nothing about
    the path inside it. A gap through the stop fills at the open, not at the
    stop -- otherwise the backtest reports a capped loss on an uncapped move,
    flattering exactly the scenario a stop exists for.

    The fill bar can also be the bar that resolves the trade. Skipping it would
    let a limit that fills and is then blown through in the same candle record
    a fill and no loss. With a stop this close to the entry, that single
    off-by-one manufactures a large fake edge.
    """
    idx = df.index
    try:
        start = idx.get_loc(setup.signal_ts)
    except KeyError:
        return None
    if isinstance(start, slice):
        start = start.start

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    day = idx[start].normalize()
    d = setup.direction
    entry, stop, target = setup.entry_px, setup.stop_px, setup.target_px

    flat_by = None
    if session is not None:
        flat_by = session.flatten_deadline(cfg.flatten_minutes_before_close).time()

    # -- wait for the limit to fill -------------------------------------
    fill_i = None
    for j in range(start + 1, min(start + 1 + setup.expires_after_bars, n)):
        if idx[j].normalize() != day:
            break
        touched = (l[j] <= entry) if d > 0 else (h[j] >= entry)
        if touched:
            fill_i = j
            break
    if fill_i is None:
        return None

    # -- resolve, starting with the fill bar itself ----------------------
    for j in range(fill_i, min(fill_i + cfg.max_hold_bars, n)):
        if idx[j].normalize() != day:
            return _close(setup, idx, fill_i, j - 1, c[j - 1], "eod_rollover")

        hit_stop = (l[j] <= stop) if d > 0 else (h[j] >= stop)
        hit_tp = (h[j] >= target) if d > 0 else (l[j] <= target)
        ref = o[j] if j > fill_i else entry

        if hit_stop:
            fill = min(ref, stop) if d > 0 else max(ref, stop)
            return _close(setup, idx, fill_i, j, fill, "stop")
        if hit_tp:
            fill = max(ref, target) if d > 0 else min(ref, target)
            return _close(setup, idx, fill_i, j, fill, "target")
        if flat_by is not None and idx[j].time() >= flat_by:
            return _close(setup, idx, fill_i, j, c[j], "flatten")

    j = min(fill_i + cfg.max_hold_bars, n - 1)
    return _close(setup, idx, fill_i, j, c[j], "timeout")


def _close(setup: Setup, idx, fill_i: int, exit_i: int, exit_px: float,
           reason: str) -> dict:
    risk = setup.risk_per_share
    pnl_per_share = (exit_px - setup.entry_px) * setup.direction
    return {
        "symbol": setup.symbol,
        "signal_ts": setup.signal_ts,
        "entry_ts": idx[fill_i],
        "exit_ts": idx[exit_i],
        "direction": setup.direction,
        "entry_px": setup.entry_px,
        "exit_px": float(exit_px),
        "stop_px": setup.stop_px,
        "target_px": setup.target_px,
        "risk_per_share": risk,
        "r": pnl_per_share / risk if risk else 0.0,
        "ret_pct": pnl_per_share / setup.entry_px * 100.0,
        "bars_held": exit_i - fill_i,
        "reason": reason,
    }


def backtest(bars: dict[str, pd.DataFrame], cfg: DayTradeConfig | None = None,
             calendar: MarketCalendar | None = None,
             cost_bps: float = 0.0) -> pd.DataFrame:
    """Run the rule over several symbols. One row per filled trade.

    `cost_bps` is charged round trip, subtracted from the return. It defaults
    to zero so the frictionless number is visible, but a frictionless number
    is not a result -- see `summarise`.
    """
    cfg = cfg or DayTradeConfig()

    if calendar is not None:
        spans = [(df.index.min(), df.index.max())
                 for df in bars.values() if df is not None and not df.empty]
        if spans:
            # Up front and loudly. `is_trading_day` answers False for a date the
            # calendar has never heard of, so without this the loop below would
            # skip every uncovered session in silence and still report a tidy
            # result computed from whatever fraction happened to be covered.
            calendar.require_covers(min(s for s, _ in spans), max(e for _, e in spans))

    rows = []
    for symbol, df in bars.items():
        if df is None or df.empty:
            continue
        for day, chunk in df.groupby(df.index.normalize()):
            sess = None
            if calendar is not None:
                if not calendar.is_trading_day(day):
                    continue
                sess = calendar.session(day)
            for setup in find_setups(chunk, symbol, cfg, sess):
                res = simulate(chunk, setup, cfg, sess)
                if res is not None:
                    res["ret_pct"] -= cost_bps / 100.0
                    rows.append(res)
    out = pd.DataFrame(rows)
    return out.sort_values("entry_ts").reset_index(drop=True) if len(out) else out


def summarise(trades: pd.DataFrame, cost_bps: float = 0.0) -> dict:
    """Headline numbers, including the one that decides everything.

    `breakeven_bps` is the round-trip cost at which the edge is exactly zero.
    Compare it against the measured spread: 1.59 bps for SPY mid-day, 2.12 for
    QQQ. A break-even below that means the rule does not survive being traded.
    """
    if trades is None or trades.empty:
        return {"trades": 0}
    r = trades["r"]
    ret = trades["ret_pct"]
    n = len(trades)
    se = ret.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {
        "trades": n,
        "symbols": int(trades["symbol"].nunique()),
        "win_rate": float((r > 0).mean()),
        "mean_r": float(r.mean()),
        "mean_ret_bps": float(ret.mean() * 100.0),
        "t_stat": float(ret.mean() / se) if se and se == se and se > 0 else np.nan,
        "breakeven_bps": float(ret.mean() * 100.0 + cost_bps),
        "target_rate": float((trades["reason"] == "target").mean()),
        "stop_rate": float((trades["reason"] == "stop").mean()),
        "median_risk_pct": float((trades["risk_per_share"] / trades["entry_px"] * 100).median()),
    }
