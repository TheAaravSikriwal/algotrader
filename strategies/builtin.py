"""Reference strategies.

These are working *examples*, not recommendations -- they exist so the engine
and the UI have something to chew on, and so you have a template to copy when
you write your own. Every one of them is a well-known textbook rule that plenty
of people have already traded into the ground; treat the numbers they produce
as a test of the plumbing, not as an edge.

To add your own: copy a class, change ``name``, ``params`` and
``generate_signals``, and it appears in the UI automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import atr, bollinger, donchian, ema, macd, rsi, sma
from core.strategy import Param, Strategy, register


@register
class BuyAndHold(Strategy):
    name = "Buy and hold"
    description = "Always long. The baseline every other strategy has to beat."
    params: list[Param] = []

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


@register
class SmaCrossover(Strategy):
    name = "SMA crossover"
    description = ("Long while the fast moving average is above the slow one. "
                   "The canonical trend-following rule.")
    params = [
        Param("fast", 20, "Fast MA", "int", 2, 200, 1, help="Lookback of the fast average, in bars"),
        Param("slow", 50, "Slow MA", "int", 5, 400, 1, help="Lookback of the slow average, in bars"),
        Param("use_ema", False, "Use EMA instead of SMA", "bool"),
        Param("short_on_cross_down", False, "Go short on the down-cross", "bool",
              help="Otherwise the strategy simply goes flat"),
    ]

    def _mas(self, df):
        f = ema if self.use_ema else sma
        return f(df["close"], int(self.fast)), f(df["close"], int(self.slow))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast, slow = self._mas(df)
        sig = pd.Series(0.0, index=df.index)
        sig[fast > slow] = 1.0
        if self.short_on_cross_down:
            sig[(fast < slow) & fast.notna() & slow.notna()] = -1.0
        return sig.where(fast.notna() & slow.notna(), 0.0)

    def indicators(self, df):
        fast, slow = self._mas(df)
        label = "EMA" if self.use_ema else "SMA"
        return {f"{label} {int(self.fast)}": fast, f"{label} {int(self.slow)}": slow}


@register
class RsiMeanReversion(Strategy):
    name = "RSI mean reversion"
    description = ("Buy when RSI drops below the oversold level, exit when it "
                   "recovers past the exit level. Fights the trend by design.")
    params = [
        Param("period", 14, "RSI period", "int", 2, 100, 1),
        Param("oversold", 30, "Entry (oversold) level", "int", 5, 50, 1),
        Param("exit_level", 55, "Exit level", "int", 20, 95, 1),
        Param("trend_filter", 0, "Only trade above this SMA", "int", 0, 400, 10,
              help="0 turns the filter off"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["close"], int(self.period)).to_numpy()
        allowed = np.ones(len(df), dtype=bool)
        if int(self.trend_filter) > 0:
            trend = sma(df["close"], int(self.trend_filter))
            allowed = (df["close"] > trend).fillna(False).to_numpy()

        sig = np.zeros(len(df))
        holding = False
        for i in range(len(df)):
            if np.isnan(r[i]):
                continue
            if holding:
                if r[i] >= self.exit_level:
                    holding = False
            elif r[i] <= self.oversold and allowed[i]:
                holding = True
            sig[i] = 1.0 if holding else 0.0
        return pd.Series(sig, index=df.index)

    def indicators(self, df):
        if int(self.trend_filter) > 0:
            return {f"SMA {int(self.trend_filter)}": sma(df["close"], int(self.trend_filter))}
        return {}


@register
class DonchianBreakout(Strategy):
    name = "Donchian breakout"
    description = ("Buy a break above the highest high of the last N bars, exit "
                   "on a break below the lowest low of the last M. The turtle rule.")
    params = [
        Param("entry_lookback", 20, "Entry channel (bars)", "int", 5, 300, 1),
        Param("exit_lookback", 10, "Exit channel (bars)", "int", 3, 300, 1),
        Param("mirror_short", False, "Take the mirror-image short", "bool"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        entry_lo, entry_hi = donchian(df, int(self.entry_lookback))
        exit_lo, exit_hi = donchian(df, int(self.exit_lookback))

        c = df["close"].to_numpy()
        ehi, elo = entry_hi.to_numpy(), entry_lo.to_numpy()
        xlo, xhi = exit_lo.to_numpy(), exit_hi.to_numpy()

        sig = np.zeros(len(df))
        pos = 0
        for i in range(len(df)):
            if pos == 0:
                if not np.isnan(ehi[i]) and c[i] > ehi[i]:
                    pos = 1
                elif self.mirror_short and not np.isnan(elo[i]) and c[i] < elo[i]:
                    pos = -1
            elif pos == 1 and not np.isnan(xlo[i]) and c[i] < xlo[i]:
                pos = 0
            elif pos == -1 and not np.isnan(xhi[i]) and c[i] > xhi[i]:
                pos = 0
            sig[i] = pos
        return pd.Series(sig, index=df.index)

    def indicators(self, df):
        lo, hi = donchian(df, int(self.entry_lookback))
        n = int(self.entry_lookback)
        return {f"Upper {n}": hi, f"Lower {n}": lo}


@register
class BollingerReversion(Strategy):
    name = "Bollinger reversion"
    description = "Buy a close below the lower band, exit on a return to the middle band."
    params = [
        Param("period", 20, "Band period", "int", 5, 200, 1),
        Param("num_std", 2.0, "Band width (std devs)", "float", 0.5, 4.0, 0.1),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lower, mid, _ = bollinger(df["close"], int(self.period), float(self.num_std))
        c, lo, m = df["close"].to_numpy(), lower.to_numpy(), mid.to_numpy()

        sig = np.zeros(len(df))
        holding = False
        for i in range(len(df)):
            if np.isnan(lo[i]):
                continue
            if holding:
                if c[i] >= m[i]:
                    holding = False
            elif c[i] < lo[i]:
                holding = True
            sig[i] = 1.0 if holding else 0.0
        return pd.Series(sig, index=df.index)

    def indicators(self, df):
        lower, mid, upper = bollinger(df["close"], int(self.period), float(self.num_std))
        return {"Lower band": lower, "Middle band": mid, "Upper band": upper}


@register
class MacdTrend(Strategy):
    name = "MACD trend"
    description = "Long while the MACD line sits above its signal line."
    params = [
        Param("fast", 12, "Fast EMA", "int", 2, 100, 1),
        Param("slow", 26, "Slow EMA", "int", 5, 200, 1),
        Param("signal", 9, "Signal EMA", "int", 2, 50, 1),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        line, sig_line, _ = macd(df["close"], int(self.fast), int(self.slow), int(self.signal))
        return (line > sig_line).astype(float).where(line.notna() & sig_line.notna(), 0.0)


@register
class VolatilityBreakout(Strategy):
    name = "Volatility breakout"
    description = ("Buy when price closes more than N ATRs above its recent "
                   "average, exit when it falls back below that average.")
    params = [
        Param("lookback", 20, "Reference SMA (bars)", "int", 5, 200, 1),
        Param("atr_period", 14, "ATR period", "int", 2, 100, 1),
        Param("threshold", 1.0, "Entry threshold (ATRs)", "float", 0.1, 5.0, 0.1),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        base = sma(df["close"], int(self.lookback))
        upper = base + float(self.threshold) * atr(df, int(self.atr_period))
        c, u, b = df["close"].to_numpy(), upper.to_numpy(), base.to_numpy()

        sig = np.zeros(len(df))
        holding = False
        for i in range(len(df)):
            if np.isnan(u[i]):
                continue
            if holding:
                if c[i] < b[i]:
                    holding = False
            elif c[i] > u[i]:
                holding = True
            sig[i] = 1.0 if holding else 0.0
        return pd.Series(sig, index=df.index)

    def indicators(self, df):
        base = sma(df["close"], int(self.lookback))
        return {
            f"SMA {int(self.lookback)}": base,
            "Breakout level": base + float(self.threshold) * atr(df, int(self.atr_period)),
        }
