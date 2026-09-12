"""The classics that were missing from `builtin.py`.

These are the rules that turn up in every introductory book and course. They
are here to be *measured*, not recommended -- several of them are famous
precisely because they were published, and a rule that has been in print for
forty years has had forty years to be arbitraged.

Each one is written as literally as the usual description allows. Where the
common statement is ambiguous, the docstring says which reading was taken,
because "momentum" and "mean reversion" each name half a dozen different rules
and quietly picking the flattering one is how a strategy library becomes
useless.

Every signal is decided from information available at the close of its own bar;
`core.engine` fills it at the next bar's open. Nothing here may peek.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import atr, ema, rsi, sma
from core.strategy import Param, Strategy, register


@register
class TimeSeriesMomentum(Strategy):
    name = "Time-series momentum"
    description = ("Long when the trailing return over the lookback is positive. "
                   "Moskowitz, Ooi and Pedersen (2012); the simplest trend rule "
                   "there is.")
    params = [
        Param("lookback", 252, "Lookback", "int", 5, 756, 1,
              help="Bars of trailing return to judge. 252 is about a year of daily bars"),
        Param("short_when_negative", False, "Short when the return is negative", "bool",
              help="Otherwise it simply goes flat"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        n = int(self.lookback)
        trailing = df["close"].pct_change(n)
        sig = pd.Series(0.0, index=df.index)
        sig[trailing > 0] = 1.0
        if self.short_when_negative:
            sig[trailing < 0] = -1.0
        return sig.where(trailing.notna(), 0.0)

    def indicators(self, df):
        return {f"Return over {int(self.lookback)}": df["close"].pct_change(int(self.lookback))}


@register
class MovingAverageDistance(Strategy):
    name = "Price vs moving average"
    description = ("Long while price is above its own moving average. The "
                   "single most-cited technical rule, and the one source 11 "
                   "states as 'price above the MA signals a bullish trend'.")
    params = [
        Param("window", 200, "Moving average", "int", 5, 400, 1),
        Param("buffer_pct", 0.0, "Dead band (%)", "float", 0.0, 10.0, 0.25,
              help="Price must clear the average by this much before flipping. "
                   "Reduces whipsaw at the cost of lateness"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ma = sma(df["close"], int(self.window))
        band = float(self.buffer_pct) / 100.0
        sig = pd.Series(np.nan, index=df.index)
        sig[df["close"] > ma * (1 + band)] = 1.0
        sig[df["close"] < ma * (1 - band)] = 0.0
        # Inside the dead band, hold whatever was held. That is what a dead band
        # means -- filling it with zero would flatten on every quiet day and
        # turn the buffer into extra trading rather than less.
        return sig.ffill().fillna(0.0).where(ma.notna(), 0.0)

    def indicators(self, df):
        return {f"SMA {int(self.window)}": sma(df["close"], int(self.window))}


@register
class GoldenCross(Strategy):
    name = "Golden cross"
    description = ("The 50/200 crossover by name. A special case of the SMA "
                   "crossover, included because it is quoted as its own "
                   "strategy often enough to deserve measuring on its own.")
    params = [
        Param("fast", 50, "Fast MA", "int", 5, 200, 1),
        Param("slow", 200, "Slow MA", "int", 20, 400, 1),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        f, s = sma(df["close"], int(self.fast)), sma(df["close"], int(self.slow))
        return (f > s).astype(float).where(f.notna() & s.notna(), 0.0)


@register
class BuyTheDip(Strategy):
    name = "Buy the dip"
    description = ("Buy after a drop of a given size, hold a fixed number of "
                   "bars. The folk strategy, stated precisely enough to test.")
    params = [
        Param("drop_pct", 3.0, "Drop that triggers a buy (%)", "float", 0.5, 25.0, 0.5),
        Param("lookback", 5, "Measured over this many bars", "int", 1, 60, 1),
        Param("hold", 10, "Hold for this many bars", "int", 1, 120, 1),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        drawdown = df["close"].pct_change(int(self.lookback)) * 100.0
        trigger = drawdown <= -float(self.drop_pct)
        # A fresh trigger restarts the clock rather than extending it, so a
        # long slide is one position held `hold` bars from the last trigger,
        # not a position that never closes.
        held = trigger.rolling(int(self.hold), min_periods=1).max()
        return held.fillna(0.0).astype(float)


@register
class FiftyTwoWeekHigh(Strategy):
    name = "52-week high"
    description = ("Long while price is within a whisker of its one-year high. "
                   "George and Hwang (2004) found this carried most of what "
                   "momentum was measuring.")
    params = [
        Param("window", 252, "Lookback", "int", 20, 756, 1),
        Param("within_pct", 2.0, "Within this much of the high (%)", "float",
              0.0, 20.0, 0.5),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # shift(1) so today's own high cannot define the level today's signal
        # is compared against -- without it the rule trivially triggers on
        # every new high, using that bar's own data.
        high = df["high"].rolling(int(self.window)).max().shift(1)
        close_enough = df["close"] >= high * (1 - float(self.within_pct) / 100.0)
        return close_enough.astype(float).where(high.notna(), 0.0)

    def indicators(self, df):
        return {f"{int(self.window)}-bar high":
                df["high"].rolling(int(self.window)).max().shift(1)}


@register
class StochasticReversion(Strategy):
    name = "Stochastic reversion"
    description = ("Buy when the stochastic leaves oversold, exit when it "
                   "leaves overbought. Source 11's oscillator rule, and the "
                   "confirmation step source 09 never fully specified.")
    params = [
        Param("k", 14, "%K lookback", "int", 2, 100, 1),
        Param("smooth", 3, "%K smoothing", "int", 1, 20, 1),
        Param("oversold", 20, "Oversold level", "int", 1, 49, 1),
        Param("overbought", 80, "Overbought level", "int", 51, 99, 1),
    ]

    def _stoch(self, df):
        n = int(self.k)
        lo = df["low"].rolling(n).min()
        hi = df["high"].rolling(n).max()
        raw = 100.0 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
        return raw.rolling(int(self.smooth)).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        st = self._stoch(df)
        sig = pd.Series(np.nan, index=df.index)
        sig[(st > float(self.oversold)) & (st.shift(1) <= float(self.oversold))] = 1.0
        sig[(st < float(self.overbought)) & (st.shift(1) >= float(self.overbought))] = 0.0
        return sig.ffill().fillna(0.0).where(st.notna(), 0.0)

    def indicators(self, df):
        return {"Stochastic %K": self._stoch(df)}


@register
class KeltnerBreakout(Strategy):
    name = "Keltner breakout"
    description = ("Long on a close above an ATR band around the EMA. The "
                   "volatility-scaled cousin of Donchian, so the threshold "
                   "widens when the market is already moving.")
    params = [
        Param("window", 20, "EMA window", "int", 5, 200, 1),
        Param("atr_window", 20, "ATR window", "int", 5, 200, 1),
        Param("mult", 2.0, "Band width in ATRs", "float", 0.5, 6.0, 0.25),
    ]

    def _bands(self, df):
        mid = ema(df["close"], int(self.window))
        width = atr(df, int(self.atr_window)) * float(self.mult)
        return mid, mid + width, mid - width

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        mid, up, lo = self._bands(df)
        sig = pd.Series(np.nan, index=df.index)
        sig[df["close"] > up] = 1.0
        sig[df["close"] < mid] = 0.0          # give back to the middle, not the lower band
        return sig.ffill().fillna(0.0).where(up.notna(), 0.0)

    def indicators(self, df):
        mid, up, lo = self._bands(df)
        return {"Keltner mid": mid, "Keltner upper": up, "Keltner lower": lo}


@register
class DayOfWeek(Strategy):
    name = "Day of week"
    description = ("Hold only on chosen weekdays. A pure calendar effect, and "
                   "a useful null: if this beats the others, the others are "
                   "noise.")
    params = [
        Param("weekday", 0, "Weekday to hold", "choice", choices=[0, 1, 2, 3, 4],
              help="0 Monday through 4 Friday"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # The position is decided at the previous close and filled at this
        # bar's open, so the day that matters is the NEXT bar's weekday.
        next_day = pd.Series(df.index, index=df.index).shift(-1).dt.weekday
        return (next_day == int(self.weekday)).astype(float)


@register
class GapFade(Strategy):
    name = "Gap fade"
    description = ("Bet against an overnight gap. Source 05 trades gaps in the "
                   "other direction, so measuring this is how you find out "
                   "which way the drift actually runs.")
    params = [
        Param("gap_pct", 1.0, "Gap size that triggers (%)", "float", 0.1, 20.0, 0.1),
        Param("hold", 1, "Hold for this many bars", "int", 1, 30, 1),
        Param("fade_up", True, "Short gaps up", "bool"),
        Param("fade_down", True, "Buy gaps down", "bool"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        gap = (df["open"] / df["close"].shift(1) - 1.0) * 100.0
        sig = pd.Series(0.0, index=df.index)
        if self.fade_down:
            sig[gap <= -float(self.gap_pct)] = 1.0
        if self.fade_up:
            sig[gap >= float(self.gap_pct)] = -1.0
        if int(self.hold) > 1:
            # Hold the most recent non-zero signal for `hold` bars.
            sig = sig.replace(0.0, np.nan).ffill(limit=int(self.hold) - 1).fillna(0.0)
        return sig.where(gap.notna(), 0.0)


@register
class TurtleBreakout(Strategy):
    name = "Turtle breakout"
    description = ("Enter on a 20-bar high, exit on a 10-bar low. The Dennis "
                   "and Eckhardt rule, published in 1983 and traded ever since.")
    params = [
        Param("entry", 20, "Entry breakout", "int", 5, 200, 1),
        Param("exit", 10, "Exit breakout", "int", 2, 100, 1),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        hi = df["high"].rolling(int(self.entry)).max().shift(1)
        lo = df["low"].rolling(int(self.exit)).min().shift(1)
        sig = pd.Series(np.nan, index=df.index)
        sig[df["close"] > hi] = 1.0
        sig[df["close"] < lo] = 0.0
        return sig.ffill().fillna(0.0).where(hi.notna() & lo.notna(), 0.0)

    def indicators(self, df):
        return {f"{int(self.entry)}-bar high": df["high"].rolling(int(self.entry)).max().shift(1),
                f"{int(self.exit)}-bar low": df["low"].rolling(int(self.exit)).min().shift(1)}
