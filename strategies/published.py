"""Strategies with a named, published source.

Unlike `builtin.py`, which holds generic textbook rules, every strategy here
traces to a specific author and a specific paper or book. That matters for one
reason: a published rule has a **fixed specification you did not choose**, so
running it is a real test rather than a search for whatever fits your data.

The catch is the other side of publication. A rule that worked before it was
published is not evidence it works after — being widely known is precisely
what arbitrages an edge away. Faber's paper appeared in 2007; Antonacci's book
in 2014. Treat these as hypotheses with a date stamp on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import adx, rolling_vwap, sma
from core.strategy import Param, Strategy, register


@register
class FaberTAA(Strategy):
    name = "Faber TAA"
    description = ("Hold the asset while it closes above its 10-month moving "
                   "average, otherwise sit in cash. From Meb Faber's 'A "
                   "Quantitative Approach to Tactical Asset Allocation' (2007).")
    params = [
        Param("months", 10, "Moving average (months)", "int", 1, 24, 1,
              help="Faber used 10 months; on daily bars that is ~200 sessions"),
        Param("bars_per_month", 21, "Bars per month", "int", 15, 25, 1),
    ]

    def _trend(self, df: pd.DataFrame) -> pd.Series:
        window = int(self.months) * int(self.bars_per_month)
        return sma(df["close"], window)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        trend = self._trend(df)
        return (df["close"] > trend).astype(float).where(trend.notna(), 0.0)

    def indicators(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        window = int(self.months) * int(self.bars_per_month)
        return {f"SMA {window}": self._trend(df)}


@register
class TurnOfMonth(Strategy):
    name = "Turn of month"
    description = ("Hold only across the turn of the month — the last few "
                   "sessions and the first few. A long-documented calendar "
                   "pattern, usually attributed to pension and payroll flows.")
    params = [
        Param("days_before", 3, "Sessions before month end", "int", 0, 10, 1),
        Param("days_after", 3, "Sessions after month start", "int", 0, 10, 1),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        index = pd.DatetimeIndex(df.index)
        periods = index.to_period("M")
        signals = np.zeros(len(df))

        before, after = int(self.days_before), int(self.days_after)
        for _, positions in pd.Series(np.arange(len(df)), index=periods).groupby(level=0):
            rows = positions.to_numpy()
            if after > 0:
                signals[rows[:after]] = 1.0
            if before > 0:
                signals[rows[-before:]] = 1.0
        return pd.Series(signals, index=df.index)


@register
class AdxTrend(Strategy):
    name = "ADX trend"
    description = ("Long when +DI is above -DI and ADX confirms the trend is "
                   "strong enough to be worth following. Suggested by "
                   "r/algotrading user Rooster_Odd.")
    params = [
        Param("period", 14, "ADX period", "int", 5, 60, 1),
        Param("adx_min", 25, "Minimum ADX", "int", 0, 60, 1,
              help="Below ~20 the market is rangebound and DI crossovers whipsaw"),
        Param("trade_negative", False, "Short when -DI leads", "bool"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        strength, plus_di, minus_di = adx(df, int(self.period))
        strong = strength >= float(self.adx_min)

        signals = pd.Series(0.0, index=df.index)
        signals[strong & (plus_di > minus_di)] = 1.0
        if self.trade_negative:
            signals[strong & (minus_di > plus_di)] = -1.0
        return signals.where(strength.notna(), 0.0)

    def indicators(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {}


@register
class VwapReversion(Strategy):
    name = "VWAP reversion"
    description = ("Buy when price falls a set distance below its rolling "
                   "volume-weighted average price, exit on the way back. On "
                   "daily bars this is an approximation of the intraday measure.")
    params = [
        Param("period", 20, "VWAP window (bars)", "int", 5, 120, 1),
        Param("entry_pct", 3.0, "Entry distance below VWAP (%)", "float", 0.5, 15.0, 0.5),
        Param("exit_pct", 0.0, "Exit distance from VWAP (%)", "float", -5.0, 5.0, 0.5),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        vwap = rolling_vwap(df, int(self.period))
        gap = (df["close"] / vwap - 1.0) * 100

        entry, exit_level = -float(self.entry_pct), float(self.exit_pct)
        values = gap.to_numpy()
        signals = np.zeros(len(df))
        holding = False

        for i in range(len(df)):
            if np.isnan(values[i]):
                continue
            if holding:
                if values[i] >= exit_level:
                    holding = False
            elif values[i] <= entry:
                holding = True
            signals[i] = 1.0 if holding else 0.0
        return pd.Series(signals, index=df.index)

    def indicators(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {f"VWAP {int(self.period)}": rolling_vwap(df, int(self.period))}


@register
class SellInMay(Strategy):
    name = "Sell in May"
    description = ("Hold from November through April, stay out May through "
                   "October. The Halloween indicator — the best known and most "
                   "argued-over seasonal in equities.")
    params = [
        Param("enter_month", 11, "Enter in month", "int", 1, 12, 1),
        Param("exit_month", 5, "Exit in month", "int", 1, 12, 1),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        months = pd.DatetimeIndex(df.index).month
        enter, leave = int(self.enter_month), int(self.exit_month)

        if enter <= leave:
            holding = (months >= enter) & (months < leave)
        else:                       # the window wraps around new year
            holding = (months >= enter) | (months < leave)
        return pd.Series(holding.astype(float), index=df.index)
