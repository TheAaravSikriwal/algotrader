"""Complex strategies: regime filters, volatility targeting, ensembles, ML.

Complexity here is a hypothesis, not an advantage. Each of these adds moving
parts to a simpler rule, and the whole point of measuring them alongside the
classics is to find out whether the extra machinery pays for itself. The audit
of the eleven sources already found one case where it did not -- source 10's
break-of-structure and change-of-character layers contributed nothing on top of
a bare limit order -- so the prior is not kind.

Two things every strategy in this file is careful about:

**Lookahead.** Anything fitted -- a scaler, a regression, a classifier -- is
fitted only on data before the bar it predicts. An expanding-window fit is used
rather than a full-sample one, which is slower and much less flattering.

**Degrees of freedom.** Each strategy reports how many parameters it has in its
description, because a rule with nine knobs that beats one with two has not
necessarily found anything. The tournament applies a multiple-comparison
correction for exactly this reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import atr, adx, ema, macd, rsi, sma
from core.strategy import Param, Strategy, register


def _realised_vol(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change().rolling(window).std()


@register
class VolatilityTargeted(Strategy):
    name = "Volatility-targeted trend"
    description = ("A trend rule whose position size scales inversely with "
                   "recent volatility, so each bar risks a similar amount. "
                   "Four parameters.")
    params = [
        Param("fast", 20, "Fast MA", "int", 2, 200, 1),
        Param("slow", 100, "Slow MA", "int", 5, 400, 1),
        Param("vol_window", 20, "Volatility lookback", "int", 5, 200, 1),
        Param("target_vol", 15.0, "Target annual volatility (%)", "float",
              1.0, 60.0, 1.0),
        Param("max_leverage", 1.0, "Cap on position size", "float", 0.25, 3.0, 0.25,
              help="1.0 means never more than fully invested"),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        f, s = sma(df["close"], int(self.fast)), sma(df["close"], int(self.slow))
        direction = (f > s).astype(float)

        vol = _realised_vol(df["close"], int(self.vol_window)) * np.sqrt(252.0)
        # Scale by the volatility known at this close. Using the realised vol
        # of the bar being sized would be lookahead, and a very effective one:
        # it shrinks exactly the bars that turn out to be large.
        scale = (float(self.target_vol) / 100.0) / vol.replace(0, np.nan)
        scale = scale.clip(upper=float(self.max_leverage)).fillna(0.0)

        return (direction * scale).where(f.notna() & s.notna(), 0.0)

    def indicators(self, df):
        vol = _realised_vol(df["close"], int(self.vol_window)) * np.sqrt(252.0) * 100
        return {"Annualised vol %": vol}


@register
class RegimeFilteredTrend(Strategy):
    name = "Regime-filtered trend"
    description = ("Trend-follow, but only while the market is in a calm "
                   "regime measured by its own volatility percentile. Stands "
                   "aside when volatility is in the top band. Four parameters.")
    params = [
        Param("fast", 20, "Fast MA", "int", 2, 200, 1),
        Param("slow", 100, "Slow MA", "int", 5, 400, 1),
        Param("vol_window", 20, "Volatility lookback", "int", 5, 200, 1),
        Param("rank_window", 252, "Percentile window", "int", 30, 1000, 1),
        Param("max_percentile", 80, "Stand aside above this vol percentile",
              "int", 10, 100, 5),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        f, s = sma(df["close"], int(self.fast)), sma(df["close"], int(self.slow))
        vol = _realised_vol(df["close"], int(self.vol_window))
        # Percentile against this bar's own trailing history only. A full-sample
        # percentile would know how volatile 2020 was going to be in 2015.
        pct = vol.rolling(int(self.rank_window)).rank(pct=True) * 100.0
        calm = pct <= float(self.max_percentile)
        return ((f > s) & calm).astype(float).where(
            f.notna() & s.notna() & pct.notna(), 0.0)

    def indicators(self, df):
        vol = _realised_vol(df["close"], int(self.vol_window))
        return {"Vol percentile": vol.rolling(int(self.rank_window)).rank(pct=True) * 100}


@register
class TrendStrengthGated(Strategy):
    name = "Trend strength gated"
    description = ("Only take the moving-average signal when ADX says a trend "
                   "actually exists. Source 11 pairs momentum indicators with "
                   "trending markets and oscillators with choppy ones; this is "
                   "that claim, made testable.")
    params = [
        Param("fast", 20, "Fast MA", "int", 2, 200, 1),
        Param("slow", 100, "Slow MA", "int", 5, 400, 1),
        Param("adx_window", 14, "ADX window", "int", 5, 100, 1),
        Param("adx_min", 25.0, "Minimum ADX to trade", "float", 5.0, 60.0, 1.0),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        f, s = sma(df["close"], int(self.fast)), sma(df["close"], int(self.slow))
        strength = adx(df, int(self.adx_window))
        if isinstance(strength, tuple):
            strength = strength[0]
        trending = strength >= float(self.adx_min)
        return ((f > s) & trending).astype(float).where(
            f.notna() & s.notna() & strength.notna(), 0.0)


@register
class EnsembleVote(Strategy):
    name = "Ensemble vote"
    description = ("Five independent rules vote; hold when enough agree. The "
                   "standard argument for ensembles is that errors cancel -- "
                   "which only holds if the rules are actually independent, "
                   "and trend rules rarely are.")
    params = [
        Param("threshold", 3, "Votes needed to hold", "int", 1, 5, 1),
        Param("fast", 20, "Fast MA", "int", 2, 200, 1),
        Param("slow", 100, "Slow MA", "int", 5, 400, 1),
        Param("rsi_window", 14, "RSI window", "int", 2, 100, 1),
        Param("breakout", 20, "Breakout lookback", "int", 5, 200, 1),
    ]

    def _votes(self, df: pd.DataFrame) -> pd.DataFrame:
        f, s = sma(df["close"], int(self.fast)), sma(df["close"], int(self.slow))
        line, signal, _ = macd(df["close"])
        hi = df["high"].rolling(int(self.breakout)).max().shift(1)
        return pd.DataFrame({
            "ma_cross": (f > s).astype(float),
            "above_slow": (df["close"] > s).astype(float),
            "macd": (line > signal).astype(float),
            "rsi": (rsi(df["close"], int(self.rsi_window)) > 50).astype(float),
            "breakout": (df["close"] > hi).astype(float),
        })

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        votes = self._votes(df)
        return (votes.sum(axis=1) >= int(self.threshold)).astype(float)

    def indicators(self, df):
        return {"Votes in favour": self._votes(df).sum(axis=1)}


@register
class DualMomentumSingle(Strategy):
    name = "Absolute momentum with trend filter"
    description = ("Hold only when BOTH the trailing return is positive and "
                   "price is above its long average. Antonacci's absolute "
                   "momentum, reduced to one instrument.")
    params = [
        Param("lookback", 252, "Return lookback", "int", 20, 756, 1),
        Param("ma_window", 200, "Trend filter", "int", 20, 400, 1),
    ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        trailing = df["close"].pct_change(int(self.lookback))
        ma = sma(df["close"], int(self.ma_window))
        return ((trailing > 0) & (df["close"] > ma)).astype(float).where(
            trailing.notna() & ma.notna(), 0.0)


@register
class MeanReversionZScore(Strategy):
    name = "Z-score mean reversion"
    description = ("Buy when price is unusually far below its own mean in "
                   "standard deviations, exit on the way back. The statistical "
                   "form of 'buy the dip', so the two can be compared.")
    params = [
        Param("window", 20, "Lookback", "int", 5, 250, 1),
        Param("entry_z", -2.0, "Buy below this z-score", "float", -5.0, 0.0, 0.25),
        Param("exit_z", -0.5, "Exit above this z-score", "float", -3.0, 3.0, 0.25),
        Param("trend_filter", False, "Only in an uptrend", "bool",
              help="Requires price above its 200-bar average. Mean reversion "
                   "into a downtrend is how a dip becomes a hole"),
    ]

    def _z(self, df):
        m = df["close"].rolling(int(self.window)).mean()
        sd = df["close"].rolling(int(self.window)).std()
        return (df["close"] - m) / sd.replace(0, np.nan)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        z = self._z(df)
        sig = pd.Series(np.nan, index=df.index)
        sig[z <= float(self.entry_z)] = 1.0
        sig[z >= float(self.exit_z)] = 0.0
        out = sig.ffill().fillna(0.0)
        if self.trend_filter:
            out = out.where(df["close"] > sma(df["close"], 200), 0.0)
        return out.where(z.notna(), 0.0)

    def indicators(self, df):
        return {"Z-score": self._z(df)}


@register
class WalkForwardLogit(Strategy):
    name = "Walk-forward logistic"
    description = ("A logistic regression on a handful of technical features, "
                   "refitted on an expanding window of past bars only. The "
                   "honest version of 'let the machine find the pattern'.")
    params = [
        Param("min_train", 500, "Bars before the first fit", "int", 100, 2000, 50),
        Param("refit_every", 63, "Refit interval", "int", 5, 252, 1,
              help="Refitting every bar is slower and changes very little"),
        Param("threshold", 0.55, "Probability needed to hold", "float",
              0.5, 0.8, 0.01),
    ]

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        feats = pd.DataFrame({
            "ret1": close.pct_change(),
            "ret5": close.pct_change(5),
            "ret20": close.pct_change(20),
            "rsi": rsi(close, 14) / 100.0,
            "ma_gap": close / sma(close, 50) - 1.0,
            "vol": _realised_vol(close, 20),
            "range": (df["high"] - df["low"]) / close,
        })
        return feats

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            # No sklearn: stand aside rather than silently degrade to a
            # different rule wearing this one's name.
            return pd.Series(0.0, index=df.index)

        X = self._features(df)
        # The label is the NEXT bar's direction. It is only ever used to fit on
        # rows strictly before the bar being predicted.
        y = (df["close"].shift(-1) > df["close"]).astype(int)

        sig = pd.Series(0.0, index=df.index)
        n = len(df)
        start = int(self.min_train)
        if n <= start + 1:
            return sig

        model = scaler = None
        for i in range(start, n):
            if model is None or (i - start) % int(self.refit_every) == 0:
                # Rows 0..i-2 only. Row i-1's label depends on row i's close,
                # which is the bar being predicted -- including it leaks.
                tr = X.iloc[:i - 1].copy()
                lab = y.iloc[:i - 1]
                ok = tr.notna().all(axis=1) & lab.notna()
                tr, lab = tr[ok], lab[ok]
                if len(tr) < 100 or lab.nunique() < 2:
                    continue
                scaler = StandardScaler().fit(tr)
                model = LogisticRegression(max_iter=400, C=1.0).fit(
                    scaler.transform(tr), lab)

            row = X.iloc[[i]]
            if model is None or row.isna().any(axis=1).iloc[0]:
                continue
            p = model.predict_proba(scaler.transform(row))[0, 1]
            sig.iloc[i] = 1.0 if p >= float(self.threshold) else 0.0

        return sig


@register
class AdaptiveChannel(Strategy):
    name = "Adaptive channel"
    description = ("A breakout channel whose lookback lengthens when the "
                   "market is choppy and shortens when it trends, judged by "
                   "the efficiency ratio. Kaufman's adaptive idea applied to a "
                   "channel rather than an average.")
    params = [
        Param("base", 20, "Base lookback", "int", 5, 200, 1),
        Param("er_window", 20, "Efficiency-ratio window", "int", 5, 200, 1),
        Param("stretch", 3.0, "How far the lookback may stretch", "float",
              1.0, 6.0, 0.5),
    ]

    def _efficiency(self, df: pd.DataFrame) -> pd.Series:
        n = int(self.er_window)
        direction = (df["close"] - df["close"].shift(n)).abs()
        volatility = df["close"].diff().abs().rolling(n).sum()
        return (direction / volatility.replace(0, np.nan)).clip(0, 1)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        er = self._efficiency(df)
        base = int(self.base)
        # Efficient (trending) -> shorter channel, reacts sooner.
        # Inefficient (choppy) -> longer channel, harder to trigger.
        length = (base * (1 + (1 - er.fillna(0.5)) * (float(self.stretch) - 1))).round()

        close = df["close"].to_numpy(float)
        highs = df["high"].to_numpy(float)
        lens = length.fillna(base).to_numpy(int)
        n = len(df)
        sig = np.zeros(n)
        for i in range(n):
            L = max(int(lens[i]), 2)
            if i < L + 1:
                continue
            # Window ends at i-1: this bar's own high cannot define the level
            # this bar is asked to break.
            level = highs[i - L:i].max()
            sig[i] = 1.0 if close[i] > level else 0.0
        return pd.Series(sig, index=df.index)

    def indicators(self, df):
        return {"Efficiency ratio": self._efficiency(df)}
