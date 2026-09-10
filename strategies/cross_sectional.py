"""Cross-sectional strategies -- rank a universe, hold the best.

These express what a single-symbol strategy structurally cannot: the signal is
the *comparison between names*, not any one name's own history.

Two of these -- momentum and low volatility -- are among the most replicated
findings in finance. That is a reason to test them carefully, not a reason to
expect them to work on your universe over your window. Published anomalies get
arbitraged, and a five-stock universe is not a cross-section.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy import Param

REGISTRY: dict[str, type] = {}


def register_xs(cls):
    REGISTRY[cls.name] = cls
    return cls


class CrossSectionalStrategy:
    """Base class. Subclasses turn a Panel into a weight matrix."""

    name: str = "Unnamed"
    description: str = ""
    params: list[Param] = []

    def __init__(self, **kwargs):
        for p in self.params:
            setattr(self, p.name, kwargs.get(p.name, p.default))
        unknown = set(kwargs) - {p.name for p in self.params}
        if unknown:
            raise TypeError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")

    def generate_weights(self, panel) -> pd.DataFrame:
        """Target weight per symbol per bar. Row t is filled at bar t+1's open."""
        raise NotImplementedError

    @property
    def settings(self) -> dict:
        return {p.name: getattr(self, p.name) for p in self.params}

    def __repr__(self):
        bits = ", ".join(f"{k}={v}" for k, v in self.settings.items())
        return f"{self.name}({bits})"

    # -- helpers shared by the ranked strategies ---------------------------
    @staticmethod
    def _rebalance_mask(index: pd.DatetimeIndex, every: int) -> np.ndarray:
        """True on bars where the portfolio is allowed to change.

        `every` is in bars; 21 is roughly monthly. Holding weights constant
        between rebalances is what keeps turnover -- and therefore cost --
        anywhere near realistic.
        """
        mask = np.zeros(len(index), dtype=bool)
        mask[::max(int(every), 1)] = True
        return mask

    def _top_n_weights(self, scores: pd.DataFrame, valid: pd.DataFrame,
                       n: int, every: int, ascending: bool) -> pd.DataFrame:
        """Equal-weight the best `n` names, refreshed every `every` bars."""
        weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        rebalance = self._rebalance_mask(scores.index, every)

        current = pd.Series(0.0, index=scores.columns)
        for i in range(len(scores)):
            if rebalance[i]:
                row = scores.iloc[i].where(valid.iloc[i])
                row = row.dropna()
                if len(row) >= 2:
                    ranked = row.sort_values(ascending=ascending)
                    picks = ranked.head(min(int(n), len(ranked))).index
                    current = pd.Series(0.0, index=scores.columns)
                    if len(picks):
                        current[picks] = 1.0 / len(picks)
            weights.iloc[i] = current.to_numpy()
        return weights


@register_xs
class CrossSectionalMomentum(CrossSectionalStrategy):
    name = "Cross-sectional momentum"
    description = ("Rank the universe by trailing return, hold the strongest N. "
                   "The classic version skips the most recent month, because "
                   "very-short-term returns tend to reverse rather than persist.")
    params = [
        Param("lookback", 252, "Lookback (bars)", "int", 20, 500, 1,
              help="252 bars is the standard 12-month formation window"),
        Param("skip", 21, "Skip most recent (bars)", "int", 0, 60, 1,
              help="Excludes the last month to avoid short-term reversal"),
        Param("top_n", 5, "Names to hold", "int", 1, 50, 1),
        Param("rebalance_every", 21, "Rebalance every (bars)", "int", 1, 126, 1),
        Param("min_history", 0, "Require full lookback", "int", 0, 1, 1),
    ]

    def generate_weights(self, panel) -> pd.DataFrame:
        closes = panel.closes
        lookback, skip = int(self.lookback), int(self.skip)

        past = closes.shift(lookback + skip)
        recent = closes.shift(skip)
        scores = recent / past - 1.0

        valid = scores.notna() & panel.tradeable()
        if int(self.min_history):
            valid &= closes.shift(lookback + skip).notna()

        return self._top_n_weights(scores, valid, int(self.top_n),
                                   int(self.rebalance_every), ascending=False)


@register_xs
class LowVolatility(CrossSectionalStrategy):
    name = "Low volatility"
    description = ("Hold the least volatile N names. The low-volatility anomaly "
                   "is the finding that these have historically returned as much "
                   "as high-volatility names with far less risk -- which basic "
                   "theory says should not happen.")
    params = [
        Param("lookback", 126, "Volatility window (bars)", "int", 20, 500, 1),
        Param("top_n", 5, "Names to hold", "int", 1, 50, 1),
        Param("rebalance_every", 21, "Rebalance every (bars)", "int", 1, 126, 1),
    ]

    def generate_weights(self, panel) -> pd.DataFrame:
        returns = panel.closes.pct_change()
        vol = returns.rolling(int(self.lookback), min_periods=int(self.lookback)).std()
        valid = vol.notna() & (vol > 0) & panel.tradeable()
        return self._top_n_weights(vol, valid, int(self.top_n),
                                   int(self.rebalance_every), ascending=True)


@register_xs
class CrossSectionalReversal(CrossSectionalStrategy):
    name = "Cross-sectional reversal"
    description = ("Buy the recent worst performers. The short-horizon mirror of "
                   "momentum -- over days to weeks, relative losers have tended "
                   "to bounce back.")
    params = [
        Param("lookback", 21, "Lookback (bars)", "int", 2, 126, 1),
        Param("top_n", 5, "Names to hold", "int", 1, 50, 1),
        Param("rebalance_every", 5, "Rebalance every (bars)", "int", 1, 63, 1),
    ]

    def generate_weights(self, panel) -> pd.DataFrame:
        closes = panel.closes
        scores = closes / closes.shift(int(self.lookback)) - 1.0
        valid = scores.notna() & panel.tradeable()
        return self._top_n_weights(scores, valid, int(self.top_n),
                                   int(self.rebalance_every), ascending=True)


@register_xs
class RiskParityLite(CrossSectionalStrategy):
    name = "Inverse volatility"
    description = ("Hold everything, weighted by the inverse of each name's "
                   "volatility. Not a ranking bet -- a way to stop one wild "
                   "name dominating the portfolio's risk.")
    params = [
        Param("lookback", 63, "Volatility window (bars)", "int", 20, 252, 1),
        Param("rebalance_every", 21, "Rebalance every (bars)", "int", 1, 126, 1),
    ]

    def generate_weights(self, panel) -> pd.DataFrame:
        returns = panel.closes.pct_change()
        vol = returns.rolling(int(self.lookback), min_periods=int(self.lookback)).std()
        inv = (1.0 / vol.where(vol > 0)).where(panel.tradeable())

        weights = pd.DataFrame(0.0, index=vol.index, columns=vol.columns)
        rebalance = self._rebalance_mask(vol.index, int(self.rebalance_every))
        current = pd.Series(0.0, index=vol.columns)

        for i in range(len(vol)):
            if rebalance[i]:
                row = inv.iloc[i].dropna()
                if row.sum() > 0:
                    current = pd.Series(0.0, index=vol.columns)
                    current[row.index] = row / row.sum()
            weights.iloc[i] = current.to_numpy()
        return weights


@register_xs
class EqualWeightAll(CrossSectionalStrategy):
    name = "Equal weight all"
    description = ("Hold every name equally. The benchmark a ranked strategy has "
                   "to beat -- if it cannot, the ranking is adding nothing.")
    params = [
        Param("rebalance_every", 21, "Rebalance every (bars)", "int", 1, 252, 1),
    ]

    def generate_weights(self, panel) -> pd.DataFrame:
        valid = panel.tradeable()
        weights = pd.DataFrame(0.0, index=valid.index, columns=valid.columns)
        rebalance = self._rebalance_mask(valid.index, int(self.rebalance_every))
        current = pd.Series(0.0, index=valid.columns)

        for i in range(len(valid)):
            if rebalance[i]:
                live = valid.iloc[i]
                if live.any():
                    current = pd.Series(0.0, index=valid.columns)
                    current[live] = 1.0 / int(live.sum())
            weights.iloc[i] = current.to_numpy()
        return weights


def available_xs() -> list[str]:
    return sorted(REGISTRY)


def get_xs_strategy(name: str):
    if name not in REGISTRY:
        raise KeyError(f"No cross-sectional strategy {name!r}. "
                       f"Available: {available_xs()}")
    return REGISTRY[name]


@register_xs
class DualMomentum(CrossSectionalStrategy):
    name = "Dual momentum"
    description = ("Gary Antonacci's Global Equities Momentum. Two filters, not "
                   "one: pick the strongest asset by RELATIVE momentum, but only "
                   "hold it if its ABSOLUTE momentum is positive -- otherwise go "
                   "to cash. The absolute filter is the whole point; it is what "
                   "sidesteps bear markets that a pure ranking rides straight down.")
    params = [
        Param("lookback", 252, "Lookback (bars)", "int", 20, 500, 1,
              help="Antonacci uses 12 months"),
        Param("top_n", 1, "Names to hold", "int", 1, 20, 1),
        Param("rebalance_every", 21, "Rebalance every (bars)", "int", 1, 126, 1),
        Param("absolute_filter", 1, "Require positive absolute momentum", "int", 0, 1, 1),
    ]

    def generate_weights(self, panel) -> pd.DataFrame:
        closes = panel.closes
        lookback = int(self.lookback)
        scores = closes / closes.shift(lookback) - 1.0
        valid = scores.notna() & panel.tradeable()

        weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        rebalance = self._rebalance_mask(scores.index, int(self.rebalance_every))
        current = pd.Series(0.0, index=scores.columns)
        require_positive = bool(int(self.absolute_filter))

        for i in range(len(scores)):
            if rebalance[i]:
                row = scores.iloc[i].where(valid.iloc[i]).dropna()
                current = pd.Series(0.0, index=scores.columns)
                if len(row) >= 2:
                    picks = row.sort_values(ascending=False).head(
                        min(int(self.top_n), len(row)))
                    if require_positive:
                        picks = picks[picks > 0]     # else stay in cash
                    if len(picks):
                        current[picks.index] = 1.0 / len(picks)
            weights.iloc[i] = current.to_numpy()
        return weights


@register_xs
class PairsTrading(CrossSectionalStrategy):
    name = "Pairs trading"
    description = ("Statistical arbitrage on the most correlated pair in the "
                   "universe. When their price ratio strays far enough from its "
                   "own mean, buy the laggard and short the leader, betting the "
                   "gap closes. Market-neutral by construction -- it needs "
                   "allow_short enabled to express the other leg.")
    params = [
        Param("formation", 252, "Formation window (bars)", "int", 60, 750, 1,
              help="History used to pick the pair and measure its normal spread"),
        Param("zscore_window", 63, "Z-score window (bars)", "int", 10, 252, 1),
        Param("entry_z", 2.0, "Entry z-score", "float", 0.5, 4.0, 0.25),
        Param("exit_z", 0.5, "Exit z-score", "float", 0.0, 2.0, 0.25),
    ]

    def generate_weights(self, panel) -> pd.DataFrame:
        closes = panel.closes
        symbols = panel.symbols
        weights = pd.DataFrame(0.0, index=closes.index, columns=symbols)
        formation = int(self.formation)
        if len(closes) <= formation + 10 or len(symbols) < 2:
            return weights

        # pick the pair on the formation window only -- choosing it using the
        # whole history would be lookahead dressed up as pair selection
        window = closes.iloc[:formation].pct_change().dropna(how="all")
        correlations = window.corr()
        best, best_corr = None, -np.inf
        for a_i, a in enumerate(symbols):
            for b in symbols[a_i + 1:]:
                value = correlations.loc[a, b]
                if pd.notna(value) and value > best_corr:
                    best, best_corr = (a, b), value
        if best is None:
            return weights

        left, right = best
        ratio = np.log(closes[left] / closes[right])
        mean = ratio.rolling(int(self.zscore_window), min_periods=int(self.zscore_window)).mean()
        sd = ratio.rolling(int(self.zscore_window), min_periods=int(self.zscore_window)).std(ddof=0)
        z = ((ratio - mean) / sd.replace(0, np.nan)).to_numpy()

        entry, exit_z = float(self.entry_z), float(self.exit_z)
        position = 0.0
        for i in range(len(closes)):
            if i < formation or np.isnan(z[i]):
                continue
            if position != 0.0 and abs(z[i]) <= exit_z:
                position = 0.0
            elif position == 0.0:
                if z[i] >= entry:
                    position = -1.0          # left rich: short it, buy right
                elif z[i] <= -entry:
                    position = 1.0           # left cheap: buy it, short right

            weights.iloc[i, weights.columns.get_loc(left)] = 0.5 * position
            weights.iloc[i, weights.columns.get_loc(right)] = -0.5 * position
        return weights


@register_xs
class ResidualMomentum(CrossSectionalStrategy):
    name = "Residual momentum"
    description = ("Rank on the part of each stock's return the market does NOT "
                   "explain. Plain momentum quietly favours high-beta names, so "
                   "in a rising market it is partly a leveraged index bet; "
                   "regressing that out leaves stock-specific momentum. "
                   "Kakushadze and Serur, 151 Trading Strategies, section 3.7.")
    params = [
        Param("beta_window", 756, "Beta estimation window (bars)", "int", 120, 1000, 1,
              help="The paper uses 36 months"),
        Param("formation", 252, "Formation window (bars)", "int", 40, 500, 1,
              help="The paper uses 12 months"),
        Param("skip", 21, "Skip most recent (bars)", "int", 0, 60, 1),
        Param("top_n", 3, "Names per side", "int", 1, 50, 1),
        Param("rebalance_every", 21, "Rebalance every (bars)", "int", 1, 126, 1),
        Param("dollar_neutral", True, "Short the bottom ranks too", "bool"),
    ]

    def generate_weights(self, panel) -> pd.DataFrame:
        returns = panel.closes.pct_change()
        # No Fama-French factors here, so the market proxy is the equal-weighted
        # universe. A single-factor simplification of the paper's three-factor
        # regression -- it removes market beta but not size or value tilts.
        market = returns.mean(axis=1, skipna=True)

        beta_window = int(self.beta_window)
        cov = returns.rolling(beta_window, min_periods=beta_window // 2).cov(market)
        var = market.rolling(beta_window, min_periods=beta_window // 2).var()
        beta = cov.div(var.replace(0, np.nan), axis=0)

        # residual return: what the market did not account for
        residual = returns.sub(beta.mul(market, axis=0))

        formation, skip = int(self.formation), int(self.skip)
        mean = residual.rolling(formation, min_periods=formation // 2).mean()
        sd = residual.rolling(formation, min_periods=formation // 2).std(ddof=1)
        # risk-adjusted residual return, as of `skip` bars ago
        scores = (mean / sd.replace(0, np.nan)).shift(skip)

        valid = scores.notna() & panel.tradeable()
        if not self.dollar_neutral:
            return self._top_n_weights(scores, valid, int(self.top_n),
                                       int(self.rebalance_every), ascending=False)

        weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        rebalance = self._rebalance_mask(scores.index, int(self.rebalance_every))
        current = pd.Series(0.0, index=scores.columns)
        n = int(self.top_n)

        for i in range(len(scores)):
            if rebalance[i]:
                row = scores.iloc[i].where(valid.iloc[i]).dropna()
                current = pd.Series(0.0, index=scores.columns)
                if len(row) >= 2 * n:
                    ranked = row.sort_values(ascending=False)
                    longs, shorts = ranked.head(n).index, ranked.tail(n).index
                    current[longs] = 0.5 / n
                    current[shorts] = -0.5 / n
            weights.iloc[i] = current.to_numpy()
        return weights


@register_xs
class ClusterMeanReversion(CrossSectionalStrategy):
    name = "Cluster mean reversion"
    description = ("Pairs trading generalised past two names. Demean the "
                   "cluster's returns, then hold each stock in proportion to how "
                   "far it strayed -- short the leaders, buy the laggards. "
                   "Dollar-neutral by construction. Kakushadze and Serur, "
                   "151 Trading Strategies, section 3.9.")
    params = [
        Param("lookback", 5, "Return window (bars)", "int", 1, 60, 1,
              help="Short windows suit this; reversal is a days-to-weeks effect"),
        Param("rebalance_every", 5, "Rebalance every (bars)", "int", 1, 63, 1),
        Param("gross", 1.0, "Gross exposure", "float", 0.1, 1.0, 0.1,
              help="Total of the absolute dollar positions"),
    ]

    def generate_weights(self, panel) -> pd.DataFrame:
        closes = panel.closes
        lookback = int(self.lookback)
        log_returns = np.log(closes / closes.shift(lookback))
        demeaned = log_returns.sub(log_returns.mean(axis=1, skipna=True), axis=0)
        demeaned = demeaned.where(panel.tradeable())

        weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        rebalance = self._rebalance_mask(closes.index, int(self.rebalance_every))
        current = pd.Series(0.0, index=closes.columns)
        gross = float(self.gross)

        for i in range(len(closes)):
            if rebalance[i]:
                row = demeaned.iloc[i].dropna()
                current = pd.Series(0.0, index=closes.columns)
                if len(row) >= 3:
                    scale = row.abs().sum()
                    if scale > 0:
                        # positions opposite the deviation; sums to zero, so the
                        # book is dollar-neutral without an extra constraint
                        current[row.index] = -gross * row / scale
            weights.iloc[i] = current.to_numpy()
        return weights
