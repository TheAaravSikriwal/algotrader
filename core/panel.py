"""Cross-sectional backtesting -- many symbols at once.

The single-symbol engine answers "should I be long SPY today?". This one
answers "of these 100 stocks, which 10 should I hold?" -- which is the shape of
the best-replicated findings in finance (cross-sectional momentum, the
low-volatility anomaly). Those cannot be expressed one symbol at a time,
because the signal *is* the comparison between names.

Same honesty rules as the single-symbol engine: weights decided from bar t are
filled at the open of bar t+1, and a symbol with no bar on a date simply cannot
be traded that day.

The bias to worry about here is survivorship, and it is not one the code can
fix. If you hand it today's index members and run them back ten years, you have
silently excluded everything that went bankrupt or got delisted, and the result
will look far better than reality. `Panel.survivorship_warning()` reports what
it can detect; the rest is on your choice of universe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Panel:
    """Aligned price history for a universe of symbols."""
    opens: pd.DataFrame
    highs: pd.DataFrame
    lows: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame

    @property
    def symbols(self) -> list[str]:
        return list(self.closes.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.closes.index)

    def __len__(self) -> int:
        return len(self.closes)

    @classmethod
    def from_bars(cls, bars: dict[str, pd.DataFrame]) -> "Panel":
        """Align per-symbol frames onto one shared date index."""
        if not bars:
            raise ValueError("panel needs at least one symbol")

        frames = {f: {} for f in FIELDS}
        for symbol, df in bars.items():
            if df is None or df.empty:
                continue
            symbol = symbol.upper()
            for f in FIELDS:
                frames[f][symbol] = df[f].astype(float)

        if not frames["close"]:
            raise ValueError("no usable bars in the panel")

        built = {f: pd.DataFrame(frames[f]).sort_index() for f in FIELDS}
        index = built["close"].index
        for f in FIELDS:
            built[f] = built[f].reindex(index)

        return cls(opens=built["open"], highs=built["high"], lows=built["low"],
                   closes=built["close"], volumes=built["volume"])

    def tradeable(self) -> pd.DataFrame:
        """True where a symbol has a real, positive open price that day."""
        return self.opens.notna() & (self.opens > 0)

    def coverage(self) -> pd.Series:
        """Fraction of the window each symbol actually has data for."""
        return self.closes.notna().mean().sort_values()

    def survivorship_warning(self) -> list[str]:
        """Detectable signs that the universe was chosen with hindsight."""
        notes = []
        coverage = self.coverage()

        late = coverage[coverage < 0.95]
        if len(late):
            notes.append(
                f"{len(late)} of {len(self.symbols)} symbols lack full history "
                f"(thinnest: {', '.join(late.head(3).index)}). Names that only "
                "appear partway through were still picked by you today.")

        if (coverage >= 0.999).all():
            notes.append(
                "Every symbol has complete history over the whole window -- which "
                "is exactly what a survivorship-biased universe looks like. "
                "Nothing here was delisted, acquired or went bankrupt, because "
                "you chose the list knowing which ones made it.")
        return notes


@dataclass
class PanelConfig:
    initial_cash: float = 100_000.0
    slippage_bps: float = 5.0
    commission_pct: float = 0.0
    min_weight_change: float = 0.001   # ignore rebalances smaller than this
    allow_short: bool = False
    max_gross: float = 1.0             # total absolute exposure cap


@dataclass
class PanelResult:
    equity: pd.Series
    weights: pd.DataFrame             # realised weights, per bar
    trades: pd.DataFrame
    turnover: pd.Series
    config: PanelConfig
    benchmark: pd.Series = field(default_factory=pd.Series)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    @property
    def drawdown(self) -> pd.Series:
        return self.equity / self.equity.cummax() - 1.0

    @property
    def annual_turnover(self) -> float:
        if self.turnover.empty:
            return 0.0
        years = max((self.turnover.index[-1] - self.turnover.index[0]).days / 365.25,
                    1e-9)
        return float(self.turnover.sum() / years)


TRADE_COLUMNS = ["timestamp", "symbol", "side", "qty", "price", "notional", "fees"]


def equal_weight_benchmark(panel: Panel, cash: float = 100_000.0) -> pd.Series:
    """Buy-and-hold an equal-weighted basket of the universe.

    This is the benchmark that matters for a ranked strategy. Beating SPY might
    just mean your universe outperformed; beating the equal-weighted universe
    means the *ranking* added something.
    """
    returns = panel.closes.pct_change()
    basket = returns.mean(axis=1, skipna=True).fillna(0.0)
    return (cash * (1 + basket).cumprod()).rename("equal_weight")


def run_panel_backtest(panel: Panel, weights: pd.DataFrame,
                       config: PanelConfig | None = None) -> PanelResult:
    """Simulate target weights across a universe.

    `weights` is indexed like the panel, one column per symbol, each value the
    fraction of equity to hold. Row t is filled at the open of bar t+1.
    """
    cfg = config or PanelConfig()
    if len(panel) < 2:
        raise ValueError("panel needs at least two bars")

    symbols = panel.symbols
    weights = weights.reindex(index=panel.index, columns=symbols).fillna(0.0)
    if not cfg.allow_short:
        weights = weights.clip(lower=0.0)

    # never commit more than max_gross in total
    gross = weights.abs().sum(axis=1)
    scale = np.where(gross > cfg.max_gross, cfg.max_gross / gross.replace(0, np.nan), 1.0)
    weights = weights.mul(pd.Series(scale, index=weights.index).fillna(1.0), axis=0)

    opens = panel.opens.to_numpy(dtype=float)
    closes = panel.closes.to_numpy(dtype=float)
    targets = weights.to_numpy(dtype=float)
    tradeable = panel.tradeable().to_numpy()
    index = panel.index
    n_bars, n_syms = closes.shape

    slip = cfg.slippage_bps / 10_000.0
    cash = cfg.initial_cash
    shares = np.zeros(n_syms)
    desired = np.zeros(n_syms)
    applied = np.zeros(n_syms)

    # last known price, so a halted or missing name still marks to something
    last_price = np.full(n_syms, np.nan)

    equity_curve = np.empty(n_bars)
    realised_weights = np.zeros((n_bars, n_syms))
    turnover = np.zeros(n_bars)
    trades: list[dict] = []

    for i in range(n_bars):
        px_open = opens[i]
        px_close = closes[i]
        live = np.where(np.isnan(px_close), last_price, px_close)
        last_price = np.where(np.isnan(px_close), last_price, px_close)

        # ---- fill the previous bar's decision at this open -----------------
        if np.any(np.abs(desired - applied) > cfg.min_weight_change):
            mark = np.where(np.isnan(px_open), live, px_open)
            equity_now = cash + np.nansum(shares * np.nan_to_num(mark))

            if equity_now > 0:
                traded_notional = 0.0
                for s in range(n_syms):
                    if not tradeable[i, s]:
                        continue          # no bar today: cannot trade this name
                    price = px_open[s]
                    target_shares = desired[s] * equity_now / price
                    delta = target_shares - shares[s]
                    if abs(delta * price) < cfg.min_weight_change * equity_now:
                        continue

                    exec_px = price * (1 + slip) if delta > 0 else price * (1 - slip)
                    fee = abs(delta * exec_px) * cfg.commission_pct
                    cash -= delta * exec_px + fee
                    shares[s] += delta
                    applied[s] = desired[s]
                    traded_notional += abs(delta * exec_px)

                    trades.append({
                        "timestamp": index[i], "symbol": symbols[s],
                        "side": "buy" if delta > 0 else "sell",
                        "qty": abs(delta), "price": exec_px,
                        "notional": abs(delta * exec_px), "fees": fee,
                    })

                # symbols that could not trade keep their old target on the books
                for s in range(n_syms):
                    if tradeable[i, s]:
                        applied[s] = desired[s]
                turnover[i] = traded_notional / equity_now if equity_now > 0 else 0.0

        equity = cash + np.nansum(shares * np.nan_to_num(live))
        equity_curve[i] = equity
        if equity > 0:
            realised_weights[i] = shares * np.nan_to_num(live) / equity

        desired = targets[i]

    return PanelResult(
        equity=pd.Series(equity_curve, index=index, name="equity"),
        weights=pd.DataFrame(realised_weights, index=index, columns=symbols),
        trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
        turnover=pd.Series(turnover, index=index, name="turnover"),
        config=cfg,
        benchmark=equal_weight_benchmark(panel, cfg.initial_cash),
    )
