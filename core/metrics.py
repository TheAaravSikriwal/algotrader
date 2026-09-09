"""Performance statistics computed from an equity curve and a trade log."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import bars_per_year


def _safe(x, default=0.0):
    return default if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x


def equity_stats(equity: pd.Series, periods_per_year: float | None = None) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}

    ppy = periods_per_year or bars_per_year(equity.index)
    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0 if equity.iloc[0] > 0 else 0.0

    vol = rets.std(ddof=1) * np.sqrt(ppy)
    sharpe = (rets.mean() * ppy) / vol if vol > 0 else 0.0

    downside = rets[rets < 0]
    dvol = downside.std(ddof=1) * np.sqrt(ppy) if len(downside) > 1 else 0.0
    sortino = (rets.mean() * ppy) / dvol if dvol > 0 else 0.0

    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()

    # longest stretch below the previous high-water mark
    under = dd < -1e-12
    longest, run = 0, 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)

    return {
        "Total return": total_return,
        "CAGR": _safe(cagr),
        "Sharpe": _safe(sharpe),
        "Sortino": _safe(sortino),
        "Volatility (ann.)": _safe(vol),
        "Max drawdown": _safe(max_dd),
        "Calmar": _safe(cagr / abs(max_dd)) if max_dd < -1e-9 else 0.0,
        "Longest drawdown (bars)": int(longest),
        "Final equity": float(equity.iloc[-1]),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"Trades": 0, "Win rate": 0.0, "Profit factor": 0.0,
                "Avg trade": 0.0, "Best trade": 0.0, "Worst trade": 0.0,
                "Avg bars held": 0.0, "Total commissions": 0.0}

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_win = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())

    return {
        "Trades": int(len(trades)),
        "Win rate": len(wins) / len(trades),
        "Profit factor": _safe(gross_win / gross_loss) if gross_loss > 1e-9 else (np.inf if gross_win > 0 else 0.0),
        "Avg trade": float(trades["pnl"].mean()),
        "Avg win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "Avg loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "Best trade": float(trades["pnl"].max()),
        "Worst trade": float(trades["pnl"].min()),
        "Avg bars held": float(trades["bars_held"].mean()),
        "Total commissions": float(trades["fees"].sum()),
    }


def summarise(result, periods_per_year: float | None = None) -> dict:
    """Full report: strategy stats, trade stats, and the buy-and-hold baseline."""
    ppy = periods_per_year or bars_per_year(result.equity.index)
    stats = equity_stats(result.equity, ppy)
    stats.update(trade_stats(result.trades))
    stats["Time in market"] = float((result.exposure.abs() > 1e-12).mean())

    if result.benchmark is not None and len(result.benchmark) > 1:
        bh = equity_stats(result.benchmark, ppy)
        stats["Buy & hold return"] = bh.get("Total return", 0.0)
        stats["Buy & hold Sharpe"] = bh.get("Sharpe", 0.0)
        stats["Buy & hold max DD"] = bh.get("Max drawdown", 0.0)
        stats["Excess return"] = stats["Total return"] - stats["Buy & hold return"]
    return stats


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Year x month grid of returns, as fractions."""
    if equity.empty:
        return pd.DataFrame()
    monthly = equity.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return pd.DataFrame()
    grid = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    return grid.pivot(index="year", columns="month", values="ret").sort_index(ascending=False)
