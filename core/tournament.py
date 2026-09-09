"""Walk-forward evaluation and the strategy tournament.

The problem this solves: if you search a thousand strategy/parameter
combinations over one stretch of history and keep the best, you have not found
an edge -- you have found the combination that best fits that particular noise.
It will look brilliant and then lose money.

Walk-forward fixes it by never scoring a strategy on data used to choose its
parameters:

    |--- train ---|- test -|                     fold 1
              |--- train ---|- test -|           fold 2
                        |--- train ---|- test -| fold 3

Parameters are optimised on each train window, then applied *unchanged* to the
test window that follows it. Only those test segments are stitched together
into the reported equity curve. That curve is what the leaderboard ranks on, so
a strategy earns its place by performing on data it has never seen.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import bars_per_year
from .engine import BacktestConfig, run_backtest
from .metrics import equity_stats
from .strategy import Param, Strategy


@dataclass
class WalkForwardConfig:
    train_bars: int = 504          # ~2 years of daily bars
    test_bars: int = 126           # ~6 months held out
    step_bars: int | None = None   # defaults to test_bars (non-overlapping tests)
    anchored: bool = False         # True keeps every fold's train window starting at bar 0
    min_folds: int = 2
    rank_metric: str = "Sharpe"
    max_grid: int = 240            # hard cap on parameter combinations per fold


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict
    in_sample: float               # rank metric on the train window
    out_of_sample: float           # same metric on the test window
    returns: pd.Series = field(repr=False, default_factory=pd.Series)


@dataclass
class WalkForwardResult:
    strategy: str
    symbol: str
    folds: list[FoldResult]
    equity: pd.Series              # stitched out-of-sample equity curve
    stats: dict
    param_stability: float         # 0..1, how often the chosen params stayed put
    edge_fraction: float = 0.0     # 0..1, share of chosen params pinned to a grid edge

    @property
    def chosen_params(self) -> dict:
        """The most recent fold's parameters -- what you would trade next."""
        return self.folds[-1].params if self.folds else {}


def auto_grid(cls: type[Strategy], levels: int = 4, max_combos: int = 240) -> list[dict]:
    """Build a coarse parameter grid from a strategy's own Param declarations."""
    axes: list[list] = []
    names: list[str] = []

    for p in cls.params:
        if p.kind == "bool":
            values = [False, True]
        elif p.kind == "choice":
            values = list(p.choices) or [p.default]
        elif p.min is None or p.max is None:
            values = [p.default]
        else:
            raw = np.linspace(float(p.min), float(p.max), levels)
            values = sorted({int(round(v)) for v in raw}) if p.kind == "int" \
                else sorted({round(float(v), 4) for v in raw})
        names.append(p.name)
        axes.append(values)

    if not axes:
        return [{}]

    combos = [dict(zip(names, c)) for c in itertools.product(*axes)]
    if len(combos) > max_combos:                 # thin evenly rather than truncating
        idx = np.linspace(0, len(combos) - 1, max_combos).astype(int)
        combos = [combos[i] for i in sorted(set(idx))]
    return combos


def make_folds(n_bars: int, cfg: WalkForwardConfig) -> list[tuple[int, int, int, int]]:
    """(train_start, train_end, test_start, test_end) index tuples."""
    step = cfg.step_bars or cfg.test_bars
    folds = []
    train_end = cfg.train_bars
    while train_end + cfg.test_bars <= n_bars:
        train_start = 0 if cfg.anchored else max(0, train_end - cfg.train_bars)
        folds.append((train_start, train_end, train_end, train_end + cfg.test_bars))
        train_end += step
    return folds


def edge_fraction(params: dict, grid: list[dict]) -> float:
    """How much of the chosen parameter set sits at the edge of the search grid.

    A parameter pinned to its minimum or maximum usually means one of two
    things: the optimum lies outside the range you searched, or the parameter
    does not matter and the optimiser is picking arbitrarily. Either way the
    number on the leaderboard is not measuring what its name suggests -- an RSI
    rule that optimises to "oversold=50, exit=95" has quietly become
    buy-and-hold.
    """
    if not params or not grid:
        return 0.0
    at_edge = 0
    for name, value in params.items():
        axis = sorted({c[name] for c in grid if name in c},
                      key=lambda v: (v is True, v))
        if len(axis) < 2:
            continue
        if value in (axis[0], axis[-1]):
            at_edge += 1
    return at_edge / max(len(params), 1)


def _score(equity: pd.Series, metric: str, ppy: float) -> float:
    stats = equity_stats(equity, ppy)
    value = stats.get(metric)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return -np.inf
    return float(value)


def walk_forward(df: pd.DataFrame, cls: type[Strategy], symbol: str = "",
                 grid: list[dict] | None = None,
                 wf: WalkForwardConfig | None = None,
                 backtest: BacktestConfig | None = None) -> WalkForwardResult | None:
    """Optimise on each train window, score on the untouched test window."""
    wf = wf or WalkForwardConfig()
    backtest = backtest or BacktestConfig()
    grid = grid if grid is not None else auto_grid(cls, max_combos=wf.max_grid)

    folds_idx = make_folds(len(df), wf)
    if len(folds_idx) < wf.min_folds:
        return None                    # not enough history to judge anything

    ppy = bars_per_year(df.index)
    results: list[FoldResult] = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds_idx):
        train = df.iloc[tr_s:tr_e]

        best_params, best_score = None, -np.inf
        for params in grid:
            try:
                strat = cls(**params)
                res = run_backtest(train, strat.generate_signals(train), backtest)
            except Exception:  # noqa: BLE001 -- a bad combo must not sink the sweep
                continue
            score = _score(res.equity, wf.rank_metric, ppy)
            if score > best_score:
                best_params, best_score = params, score

        if best_params is None:
            continue

        # Re-run across train+test so indicators are warm, then keep only the
        # test segment. The parameters are frozen -- the test window never
        # influenced them.
        window = df.iloc[tr_s:te_e]
        strat = cls(**best_params)
        res = run_backtest(window, strat.generate_signals(window), backtest)
        oos_returns = res.returns.loc[df.index[te_s]:df.index[te_e - 1]]

        oos_equity = backtest.initial_cash * (1 + oos_returns).cumprod()
        results.append(FoldResult(
            fold=i, train_start=df.index[tr_s], train_end=df.index[tr_e - 1],
            test_start=df.index[te_s], test_end=df.index[te_e - 1],
            params=best_params, in_sample=best_score,
            out_of_sample=_score(oos_equity, wf.rank_metric, ppy),
            returns=oos_returns,
        ))

    if not results:
        return None

    stitched = pd.concat([f.returns for f in results])
    stitched = stitched[~stitched.index.duplicated(keep="first")].sort_index()
    equity = backtest.initial_cash * (1 + stitched).cumprod()

    changes = sum(1 for a, b in zip(results, results[1:]) if a.params != b.params)
    stability = 1.0 - (changes / max(len(results) - 1, 1))

    return WalkForwardResult(
        strategy=cls.name, symbol=symbol.upper(), folds=results,
        equity=equity, stats=equity_stats(equity, ppy), param_stability=stability,
        edge_fraction=edge_fraction(results[-1].params, grid),
    )


def run_tournament(data: dict[str, pd.DataFrame], strategies: list[type[Strategy]],
                   wf: WalkForwardConfig | None = None,
                   backtest: BacktestConfig | None = None,
                   progress=None) -> pd.DataFrame:
    """Every strategy against every symbol, ranked on out-of-sample results only."""
    wf = wf or WalkForwardConfig()
    rows, total, done = [], len(data) * len(strategies), 0

    for symbol, df in data.items():
        for cls in strategies:
            done += 1
            if progress:
                progress(done, total, f"{cls.name} on {symbol}")
            if not cls.can_run_on(df):
                continue           # e.g. a news strategy on bars carrying no news
            try:
                res = walk_forward(df, cls, symbol, wf=wf, backtest=backtest)
            except Exception as exc:  # noqa: BLE001
                rows.append({"strategy": cls.name, "symbol": symbol,
                             "error": str(exc)[:80]})
                continue
            if res is None:
                continue

            s = res.stats
            rows.append({
                "strategy": res.strategy,
                "symbol": res.symbol,
                "oos_return_%": s.get("Total return", 0.0) * 100,
                "oos_sharpe": s.get("Sharpe", 0.0),
                "oos_max_dd_%": s.get("Max drawdown", 0.0) * 100,
                "oos_calmar": s.get("Calmar", 0.0),
                "folds": len(res.folds),
                "fold_win_rate": float(np.mean([f.out_of_sample > 0 for f in res.folds])),
                "param_stability": res.param_stability,
                "params_at_edge": res.edge_fraction,
                "params": res.chosen_params,
            })

    table = pd.DataFrame(rows)
    if "oos_sharpe" in table.columns:
        table = table.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
    return table


def leaders(table: pd.DataFrame, n: int = 5, min_folds: int = 3,
            min_fold_win_rate: float = 0.5,
            max_edge_fraction: float = 0.75) -> pd.DataFrame:
    """Top N, after discarding results too thin or too erratic to believe.

    A high Sharpe from two folds, or one that came from a single lucky window,
    is not evidence. These filters are the difference between a leaderboard and
    a list of coincidences.
    """
    if table.empty or "oos_sharpe" not in table.columns:
        return table
    keep = table[
        (table["folds"] >= min_folds)
        & (table["fold_win_rate"] >= min_fold_win_rate)
        & (table["oos_sharpe"] > 0)
        & (table.get("params_at_edge", pd.Series(0.0, index=table.index))
           <= max_edge_fraction)
    ]
    return keep.head(n).reset_index(drop=True)
