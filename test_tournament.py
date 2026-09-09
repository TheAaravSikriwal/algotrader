"""Walk-forward and tournament checks. Synthetic data, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.engine import BacktestConfig
from core.strategy import Param, Strategy, get_strategy
from core.tournament import (WalkForwardConfig, auto_grid, leaders, make_folds,
                             run_tournament, walk_forward)

import strategies  # noqa: F401


def synth(n=1500, seed=1, drift=0.0004, vol=0.011):
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"open": prices, "high": prices * 1.004,
                         "low": prices * 0.996, "close": prices,
                         "volume": np.full(n, 1e6)}, index=idx)


def test_folds_never_overlap_train_and_test():
    folds = make_folds(1500, WalkForwardConfig(train_bars=500, test_bars=100))
    assert folds, "no folds produced"
    for tr_s, tr_e, te_s, te_e in folds:
        assert tr_e <= te_s, "train window bleeds into the test window"
        assert te_e <= 1500


def test_folds_advance_forward_in_time():
    folds = make_folds(1500, WalkForwardConfig(train_bars=500, test_bars=100))
    for a, b in zip(folds, folds[1:]):
        assert b[2] > a[2], "test windows are not moving forward"


def test_test_windows_do_not_overlap_each_other():
    folds = make_folds(2000, WalkForwardConfig(train_bars=500, test_bars=100))
    for a, b in zip(folds, folds[1:]):
        assert b[2] >= a[3], "test windows overlap, double-counting the same bars"


def test_anchored_mode_keeps_the_start_fixed():
    folds = make_folds(2000, WalkForwardConfig(train_bars=500, test_bars=100,
                                               anchored=True))
    assert all(f[0] == 0 for f in folds)


def test_auto_grid_respects_the_cap():
    cls = get_strategy("SMA crossover")
    grid = auto_grid(cls, levels=6, max_combos=20)
    assert 0 < len(grid) <= 20
    assert all(set(c) <= {p.name for p in cls.params} for c in grid)


def test_walk_forward_produces_out_of_sample_only():
    df = synth()
    res = walk_forward(df, get_strategy("SMA crossover"), "TEST",
                       wf=WalkForwardConfig(train_bars=400, test_bars=120))
    assert res is not None
    assert len(res.folds) >= 2
    # the stitched curve must start at the first test window, not at bar 0
    assert res.equity.index[0] >= res.folds[0].test_start
    assert res.equity.index[0] > df.index[0], "equity includes in-sample bars"


def test_walk_forward_needs_enough_history():
    short = synth(n=300)
    res = walk_forward(short, get_strategy("SMA crossover"), "TEST",
                       wf=WalkForwardConfig(train_bars=250, test_bars=120))
    assert res is None, "judged a strategy on too little data"


def test_params_are_frozen_within_a_fold():
    """Each fold reports one parameter set, chosen before its test window."""
    df = synth()
    res = walk_forward(df, get_strategy("SMA crossover"), "TEST",
                       wf=WalkForwardConfig(train_bars=400, test_bars=120))
    for fold in res.folds:
        assert isinstance(fold.params, dict) and fold.params
        assert fold.train_end < fold.test_start


def test_one_bar_lookahead_earns_nothing():
    """A signal built from the *next* bar's close is worthless here, because the
    engine fills at that same next open -- the move is already in the price you
    pay. This is the no-lookahead rule doing its job."""

    class PeeksOneBar(Strategy):
        name = "Peeks one bar (test)"
        params: list[Param] = []

        def generate_signals(self, df):
            return (df["close"].shift(-1) > df["close"]).astype(float).fillna(0.0)

    df = synth()
    res = walk_forward(df, PeeksOneBar, "TEST",
                       wf=WalkForwardConfig(train_bars=400, test_bars=120))
    assert res is not None
    assert all(abs(f.in_sample) < 3 for f in res.folds), \
        "a one-bar peek produced free money -- the execution delay is broken"


def test_real_lookahead_shows_up_in_sample():
    """A two-bar peek *does* beat the execution delay, and must show as an absurd
    in-sample score -- proving the harness would surface a leaky strategy."""

    class PeeksTwoBars(Strategy):
        name = "Peeks two bars (test)"
        params: list[Param] = []

        def generate_signals(self, df):
            future = df["close"].shift(-2) > df["close"].shift(-1)
            return future.astype(float).fillna(0.0)

    df = synth()
    res = walk_forward(df, PeeksTwoBars, "TEST",
                       wf=WalkForwardConfig(train_bars=400, test_bars=120))
    assert res is not None
    assert all(f.in_sample > 3 for f in res.folds), \
        "a genuine lookahead did not produce a suspicious in-sample Sharpe"


def test_tournament_ranks_by_out_of_sample_sharpe():
    data = {"AAA": synth(seed=2), "BBB": synth(seed=3, drift=0.0)}
    strats = [get_strategy(n) for n in ("SMA crossover", "MACD trend", "Buy and hold")]
    table = run_tournament(data, strats,
                           wf=WalkForwardConfig(train_bars=400, test_bars=120),
                           backtest=BacktestConfig(initial_cash=10_000))
    assert not table.empty
    assert "oos_sharpe" in table.columns
    sharpes = table["oos_sharpe"].dropna().tolist()
    assert sharpes == sorted(sharpes, reverse=True), "leaderboard is not sorted"


def test_leaders_filters_out_thin_evidence():
    table = pd.DataFrame([
        {"strategy": "lucky", "symbol": "A", "oos_sharpe": 9.0, "folds": 2,
         "fold_win_rate": 1.0},
        {"strategy": "erratic", "symbol": "B", "oos_sharpe": 4.0, "folds": 8,
         "fold_win_rate": 0.25},
        {"strategy": "solid", "symbol": "C", "oos_sharpe": 1.2, "folds": 8,
         "fold_win_rate": 0.75},
        {"strategy": "losing", "symbol": "D", "oos_sharpe": -0.5, "folds": 8,
         "fold_win_rate": 0.6},
    ])
    top = leaders(table, n=5)
    names = top["strategy"].tolist()
    assert names == ["solid"], f"kept unreliable entries: {names}"


def test_tournament_survives_a_broken_strategy():
    class Broken(Strategy):
        name = "Broken (test)"
        params: list[Param] = []

        def generate_signals(self, df):
            raise RuntimeError("boom")

    data = {"AAA": synth(seed=4)}
    table = run_tournament(data, [Broken, get_strategy("Buy and hold")],
                           wf=WalkForwardConfig(train_bars=400, test_bars=120))
    assert not table.empty, "one broken strategy killed the whole tournament"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc or 'assertion failed'}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print("\nall green" if not failures else f"\n{failures} failing")
    raise SystemExit(1 if failures else 0)
