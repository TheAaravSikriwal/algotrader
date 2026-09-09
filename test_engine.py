"""Engine correctness checks. Run with:  python -m pytest -q  (or  python test_engine.py)

These use synthetic bars, so they need no network and no API keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.engine import BacktestConfig, run_backtest
from core.metrics import summarise
from core.strategy import get_strategy

import strategies  # noqa: F401


def make_bars(closes, spread=0.0):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": c, "high": c + spread, "low": c - spread, "close": c,
        "volume": np.full(len(c), 1e6),
    }, index=idx)


def test_frictionless_buy_and_hold_matches_price():
    """With no costs, always-long must track the price exactly from bar 1 onward."""
    df = make_bars([100, 101, 103, 102, 108])
    signals = pd.Series(1.0, index=df.index)
    res = run_backtest(df, signals, BacktestConfig(initial_cash=1000, slippage_bps=0))

    # entry happens at bar 1's open (101), not bar 0 -- that is the no-lookahead rule
    assert res.position.iloc[0] == 0
    assert res.position.iloc[1] > 0
    expected = 1000 * df["close"].iloc[-1] / df["open"].iloc[1]
    assert abs(res.equity.iloc[-1] - expected) < 1e-6


def test_no_lookahead():
    """A one-bar spike the signal only sees at its close must not be captured."""
    df = make_bars([100, 100, 130, 100, 100])
    signals = pd.Series([0, 0, 1, 0, 0], index=df.index, dtype=float)
    res = run_backtest(df, signals, BacktestConfig(initial_cash=1000, slippage_bps=0))
    # signal fires on the spike bar, fills at the *next* open (100) -> no gain from it
    assert res.equity.iloc[-1] <= 1000 + 1e-9


def test_slippage_costs_money():
    df = make_bars([100] * 10)
    signals = pd.Series([1, 1, 0, 0, 1, 1, 0, 0, 0, 0], index=df.index, dtype=float)
    free = run_backtest(df, signals, BacktestConfig(initial_cash=1000, slippage_bps=0))
    costly = run_backtest(df, signals, BacktestConfig(initial_cash=1000, slippage_bps=25))
    assert abs(free.equity.iloc[-1] - 1000) < 1e-9      # flat price, no costs -> no change
    assert costly.equity.iloc[-1] < 1000                # every round trip pays the spread


def test_stop_loss_caps_the_loss():
    df = make_bars([100, 100, 100, 80, 60, 40])
    signals = pd.Series(1.0, index=df.index)
    res = run_backtest(df, signals, BacktestConfig(
        initial_cash=1000, slippage_bps=0, stop_loss_pct=10))

    stopped = res.trades[res.trades["exit_reason"] == "stop"]
    assert len(stopped) == 1
    assert abs(stopped.iloc[0]["return_pct"] + 10) < 1e-6   # exactly -10%
    assert res.equity.iloc[-1] > 850                        # far better than riding to 40


def test_take_profit_fires():
    df = make_bars([100, 100, 110, 120, 130])
    signals = pd.Series(1.0, index=df.index)
    res = run_backtest(df, signals, BacktestConfig(
        initial_cash=1000, slippage_bps=0, take_profit_pct=5))
    assert (res.trades["exit_reason"] == "target").any()


def test_short_side_profits_when_price_falls():
    df = make_bars([100, 100, 90, 80, 70])
    signals = pd.Series(-1.0, index=df.index)
    res = run_backtest(df, signals, BacktestConfig(
        initial_cash=1000, slippage_bps=0, allow_short=True))
    assert res.equity.iloc[-1] > 1000
    assert (res.trades["direction"] == "short").all()


def test_shorts_are_blocked_by_default():
    df = make_bars([100, 100, 90, 80])
    res = run_backtest(df, pd.Series(-1.0, index=df.index), BacktestConfig(initial_cash=1000))
    assert (res.position == 0).all()
    assert abs(res.equity.iloc[-1] - 1000) < 1e-9


def test_trade_pnl_reconciles_with_equity():
    """Realised P&L plus the open position must equal the change in equity."""
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    df = make_bars(prices, spread=0.05)
    sig = get_strategy("SMA crossover")(fast=10, slow=30).generate_signals(df)
    res = run_backtest(df, sig, BacktestConfig(initial_cash=10_000, slippage_bps=2))

    realised = res.trades["pnl"].sum()
    change = res.equity.iloc[-1] - 10_000
    assert abs(realised - change) < 1e-6 * max(1.0, abs(change)) + 0.01


def test_every_builtin_runs():
    rng = np.random.default_rng(11)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 600)))
    df = make_bars(prices, spread=0.2)

    # give the news strategies the columns they declare, all zeroed -- this also
    # checks they stay flat rather than misbehaving on a symbol with no coverage
    from core.newsfeatures import EVENT_NAMES
    news_cols = ["news_count", "news_sentiment", "news_sentiment_weighted",
                 "news_sentiment_ewm3", "news_sentiment_ewm10"]
    news_cols += [f"event_{e}" for e in EVENT_NAMES]
    df = df.join(pd.DataFrame(0.0, index=df.index, columns=news_cols))

    from core.strategy import available
    for name in available():
        strat = get_strategy(name)()
        sig = strat.generate_signals(df)
        assert sig.index.equals(df.index), f"{name} returned a misaligned signal index"
        assert sig.notna().all(), f"{name} returned NaN signals"
        res = run_backtest(df, sig, BacktestConfig(initial_cash=10_000))
        stats = summarise(res)
        assert np.isfinite(res.equity).all(), f"{name} produced a non-finite equity curve"
        assert stats["Trades"] == len(res.trades)


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
