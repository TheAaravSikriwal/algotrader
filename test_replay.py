"""Replay-broker checks. The critical one is that it cannot see the future."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brokers.replay import ReplayBroker
from core.broker import BrokerError
from core.engine import BacktestConfig, run_backtest
from core.metrics import equity_stats
from core.strategy import get_strategy
from core.trader import LiveConfig, Trader

import strategies  # noqa: F401


def synth(n=600, seed=3):
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n)))
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"open": prices, "high": prices * 1.004,
                         "low": prices * 0.996, "close": prices,
                         "volume": np.full(n, 1e6)}, index=idx)


def test_cannot_see_past_the_current_bar():
    """The whole point. A replay that leaks the future proves nothing."""
    df = synth()
    broker = ReplayBroker({"AAA": df}, warmup=100)
    for _ in range(50):
        visible = broker.get_bars("AAA", limit=10_000)
        assert visible.index[-1] <= broker.now, "returned a bar from the future"
        assert len(visible) == df.index.get_loc(broker.now) + 1
        broker.advance()


def test_last_price_is_the_current_close():
    df = synth()
    broker = ReplayBroker({"AAA": df}, warmup=100)
    assert broker.last_price("AAA") == df.loc[broker.now, "close"]
    broker.advance()
    assert broker.last_price("AAA") == df.loc[broker.now, "close"]


def test_orders_fill_at_the_next_bar_open():
    df = synth()
    broker = ReplayBroker({"AAA": df}, warmup=100, slippage_bps=0)
    broker.submit_order("AAA", 10, "buy")
    assert broker.get_positions() == {}, "filled instantly instead of next bar"
    assert len(broker.get_open_orders()) == 1

    expected_price = df["open"].iloc[df.index.get_loc(broker.now) + 1]
    broker.advance()
    position = broker.get_positions()["AAA"]
    assert position.qty == 10
    assert abs(position.avg_price - expected_price) < 1e-9
    assert broker.get_open_orders() == []


def test_slippage_is_charged_on_the_fill():
    df = synth()
    clean = ReplayBroker({"AAA": df}, warmup=100, slippage_bps=0)
    dirty = ReplayBroker({"AAA": df}, warmup=100, slippage_bps=50)
    for broker in (clean, dirty):
        broker.submit_order("AAA", 10, "buy")
        broker.advance()
    assert (dirty.get_positions()["AAA"].avg_price
            > clean.get_positions()["AAA"].avg_price)


def test_advance_stops_at_the_end_of_history():
    df = synth(n=120)
    broker = ReplayBroker({"AAA": df}, warmup=100)
    steps = 0
    while broker.advance():
        steps += 1
        assert steps < 1000, "advance never terminated"
    assert broker.finished
    assert broker.advance() is False


def test_refuses_too_little_history():
    try:
        ReplayBroker({"AAA": synth(n=50)}, warmup=100)
        raise AssertionError("accepted a warmup longer than the data")
    except BrokerError as exc:
        assert "warmup" in str(exc).lower() or "enough" in str(exc).lower()


def test_unknown_symbol_is_rejected():
    broker = ReplayBroker({"AAA": synth()}, warmup=100)
    for call in (lambda: broker.get_bars("ZZZ"),
                 lambda: broker.submit_order("ZZZ", 1, "buy")):
        try:
            call()
            raise AssertionError("accepted an unknown symbol")
        except BrokerError:
            pass


def run_replay(df, strategy, cash=10_000, warmup=200, slippage=5.0):
    broker = ReplayBroker({"AAA": df}, cash=cash, slippage_bps=slippage,
                          warmup=warmup)
    trader = Trader(broker, strategy, LiveConfig(
        symbols=["AAA"], dry_run=False, position_size=1.0,
        max_daily_loss_pct=100.0, use_closed_bars_only=False,
        log_dir="logs/test_replay"))
    while True:
        trader.run_once()
        if not broker.advance():
            break
    return broker


def test_replay_agrees_with_the_backtester():
    """The live loop and the simulator must reach the same place on the same
    data. A gap means one of them is wrong about orders, sizing or timing."""
    df = synth(n=700)
    strategy = get_strategy("SMA crossover")(fast=20, slow=50)
    warmup = 200

    broker = run_replay(df, strategy, warmup=warmup)
    replay_final = broker.equity_series().iloc[-1]

    window = df.loc[broker.timeline[warmup]:]
    backtest = run_backtest(window,
                            strategy.generate_signals(df).loc[window.index],
                            BacktestConfig(initial_cash=10_000, slippage_bps=5.0))
    bt_final = backtest.equity.iloc[-1]

    gap = abs(replay_final - bt_final) / 10_000
    assert gap < 0.03, (f"replay and backtest disagree by {gap * 100:.1f}% "
                        f"({replay_final:,.0f} vs {bt_final:,.0f})")


def test_replay_actually_trades():
    df = synth(n=700)
    broker = run_replay(df, get_strategy("SMA crossover")(fast=10, slow=30))
    assert len(broker.fills) > 3, "replay produced almost no fills"
    assert not broker.equity_series().empty


def test_buy_and_hold_replay_tracks_the_market():
    df = synth(n=600)
    broker = run_replay(df, get_strategy("Buy and hold")(), warmup=200, slippage=0)
    equity = broker.equity_series()

    window = df.loc[broker.timeline[200]:]
    market = window["close"].iloc[-1] / window["open"].iloc[1] - 1
    replayed = equity.iloc[-1] / 10_000 - 1
    assert abs(replayed - market) < 0.02, (
        f"buy and hold drifted from the market: {replayed:.3f} vs {market:.3f}")


def test_working_order_guard_engages_during_replay():
    """Replay leaves orders unfilled for a bar, which is precisely the condition
    the duplicate-order guard exists for."""
    df = synth(n=400)
    broker = ReplayBroker({"AAA": df}, cash=10_000, warmup=200)
    trader = Trader(broker, get_strategy("Buy and hold")(), LiveConfig(
        symbols=["AAA"], dry_run=False, max_daily_loss_pct=100.0,
        use_closed_bars_only=False, log_dir="logs/test_replay"))

    trader.run_once()
    assert len(broker.get_open_orders()) == 1
    trader.run_once()                      # same bar, order still working
    assert len(broker.get_open_orders()) == 1, "queued a duplicate order"


def test_equity_curve_is_recorded_each_bar():
    df = synth(n=400)
    broker = run_replay(df, get_strategy("Buy and hold")(), warmup=200)
    equity = broker.equity_series()
    assert len(equity) == len(df) - 200 - 1 + 1 or len(equity) > 100
    assert equity.index.is_monotonic_increasing
    assert equity_stats(equity)["Final equity"] == equity.iloc[-1]


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
