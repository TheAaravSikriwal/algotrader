"""Trading-loop checks against the in-memory broker. No network, no API keys.

Run with:  python test_trader.py
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from brokers.fake import FakeBroker
from core.strategy import Param, Strategy
from core.trader import Halted, LiveConfig, Trader

import strategies  # noqa: F401


class FixedSignal(Strategy):
    """Holds whatever target you hand it -- lets the tests drive the loop directly."""
    name = "Fixed signal (test)"
    params = [Param("target", 1.0, "Target", "float", -1.0, 1.0, 0.1)]

    def generate_signals(self, df):
        return pd.Series(float(self.target), index=df.index)


def make_bars(n=60, price=100.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = np.full(n, float(price))
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c,
                         "volume": np.full(n, 1e6)}, index=idx)


def make_trader(target=1.0, cash=10_000.0, price=100.0, **overrides):
    tmp = tempfile.mkdtemp()
    broker = FakeBroker({"SPY": make_bars(price=price)}, cash=cash)
    cfg = LiveConfig(symbols=["SPY"], dry_run=False, log_dir=tmp,
                     halt_file=str(Path(tmp) / "HALT"), **overrides)
    return Trader(broker, FixedSignal(target=target), cfg), broker, tmp


def test_dry_run_sends_nothing():
    trader, broker, tmp = make_trader()
    trader.cfg.dry_run = True
    trader.run_once()
    assert broker.submitted == [], "dry run must not place orders"
    shutil.rmtree(tmp, ignore_errors=True)


def test_buys_up_to_the_target():
    trader, broker, tmp = make_trader(target=1.0, cash=10_000, price=100)
    trader.run_once()
    # 95% of $10k at $100 = 95 shares
    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == "buy"
    assert broker.submitted[0].qty == 95
    shutil.rmtree(tmp, ignore_errors=True)


def test_second_cycle_is_a_no_op_when_nothing_changed():
    """The loop trades the *difference*, so a steady signal must not churn."""
    trader, broker, tmp = make_trader(target=1.0)
    trader.run_once()
    first = len(broker.submitted)
    trader.run_once()
    assert len(broker.submitted) == first, "re-sent an order despite no change in target"
    shutil.rmtree(tmp, ignore_errors=True)


def test_signal_to_zero_closes_the_position():
    trader, broker, tmp = make_trader(target=1.0)
    trader.run_once()
    trader.strategy.target = 0.0
    trader.run_once()
    assert broker.submitted[-1].side == "sell"
    assert not broker.get_positions(), "position should be flat"
    shutil.rmtree(tmp, ignore_errors=True)


def test_shorts_blocked_unless_enabled():
    trader, broker, tmp = make_trader(target=-1.0)
    trader.run_once()
    assert broker.submitted == [], "went short with allow_short=False"
    shutil.rmtree(tmp, ignore_errors=True)


def test_shorts_allowed_when_enabled():
    trader, broker, tmp = make_trader(target=-1.0, allow_short=True)
    trader.run_once()
    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == "sell"
    shutil.rmtree(tmp, ignore_errors=True)


def test_closed_market_skips_the_cycle():
    trader, broker, tmp = make_trader()
    broker.set_market_open(False)
    summary = trader.run_once()
    assert summary["skipped"] and broker.submitted == []
    shutil.rmtree(tmp, ignore_errors=True)


def test_halt_file_stops_trading():
    trader, broker, tmp = make_trader()
    Path(trader.cfg.halt_file).write_text("stop")
    try:
        trader.run_once()
        raise AssertionError("HALT file did not stop the loop")
    except Halted:
        pass
    assert broker.submitted == []
    shutil.rmtree(tmp, ignore_errors=True)


def test_daily_loss_limit_halts_and_flattens():
    trader, broker, tmp = make_trader(target=1.0, cash=10_000, price=100)
    trader.run_once()                       # buy 95 shares at $100
    assert broker.get_positions()

    broker._bars["SPY"].loc[:, ["open", "high", "low", "close"]] = 90.0  # -9.5% on equity
    try:
        trader.run_once()
        raise AssertionError("daily-loss rail did not trip")
    except Halted as exc:
        assert "daily loss" in str(exc)
    assert not broker.get_positions(), "halt should have flattened the book"
    shutil.rmtree(tmp, ignore_errors=True)


def test_notional_cap_blocks_oversized_orders():
    trader, broker, tmp = make_trader(target=1.0, cash=1_000_000, price=100,
                                      max_order_notional=5_000)
    trader.run_once()
    assert broker.submitted == [], "order above the notional cap was still sent"
    shutil.rmtree(tmp, ignore_errors=True)


def test_partial_bar_is_dropped():
    """The last bar is still forming intraday, so the loop must ignore it."""
    bars = make_bars(n=40, price=100.0)
    bars.iloc[-1, bars.columns.get_loc("close")] = 999.0   # today's partial print
    tmp = tempfile.mkdtemp()
    broker = FakeBroker({"SPY": bars}, cash=10_000)
    cfg = LiveConfig(symbols=["SPY"], dry_run=False, log_dir=tmp,
                     halt_file=str(Path(tmp) / "HALT"))
    trader = Trader(broker, FixedSignal(target=1.0), cfg)
    intents = trader.plan(broker.get_account())
    assert intents[0].price == 100.0, "sized off the unfinished bar"
    shutil.rmtree(tmp, ignore_errors=True)


def test_working_order_blocks_a_duplicate():
    """The bug this prevents: a market order that has not filled yet would make
    the next cycle think the target was never reached, and double the order."""
    from core.broker import Order
    trader, broker, tmp = make_trader(target=1.0)
    broker.pending.append(Order(id="working-1", symbol="SPY", qty=95,
                                side="buy", status="new"))
    trader.run_once()
    assert broker.submitted == [], "ordered again while an order was still working"
    shutil.rmtree(tmp, ignore_errors=True)


def test_order_is_capped_by_buying_power():
    trader, broker, tmp = make_trader(target=1.0, cash=10_000, price=100)
    broker._cash = 10_000
    # equity says buy 95 shares ($9,500) but only $500 of buying power is free
    original = broker.get_account

    def squeezed():
        acct = original()
        acct.buying_power = 500.0
        return acct

    broker.get_account = squeezed
    trader.run_once()
    assert broker.submitted == [], "spent buying power it did not have"
    shutil.rmtree(tmp, ignore_errors=True)


def test_closing_a_position_ignores_buying_power():
    """Selling frees capital, so a low buying power must not block an exit."""
    trader, broker, tmp = make_trader(target=1.0, cash=10_000, price=100)
    trader.run_once()
    assert broker.get_positions()

    original = broker.get_account

    def squeezed():
        acct = original()
        acct.buying_power = 0.0
        return acct

    broker.get_account = squeezed
    trader.strategy.target = 0.0
    trader.run_once()
    assert not broker.get_positions(), "could not exit with zero buying power"
    shutil.rmtree(tmp, ignore_errors=True)


def test_activity_log_is_written():
    trader, broker, tmp = make_trader()
    trader.run_once()
    rows = [json.loads(line) for line in trader.activity_log.read_text().splitlines()]
    assert any(r["event"] == "order_sent" for r in rows)
    assert any(r["event"] == "cycle" for r in rows)
    shutil.rmtree(tmp, ignore_errors=True)


def test_real_strategy_drives_the_loop():
    """End to end with a registered strategy rather than the test stub."""
    from core.strategy import get_strategy
    rng = np.random.default_rng(3)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 200)))
    idx = pd.bdate_range("2024-01-01", periods=200)
    bars = pd.DataFrame({"open": prices, "high": prices * 1.005, "low": prices * 0.995,
                         "close": prices, "volume": np.full(200, 1e6)}, index=idx)

    tmp = tempfile.mkdtemp()
    broker = FakeBroker({"AAPL": bars}, cash=50_000)
    cfg = LiveConfig(symbols=["AAPL"], dry_run=False, log_dir=tmp,
                     halt_file=str(Path(tmp) / "HALT"))
    trader = Trader(broker, get_strategy("SMA crossover")(fast=10, slow=30), cfg)
    summary = trader.run_once()
    assert not summary["skipped"]
    assert summary["intents"][0]["symbol"] == "AAPL"
    shutil.rmtree(tmp, ignore_errors=True)


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
