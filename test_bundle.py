"""Round-trip checks for run bundles. No network needed."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from core import engine as engine_mod
from core.bundle import (BundleError, build_bundle, bundle_equity, describe,
                         load_bundle, replay, save_bundle)
from core.engine import BacktestConfig, run_backtest
from core.metrics import summarise
from core.strategy import get_strategy

import strategies  # noqa: F401


def make_run(fast=10, slow=30):
    rng = np.random.default_rng(5)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, 500)))
    idx = pd.bdate_range("2023-01-02", periods=500)
    df = pd.DataFrame({"open": prices, "high": prices * 1.004, "low": prices * 0.996,
                       "close": prices, "volume": np.full(500, 1e6)}, index=idx)
    strategy = get_strategy("SMA crossover")(fast=fast, slow=slow)
    result = run_backtest(df, strategy.generate_signals(df),
                          BacktestConfig(initial_cash=25_000, slippage_bps=3))
    return df, strategy, result


def make_bundle():
    df, strategy, result = make_run()
    return df, build_bundle(symbol="TEST", start="2023-01-02", end="2024-12-01",
                            timeframe="1Day", source="yfinance", strategy=strategy,
                            result=result, stats=summarise(result)), result


def test_bundle_carries_everything_needed():
    _, bundle, _ = make_bundle()
    assert bundle["strategy"]["name"] == "SMA crossover"
    assert bundle["strategy"]["params"]["fast"] == 10
    assert bundle["execution"]["initial_cash"] == 25_000
    assert bundle["run"]["symbol"] == "TEST"
    assert bundle["equity"]["values"], "no equity curve saved"
    assert "Sharpe" in bundle["metrics"]


def test_survives_a_file_round_trip():
    _, bundle, result = make_bundle()
    tmp = Path(tempfile.mkdtemp()) / "run"
    path = save_bundle(bundle, tmp)
    assert path.suffix == ".json", "extension was not normalised"

    reloaded = load_bundle(path)
    assert reloaded["strategy"] == bundle["strategy"]
    equity = bundle_equity(reloaded)
    assert len(equity) == len(result.equity)
    assert abs(equity.iloc[-1] - result.equity.iloc[-1]) < 1e-6


def test_replay_reproduces_the_original():
    df, bundle, result = make_bundle()
    replayed, strategy, drift = replay(bundle, lambda *a, **k: df,
                                       engine_mod, get_strategy)
    assert abs(drift) < 1e-6, f"replay drifted by {drift}"
    assert strategy.settings == bundle["strategy"]["params"]
    assert len(replayed.trades) == len(result.trades)


def test_replay_detects_drift():
    """A bundle whose saved result no longer matches must say so, not stay quiet."""
    df, bundle, _ = make_bundle()
    bundle["equity"]["values"][-1] += 1_000.0
    _, _, drift = replay(bundle, lambda *a, **k: df, engine_mod, get_strategy)
    assert abs(drift + 1_000.0) < 1e-6


def test_rejects_a_foreign_file():
    for junk in ('{"hello": "world"}', "not json at all"):
        try:
            load_bundle(junk)
            raise AssertionError(f"accepted junk: {junk!r}")
        except BundleError:
            pass


def test_rejects_a_future_version():
    _, bundle, _ = make_bundle()
    bundle["version"] = 99
    try:
        load_bundle(json.dumps(bundle))
        raise AssertionError("accepted a newer bundle version")
    except BundleError as exc:
        assert "newer" in str(exc)


def test_reports_a_renamed_parameter():
    """If a strategy's params change, replay must explain rather than crash."""
    _, bundle, _ = make_bundle()
    bundle["strategy"]["params"]["a_param_that_no_longer_exists"] = 7
    try:
        replay(bundle, lambda *a, **k: make_run()[0], engine_mod, get_strategy)
        raise AssertionError("replayed a bundle with an unknown parameter")
    except BundleError as exc:
        assert "no longer accepts" in str(exc)


def test_describe_is_human_readable():
    _, bundle, _ = make_bundle()
    text = describe(bundle)
    assert "TEST" in text and "SMA crossover" in text and "fast=10" in text


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


def test_plain_renders_bold_rather_than_showing_the_stars(monkeypatch):
    """`plain` emits raw HTML, so Streamlit's markdown never runs on it.

    Without the conversion, "**matters**" reaches the page as literal
    asterisks -- which is exactly how it shipped the first time.
    """
    import streamlit as st

    from core.ui import plain

    captured = []
    monkeypatch.setattr(st, "markdown", lambda body, **kw: captured.append(body))
    plain("**This matters.** This does not.")

    assert "<b>This matters.</b>" in captured[0]
    assert "**" not in captured[0]
