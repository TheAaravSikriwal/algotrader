"""Cross-sectional engine checks. Synthetic data, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.panel import (Panel, PanelConfig, equal_weight_benchmark,
                        run_panel_backtest)
from strategies.cross_sectional import available_xs, get_xs_strategy


def make_panel(n=600, symbols=("AAA", "BBB", "CCC", "DDD"), seed=5, drifts=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    drifts = drifts or {s: 0.0002 for s in symbols}

    bars = {}
    for s in symbols:
        r = rng.normal(drifts[s], 0.012, n)
        px = 100 * np.exp(np.cumsum(r))
        bars[s] = pd.DataFrame({"open": px, "high": px * 1.005, "low": px * 0.995,
                                "close": px, "volume": np.full(n, 1e6)}, index=idx)
    return Panel.from_bars(bars)


def test_panel_alignment():
    panel = make_panel()
    assert panel.symbols == ["AAA", "BBB", "CCC", "DDD"]
    assert len(panel) == 600
    assert panel.closes.shape == (600, 4)
    assert panel.tradeable().all().all()


def test_panel_handles_ragged_history():
    """A symbol that lists partway through must not break alignment."""
    panel_full = make_panel(n=300, symbols=("AAA", "BBB"))
    bars = {"AAA": pd.DataFrame({
        "open": panel_full.opens["AAA"], "high": panel_full.highs["AAA"],
        "low": panel_full.lows["AAA"], "close": panel_full.closes["AAA"],
        "volume": panel_full.volumes["AAA"]})}
    late = panel_full.closes["BBB"].iloc[150:]
    bars["BBB"] = pd.DataFrame({"open": late, "high": late, "low": late,
                                "close": late, "volume": 1e6})

    panel = Panel.from_bars(bars)
    assert len(panel) == 300
    assert not panel.tradeable()["BBB"].iloc[10]
    assert panel.tradeable()["BBB"].iloc[200]
    assert panel.coverage()["BBB"] < 0.6


def test_weights_are_filled_next_bar():
    """The no-lookahead rule, at panel scale."""
    panel = make_panel(n=50, symbols=("AAA", "BBB"))
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    weights.iloc[10:, 0] = 1.0

    result = run_panel_backtest(panel, weights,
                                PanelConfig(initial_cash=10_000, slippage_bps=0))
    assert result.weights.iloc[10].sum() < 1e-9, "traded on the signal bar itself"
    assert result.weights.iloc[11]["AAA"] > 0.9, "did not fill at the next open"


def test_equal_weight_matches_its_benchmark():
    panel = make_panel(n=400)
    strategy = get_xs_strategy("Equal weight all")(rebalance_every=1)
    weights = strategy.generate_weights(panel)
    result = run_panel_backtest(panel, weights,
                                PanelConfig(initial_cash=100_000, slippage_bps=0))

    bench = equal_weight_benchmark(panel, 100_000)
    drift = abs(result.equity.iloc[-1] - bench.iloc[-1]) / 100_000
    assert drift < 0.05, f"equal weight drifted {drift:.1%} from its benchmark"


def test_top_n_holds_exactly_n_names():
    panel = make_panel(n=400, symbols=("AAA", "BBB", "CCC", "DDD", "EEE"))
    strategy = get_xs_strategy("Cross-sectional momentum")(
        lookback=60, skip=5, top_n=2, rebalance_every=21)
    weights = strategy.generate_weights(panel)

    active = weights[weights.sum(axis=1) > 0]
    assert not active.empty, "never took a position"
    counts = (active > 0).sum(axis=1)
    assert counts.max() <= 2, f"held {counts.max()} names with top_n=2"
    assert np.allclose(active.sum(axis=1), 1.0), "weights did not sum to 1"


def test_momentum_picks_the_winner():
    """With one symbol given a large planted drift, momentum must find it."""
    panel = make_panel(n=500, symbols=("FLAT1", "FLAT2", "FLAT3", "WINNER"),
                       drifts={"FLAT1": 0.0, "FLAT2": 0.0, "FLAT3": 0.0,
                               "WINNER": 0.0025})
    strategy = get_xs_strategy("Cross-sectional momentum")(
        lookback=120, skip=5, top_n=1, rebalance_every=21)
    weights = strategy.generate_weights(panel)

    held = weights[weights.sum(axis=1) > 0]
    share = (held["WINNER"] > 0).mean()
    assert share > 0.8, f"only held the winner {share:.0%} of the time"


def test_low_volatility_picks_the_calm_name():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2021-01-04", periods=400)
    bars = {}
    for s, vol in [("CALM", 0.004), ("MID", 0.012), ("WILD", 0.030)]:
        px = 100 * np.exp(np.cumsum(rng.normal(0.0002, vol, 400)))
        bars[s] = pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                                "volume": 1e6}, index=idx)
    panel = Panel.from_bars(bars)

    weights = get_xs_strategy("Low volatility")(
        lookback=60, top_n=1, rebalance_every=21).generate_weights(panel)
    held = weights[weights.sum(axis=1) > 0]
    assert (held["CALM"] > 0).mean() > 0.9


def test_costs_reduce_returns():
    panel = make_panel(n=400)
    weights = get_xs_strategy("Cross-sectional reversal")(
        lookback=10, top_n=2, rebalance_every=5).generate_weights(panel)

    free = run_panel_backtest(panel, weights, PanelConfig(slippage_bps=0))
    costly = run_panel_backtest(panel, weights, PanelConfig(slippage_bps=50))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.trades["fees"].sum() >= 0


def test_turnover_is_recorded():
    panel = make_panel(n=400)
    weights = get_xs_strategy("Cross-sectional momentum")(
        lookback=60, top_n=2, rebalance_every=21).generate_weights(panel)
    result = run_panel_backtest(panel, weights)
    assert result.turnover.sum() > 0
    assert result.annual_turnover > 0


def test_gross_exposure_is_capped():
    panel = make_panel(n=100, symbols=("AAA", "BBB"))
    weights = pd.DataFrame(5.0, index=panel.index, columns=panel.symbols)
    result = run_panel_backtest(panel, weights,
                                PanelConfig(initial_cash=10_000, max_gross=1.0,
                                            slippage_bps=0))
    assert result.weights.abs().sum(axis=1).max() < 1.15, "leveraged past the cap"


def test_shorts_blocked_unless_enabled():
    panel = make_panel(n=100, symbols=("AAA", "BBB"))
    weights = pd.DataFrame(-1.0, index=panel.index, columns=panel.symbols)
    result = run_panel_backtest(panel, weights, PanelConfig(initial_cash=10_000))
    assert (result.weights >= -1e-9).all().all()


def test_untradeable_symbols_are_skipped():
    """A name with no bar that day cannot be bought, however good it ranks."""
    panel = make_panel(n=200, symbols=("AAA", "BBB"))
    panel.opens.iloc[50:60, 1] = np.nan
    panel.closes.iloc[50:60, 1] = np.nan

    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)
    weights["BBB"] = 1.0
    result = run_panel_backtest(panel, weights,
                                PanelConfig(initial_cash=10_000, slippage_bps=0))
    assert np.isfinite(result.equity).all()
    for _, row in result.trades.iterrows():
        stamp = row["timestamp"]
        if row["symbol"] == "BBB":
            assert not np.isnan(panel.opens.loc[stamp, "BBB"])


def test_survivorship_warning_fires_on_complete_history():
    panel = make_panel(n=300)
    notes = panel.survivorship_warning()
    assert any("survivorship" in n.lower() or "delisted" in n.lower() for n in notes)


def test_every_xs_strategy_runs():
    panel = make_panel(n=500, symbols=("AAA", "BBB", "CCC", "DDD", "EEE"))
    for name in available_xs():
        strategy = get_xs_strategy(name)()
        weights = strategy.generate_weights(panel)
        assert weights.shape == (len(panel), len(panel.symbols)), name
        assert weights.notna().all().all(), f"{name} produced NaN weights"
        result = run_panel_backtest(panel, weights)
        assert np.isfinite(result.equity).all(), f"{name} broke the equity curve"


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
