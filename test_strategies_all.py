"""Properties every registered strategy must satisfy.

These run against the whole registry rather than one class at a time, so a
strategy added later is covered the moment it is registered. The important one
is the lookahead check: it rewrites the future and asserts the past does not
move. A rule that fails it is not a strategy, it is a time machine, and its
backtest is worthless no matter how good it looks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import strategies  # noqa: F401  -- registers everything
from core.strategy import REGISTRY, Strategy

# The news strategies need feature columns a plain OHLCV frame does not carry.
# They are covered by test_news.py; skipping them here keeps this file about
# the property rather than about fixture plumbing.
# "ZZ " names are deliberate test fixtures from other modules -- oracles that
# read ahead, rules that throw. They register globally, and the lookahead check
# below correctly fails them, which is the check working rather than a problem
# with the strategy library.
PRICE_ONLY = {name: cls for name, cls in REGISTRY.items()
              if not cls.requires_features and not name.startswith("ZZ ")}


def _series(n: int, seed: int) -> pd.DataFrame:
    """A deterministic random walk with enough structure to trigger rules."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    spread = np.abs(rng.normal(0, 0.006, n)) * close
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": rng.integers(1_000_000, 9_000_000, n).astype(float),
    }, index=pd.bdate_range("2015-01-01", periods=n))


@pytest.fixture(scope="module")
def bars():
    return _series(900, seed=7)


@pytest.mark.parametrize("name", sorted(PRICE_ONLY))
def test_signals_are_well_formed(name, bars):
    sig = PRICE_ONLY[name]().generate_signals(bars)
    assert isinstance(sig, pd.Series)
    assert len(sig) == len(bars)
    assert sig.index.equals(bars.index)
    assert not sig.isna().any(), "a NaN position is neither in nor out"
    assert np.isfinite(sig.to_numpy(float)).all()


@pytest.mark.parametrize("name", sorted(PRICE_ONLY))
def test_positions_are_within_a_sane_range(name, bars):
    """No strategy here is allowed to imply leverage it never declared."""
    sig = PRICE_ONLY[name]().generate_signals(bars)
    assert sig.abs().max() <= 3.0, f"{name} asks for {sig.abs().max():.1f}x"


def _splice(bars: pd.DataFrame, cut: int, seed: int, drift: float) -> pd.DataFrame:
    """Replace everything from `cut` onward with a different future."""
    n = len(bars) - cut
    tail = _series(n, seed=seed)
    # The level shift is applied from the very FIRST spliced bar, not ramped in.
    # Two futures that begin at the same price leave a one-bar leak reading the
    # same number in both versions -- which is how an outright
    # ``close.shift(-1)`` oracle passed two earlier drafts of this test.
    scale = bars["close"].iloc[cut - 1] / tail["close"].iloc[0]
    factor = (1.0 + drift) * np.exp(np.linspace(0, drift, n))
    out = bars.copy()
    for col in ["open", "high", "low", "close"]:
        out.iloc[cut:, out.columns.get_loc(col)] = tail[col].to_numpy() * scale * factor
    out.iloc[cut:, out.columns.get_loc("volume")] = tail["volume"].to_numpy()
    return out


@pytest.mark.parametrize("name", sorted(PRICE_ONLY))
def test_no_lookahead(name, bars):
    """Give the strategy two different futures; the past must come out identical.

    Comparing two spliced futures against *each other* rather than against the
    original is what makes this sensitive. A one-bar leak is only observable on
    the single bar before the splice, and if that bar is compared against a
    price-continuous future it can agree by coincidence -- an outright
    ``close.shift(-1)`` oracle passed the first version of this test for
    exactly that reason. Two futures that diverge immediately and in opposite
    directions leave it nowhere to hide.

    This is the check that separates a strategy from a backtest artefact.
    """
    cls = PRICE_ONLY[name]
    cut = len(bars) - 200

    up = cls().generate_signals(_splice(bars, cut, seed=99, drift=+0.6)).iloc[:cut]
    down = cls().generate_signals(_splice(bars, cut, seed=1234, drift=-0.6)).iloc[:cut]

    where = np.flatnonzero(up.to_numpy(float) != down.to_numpy(float))
    assert not len(where), (
        f"{name} changed {len(where)} past signal(s) when the future was "
        f"rewritten; first at index {where[0]} of {cut}")


@pytest.mark.parametrize("name", sorted(PRICE_ONLY))
def test_a_strategy_is_deterministic(name, bars):
    """Same input, same output. A rule that drifts cannot be evaluated."""
    a = PRICE_ONLY[name]().generate_signals(bars)
    b = PRICE_ONLY[name]().generate_signals(bars)
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("name", sorted(PRICE_ONLY))
def test_short_history_does_not_crash(name):
    """Real universes contain recent listings with almost no history."""
    sig = PRICE_ONLY[name]().generate_signals(_series(12, seed=3))
    assert len(sig) == 12
    assert not sig.isna().any()


@pytest.mark.parametrize("name", sorted(PRICE_ONLY))
def test_unknown_parameters_are_rejected(name):
    with pytest.raises(TypeError):
        PRICE_ONLY[name](definitely_not_a_real_parameter=1)


def test_the_fixtures_are_excluded_but_would_have_been_caught():
    """The exclusion is by naming convention, so check it is not hiding a leak.

    The intraday fixtures genuinely read ahead. If the lookahead test did not
    fail them, the convention would be hiding a broken check rather than an
    irrelevant one.
    """
    import pytest as _pytest
    if "ZZ close oracle" not in REGISTRY:
        _pytest.skip("intraday fixtures not loaded in this run")
    assert "ZZ close oracle" not in PRICE_ONLY
    with _pytest.raises(AssertionError):
        test_no_lookahead.__wrapped__("ZZ close oracle", _series(900, seed=7))             if hasattr(test_no_lookahead, "__wrapped__") else _leak_check(
                REGISTRY["ZZ close oracle"], _series(900, seed=7))


def _leak_check(cls, bars):
    cut = len(bars) - 200
    up = cls().generate_signals(_splice(bars, cut, seed=99, drift=+0.6)).iloc[:cut]
    down = cls().generate_signals(_splice(bars, cut, seed=1234, drift=-0.6)).iloc[:cut]
    assert not np.flatnonzero(up.to_numpy(float) != down.to_numpy(float)).size


def test_the_registry_actually_grew():
    """Guards against a module silently failing to import.

    A strategy file that raises on import just vanishes from the registry --
    the tournament then reports on whatever is left and looks perfectly
    healthy.
    """
    assert len(PRICE_ONLY) >= 28, f"only {len(PRICE_ONLY)} registered"
    for expected in ["Buy and hold", "Turtle breakout", "Ensemble vote",
                     "Walk-forward logistic", "Volatility-targeted trend"]:
        assert expected in REGISTRY


def test_buy_and_hold_is_always_invested(bars):
    """The benchmark every other result is judged against. If this is wrong,
    every relative number in the tournament is wrong with it."""
    sig = REGISTRY["Buy and hold"]().generate_signals(bars)
    assert (sig == 1.0).all()
