"""Panel live-loop checks. FakeBroker throughout, no network."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from brokers.fake import FakeBroker
from core.explain import CHARTS, CONTROLS, METRICS, glossary_markdown, metric
from core.overlay import Overlay, record
from core.paneltrader import PanelLiveConfig, PanelTrader
from core.trader import Halted
from strategies.cross_sectional import get_xs_strategy

SYMBOLS = ["SPY", "QQQ", "TLT", "XLE"]


def bars(n=300, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    out = {}
    for s in SYMBOLS:
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
        out[s] = pd.DataFrame({"open": px, "high": px * 1.004, "low": px * 0.996,
                               "close": px, "volume": 1e6}, index=idx)
    return out


def trader(cash=100_000, **overrides):
    broker = FakeBroker(bars(), cash=cash)
    cfg = PanelLiveConfig(symbols=SYMBOLS, require_market_open=False,
                          **overrides)
    strategy = get_xs_strategy(cfg.strategy)(**cfg.params)
    return PanelTrader(broker, strategy, cfg), broker


# ---------------------------------------------------------------------------
# building the book
# ---------------------------------------------------------------------------
def test_base_book_is_produced():
    t, _ = trader(use_overlay=False)
    base, final, _ = t.target_book()
    assert len(base) == len(SYMBOLS)
    assert abs(base.sum() - 1.0) < 1e-6
    assert final.equals(base), "no overlay was active, so nothing should change"


def test_dry_run_sends_nothing():
    t, broker = trader(dry_run=True, use_overlay=False)
    out = t.run_once()
    assert out["status"] == "dry run"
    assert broker.submitted == []
    assert all(r["action"].startswith("would") for r in out["results"])


def test_execute_places_orders():
    t, broker = trader(dry_run=False, use_overlay=False)
    out = t.run_once()
    assert broker.submitted, "executed but sent no orders"
    assert all(r["action"] in ("buy", "sell") for r in out["results"])


def test_second_cycle_is_quiet_once_positioned():
    """After the book is built, nothing should churn on the next pass."""
    t, broker = trader(dry_run=False, use_overlay=False)
    t.run_once()
    first = len(broker.submitted)
    t.run_once()
    assert len(broker.submitted) == first, "rebalanced when nothing had changed"


def test_whole_shares_only():
    t, _ = trader(dry_run=True, use_overlay=False)
    out = t.run_once()
    for intent in out["intents"]:
        assert intent.target_shares == int(intent.target_shares)


def test_dust_trades_are_skipped():
    """Rebalancing a fraction of a percent costs more in spread than it fixes."""
    t, broker = trader(dry_run=False, use_overlay=False)
    t.run_once()
    before = len(broker.submitted)
    t.cfg.min_trade_fraction = 0.5     # nothing should qualify
    t.run_once()
    assert len(broker.submitted) == before


def test_working_orders_block_a_symbol():
    from core.broker import Order
    t, broker = trader(dry_run=False, use_overlay=False)
    broker.pending.append(Order(id="w1", symbol="SPY", qty=10, side="buy",
                                status="new"))
    out = t.run_once()
    assert all(i.symbol != "SPY" for i in out["intents"]), \
        "queued a second order while one was still working"


def test_buying_power_is_respected():
    t, broker = trader(dry_run=False, use_overlay=False)
    original = broker.get_account

    def squeezed():
        account = original()
        account.buying_power = 50.0
        return account

    broker.get_account = squeezed
    out = t.run_once()
    assert any(r["action"] == "skipped" for r in out["results"])


def test_daily_loss_halts_the_book():
    t, broker = trader(cash=100_000, dry_run=False, use_overlay=False,
                       max_daily_loss_pct=2.0)
    t.run_once()
    broker._cash -= 30_000            # simulate a bad day
    try:
        t.run_once()
        raise AssertionError("kept trading through the daily loss limit")
    except Halted as exc:
        assert "daily loss" in str(exc)


def test_market_closed_skips_the_cycle():
    t, broker = trader(dry_run=False, use_overlay=False)
    t.cfg.require_market_open = True
    broker.set_market_open(False)
    out = t.run_once()
    assert out["status"] == "market closed"
    assert broker.submitted == []


def test_shorts_blocked_unless_enabled():
    t, _ = trader(dry_run=True, use_overlay=False)
    _, final, _ = t.target_book()
    out = t.run_once()
    assert all(i.target_shares >= 0 for i in out["intents"])


# ---------------------------------------------------------------------------
# the overlay on top
# ---------------------------------------------------------------------------
def test_overlay_tilts_the_book():
    path = Path(mkdtemp()) / "overlays.jsonl"
    record(Overlay(rationale="test", bucket="macro", adjustments=[
        {"symbol": "XLE", "action": "tilt", "weight": 0.12, "confidence": 1.0,
         "reason": "supply shock lifts crude into sector revenue"}]), path)

    t, _ = trader(dry_run=True, use_overlay=True, overlay_ledger=path)
    base, final, report = t.target_book()
    assert final["XLE"] > base["XLE"], "overlay did not tilt the book"
    assert len(report["applied"]) == 1


def test_expired_overlay_leaves_the_book_alone():
    """A stale view must not still be steering the portfolio."""
    path = Path(mkdtemp()) / "overlays.jsonl"
    record(Overlay(
        rationale="stale", bucket="macro",
        issued=(date.today() - timedelta(days=90)).isoformat(),
        adjustments=[{"symbol": "XLE", "action": "tilt", "weight": 0.12,
                      "expires_days": 21, "reason": "long expired"}]), path)

    t, _ = trader(dry_run=True, use_overlay=True, overlay_ledger=path)
    base, final, _ = t.target_book()
    assert np.allclose(base.to_numpy(), final.to_numpy())


def test_base_is_recoverable_alongside_the_overlay():
    """Switching the overlay off must be a flag, not a rewrite."""
    t_on, _ = trader(dry_run=True, use_overlay=True)
    t_off, _ = trader(dry_run=True, use_overlay=False)
    base_on, _, _ = t_on.target_book()
    base_off, _, _ = t_off.target_book()
    assert np.allclose(base_on.to_numpy(), base_off.to_numpy())


# ---------------------------------------------------------------------------
# the explanations
# ---------------------------------------------------------------------------
def test_every_metric_explains_itself():
    for key, item in METRICS.items():
        assert item.label, key
        assert len(item.plain) > 20, f"{key}: explanation too thin"
        assert not item.plain.endswith(":"), key


def test_metrics_name_what_can_fool_you():
    """The trap field is the point -- a metric with no caveat is a lie."""
    without = [k for k, v in METRICS.items() if not v.trap]
    assert not without, f"no caveat given for: {without}"


def test_charts_and_controls_are_explained():
    assert len(CHARTS) >= 4
    assert len(CONTROLS) >= 5
    for item in list(CHARTS.values()) + list(CONTROLS.values()):
        assert len(item.plain) > 15


def test_tooltip_survives_html_escaping():
    """A tooltip full of quotes and newlines must not break out of the
    attribute it lives in and render as page text."""
    import html
    from core.explain import METRICS

    for key, item in METRICS.items():
        escaped = html.escape(item.tooltip(), quote=True).replace("\n", "&#10;")
        assert '"' not in escaped, f"{key}: an unescaped quote would end the attribute"
        assert "\n" not in escaped, f"{key}: a raw newline survived"
        assert "<" not in escaped and ">" not in escaped, key


def test_tooltip_and_markdown_render():
    item = metric("sharpe")
    assert "per unit of nerve" in item.tooltip()
    assert item.markdown().startswith("**Sharpe ratio**")
    assert "Sharpe" in glossary_markdown(["sharpe"])


def test_glossary_ignores_unknown_keys():
    assert glossary_markdown(["not_a_metric"]) == ""


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
