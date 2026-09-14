"""The automatic loop.

Two things matter here. The loop must not be able to do anything a person
could not do by hand on the same rails -- paper only, session window,
flatten deadline, loss limit. And an instruction from Claude must actually
reach it, exactly once, bounded.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from core.autotrader import AutoConfig, AutoTrader
from core.livestate import Bridge, Instruction
from core.marketclock import MarketCalendar
from core.recommend import Candidate
from test_daytrader import FakeBroker, _gap_bars      # reuse the fixtures

DAY = date(2026, 9, 14)


def _cal():
    return MarketCalendar.from_rows([
        {"date": DAY, "open": "09:30", "close": "16:00"},
        {"date": date(2026, 11, 27), "open": "09:30", "close": "13:00"},
    ])


def _auto(tmp_path, broker=None, **kw):
    broker = broker or FakeBroker(bars=_gap_bars())
    cfg = AutoConfig(symbols=("SPY",), **kw)
    return AutoTrader(broker, cfg, _cal(), Bridge(tmp_path)), broker


def _cand(strategy, bps=+14.0, t=1.1, trades=400.0, symbol="SPY"):
    return Candidate(strategy=strategy, symbol=symbol, expectancy_pct=bps,
                     t_stat=t, trades=trades, exposure=1.0,
                     avg_hold_bars=6.0, horizon="short")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Keep the loop away from the project's real logs and fill history."""
    import core.daytrader as mod
    monkeypatch.setattr(mod, "FILLS", tmp_path / "fills.jsonl")
    monkeypatch.setattr(mod, "STATE", tmp_path / "trader_state.json")
    monkeypatch.setattr(mod, "HALT", tmp_path / "HALT")


# -- the rails still apply ------------------------------------------------

def test_the_loop_cannot_trade_a_live_account(tmp_path):
    """Automation must not widen what is permitted."""
    from core.daytrader import NotPaper
    auto, _ = _auto(tmp_path, broker=FakeBroker(bars=_gap_bars(), is_paper=False))
    monkey = auto.candidates
    auto.candidates = lambda: [_cand("X")]
    auto._running = "X"
    c = auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    assert not c.sent
    assert any("paper" in b.lower() for b in c.blocks)
    auto.candidates = monkey


def test_nothing_opens_outside_the_session(tmp_path):
    auto, broker = _auto(tmp_path)
    auto.candidates = lambda: [_cand("X")]
    c = auto.cycle(now=datetime(2026, 9, 14, 8, 0))
    assert not c.sent and broker.submitted == []
    assert any("outside the session" in b for b in c.blocks)


def test_the_deadline_flattens_rather_than_trades(tmp_path):
    from core.broker import Position
    broker = FakeBroker(bars=_gap_bars(),
                        positions={"SPY": Position("SPY", 10, 11.0)})
    auto, _ = _auto(tmp_path, broker=broker)
    auto.candidates = lambda: [_cand("X")]
    c = auto.cycle(now=datetime(2026, 9, 14, 15, 57))
    assert c.flattened
    assert broker.closed == ["SPY"]
    assert broker.submitted == []


def test_stale_data_blocks_the_loop(tmp_path):
    bars = _gap_bars(start="2026-09-14 10:30")
    auto, broker = _auto(tmp_path, broker=FakeBroker(bars=bars))
    auto.candidates = lambda: [_cand("X")]
    late = bars.index[-1].to_pydatetime() + timedelta(minutes=20)
    c = auto.cycle(now=late)
    assert not c.sent
    assert any("behind" in b for b in c.blocks)


# -- choosing -------------------------------------------------------------

def test_it_refuses_to_run_anything_when_nothing_has_an_edge(tmp_path):
    """The loop standing down is a result, not a failure."""
    auto, broker = _auto(tmp_path)
    auto.candidates = lambda: [_cand("Loser", bps=-5.0)]
    c = auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    assert c.running == ""
    assert broker.submitted == []
    assert any("usable edge" in b for b in c.blocks)


def test_it_picks_the_best_evidenced_rule(tmp_path):
    auto, _ = _auto(tmp_path)
    auto.candidates = lambda: [_cand("Weak", bps=+2.0), _cand("Strong", bps=+20.0)]
    c = auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    assert c.running == "Strong"


def test_it_does_not_re_choose_every_cycle(tmp_path):
    """A recommender consulted constantly will find a reason to switch, and
    each switch costs the spread."""
    auto, _ = _auto(tmp_path, review_every_minutes=15.0)
    auto.candidates = lambda: [_cand("First", bps=+20.0)]
    auto.cycle(now=datetime(2026, 9, 14, 10, 40))

    auto.candidates = lambda: [_cand("Second", bps=+99.0)]
    soon = auto.cycle(now=datetime(2026, 9, 14, 10, 45))
    assert soon.running == "First", "too soon to reconsider"

    later = auto.cycle(now=datetime(2026, 9, 14, 11, 5))
    assert later.running == "Second"


# -- instructions from Claude ---------------------------------------------

def test_a_pause_instruction_stops_new_positions(tmp_path):
    auto, broker = _auto(tmp_path)
    auto.candidates = lambda: [_cand("X")]
    auto.bridge.instruct(Instruction(action="pause", reason="stepping in"))
    c = auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    assert "paused" in c.instruction
    assert broker.submitted == []
    assert any("paused" in b for b in c.blocks)


def test_resume_undoes_a_pause(tmp_path):
    auto, _ = _auto(tmp_path, paused=True)
    auto.candidates = lambda: [_cand("X")]
    auto.bridge.instruct(Instruction(action="resume", reason="carry on"))
    auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    assert auto.cfg.paused is False


def test_a_switch_instruction_overrides_the_choice(tmp_path):
    auto, _ = _auto(tmp_path)
    auto.candidates = lambda: [_cand("Chosen", bps=+20.0)]
    auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    auto.bridge.instruct(Instruction(action="switch", strategy="Forced",
                                     reason="news"))
    c = auto.cycle(now=datetime(2026, 9, 14, 10, 41))
    assert auto._running == "Forced"
    assert "Forced" in c.instruction


def test_a_close_instruction_flattens(tmp_path):
    from core.broker import Position
    broker = FakeBroker(bars=_gap_bars(),
                        positions={"SPY": Position("SPY", 5, 11.0)})
    auto, _ = _auto(tmp_path, broker=broker)
    auto.candidates = lambda: [_cand("X")]
    auto.bridge.instruct(Instruction(action="close", reason="get out"))
    auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    assert broker.closed == ["SPY"]


@pytest.mark.parametrize("risk", [0.0, -0.01, 0.5, 1.0])
def test_a_size_instruction_outside_sane_bounds_is_ignored(tmp_path, risk):
    """The instruction that can do real damage is bounded, not trusted."""
    auto, _ = _auto(tmp_path)
    before = auto.cfg.risk_frac
    auto.candidates = lambda: [_cand("X")]
    auto.bridge.instruct(Instruction(action="size", risk_frac=risk, reason="!"))
    c = auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    assert auto.cfg.risk_frac == before
    assert "ignored" in c.instruction


def test_a_sane_size_instruction_is_applied(tmp_path):
    auto, _ = _auto(tmp_path)
    auto.candidates = lambda: [_cand("X")]
    auto.bridge.instruct(Instruction(action="size", risk_frac=0.01, reason="ok"))
    auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    assert auto.cfg.risk_frac == pytest.approx(0.01)


def test_an_instruction_fires_only_once(tmp_path):
    auto, _ = _auto(tmp_path)
    auto.candidates = lambda: [_cand("X")]
    auto.bridge.instruct(Instruction(action="pause", reason="once"))
    first = auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    second = auto.cycle(now=datetime(2026, 9, 14, 10, 41))
    assert "paused" in first.instruction
    assert second.instruction == ""


# -- what it publishes ----------------------------------------------------

def test_every_cycle_publishes_state(tmp_path):
    auto, _ = _auto(tmp_path)
    auto.candidates = lambda: [_cand("X")]
    auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    s = auto.bridge.state()
    assert s["mode"] == "auto"
    assert s["running"] == "X"
    assert "headline" in s and "session" in s and "config" in s


def test_doing_nothing_is_logged_as_well_as_acting(tmp_path):
    auto, _ = _auto(tmp_path)
    auto.candidates = lambda: [_cand("Loser", bps=-5.0)]
    auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    rows = auto.bridge.decisions()
    assert any(r["event"] == "cycle" for r in rows)
    assert any("no" in str(r.get("headline", "")).lower() for r in rows)


def test_a_broker_failure_is_reported_not_raised(tmp_path):
    from core.broker import BrokerError

    class Broken(FakeBroker):
        def get_account(self):
            raise BrokerError("network down")

    auto, _ = _auto(tmp_path, broker=Broken(bars=_gap_bars()))
    c = auto.cycle(now=datetime(2026, 9, 14, 10, 40))
    assert any("unreachable" in b for b in c.blocks)
    assert auto.bridge.state()["blocks"]


# -- the throttle ---------------------------------------------------------

def test_cycles_cannot_be_run_back_to_back(tmp_path):
    """A second cycle inside one five-minute bar reaches the same conclusion.

    Except when it reaches it again as a second order against the same
    signal, which is how one setup quietly becomes two positions. Clicking
    the button twice must not do that.
    """
    auto, broker = _auto(tmp_path)
    auto.candidates = lambda: [_cand("X")]
    t0 = datetime(2026, 9, 14, 10, 40)

    first = auto.cycle(now=t0)
    assert not first.throttled

    again = auto.cycle(now=t0 + timedelta(seconds=5))
    assert again.throttled
    assert any("too soon" in b for b in again.blocks)
    assert len(broker.submitted) <= 1, "the second click must not order again"


def test_the_cooldown_expires(tmp_path):
    auto, _ = _auto(tmp_path)
    auto.candidates = lambda: [_cand("X")]
    t0 = datetime(2026, 9, 14, 10, 40)
    auto.cycle(now=t0)
    assert auto.cycle(now=t0 + timedelta(seconds=45)).throttled is False


def test_seconds_until_ready_counts_down(tmp_path):
    auto, _ = _auto(tmp_path)
    auto.candidates = lambda: [_cand("X")]
    t0 = datetime(2026, 9, 14, 10, 40)
    assert auto.seconds_until_ready(t0) == 0.0, "nothing has run yet"
    auto.cycle(now=t0)
    assert auto.seconds_until_ready(t0 + timedelta(seconds=10)) == pytest.approx(20.0)
    assert auto.seconds_until_ready(t0 + timedelta(seconds=60)) == 0.0


def test_a_throttled_cycle_does_not_publish_over_the_real_state(tmp_path):
    """The dashboard must not show 'too soon' as if it were the loop's view."""
    auto, _ = _auto(tmp_path)
    auto.candidates = lambda: [_cand("X")]
    t0 = datetime(2026, 9, 14, 10, 40)
    auto.cycle(now=t0)
    before = auto.bridge.state()
    auto.cycle(now=t0 + timedelta(seconds=2))
    assert auto.bridge.state()["headline"] == before["headline"]
