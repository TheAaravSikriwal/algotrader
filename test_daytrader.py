"""The intraday paper loop.

The rails are the point. This rule is not expected to make money, so the only
way it can hurt is by doing something the user did not sanction -- trading
live, holding overnight, or sizing past the limit it promised.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from core.broker import Account, Broker, Order, Position
from core.daytrade import DayTradeConfig
from core.daytrader import DayTrader, DayTraderConfig, NotPaper
from core.marketclock import MarketCalendar

DAY = date(2026, 9, 14)


def _cal():
    return MarketCalendar.from_rows([
        {"date": DAY, "open": "09:30", "close": "16:00"},
        {"date": date(2026, 11, 27), "open": "09:30", "close": "13:00"},
    ])



@pytest.fixture(autouse=True)
def _isolate_logs(tmp_path, monkeypatch):
    """Keep tests out of the project's real fill log.

    Two tests called execute() without redirecting the log and wrote 105 fake
    orders into logs/daytrade_fills.jsonl, which then showed up in the UI as a
    0% fill rate on orders that were never sent. Test data reaching a file the
    app reads is worse than a failing test: it looks like evidence.
    """
    import core.daytrader as mod
    monkeypatch.setattr(mod, "FILLS", tmp_path / "fills.jsonl")
    monkeypatch.setattr(mod, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(mod, "HALT", tmp_path / "HALT")

class FakeBroker(Broker):
    """Records what it was asked to do; invents nothing."""
    name = "fake"

    def __init__(self, bars=None, equity=100_000.0, is_paper=True, positions=None):
        self.is_paper = is_paper
        self._bars = bars if bars is not None else pd.DataFrame()
        self._equity = equity
        self._positions = positions or {}
        self.submitted: list[dict] = []
        self.cancelled = 0
        self.closed: list[str] = []

    def get_account(self):
        return Account(cash=self._equity, equity=self._equity,
                       buying_power=self._equity * 4, is_paper=self.is_paper)

    def get_positions(self):
        return dict(self._positions)

    def get_clock(self):
        raise NotImplementedError

    def get_bars(self, symbol, timeframe="5Min", limit=300):
        return self._bars

    def submit_order(self, symbol, qty, side, order_type="market",
                     limit_price=None, time_in_force="day",
                     stop_loss=None, take_profit=None):
        self.submitted.append({"symbol": symbol, "qty": qty, "side": side,
                               "type": order_type, "limit": limit_price,
                               "stop_loss": stop_loss, "take_profit": take_profit})
        return Order(id=f"o{len(self.submitted)}", symbol=symbol, qty=qty,
                     side=side, status="new")

    def get_open_orders(self):
        return []

    def cancel_all_orders(self):
        self.cancelled += 1
        return 1

    def close_position(self, symbol):
        self.closed.append(symbol)
        return Order(id="c1", symbol=symbol, qty=1, side="sell", status="new")


def _gap_bars(start="2026-09-14 10:30", n_pad=0):
    """Bars ending on a confirmed bullish gap: 11..12, midpoint 11.5."""
    rows = [(10, 11, 10, 11)] * (1 + n_pad) + [(11, 13, 11, 13), (13, 14, 12, 13)]
    idx = pd.date_range(start, periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=idx).assign(volume=1000.0)


def _trader(**kw):
    broker = kw.pop("broker", FakeBroker(bars=_gap_bars()))
    cfg = kw.pop("cfg", DayTraderConfig(symbols=("SPY",)))
    return DayTrader(broker, cfg, _cal()), broker


# -- live is refused ------------------------------------------------------

def test_execute_refuses_a_live_account():
    """No override flag. Going live means editing the file and noticing."""
    t, _ = _trader(broker=FakeBroker(bars=_gap_bars(), is_paper=False))
    plan = t.plan(now=datetime(2026, 9, 14, 10, 40))
    with pytest.raises(NotPaper):
        t.execute(plan["intents"])


def test_a_paper_account_is_allowed_through():
    t, broker = _trader()
    t.execute(t.plan(now=datetime(2026, 9, 14, 10, 40))["intents"])
    assert broker.submitted


# -- orders are limits, not markets --------------------------------------

def test_entries_are_resting_limits():
    """The whole measured effect is passive execution. Entering at market
    scored -0.009R, so a market entry here would not be the same strategy."""
    t, broker = _trader()
    t.execute(t.plan(now=datetime(2026, 9, 14, 10, 40))["intents"])
    assert broker.submitted[0]["type"] == "limit"
    assert broker.submitted[0]["limit"] == pytest.approx(11.5)


def test_every_send_is_written_to_the_fill_log(tmp_path):
    t, _ = _trader()
    t.fills.path = tmp_path / "f.jsonl"
    t.execute(t.plan(now=datetime(2026, 9, 14, 10, 40))["intents"])
    df = t.fills.frame()
    assert len(df) == 1
    assert df["order_type"].iloc[0] == "limit"


# -- rails ----------------------------------------------------------------

def test_nothing_is_opened_outside_the_session():
    t, _ = _trader()
    plan = t.plan(now=datetime(2026, 9, 14, 8, 0))
    assert plan["intents"] == []
    assert any("outside the session" in b for b in plan["blocks"])


def test_nothing_is_opened_past_the_flatten_deadline():
    t, _ = _trader()
    plan = t.plan(now=datetime(2026, 9, 14, 15, 56))
    assert plan["intents"] == []
    assert plan["closing"]
    # The reason is part of the contract, not decoration. `closing` alone stops
    # the trading, but a UI showing an empty plan with no explanation is how a
    # user concludes the loop is broken and starts overriding it.
    assert any("flatten deadline" in b for b in plan["blocks"])


def test_the_deadline_moves_on_a_half_day():
    """13:10 is mid-afternoon on a normal day and past the close on this one."""
    bars = _gap_bars(start="2026-11-27 11:00")
    t, _ = _trader(broker=FakeBroker(bars=bars))
    assert t.plan(now=datetime(2026, 11, 27, 12, 56))["closing"]
    assert not t.plan(now=datetime(2026, 9, 14, 12, 56))["closing"]


def test_a_halt_file_stops_everything(tmp_path):
    (tmp_path / "HALT").write_text("stop")
    t, _ = _trader()
    plan = t.plan(now=datetime(2026, 9, 14, 10, 40))
    assert plan["intents"] == []
    assert any("HALT" in b for b in plan["blocks"])


def test_a_blocked_account_stops_everything():
    class Blocked(FakeBroker):
        def get_account(self):
            a = super().get_account()
            a.blocked = True
            return a
    t, _ = _trader(broker=Blocked(bars=_gap_bars()))
    assert any("blocked" in b for b in t.plan(now=datetime(2026, 9, 14, 10, 40))["blocks"])


def test_the_daily_loss_rail_uses_a_persisted_baseline(tmp_path):
    """A Streamlit page rebuilds its objects on every click.

    An in-memory baseline resets each time, so the rail silently never fires.
    """
    broker = FakeBroker(bars=_gap_bars(), equity=100_000.0)
    t = DayTrader(broker, DayTraderConfig(symbols=("SPY",)), _cal())
    t.state_path = tmp_path / "state.json"
    t.plan(now=datetime(2026, 9, 14, 10, 40))          # records the baseline

    broker._equity = 97_000.0                          # down 3%, limit is 2%
    fresh = DayTrader(broker, DayTraderConfig(symbols=("SPY",)), _cal())
    fresh.state_path = tmp_path / "state.json"         # a new object, same file
    plan = fresh.plan(now=datetime(2026, 9, 14, 10, 45))
    assert any("daily limit" in b for b in plan["blocks"])
    assert plan["intents"] == []


def test_no_new_position_in_a_symbol_already_held():
    t, _ = _trader(broker=FakeBroker(
        bars=_gap_bars(),
        positions={"SPY": Position(symbol="SPY", qty=10, avg_price=11.0)}))
    assert t.plan(now=datetime(2026, 9, 14, 10, 40))["intents"] == []


def test_open_positions_are_capped():
    cfg = DayTraderConfig(symbols=("SPY", "QQQ"), max_open_positions=1)
    t, _ = _trader(broker=FakeBroker(bars=_gap_bars()), cfg=cfg)
    assert len(t.plan(now=datetime(2026, 9, 14, 10, 40))["intents"]) == 1


def test_a_single_order_cannot_exceed_the_notional_cap():
    cfg = DayTraderConfig(symbols=("SPY",), max_order_notional=100.0)
    t, _ = _trader(broker=FakeBroker(bars=_gap_bars(), equity=1_000_000.0), cfg=cfg)
    intents = t.plan(now=datetime(2026, 9, 14, 10, 40))["intents"]
    assert intents and intents[0].notional <= 100.0


# -- staleness ------------------------------------------------------------

def test_only_a_setup_from_the_bar_that_just_closed_is_acted_on():
    """An older level has already had its price traded through."""
    bars = _gap_bars()
    extra = pd.DataFrame([(13, 13.5, 12.8, 13.2)] * 4,
                         columns=["open", "high", "low", "close"],
                         index=pd.date_range(bars.index[-1] + pd.Timedelta("5min"),
                                             periods=4, freq="5min")).assign(volume=1000.0)
    t, _ = _trader(broker=FakeBroker(bars=pd.concat([bars, extra])))
    assert t.plan(now=datetime(2026, 9, 14, 11, 10))["intents"] == []


# -- flatten --------------------------------------------------------------

def test_flatten_cancels_orders_and_closes_positions():
    broker = FakeBroker(bars=_gap_bars(),
                        positions={"SPY": Position(symbol="SPY", qty=10, avg_price=11.0)})
    t, _ = _trader(broker=broker)
    t.flatten()
    assert broker.cancelled == 1
    assert broker.closed == ["SPY"]


def test_run_once_flattens_instead_of_trading_at_the_deadline():
    broker = FakeBroker(bars=_gap_bars(),
                        positions={"SPY": Position(symbol="SPY", qty=10, avg_price=11.0)})
    t, _ = _trader(broker=broker)
    out = t.run_once(now=datetime(2026, 9, 14, 15, 57))
    assert "flattened" in out
    assert broker.closed == ["SPY"]
    assert broker.submitted == []


def test_run_once_sends_the_approved_plan_not_a_recomputed_one():
    """Bars move between the click and the send.

    Recomputing inside execute would mean the orders sent are not the ones
    the user saw and approved.
    """
    t, broker = _trader()
    plan = t.plan(now=datetime(2026, 9, 14, 10, 40))
    approved = plan["intents"][:1]
    approved[0].qty = 3
    t.run_once(intents=approved, now=datetime(2026, 9, 14, 10, 40))
    assert broker.submitted[0]["qty"] == 3


def test_blocked_plans_send_nothing_even_when_intents_are_passed():
    t, broker = _trader()
    good = t.plan(now=datetime(2026, 9, 14, 10, 40))["intents"]
    t.run_once(intents=good, now=datetime(2026, 9, 14, 8, 0))
    assert broker.submitted == []


def test_an_uncovered_date_blocks_rather_than_assuming_hours():
    t, _ = _trader()
    plan = t.plan(now=datetime(2030, 6, 3, 11, 0))
    assert plan["intents"] == []
    assert plan["blocks"]


# -- exits live at the venue ---------------------------------------------

def test_entries_carry_their_stop_and_target_to_the_broker():
    """A stop held only in this process is not a stop.

    If the laptop sleeps or the script dies with a position open, the only
    protection left is whatever the venue was told about.
    """
    t, broker = _trader()
    plan = t.plan(now=datetime(2026, 9, 14, 10, 40))
    t.execute(plan["intents"])
    sent = broker.submitted[0]
    s = plan["intents"][0].setup
    assert sent["stop_loss"] == pytest.approx(round(s.stop_px, 2))
    assert sent["take_profit"] == pytest.approx(round(s.target_px, 2))


def test_the_bracket_brackets_the_entry():
    """Long: stop below, target above. A sign slip here inverts the trade."""
    t, broker = _trader()
    plan = t.plan(now=datetime(2026, 9, 14, 10, 40))
    t.execute(plan["intents"])
    sent = broker.submitted[0]
    assert sent["side"] == "buy"
    assert sent["stop_loss"] < sent["limit"] < sent["take_profit"]


def test_only_regular_hours_bars_feed_the_rule():
    """Alpaca returns pre- and post-market bars by default.

    A gap formed at 04:15 on two hundred shares is not the object the rule was
    measured on, and its level would never be revisited in the session.
    """
    pre = pd.DataFrame(
        [(10, 11, 10, 11), (11, 13, 11, 13), (13, 14, 12, 13)],
        columns=["open", "high", "low", "close"],
        index=pd.date_range("2026-09-14 04:15", periods=3, freq="5min"),
    ).assign(volume=200.0)
    t, _ = _trader(broker=FakeBroker(bars=pre))
    assert t.plan(now=datetime(2026, 9, 14, 10, 40))["intents"] == []


def test_bars_stamped_after_the_planning_moment_are_ignored():
    """Nothing later than `now` may inform the decision.

    In live use the newest bar is the current one, so this is invisible. It
    matters under clock skew, and it is what makes replaying a past moment
    show what was actually visible then rather than the whole day.
    """
    bars = _gap_bars()                       # gap confirmed on the third bar
    later = pd.DataFrame(
        [(13, 20, 12, 19)],                  # a huge bar that has not happened yet
        columns=["open", "high", "low", "close"],
        index=[bars.index[-1] + pd.Timedelta("5min")]).assign(volume=1000.0)
    t, _ = _trader(broker=FakeBroker(bars=pd.concat([bars, later])))

    # Planning at the gap bar: the future bar is invisible, so the setup stands.
    at_gap = t.plan(now=bars.index[-1].to_pydatetime())
    assert len(at_gap["intents"]) == 1

    # Planning one bar later: that bar is now the newest, so the gap is stale.
    assert t.plan(now=later.index[0].to_pydatetime())["intents"] == []
