"""The intraday paper loop for `core.daytrade`.

This exists to test *execution*, not to make money. The rule it runs prices
negative after costs in every window except the open (see `core.daytrade`),
and even there the margin is inside the spread's own dispersion. What is
genuinely unknown is whether a resting limit fills in life the way the
backtest assumes it does, and that question can only be answered by sending
orders and measuring what comes back.

So the loop's real output is `core.fills`, not P&L. Read the fill rate and the
slippage; the equity curve over fifty trades cannot distinguish a 3 bps edge
from zero and should not be consulted as though it could.

Safety, in the order it is enforced:

  1. **Paper only.** `run_once` refuses a broker that is not paper. There is no
     flag to override it -- promoting this to live means editing the file and
     noticing you did.
  2. **Plan, then execute.** `plan()` reads and decides; `execute()` sends.
     They are separate calls so a UI can show the orders before any exist, and
     so the tests can assert on a plan without a broker.
  3. **Session-derived times.** Nothing is hardcoded to 16:00; the flatten
     deadline comes from the venue calendar, so a half-day closes on time.
  4. **Exits live at the venue.** Entries go out as brackets, so the stop and
     target are attached by the broker the moment the entry fills. A stop that
     exists only inside this loop protects nothing if the process dies, the
     laptop sleeps, or the network drops while a position is open.
  5. **Flat at the end.** Every position is closed before the deadline. An
     overnight hold is a bug in a strategy defined by not having them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.broker import Broker, BrokerError
from core.data import session as rth
from core.daytrade import DayTradeConfig, Setup, find_setups
from core.fills import FillLog, FillRecord, mid
from core.marketclock import CalendarError, MarketCalendar, Session

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "logs" / "daytrader_state.json"
FILLS = ROOT / "logs" / "daytrade_fills.jsonl"
HALT = ROOT / "HALT"


class NotPaper(RuntimeError):
    """Raised when asked to run against anything but a paper account."""


@dataclass
class Intent:
    """One order the loop wants to send. Nothing has been sent yet."""
    setup: Setup
    qty: int
    reference_px: float
    note: str = ""

    @property
    def notional(self) -> float:
        return self.qty * self.setup.entry_px


@dataclass
class DayTraderConfig:
    symbols: tuple[str, ...] = ("SPY", "QQQ")
    rule: DayTradeConfig = field(default_factory=DayTradeConfig)
    max_daily_loss_frac: float = 0.02      # stand down for the day past this
    max_open_positions: int = 2
    max_order_notional: float | None = None
    bars_lookback: int = 120               # enough for swings plus context


class DayTrader:
    def __init__(self, broker: Broker, cfg: DayTraderConfig,
                 calendar: MarketCalendar | None = None):
        self.broker = broker
        self.cfg = cfg
        self.calendar = calendar or MarketCalendar.load()
        self.fills = FillLog(FILLS)
        self.state_path = STATE
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    # -- state -----------------------------------------------------------

    def _state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self, **fields):
        s = self._state()
        s.update(fields)
        self.state_path.write_text(json.dumps(s, indent=1, default=str),
                                   encoding="utf-8")

    def session_baseline(self, account) -> float:
        """Equity at the session's first look, persisted.

        On disk rather than on the instance: a Streamlit page rebuilds its
        objects on every click, so an in-memory baseline resets constantly and
        the daily-loss rail silently never fires.
        """
        today = str(pd.Timestamp.now().date())
        s = self._state()
        if s.get("baseline_day") != today:
            self._save(baseline_day=today, baseline_equity=float(account.equity))
            return float(account.equity)
        return float(s.get("baseline_equity", account.equity))

    # -- rails -----------------------------------------------------------

    def check_rails(self, account, session: Session, now: datetime) -> list[str]:
        """Every reason not to open something new. Empty means clear."""
        blocks: list[str] = []

        if HALT.exists():
            blocks.append("HALT file present -- delete it to resume")
        if getattr(account, "blocked", False):
            blocks.append("the broker has blocked this account")
        if not session.contains(now):
            blocks.append(f"outside the session ({session.open:%H:%M}-{session.close:%H:%M})")
        if now >= session.flatten_deadline(self.cfg.rule.flatten_minutes_before_close):
            blocks.append("past the flatten deadline -- closing only")

        base = self.session_baseline(account)
        if base > 0:
            drop = (base - float(account.equity)) / base
            if drop >= self.cfg.max_daily_loss_frac:
                blocks.append(f"down {drop:.1%} today, past the "
                              f"{self.cfg.max_daily_loss_frac:.0%} daily limit")
        return blocks

    def order_cap(self, account) -> float:
        if self.cfg.max_order_notional:
            return float(self.cfg.max_order_notional)
        return max(float(account.equity) * 0.25, 1_000.0)

    # -- deciding --------------------------------------------------------

    def plan(self, now: datetime | None = None) -> dict:
        """What the loop would do. Reads only -- sends nothing.

        Returns the intents alongside every reason it might not act, so a UI
        can show the whole picture rather than an empty list with no
        explanation.
        """
        now = now or datetime.now()
        account = self.broker.get_account()

        try:
            session = self.calendar.session(now)
        except CalendarError as exc:
            return {"intents": [], "blocks": [str(exc)], "session": None,
                    "positions": {}, "equity": float(account.equity)}

        positions = self.broker.get_positions()
        blocks = self.check_rails(account, session, now)

        flatten_at = session.flatten_deadline(self.cfg.rule.flatten_minutes_before_close)
        closing = now >= flatten_at

        intents: list[Intent] = []
        if not blocks and not closing:
            room = self.cfg.max_open_positions - len(positions)
            cap = self.order_cap(account)
            for symbol in self.cfg.symbols:
                if room <= 0:
                    break
                if symbol.upper() in positions:
                    continue
                try:
                    bars = self.broker.get_bars(symbol, self.cfg.rule.timeframe,
                                                self.cfg.bars_lookback)
                except BrokerError as exc:
                    blocks.append(f"{symbol}: no bars ({exc})")
                    continue
                if bars is None or len(bars) < 3:
                    continue

                # Regular hours only, matching the backtest. Alpaca hands back
                # pre- and post-market bars by default, and a gap formed at
                # 04:15 on two hundred shares of volume is not the same object
                # the rule was measured on.
                # Never look at a bar stamped later than the moment being
                # planned for. In live use the newest bar is the current one so
                # this changes nothing, but it makes the loop honest under
                # clock skew, and lets a past timestamp be replayed as it
                # actually looked rather than with the rest of the day visible.
                bars = bars[bars.index <= pd.Timestamp(now)]
                today = rth(bars[bars.index.normalize() == pd.Timestamp(now.date())])
                if len(today) < 3:
                    continue

                setups = find_setups(today, symbol, self.cfg.rule, session)
                if not setups:
                    continue
                setup = setups[-1]
                # Only act on a setup from the bar that just closed. An older
                # one has had its level traded through already.
                if setup.signal_ts != today.index[-1]:
                    continue

                qty = setup.shares(float(account.equity), self.cfg.rule.risk_frac)
                if qty <= 0:
                    continue
                if qty * setup.entry_px > cap:
                    qty = int(cap // setup.entry_px)
                if qty <= 0:
                    continue

                intents.append(Intent(setup=setup, qty=qty,
                                      reference_px=float(today["close"].iloc[-1])))
                room -= 1

        return {
            "intents": intents,
            "blocks": blocks,
            "closing": closing,
            "session": session,
            "flatten_at": flatten_at,
            "positions": positions,
            "equity": float(account.equity),
            "baseline": self.session_baseline(account),
            "now": now,
        }

    # -- acting ----------------------------------------------------------

    def execute(self, intents: list[Intent]) -> list[dict]:
        """Send the approved intents. Paper accounts only."""
        if not getattr(self.broker, "is_paper", False):
            raise NotPaper(
                "DayTrader runs on paper accounts only. This rule prices "
                "negative after costs in every window except the open; it is "
                "here to measure fills, not to be funded.")

        sent = []
        for it in intents:
            s = it.setup
            try:
                order = self.broker.submit_order(
                    s.symbol, it.qty, s.side, order_type="limit",
                    limit_price=round(s.entry_px, 2), time_in_force="day",
                    stop_loss=round(s.stop_px, 2),
                    take_profit=round(s.target_px, 2))
            except BrokerError as exc:
                log.error("%s rejected: %s", s.symbol, exc)
                sent.append({"symbol": s.symbol, "status": "rejected", "error": str(exc)})
                continue

            self.fills.record(FillRecord(
                symbol=s.symbol, side=s.side, qty=float(it.qty),
                reference_price=it.reference_px, order_type="limit",
                limit_price=float(s.entry_px),
                filled_qty=float(getattr(order, "filled_qty", 0.0) or 0.0),
                filled_price=getattr(order, "filled_price", None),
                status=getattr(order, "status", "submitted"),
                strategy="daytrade_fvg_limit", order_id=getattr(order, "id", "")))

            sent.append({"symbol": s.symbol, "qty": it.qty, "side": s.side,
                         "limit": round(s.entry_px, 2), "stop": round(s.stop_px, 2),
                         "target": round(s.target_px, 2),
                         "order_id": getattr(order, "id", ""),
                         "status": getattr(order, "status", "submitted")})
        return sent

    def flatten(self) -> list[dict]:
        """Close everything and cancel working orders.

        Called at the deadline, and safe to call at any time -- the honest
        response to "I do not know what state I am in" is to be flat.
        """
        out = []
        try:
            cancelled = self.broker.cancel_all_orders()
            out.append({"action": "cancel_all", "count": cancelled})
        except BrokerError as exc:
            out.append({"action": "cancel_all", "error": str(exc)})

        for symbol in list(self.broker.get_positions()):
            try:
                self.broker.close_position(symbol)
                out.append({"action": "close", "symbol": symbol})
            except BrokerError as exc:
                out.append({"action": "close", "symbol": symbol, "error": str(exc)})
        return out

    def run_once(self, intents: list[Intent] | None = None,
                 now: datetime | None = None) -> dict:
        """One cycle. Pass `intents` to send a plan the user actually saw.

        Recomputing inside execute would mean the orders sent are not
        necessarily the ones approved -- the bars move between the click and
        the send.
        """
        plan = self.plan(now=now)
        if plan.get("closing"):
            plan["flattened"] = self.flatten()
            return plan
        chosen = plan["intents"] if intents is None else intents
        if plan["blocks"]:
            plan["sent"] = []
            return plan
        plan["sent"] = self.execute(chosen)
        return plan
