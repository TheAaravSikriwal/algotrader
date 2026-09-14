"""The automatic loop: pick an algorithm, trade it, and say why.

One cycle does four things, in order:

  1. **Obey.** If Claude has left an instruction in `live/instructions.json`,
     act on it first and mark it consumed. A human-issued instruction outranks
     whatever the loop was about to do on its own.
  2. **Choose.** Pick the best-evidenced rule for the current conditions from
     what has been backtested as a day trade. Sticking beats switching unless
     the challenger is clearly better -- switching on noise converts it into
     turnover and turnover into spread.
  3. **Act.** Run the chosen rule's signal, size it, place or exit.
  4. **Publish.** Write the whole picture to `live/state.json` so it can be
     seen, and log the decision -- including the decision to do nothing,
     because "why didn't it trade?" is the question that actually gets asked.

Every rail from the manual path still applies: paper accounts only, session
window, flatten deadline, daily loss limit, position caps, stale-data refusal.
The loop cannot do anything a person could not do by hand on the same page.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from core.broker import Broker, BrokerError
from core.daytrade import DayTradeConfig
from core.daytrader import DayTrader, DayTraderConfig, NotPaper
from core.livestate import Bridge, Instruction
from core.marketclock import CalendarError, MarketCalendar
from core.recommend import Candidate, best_intraday, load_intraday, rank, review

log = logging.getLogger(__name__)


@dataclass
class AutoConfig:
    symbols: tuple[str, ...] = ("SPY", "QQQ")
    risk_frac: float = 0.005
    max_daily_loss_frac: float = 0.02
    max_open_positions: int = 2
    max_bar_age_minutes: float = 6.0
    #: Re-check which algorithm to run this often. Every cycle would churn.
    review_every_minutes: float = 15.0
    #: Shortest gap between cycles. The rule works on five-minute bars, so
    #: running it twice inside one bar can only produce the same answer --
    #: except when it produces a second order against the same signal, which
    #: is how one setup becomes two positions.
    min_seconds_between_cycles: float = 30.0
    paused: bool = False


@dataclass
class Cycle:
    """What one pass of the loop saw, chose and did."""
    at: datetime
    running: str = ""
    chose_because: str = ""
    switched_from: str = ""
    instruction: str = ""
    blocks: list[str] = field(default_factory=list)
    intents: list[dict] = field(default_factory=list)
    sent: list[dict] = field(default_factory=list)
    flattened: list[dict] = field(default_factory=list)
    equity: float = 0.0
    positions: dict = field(default_factory=dict)
    candidate: dict = field(default_factory=dict)
    throttled: bool = False

    @property
    def acted(self) -> bool:
        return bool(self.sent or self.flattened)

    def headline(self) -> str:
        if self.flattened:
            return "Closed out"
        if self.sent:
            return f"Placed {len(self.sent)} order(s)"
        if self.blocks:
            return self.blocks[0]
        if self.intents:
            return "Setup found, awaiting approval"
        return "Watching — no setup"


class AutoTrader:
    """Runs the day-trading loop without asking, on a paper account."""

    def __init__(self, broker: Broker, cfg: AutoConfig | None = None,
                 calendar: MarketCalendar | None = None,
                 bridge: Bridge | None = None):
        self.cfg = cfg or AutoConfig()
        self.broker = broker
        self.calendar = calendar or MarketCalendar.load()
        self.bridge = bridge or Bridge()
        self._last_review: datetime | None = None
        self._last_cycle_at: datetime | None = None
        self._running: str = ""
        self._trader = self._build_trader()

    def _build_trader(self) -> DayTrader:
        rule = DayTradeConfig(symbols=self.cfg.symbols,
                              risk_frac=self.cfg.risk_frac)
        return DayTrader(
            self.broker,
            DayTraderConfig(symbols=self.cfg.symbols, rule=rule,
                            max_daily_loss_frac=self.cfg.max_daily_loss_frac,
                            max_open_positions=self.cfg.max_open_positions,
                            max_bar_age_minutes=self.cfg.max_bar_age_minutes),
            self.calendar)

    # -- choosing ---------------------------------------------------------

    def candidates(self) -> list[Candidate]:
        return [c for c in load_intraday() if c.symbol in self.cfg.symbols]

    def choose(self, now: datetime, force: bool = False) -> tuple[str, str, str]:
        """Which rule to run. Returns (running, reason, switched_from).

        Re-checked on a timer rather than every cycle. A recommender consulted
        every thirty seconds will eventually find a reason to switch, and each
        switch costs the spread.
        """
        due = force or self._last_review is None or (
            (now - self._last_review).total_seconds()
            >= self.cfg.review_every_minutes * 60)
        if not due:
            return self._running, "", ""

        self._last_review = now
        pool = self.candidates()
        rec = (best_intraday(candidates=pool) if not self._running
               else review(self._running, candidates=pool))

        if rec.action == "stand_aside" or rec.best is None:
            was, self._running = self._running, ""
            return "", rec.reason, was
        if rec.action == "hold":
            return self._running, rec.reason, ""

        was, self._running = self._running, rec.best.strategy
        return self._running, rec.reason, was

    # -- instructions -----------------------------------------------------

    def apply_instruction(self, ins: Instruction) -> str:
        """Act on one instruction and describe what it did."""
        if ins.action == "pause":
            self.cfg.paused = True
            return f"paused — {ins.reason}"
        if ins.action == "resume":
            self.cfg.paused = False
            return f"resumed — {ins.reason}"
        if ins.action == "switch":
            if not ins.strategy:
                return "ignored: switch with no strategy named"
            was, self._running = self._running, ins.strategy
            self._last_review = datetime.now()
            return f"switched {was or 'nothing'} -> {ins.strategy} — {ins.reason}"
        if ins.action == "size":
            if ins.risk_frac is None or not (0 < ins.risk_frac <= 0.05):
                # A size instruction is the one that can do real damage, so
                # it is bounded here rather than trusted.
                return f"ignored: risk {ins.risk_frac} outside 0-5%"
            self.cfg.risk_frac = float(ins.risk_frac)
            self._trader = self._build_trader()
            return f"risk per trade set to {ins.risk_frac:.2%} — {ins.reason}"
        if ins.action == "close":
            done = self._trader.flatten()
            return f"closed {ins.symbol or 'everything'} — {ins.reason}"
        return f"hold — {ins.reason}"

    # -- one pass ---------------------------------------------------------

    def seconds_until_ready(self, now: datetime | None = None) -> float:
        """How long before another cycle is allowed. Zero means now.

        The rule reads five-minute bars, so a second cycle inside one bar can
        only reach the same conclusion -- except when it reaches it again as a
        second order against the same signal, which is how one setup quietly
        becomes two positions.
        """
        if self._last_cycle_at is None:
            return 0.0
        waited = ((now or datetime.now()) - self._last_cycle_at).total_seconds()
        return max(self.cfg.min_seconds_between_cycles - waited, 0.0)

    def cycle(self, now: datetime | None = None,
              execute: bool = True) -> Cycle:
        now = now or datetime.now()

        wait = self.seconds_until_ready(now)
        if wait > 0:
            c = Cycle(at=now, running=self._running)
            c.blocks = [f"too soon — another cycle in {wait:.0f}s"]
            c.throttled = True
            return c

        self._last_cycle_at = now
        c = Cycle(at=now)

        ins = self.bridge.pending()
        if ins is not None:
            c.instruction = self.apply_instruction(ins)
            self.bridge.consume(ins.id)
            self.bridge.record("instruction", id=ins.id, action=ins.action,
                               result=c.instruction)

        running, why, was = self.choose(now)
        c.running, c.chose_because, c.switched_from = running, why, was
        if was and running and was != running:
            self.bridge.record("switch", **{"from": was, "to": running,
                                            "why": why})

        try:
            account = self.broker.get_account()
            c.equity = float(account.equity)
            c.positions = {s: {"qty": p.qty, "avg_price": p.avg_price,
                               "market_value": p.market_value,
                               "unrealized_pl": p.unrealized_pl}
                           for s, p in self.broker.get_positions().items()}
        except BrokerError as exc:
            c.blocks = [f"broker unreachable: {exc}"]
            self._publish(c)
            return c

        plan = self._trader.plan(now=now)
        c.blocks = list(plan.get("blocks", []))
        c.intents = [{"symbol": i.setup.symbol, "side": i.setup.side,
                      "qty": i.qty, "limit": round(i.setup.entry_px, 2),
                      "stop": round(i.setup.stop_px, 2),
                      "target": round(i.setup.target_px, 2),
                      "risk": round(i.qty * i.setup.risk_per_share, 2)}
                     for i in plan.get("intents", [])]

        if plan.get("closing"):
            c.flattened = plan.get("flattened") or self._trader.flatten()
            self.bridge.record("flatten", reason="past the deadline",
                               result=c.flattened)
        elif self.cfg.paused:
            c.blocks.append("paused by instruction")
        elif not running:
            c.blocks.append("no algorithm running — nothing has a usable edge")
        elif execute and plan.get("intents") and not plan["blocks"]:
            try:
                c.sent = self._trader.execute(plan["intents"])
                self.bridge.record("orders", running=running, sent=c.sent)
            except NotPaper as exc:
                c.blocks.append(str(exc))
            except Exception as exc:                      # noqa: BLE001
                c.blocks.append(f"order failed: {exc}")

        best = next((x for x in rank(self.candidates())
                     if x.strategy == running), None)
        if best:
            c.candidate = {"strategy": best.strategy, "symbol": best.symbol,
                           "expectancy_bps": best.expectancy_pct,
                           "t_stat": best.t_stat, "trades": best.trades,
                           "credible": best.credible, "note": best.note}

        self._publish(c)
        self.bridge.record("cycle", headline=c.headline(), running=running,
                           blocks=c.blocks, intents=len(c.intents),
                           sent=len(c.sent), equity=c.equity)
        return c

    def _publish(self, c: Cycle):
        try:
            session = self.calendar.session(c.at)
            window = {"open": f"{session.open:%H:%M}",
                      "close": f"{session.close:%H:%M}",
                      "flatten_at": f"{session.flatten_deadline(5):%H:%M}",
                      "is_open": session.contains(c.at)}
        except CalendarError:
            window = {"is_open": False}

        self.bridge.publish({
            "at": c.at.isoformat(),
            "mode": "auto",
            "paused": self.cfg.paused,
            "running": c.running,
            "chose_because": c.chose_because,
            "switched_from": c.switched_from,
            "instruction_applied": c.instruction,
            "headline": c.headline(),
            "blocks": c.blocks,
            "intents": c.intents,
            "sent": c.sent,
            "flattened": c.flattened,
            "equity": c.equity,
            "positions": c.positions,
            "candidate": c.candidate,
            "session": window,
            "config": {"symbols": list(self.cfg.symbols),
                       "risk_frac": self.cfg.risk_frac,
                       "max_daily_loss_frac": self.cfg.max_daily_loss_frac,
                       "max_bar_age_minutes": self.cfg.max_bar_age_minutes},
        })
