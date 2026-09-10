"""Trading a whole book, with the reasoning overlay applied on top.

The single-symbol loop in `core.trader` asks "should I hold SPY today?". This
one asks "of these fourteen, what should the whole book look like today?" --
which is the shape of everything built here since the panel engine.

The order matters and is deliberate:

    base strategy  ->  overlay tilts  ->  whole-share rounding  ->  orders

The base runs first and alone, so its target is always recoverable. The overlay
is a separate, bounded, expiring adjustment on top. That separation is what
makes the reasoning layer's contribution measurable rather than tangled into
the strategy -- and it means switching the overlay off is one flag, not a
rewrite.

The rails match the single-symbol loop, because the ways to lose money by
accident do not change with the number of instruments: a HALT file, a persisted
daily-loss baseline, a per-order notional cap, a gross-exposure ceiling, whole
shares only, no second order while one is still working, and an audit log of
every cycle that sent anything.

Two of those exist because of specific failures. The daily-loss baseline is
persisted rather than held on the instance: Streamlit rebuilds this object on
every button press, and an in-memory baseline silently re-anchors to the
already-drawn-down equity. And `run_once` accepts a pre-computed plan, so a
confirmation button sends the orders the user actually saw rather than
recomputing a fresh book behind their back.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .broker import Broker, BrokerError
from .overlay import apply_overlay, active_overlays
from .panel import Panel
from .trader import Halted

log = logging.getLogger(__name__)


@dataclass
class PanelLiveConfig:
    symbols: list
    strategy: str = "Equal weight all"
    params: dict = field(default_factory=dict)
    lookback: int = 400
    timeframe: str = "1Day"

    gross_target: float = 0.95        # share of equity to deploy
    min_trade_fraction: float = 0.005  # ignore rebalances smaller than this
    use_overlay: bool = True
    overlay_bucket: str = ""          # "" means any
    overlay_ledger: object = None     # None uses the project ledger
    max_tilt: float = 0.30

    dry_run: bool = True
    allow_short: bool = False
    require_market_open: bool = True
    use_closed_bars_only: bool = True
    max_daily_loss_pct: float = 3.0
    max_order_notional: float | None = None   # None derives it from the book
    max_gross: float = 1.05                   # refuse to lever past this
    halt_file: str = "HALT"
    log_dir: str = "logs/panel"


@dataclass
class PanelIntent:
    symbol: str
    target_weight: float
    current_weight: float
    target_shares: float
    current_shares: float
    price: float
    reason: str = "base"

    @property
    def delta(self) -> float:
        return self.target_shares - self.current_shares

    @property
    def notional(self) -> float:
        return abs(self.delta * self.price)


class PanelTrader:
    """Drives a whole book against a broker, base strategy plus overlay."""

    def __init__(self, broker: Broker, strategy, cfg: PanelLiveConfig):
        self.broker = broker
        self.strategy = strategy
        self.cfg = cfg
        # Persisted, not held on the instance. Streamlit builds a fresh trader
        # on every button press, so an in-memory baseline re-anchors to the
        # already-drawn-down equity and the daily-loss rail never fires.
        self.log_dir = Path(cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.log_dir / "state.json"
        self.activity_log = self.log_dir / "activity.jsonl"

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except json.JSONDecodeError:
                log.warning("state file unreadable; starting a fresh session")
        return {}

    def _session_baseline(self, account) -> float:
        today = str(date.today())
        state = self._load_state()
        if state.get("date") != today:
            state = {"date": today, "start_equity": account.equity}
            self.state_file.write_text(json.dumps(state, indent=2, default=str))
        return float(state.get("start_equity", account.equity))

    def _record(self, event: str, **fields):
        row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event,
               "broker": self.broker.name, "paper": self.broker.is_paper, **fields}
        with self.activity_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        return row

    # -- building today's target ------------------------------------------
    def target_book(self) -> tuple[pd.Series, pd.Series, dict]:
        """Returns (base weights, final weights, overlay report)."""
        bars = {}
        for symbol in self.cfg.symbols:
            try:
                df = self.broker.get_bars(symbol, self.cfg.timeframe,
                                          self.cfg.lookback)
            except (BrokerError, KeyError) as exc:
                log.error("%s: no bars (%s)", symbol, exc)
                continue
            if self.cfg.use_closed_bars_only and len(df) > 1:
                df = df.iloc[:-1]          # today's bar is still forming
            if len(df) > 30:
                bars[symbol] = df

        if len(bars) < 3:
            raise BrokerError("fewer than three symbols returned usable bars")

        weights = self.strategy.generate_weights(Panel.from_bars(bars))
        base = weights.iloc[-1].astype(float)

        report = {"applied": [], "overlay_id": None}
        final = base.copy()
        if self.cfg.use_overlay:
            bucket = self.cfg.overlay_bucket or None
            for overlay in active_overlays(bucket=bucket,
                                           path=self.cfg.overlay_ledger):
                final, report = apply_overlay(
                    final, overlay, max_tilt=self.cfg.max_tilt,
                    allow_short=self.cfg.allow_short)
        return base, final, report

    # -- reconciliation ----------------------------------------------------
    def plan(self, account, final: pd.Series) -> list[PanelIntent]:
        """Turn target weights into the trades that would get us there."""
        try:
            working = {o.symbol.upper() for o in self.broker.get_open_orders()}
        except BrokerError as exc:
            log.error("cannot read open orders (%s); standing down", exc)
            return []
        if working:
            log.info("orders still working on %s -- leaving those alone",
                     ", ".join(sorted(working)))

        positions = self.broker.get_positions()
        equity = account.equity
        deployable = equity * self.cfg.gross_target
        intents = []

        # A symbol whose data feed dropped out disappears from `final`, so it
        # would never be reconciled -- the position sits there untouched while
        # the strategy re-weights across the survivors and total exposure
        # climbs. Anything held but no longer in the book gets a target of zero.
        held_symbols = {s.upper() for s in positions}
        in_book = {str(s).upper() for s in final.index}
        for symbol in sorted(held_symbols - in_book):
            if symbol in working:
                continue
            held = positions[symbol]
            price = float(held.market_value / held.qty) if held.qty else 0.0
            if price <= 0:
                log.error("%s: held but not in the book, and unpriceable", symbol)
                continue
            log.warning("%s: held but no longer in the book -- flattening", symbol)
            intents.append(PanelIntent(
                symbol=symbol, target_weight=0.0,
                current_weight=held.qty * price / equity if equity > 0 else 0.0,
                target_shares=0.0, current_shares=float(held.qty),
                price=price, reason="not in the book"))

        for symbol, weight in final.items():
            symbol = symbol.upper()
            if symbol in working:
                continue
            if not self.cfg.allow_short:
                weight = max(float(weight), 0.0)

            try:
                price = float(self.broker.get_bars(
                    symbol, self.cfg.timeframe, 2)["close"].iloc[-1])
            except (BrokerError, KeyError, IndexError):
                continue
            if price <= 0:
                continue

            held = positions.get(symbol)
            current_shares = float(held.qty) if held else 0.0
            target_shares = float(np.trunc(weight * deployable / price))

            current_weight = current_shares * price / equity if equity > 0 else 0.0
            intent = PanelIntent(
                symbol=symbol, target_weight=float(weight),
                current_weight=current_weight, target_shares=target_shares,
                current_shares=current_shares, price=price)

            # ignore dust: rebalancing a fraction of a percent costs more in
            # spread than the tracking error it removes
            if intent.notional < self.cfg.min_trade_fraction * equity:
                continue
            intents.append(intent)

        return intents

    # -- execution ---------------------------------------------------------
    def execute(self, intents: list[PanelIntent], account) -> list[dict]:
        results = []
        remaining_bp = max(account.buying_power, 0.0)
        cap = self.order_cap(account)

        # sells first: they free the buying power the buys need
        ordered = sorted(intents, key=lambda i: i.delta)
        for intent in ordered:
            # Whole shares. target_shares is truncated but current_shares comes
            # from the broker and can be fractional, so the difference can be
            # too -- and a fractional short leg is rejected outright.
            qty = float(np.trunc(abs(intent.delta)))
            if qty < 1:
                continue
            side = "buy" if intent.delta > 0 else "sell"

            notional = qty * intent.price
            if notional > cap:
                log.warning("%s: skipped, $%s exceeds the $%s single-order cap",
                            intent.symbol, f"{notional:,.0f}", f"{cap:,.0f}")
                results.append({"symbol": intent.symbol, "action": "skipped",
                                "reason": f"exceeds the ${cap:,.0f} order cap"})
                continue

            if side == "buy":
                needed = notional
                if needed > remaining_bp + 1e-9:
                    log.warning("%s: skipped, needs $%s of buying power, $%s left",
                                intent.symbol, f"{needed:,.0f}", f"{remaining_bp:,.0f}")
                    results.append({"symbol": intent.symbol, "action": "skipped",
                                    "reason": "insufficient buying power"})
                    continue
                remaining_bp -= needed

            if self.cfg.dry_run:
                results.append({"symbol": intent.symbol, "action": "would " + side,
                                "qty": qty, "price": intent.price,
                                "notional": intent.notional})
                continue

            try:
                order = self.broker.submit_order(intent.symbol, qty, side)
                results.append({"symbol": intent.symbol, "action": side,
                                "qty": qty, "order_id": order.id,
                                "notional": intent.notional})
            except BrokerError as exc:
                log.error("%s: order rejected (%s)", intent.symbol, exc)
                results.append({"symbol": intent.symbol, "action": "rejected",
                                "reason": str(exc)})
        return results

    # -- rails -------------------------------------------------------------
    def check_rails(self, account):
        # Resolved against this file's project root, not the working directory:
        # launched from elsewhere, a cwd-relative kill switch silently does
        # nothing, which is the worst possible failure for a kill switch.
        halt = Path(self.cfg.halt_file)
        if not halt.is_absolute():
            halt = Path(__file__).resolve().parent.parent / halt
        if halt.exists():
            raise Halted(f"{halt.name} file present")

        if account.blocked:
            raise Halted("the broker has this account blocked from trading")
        if account.equity <= 0:
            raise Halted("account equity is zero or negative")

        baseline = self._session_baseline(account)
        if baseline > 0:
            change = account.equity / baseline - 1.0
            if change <= -abs(self.cfg.max_daily_loss_pct) / 100.0:
                raise Halted(f"daily loss {change:.2%} hit the "
                             f"{self.cfg.max_daily_loss_pct:.2f}% limit")

    def order_cap(self, account) -> float:
        """Largest single order allowed, so one bad target cannot dominate."""
        if self.cfg.max_order_notional:
            return float(self.cfg.max_order_notional)
        per_symbol = account.equity * self.cfg.gross_target / max(
            len(self.cfg.symbols), 1)
        # a full reversal is twice a one-way position, so allow for it
        return per_symbol * (2.5 if self.cfg.allow_short else 1.5)

    def run_once(self, intents: list[PanelIntent] | None = None) -> dict:
        """One full cycle. Returns everything that happened, for display.

        Pass `intents` to execute a plan the user has already seen and
        approved. Without it the book is recomputed, which means the orders
        sent can differ in count, size and side from the ones consented to --
        fine for an unattended loop, wrong behind a confirmation button.
        """
        account = self.broker.get_account()
        self.check_rails(account)

        if self.cfg.require_market_open:
            clock = self.broker.get_clock()
            if not clock.is_open:
                return {"status": "market closed", "intents": [], "results": []}

        base, final, overlay_report = self.target_book()
        if intents is None:
            intents = self.plan(account, final)

        # Refuse to lever past the cap, whatever the targets say.
        planned_gross = float(final.abs().sum())
        if planned_gross > self.cfg.max_gross:
            raise Halted(f"target book is {planned_gross:.2f}x gross, above the "
                         f"{self.cfg.max_gross:.2f}x cap")

        results = self.execute(intents, account)
        if not self.cfg.dry_run and results:
            self._record("cycle", equity=account.equity, results=results,
                         overlay=overlay_report.get("applied", []))

        return {
            "status": "dry run" if self.cfg.dry_run else "executed",
            "equity": account.equity,
            "buying_power": account.buying_power,
            "base": base, "final": final,
            "overlay": overlay_report,
            "intents": intents, "results": results,
            "planned_gross": planned_gross,
        }
