"""The live trading loop.

Each cycle does the same five things the backtest engine does, except step 4
talks to a real venue:

  1. pull the latest bars
  2. ask the strategy what it *should* hold
  3. read what it *actually* holds at the broker
  4. send the difference as orders
  5. write it all down, and check the kill switch

Safety posture: `dry_run` is on by default, orders are capped by notional, a
daily-loss limit flattens and halts, and a `HALT` file in the working directory
stops everything on the next cycle without needing to kill the process.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from core.broker import Account, Broker, BrokerError
from core.strategy import Strategy

log = logging.getLogger("trader")


@dataclass
class LiveConfig:
    symbols: list[str]
    timeframe: str = "1Day"
    lookback: int = 400                 # bars fetched per cycle, needs to cover the slowest indicator
    position_size: float = 0.95         # fraction of equity to deploy across all symbols
    allow_short: bool = False
    dry_run: bool = True                # log intended orders, send nothing

    # --- risk rails ---
    max_daily_loss_pct: float = 3.0     # flatten and halt past this drawdown on the day
    # None derives the cap from account size: 1.5x a full intended position. That
    # catches a sizing bug that asks for 10x, without blocking ordinary trades the
    # way a fixed dollar figure does once the account outgrows it.
    max_order_notional: float | None = None
    max_order_notional_multiple: float = 1.5
    min_order_notional: float = 1.0     # ignore dust rebalances
    whole_shares: bool = True
    require_market_open: bool = True
    flatten_on_halt: bool = True
    use_closed_bars_only: bool = True   # drop today's partial bar, matching the backtest

    halt_file: str = "HALT"
    log_dir: str = "logs"


@dataclass
class Intent:
    """What the loop wants to do about one symbol, before it does it."""
    symbol: str
    price: float
    target_shares: float
    current_shares: float
    signal: float

    @property
    def delta(self) -> float:
        return self.target_shares - self.current_shares

    @property
    def notional(self) -> float:
        return abs(self.delta) * self.price

    @property
    def side(self) -> str:
        return "buy" if self.delta > 0 else "sell"


class Halted(RuntimeError):
    """Raised when a rail trips. The loop stops; it does not retry."""


class Trader:
    def __init__(self, broker: Broker, strategy: Strategy, config: LiveConfig):
        if not config.symbols:
            raise ValueError("no symbols configured")
        self.broker = broker
        self.strategy = strategy
        self.cfg = config

        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.activity_log = self.log_dir / "activity.jsonl"
        self.state_file = self.log_dir / "state.json"
        self.halted = False

    # ---- persistence ----------------------------------------------------
    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except json.JSONDecodeError:
                log.warning("state file was unreadable; starting a fresh session")
        return {}

    def _save_state(self, state: dict):
        self.state_file.write_text(json.dumps(state, indent=2, default=str))

    def _session_baseline(self, account: Account) -> float:
        """Equity at the start of today, so the daily-loss rail has a reference."""
        today = str(date.today())
        state = self._load_state()
        if state.get("date") != today:
            state = {"date": today, "start_equity": account.equity}
            self._save_state(state)
        return float(state.get("start_equity", account.equity))

    def _record(self, event: str, **fields):
        row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event,
               "broker": self.broker.name, "paper": self.broker.is_paper, **fields}
        with self.activity_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        return row

    # ---- rails ----------------------------------------------------------
    def check_guards(self, account: Account) -> str | None:
        """Return a reason to stop trading, or None to proceed."""
        if Path(self.cfg.halt_file).exists():
            return f"{self.cfg.halt_file} file present"
        if account.blocked:
            return "the broker has blocked trading on this account"
        if account.equity <= 0:
            return "account equity is zero or negative"

        baseline = self._session_baseline(account)
        if baseline > 0:
            loss_pct = (baseline - account.equity) / baseline * 100
            if loss_pct >= self.cfg.max_daily_loss_pct:
                return (f"daily loss {loss_pct:.2f}% reached the "
                        f"{self.cfg.max_daily_loss_pct:.2f}% limit")
        return None

    def halt(self, reason: str):
        self.halted = True
        log.error("HALTED: %s", reason)
        self._record("halt", reason=reason)
        if self.cfg.flatten_on_halt and not self.cfg.dry_run:
            try:
                self.broker.cancel_all_orders()
                for symbol in self.broker.get_positions():
                    order = self.broker.close_position(symbol)
                    if order:
                        log.warning("flattened %s", symbol)
                        self._record("flatten", symbol=symbol, order_id=order.id)
            except BrokerError as exc:
                log.error("could not flatten positions: %s", exc)
        raise Halted(reason)

    # ---- the cycle ------------------------------------------------------
    def plan(self, account: Account) -> list[Intent]:
        """Work out the target position for every symbol. Sends nothing."""
        positions = self.broker.get_positions()
        per_symbol = account.equity * self.cfg.position_size / len(self.cfg.symbols)
        intents: list[Intent] = []

        # A working order has not moved the position yet, so acting on the current
        # position would re-order the same thing. Leave those symbols alone until
        # the order fills or is cancelled.
        try:
            working = {o.symbol.upper() for o in self.broker.get_open_orders()}
        except BrokerError as exc:
            log.error("could not read open orders (%s); standing down this cycle", exc)
            return []
        if working:
            log.info("orders still working on %s -- skipping those this cycle",
                     ", ".join(sorted(working)))

        for symbol in self.cfg.symbols:
            symbol = symbol.upper()
            if symbol in working:
                continue
            try:
                bars = self.broker.get_bars(symbol, self.cfg.timeframe, self.cfg.lookback)
            except (BrokerError, KeyError) as exc:
                log.error("%s: could not fetch bars (%s)", symbol, exc)
                continue

            if self.cfg.use_closed_bars_only and len(bars) > 1:
                bars = bars.iloc[:-1]      # today's bar is still forming
            if bars.empty:
                log.error("%s: no usable bars", symbol)
                continue

            signals = self.strategy.generate_signals(bars)
            signal = float(signals.iloc[-1])
            if not self.cfg.allow_short:
                signal = max(signal, 0.0)
            signal = max(-1.0, min(1.0, signal))

            price = float(bars["close"].iloc[-1])
            if price <= 0:
                continue

            target = signal * per_symbol / price
            if self.cfg.whole_shares:
                # round toward zero so rounding never increases exposure
                target = float(int(target))

            held = positions.get(symbol)
            intents.append(Intent(symbol, price, target,
                                  held.qty if held else 0.0, signal))
        return intents

    def order_cap(self, account: Account) -> float:
        """Largest notional a single order may carry."""
        if self.cfg.max_order_notional is not None:
            return self.cfg.max_order_notional
        per_symbol = account.equity * self.cfg.position_size / max(len(self.cfg.symbols), 1)
        return max(per_symbol * self.cfg.max_order_notional_multiple, 1.0)

    @staticmethod
    def _exposure_increase(it: Intent) -> float:
        """Notional of the part of this order that *adds* risk.

        Buying power is consumed by opening or growing a position, not by
        closing one. A flip through zero only spends it on the new leg.
        """
        cur, tgt = it.current_shares, it.target_shares
        if cur == 0 or (cur > 0) == (tgt > 0):
            return max(abs(tgt) - abs(cur), 0.0) * it.price
        return abs(tgt) * it.price          # closed the old side, opened a new one

    def execute(self, intents: list[Intent], account: Account) -> list[dict]:
        results = []
        cap = self.order_cap(account)
        # Alpaca rejects orders that would create a margin deficit, so stay inside
        # buying power rather than discovering the limit through rejections.
        remaining_bp = max(account.buying_power, 0.0)

        for it in intents:
            if it.notional < self.cfg.min_order_notional or abs(it.delta) < 1e-9:
                continue

            needed = self._exposure_increase(it)
            if needed > remaining_bp + 1e-9:
                log.warning("%s: skipping -- needs $%s of buying power, $%s left",
                            it.symbol, f"{needed:,.0f}", f"{remaining_bp:,.0f}")
                results.append(self._record("order_skipped", symbol=it.symbol,
                                            reason="insufficient buying power",
                                            needed=needed, available=remaining_bp))
                continue
            remaining_bp -= needed

            if it.notional > cap:
                log.warning("%s: skipping -- order notional $%s exceeds the $%s cap",
                            it.symbol, f"{it.notional:,.0f}", f"{cap:,.0f}")
                results.append(self._record("order_skipped", symbol=it.symbol,
                                            reason="notional cap",
                                            notional=it.notional, cap=cap))
                continue

            qty = abs(it.delta)
            if self.cfg.whole_shares:
                qty = float(int(qty))
                if qty < 1:
                    continue

            if self.cfg.dry_run:
                log.info("[dry run] would %s %g %s @ ~$%.2f (signal %.2f)",
                         it.side, qty, it.symbol, it.price, it.signal)
                results.append(self._record("order_dry_run", symbol=it.symbol,
                                            side=it.side, qty=qty, price=it.price,
                                            signal=it.signal))
                continue

            try:
                order = self.broker.submit_order(it.symbol, qty, it.side)
            except BrokerError as exc:
                log.error("%s: order rejected -- %s", it.symbol, exc)
                results.append(self._record("order_rejected", symbol=it.symbol,
                                            side=it.side, qty=qty, error=str(exc)))
                continue

            log.info("%s %g %s -> order %s (%s)", it.side, qty, it.symbol,
                     order.id, order.status)
            results.append(self._record("order_sent", symbol=it.symbol, side=it.side,
                                        qty=qty, order_id=order.id,
                                        status=order.status, signal=it.signal))
        return results

    def run_once(self) -> dict:
        """One full cycle. Returns a summary dict."""
        if self.halted:
            raise Halted("trader is already halted")

        account = self.broker.get_account()
        reason = self.check_guards(account)
        if reason:
            self.halt(reason)

        clock = self.broker.get_clock()
        if self.cfg.require_market_open and not clock.is_open:
            log.info("market is closed; skipping this cycle")
            return {"skipped": True, "reason": "market closed",
                    "equity": account.equity, "orders": []}

        intents = self.plan(account)
        orders = self.execute(intents, account)

        summary = {
            "skipped": False,
            "equity": account.equity,
            "cash": account.cash,
            "intents": [asdict(i) | {"delta": i.delta} for i in intents],
            "orders": orders,
        }
        self._record("cycle", equity=account.equity,
                     positions={i.symbol: i.current_shares for i in intents},
                     targets={i.symbol: i.target_shares for i in intents})
        return summary

    def run_forever(self, interval_seconds: int = 60, max_cycles: int | None = None):
        """Poll on a fixed interval until halted, interrupted, or out of cycles."""
        mode = "DRY RUN" if self.cfg.dry_run else ("PAPER" if self.broker.is_paper else "LIVE")
        log.info("starting %s loop on %s every %ss -- %s",
                 mode, ", ".join(self.cfg.symbols), interval_seconds, self.strategy)

        cycles = 0
        try:
            while max_cycles is None or cycles < max_cycles:
                try:
                    self.run_once()
                except Halted:
                    break
                except Exception as exc:  # noqa: BLE001 -- a bad cycle must not kill the loop
                    log.exception("cycle failed: %s", exc)
                    self._record("cycle_error", error=str(exc))
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            log.info("interrupted; leaving positions untouched")
        return cycles
