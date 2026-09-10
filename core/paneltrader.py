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

Same safety rails as the single-symbol loop, because the ways to lose money by
accident do not change with the number of instruments: fill at the next open,
never act while an order is still working, respect buying power, and stop on a
daily loss.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

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
        self._session_start_equity: float | None = None
        self._session_day: date | None = None

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

        # sells first: they free the buying power the buys need
        ordered = sorted(intents, key=lambda i: i.delta)
        for intent in ordered:
            qty = abs(intent.delta)
            if qty < 1:
                continue
            side = "buy" if intent.delta > 0 else "sell"

            if side == "buy":
                needed = qty * intent.price
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
        today = date.today()
        if self._session_day != today:
            self._session_day = today
            self._session_start_equity = account.equity

        if self._session_start_equity and self._session_start_equity > 0:
            change = account.equity / self._session_start_equity - 1.0
            if change <= -abs(self.cfg.max_daily_loss_pct) / 100.0:
                raise Halted(f"daily loss {change:.2%} hit the "
                             f"{self.cfg.max_daily_loss_pct:.2f}% limit")

        if account.blocked:
            raise Halted("the broker has this account blocked from trading")

    def run_once(self) -> dict:
        """One full cycle. Returns everything that happened, for display."""
        account = self.broker.get_account()
        self.check_rails(account)

        if self.cfg.require_market_open:
            clock = self.broker.get_clock()
            if not clock.is_open:
                return {"status": "market closed", "intents": [], "results": []}

        base, final, overlay_report = self.target_book()
        intents = self.plan(account, final)
        results = self.execute(intents, account)

        return {
            "status": "dry run" if self.cfg.dry_run else "executed",
            "equity": account.equity,
            "buying_power": account.buying_power,
            "base": base, "final": final,
            "overlay": overlay_report,
            "intents": intents, "results": results,
        }
