"""Event-driven backtest engine.

The one rule that keeps results honest: a signal computed from bar *t* is
filled at the **open of bar t+1**. Nothing in here can see a price it would not
have had at decision time.

Optional stop-loss / take-profit levels are checked intrabar against the bar's
high and low. When a bar's range covers both levels the stop is assumed to hit
first -- the pessimistic reading, since a single bar tells you nothing about
the path the price took inside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    position_size: float = 1.0        # fraction of equity committed per trade
    slippage_bps: float = 5.0         # each side, in basis points of price
    commission_pct: float = 0.0       # Alpaca US equities: 0
    commission_per_share: float = 0.0
    allow_short: bool = False
    stop_loss_pct: float = 0.0        # 0 disables; 5.0 means exit 5% against entry
    take_profit_pct: float = 0.0      # 0 disables


@dataclass
class BacktestResult:
    equity: pd.Series
    position: pd.Series               # shares held, per bar
    exposure: pd.Series               # target position (-1..1) actually applied
    trades: pd.DataFrame
    price: pd.Series
    config: BacktestConfig
    benchmark: pd.Series = field(default_factory=pd.Series)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    @property
    def drawdown(self) -> pd.Series:
        return self.equity / self.equity.cummax() - 1.0


TRADE_COLUMNS = [
    "entry_time", "exit_time", "direction", "qty", "entry_price", "exit_price",
    "fees", "pnl", "return_pct", "bars_held", "exit_reason",
]


def run_backtest(df: pd.DataFrame, signals: pd.Series,
                 config: BacktestConfig | None = None) -> BacktestResult:
    """Simulate `signals` over `df` and return equity, positions and trades."""
    cfg = config or BacktestConfig()
    if df.empty:
        raise ValueError("no bars to backtest")

    signals = signals.reindex(df.index).fillna(0.0).astype(float).clip(-1.0, 1.0)
    if not cfg.allow_short:
        signals = signals.clip(lower=0.0)

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    times = df.index
    targets = signals.to_numpy(dtype=float)
    n = len(df)

    slip = cfg.slippage_bps / 10_000.0
    stop_frac = cfg.stop_loss_pct / 100.0
    tp_frac = cfg.take_profit_pct / 100.0

    cash = cfg.initial_cash
    shares = 0.0
    desired = 0.0        # target set at the previous close, filled at this open
    applied = 0.0        # target currently expressed in `shares`
    blocked_side = 0.0   # after a stop-out, don't re-enter this side until the signal resets

    equity_curve = np.empty(n)
    position_curve = np.zeros(n)
    exposure_curve = np.zeros(n)
    trades: list[dict] = []
    open_trade: dict | None = None

    def fees_for(price: float, qty: float) -> float:
        return abs(qty) * cfg.commission_per_share + abs(qty * price) * cfg.commission_pct

    def close_trade(exit_px: float, exit_idx: int, exit_fees: float, reason: str):
        nonlocal open_trade
        t = open_trade
        gross = (exit_px - t["entry_price"]) * t["qty"] * t["sign"]
        total_fees = t["fees"] + exit_fees
        pnl = gross - total_fees
        basis = abs(t["entry_price"] * t["qty"])
        trades.append({
            "entry_time": t["entry_time"], "exit_time": times[exit_idx],
            "direction": "long" if t["sign"] > 0 else "short",
            "qty": t["qty"], "entry_price": t["entry_price"], "exit_price": exit_px,
            "fees": total_fees, "pnl": pnl,
            "return_pct": pnl / basis * 100 if basis else 0.0,
            "bars_held": exit_idx - t["entry_index"], "exit_reason": reason,
        })
        open_trade = None

    for i in range(n):
        px_open = opens[i]

        # 1 -- fill the previous bar's decision at this bar's open ------------
        if blocked_side != 0.0 and (desired == 0.0 or np.sign(desired) != blocked_side):
            blocked_side = 0.0           # signal reset -- lockout lifted
        want = 0.0 if (blocked_side != 0.0 and np.sign(desired) == blocked_side) else desired

        if want != applied and px_open > 0:
            equity_now = cash + shares * px_open
            target_shares = want * cfg.position_size * equity_now / px_open
            delta = target_shares - shares

            if abs(delta) > 1e-12:
                exec_px = px_open * (1 + slip) if delta > 0 else px_open * (1 - slip)

                if open_trade is not None and (want == 0 or np.sign(want) != open_trade["sign"]):
                    close_trade(exec_px, i, fees_for(exec_px, open_trade["qty"]), "signal")

                cash -= delta * exec_px + fees_for(exec_px, delta)
                shares += delta

                if open_trade is None and abs(shares) > 1e-12:
                    open_trade = {
                        "entry_time": times[i], "entry_index": i, "entry_price": exec_px,
                        "qty": abs(shares), "sign": float(np.sign(shares)),
                        "fees": fees_for(exec_px, shares),
                    }
                applied = want

        # 2 -- stop-loss / take-profit, checked against this bar's range ------
        if open_trade is not None and (stop_frac > 0 or tp_frac > 0):
            entry = open_trade["entry_price"]
            long = open_trade["sign"] > 0
            stop_px = entry * (1 - stop_frac) if long else entry * (1 + stop_frac)
            tp_px = entry * (1 + tp_frac) if long else entry * (1 - tp_frac)

            hit_px, reason = None, ""
            if stop_frac > 0 and ((long and lows[i] <= stop_px) or (not long and highs[i] >= stop_px)):
                hit_px, reason = stop_px, "stop"
            elif tp_frac > 0 and ((long and highs[i] >= tp_px) or (not long and lows[i] <= tp_px)):
                hit_px, reason = tp_px, "target"

            if hit_px is not None:
                exec_px = hit_px * (1 - slip) if long else hit_px * (1 + slip)
                qty = shares
                cash += qty * exec_px - fees_for(exec_px, qty)
                close_trade(exec_px, i, fees_for(exec_px, qty), reason)
                shares = 0.0
                applied = 0.0
                blocked_side = 1.0 if long else -1.0

        equity_curve[i] = cash + shares * closes[i]
        position_curve[i] = shares
        exposure_curve[i] = applied

        # 3 -- decide what to hold on the next bar ---------------------------
        desired = targets[i]

    if open_trade is not None:                # mark the open trade to the last close
        close_trade(closes[-1], n - 1, 0.0, "end of data")

    return BacktestResult(
        equity=pd.Series(equity_curve, index=df.index, name="equity"),
        position=pd.Series(position_curve, index=df.index, name="shares"),
        exposure=pd.Series(exposure_curve, index=df.index, name="exposure"),
        trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
        price=df["close"].copy(),
        config=cfg,
        benchmark=(cfg.initial_cash * df["close"] / df["close"].iloc[0]).rename("buy_and_hold"),
    )
