"""Alpaca implementation of the Broker interface.

Paper and live are the same API with a different base URL, so this class covers
both. `paper=True` is the default everywhere and you have to go out of your way
to change it.

Paper and live use **separate credentials**, and this class reads them from
separate environment variables. That is deliberate: it makes sending a real
order with paper intent structurally impossible rather than merely discouraged.
If the live variables are unset, live mode cannot start at all.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from core.broker import Account, Broker, BrokerError, Clock, Order, Position
from core.data import COLUMNS, normalise


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class AlpacaBroker(Broker):
    name = "alpaca"

    #: Which market-data feed live reads use. "iex" is real-time and free;
    #: "sip" is the full tape but fifteen minutes behind on the free plan.
    feed: str = "iex"

    def __init__(self, paper: bool = True, key: str | None = None,
                 secret: str | None = None):
        self.is_paper = bool(paper)

        if self.is_paper:
            key = key or os.getenv("ALPACA_API_KEY_ID")
            secret = secret or os.getenv("ALPACA_API_SECRET_KEY")
            if not key or not secret:
                raise BrokerError(
                    "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set. "
                    "Copy .env.example to .env and paste your PAPER keys in. "
                    "Paper keys are free at https://app.alpaca.markets"
                )
        else:
            key = key or os.getenv("ALPACA_LIVE_API_KEY_ID")
            secret = secret or os.getenv("ALPACA_LIVE_API_SECRET_KEY")
            if not key or not secret:
                raise BrokerError(
                    "Live mode needs ALPACA_LIVE_API_KEY_ID / "
                    "ALPACA_LIVE_API_SECRET_KEY, which are not set.\n"
                    "These are DIFFERENT from your paper keys -- generate them "
                    "under the Live section of the Alpaca dashboard.\n"
                    "Keeping them in separate variables is what stops a paper "
                    "run from ever reaching your real account."
                )
            if key == os.getenv("ALPACA_API_KEY_ID"):
                raise BrokerError(
                    "ALPACA_LIVE_API_KEY_ID is the same as your paper key. "
                    "Paper and live issue different credentials -- one of these "
                    "is pasted in the wrong place."
                )

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise BrokerError("alpaca-py is not installed -- pip install alpaca-py") from exc

        self._trading = TradingClient(key, secret, paper=self.is_paper)
        self._data = StockHistoricalDataClient(key, secret)
        self._key, self._secret = key, secret

    # ---- account state --------------------------------------------------
    def get_account(self) -> Account:
        a = self._trading.get_account()
        return Account(
            cash=_f(a.cash),
            equity=_f(a.equity),
            buying_power=_f(a.buying_power),
            currency=getattr(a, "currency", "USD") or "USD",
            is_paper=self.is_paper,
            blocked=bool(getattr(a, "trading_blocked", False)
                         or getattr(a, "account_blocked", False)),
        )

    def get_open_orders(self) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        raw = self._trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return [
            Order(id=str(o.id), symbol=o.symbol.upper(), qty=_f(o.qty),
                  limit_price=_f(getattr(o, "limit_price", None), None) or None,
                  side=str(getattr(o.side, "value", o.side)).lower(),
                  status=str(getattr(o.status, "value", o.status)),
                  filled_qty=_f(getattr(o, "filled_qty", 0)),
                  submitted_at=getattr(o, "submitted_at", None))
            for o in (raw or [])
        ]

    def get_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self._trading.get_all_positions():
            out[p.symbol.upper()] = Position(
                symbol=p.symbol.upper(),
                qty=_f(p.qty),
                avg_price=_f(p.avg_entry_price),
                market_value=_f(p.market_value),
                unrealized_pl=_f(p.unrealized_pl),
            )
        return out

    def get_clock(self) -> Clock:
        c = self._trading.get_clock()
        return Clock(
            timestamp=c.timestamp,
            is_open=bool(c.is_open),
            next_open=getattr(c, "next_open", None),
            next_close=getattr(c, "next_close", None),
        )

    # ---- market data ----------------------------------------------------
    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 300,
                 feed: str | None = None) -> pd.DataFrame:
        """Recent bars. `feed` is "iex" or "sip"; None uses the account default.

        This matters for anything trading live. The free plan delays the SIP
        tape by fifteen minutes, so a loop reading SIP is deciding on prices a
        quarter of an hour old and will refuse to act on them. IEX is
        real-time and free, at the cost of carrying roughly 4% of the volume --
        a different series rather than a faster one, which is why the
        backtests were re-run on it before it was wired in here.
        """
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf = {
            "1Day": TimeFrame.Day,
            "1Hour": TimeFrame.Hour,
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "1Min": TimeFrame.Minute,
        }[timeframe]

        # `limit` alone is not enough: without a start date Alpaca returns only
        # the most recent sliver -- one bar, in practice -- and every caller
        # asking for history silently gets nothing usable. Ask for a window wide
        # enough to contain `limit` bars, then trim.
        span = {"1Day": 1.0, "1Hour": 1 / 6.5, "15Min": 1 / 26,
                "5Min": 1 / 78, "1Min": 1 / 390}[timeframe]
        calendar_days = max(int(limit * span * 1.6) + 7, 7)
        start = datetime.now(timezone.utc) - timedelta(days=calendar_days)

        # No `limit` on the request. Alpaca applies it from `start` *forward*,
        # so pairing a wide window with a limit returns the OLDEST n bars --
        # a strategy would then be reading months-stale prices while looking
        # perfectly healthy. Fetch the window and take the tail instead.
        chosen = (feed or self.feed or "sip").lower()
        bars = self._data.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol.upper(), timeframe=tf, start=start,
            feed=DataFeed.IEX if chosen == "iex" else DataFeed.SIP))
        df = bars.df
        if df is None or df.empty:
            raise BrokerError(f"Alpaca returned no bars for {symbol}")
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol.upper(), level="symbol")

        # One normalisation, shared with core.data. A second copy here is what
        # let the UTC bug survive being fixed there: the backtests read the
        # fixed path while the live loop read this one, so bars arrived stamped
        # 23:55 and a 10:30-15:30 window selected the wrong five hours.
        return normalise(df).tail(limit)

    # ---- orders ---------------------------------------------------------
    def get_quote(self, symbol: str) -> tuple[float, float]:
        """Live bid and ask, for pricing what a position is actually worth.

        Reads the same feed as the bars. On the free plan that has to be IEX:
        recent SIP quotes are refused outright rather than delayed, so asking
        for them raises instead of returning something stale.

        Returns (0.0, 0.0) when no usable quote came back, which callers treat
        as "no valuation" rather than as a price of zero.
        """
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestQuoteRequest

        chosen = (self.feed or "iex").lower()
        try:
            q = self._data.get_stock_latest_quote(StockLatestQuoteRequest(
                symbol_or_symbols=symbol.upper(),
                feed=DataFeed.IEX if chosen == "iex" else DataFeed.SIP,
            ))[symbol.upper()]
        except Exception as exc:                      # noqa: BLE001
            raise BrokerError(f"no quote for {symbol}: {exc}") from exc
        return float(q.bid_price or 0.0), float(q.ask_price or 0.0)

    def submit_order(self, symbol: str, qty: float, side: str,
                     order_type: str = "market", limit_price: float | None = None,
                     time_in_force: str = "day",
                     stop_loss: float | None = None,
                     take_profit: float | None = None) -> Order:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (LimitOrderRequest, MarketOrderRequest,
                                             StopLossRequest, TakeProfitRequest)

        if qty <= 0:
            raise BrokerError(f"order qty must be positive, got {qty}")

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC,
               "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK}[time_in_force.lower()]

        extra = {}
        if stop_loss is not None or take_profit is not None:
            # Alpaca rejects a bracket on anything but DAY or GTC, and rejects
            # exits on the wrong side of the entry, so catch both here rather
            # than reading it back off a 422.
            if time_in_force.lower() not in {"day", "gtc"}:
                raise BrokerError("a bracket order needs time_in_force day or gtc")
            ref = limit_price
            if ref is not None:
                long = side.lower() == "buy"
                if stop_loss is not None and ((long and stop_loss >= ref) or
                                              (not long and stop_loss <= ref)):
                    raise BrokerError(
                        f"stop {stop_loss} is on the wrong side of entry {ref} "
                        f"for a {side}")
                if take_profit is not None and ((long and take_profit <= ref) or
                                                (not long and take_profit >= ref)):
                    raise BrokerError(
                        f"target {take_profit} is on the wrong side of entry "
                        f"{ref} for a {side}")
            extra["order_class"] = OrderClass.BRACKET
            if stop_loss is not None:
                extra["stop_loss"] = StopLossRequest(stop_price=round(stop_loss, 2))
            if take_profit is not None:
                extra["take_profit"] = TakeProfitRequest(limit_price=round(take_profit, 2))

        if order_type.lower() == "limit":
            if limit_price is None:
                raise BrokerError("a limit order needs a limit_price")
            req = LimitOrderRequest(symbol=symbol.upper(), qty=qty, side=side_enum,
                                    time_in_force=tif, limit_price=round(limit_price, 2),
                                    **extra)
        else:
            req = MarketOrderRequest(symbol=symbol.upper(), qty=qty,
                                     side=side_enum, time_in_force=tif, **extra)

        o = self._trading.submit_order(order_data=req)
        return Order(
            id=str(o.id), symbol=o.symbol.upper(), qty=_f(o.qty), side=side.lower(),
            status=str(getattr(o.status, "value", o.status)),
            filled_qty=_f(getattr(o, "filled_qty", 0)),
            filled_price=_f(getattr(o, "filled_avg_price", None), None) or None,
            limit_price=_f(getattr(o, "limit_price", None), None) or None,
            submitted_at=getattr(o, "submitted_at", datetime.now(timezone.utc)),
        )

    def cancel_all_orders(self) -> int:
        responses = self._trading.cancel_orders()
        return len(responses or [])

    def close_position(self, symbol: str) -> Order | None:
        pos = self.get_position(symbol)
        if pos is None or pos.qty == 0:
            return None
        o = self._trading.close_position(symbol.upper())
        return Order(
            id=str(o.id), symbol=symbol.upper(), qty=_f(o.qty),
            side="sell" if pos.qty > 0 else "buy",
            status=str(getattr(o.status, "value", o.status)),
        )
