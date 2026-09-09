"""Preflight check -- tells you exactly what works before you rely on it.

    python check_setup.py

Reads .env, connects to Alpaca paper, and confirms account access, market data,
news access and the trading loop. Places no orders and never touches your live
account: it only reports whether live credentials are present.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.env import load_env

load_env()

import os  # noqa: E402

OK, WARN, BAD, INFO = "  OK  ", " WARN ", " FAIL ", " ---  "
results: list[tuple[str, str]] = []


def record(status: str, message: str):
    results.append((status, message))
    print(f"[{status}] {message}", flush=True)


def section(title: str):
    print(f"\n{title}\n{'-' * len(title)}")


def check_dependencies():
    section("Dependencies")
    for module, label in [("pandas", "pandas"), ("numpy", "numpy"),
                          ("plotly", "plotly"), ("streamlit", "streamlit"),
                          ("alpaca", "alpaca-py"), ("yfinance", "yfinance")]:
        try:
            __import__(module)
            record(OK, label)
        except ImportError:
            record(BAD, f"{label} is missing -- pip install -r requirements.txt")

    try:
        __import__("transformers")
        record(OK, "transformers (FinBERT scoring available)")
    except ImportError:
        record(INFO, "transformers not installed -- lexicon scorer will be used")


def check_env():
    section("Credentials")
    if not Path(".env").exists():
        record(WARN, ".env not found -- copy .env.example to .env")

    paper_key = os.getenv("ALPACA_API_KEY_ID")
    paper_secret = os.getenv("ALPACA_API_SECRET_KEY")
    if paper_key and paper_secret:
        record(OK, f"paper keys present ({paper_key[:6]}...)")
    else:
        record(BAD, "paper keys missing -- set ALPACA_API_KEY_ID and "
                    "ALPACA_API_SECRET_KEY in .env")

    live_key = os.getenv("ALPACA_LIVE_API_KEY_ID")
    if live_key and os.getenv("ALPACA_LIVE_API_SECRET_KEY"):
        if live_key == paper_key:
            record(BAD, "live key is identical to the paper key -- one is pasted "
                        "in the wrong variable")
        else:
            record(WARN, "LIVE keys are set. Real money is reachable with "
                         "--live --i-understand-this-is-real-money")
    else:
        record(OK, "live keys not set -- live mode cannot start (good for now)")

    return bool(paper_key and paper_secret)


def check_broker():
    section("Alpaca paper account")
    from brokers.alpaca import AlpacaBroker
    from core.broker import BrokerError

    try:
        broker = AlpacaBroker(paper=True)
    except BrokerError as exc:
        record(BAD, f"could not connect: {exc}")
        return None

    try:
        account = broker.get_account()
    except Exception as exc:  # noqa: BLE001
        record(BAD, f"credentials rejected: {exc}")
        record(INFO, "Check you copied the PAPER keys, not the live ones.")
        return None

    record(OK, f"connected -- equity ${account.equity:,.2f}, "
               f"buying power ${account.buying_power:,.2f}")
    if account.blocked:
        record(BAD, "this account is blocked from trading")
    if account.equity <= 0:
        record(WARN, "paper account has no equity -- reset it in the dashboard")

    try:
        clock = broker.get_clock()
        state = "open" if clock.is_open else "closed"
        record(OK, f"market clock reachable -- market is {state}")
    except Exception as exc:  # noqa: BLE001
        record(WARN, f"clock unavailable: {exc}")

    try:
        positions = broker.get_positions()
        record(OK, f"{len(positions)} open position(s)" if positions
                   else "no open positions")
    except Exception as exc:  # noqa: BLE001
        record(WARN, f"positions unavailable: {exc}")

    try:
        orders = broker.get_open_orders()
        record(OK, f"{len(orders)} working order(s)")
    except Exception as exc:  # noqa: BLE001
        record(WARN, f"open orders unavailable: {exc}")

    return broker


def check_market_data(broker):
    section("Market data")
    from core.data import load_bars

    try:
        df = load_bars("SPY", date.today() - timedelta(days=120), date.today(),
                       "1Day", "yfinance")
        record(OK, f"yfinance -- {len(df)} SPY bars to {df.index[-1]:%Y-%m-%d}")
    except Exception as exc:  # noqa: BLE001
        record(WARN, f"yfinance failed: {exc}")

    if broker is None:
        return
    try:
        bars = broker.get_bars("SPY", "1Day", 30)
        record(OK, f"Alpaca bars -- {len(bars)} SPY bars to "
                   f"{bars.index[-1]:%Y-%m-%d}")
    except Exception as exc:  # noqa: BLE001
        record(WARN, f"Alpaca bars failed ({exc})")
        record(INFO, "The free plan serves IEX data only; that is normal.")


def check_news():
    section("News")
    from core.news import NewsError, load_news, to_frame
    from core.sentiment import LexiconScorer, score_frame

    try:
        items = load_news(["AAPL"], date.today() - timedelta(days=14), date.today())
    except NewsError as exc:
        record(BAD, f"news unavailable: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        record(BAD, f"news request failed: {exc}")
        return

    if not items:
        record(WARN, "no articles returned for AAPL over the last 14 days")
        return

    record(OK, f"{len(items)} AAPL articles in the last 14 days")
    scored = score_frame(to_frame(items), LexiconScorer())
    if not scored.empty:
        record(OK, f"scored -- mean sentiment {scored['sentiment'].mean():+.3f}")
        top = scored.reindex(scored["sentiment"].abs().sort_values(
            ascending=False).index).iloc[0]
        record(INFO, f"strongest: [{top['sentiment']:+.2f}] "
                     f"{top['headline'][:70]}")


def check_loop(broker):
    section("Trading loop (dry run -- sends nothing)")
    if broker is None:
        record(INFO, "skipped, no broker connection")
        return

    from core.strategy import get_strategy
    from core.trader import LiveConfig, Trader
    import strategies  # noqa: F401

    try:
        trader = Trader(broker, get_strategy("SMA crossover")(fast=20, slow=50),
                        LiveConfig(symbols=["SPY"], dry_run=True,
                                   require_market_open=False,
                                   log_dir="logs/preflight"))
        events = trader.run_once()
        record(OK, f"cycle completed, {len(events)} event(s), nothing sent")
    except Exception as exc:  # noqa: BLE001
        record(BAD, f"loop failed: {type(exc).__name__}: {exc}")


def main() -> int:
    print("Preflight check -- no orders are placed and your live account is "
          "never contacted.")

    check_dependencies()
    have_keys = check_env()

    broker = check_broker() if have_keys else None
    check_market_data(broker)
    if have_keys:
        check_news()
    else:
        section("News")
        record(INFO, "skipped, needs paper keys")
    check_loop(broker)

    failures = sum(1 for status, _ in results if status == BAD)
    warnings = sum(1 for status, _ in results if status == WARN)

    section("Summary")
    if failures:
        print(f"{failures} problem(s) and {warnings} warning(s). "
              "Fix the FAIL lines above first.")
        return 1
    print(f"Everything works. {warnings} warning(s), none blocking.\n")
    print("Next:")
    print("  python news_backtest.py --symbols AAPL,MSFT,NVDA --decay")
    print("  python tournament.py --symbols SPY,QQQ,AAPL --top 5")
    print("  python run_live.py --symbols SPY --strategy \"SMA crossover\" --once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
