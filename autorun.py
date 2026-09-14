"""The background loop. Auto, actually automatic.

Until this existed, "Auto" meant one cycle per button press. A short sat five
minutes past its own 15:55 deadline and then overnight, while the page said
"closes automatically at 15:55 — nothing is held overnight". Nothing was
running to do it.

    python autorun.py                 run until the session ends
    python autorun.py --dry-run       decide everything, send nothing
    python autorun.py --once          a single cycle, then stop
    python autorun.py --name nvda     a second copy, its own state and log

What it does, every `--every` seconds:

  1. Beat, so the app can tell it is alive.
  2. Run one cycle -- obey any instruction, choose a rule, act, publish.
  3. Stop when the session is over and nothing is left open.

Three things it must get right, because each one is a way to lose money
quietly:

  * **The deadline is not optional.** Past the flatten time it keeps running
    until the position is actually gone, even though it opens nothing new.
    Exiting "because the window closed" is what left a position overnight.
  * **A crash must not look like silence.** The heartbeat is cleared on the
    way out, whether that is a clean stop, Ctrl-C, or an exception, so the
    page says "stopped" rather than going quietly stale.
  * **It never widens what is permitted.** Every rail lives in `DayTrader`
    and is enforced there. This only decides *when* to ask.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from core.env import load_env  # noqa: E402

load_env()

from brokers.alpaca import AlpacaBroker  # noqa: E402
from core import heartbeat as hb  # noqa: E402
from core.autotrader import AutoConfig, AutoTrader  # noqa: E402
from core.broker import BrokerError  # noqa: E402
from core.livestate import Bridge  # noqa: E402
from core.marketclock import CalendarError, MarketCalendar  # noqa: E402

_stop = False


def _ask_to_stop(*_):
    """Ctrl-C once means finish this cycle and shut down tidily."""
    global _stop
    _stop = True
    print("\nstopping after this cycle...")


def still_needed(broker, calendar, now: datetime) -> tuple[bool, str]:
    """Whether there is any reason to keep looping.

    Deliberately not "is the window open". The window closes at 15:30 and the
    flatten happens at 15:55; a loop that stopped at the window would leave
    the position for the night, which is the bug this file exists to fix.
    """
    try:
        session = calendar.session(now)
    except CalendarError:
        return False, "not a trading day"

    try:
        open_positions = bool(broker.get_positions())
    except BrokerError:
        open_positions = True          # cannot tell: assume there is work

    if now >= session.close:
        return (True, "market closed but something is still open") \
            if open_positions else (False, "session over, flat")
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=30.0,
                    help="seconds between cycles (default 30)")
    ap.add_argument("--once", action="store_true", help="one cycle, then stop")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide everything, send nothing")
    ap.add_argument("--name", default="", help="name this copy")
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--risk", type=float, default=0.5,
                    help="percent of equity per trade")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _ask_to_stop)

    broker = AlpacaBroker(paper=True)
    account = broker.get_account()
    if not account.is_paper:
        print("Refusing: this is not a paper account.")
        return 1

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    bridge = Bridge(name=args.name)
    calendar = MarketCalendar.load()
    auto = AutoTrader(broker, AutoConfig(symbols=symbols,
                                         risk_frac=args.risk / 100.0,
                                         max_open_positions=len(symbols)),
                      calendar, bridge)

    mode = "DRY RUN - nothing will be sent" if args.dry_run else "live on paper"
    print(f"autorun  {mode}")
    print(f"         {', '.join(symbols)}  |  every {args.every:g}s  "
          f"|  instance {bridge.label}")
    print(f"         equity ${account.equity:,.2f}")
    print("         Ctrl-C to stop\n")

    cycles = 0
    try:
        while True:
            now = datetime.now()
            keep, why = still_needed(broker, calendar, now)
            if not keep:
                print(f"{now:%H:%M:%S}  done - {why}")
                break
            if why:
                print(f"{now:%H:%M:%S}  {why}")

            hb.beat(name=args.name, cycles=cycles, running=auto._running,
                    dry_run=args.dry_run)
            try:
                c = auto.cycle(now=now, execute=not args.dry_run)
                cycles += 1
                if not c.throttled:
                    print(f"{now:%H:%M:%S}  {c.headline()}")
            except Exception as exc:                      # noqa: BLE001
                # One bad cycle -- a dropped connection, a malformed bar --
                # must not end the loop and leave a position unwatched.
                print(f"{now:%H:%M:%S}  cycle failed: {exc}")
                bridge.record("error", where="autorun", error=str(exc),
                              trace=traceback.format_exc()[-800:])

            if args.once or _stop:
                break
            time.sleep(max(1.0, args.every))
    finally:
        # Whatever happened -- clean stop, Ctrl-C, or a crash on the way out
        # -- the page must stop claiming anything is being watched.
        hb.clear(name=args.name)
        print(f"\nstopped after {cycles} cycle(s). "
              f"Nothing is watching your positions now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
