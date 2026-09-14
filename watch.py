"""The console Claude works through: read the loop's state, leave instructions.

The app publishes what it is doing to `live/state.json` each cycle and reads
`live/instructions.json` before deciding anything. This is the other end of
that pipe.

    python watch.py                          what the loop is doing now
    python watch.py --follow                 same, refreshing
    python watch.py log                      the decision trail
    python watch.py pause   --why "..."      stop opening new positions
    python watch.py resume  --why "..."
    python watch.py switch  "Keltner breakout" --why "..."
    python watch.py close   --why "..."      flatten everything
    python watch.py size    0.25 --why "..." risk per trade, in percent

Every instruction carries a reason and an expiry. The reason is not decoration
-- it is what you read three days later when you are working out why the loop
did something odd, and an instruction with no expiry fires long after the
situation that justified it has gone.

Nothing here places an order. It writes a file; the running app decides
whether to act on it, and still refuses anything its rails forbid.
"""
from __future__ import annotations

import argparse
import sys
import time as clock
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from core.livestate import ACTIONS, Bridge, Instruction  # noqa: E402

STALE_SECONDS = 180


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    return f"{seconds / 60:.0f}m ago"


def show(bridge: Bridge) -> int:
    s = bridge.state()
    if not s:
        print("The loop has not published anything. Open the app and switch "
              "to Auto.")
        return 1

    age = bridge.state_age_seconds()
    stale = age is not None and age > STALE_SECONDS

    print(f"  as of        {_fmt_age(age)}"
          + ("   << STALE: the app may not be running" if stale else ""))
    print(f"  mode         {s.get('mode', '?')}"
          + ("  (PAUSED)" if s.get("paused") else ""))
    print(f"  running      {s.get('running') or 'nothing'}")
    if s.get("chose_because"):
        print(f"  because      {s['chose_because'][:100]}")
    if s.get("switched_from"):
        print(f"  switched     from {s['switched_from']}")
    print(f"  headline     {s.get('headline', '')}")

    cand = s.get("candidate") or {}
    if cand:
        print(f"  its edge     {cand.get('expectancy_bps', 0):+.2f} bps/trade"
              f"   t={cand.get('t_stat', 0):+.2f}"
              f"   {int(cand.get('trades', 0))} trades"
              f"   {'usable' if cand.get('credible') else 'TOO FEW'}")

    sess = s.get("session") or {}
    print(f"  session      {sess.get('open', '?')}-{sess.get('close', '?')}"
          f"   flatten {sess.get('flatten_at', '?')}"
          f"   {'open' if sess.get('is_open') else 'closed'}")
    print(f"  equity       ${s.get('equity', 0):,.2f}")

    for sym, p in (s.get("positions") or {}).items():
        print(f"  position     {p.get('qty')} {sym} @ "
              f"{p.get('avg_price', 0):.2f}   "
              f"unrealised ${p.get('unrealized_pl', 0):+,.2f}")
    if not s.get("positions"):
        print("  position     flat")

    for b in s.get("blocks") or []:
        print(f"  BLOCKED      {b}")
    if s.get("instruction_applied"):
        print(f"  last instr   {s['instruction_applied']}")

    live = bridge.instructions(live_only=True)
    if live:
        print(f"  QUEUED       {len(live)} instruction(s) not yet picked up:")
        for i in live:
            print(f"                 {i.action} — {i.reason[:60]}")
    return 0


def log(bridge: Bridge, limit: int) -> int:
    rows = bridge.decisions(limit=limit)
    if not rows:
        print("Nothing logged yet.")
        return 1
    for r in rows:
        when = str(r.get("ts", ""))[11:19]
        what = r.get("headline") or r.get("result") or ""
        print(f"  {when}  {r.get('event', ''):<14} {str(what)[:88]}")
    return 0


def instruct(bridge: Bridge, args) -> int:
    age = bridge.state_age_seconds()
    if age is None:
        print("Warning: the loop has never published. The instruction will "
              "wait, and expires in 30 minutes.")
    elif age > STALE_SECONDS:
        print(f"Warning: last published {_fmt_age(age)} — the app may not be "
              f"running. Writing it anyway; it expires in 30 minutes.")

    kwargs = {"action": args.action, "reason": args.why}
    if args.action == "switch":
        kwargs["strategy"] = args.value
    elif args.action == "size":
        try:
            pct = float(args.value)
        except (TypeError, ValueError):
            print("size needs a number, in percent (e.g. 0.25)")
            return 2
        kwargs["risk_frac"] = pct / 100.0
    elif args.action == "close" and args.value:
        kwargs["symbol"] = args.value

    if args.minutes:
        from datetime import timedelta
        kwargs["expires_at"] = (datetime.now(timezone.utc)
                                + timedelta(minutes=args.minutes)).isoformat()

    ins = bridge.instruct(Instruction(**kwargs))
    print(f"queued {ins.action} ({ins.id})")
    print(f"  reason  {ins.reason or '(none given)'}")
    print(f"  expires {ins.expires_at[11:19]} UTC")
    print("The loop picks it up on its next cycle.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="status",
                    choices=sorted(ACTIONS | {"status", "log"}))
    ap.add_argument("value", nargs="?", default="",
                    help="strategy name for switch, percent for size, "
                         "symbol for close")
    ap.add_argument("--why", default="", help="why you are doing this")
    ap.add_argument("--minutes", type=int, default=0,
                    help="expire after this many minutes (default 30)")
    ap.add_argument("--follow", action="store_true",
                    help="keep refreshing the status")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    bridge = Bridge()

    if args.action == "log":
        return log(bridge, args.limit)
    if args.action == "status":
        if not args.follow:
            return show(bridge)
        try:
            while True:
                print("\033[2J\033[H", end="")     # clear, home
                print(f"  {datetime.now():%H:%M:%S}  watching live/state.json"
                      f"   (ctrl-c to stop)\n")
                show(bridge)
                clock.sleep(15)
        except KeyboardInterrupt:
            return 0

    if not args.why:
        print("An instruction needs --why. You will want it later, and the "
              "loop records it against whatever it does.")
        return 2
    return instruct(bridge, args)


if __name__ == "__main__":
    sys.exit(main())
