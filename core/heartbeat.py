"""Whether a background loop is actually alive.

The app promised "closes automatically at 15:55 — nothing is held overnight"
while nothing at all ran between button presses. A position sat five minutes
past its own deadline and then overnight, and the page kept saying it would
be handled.

The fix is not a nicer sentence. The page has to be able to *tell*, so a
runner writes a heartbeat and everything else reads it.

Freshness alone is not enough. A file written two seconds before a process
was killed still looks fresh two seconds later, and on Windows a stale file
outlives the process that wrote it indefinitely. So the heartbeat carries the
pid, and `alive()` checks the process is really there before believing the
timestamp.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LIVE = Path(__file__).resolve().parent.parent / "live"

#: A heartbeat older than this means the loop is gone, whatever the pid says.
#: Generous next to a 30s cycle, because one slow broker call must not read
#: as a dead process.
STALE_AFTER_SECONDS = 150.0


def path(root: Path | None = None, name: str = "") -> Path:
    slug = "".join(c if (c.isalnum() or c in "-_") else "-"
                   for c in str(name).strip().lower()).strip("-")[:32]
    return Path(root or LIVE) / (f"runner-{slug}.json" if slug else "runner.json")


def beat(root: Path | None = None, name: str = "", **fields) -> dict:
    """Say the loop is still here. Called once per cycle."""
    p = path(root, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"pid": os.getpid(), "at": datetime.now(timezone.utc).isoformat(),
           **fields}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return row


def read(root: Path | None = None, name: str = "") -> dict:
    p = path(root, name)
    if not p.exists():
        return {}
    try:
        row = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return row if isinstance(row, dict) else {}


def clear(root: Path | None = None, name: str = "") -> None:
    """Called on a clean shutdown, so the page knows immediately."""
    path(root, name).unlink(missing_ok=True)


def age_seconds(root: Path | None = None, name: str = "") -> float | None:
    row = read(root, name)
    if not row.get("at"):
        return None
    try:
        then = datetime.fromisoformat(row["at"])
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - then).total_seconds()


def pid_alive(pid: int) -> bool:
    """Whether that process still exists.

    `os.kill(pid, 0)` is the portable check. A pid we are not allowed to
    signal is still a running process, so PermissionError counts as alive.

    The exception differs by platform, which is why the catch is broad.
    POSIX raises ProcessLookupError for a pid that is gone; Windows raises
    `OSError: [WinError 87] The parameter is incorrect`, so on Windows the
    ProcessLookupError branch never fires and the generic OSError catch does
    the work.

    **It is not instant on Windows.** A process that exited seconds ago stays
    openable for a while and reports as alive here. Measured, not assumed:
    a child that had already exited still answered True. That is why `alive()`
    pairs this with a staleness window, and why a clean shutdown calls
    `clear()` rather than relying on the pid going away -- the pid check
    catches a loop that died a while back, the timestamp catches one that
    hung, and `clear()` makes an orderly stop show up immediately.
    """
    if not pid or pid < 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


def alive(root: Path | None = None, name: str = "") -> bool:
    """Is a runner really going right now?

    Both tests have to pass. A fresh timestamp from a process that has since
    been killed is exactly the case that let the app keep promising an
    automatic close while nothing was running.
    """
    age = age_seconds(root, name)
    if age is None or age > STALE_AFTER_SECONDS:
        return False
    return pid_alive(read(root, name).get("pid", 0))


def describe(root: Path | None = None, name: str = "") -> str:
    """One line for the page. Never optimistic."""
    if alive(root, name):
        age = age_seconds(root, name) or 0.0
        return f"running — last check {age:.0f}s ago"
    if read(root, name):
        return "not running — it stopped, and nothing is watching now"
    return "not running — nothing happens unless you press Run one cycle"
