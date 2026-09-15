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

#: Overridable so a test -- or a second install -- can point a *child
#: process* somewhere else. Monkeypatching the module global cannot reach a
#: process that imports it fresh, and the loop is always a separate process.
LIVE = Path(os.environ.get("ALGOTRADER_LIVE_DIR")
            or Path(__file__).resolve().parent.parent / "live")

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


#: Windows: the least privilege that lets us ask about a process we did not
#: create. `os.kill` asks for PROCESS_ALL_ACCESS, which is both more than is
#: needed and not always granted.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


def _pid_alive_windows(pid: int) -> bool:
    """Open the process and ask whether it has an exit code yet.

    `os.kill(pid, 0)` is not good enough here, for two separate reasons that
    both showed up in practice:

      * It only checks that OpenProcess succeeded. A handle stays valid after
        the process exits, so anything still holding one -- a `Popen` object,
        for instance -- makes a dead process look alive. Measured: a killed
        child opened fine and reported exit code 1.
      * It asks for PROCESS_ALL_ACCESS, far more than is needed to ask a
        question, and a live process we do not own a handle to can come back
        as ERROR_INVALID_PARAMETER -- indistinguishable from "no such pid".
        That made a running loop read as dead, which is the worse direction:
        the page would offer to start a second one.

    GetExitCodeProcess answers directly. The known wart is that a process
    which genuinely exits with code 259 is indistinguishable from a running
    one; 259 is STILL_ACTIVE and nothing can be done about that from here.
    """
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION,
                                  False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """Whether that process still exists.

    Windows and POSIX need genuinely different checks; see
    `_pid_alive_windows` for why `os.kill(pid, 0)` is not usable there.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False

    if os.name == "nt":
        try:
            return _pid_alive_windows(pid)
        except OSError:
            return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                    # running, just not ours to signal
    except OSError:
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
