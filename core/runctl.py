"""Start and stop the background loop from the app.

`autorun.py` works, but it needs a terminal, and the whole point of the page
is not needing one. This starts it as a detached process so it survives the
page reloading -- Streamlit reruns the whole script on every interaction, and
a child tied to that would die on the next click.

Two things it has to refuse to do, because both are worse than not starting:

  * **Never two at once.** Two loops on one account both see the same setup
    and both act on it, so you get double the position you sized for. The
    heartbeat is the interlock.
  * **Never pretend.** If the process dies a second after launching, `start`
    has to say so rather than returning cheerfully and leaving the page
    claiming something is watching.

Output goes to a log file rather than a pipe. A pipe nobody reads fills its
buffer and blocks the writer, which would freeze the loop mid-session -- the
one process that must not stop.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from core import heartbeat as hb

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "autorun.py"
LOGS = REPO / "logs"

#: How long to wait for the first heartbeat before calling the launch failed.
#: The first cycle talks to the broker, so this is not instant.
START_TIMEOUT = 25.0


def log_path(name: str = "") -> Path:
    slug = "".join(c if (c.isalnum() or c in "-_") else "-"
                   for c in str(name).strip().lower()).strip("-")[:32]
    return LOGS / (f"autorun-{slug}.log" if slug else "autorun.log")


#: How the loop is launched on Windows, decided by measurement rather than by
#: what the flag names suggest.
#:
#: CREATE_NO_WINDOW gives the child no console window, which is the point --
#: a black window flashing up on every start is not acceptable. It also
#: survives the app closing: on Windows a child is independent of its parent
#: unless a Job object says otherwise, and nothing here uses one.
#:
#: DETACHED_PROCESS looks like the more thorough choice and is a trap. It
#: works when the parent has a normal console and kills the child with
#: 0xC000013A (STATUS_CONTROL_C_EXIT) when the parent has none -- under a test
#: runner, for instance. Measured both ways:
#:
#:     flags                  plain script     under pytest
#:     DETACHED_PROCESS       alive            dies 0xC000013A
#:     CREATE_NO_WINDOW       alive            alive
#:
#: The two are also mutually exclusive, and combining them kills the child
#: outright in every context. So: no-window, never detached.
_CREATE_NO_WINDOW = 0x08000000


def _detached_flags() -> dict:
    """Keep the loop alive when the app goes away, and off the screen."""
    if os.name == "nt":
        # The new process group stops a Ctrl-C in the parent's console from
        # reaching the loop, which is the one signal that should not.
        return {"creationflags": (subprocess.CREATE_NEW_PROCESS_GROUP
                                  | _CREATE_NO_WINDOW)}
    return {"start_new_session": True}


def running(name: str = "") -> bool:
    return hb.alive(name=name)


def start(name: str = "", symbols=("SPY", "QQQ"), risk_pct: float = 0.5,
          every: float = 30.0, dry_run: bool = False,
          timeout: float = START_TIMEOUT) -> tuple[bool, str]:
    """Launch the loop. Returns (started, what to tell the user).

    Waits for the first heartbeat before reporting success, so "started" means
    the process reached its loop rather than merely that Popen returned.
    """
    if running(name):
        return False, "It is already running — nothing to do."
    if not SCRIPT.exists():
        return False, f"Cannot find {SCRIPT.name}."

    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCRIPT),
           "--every", str(every),
           "--symbols", ",".join(symbols),
           "--risk", str(risk_pct)]
    if name:
        cmd += ["--name", name]
    if dry_run:
        cmd.append("--dry-run")

    # The child imports `heartbeat` fresh, so it has to be told where to
    # write. Passing it explicitly also keeps a test's temporary directory
    # from being contradicted by the real one.
    env = {**os.environ, "ALGOTRADER_LIVE_DIR": str(hb.LIVE)}

    log = log_path(name)
    try:
        # A pipe nobody drains fills up and blocks the writer. The one process
        # that must never stall is this one.
        with log.open("a", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=fh,
                                    stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env,
                                    **_detached_flags())
    except OSError as exc:
        return False, f"Could not start it: {exc}"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if running(name):
            return True, f"Running. Output goes to {log.name}."
        if proc.poll() is not None:
            tail = _tail(log)
            return False, (f"It exited immediately (code {proc.returncode}). "
                           + (f"Last line: {tail}" if tail else
                              f"See {log.name}."))
        time.sleep(0.4)

    return False, (f"Started, but it has not checked in after {timeout:.0f}s. "
                   f"See {log.name}.")


def stop(name: str = "", timeout: float = 12.0) -> tuple[bool, str]:
    """Ask the loop to stop, and confirm it actually did."""
    row = hb.read(name=name)
    pid = int(row.get("pid") or 0)
    if not running(name):
        hb.clear(name=name)
        return True, "It was not running."
    if not pid:
        return False, "No process id recorded — stop it from its terminal."
    if pid == os.getpid():
        # A heartbeat naming this very process means something wrote it that
        # is not the loop. Killing ourselves here would take the app down
        # with it, and it hung the test run once before this guard existed.
        hb.clear(name=name)
        return False, ("That heartbeat names this process, not a background "
                       "loop. Cleared it rather than stopping anything.")

    try:
        os.kill(pid, 9 if os.name == "nt" else 15)
    except PermissionError:
        return False, f"Not allowed to stop process {pid}."
    except OSError as exc:
        return False, f"Could not stop it: {exc}"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not hb.pid_alive(pid):
            # A killed process never reaches its own cleanup, so the page
            # would keep reading a fresh-looking heartbeat until it goes
            # stale. Clear it here instead.
            hb.clear(name=name)
            return True, "Stopped. Nothing is watching your positions now."
        time.sleep(0.3)
    return False, f"Asked process {pid} to stop, but it is still going."


def _tail(path: Path, lines: int = 1) -> str:
    try:
        rows = [r.strip() for r in
                path.read_text(encoding="utf-8", errors="replace").splitlines()
                if r.strip()]
    except OSError:
        return ""
    return " / ".join(rows[-lines:])


def recent_output(name: str = "", lines: int = 12) -> list[str]:
    """The last few lines, for showing on the page."""
    try:
        rows = log_path(name).read_text(encoding="utf-8",
                                        errors="replace").splitlines()
    except OSError:
        return []
    return [r for r in rows if r.strip()][-lines:]
