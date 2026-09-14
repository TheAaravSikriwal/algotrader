"""Which version of the code the running app is actually serving.

The desktop launcher runs straight out of the repo, so there is no copy to
keep in sync -- but "it should be up to date" and "it is up to date" are
different claims, and only one of them is checkable. This makes it checkable.

It also catches the specific way this app can lie about itself: Streamlit
caches imported modules, so a server started before a code change keeps
serving the old `core/` while rendering the new pages. The process start time
sits next to the commit time for exactly that reason -- if the commit is newer
than the process, the app is running code it has not loaded.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Version:
    commit: str = ""
    committed_at: datetime | None = None
    subject: str = ""
    dirty: bool = False
    started_at: datetime | None = None

    @property
    def known(self) -> bool:
        return bool(self.commit)

    @property
    def stale(self) -> bool:
        """True when the code on disk is newer than the running process.

        Means: restart. The pages you are looking at may be newer than the
        modules behind them.
        """
        if not (self.committed_at and self.started_at):
            return False
        return self.committed_at > self.started_at

    def label(self) -> str:
        if not self.known:
            return "not a git checkout"
        bits = [self.commit]
        if self.dirty:
            bits.append("+ uncommitted changes")
        if self.committed_at:
            bits.append(self.committed_at.astimezone().strftime("%d %b %H:%M"))
        return "  ·  ".join(bits)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(("git", *args), cwd=REPO, capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _process_start() -> datetime | None:
    """When this Python process began, which is when modules were imported."""
    try:
        import os

        import psutil  # noqa: F401  -- optional
        return datetime.fromtimestamp(
            psutil.Process(os.getpid()).create_time(), tz=timezone.utc)
    except Exception:
        # No psutil: fall back to when this module was first imported, which
        # is close enough for the staleness check and needs no dependency.
        return _IMPORTED_AT


_IMPORTED_AT = datetime.now(timezone.utc)


def current() -> Version:
    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        return Version(started_at=_IMPORTED_AT)

    when = _git("log", "-1", "--format=%cI")
    committed_at = None
    if when:
        try:
            committed_at = datetime.fromisoformat(when)
        except ValueError:
            committed_at = None

    return Version(
        commit=commit,
        committed_at=committed_at,
        subject=_git("log", "-1", "--format=%s"),
        dirty=bool(_git("status", "--porcelain")),
        started_at=_process_start(),
    )
