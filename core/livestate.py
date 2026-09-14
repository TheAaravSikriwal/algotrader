"""The bridge between the running app and Claude.

The app writes what it is doing to `live/state.json`. Claude reads that file,
decides whether to step in, and writes back to `live/instructions.json`. The
app picks the instruction up on its next cycle.

Two files rather than one, because the direction of travel matters: the app
owns its state and never reads instructions into it, and Claude owns
instructions and never edits state. Nothing can get into a fight over the same
file.

Three properties this has to have, and each exists because of a way a shared
file goes wrong:

  * **Atomic writes.** Written to a temporary file and renamed, so a reader
    that arrives mid-write sees the old file rather than half of the new one.
  * **Instructions expire.** An instruction with no expiry is an instruction
    that fires days later against a position that no longer exists. Every one
    carries a deadline and is ignored past it.
  * **Instructions are consumed once.** Acting on the same "close the
    position" twice means selling something you no longer hold.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

LIVE = Path(__file__).resolve().parent.parent / "live"
STATE = LIVE / "state.json"
INSTRUCTIONS = LIVE / "instructions.json"
DECISIONS = LIVE / "decisions.jsonl"

#: An instruction older than this is ignored even if it was never consumed.
DEFAULT_TTL_MINUTES = 30

ACTIONS = {
    "hold",          # leave everything alone
    "switch",        # run a different algorithm from now on
    "close",         # flatten the named symbol, or everything
    "pause",         # stop opening new positions, keep managing open ones
    "resume",        # undo a pause
    "size",          # change the risk per trade
}


def _slug(name: str) -> str:
    """A filename-safe instance name. Empty stays empty, meaning the default."""
    keep = "".join(c if (c.isalnum() or c in "-_") else "-"
                   for c in str(name).strip().lower())
    return keep.strip("-")[:32]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, payload: Any):
    """Write via a temp file and rename, so no reader ever sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
        os.replace(tmp, path)          # atomic on Windows and POSIX
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class Instruction:
    """Something Claude wants the loop to do."""
    action: str
    reason: str = ""
    strategy: str = ""                 # for "switch"
    symbol: str = ""                   # for "close"; blank means everything
    risk_frac: float | None = None     # for "size"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    issued_at: str = field(default_factory=_utc)
    expires_at: str = ""
    consumed_at: str = ""

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}; "
                             f"expected one of {sorted(ACTIONS)}")
        if not self.expires_at:
            self.expires_at = (
                datetime.now(timezone.utc)
                + timedelta(minutes=DEFAULT_TTL_MINUTES)).isoformat()

    @property
    def consumed(self) -> bool:
        return bool(self.consumed_at)

    @property
    def expired(self) -> bool:
        try:
            return datetime.now(timezone.utc) > datetime.fromisoformat(self.expires_at)
        except ValueError:
            return True                # unparseable means do not act on it

    @property
    def live(self) -> bool:
        return not self.consumed and not self.expired


class Bridge:
    """Read and write the shared files."""

    def __init__(self, root: Path | None = None, name: str = ""):
        """`name` separates one running copy from another.

        Several instances can trade different symbols on the same account, so
        each writes its own state and decision files. They share the broker,
        which is the single source of truth for positions -- an instance that
        published its own idea of the position could disagree with the account
        and there would be no way to tell which was right.
        """
        self.root = Path(root or LIVE)
        self.name = _slug(name)
        suffix = f"-{self.name}" if self.name else ""
        self.state_path = self.root / f"state{suffix}.json"
        self.instructions_path = self.root / f"instructions{suffix}.json"
        self.decisions_path = self.root / f"decisions{suffix}.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def all_instances(cls, root: Path | None = None) -> list["Bridge"]:
        """Every instance that has published, newest first."""
        root = Path(root or LIVE)
        if not root.exists():
            return []
        out = []
        for f in sorted(root.glob("state*.json")):
            stem = f.stem                      # "state" or "state-nvda"
            out.append(cls(root, stem[6:] if stem.startswith("state-") else ""))
        return sorted(out, key=lambda b: b.state_path.stat().st_mtime,
                      reverse=True)

    @property
    def label(self) -> str:
        return self.name or "main"

    # -- app -> Claude ----------------------------------------------------

    def publish(self, state: dict) -> dict:
        """Write what the loop is doing right now."""
        payload = {"written_at": _utc(), **state}
        _write_atomic(self.state_path, payload)
        return payload

    def state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def state_age_seconds(self) -> float | None:
        """How long since the loop last published. None when it never has.

        A stale state file means the app is not running, and an instruction
        written against it would be acting on a picture of the past.
        """
        s = self.state()
        if not s.get("written_at"):
            return None
        try:
            then = datetime.fromisoformat(s["written_at"])
        except ValueError:
            return None
        return (datetime.now(timezone.utc) - then).total_seconds()

    # -- Claude -> app ----------------------------------------------------

    def instruct(self, instruction: Instruction) -> Instruction:
        """Queue an instruction for the loop."""
        rows = self._read_instructions()
        rows.append(asdict(instruction))
        _write_atomic(self.instructions_path, rows)
        return instruction

    def _read_instructions(self) -> list[dict]:
        if not self.instructions_path.exists():
            return []
        try:
            raw = json.loads(self.instructions_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return raw if isinstance(raw, list) else []

    def instructions(self, live_only: bool = False) -> list[Instruction]:
        out = []
        for row in self._read_instructions():
            try:
                out.append(Instruction(**row))
            except (TypeError, ValueError):
                continue               # a row from an older shape, skipped
        return [i for i in out if i.live] if live_only else out

    def pending(self) -> Instruction | None:
        """The oldest instruction still worth acting on."""
        live = self.instructions(live_only=True)
        return live[0] if live else None

    def consume(self, instruction_id: str) -> bool:
        """Mark one as acted on, so it cannot fire twice."""
        rows, hit = self._read_instructions(), False
        for row in rows:
            if row.get("id") == instruction_id and not row.get("consumed_at"):
                row["consumed_at"] = _utc()
                hit = True
        if hit:
            _write_atomic(self.instructions_path, rows)
        return hit

    # -- the audit trail --------------------------------------------------

    def record(self, event: str, **fields) -> dict:
        """Append to the decision log.

        Every choice the loop makes goes here, including the ones where it
        decided to do nothing. A log of only the actions taken cannot answer
        "why didn't it trade?", which is the question that actually comes up.
        """
        row = {"ts": _utc(), "event": event, **fields}
        self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with self.decisions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        return row

    def decisions(self, limit: int = 200) -> list[dict]:
        if not self.decisions_path.exists():
            return []
        rows = []
        with self.decisions_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue           # a half-written line from a kill
        return rows[-limit:]
