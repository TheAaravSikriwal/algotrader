"""The shared file between the app and Claude.

A file two processes share goes wrong in specific ways, and each test here is
one of them: a reader arriving mid-write, an instruction firing twice, an
instruction firing days later against a position that no longer exists, and a
half-written line from a killed process.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from core.livestate import Bridge, Instruction


@pytest.fixture
def bridge(tmp_path):
    return Bridge(tmp_path)


# -- instructions ---------------------------------------------------------

def test_an_unknown_action_is_rejected_at_construction():
    """A typo that reached the file would be acted on by something."""
    with pytest.raises(ValueError, match="unknown action"):
        Instruction(action="sel1 everything")


def test_an_instruction_expires_by_default():
    """One with no deadline fires days later against a position that is gone."""
    ins = Instruction(action="hold")
    assert ins.expires_at
    assert not ins.expired


def test_an_expired_instruction_is_not_live(bridge):
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    bridge.instruct(Instruction(action="close", expires_at=past))
    assert bridge.pending() is None
    assert len(bridge.instructions()) == 1, "it is kept, just not acted on"


def test_an_unparseable_expiry_counts_as_expired():
    """Better to ignore an instruction than to act on a broken one."""
    assert Instruction(action="hold", expires_at="whenever").expired


def test_an_instruction_is_consumed_once(bridge):
    """Acting on 'close' twice means selling something you no longer hold."""
    ins = bridge.instruct(Instruction(action="close", reason="test"))
    assert bridge.pending().id == ins.id
    assert bridge.consume(ins.id) is True
    assert bridge.pending() is None
    assert bridge.consume(ins.id) is False, "consuming again changes nothing"


def test_the_oldest_live_instruction_comes_first(bridge):
    bridge.instruct(Instruction(action="pause", reason="first"))
    bridge.instruct(Instruction(action="resume", reason="second"))
    assert bridge.pending().reason == "first"


def test_a_malformed_row_is_skipped_not_fatal(bridge):
    bridge.instruct(Instruction(action="hold"))
    rows = json.loads(bridge.instructions_path.read_text(encoding="utf-8"))
    rows.append({"action": "hold", "unexpected_field": 1})
    bridge.instructions_path.write_text(json.dumps(rows), encoding="utf-8")
    assert len(bridge.instructions()) == 1


def test_no_instructions_file_is_empty_not_an_error(bridge):
    assert bridge.instructions() == []
    assert bridge.pending() is None


# -- state ----------------------------------------------------------------

def test_state_round_trips(bridge):
    bridge.publish({"running": "RSI mean reversion", "equity": 5000.0})
    s = bridge.state()
    assert s["running"] == "RSI mean reversion"
    assert s["written_at"]


def test_state_age_says_how_stale_the_picture_is(bridge):
    """An instruction written against a stale state acts on the past."""
    assert bridge.state_age_seconds() is None
    bridge.publish({"running": "x"})
    assert 0 <= bridge.state_age_seconds() < 5


def test_a_corrupt_state_file_reads_as_empty(bridge):
    bridge.state_path.write_text("{not json", encoding="utf-8")
    assert bridge.state() == {}
    assert bridge.state_age_seconds() is None


def test_writes_are_atomic(bridge, monkeypatch):
    """A reader arriving mid-write must see the old file, never half of one.

    Simulated by making the serialisation blow up after the temp file exists:
    the real file must be untouched, and no stray temp files left behind.
    """
    bridge.publish({"running": "original"})

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(json, "dump", boom)
    with pytest.raises(RuntimeError):
        bridge.publish({"running": "replacement"})

    assert bridge.state()["running"] == "original"
    assert not list(bridge.root.glob("*.tmp")), "temp file left behind"


# -- the decision log -----------------------------------------------------

def test_decisions_record_doing_nothing_too(bridge):
    """A log of only the actions taken cannot answer 'why didn't it trade?'."""
    bridge.record("cycle", headline="Watching — no setup", sent=0)
    bridge.record("cycle", headline="Placed 1 order(s)", sent=1)
    rows = bridge.decisions()
    assert len(rows) == 2
    assert rows[0]["headline"] == "Watching — no setup"
    assert all("ts" in r for r in rows)


def test_a_truncated_decision_line_is_skipped(bridge):
    bridge.record("cycle", headline="fine")
    with bridge.decisions_path.open("a", encoding="utf-8") as fh:
        fh.write('{"ts": "2026-09-14", "event": "cy')
    assert len(bridge.decisions()) == 1


def test_decisions_are_limited_to_the_tail(bridge):
    for i in range(50):
        bridge.record("cycle", n=i)
    rows = bridge.decisions(limit=10)
    assert len(rows) == 10
    assert rows[-1]["n"] == 49


def test_no_decision_log_is_empty_not_an_error(bridge):
    assert bridge.decisions() == []


# -- several running copies ----------------------------------------------

def test_named_instances_write_to_separate_files(tmp_path):
    """Two copies trading different symbols must not overwrite each other."""
    a = Bridge(tmp_path, name="SPY desk")
    b = Bridge(tmp_path, name="nvda")
    a.publish({"running": "Keltner breakout"})
    b.publish({"running": "Day of week"})
    assert a.state()["running"] == "Keltner breakout"
    assert b.state()["running"] == "Day of week"
    assert a.state_path != b.state_path


def test_an_instance_name_is_made_filename_safe(tmp_path):
    bridge = Bridge(tmp_path, name="SPY / QQQ  desk!")
    bridge.publish({"running": "x"})
    assert bridge.state_path.exists()
    assert "/" not in bridge.state_path.name


def test_the_unnamed_instance_keeps_the_plain_filename(tmp_path):
    """Existing installs keep working without being renamed."""
    assert Bridge(tmp_path).state_path.name == "state.json"
    assert Bridge(tmp_path).label == "main"


def test_all_instances_finds_every_one_that_published(tmp_path):
    Bridge(tmp_path).publish({"running": "a"})
    Bridge(tmp_path, name="two").publish({"running": "b"})
    Bridge(tmp_path, name="three").publish({"running": "c"})
    found = {b.label for b in Bridge.all_instances(tmp_path)}
    assert found == {"main", "two", "three"}


def test_instructions_are_addressed_to_one_instance(tmp_path):
    """Pausing one copy must not pause the others."""
    a, b = Bridge(tmp_path, name="a"), Bridge(tmp_path, name="b")
    a.instruct(Instruction(action="pause", reason="only a"))
    assert a.pending() is not None
    assert b.pending() is None


def test_no_instances_at_all_is_empty_not_an_error(tmp_path):
    assert Bridge.all_instances(tmp_path / "nothing_here") == []
