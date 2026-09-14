"""Whether the app can tell that a background loop is really running.

This exists because of a specific failure: the page said "closes
automatically at 15:55 — nothing is held overnight" while nothing ran between
button presses, and a position went five minutes past its deadline and then
overnight. Every test here is a way the page could be fooled into saying that
again.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from core import heartbeat as hb


def _at(root, seconds_ago: float, pid: int | None = None):
    """Write a heartbeat as though it happened `seconds_ago`."""
    row = {"pid": os.getpid() if pid is None else pid,
           "at": (datetime.now(timezone.utc)
                  - timedelta(seconds=seconds_ago)).isoformat()}
    p = hb.path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(row), encoding="utf-8")


# -- the basic signal -----------------------------------------------------

def test_a_fresh_beat_from_a_live_process_is_alive(tmp_path):
    hb.beat(tmp_path)
    assert hb.alive(tmp_path)
    assert 0 <= hb.age_seconds(tmp_path) < 5


def test_no_heartbeat_at_all_is_not_running(tmp_path):
    assert not hb.alive(tmp_path)
    assert hb.age_seconds(tmp_path) is None
    assert "press Run one cycle" in hb.describe(tmp_path)


def test_a_stale_beat_is_not_running_however_alive_the_pid(tmp_path):
    """A loop that hung an hour ago is not watching your position, even
    though the process it left behind is still there."""
    _at(tmp_path, hb.STALE_AFTER_SECONDS + 30)
    assert not hb.alive(tmp_path)


def test_a_fresh_beat_from_a_dead_process_is_not_running(tmp_path):
    """The case freshness alone cannot catch. A file written two seconds
    before the process was killed still looks fresh two seconds later, and on
    Windows it outlives the process indefinitely."""
    _at(tmp_path, 1.0, pid=999_999_998)
    assert hb.age_seconds(tmp_path) < hb.STALE_AFTER_SECONDS
    assert not hb.alive(tmp_path)


def test_a_slow_cycle_does_not_read_as_a_dead_loop(tmp_path):
    """One slow broker call must not flip the page to 'not running'."""
    _at(tmp_path, 45.0)
    assert hb.alive(tmp_path)


# -- shutting down --------------------------------------------------------

def test_clearing_says_stopped_immediately_rather_than_waiting_to_go_stale(tmp_path):
    hb.beat(tmp_path)
    hb.clear(tmp_path)
    assert not hb.alive(tmp_path)
    assert "press Run one cycle" in hb.describe(tmp_path)


def test_a_loop_that_died_without_clearing_says_it_stopped(tmp_path):
    """Different sentence from 'never started': one of them means something
    went wrong."""
    _at(tmp_path, hb.STALE_AFTER_SECONDS + 30)
    assert "it stopped" in hb.describe(tmp_path)


# -- refusing to guess ----------------------------------------------------

def test_a_corrupt_heartbeat_is_not_running(tmp_path):
    """Never resolve doubt in favour of 'it is handled'."""
    p = hb.path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert hb.read(tmp_path) == {}
    assert not hb.alive(tmp_path)


def test_an_unparseable_timestamp_is_not_running(tmp_path):
    p = hb.path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pid": os.getpid(), "at": "whenever"}),
                 encoding="utf-8")
    assert hb.age_seconds(tmp_path) is None
    assert not hb.alive(tmp_path)


def test_a_missing_pid_is_not_running(tmp_path):
    p = hb.path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat()}),
                 encoding="utf-8")
    assert not hb.alive(tmp_path)


def test_this_process_counts_as_alive():
    """Pins the pid check itself: if os.kill(pid, 0) stopped working the
    whole signal would silently read as 'nothing is running'."""
    assert hb.pid_alive(os.getpid())
    assert not hb.pid_alive(999_999_998)
    assert not hb.pid_alive(0)


# -- several copies -------------------------------------------------------

def test_named_instances_have_their_own_heartbeat(tmp_path):
    """One copy running must not make another copy look alive."""
    hb.beat(tmp_path, name="spy desk")
    assert hb.alive(tmp_path, name="spy desk")
    assert not hb.alive(tmp_path, name="nvda")
    assert not hb.alive(tmp_path)


def test_an_instance_name_is_made_filename_safe(tmp_path):
    hb.beat(tmp_path, name="SPY / QQQ!")
    assert "/" not in hb.path(tmp_path, name="SPY / QQQ!").name
    assert hb.alive(tmp_path, name="SPY / QQQ!")


# -- what it carries ------------------------------------------------------

def test_extra_fields_ride_along_for_the_page(tmp_path):
    hb.beat(tmp_path, cycles=12, running="Fair value gap")
    row = hb.read(tmp_path)
    assert row["cycles"] == 12
    assert row["running"] == "Fair value gap"


def test_no_temp_file_is_left_behind(tmp_path):
    hb.beat(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_a_long_gone_pid_is_dead_whatever_the_platform_calls_it():
    """POSIX raises ProcessLookupError for a pid that is gone; Windows raises
    `OSError: [WinError 87]`. Pinned on the answer rather than the exception,
    so it keeps meaning the same thing on either.

    Deliberately not a just-exited child: Windows keeps such a pid openable
    for a while and reports it alive, which is why `alive()` does not rely on
    this check alone. See the note in `pid_alive`.
    """
    assert not hb.pid_alive(999_999_998)
    assert not hb.pid_alive(0)
    assert not hb.pid_alive(-1)
    assert hb.pid_alive(os.getpid())
