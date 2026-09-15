"""Starting and stopping the background loop from the page.

Two failures matter more than the rest, and each has a test here:

  * Two loops on one account both act on the same setup, so you end up with
    double the position you sized for.
  * A `start` that returns cheerfully when the process has already died leaves
    the page claiming something is watching when nothing is -- which is the
    exact bug the heartbeat was built to end.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

from core import heartbeat as hb
from core import runctl


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Keep heartbeats and logs out of the real live/ and logs/ folders."""
    monkeypatch.setattr(hb, "LIVE", tmp_path / "live")
    monkeypatch.setattr(runctl, "LOGS", tmp_path / "logs")
    return tmp_path


# -- refusing to start a second one ---------------------------------------

def test_it_will_not_start_a_second_loop(tmp_path):
    """Two loops both see the same setup and both act, so you get double the
    position you sized for. The heartbeat is the interlock."""
    hb.beat(name="")                       # pretend one is already going
    started, msg = runctl.start()
    assert not started
    assert "already running" in msg


def test_a_named_copy_can_start_while_another_runs(tmp_path, monkeypatch):
    """Separate instances are the supported way to trade different symbols,
    so the interlock must be per-instance, not global."""
    hb.beat(name="")
    assert runctl.running("") is True
    assert runctl.running("nvda") is False


def test_running_is_false_when_nothing_has_ever_run(tmp_path):
    assert not runctl.running()


# -- reporting honestly ---------------------------------------------------

def test_a_process_that_dies_immediately_is_reported_as_failed(tmp_path, monkeypatch):
    """The failure mode the heartbeat exists to prevent: start() returning
    success while nothing is actually running."""
    monkeypatch.setattr(runctl, "SCRIPT", tmp_path / "boom.py")
    runctl.SCRIPT.write_text("import sys; sys.exit(3)", encoding="utf-8")
    started, msg = runctl.start(timeout=8.0)
    assert not started
    assert "exited immediately" in msg
    assert "3" in msg
    assert not runctl.running()


def test_a_missing_script_is_reported_rather_than_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(runctl, "SCRIPT", tmp_path / "not_here.py")
    started, msg = runctl.start()
    assert not started
    assert "Cannot find" in msg


def test_a_process_that_never_checks_in_is_not_called_running(tmp_path, monkeypatch):
    """Alive is not the same as working. A loop that starts and then hangs
    before its first beat must not light the page up green."""
    monkeypatch.setattr(runctl, "SCRIPT", tmp_path / "sleeper.py")
    runctl.SCRIPT.write_text("import time; time.sleep(30)", encoding="utf-8")
    started, msg = runctl.start(timeout=3.0)
    assert not started
    assert "not checked in" in msg
    runctl.stop()


# -- the round trip -------------------------------------------------------

def test_starting_and_stopping_a_real_process(tmp_path, monkeypatch):
    """End to end against a real detached process, because every interesting
    part of this -- detaching, the heartbeat, the pid -- is only exercised for
    real. The child reads ALGOTRADER_LIVE_DIR, which is the only way to reach
    a process that imports the module fresh."""
    script = tmp_path / "beater.py"
    script.write_text("\n".join([
        "import sys, time",
        f"sys.path.insert(0, {str(runctl.REPO)!r})",
        "from core import heartbeat as hb",
        "while True:",
        "    hb.beat()",
        "    time.sleep(0.5)",
    ]), encoding="utf-8")
    monkeypatch.setattr(runctl, "SCRIPT", script)

    started, msg = runctl.start(timeout=20.0)
    assert started, msg
    assert runctl.running()
    assert hb.read()["pid"], "the heartbeat has to carry a pid to be stoppable"

    stopped, msg = runctl.stop()
    assert stopped, msg
    assert not runctl.running()


def test_stopping_something_that_is_not_running_is_not_an_error(tmp_path):
    stopped, msg = runctl.stop()
    assert stopped
    assert "was not running" in msg


def test_stop_refuses_to_kill_the_process_asking(tmp_path):
    """`hb.beat()` records whoever called it. A heartbeat naming this very
    process means something other than the loop wrote it, and killing it
    would take the app down -- it hung a whole test run once."""
    hb.beat(name="")                       # records this pytest process
    stopped, msg = runctl.stop()
    assert not stopped
    assert "names this process" in msg
    assert not runctl.running(), "the bogus heartbeat is cleared"


# -- the log --------------------------------------------------------------

def test_output_goes_to_a_file_not_a_pipe(tmp_path, monkeypatch):
    """A pipe nobody drains fills its buffer and blocks the writer, which
    would freeze the one process that must not stop."""
    monkeypatch.setattr(runctl, "SCRIPT", tmp_path / "chatty.py")
    runctl.SCRIPT.write_text("print('hello from the loop')", encoding="utf-8")
    runctl.start(timeout=5.0)
    assert runctl.log_path().exists()
    assert any("hello from the loop" in r for r in runctl.recent_output())


def test_no_log_yet_reads_as_empty_rather_than_an_error(tmp_path):
    assert runctl.recent_output() == []


def test_each_instance_writes_its_own_log(tmp_path):
    assert runctl.log_path("nvda") != runctl.log_path()
    assert "nvda" in runctl.log_path("nvda").name


def test_an_instance_name_is_made_filename_safe(tmp_path):
    assert "/" not in runctl.log_path("SPY / QQQ!").name


# -- how it is launched ---------------------------------------------------

def test_it_launches_detached_so_it_survives_the_page_reloading(monkeypatch):
    """Streamlit reruns the whole script on every interaction. A child tied to
    that process would die on the next click."""
    flags = runctl._detached_flags()
    if os.name == "nt":
        assert flags["creationflags"] & runctl._DETACHED_PROCESS
    else:
        assert flags["start_new_session"] is True


def test_it_does_not_put_a_console_window_on_the_screen():
    """python.exe is a console application, so DETACHED_PROCESS alone makes
    Windows open a *new* console for it -- a black window flashing up every
    time the loop starts, and staying there while it runs."""
    if os.name != "nt":
        return
    assert runctl._detached_flags()["creationflags"] & runctl._CREATE_NO_WINDOW


def test_the_command_carries_the_settings_the_page_is_showing(tmp_path, monkeypatch):
    seen = {}

    class FakeProc:
        returncode = None

        def poll(self):
            return 0

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(runctl.subprocess, "Popen", fake_popen)
    runctl.start(name="nvda", symbols=("NVDA", "AMD"), risk_pct=1.25,
                 every=45.0, dry_run=True, timeout=1.0)
    cmd = seen["cmd"]
    assert cmd[0] == sys.executable
    assert "--symbols" in cmd and "NVDA,AMD" in cmd
    assert "--risk" in cmd and "1.25" in cmd
    assert "--every" in cmd and "45.0" in cmd
    assert "--name" in cmd and "nvda" in cmd
    assert "--dry-run" in cmd


def test_a_real_run_does_not_get_the_dry_run_flag(tmp_path, monkeypatch):
    seen = {}

    class FakeProc:
        returncode = None

        def poll(self):
            return 0

    monkeypatch.setattr(runctl.subprocess, "Popen",
                        lambda cmd, **kw: (seen.update(cmd=cmd), FakeProc())[1])
    runctl.start(dry_run=False, timeout=1.0)
    assert "--dry-run" not in seen["cmd"]
