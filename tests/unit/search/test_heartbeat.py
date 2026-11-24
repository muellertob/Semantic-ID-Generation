"""Tests for search/heartbeat.py with the clock mocked; the beats of a short interval are awaited by polling."""
import itertools
import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from unittest.mock import Mock, call

import pytest

from search import heartbeat, json_files
from search.heartbeat import Heartbeat

T0 = datetime(2026, 11, 24, 10, 0, 0, tzinfo=timezone.utc)


def _beat(second):
    return {"host": socket.gethostname(), "pid": os.getpid(), "time": f"2026-11-24T10:00:{second:02d}+00:00"}


def _wait_until(condition, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.001)


def _heartbeat_threads():
    return [thread for thread in threading.enumerate() if thread.name == "heartbeat"]


@pytest.fixture
def writes(monkeypatch):
    """Records every write of the heartbeat; the clock reads 10:00:00, 10:00:01, ... one second per beat."""
    seconds = itertools.count()
    monkeypatch.setattr(heartbeat, "_now", lambda: T0.replace(second=next(seconds)))
    spy = Mock(wraps=json_files.replace_json)
    monkeypatch.setattr(heartbeat, "replace_json", spy)
    return spy


def test_the_first_beat_is_written_on_entry(tmp_path, writes):
    with Heartbeat(tmp_path / "heartbeat", interval_s=3600):
        assert json.loads((tmp_path / "heartbeat").read_text()) == _beat(0)


def test_the_beats_repeat_every_interval(tmp_path, writes):
    path = tmp_path / "heartbeat"
    with Heartbeat(path, interval_s=0.01):
        _wait_until(lambda: writes.call_count >= 3)

    assert writes.call_args_list[:3] == [call(path, _beat(0)), call(path, _beat(1)), call(path, _beat(2))]


def test_exit_stops_the_thread_at_once(tmp_path, writes):
    with Heartbeat(tmp_path / "heartbeat", interval_s=3600):
        start = time.monotonic()
    took = time.monotonic() - start

    # only the entry beat: the thread was waiting out its first hour when the run ended
    assert (took < 1, _heartbeat_threads(), writes.call_count) == (True, [], 1)


def test_a_failed_write_is_logged_and_the_next_beat_still_comes(tmp_path, writes, caplog):
    path = tmp_path / "heartbeat"
    calls = itertools.count()

    def entry_beat_fails(target, data):
        if next(calls) == 0:
            raise OSError("stale NFS handle")
        return json_files.replace_json(target, data)
    writes.side_effect = entry_beat_fails

    with Heartbeat(path, interval_s=0.01):
        _wait_until(path.exists)

    logged = [(record.levelname, record.getMessage()) for record in caplog.records if record.name == heartbeat.__name__]
    assert (logged, writes.call_args_list[:2]) == (
        [("ERROR", f"heartbeat write failed: {path}")], [call(path, _beat(0)), call(path, _beat(1))])


def test_an_exception_in_the_block_propagates_and_stops_the_thread(tmp_path, writes):
    with pytest.raises(RuntimeError, match="boom"):
        with Heartbeat(tmp_path / "heartbeat", interval_s=3600):
            raise RuntimeError("boom")

    assert _heartbeat_threads() == []
