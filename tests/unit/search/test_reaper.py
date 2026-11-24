"""
Tests for search/reaper.py on in-memory Optuna studies. `now` is passed in: for heartbeat cases a fixed UTC time,
for cases without a heartbeat the real time shifted against Optuna's start time (the asking host's local clock).
"""
import json
import logging
import os
import socket
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import optuna
import pytest
from optuna.distributions import FloatDistribution
from optuna.exceptions import UpdateFinishedTrialError
from optuna.trial import TrialState

from search.layout import trial_folder
from search.reaper import reap

optuna.logging.set_verbosity(optuna.logging.WARNING)

NOW = datetime(2026, 11, 24, 12, 0, 0, tzinfo=timezone.utc)
SPACE = {"x": FloatDistribution(0.0, 1.0)}


def _running(study, params, user_attrs=None):
    """A RUNNING trial with exactly `params`, as a worker's ask() of an enqueued trial gives it."""
    study.enqueue_trial(params, user_attrs=user_attrs)
    return study.ask(SPACE)


def _beat(study_dir, number, minutes_ago):
    folder = trial_folder(study_dir, number)
    folder.mkdir(parents=True, exist_ok=True)
    time = (NOW - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")
    (folder / "heartbeat").write_text(json.dumps({"host": "gpu1", "pid": 1, "time": time}))


def _states(study):
    return [trial.state for trial in study.get_trials()]


def _next_trial(study):
    """What the next worker gets: the parameters and user attrs of the next ask()."""
    trial = study.ask(SPACE)
    return trial.params, trial.user_attrs


@pytest.fixture
def study():
    return optuna.create_study(direction="maximize")


def test_a_stale_heartbeat_fails_the_trial_and_enqueues_its_parameters_once(study, tmp_path):
    _running(study, {"x": 0.3})
    _beat(tmp_path, 0, minutes_ago=16)

    reaped = reap(study, tmp_path, NOW)

    assert (reaped, _states(study)) == ([0], [TrialState.FAIL, TrialState.WAITING])
    assert json.loads((trial_folder(tmp_path, 0) / "reaped").read_text()) == {
        "host": socket.gethostname(), "pid": os.getpid(), "time": "2026-11-24T12:00:00+00:00"}
    assert _next_trial(study) == ({"x": 0.3}, {"retry_of": 0})


# | heartbeat age | stale? | why                              |
# |---------------|--------|----------------------------------|
# | 14 min        | no     | inside the 15-minute grace       |
# | 15 min        | no     | stale means older than the grace |
@pytest.mark.parametrize("minutes_ago", [14, 15])
def test_a_trial_with_a_heartbeat_inside_the_grace_is_left_alone(study, tmp_path, minutes_ago):
    _running(study, {"x": 0.3})
    _beat(tmp_path, 0, minutes_ago)

    reaped = reap(study, tmp_path, NOW)

    assert (reaped, _states(study), (trial_folder(tmp_path, 0) / "reaped").exists()) == (
        [], [TrialState.RUNNING], False)


# the worker died between ask() and its first beat: with or without having made the folder
@pytest.mark.parametrize("with_folder", [True, False], ids=["empty-folder", "no-folder"])
def test_a_trial_without_heartbeat_that_started_long_ago_is_stale(study, tmp_path, with_folder):
    _running(study, {"x": 0.3})
    if with_folder:
        trial_folder(tmp_path, 0).mkdir()

    reaped = reap(study, tmp_path, datetime.now(timezone.utc) + timedelta(minutes=16))

    # the claim's time is the real clock here, so only its existence is fixed
    assert (reaped, _states(study), (trial_folder(tmp_path, 0) / "reaped").exists()) == (
        [0], [TrialState.FAIL, TrialState.WAITING], True)
    assert _next_trial(study) == ({"x": 0.3}, {"retry_of": 0})


def test_a_trial_without_heartbeat_that_just_started_is_left_alone(study, tmp_path):
    _running(study, {"x": 0.3})

    reaped = reap(study, tmp_path, datetime.now(timezone.utc) + timedelta(minutes=1))

    assert (reaped, _states(study)) == ([], [TrialState.RUNNING])


def test_a_trial_another_reaper_claimed_is_skipped(study, tmp_path):
    # the other reaper told FAIL and enqueued already, or is about to
    _running(study, {"x": 0.3})
    _beat(tmp_path, 0, minutes_ago=16)
    (trial_folder(tmp_path, 0) / "reaped").write_text("{}")

    reaped = reap(study, tmp_path, NOW)

    assert (reaped, _states(study)) == ([], [TrialState.RUNNING])


def test_reaping_twice_enqueues_one_retry(study, tmp_path):
    _running(study, {"x": 0.3})
    _beat(tmp_path, 0, minutes_ago=16)

    reaped = (reap(study, tmp_path, NOW), reap(study, tmp_path, NOW))

    assert (reaped, _states(study)) == (([0], []), [TrialState.FAIL, TrialState.WAITING])


def test_finished_trials_are_never_reaped(study, tmp_path):
    # their heartbeat stopped because they ended, not because the worker died
    study.tell(_running(study, {"x": 0.3}), 0.5)
    study.tell(_running(study, {"x": 0.4}), state=TrialState.FAIL)
    _beat(tmp_path, 0, minutes_ago=600)
    _beat(tmp_path, 1, minutes_ago=600)

    reaped = reap(study, tmp_path, NOW)

    assert (reaped, _states(study)) == ([], [TrialState.COMPLETE, TrialState.FAIL])


def _chain(study, length):
    """Trials 0..length-1 of one parameter set, each a retry of the one before; all FAIL except the last."""
    trial = _running(study, {"x": 0.3})
    for number in range(1, length):
        study.tell(trial, state=TrialState.FAIL)
        trial = _running(study, {"x": 0.3}, {"retry_of": number - 1})
    return trial.number


def test_a_first_retry_that_goes_stale_is_retried_again(study, tmp_path):
    last = _chain(study, 2)  # 0 <- 1
    _beat(tmp_path, last, minutes_ago=16)

    reap(study, tmp_path, NOW)

    assert _next_trial(study) == ({"x": 0.3}, {"retry_of": 1})


def test_a_parameter_set_gets_at_most_two_retries(study, tmp_path, caplog):
    last = _chain(study, 3)  # 0 <- 1 <- 2: trial 2 is the second retry
    _beat(tmp_path, last, minutes_ago=16)

    with caplog.at_level(logging.WARNING, logger="search.reaper"):
        reaped = reap(study, tmp_path, NOW)

    assert (reaped, _states(study)) == ([2], [TrialState.FAIL] * 3)
    assert [record.getMessage() for record in caplog.records] == [
        "trial 2 is stale: told FAIL", "trial 2 is not retried: its parameters had 2 retries"]


def test_a_trial_that_died_before_drawing_its_enqueued_parameters_is_retried_with_them(study, tmp_path):
    # before suggest_* the values sit only in Optuna's fixed_params; params is still empty
    study.enqueue_trial({"x": 0.3})
    study.ask()
    _beat(tmp_path, 0, minutes_ago=16)

    reap(study, tmp_path, NOW)

    assert _next_trial(study) == ({"x": 0.3}, {"retry_of": 0})


def test_a_trial_that_died_before_any_parameter_was_drawn_is_not_retried(study, tmp_path, caplog):
    # a sampled trial without values: a "retry" would be a fresh draw under a false label
    study.ask()
    _beat(tmp_path, 0, minutes_ago=16)

    with caplog.at_level(logging.WARNING, logger="search.reaper"):
        reaped = reap(study, tmp_path, NOW)

    assert (reaped, _states(study)) == ([0], [TrialState.FAIL])
    assert [record.getMessage() for record in caplog.records] == [
        "trial 0 is stale: told FAIL", "trial 0 is not retried: it died before any parameter was drawn"]


def test_a_trial_its_worker_completes_while_being_reaped_is_not_retried(study, tmp_path, monkeypatch):
    # the beats did not arrive for 15 min, but the worker lived and tells COMPLETE between our read and our tell
    trial = _running(study, {"x": 0.3})
    _beat(tmp_path, 0, minutes_ago=16)
    real_tell = study.tell

    def worker_completes_first(*args, **kwargs):
        real_tell(trial, 0.5)
        return real_tell(*args, **kwargs)
    monkeypatch.setattr(study, "tell", worker_completes_first)

    reaped = reap(study, tmp_path, NOW)

    assert (reaped, _states(study)) == ([], [TrialState.COMPLETE])


def test_a_tell_that_finds_the_trial_finished_does_not_stop_the_reaper(study, tmp_path, monkeypatch):
    # the journal replays our FAIL after the worker's COMPLETE: the reaping worker must go on to its ask()
    _running(study, {"x": 0.3})
    _beat(tmp_path, 0, minutes_ago=16)
    monkeypatch.setattr(study, "tell", Mock(side_effect=UpdateFinishedTrialError("finished")))

    assert reap(study, tmp_path, NOW) == []
    assert _states(study) == [TrialState.RUNNING]


def test_a_now_without_time_zone_is_rejected(study, tmp_path):
    # it would be read as local time without a heartbeat and fail only with one
    _running(study, {"x": 0.3})

    with pytest.raises(ValueError, match="timezone-aware"):
        reap(study, tmp_path, datetime(2026, 11, 24, 12, 0, 0))
