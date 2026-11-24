"""
The reaper: ends RUNNING trials whose worker died and enqueues their parameters again.

A dead worker never tells Optuna its trial ended, so the trial would stay RUNNING for good, holding a budget slot.
A trial is stale when its heartbeat is older than GRACE, or, without one, when it started more than GRACE ago.
The reaper tells FAIL and enqueues the same parameters as a new trial, at most MAX_RETRIES times per parameter set.

Every worker reaps before it asks for a trial, so reapers can meet: only the one that creates <trial>/reaped acts.
"""
import json
import logging
import os
import socket
from datetime import datetime, timedelta
from pathlib import Path

import optuna
from optuna.exceptions import UpdateFinishedTrialError
from optuna.trial import FrozenTrial, TrialState

from search.json_files import create_json
from search.layout import trial_folder

logger = logging.getLogger(__name__)

GRACE = timedelta(minutes=15)  # heartbeat every 60 s; the clocks of the servers need to agree to minutes only
MAX_RETRIES = 2


def is_stale(folder: str | Path, trial: FrozenTrial, now: datetime) -> bool:
    """`now` is timezone-aware; the heartbeat holds UTC, Optuna's start time is the asking host's local time."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    heartbeat = Path(folder) / "heartbeat"
    if heartbeat.exists():
        last_beat = datetime.fromisoformat(json.loads(heartbeat.read_text())["time"])
        return now - last_beat > GRACE
    # astimezone() reads the naive start as local time
    # the difference is then taken in UTC, across Daylight Saving Time switches too
    return now - trial.datetime_start.astimezone() > GRACE


def reap(study: optuna.Study, study_dir: str | Path, now: datetime) -> list[int]:
    """End the stale RUNNING trials of `study` and enqueue their retries; returns the numbers this call reaped."""
    reaped = []
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        folder = trial_folder(study_dir, trial.number)
        if not is_stale(folder, trial, now):
            continue
        folder.mkdir(parents=True, exist_ok=True)  # a worker that died right after ask() left no folder
        try:
            create_json(folder / "reaped", {"host": socket.gethostname(), "pid": os.getpid(),
                                            "time": now.isoformat(timespec="seconds")})
        except FileExistsError:
            continue  # another reaper has this trial
        try:
            told = study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)
        except UpdateFinishedTrialError:  # its worker finished it between our read and our write
            continue
        if told.state != TrialState.FAIL or "failure" in told.user_attrs:
            continue  # its worker finished it after all (a worker's own FAIL carries its failure class)
        logger.warning("trial %d is stale: told FAIL", trial.number)
        enqueue_retry(study, trial)
        reaped.append(trial.number)
    return reaped


def enqueue_retry(study: optuna.Study, trial: FrozenTrial) -> bool:
    """Enqueue `trial`'s parameters as a new trial with user attr retry_of, unless MAX_RETRIES were used up."""
    # an enqueued trial holds its values in fixed_params until they are suggested; a worker that died before its
    # space was drawn leaves params empty
    params = {**trial.system_attrs.get("fixed_params", {}), **trial.params}
    if not params:
        logger.warning("trial %d is not retried: it died before any parameter was drawn", trial.number)
        return False
    if _retries_before(study, trial) >= MAX_RETRIES:
        logger.warning("trial %d is not retried: its parameters had %d retries", trial.number, MAX_RETRIES)
        return False
    study.enqueue_trial(params, user_attrs={"retry_of": trial.number})
    return True


def _retries_before(study: optuna.Study, trial: FrozenTrial) -> int:
    by_number = {t.number: t for t in study.get_trials(deepcopy=False)}
    retries = 0
    while "retry_of" in trial.user_attrs:
        retries += 1
        trial = by_number[trial.user_attrs["retry_of"]]
    return retries
