"""
Tests for search/runner.py with the stage functions and the clean-up calls mocked (no GPU); the Stage-2 config is
resolved for real from a small base, so the metric is read as in a run.
"""
import itertools
import json
import logging
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, Mock, call

import optuna
import pytest

from generate_semids import CollisionOverflowError
from search import heartbeat, runner, stages
from search.runner import TrialOutcome, run_trial
from search.spec import TrialSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPEC = TrialSpec(dataset="toys", quantizer="rqvae", phase="A", stage1_overrides={"model.beta": 0.25},
                 stage2_overrides={}, stage1_seed=42, stage2_seed=42, semids_from=None, max_steps=100_000,
                 run_test=False, source={"study": "toys.A.rqvae", "trial_number": 7})
SPEC_B = TrialSpec(dataset="toys", quantizer="rqvae", phase="B", stage1_overrides={}, stage2_overrides={},
                   stage1_seed=42, stage2_seed=42, semids_from="/search/toys/A-rqvae/trial_0003", max_steps=100_000,
                   run_test=False, source={})
CONTEXT = {"host": "gpu1", "compile": False}

STAGE2_BASE_YAML = """\
seq2seq:
  checkpoint_metric:
    metric: ndcg
    k: 10
general:
  seed: 0
"""


def _record(step, ndcg, best, recall=0.0, best_recall=0.0):
    return {"epoch": step // 68, "global_step": step, "recall@10": recall, "ndcg@10": ndcg,
            "best_recall@10": best_recall, "best_ndcg@10": best}


# The peak is neither first nor last:
# | step | ndcg@10 | best_ndcg@10 |
# |------|---------|--------------|
# | 68   | 0.30    | 0.30         |
# | 136  | 0.45    | 0.45         |
# | 204  | 0.40    | 0.45         |
CURVE = [_record(68, 0.30, 0.30), _record(136, 0.45, 0.45), _record(204, 0.40, 0.45)]

# the clock advances 10 s per reading, so every finished stage took 10 s
TEN_SECONDS_EACH = {"stage1_seconds": 10, "stage2_seconds": 10, "test_seconds": None}

# the heartbeat's wall clock is fixed, so every beat has the same content
BEAT = {"host": socket.gethostname(), "pid": os.getpid(), "time": "2026-11-24T10:00:00+00:00"}


def _stage2_sending(records):
    def stage2(spec, folder, config, on_validation):
        for record in records:
            on_validation(record)
        return stages.Stage2Result(Path(folder) / "checkpoints" / "stage2_best.pt", {})
    return stage2


@pytest.fixture
def mocks(tmp_path, monkeypatch):
    """Mocks the stages (except stage2_config), the clock and the clean-up; Stage 2 sends CURVE."""
    base = tmp_path / "stage2_base.yaml"
    base.write_text(STAGE2_BASE_YAML)
    monkeypatch.setattr(stages, "BASE_CONFIGS", {("toys", "rqvae"): ("unused.yaml", str(base))})
    found = {"stage1": Mock(), "stage2": Mock(side_effect=_stage2_sending(CURVE)), "test": Mock(),
             "wandb": Mock(run=None), "gc": Mock(), "torch": Mock()}
    for name in ("stage1", "stage2", "test"):
        monkeypatch.setattr(stages, name, found[name])
    for name in ("wandb", "gc", "torch"):
        monkeypatch.setattr(runner, name, found[name])
    monkeypatch.setattr(runner.time, "monotonic", Mock(side_effect=itertools.count(0, 10)))
    monkeypatch.setattr(heartbeat, "_now", lambda: datetime(2026, 11, 24, 10, tzinfo=timezone.utc))
    return found


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "trial_0007"


def _run(spec, folder, trial=None):
    return run_trial(spec, folder, trial, compile=False, context=CONTEXT)


def _trial_json(folder):
    return json.loads((folder / "trial.json").read_text())


# reruns have no Optuna trial
@pytest.mark.parametrize("with_trial", [True, False], ids=["search-trial", "rerun"])
def test_a_complete_run_is_worth_the_best_validation_value(mocks, folder, with_trial):
    trial = optuna.create_study(direction="maximize").ask() if with_trial else None

    assert _run(SPEC, folder, trial) == TrialOutcome("complete", 0.45, None, None, TEN_SECONDS_EACH)


def test_trial_json_holds_spec_context_and_outcome(mocks, folder):
    _run(SPEC, folder)

    assert _trial_json(folder) == {"spec": SPEC.to_json(), "context": CONTEXT,
                                   "outcome": {"status": "complete", "value": 0.45, "failure_class": None,
                                               "error": None, "timings": TEN_SECONDS_EACH}}


def test_trial_json_is_written_before_stage1_runs(mocks, folder):
    seen = []
    mocks["stage1"].side_effect = lambda *args: seen.append(_trial_json(folder))

    _run(SPEC, folder)

    assert seen == [{"spec": SPEC.to_json(), "context": CONTEXT}]


def test_an_existing_trial_json_is_rejected_before_any_stage(mocks, folder):
    folder.mkdir()
    (folder / "trial.json").write_text('{"earlier": "run"}')

    with pytest.raises(FileExistsError):
        _run(SPEC, folder)

    assert (mocks["stage1"].call_count, mocks["stage2"].call_count) == (0, 0)
    # no heartbeat either: the folder still belongs to the earlier run
    assert (sorted(path.name for path in folder.iterdir()), (folder / "trial.json").read_text()) == (
        ["trial.json"], '{"earlier": "run"}')


def test_a_context_that_is_not_json_leaves_no_trial_json(mocks, folder):
    # a half-written trial.json would lock the folder and break every reader
    with pytest.raises(TypeError):
        run_trial(SPEC, folder, None, compile=False, context={"host": Path("/gpu1")})

    assert (sorted(folder.iterdir()), mocks["stage1"].call_count) == ([], 0)


# reruns need it too: the queue judges a stale claim by the run folder's heartbeat
@pytest.mark.parametrize("with_trial", [True, False], ids=["search-trial", "rerun"])
def test_the_heartbeat_is_written_before_stage1_runs(mocks, folder, with_trial):
    trial = optuna.create_study(direction="maximize").ask() if with_trial else None
    seen = []
    mocks["stage1"].side_effect = lambda *args: seen.append(json.loads((folder / "heartbeat").read_text()))

    _run(SPEC, folder, trial)

    assert seen == [BEAT]


@pytest.mark.parametrize("stage2_raises", [None, RuntimeError("boom"), KeyboardInterrupt()],
                         ids=["complete", "crash", "interrupt"])
def test_the_heartbeat_thread_is_stopped_after_every_trial(mocks, folder, stage2_raises):
    if stage2_raises is not None:
        mocks["stage2"].side_effect = stage2_raises

    try:
        _run(SPEC, folder)
    except KeyboardInterrupt:
        pass

    assert [thread for thread in threading.enumerate() if thread.name == "heartbeat"] == []


def test_semids_from_skips_stage1(mocks, folder):
    outcome = _run(SPEC_B, folder)

    assert mocks["stage1"].call_count == 0
    assert mocks["stage2"].call_args_list == [call(SPEC_B, folder, folder / "configs" / "stage2.yaml", ANY)]
    assert outcome.timings == {"stage1_seconds": None, "stage2_seconds": 10, "test_seconds": None}


@pytest.mark.parametrize("run_test, expected_calls", [(True, 1), (False, 0)])
def test_run_test_decides_whether_the_test_set_is_evaluated(mocks, folder, run_test, expected_calls):
    spec = TrialSpec.from_json({**SPEC.to_json(), "run_test": run_test})

    _run(spec, folder)

    best = folder / "checkpoints" / "stage2_best.pt"
    assert mocks["test"].call_args_list == [call(spec, folder, best)] * expected_calls


def test_the_value_stays_the_validation_maximum_when_the_test_runs(mocks, folder):
    mocks["test"].return_value = {"metrics": {"ndcg@10": 0.99}}
    spec = TrialSpec.from_json({**SPEC.to_json(), "run_test": True})

    assert _run(spec, folder) == TrialOutcome("complete", 0.45, None, None, {**TEN_SECONDS_EACH, "test_seconds": 10})


def test_the_metric_comes_from_the_resolved_stage2_config(mocks, folder):
    # selected on recall@10 by the spec; ndcg@10 peaks elsewhere, so reading it would give 0.35:
    # | step | recall@10 | ndcg@10 | best_recall@10 |
    # |------|-----------|---------|----------------|
    # | 68   | 0.6       | 0.30    | 0.6            |
    # | 136  | 0.5       | 0.35    | 0.6            |
    mocks["stage2"].side_effect = _stage2_sending([_record(68, 0.30, 0.30, recall=0.6, best_recall=0.6),
                                                   _record(136, 0.35, 0.35, recall=0.5, best_recall=0.6)])
    spec = TrialSpec.from_json({**SPEC.to_json(), "stage2_overrides": {"seq2seq.checkpoint_metric.metric": "recall"}})
    study = optuna.create_study(direction="maximize")

    outcome = _run(spec, folder, study.ask())

    assert (outcome.value, study.trials[0].intermediate_values) == (0.6, {68: 0.6, 136: 0.6})


def test_the_optuna_trial_gets_the_best_so_far_curve(mocks, folder):
    study = optuna.create_study(direction="maximize")

    _run(SPEC, folder, study.ask())

    assert study.trials[0].intermediate_values == {68: 0.30, 136: 0.45, 204: 0.45}


# the clock reads 10 s per finished stage; the failing stage and those after it have no time
@pytest.mark.parametrize("stage, timings", [
    ("stage1", {"stage1_seconds": None, "stage2_seconds": None, "test_seconds": None}),
    ("stage2", {"stage1_seconds": 10, "stage2_seconds": None, "test_seconds": None}),
    ("test", {"stage1_seconds": 10, "stage2_seconds": 10, "test_seconds": None}),
])
def test_a_raising_stage_is_a_crash(mocks, folder, stage, timings):
    mocks[stage].side_effect = RuntimeError("boom")
    spec = TrialSpec.from_json({**SPEC.to_json(), "run_test": True})

    assert _run(spec, folder) == TrialOutcome("failed", None, "crash", "RuntimeError: boom", timings)


def test_a_crash_is_recorded_in_trial_json(mocks, folder):
    mocks["stage1"].side_effect = RuntimeError("boom")

    _run(SPEC, folder)

    assert _trial_json(folder)["outcome"] == {"status": "failed", "value": None, "failure_class": "crash",
                                              "error": "RuntimeError: boom",
                                              "timings": {"stage1_seconds": None, "stage2_seconds": None,
                                                          "test_seconds": None}}


def test_a_collision_overflow_is_infeasible(mocks, folder):
    message = "Collision overflow! Max depth 2 exceeds limit 2."
    mocks["stage1"].side_effect = CollisionOverflowError(message)

    assert _run(SPEC, folder) == TrialOutcome("failed", None, "infeasible", f"CollisionOverflowError: {message}",
                                              {"stage1_seconds": None, "stage2_seconds": None, "test_seconds": None})


def test_the_failure_class_comes_from_the_type_not_the_message(mocks, folder):
    # same message, plain ValueError: a crash
    mocks["stage1"].side_effect = ValueError("Collision overflow! Max depth 2 exceeds limit 2.")

    assert _run(SPEC, folder) == TrialOutcome("failed", None, "crash",
                                              "ValueError: Collision overflow! Max depth 2 exceeds limit 2.",
                                              {"stage1_seconds": None, "stage2_seconds": None, "test_seconds": None})


def test_stage2_without_records_is_a_crash(mocks, folder):
    mocks["stage2"].side_effect = _stage2_sending([])

    # Stage 2 itself returns; the missing curve fails afterwards, so its time is recorded
    assert _run(SPEC, folder) == TrialOutcome("failed", None, "crash", "ValueError: no validation records",
                                              TEN_SECONDS_EACH)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_propagates_without_an_outcome(mocks, folder, interrupt):
    handlers_before = list(logging.getLogger().handlers)
    mocks["stage2"].side_effect = interrupt()

    with pytest.raises(interrupt):
        _run(SPEC, folder)

    # no outcome: the reaper, not the worker, decides about a killed run
    assert _trial_json(folder) == {"spec": SPEC.to_json(), "context": CONTEXT}
    assert logging.getLogger().handlers == handlers_before


def test_stage_log_lines_and_a_crash_traceback_go_to_trial_log(mocks, folder):
    # WARNING, so the line passes whatever level the root logger has in the test run
    def stage1(*args):
        logging.getLogger("train_quantizer").warning("epoch 1 done")
        raise RuntimeError("boom")
    mocks["stage1"].side_effect = stage1

    _run(SPEC, folder)

    # record lines start with the date and time; the traceback follows the error line
    lines = (folder / "trial.log").read_text().splitlines()
    records = [line.split(" ", 2)[2] for line in lines if line[:4].isdigit()]
    assert records == ["WARNING train_quantizer: epoch 1 done", "ERROR search.runner: trial failed"]
    assert (lines[2], lines[-1]) == ("Traceback (most recent call last):", "RuntimeError: boom")


@pytest.mark.parametrize("fails", [False, True], ids=["complete", "crash"])
def test_the_log_handler_is_removed_after_the_trial(mocks, folder, fails):
    handlers_before = list(logging.getLogger().handlers)
    if fails:
        mocks["stage1"].side_effect = RuntimeError("boom")

    _run(SPEC, folder)

    assert logging.getLogger().handlers == handlers_before


@pytest.mark.parametrize("stage2_raises", [None, RuntimeError("boom"), KeyboardInterrupt()],
                         ids=["complete", "crash", "interrupt"])
def test_the_clean_up_runs_once_after_every_trial(mocks, folder, stage2_raises):
    if stage2_raises is not None:
        mocks["stage2"].side_effect = stage2_raises

    try:
        _run(SPEC, folder)
    except KeyboardInterrupt:
        pass

    assert (mocks["gc"].collect.call_count, mocks["torch"].cuda.empty_cache.call_count,
            mocks["torch"]._dynamo.reset.call_count) == (1, 1, 1)


@pytest.mark.parametrize("run, finishes", [(None, []), (Mock(), [call(exit_code=1)])], ids=["no-run", "open-run"])
def test_an_open_wandb_run_is_finished_as_failed(mocks, folder, run, finishes):
    # only a crashed stage leaves its W&B run open; without an open run there is nothing to finish
    mocks["wandb"].run = run

    _run(SPEC, folder)

    assert mocks["wandb"].finish.call_args_list == finishes


def test_a_failing_wandb_finish_still_cleans_up_and_returns_the_outcome(mocks, folder):
    mocks["wandb"].run = Mock()
    mocks["wandb"].finish.side_effect = RuntimeError("wandb service died")

    outcome = _run(SPEC, folder)

    assert outcome == TrialOutcome("complete", 0.45, None, None, TEN_SECONDS_EACH)
    assert (mocks["gc"].collect.call_count, mocks["torch"].cuda.empty_cache.call_count,
            mocks["torch"]._dynamo.reset.call_count) == (1, 1, 1)
