"""
One run of the search driver, a search trial or a rerun, in its own folder.

trial.json is written first, so a folder always says what ran; the outcome is added at the end ("has an outcome"
means done). A heartbeat shows that the run lives. A stage's exception becomes a failure class instead of being raised:
- "infeasible": a collision overflow (deterministic: no retry, counts toward the budget);
- "reaped": the reaper already ended the Optuna trial and enqueued its retry;
- "crash": anything else.
Every run ends with the same clean-up, because a worker runs many trials in one process.
"""
import dataclasses
import gc
import logging
import time
from pathlib import Path

import optuna
from optuna.exceptions import UpdateFinishedTrialError
import torch
import wandb
from omegaconf import OmegaConf

import train_seq2seq
from generate_semids import CollisionOverflowError
from search import stages
from search.heartbeat import Heartbeat
from search.json_files import create_json, replace_json
from search.reporter import ValidationReporter, objective_value, read_records
from search.spec import TrialSpec

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TrialOutcome:
    status: str                # "complete" | "failed"
    value: float | None        # the objective value; None unless complete
    failure_class: str | None  # None | "infeasible" | "reaped" | "crash"
    error: str | None          # "<exception type>: <message>", or None
    timings: dict              # quantizer_seconds, recommender_seconds, test_seconds; None for a stage that did
                               # not finish


def run_trial(spec: TrialSpec, folder: str | Path, trial: optuna.Trial | None, *, compile: bool,
              context: dict) -> TrialOutcome:
    """
    Run the stages of `spec` in `folder`. `trial` is the Optuna trial of a search trial, None for a rerun;
    `context` (Optuna number and params, card, host, git) is written to trial.json as given.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    trial_json = folder / "trial.json"
    head = {"spec": spec.to_json(), "context": context}
    create_json(trial_json, head)  # a used folder fails here, before any stage

    handler = logging.FileHandler(folder / "trial.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    timings = {"quantizer_seconds": None, "recommender_seconds": None, "test_seconds": None}
    try:
        with Heartbeat(folder / "heartbeat"):
            try:
                value = _run_stages(spec, folder, trial, compile, timings)
                outcome = TrialOutcome("complete", value, None, None, dict(timings))
            except Exception as error:
                logger.exception("trial failed")
                failure_class = _failure_class(error)
                outcome = TrialOutcome("failed", None, failure_class, f"{type(error).__name__}: {error}",
                                       dict(timings))
            replace_json(trial_json, {**head, "outcome": dataclasses.asdict(outcome)})
            return outcome
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
        _clean_up()


def _failure_class(error: Exception) -> str:
    if isinstance(error, CollisionOverflowError):
        return "infeasible"
    if isinstance(error, UpdateFinishedTrialError):  # raised by trial.report after the reaper's FAIL
        return "reaped"
    return "crash"


def _run_stages(spec: TrialSpec, folder: Path, trial, compile: bool, timings: dict) -> float:
    if spec.semids_from is None:
        start = time.monotonic()
        stages.quantizer(spec, folder, compile)
        timings["quantizer_seconds"] = time.monotonic() - start

    config = stages.recommender_config(spec, folder, compile)
    name, k = train_seq2seq.get_checkpoint_metric(OmegaConf.load(config))
    metric = f"{name}@{k}"
    log = folder / "validation.jsonl"
    reporter = ValidationReporter(log, metric, trial)
    start = time.monotonic()
    result = stages.recommender(spec, folder, config, reporter.on_validation)
    timings["recommender_seconds"] = time.monotonic() - start
    value = objective_value(read_records(log), metric)

    if spec.run_test:
        start = time.monotonic()
        stages.test(spec, folder, result.best_checkpoint)
        timings["test_seconds"] = time.monotonic() - start
    return value


def _clean_up() -> None:
    if wandb.run is not None:  # only a crashed stage leaves its W&B run open; the stages finish their own
        try:
            wandb.finish(exit_code=1)
        except Exception:  # e.g. a dead W&B service: the rest of the clean-up and the outcome still count
            logger.exception("wandb.finish failed")
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()  # a fresh compile per trial, so the recompile limit is not reached
