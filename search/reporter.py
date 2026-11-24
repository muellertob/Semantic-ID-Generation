"""
The validation reporter of one run.

train_seq2seq.run_training calls `on_validation(record)` after each checkpoint decision. Every record goes to the
run's validation.jsonl (the curve with all metrics). A search trial also reports `best_<metric>` at `global_step` 
to its Optuna trial, a copy for the study's plots; reruns belong to no study and only write. The reporter never 
stops a run (no pruning).

`metric` is the run's checkpoint metric (train_seq2seq.get_checkpoint_metric); it has no default here, so it
cannot disagree with the config. The objective value is the metric's maximum, the best checkpoint's value.
"""
import json
from pathlib import Path

import optuna


class ValidationReporter:
    def __init__(self, path, metric: str, trial: optuna.Trial | None = None):
        path = Path(path)
        # created now and exclusively: a second reporter on this path fails at once; no validations = empty log
        try:
            open(path, "x").close()
        except FileExistsError:
            raise FileExistsError(f"validation log already exists: {path}") from None
        self._path = path
        self._trial = trial
        self._metric = metric

    def on_validation(self, record: dict) -> bool:
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self._trial is not None:
            self._trial.report(record[f"best_{self._metric}"], record["global_step"])
        return False


def read_records(path) -> list[dict]:
    """The records of a validation.jsonl, in the order they were written."""
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def objective_value(records: list[dict], metric: str) -> float:
    if not records:
        raise ValueError("no validation records")
    return max(record[metric] for record in records)
