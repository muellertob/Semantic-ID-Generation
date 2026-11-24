"""
Tests for search/reporter.py on real in-memory Optuna studies and hand-made curves.
"""
import optuna
import pytest

from search.reporter import ValidationReporter, objective_value, read_records

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _record(epoch, step, ndcg, best):
    return {"epoch": epoch, "global_step": step, "ndcg@10": ndcg, "best_ndcg@10": best}


# Current and best-so-far differ after the peak:
# | step   | ndcg@10 | best_ndcg@10 |
# |--------|---------|--------------|
# | 10,000 | 0.40    | 0.40         |
# | 25,160 | 0.20    | 0.40         |
# | 50,320 | 0.30    | 0.40         |
CURVE = [_record(147, 10_000, 0.40, 0.40), _record(370, 25_160, 0.20, 0.40), _record(740, 50_320, 0.30, 0.40)]


# Checkpoint selected on recall@10 (no best_ndcg@10 key); ndcg@10 peaks elsewhere, so using it gives 0.35 or crashes:
# | step | recall@10 | ndcg@10 | best_recall@10 |
# |------|-----------|---------|----------------|
# | 68   | 0.6       | 0.30    | 0.6            |
# | 136  | 0.5       | 0.35    | 0.6            |
RECALL_RUN = [{"epoch": 1, "global_step": 68, "recall@10": 0.6, "ndcg@10": 0.30, "best_recall@10": 0.6},
              {"epoch": 2, "global_step": 136, "recall@10": 0.5, "ndcg@10": 0.35, "best_recall@10": 0.6}]


@pytest.fixture
def study():
    return optuna.create_study(direction="maximize")


# reruns have no Optuna trial
@pytest.mark.parametrize("with_trial", [True, False], ids=["search-trial", "rerun"])
def test_every_record_is_written_once_in_order(with_trial, study, tmp_path):
    path = tmp_path / "validation.jsonl"
    reporter = ValidationReporter(path, "ndcg@10", trial=study.ask() if with_trial else None)
    for record in CURVE:
        reporter.on_validation(record)

    assert read_records(path) == CURVE


def test_an_existing_validation_log_is_rejected(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text('{"earlier": "run"}\n')

    with pytest.raises(FileExistsError) as error:
        ValidationReporter(path, "ndcg@10")

    assert str(error.value) == f"validation log already exists: {path}"
    assert path.read_text() == '{"earlier": "run"}\n'


def test_a_second_reporter_on_the_same_path_is_rejected_before_any_validation(tmp_path):
    path = tmp_path / "validation.jsonl"
    ValidationReporter(path, "ndcg@10")

    with pytest.raises(FileExistsError) as error:
        ValidationReporter(path, "ndcg@10")

    assert str(error.value) == f"validation log already exists: {path}"


def test_a_run_without_validations_leaves_an_empty_log(tmp_path):
    # e.g. the recommender crashes before its first validation
    path = tmp_path / "validation.jsonl"
    ValidationReporter(path, "ndcg@10")

    assert read_records(path) == []


def test_reports_the_best_so_far_value_at_the_global_step(study, tmp_path):
    reporter = ValidationReporter(tmp_path / "validation.jsonl", "ndcg@10", trial=study.ask())
    for record in CURVE:
        reporter.on_validation(record)

    # best-so-far, not the current 0.20 / 0.30
    assert study.trials[0].intermediate_values == {10_000: 0.40, 25_160: 0.40, 50_320: 0.40}


def test_never_stops_a_run_even_under_a_pruner(tmp_path):
    # A study with a pruner by mistake (first rung 25k; below 4 trials only the best at a rung survives):
    # | trial      | step   | best | pruner's answer if asked   |
    # |------------|--------|------|----------------------------|
    # | leader     | 25,160 | 0.50 | stays (alone at the rung)  |
    # | challenger | 10,000 | 0.10 | stays (below the rung)     |
    # | challenger | 25,160 | 0.20 | stop (0.20 < 0.50)         |
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner(
        min_resource=25_000, reduction_factor=4, min_early_stopping_rate=0, bootstrap_count=0))
    leader = study.ask()
    leader.report(0.50, 25_160)
    leader.should_prune()  # the pruner stores a trial's rung value only when asked
    study.tell(leader, 0.50)
    reporter = ValidationReporter(tmp_path / "validation.jsonl", "ndcg@10", trial=study.ask())

    stops = [reporter.on_validation(record)
             for record in [_record(147, 10_000, 0.10, 0.10), _record(370, 25_160, 0.20, 0.20)]]

    assert stops == [False, False]


def test_reports_the_best_so_far_of_the_given_metric(study, tmp_path):
    reporter = ValidationReporter(tmp_path / "validation.jsonl", "recall@10", trial=study.ask())
    for record in RECALL_RUN:
        reporter.on_validation(record)

    assert study.trials[0].intermediate_values == {68: 0.6, 136: 0.6}


def test_objective_value_is_the_maximum_of_the_given_metric():
    assert objective_value(RECALL_RUN, "recall@10") == 0.6


def test_objective_value_is_the_maximum_of_the_curve():
    # the peak is neither first nor last
    records = [_record(1, 68, 0.30, 0.30), _record(2, 136, 0.45, 0.45), _record(3, 204, 0.40, 0.45)]

    assert objective_value(records, "ndcg@10") == 0.45


def test_objective_value_without_records_is_an_error():
    with pytest.raises(ValueError) as error:
        objective_value([], "ndcg@10")

    assert str(error.value) == "no validation records"
