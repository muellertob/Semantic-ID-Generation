"""
Tests for search/spec.py: a TrialSpec survives the trip through trial.json and queue job files unchanged,
and a spec that a worker would misread is rejected when it is built.
"""
import dataclasses
import json

import pytest

from search.spec import TrialSpec

# REF: a Phase A trial; both stages run, the test set is not touched.
# REF_B: a Phase B retry of trial 3; recommender only, on the SIDs of Phase A trial 7, so the quantizer has no
# overrides.
#
# | spec  | quantizer_overrides                             | semids_from                       | run_test | retry_of |
# |-------|-------------------------------------------------|-----------------------------------|----------|----------|
# | REF   | {"model.codebook_size": 256, "train.lr": 0.001} | None                              | False    | None     |
# | REF_B | {}                                              | "/search/toys/A-rqvae/trial_0007" | True     | 3        |
REF = TrialSpec(
    dataset="toys",
    quantizer="rqvae",
    phase="A",
    quantizer_overrides={"model.codebook_size": 256, "train.lr": 0.001},
    recommender_overrides={"training.learning_rate": 0.0005, "model.num_layers": 4},
    quantizer_seed=42,
    recommender_seed=42,
    semids_from=None,
    max_steps=100_000,
    run_test=False,
    source={"study": "toys.A.rqvae", "trial_number": 7, "retry_of": None},
)
REF_B = dataclasses.replace(
    REF,
    phase="B",
    quantizer_overrides={},
    semids_from="/search/toys/A-rqvae/trial_0007",
    run_test=True,
    source={"study": "toys.B.rqvae", "trial_number": 12, "retry_of": 3},
)

# REF as written to disk, typed out by hand so a dropped or renamed field shows up
REF_JSON = {
    "dataset": "toys",
    "quantizer": "rqvae",
    "phase": "A",
    "quantizer_overrides": {"model.codebook_size": 256, "train.lr": 0.001},
    "recommender_overrides": {"training.learning_rate": 0.0005, "model.num_layers": 4},
    "quantizer_seed": 42,
    "recommender_seed": 42,
    "semids_from": None,
    "max_steps": 100000,
    "run_test": False,
    "source": {"study": "toys.A.rqvae", "trial_number": 7, "retry_of": None},
}


@pytest.mark.parametrize("spec", [REF, REF_B], ids=["phase_A", "phase_B_retry"])
def test_round_trip_through_a_json_file_is_exact(spec):
    text = json.dumps(spec.to_json())
    assert TrialSpec.from_json(json.loads(text)) == spec


def test_to_json_writes_exactly_the_eleven_fields():
    assert REF.to_json() == REF_JSON


def test_from_json_rejects_an_unknown_key():
    # a typo in a hand-edited job file must not fall back silently to the old seed
    data = {**REF_JSON, "recommender_seeds": 7}
    with pytest.raises(ValueError) as excinfo:
        TrialSpec.from_json(data)
    assert str(excinfo.value) == "TrialSpec JSON has unknown keys: ['recommender_seeds']"


def test_from_json_rejects_a_missing_key():
    # no defaults: an old or truncated file fails instead of running with a guessed budget
    data = {k: v for k, v in REF_JSON.items() if k != "max_steps"}
    with pytest.raises(ValueError) as excinfo:
        TrialSpec.from_json(data)
    assert str(excinfo.value) == "TrialSpec JSON is missing keys: ['max_steps']"


def test_phase_outside_the_four_phases_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(REF, phase="C")
    assert str(excinfo.value) == "phase must be one of ['A', 'B', 'confirm', 'final'], got 'C'"


@pytest.mark.parametrize(
    "field, value, message",
    [
        # a string seed would land unchecked in the stage config
        ("quantizer_seed", "42", "quantizer_seed must be int, got str"),
        # bool is an int subclass in Python, but True is not a step budget
        ("max_steps", True, "max_steps must be int, got bool"),
        # the string "false" is truthy and would run the test set
        ("run_test", "false", "run_test must be bool, got str"),
    ],
)
def test_field_of_the_wrong_type_is_rejected(field, value, message):
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(REF, **{field: value})
    assert str(excinfo.value) == message


def test_overrides_that_change_in_json_are_rejected():
    # a tuple comes back from JSON as a list, so a rerun's spec would differ from the trial's
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(REF, recommender_overrides={"model.layers": (4, 4)})
    assert str(excinfo.value) == (
        "recommender_overrides must survive a JSON round trip unchanged: {'model.layers': (4, 4)}"
    )


def test_quantizer_overrides_with_semids_from_are_rejected():
    # the quantizer stage does not run on reused SIDs, so trial.json would claim a config that never trained
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(REF_B, quantizer_overrides={"train.lr": 0.001})
    assert str(excinfo.value) == (
        "quantizer_overrides must be empty when semids_from is set (the quantizer stage does not run)"
    )
