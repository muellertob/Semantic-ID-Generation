"""
What one search trial runs.

A worker builds a TrialSpec, writes it into the trial folder's trial.json (and the rerun queue writes it
into job files); a rerun or another worker reads it back. The spec is therefore checked when it is built:
a spec that would be misread after a trip through JSON, or that claims a config that never trains, is
rejected instead of run.
"""
import copy
import dataclasses
import json
import types
import typing
from typing import Any

PHASES = ("A", "B", "confirm", "final")


@dataclasses.dataclass(frozen=True)
class TrialSpec:
    """
    dataset, quantizer: names the stage and space modules resolve (not checked here).
    phase: "A" (quantizer search), "B" (recommender search), "confirm" or "final" (reruns).
    quantizer_overrides, recommender_overrides: dotted config keys -> values; the quantizer's must be empty when
        semids_from is set, because the quantizer stage does not run then.
    quantizer_seed, recommender_seed: seeds of the two stages.
    semids_from: trial folder whose semids.pt the recommender uses, or None to fit the quantizer.
    max_steps: recommender step budget.
    run_test: run the test set after training (reruns only).
    source: provenance, e.g. {"study": ..., "trial_number": ..., "retry_of": ...}; keys are free.
    """
    dataset: str
    quantizer: str
    phase: str
    quantizer_overrides: dict[str, Any]
    recommender_overrides: dict[str, Any]
    quantizer_seed: int
    recommender_seed: int
    semids_from: str | None
    max_steps: int
    run_test: bool
    source: dict[str, Any]

    def __post_init__(self):
        for name, hint in typing.get_type_hints(type(self)).items():
            _check_type(name, getattr(self, name), _accepted_types(hint))
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {list(PHASES)}, got {self.phase!r}")
        for name in ("quantizer_overrides", "recommender_overrides", "source"):
            _check_json_safe(name, getattr(self, name))
        if self.semids_from is not None and self.quantizer_overrides:
            raise ValueError("quantizer_overrides must be empty when semids_from is set "
                             "(the quantizer stage does not run)")

    def to_json(self) -> dict:
        """All fields as a plain dict (a deep copy, so changing it leaves the spec alone)."""
        return copy.deepcopy({f.name: getattr(self, f.name) for f in dataclasses.fields(self)})

    @classmethod
    def from_json(cls, data: dict) -> "TrialSpec":
        """Inverse of to_json; every field is required and no other key is allowed."""
        expected = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - expected)
        if unknown:
            raise ValueError(f"TrialSpec JSON has unknown keys: {unknown}")
        missing = sorted(expected - set(data))
        if missing:
            raise ValueError(f"TrialSpec JSON is missing keys: {missing}")
        return cls(**copy.deepcopy(data))


def _accepted_types(hint) -> tuple[type, ...]:
    """`str | None` -> (str, NoneType); `dict[str, Any]` -> (dict,); `int` -> (int,)."""
    if isinstance(hint, types.UnionType):
        return tuple(t for arg in typing.get_args(hint) for t in _accepted_types(arg))
    return (typing.get_origin(hint) or hint,)


def _check_type(name: str, value, accepted: tuple[type, ...]) -> None:
    # bool is a subclass of int, but True as a step budget or seed is a mistake, not a number
    is_bool_for_int = isinstance(value, bool) and bool not in accepted
    if not isinstance(value, accepted) or is_bool_for_int:
        names = " or ".join(t.__name__ for t in accepted)
        raise ValueError(f"{name} must be {names}, got {type(value).__name__}")


def _check_json_safe(name: str, value: dict) -> None:
    try:
        survives = json.loads(json.dumps(value, allow_nan=False)) == value
    except (TypeError, ValueError):
        survives = False
    if not survives:
        raise ValueError(f"{name} must survive a JSON round trip unchanged: {value!r}")
