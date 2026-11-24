"""
Names inside a study folder, in one place, so the worker and the reaper agree on them.

A trial folder is named after the Optuna trial number, so a retried trial gets a folder of its own.
"""
from pathlib import Path


def trial_folder(study_dir: str | Path, number: int) -> Path:
    return Path(study_dir) / f"trial_{number:04d}"
