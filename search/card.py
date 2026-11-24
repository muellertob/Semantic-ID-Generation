"""
The GPU a worker runs on and what follows from it, read once when the worker starts.

The worker does not choose its GPU: CUDA_VISIBLE_DEVICES, set when it is launched, does. From the visible card follows
whether torch.compile is on, and with it the order in which the worker takes work: only compiled workers take
confirmation and final runs, so the candidates of a cell are compared under the same numerics. Every trial.json
records the run context: host, card, versions and the git state of the code.
"""
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

import optuna
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.utils._triton import has_triton

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CardInfo:
    name: str                    # e.g. "NVIDIA GeForce RTX 3090"
    uuid: str
    capability: tuple[int, int]  # CUDA compute capability (major, minor)


def detect_card() -> CardInfo | None:
    """The first visible CUDA device, or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return CardInfo(props.name, str(props.uuid), (props.major, props.minor))


def compile_default(override: bool | None = None) -> bool:
    if override is not None:
        return override
    return torch.cuda.is_available() and get_interface_for_device("cuda").is_triton_capable() and has_triton()


def git_state(repo: str | Path = REPO) -> dict:
    """The commit and the `git status --porcelain` lines of `repo` (empty = clean; untracked files count)."""
    repo = Path(repo)
    if not (repo / ".git").exists():
        raise RuntimeError(f"{repo} has no .git: sync .git/ with the code, so every run records its code state")
    commit = _git(repo, "rev-parse", "HEAD").strip()
    return {"git_commit": commit, "git_status": _git(repo, "status", "--porcelain").splitlines()}


def run_context(card: CardInfo | None, compile: bool, git: dict) -> dict:
    """What every trial.json records about where and with which code it ran."""
    return {
        "host": socket.gethostname(),
        "gpu_name": card.name if card else None,
        "gpu_uuid": card.uuid if card else None,
        "gpu_capability": f"{card.capability[0]}.{card.capability[1]}" if card else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "compile": compile,
        "torch_version": str(torch.__version__),
        "optuna_version": optuna.__version__,
        **git,
    }


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {error.stderr.strip()}") from None
