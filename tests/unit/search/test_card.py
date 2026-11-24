"""
Tests for search/card.py. torch's device queries and Triton check are mocked (no GPU); the git state is read from
a real throwaway repository in tmp_path.
"""
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import optuna
import pytest
import torch

from search import card
from search.card import CardInfo, compile_default, detect_card, git_state, run_context

RTX_3090 = CardInfo("NVIDIA GeForce RTX 3090", "GPU-3090", (8, 6))


def _torch_sees(monkeypatch, cuda, major, triton):
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=cuda))
    monkeypatch.setattr(torch.cuda, "get_device_properties", Mock(return_value=SimpleNamespace(major=major)))
    monkeypatch.setattr(card, "has_triton", Mock(return_value=triton))


# Compile also decides the work order: only compiled workers take confirmation and final runs.
# | CUDA | capability major | Triton (any backend) | compile | why                                           |
# |------|------------------|----------------------|---------|-----------------------------------------------|
# | yes  | 8 (3090)         | yes                  | on      | the backend supports the card                 |
# | yes  | 7 (2080 Ti)      | yes                  | on      | the backend supports the card                 |
# | yes  | 6 (1080 Ti)      | yes                  | off     | e.g. a Triton CPU backend: the GPU is too old |
# | yes  | 8                | no                   | off     | no Triton installed                           |
# | no   | -                | yes                  | off     | CPU / MPS run                                 |
@pytest.mark.parametrize("cuda, major, triton, expected", [(True, 8, True, True), (True, 7, True, True),
                                                           (True, 6, True, False), (True, 8, False, False),
                                                           (False, 8, True, False)],
                         ids=["3090", "2080ti", "1080ti-cpu-backend", "no-triton", "no-cuda"])
def test_compile_follows_torchs_checks_for_the_visible_cuda_device(monkeypatch, cuda, major, triton, expected):
    _torch_sees(monkeypatch, cuda, major, triton)

    assert compile_default() == expected


@pytest.mark.parametrize("major, override", [(6, True), (8, False)])
def test_an_explicit_compile_choice_wins_over_torchs_checks(monkeypatch, major, override):
    _torch_sees(monkeypatch, True, major, True)

    assert compile_default(override) == override


def test_detect_card_reads_the_first_visible_device(monkeypatch):
    props = SimpleNamespace(name="NVIDIA GeForce RTX 3090", uuid="GPU-3090", major=8, minor=6)
    get_props = Mock(return_value=props)
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_props)

    assert (detect_card(), get_props.call_args.args) == (RTX_3090, (0,))


def test_detect_card_without_cuda_is_none(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))

    assert detect_card() is None


GIT = {"git_commit": "abc123", "git_status": [" M search/runner.py"]}


@pytest.mark.parametrize("visible", ["2", None], ids=["set", "unset"])
def test_the_run_context_is_exact(monkeypatch, visible):
    monkeypatch.setattr(card.socket, "gethostname", Mock(return_value="sxst06"))
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    assert run_context(RTX_3090, True, GIT) == {
        "host": "sxst06", "gpu_name": "NVIDIA GeForce RTX 3090", "gpu_uuid": "GPU-3090", "gpu_capability": "8.6",
        "cuda_visible_devices": visible, "compile": True,
        "torch_version": str(torch.__version__), "optuna_version": optuna.__version__, "git_commit": "abc123",
        "git_status": [" M search/runner.py"]}


def test_the_run_context_survives_json(monkeypatch):
    # trial.json is created from it, so it must be plain JSON (torch.__version__ is not a plain str)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    context = run_context(RTX_3090, True, GIT)

    assert json.loads(json.dumps(context)) == context


def test_the_run_context_without_a_card_has_no_gpu_fields(monkeypatch):
    monkeypatch.setattr(card.socket, "gethostname", Mock(return_value="macbook"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert run_context(None, False, GIT) == {
        "host": "macbook", "gpu_name": None, "gpu_uuid": None, "gpu_capability": None,
        "cuda_visible_devices": None, "compile": False,
        "torch_version": str(torch.__version__), "optuna_version": optuna.__version__, "git_commit": "abc123",
        "git_status": [" M search/runner.py"]}


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    """A throwaway repository with one commit of a.py."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
         "commit", "-q", "-m", "first")
    return repo


def test_a_clean_repository_records_its_commit_and_no_status(repo):
    assert git_state(repo) == {"git_commit": _git(repo, "log", "-1", "--format=%H").strip(), "git_status": []}


def test_changed_and_untracked_files_appear_in_the_status(repo):
    # untracked code can run too, so it must be visible
    (repo / "a.py").write_text("x = 2\n")
    (repo / "b.py").write_text("y = 1\n")

    assert git_state(repo) == {"git_commit": _git(repo, "log", "-1", "--format=%H").strip(),
                               "git_status": [" M a.py", "?? b.py"]}


def test_a_folder_without_git_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match=r"sync \.git/"):
        git_state(tmp_path)


def test_a_failing_git_command_reports_gits_own_message(tmp_path):
    # a .git that git cannot read, e.g. a broken sync
    (tmp_path / ".git").write_text("not a repository\n")

    with pytest.raises(RuntimeError, match=r"git rev-parse HEAD failed in .*: fatal: "):
        git_state(tmp_path)
