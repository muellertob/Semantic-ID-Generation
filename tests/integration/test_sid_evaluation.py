"""
Integration tests for evaluate_semids (utils/sid_evaluation.py).

Tests the full orchestration: both metrics computed, report printed,
correct keys returned.
"""
import pytest
import torch
from omegaconf import OmegaConf
from utils.sid_evaluation import evaluate_semids


@pytest.fixture
def config():
    return OmegaConf.create({"model": {"codebook_clusters": 32}})


@pytest.fixture
def semids():
    torch.manual_seed(0)
    return torch.randint(0, 32, (200, 3))


def test_returns_utilisation_and_collision_keys(semids, config):
    result = evaluate_semids(semids, config)
    assert {"utilisation", "collision"}.issubset(result.keys())


def test_utilisation_has_one_entry_per_layer(semids, config):
    result = evaluate_semids(semids, config)
    assert len(result["utilisation"]["per_layer"]) == 3


def test_collision_keys_present(semids, config):
    result = evaluate_semids(semids, config)
    assert {"collision_rate", "n_collisions", "max_depth"}.issubset(
        result["collision"].keys()
    )


def test_large_dataset_does_not_raise():
    torch.manual_seed(1)
    sids = torch.randint(0, 128, (12101, 3))
    config = OmegaConf.create({"model": {"codebook_clusters": 128}})
    evaluate_semids(sids, config)
