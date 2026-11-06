"""
Tests of utils/model_id_generation.py.
"""
import pytest
from omegaconf import OmegaConf

from utils.model_id_generation import generate_model_id


def _rqvae_config(**model):
    return OmegaConf.create({
        "data": {"dataset": "amazon", "category": "toys", "batch_size": 1024, "input_normalization": "none"},
        "model": {"quantizer_type": "rqvae", "hidden_dimensions": [512, 256, 128], "latent_dimension": 32,
                  "codebook_layers": 3, "codebook_size": 256, "beta": 6.0, **model},
        "train": {"learning_rate": 0.001, "weight_decay": 0, "num_epochs": 2000},
    })


def test_rqvae_id_names_beta_with_b():
    # "-b" instead of the former "-cw": beta on summed losses must never share an ID with the averaged-loss weight
    assert generate_model_id(_rqvae_config()) == (
        "rqvae-amazon-toys-bs1024-norm_none-hd512_256_128-ld32-cb3x256-b6.0-lr0.001-wd0-ep2000")


@pytest.mark.parametrize("commit_all_levels, suffix", [(False, ""), (True, "-cal")])
def test_rqvae_id_marks_commit_all_levels(commit_all_levels, suffix):
    # the suffix keeps runs with the option apart; without the option the ID has no suffix
    assert generate_model_id(_rqvae_config(commit_all_levels=commit_all_levels)) == (
        "rqvae-amazon-toys-bs1024-norm_none-hd512_256_128-ld32-cb3x256-b6.0-lr0.001-wd0-ep2000" + suffix)
