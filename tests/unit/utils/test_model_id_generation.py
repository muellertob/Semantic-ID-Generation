"""
Tests of utils/model_id_generation.py.
"""
import pytest
from omegaconf import OmegaConf

from utils.model_id_generation import generate_model_id


@pytest.mark.parametrize("commit_all_levels, suffix", [(False, ""), (True, "-cal")])
def test_rqvae_id_marks_commit_all_levels(commit_all_levels, suffix):
    # the suffix keeps runs with the option apart; without it the ID is unchanged, so older IDs stay valid
    config = OmegaConf.create({
        "data": {"dataset": "amazon", "category": "toys", "batch_size": 1024, "input_normalization": "none"},
        "model": {"quantizer_type": "rqvae", "hidden_dimensions": [512, 256, 128], "latent_dimension": 32,
                  "codebook_layers": 3, "codebook_size": 256, "commitment_weight": 0.25,
                  "commit_all_levels": commit_all_levels},
        "train": {"learning_rate": 0.001, "weight_decay": 0, "num_epochs": 2000},
    })
    assert generate_model_id(config) == (
        "rqvae-amazon-toys-bs1024-norm_none-hd512_256_128-ld32-cb3x256-cw0.25-lr0.001-wd0-ep2000" + suffix)
