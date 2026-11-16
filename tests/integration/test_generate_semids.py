"""
Integration tests for Semantic-ID generation (generate_semids.py): model loading and the collision limit.
"""
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from generate_semids import load_trained_model, run_generation
from train_quantizer import create_model


def _rfsq_fixed_config(**model):
    return OmegaConf.create({"model": {
        "quantizer_type": "rfsq_fixed", "latent_dimension": 2, "hidden_dimensions": [4], "codebook_layers": 2,
        "level_list": [3, 2], "loss_type": "l1", "normalize_inputs": False, **model}})


def _rqvae_config(**model):
    return OmegaConf.create({"model": {
        "quantizer_type": "rqvae", "latent_dimension": 2, "hidden_dimensions": [4], "codebook_layers": 1,
        "codebook_size": 2, "beta": 0.25, "quantization_method": "ste", "distance_method": "l2",
        "normalize_inputs": False, "commit_all_levels": False, **model}})


def _run_generation_on_three_identical_items(config, tmp_path):
    config.general = {"seed": 0, "device": "cpu"}
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)
    model_path = tmp_path / "model.pt"
    torch.save(create_model(config, input_dim=8).state_dict(), model_path)
    with patch("generate_semids.load_data", return_value=torch.zeros(3, 8)):
        run_generation(str(config_path), str(model_path), str(tmp_path / "out" / "semids.pt"), run_eval=False)


def test_a_saved_and_reloaded_fixed_scale_model_gives_the_same_semantic_ids(tmp_path):
    # load_state_dict(strict=False) does not report weights that were not loaded, so a freshly initialised model
    # would pass silently unless the IDs are compared
    config = _rfsq_fixed_config()
    torch.manual_seed(0)
    model = create_model(config, input_dim=8)
    model.eval()
    x = torch.arange(32.).reshape(4, 8) / 10 - 1.5
    expected = model.get_semantic_ids(x)
    torch.save(model.state_dict(), tmp_path / "model.pt")

    loaded, _, _ = load_trained_model(tmp_path / "model.pt", config, torch.device("cpu"), 8)

    assert torch.equal(loaded.get_semantic_ids(x), expected)


def test_the_collision_limit_of_a_fixed_scale_quantizer_is_its_vocabulary_size_of_one_level(tmp_path):
    # one level with 2 possible values -> limit 2; three identical items get collision tokens 0, 1, 2;
    # token 2 is not below the limit
    config = _rfsq_fixed_config(latent_dimension=1, codebook_layers=1, level_list=[2])
    with pytest.raises(ValueError) as error:
        _run_generation_on_three_identical_items(config, tmp_path)
    assert str(error.value) == "Collision overflow! Max depth 2 exceeds limit 2."


def test_the_collision_limit_of_an_rqvae_is_its_codebook_size(tmp_path):
    # codebook of 2 vectors -> limit 2; three identical items get collision tokens 0, 1, 2;
    # token 2 is not below the limit
    with pytest.raises(ValueError) as error:
        _run_generation_on_three_identical_items(_rqvae_config(), tmp_path)
    assert str(error.value) == "Collision overflow! Max depth 2 exceeds limit 2."
