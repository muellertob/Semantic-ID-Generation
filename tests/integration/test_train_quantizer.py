"""
Integration tests for the quantizer training script (train_quantizer.py): model construction from the config.
"""
import pytest
from omegaconf import OmegaConf

from train_quantizer import create_model


def _rqvae_config(**model):
    return OmegaConf.create({"model": {
        "quantizer_type": "rqvae", "latent_dimension": 2, "hidden_dimensions": [4], "codebook_layers": 2,
        "codebook_size": 2, "beta": 0.25, "quantization_method": "ste", "distance_method": "l2",
        "normalize_inputs": False, "commit_all_levels": False, **model}})


def test_create_model_passes_commit_all_levels_to_the_quantizer():
    # the default is False, so True shows that the config value arrives
    model = create_model(_rqvae_config(commit_all_levels=True), input_dim=8)
    assert model.quantizer.commit_all_levels is True


def test_create_model_passes_beta_to_every_level():
    # the default is 0.25, so 3.0 shows that the config value arrives
    model = create_model(_rqvae_config(beta=3.0), input_dim=8)
    assert [layer.quantize_loss.beta for layer in model.quantizer.quantization_layers] == [3.0, 3.0]


def test_create_model_rejects_the_former_commitment_weight():
    # a stored config with the old key would otherwise train with a weight input_dim / latent_dim times too small
    # (24 in the real configs); the message gives the equivalent beta: 0.25 * input_dim 8 / latent_dim 2 = 1.0
    with pytest.raises(ValueError) as error:
        create_model(_rqvae_config(commitment_weight=0.25), input_dim=8)
    assert str(error.value) == (
        "model.commitment_weight (loss averaged over dimensions) was replaced by model.beta (TIGER's loss, "
        "summed over dimensions); the same weighting of the encoder terms is "
        "model.beta = commitment_weight * input_dim / latent_dim = 1.0")
