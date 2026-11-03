"""
Integration tests for the quantizer training script (train_quantizer.py): model construction from the config.
"""
from omegaconf import OmegaConf

from train_quantizer import create_model


def test_create_model_passes_commit_all_levels_to_the_quantizer():
    # the default is False, so True shows that the config value arrives
    config = OmegaConf.create({"model": {
        "quantizer_type": "rqvae", "latent_dimension": 2, "hidden_dimensions": [4], "codebook_layers": 2,
        "codebook_size": 2, "commitment_weight": 0.25, "quantization_method": "ste", "distance_method": "l2",
        "normalize_inputs": False, "commit_all_levels": True}})
    model = create_model(config, input_dim=8)
    assert model.quantizer.commit_all_levels is True
