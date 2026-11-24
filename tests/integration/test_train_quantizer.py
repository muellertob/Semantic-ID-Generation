"""
Integration tests for the quantizer training script (train_quantizer.py): model construction from the config and
the batches of the training loop.
"""
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from modules.fsq.quantization import FixedScaleResidualFSQ
from train_quantizer import create_model, train


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


def _rfsq_fixed_config(**model):
    return OmegaConf.create({"model": {
        "quantizer_type": "rfsq_fixed", "latent_dimension": 2, "hidden_dimensions": [4], "codebook_layers": 2,
        "level_list": [3, 2], "loss_type": "l1", "normalize_inputs": False, **model}})


def test_create_model_builds_the_fixed_scale_quantizer_with_level_list_and_number_of_levels():
    # the default is 3 levels, so 2 shows that the config value arrives
    model = create_model(_rfsq_fixed_config(), input_dim=8)
    assert type(model.quantizer) is FixedScaleResidualFSQ
    assert list(model.quantizer.level_list) == [3, 2]
    assert model.quantizer.codebook_layers == 2


def test_create_model_passes_the_loss_type_for_the_fixed_scale_quantizer():
    # the default is "mse"; a silent "mse" would change the numbers
    model = create_model(_rfsq_fixed_config(), input_dim=8)
    assert model.loss_type == "l1"


def test_the_shipped_fixed_scale_config_builds_with_288_values_per_level():
    # the model is only built if latent_dimension and level_list in the file fit together; 8 * 6 * 6 = 288
    model = create_model(OmegaConf.load("config/fsq/amazon_toy_rfsq_fixed_gts.yaml"), input_dim=768)
    assert model.quantizer.codebook_size == 288


def test_create_model_passes_the_quantizer_dropout_to_the_fixed_scale_quantizer():
    # the default is 0, so 0.5 shows that the config value arrives
    model = create_model(_rfsq_fixed_config(quantizer_dropout=0.5), input_dim=8)
    assert model.quantizer.quantizer_dropout == 0.5


def test_create_model_rejects_quantizer_dropout_for_another_quantizer_type():
    # the run would otherwise train without dropout under a config that claims it
    config = OmegaConf.create({"model": {
        "quantizer_type": "rfsq", "latent_dimension": 2, "hidden_dimensions": [4], "codebook_layers": 2,
        "level_list": [3, 2], "inner_dim": 4, "normalize_inputs": False, "quantizer_dropout": 0.5}})
    with pytest.raises(ValueError) as error:
        create_model(config, input_dim=8)

    assert str(error.value) == "model.quantizer_dropout is only implemented for quantizer_type 'rfsq_fixed', got 'rfsq'"


def _train_one_epoch(n_items, batch_size):
    """Trains one epoch on n_items random items with the input BatchNorm on; returns the mock that wraps the forward pass."""
    model = create_model(_rfsq_fixed_config(normalize_inputs=True), input_dim=8)
    config = OmegaConf.create({"data": {"batch_size": batch_size}, "general": {"seed": 42, "use_wandb": False},
                               "model": {"quantizer_type": "rfsq_fixed"}, "train": {}})
    with patch.object(model, "forward", wraps=model.forward) as forward:
        train(model, torch.randn(n_items, 8), torch.optim.Adam(model.parameters()), 1, torch.device("cpu"), config)
    return forward


# items = full batches + left over; the left-over items are not trained on
#   5 = 2 + 2 + 1: the Onion case (one item left, the input BatchNorm raises)
#   5 = 3 + 2:     two items left, whose BatchNorm statistics degenerate
#   4 = 2 + 2:     nothing left
@pytest.mark.parametrize("n_items, batch_size, batch_sizes", [(5, 2, [2, 2]), (5, 3, [3]), (4, 2, [2, 2])])
def test_train_uses_full_batches_only(n_items, batch_size, batch_sizes):
    forward = _train_one_epoch(n_items, batch_size)
    assert [call.args[0].shape[0] for call in forward.call_args_list] == batch_sizes


def test_train_rejects_a_batch_size_larger_than_the_data():
    # with full batches only, the epoch would have no step and the Semantic IDs would come from an untrained model
    with pytest.raises(ValueError) as error:
        _train_one_epoch(n_items=2, batch_size=3)
    assert str(error.value) == ("data.batch_size is 3 but there are only 2 items to train on: "
                                "with full batches only, an epoch would have no training step")


@pytest.mark.parametrize("path", ["config/fsq/amazon_toy_gts.yaml", "config/fsq/amazon_toy_rfsq_fixed_gts.yaml",
                                  "config/fsq/onion_gts.yaml", "config/fsq/onion_rfsq_fixed_gts.yaml"])
def test_shipped_fsq_configs_train_with_the_batch_size_of_rq_vae(path):
    # FSQ and residual FSQ use the 1024 of RQ-VAE on both datasets (the earlier 256 is no longer the setting)
    assert OmegaConf.load(path).data.batch_size == 1024
