"""
Tests of the FSQ factories in modules/fsq/model.py.
"""
import pytest

from modules.fsq.model import FSQ_AutoEncoder, ResidualFSQ_AutoEncoder, FixedScaleResidualFSQ_AutoEncoder
from modules.fsq.quantization import FixedScaleResidualFSQ


def test_fsq_keeps_the_reconstruction_averaged_over_dimensions():
    # FSQ has no beta, so its loss scale stays as in all earlier FSQ runs
    model = FSQ_AutoEncoder(input_dim=8, codebook_layers=3, hidden_dims=[4], latent_dim=3, level_list=[8, 6, 6])
    assert model.sum_over_dims is False


def test_residual_fsq_keeps_the_reconstruction_averaged_over_dimensions():
    model = ResidualFSQ_AutoEncoder(input_dim=8, codebook_layers=3, hidden_dims=[4], latent_dim=3,
                                    level_list=[8, 6, 6], inner_dim=4)
    assert model.sum_over_dims is False


def test_fixed_scale_residual_fsq_passes_level_list_and_number_of_levels():
    # the default is 3 levels, so 2 shows that the value arrives
    model = FixedScaleResidualFSQ_AutoEncoder(input_dim=8, codebook_layers=2, hidden_dims=[4], latent_dim=2,
                                              level_list=[3, 2])
    assert type(model.quantizer) is FixedScaleResidualFSQ
    assert model.quantizer.level_list == [3, 2]
    assert model.quantizer.codebook_layers == 2


def test_fixed_scale_residual_fsq_rejects_a_latent_dimension_that_differs_from_the_number_of_channels():
    # without projections the mismatch would otherwise only fail as a shape error in the first batch
    with pytest.raises(ValueError) as error:
        FixedScaleResidualFSQ_AutoEncoder(input_dim=8, codebook_layers=3, hidden_dims=[4], latent_dim=32,
                                          level_list=[8, 6, 6])
    assert str(error.value) == ("latent_dim must equal len(level_list) = 3 (one latent dimension per FSQ channel), "
                                "got 32")


def test_fixed_scale_residual_fsq_keeps_the_reconstruction_averaged_over_dimensions():
    model = FixedScaleResidualFSQ_AutoEncoder(input_dim=8, codebook_layers=3, hidden_dims=[4], latent_dim=3,
                                              level_list=[8, 6, 6])
    assert model.sum_over_dims is False
