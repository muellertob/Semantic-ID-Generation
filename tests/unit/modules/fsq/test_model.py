"""
Tests of the FSQ factories in modules/fsq/model.py.
"""
from modules.fsq.model import FSQ_AutoEncoder, ResidualFSQ_AutoEncoder


def test_fsq_keeps_the_reconstruction_averaged_over_dimensions():
    # FSQ has no beta, so its loss scale stays as in all earlier FSQ runs
    model = FSQ_AutoEncoder(input_dim=8, codebook_layers=3, hidden_dims=[4], latent_dim=3, level_list=[8, 6, 6])
    assert model.sum_over_dims is False


def test_residual_fsq_keeps_the_reconstruction_averaged_over_dimensions():
    model = ResidualFSQ_AutoEncoder(input_dim=8, codebook_layers=3, hidden_dims=[4], latent_dim=3,
                                    level_list=[8, 6, 6], inner_dim=4)
    assert model.sum_over_dims is False
