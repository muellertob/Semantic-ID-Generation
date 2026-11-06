"""
Tests of the RQ_VAE factory in modules/rqvae/model.py.
"""
from modules.rqvae.model import RQ_VAE


def test_rqvae_sums_the_reconstruction_over_dimensions():
    # beta is defined on TIGER's summed loss, so the reconstruction must use the same scale
    model = RQ_VAE(input_dim=8, latent_dim=2, hidden_dims=[4], codebook_size=2, codebook_layers=2)
    assert model.sum_over_dims is True
