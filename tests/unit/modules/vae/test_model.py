"""
Tests of the reconstruction loss of QuantizedAutoEncoder in modules/vae/model.py.

Fixture (hand-checkable; encoder and decoder are the identity, no normalization, the quantizer maps every latent
to zero with zero loss, so x_hat = 0):
  x = [[1, 0, 0, 0], [0, 0, 0, 0]]
  error per entry: one entry 1, seven entries 0
  summed over the 4 dimensions per item: 1 and 0, batch mean 0.5
  averaged over all 8 entries: 0.125
All values are exact in binary floating point, so the assertions use ==.
"""
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from modules.vae.model import QuantizedAutoEncoder
from schemas.quantization import QuantizeOutput

X = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])


def _zero_autoencoder(loss_type, sum_over_dims):
    quantizer = Mock(return_value=QuantizeOutput(embeddings=torch.zeros(2, 4), ids=torch.zeros(2, 1),
                                                 loss=torch.tensor(0.0)))
    return QuantizedAutoEncoder(encoder=nn.Identity(), decoder=nn.Identity(), quantizer=quantizer, input_dim=4,
                                latent_dim=4, loss_type=loss_type, normalize=False, sum_over_dims=sum_over_dims)


@pytest.mark.parametrize("loss_type", ["mse", "l1"])
def test_reconstruction_sums_over_dimensions_and_averages_over_the_batch(loss_type):
    # TIGER's squared norm per item: 1 and 0, batch mean 0.5 (|1| = 1^2, so l1 gives the same)
    out = _zero_autoencoder(loss_type, sum_over_dims=True)(X)
    assert out.reconstruction_loss.item() == 0.5


@pytest.mark.parametrize("loss_type", ["mse", "l1"])
def test_reconstruction_averages_over_all_entries_without_sum_over_dims(loss_type):
    # FSQ's reduction: 1 / 8
    out = _zero_autoencoder(loss_type, sum_over_dims=False)(X)
    assert out.reconstruction_loss.item() == 0.125
