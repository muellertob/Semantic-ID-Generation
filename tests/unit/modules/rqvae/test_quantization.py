"""
Tests of the residual update in modules/rqvae/quantization.py (ResidualVectorQuantizer):
which commitment terms reach the encoder, with and without commit_all_levels.

Fixture (hand-checkable, 2 levels, latent dim 2, beta = 0.25, codebooks set by hand):
  x                    = [1, 0]
  codebook 1           = {[0.5, 0], [5, 5]}  -> picks c1 = [0.5, 0]
  residual r2 = x - c1 = [0.5, 0]
  codebook 2           = {[0.25, 0], [5, 5]} -> picks c2 = [0.25, 0]
  quantized c1 + c2    = [0.75, 0]

  level | MSE (mean over 2 dims)   | level loss = MSE + beta * MSE | commitment grad wrt x = beta * (r - c)
  1     | ([0.5,0])^2  / 2 = 0.125 | 0.15625                       | [0.125, 0]
  2     | ([0.25,0])^2 / 2 = 0.03125 | 0.0390625                   | [0.0625, 0], only if r2 depends on x
  total loss = 0.1953125
  codebook-1 grad = d/dc1 MSE(sg[x], c1) = c1 - x = [-0.5, 0] for the picked row, [0, 0] for the other
All values are exact in binary floating point, so the assertions use ==.
"""
import pytest
import torch

from modules.rqvae.quantization import ResidualVectorQuantizer
from schemas.quantization import QuantizeForwardMode, QuantizeDistance


def _quantizer(commit_all_levels, forward_mode=QuantizeForwardMode.STE):
    rvq = ResidualVectorQuantizer(codebook_layers=2, latent_dim=2, codebook_size=2, commitment_weight=0.25,
                                  do_kmeans_init=False, forward_mode=forward_mode,
                                  distance_mode=QuantizeDistance.L2, commit_all_levels=commit_all_levels)
    with torch.no_grad():
        rvq.quantization_layers[0].embedding.weight.copy_(torch.tensor([[0.5, 0.0], [5.0, 5.0]]))
        rvq.quantization_layers[1].embedding.weight.copy_(torch.tensor([[0.25, 0.0], [5.0, 5.0]]))
    return rvq


def _x():
    return torch.tensor([[1.0, 0.0]], requires_grad=True)


def test_default_level2_commitment_does_not_reach_encoder():
    # TIGER's update subtracts the straight-through output, so r2 has no gradient:
    # only level 1 contributes beta * (x - c1) = [0.125, 0]
    x = _x()
    _quantizer(commit_all_levels=False)(x).loss.backward()
    assert torch.equal(x.grad, torch.tensor([[0.125, 0.0]]))


def test_commit_all_levels_every_commitment_reaches_encoder():
    # level 1 [0.125, 0] + level 2 [0.0625, 0]
    x = _x()
    _quantizer(commit_all_levels=True)(x).loss.backward()
    assert torch.equal(x.grad, torch.tensor([[0.1875, 0.0]]))


def test_commit_all_levels_codebook1_gets_only_its_own_codebook_term():
    # the level-2 terms must not move c1 (the residual subtracts the detached codeword)
    rvq = _quantizer(commit_all_levels=True)
    rvq(_x()).loss.backward()
    assert torch.equal(rvq.quantization_layers[0].embedding.weight.grad, torch.tensor([[-0.5, 0.0], [0.0, 0.0]]))


@pytest.mark.parametrize("commit_all_levels", [False, True])
def test_decoder_path_gradient_is_identity(commit_all_levels):
    # one straight-through step in total: d(quantized)/dx = I, not (number of levels) * I
    x = _x()
    _quantizer(commit_all_levels)(x).embeddings.sum().backward()
    assert torch.equal(x.grad, torch.tensor([[1.0, 1.0]]))


@pytest.mark.parametrize("commit_all_levels", [False, True])
def test_forward_values_do_not_depend_on_the_option(commit_all_levels):
    # the option changes gradients only; values from the table in the module docstring
    out = _quantizer(commit_all_levels)(_x())
    assert torch.equal(out.ids, torch.tensor([[0, 0]]))
    assert torch.equal(out.embeddings, torch.tensor([[0.75, 0.0]]))
    assert out.loss.item() == 0.1953125


def test_commit_all_levels_eval_returns_sum_of_codewords():
    # SID generation runs in eval mode: no straight-through step, c1 + c2
    rvq = _quantizer(commit_all_levels=True).eval()
    assert torch.equal(rvq(_x()).embeddings, torch.tensor([[0.75, 0.0]]))


def test_commit_all_levels_rejects_gumbel_at_construction():
    # Gumbel subtracts soft embeddings; the combination is not defined
    with pytest.raises(ValueError, match="commit_all_levels requires STE"):
        _quantizer(commit_all_levels=True, forward_mode=QuantizeForwardMode.GUMBEL_SOFTMAX)


def test_commit_all_levels_rejects_switch_to_gumbel():
    rvq = _quantizer(commit_all_levels=True)
    with pytest.raises(ValueError, match="commit_all_levels requires STE"):
        rvq.set_quantization_method(QuantizeForwardMode.GUMBEL_SOFTMAX)
