"""
Tests of the residual update in modules/rqvae/quantization.py (ResidualVectorQuantizer):
which commitment terms reach the encoder, with and without commit_all_levels.

Fixture (hand-checkable, 2 levels, latent dim 2, beta = 0.25, codebooks set by hand):
  x                    = [1, 0]
  codebook 1           = {[0.5, 0], [5, 5]}  -> picks c1 = [0.5, 0]
  residual r2 = x - c1 = [0.5, 0]
  codebook 2           = {[0.25, 0], [5, 5]} -> picks c2 = [0.25, 0]
  quantized c1 + c2    = [0.75, 0]

  level | squared error ||r - c||^2 | level loss = (1 + beta) * error | commitment grad wrt x = 2 * beta * (r - c)
  1     | ||[0.5, 0]||^2  = 0.25    | 0.3125                          | [0.25, 0]
  2     | ||[0.25, 0]||^2 = 0.0625  | 0.078125                        | [0.125, 0], only if r2 depends on x
  total loss = 0.390625 = codebook terms (0.25 + 0.0625) + commitment terms (0.0625 + 0.015625)
  codebook-1 grad = d/dc1 ||sg[x] - c1||^2 = 2 * (c1 - x) = [-1, 0] for the picked row, [0, 0] for the other
All values are exact in binary floating point, so the assertions use ==.
"""
import pytest
import torch

from modules.rqvae.quantization import QuantizeLoss, ResidualVectorQuantizer
from schemas.quantization import QuantizeForwardMode, QuantizeDistance


def _quantizer(commit_all_levels, forward_mode=QuantizeForwardMode.STE, beta=0.25):
    rvq = ResidualVectorQuantizer(codebook_layers=2, latent_dim=2, codebook_size=2, beta=beta,
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
    # only level 1 contributes 2 * beta * (x - c1) = [0.25, 0]
    x = _x()
    _quantizer(commit_all_levels=False)(x).loss.backward()
    assert torch.equal(x.grad, torch.tensor([[0.25, 0.0]]))


def test_commit_all_levels_every_commitment_reaches_encoder():
    # level 1 [0.25, 0] + level 2 [0.125, 0]
    x = _x()
    _quantizer(commit_all_levels=True)(x).loss.backward()
    assert torch.equal(x.grad, torch.tensor([[0.375, 0.0]]))


def test_commit_all_levels_codebook1_gets_only_its_own_codebook_term():
    # the level-2 terms must not move c1 (the residual subtracts the detached codeword)
    rvq = _quantizer(commit_all_levels=True)
    rvq(_x()).loss.backward()
    assert torch.equal(rvq.quantization_layers[0].embedding.weight.grad, torch.tensor([[-1.0, 0.0], [0.0, 0.0]]))


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
    assert out.loss.item() == 0.390625


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


def test_quantize_loss_sums_over_dimensions_and_averages_over_the_batch():
    # TIGER's squared norms: per item ||z - z_hat||^2 = 1 and 0, batch mean 0.5 (a mean over the 2 dimensions
    # would give 0.25); commitment term = beta * 0.5 = 0.25, returned separately for logging
    codebook, commitment = QuantizeLoss(beta=0.5)(z=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                                                  z_hat=torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
    assert (codebook.item(), commitment.item()) == (0.5, 0.25)


@pytest.mark.parametrize("commit_all_levels", [False, True])
def test_codebook_loss_per_level(commit_all_levels):
    # the option changes gradients only, so the logged values are the table's squared errors
    out = _quantizer(commit_all_levels)(_x())
    assert torch.equal(out.metrics["codebook_loss"], torch.tensor([0.25, 0.0625]))


@pytest.mark.parametrize("commit_all_levels", [False, True])
def test_commitment_loss_per_level_is_weighted_by_beta(commit_all_levels):
    # beta * squared error per level: 0.25 * 0.25 and 0.25 * 0.0625
    out = _quantizer(commit_all_levels)(_x())
    assert torch.equal(out.metrics["commitment_loss"], torch.tensor([0.0625, 0.015625]))


def test_logged_loss_terms_add_up_to_the_optimized_loss():
    out = _quantizer(commit_all_levels=False)(_x())
    assert torch.equal(out.metrics["codebook_loss"].sum() + out.metrics["commitment_loss"].sum(), out.loss)


def test_commitment_loss_is_zero_without_beta():
    out = _quantizer(commit_all_levels=False, beta=0.0)(_x())
    assert torch.equal(out.metrics["commitment_loss"], torch.tensor([0.0, 0.0]))


def test_codebook_loss_does_not_depend_on_beta():
    out = _quantizer(commit_all_levels=False, beta=0.0)(_x())
    assert torch.equal(out.metrics["codebook_loss"], torch.tensor([0.25, 0.0625]))


def test_logged_loss_terms_carry_no_graph():
    # the epoch loop adds these to running sums; a graph would stay alive across the epoch
    out = _quantizer(commit_all_levels=False)(_x())
    assert not out.metrics["codebook_loss"].requires_grad
    assert not out.metrics["commitment_loss"].requires_grad


def test_latent_norm_is_the_norm_of_the_encoder_output_not_of_a_residual():
    # x = [1, 0] has norm 1; the level-2 residual [0.5, 0] would give 0.5
    assert _quantizer(commit_all_levels=False)(_x()).metrics["latent_norm"].item() == 1.0
