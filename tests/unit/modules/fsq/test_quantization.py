import math
from unittest.mock import patch

import pytest
import torch
from modules.fsq.quantization import FSQ, round_ste, ProjectionBlock, FSQLayer, ResidualFSQ, FixedScaleResidualFSQ
from schemas.quantization import QuantizeOutput


# ==============================================================================
# Low-Level Core Math Operations
# ==============================================================================

def test_round_ste():
    x = torch.tensor([1.2, -0.6, 2.8], requires_grad=True)
    out = round_ste(x)
    
    assert torch.equal(out, torch.tensor([1.0, -1.0, 3.0]))
    
    # Check STE pass-through
    out.sum().backward()
    assert torch.equal(x.grad, torch.tensor([1.0, 1.0, 1.0]))


def test_fsq_core_indices_to_codes_invertible():
    fsq = FSQ(level_list=[8, 6, 5])
    # The max combination is 8*6*5 = 240
    indices = torch.tensor([0, 1, 10, 239])
    
    codes = fsq.indices_to_codes(indices)
    assert codes.shape == (4, 3)
    
    # Verify that indices_to_codes and codes_to_indices are exact mathematical inverses
    indices_recovered = fsq.codes_to_indices(codes)
    assert torch.equal(indices, indices_recovered)


# ==============================================================================
# Intermediate Components
# ==============================================================================

def test_projection_block():
    # Test all valid projection block types and shapes
    shapes_to_test = {
        "identity": (128, 128),
        "linear": (128, 32),
        "mlp_1_hidden": (128, 32),
        "mlp_2_hidden": (128, 32)
    }
    
    for proj_type, (dim_in, dim_out) in shapes_to_test.items():
        block = ProjectionBlock(dim_in=dim_in, inner_dim=64, dim_out=dim_out, projection_type=proj_type)
        x = torch.randn(16, dim_in)
        out = block(x)
        assert out.shape == (16, dim_out), f"Shape mismatch for projection_type={proj_type}"
        
        # Additional identity behavior check (values must be identical)
        if proj_type == "identity":
            assert torch.equal(out, x)
            
    # Test invalid projection type raises ValueError
    with pytest.raises(ValueError, match="Unknown projection_type"):
        ProjectionBlock(dim_in=128, inner_dim=64, dim_out=32, projection_type="invalid_type")


def test_fsq_layer():
    layer = FSQLayer(dim=256, level_list=[8, 6, 5], inner_dim=128, projection_type="mlp_1_hidden")
    x = torch.randn(16, 256)
    out, indices = layer(x)
    assert out.shape == (16, 256)
    assert indices.shape == (16,)


# ==============================================================================
# FSQ (Flat Quantizer)
# ==============================================================================

def test_fsq_quantization_bounds():
    fsq = FSQ(level_list=[8, 6, 5])
    z = torch.randn(10, 3)
    output = fsq(z)
    
    assert isinstance(output, QuantizeOutput)
    assert output.embeddings.shape == (10, 3)
    assert output.ids.shape == (10, 1)
    assert output.loss.item() == 0.0


def test_fsq_quantized_values_range():
    # Extreme inputs should map to bounds strictly in range [-1.0, 1.0]
    fsq = FSQ(level_list=[8, 6, 5])
    
    # Extreme positive
    z_pos = torch.ones(10, 3) * 100.0
    output_pos = fsq(z_pos)
    assert torch.all(output_pos.embeddings >= -1.0)
    assert torch.all(output_pos.embeddings <= 1.0)
    
    # Extreme negative
    z_neg = torch.ones(10, 3) * -100.0
    output_neg = fsq(z_neg)
    assert torch.all(output_neg.embeddings >= -1.0)
    assert torch.all(output_neg.embeddings <= 1.0)


def test_fsq_latent_space_chunking():
    # Test FSQ with codebook_layers = 4 and level_list of size 3 (latent_dim = 12)
    fsq = FSQ(dim=12, codebook_layers=4, level_list=[8, 6, 5])
    assert fsq.codebook_size == 240
    
    z = torch.randn(5, 12)
    output = fsq(z)
    
    assert output.embeddings.shape == (5, 12)
    assert output.ids.shape == (5, 4)  # 4 sequence chunks
    assert torch.all(output.ids >= 0)
    assert torch.all(output.ids < 240)


def test_fsq_projection_types():
    projection_types = ["identity", "linear", "mlp_1_hidden", "mlp_2_hidden"]
    for proj_type in projection_types:
        fsq = FSQ(
            dim=12,
            codebook_layers=4,
            level_list=[8, 6, 5],
            projection_type=proj_type,
            inner_dim=16
        )
        z = torch.randn(5, 12, requires_grad=True)
        output = fsq(z)
        
        assert output.embeddings.shape == (5, 12)
        assert output.ids.shape == (5, 4)
        
        # Placeholder loss to check gradient flow
        loss = output.embeddings.sum()
        loss.backward()
        assert z.grad is not None, f"Gradient did not flow back to input for projection_type={proj_type}"


# ==============================================================================
# ResidualFSQ (Hierarchical Quantizer)
# ==============================================================================

def test_residual_fsq_quantization():
    codebook_layers = 4
    quantizer = ResidualFSQ(
        dim=256, level_list=[8, 6, 5], codebook_layers=codebook_layers, inner_dim=128, projection_type="mlp_1_hidden"
    )
    x = torch.randn(16, 256, requires_grad=True)
    out = quantizer(x)
    
    assert out.embeddings.shape == (16, 256)
    assert out.ids.shape == (16, codebook_layers)
    
    # Placeholder loss to check gradient flow
    loss = out.embeddings.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input"


def test_residual_fsq_metrics_and_zero_input():
    codebook_layers = 3
    quantizer = ResidualFSQ(
        dim=64, level_list=[8, 6, 5], codebook_layers=codebook_layers, inner_dim=32, projection_type="mlp_1_hidden"
    )
    
    # Test with standard inputs
    x = torch.randn(8, 64)
    out = quantizer(x)
    
    assert "p_unique_ids" in out.metrics
    assert "layer_coverages" in out.metrics
    assert "layer_entropies" in out.metrics
    assert "first_residual_norm" in out.metrics
    assert "last_residual_norm" in out.metrics
    assert "first_residual_rel" in out.metrics
    assert "last_residual_rel" in out.metrics
    
    # Test with zero input (division by zero / numerical stability check)
    x_zero = torch.zeros(8, 64)
    out_zero = quantizer(x_zero)
    # Norms should be zero, and relative ratios should not be NaN
    assert not torch.isnan(out_zero.metrics["first_residual_rel"]).any()
    assert not torch.isnan(out_zero.metrics["last_residual_rel"]).any()
    assert out_zero.metrics["first_residual_rel"].item() == 0.0
    assert out_zero.metrics["last_residual_rel"].item() == 0.0


# ==============================================================================
# FixedScaleResidualFSQ (parameter-free residual FSQ in the latent)
# ==============================================================================

def _latent_for(inputs, level_list):
    """Latent whose soft clamp c * tanh(z / c), c = 1 + 1 / (L - 1), equals `inputs`."""
    c = 1 + 1 / (torch.tensor(level_list, dtype=torch.float32) - 1)
    return c * torch.atanh(torch.tensor(inputs) / c)


def _assert_values_close(actual, expected):
    """Whole-tensor comparison with an absolute tolerance for float32 rounding."""
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=1e-6)


def test_fixed_scale_residual_fsq_each_level_refines_the_previous_one():
    # input 0.8, L = 3 (grid {-1, 0, 1}, scales 1, 1/3, 1/9)
    # level 1: nearest grid value is 1,    rest 0.8 - 1 = -0.2
    # level 2: nearest of {-1/3, 0, 1/3} to -0.2 is -1/3, rest -0.2 + 1/3 = 0.1333
    # level 3: 0.1333 / (1/9) = 1.2, clamped to 1 -> 1/9
    # embedding 1 - 1/3 + 1/9 = 7/9; digits 2, 0, 2
    quantizer = FixedScaleResidualFSQ(level_list=[3], codebook_layers=3)
    out = quantizer(_latent_for([[0.8]], [3]))

    _assert_values_close(out.embeddings, [[7 / 9]])
    assert torch.equal(out.ids, torch.tensor([[2, 0, 2]], dtype=torch.int32))


def test_fixed_scale_residual_fsq_levels_tile_the_range_without_gaps():
    # two levels with L = 3 give the nine values k/3, k = -4..4; each input goes to the nearest one
    # input   embedding  ids
    # -1.4    -4/3       [0, 0]
    # -0.6    -2/3       [0, 2]
    #  0.2     1/3       [1, 2]
    #  0.45    1/3       [1, 2]
    #  1.3     4/3       [2, 2]
    quantizer = FixedScaleResidualFSQ(level_list=[3], codebook_layers=2)
    out = quantizer(_latent_for([[-1.4], [-0.6], [0.2], [0.45], [1.3]], [3]))

    _assert_values_close(out.embeddings, [[-4 / 3], [-2 / 3], [1 / 3], [1 / 3], [4 / 3]])
    assert torch.equal(out.ids, torch.tensor([[0, 0], [0, 2], [1, 2], [1, 2], [2, 2]], dtype=torch.int32))


def test_fixed_scale_residual_fsq_large_latents_saturate_at_the_outermost_value():
    # the largest representable value is 1 + 1/3 + 1/9 = 13/9, all levels at the edge of the grid
    quantizer = FixedScaleResidualFSQ(level_list=[3], codebook_layers=3)
    out = quantizer(torch.tensor([[100.0], [-100.0]]))

    _assert_values_close(out.embeddings, [[13 / 9], [-13 / 9]])
    assert torch.equal(out.ids, torch.tensor([[2, 2, 2], [0, 0, 0]], dtype=torch.int32))


def test_fixed_scale_residual_fsq_index_has_first_channel_as_lowest_digit():
    # L = [3, 2]: grid of channel 1 is {-1, 0, 1}, grid of channel 2 is {-1, 1}
    # input          embedding  digits  index = digit_1 + 3 * digit_2
    # (0.8, -0.3)    (1, -1)    (2, 0)  2
    # (-0.9, 0.4)    (-1, 1)    (0, 1)  3
    quantizer = FixedScaleResidualFSQ(level_list=[3, 2], codebook_layers=1)
    out = quantizer(_latent_for([[0.8, -0.3], [-0.9, 0.4]], [3, 2]))

    _assert_values_close(out.embeddings, [[1.0, -1.0], [-1.0, 1.0]])
    assert torch.equal(out.ids, torch.tensor([[2], [3]], dtype=torch.int32))


def test_fixed_scale_residual_fsq_encoder_gradient_is_counted_once():
    # input 0.2, L = 3: both levels are inside their hard clamp; the residual carries the gradient,
    # so the summed output has derivative 1 w.r.t. the input, times the soft-clamp derivative
    # 1 - (0.2 / 1.5)^2 = 0.982222 (a detached residual would double the 1)
    quantizer = FixedScaleResidualFSQ(level_list=[3], codebook_layers=2)
    latent = _latent_for([[0.2]], [3]).requires_grad_()
    quantizer(latent).embeddings.sum().backward()

    _assert_values_close(latent.grad, [[1 - (0.2 / 1.5) ** 2]])


def test_fixed_scale_residual_fsq_gradient_is_zero_beyond_the_largest_representable_value():
    # input 1.48, L = 3: level 1 clamps (1.48 > 1, rest 0.48), level 2 clamps (0.48 > 1/3, rest 0.1467),
    # level 3 clamps (0.1467 > 1/9); the output already is at its maximum 13/9, so no gradient flows
    quantizer = FixedScaleResidualFSQ(level_list=[3], codebook_layers=3)
    latent = _latent_for([[1.48]], [3]).requires_grad_()
    quantizer(latent).embeddings.sum().backward()

    assert torch.equal(latent.grad, torch.tensor([[0.0]]))


def test_fixed_scale_residual_fsq_zero_latent_goes_up_for_an_even_number_of_values():
    # L = 2, grid {-1, 1}: 0 lies exactly between both values and ties go up
    quantizer = FixedScaleResidualFSQ(level_list=[2], codebook_layers=1)
    out = quantizer(torch.tensor([[0.0]]))

    _assert_values_close(out.embeddings, [[1.0]])
    assert torch.equal(out.ids, torch.tensor([[1]], dtype=torch.int32))


def test_fixed_scale_residual_fsq_has_no_trainable_parameters():
    quantizer = FixedScaleResidualFSQ(level_list=[8, 6, 6], codebook_layers=3)

    assert list(quantizer.parameters()) == []


def test_fixed_scale_residual_fsq_has_no_quantization_loss():
    quantizer = FixedScaleResidualFSQ(level_list=[3], codebook_layers=3)
    out = quantizer(_latent_for([[0.8]], [3]))

    assert out.loss.item() == 0.0


def test_fixed_scale_residual_fsq_codebook_size_is_the_product_of_the_values_per_channel():
    # 8 * 6 * 6 = 288, 2^8 = 256
    assert FixedScaleResidualFSQ(level_list=[8, 6, 6]).codebook_size == 288
    assert FixedScaleResidualFSQ(level_list=[2] * 8).codebook_size == 256


def test_fixed_scale_residual_fsq_residual_norms_are_relative_to_the_soft_clamped_input():
    # input 0.8, L = 3: first residual |0.8 - 1| = 0.2 -> 0.2 / 0.8 = 0.25
    # last residual 0.8 - 7/9 = 0.022222 -> 0.022222 / 0.8 = 0.027778
    quantizer = FixedScaleResidualFSQ(level_list=[3], codebook_layers=3)
    out = quantizer(_latent_for([[0.8]], [3]))

    _assert_values_close(out.metrics["first_residual_rel"], 0.2 / 0.8)
    _assert_values_close(out.metrics["last_residual_rel"], (0.8 - 7 / 9) / 0.8)


def test_fixed_scale_residual_fsq_metric_names_match_residual_fsq():
    reference = ResidualFSQ(dim=4, level_list=[3], codebook_layers=3, inner_dim=4, projection_type="mlp_1_hidden")
    reference_metrics = reference(torch.zeros(2, 4)).metrics
    out = FixedScaleResidualFSQ(level_list=[3], codebook_layers=3)(torch.zeros(2, 1))

    assert set(out.metrics) == set(reference_metrics)


# Quantizer dropout. Three examples, all with soft-clamped input -0.6, L = 3, 3 levels:
# level   grid (scaled)          residual in   output   digit
# 1       {-1, 0, 1}             -0.6          -1       0
# 2       {-1/3, 0, 1/3}          0.4           1/3     2
# 3       {-1/9, 0, 1/9}          1/15          1/9     2
# sum of the first n levels: n=1: -1, n=2: -2/3, n=3: -5/9
def _dropout_latent():
    return _latent_for([[-0.6], [-0.6], [-0.6]], [3])


def _draws(rand, n):
    return (patch("torch.rand", return_value=torch.tensor(rand).reshape(3, 1)),
            patch("torch.randint", return_value=torch.tensor(n).reshape(3, 1)))


def test_quantizer_dropout_sums_only_the_first_n_levels_of_a_selected_example():
    # p = 1 selects every example, so each one gets the sum of its first n levels
    rand_draw, n_draw = _draws([0.0, 0.0, 0.0], [1, 2, 3])
    with rand_draw, n_draw:
        out = FixedScaleResidualFSQ([3], codebook_layers=3, quantizer_dropout=1.0)(_dropout_latent())

    _assert_values_close(out.embeddings, [[-1.0], [-2 / 3], [-5 / 9]])


def test_quantizer_dropout_shortens_only_examples_whose_random_number_is_below_p():
    # random number < 0.5 selects: examples 1 and 3 are selected (n = 1, 2); example 2 is not, so it keeps all
    # levels although its n is 1
    rand_draw, n_draw = _draws([0.4, 0.6, 0.4], [1, 1, 2])
    with rand_draw, n_draw:
        out = FixedScaleResidualFSQ([3], codebook_layers=3, quantizer_dropout=0.5)(_dropout_latent())

    _assert_values_close(out.embeddings, [[-1.0], [-5 / 9], [-2 / 3]])


def test_quantizer_dropout_does_not_change_the_ids():
    # the Semantic ID must not depend on the draw: every example has digits 0, 2, 2
    rand_draw, n_draw = _draws([0.0, 0.0, 0.0], [1, 2, 3])
    with rand_draw, n_draw:
        out = FixedScaleResidualFSQ([3], codebook_layers=3, quantizer_dropout=1.0)(_dropout_latent())

    assert torch.equal(out.ids, torch.tensor([[0, 2, 2]] * 3, dtype=torch.int32))


def test_quantizer_dropout_is_off_in_eval_mode():
    # Semantic-ID generation and evaluation run without randomness: all levels, whatever the draws
    quantizer = FixedScaleResidualFSQ([3], codebook_layers=3, quantizer_dropout=1.0).eval()
    rand_draw, n_draw = _draws([0.0, 0.0, 0.0], [1, 2, 3])
    with rand_draw, n_draw:
        out = quantizer(_dropout_latent())

    _assert_values_close(out.embeddings, [[-5 / 9]] * 3)


def test_quantizer_dropout_defaults_to_off():
    # draws that would shorten every example change nothing without the option, so earlier runs stay unchanged
    rand_draw, n_draw = _draws([0.0, 0.0, 0.0], [1, 1, 1])
    with rand_draw, n_draw:
        out = FixedScaleResidualFSQ([3], codebook_layers=3)(_dropout_latent())

    _assert_values_close(out.embeddings, [[-5 / 9]] * 3)


def test_no_random_numbers_are_drawn_without_quantizer_dropout():
    # runs without the option must leave the global random state as it was before the option existed
    rand_draw, n_draw = _draws([0.0, 0.0, 0.0], [1, 1, 1])
    with rand_draw as rand, n_draw as randint:
        FixedScaleResidualFSQ([3], codebook_layers=3)(_dropout_latent())

    rand.assert_not_called()
    randint.assert_not_called()


def test_quantizer_dropout_draws_one_selection_and_one_length_per_example():
    # one draw per example, not per batch; n runs from 1 (never no level) to the number of levels inclusive
    rand_draw, n_draw = _draws([0.0, 0.0, 0.0], [1, 2, 3])
    with rand_draw as rand, n_draw as randint:
        FixedScaleResidualFSQ([3], codebook_layers=3, quantizer_dropout=1.0)(_dropout_latent())

    rand.assert_called_once_with(3, 1)
    randint.assert_called_once_with(1, 4, (3, 1))


def test_quantizer_dropout_passes_no_gradient_through_a_dropped_level():
    # Two examples with soft-clamped input 1.3, L = 3, 3 levels:
    # level   residual in   share of the level's range   passes gradient   kept by
    # 1       1.3           1.3 (hard clamp at 1)        no                examples 1 and 2
    # 2       0.3           0.9                          yes               example 2
    # example 1: 0; example 2: 1 times the slope of the soft clamp, 1 - (1.3 / 1.5)^2 = 0.248889
    latent = _latent_for([[1.3], [1.3]], [3]).requires_grad_()
    rand_draw = patch("torch.rand", return_value=torch.zeros(2, 1))
    n_draw = patch("torch.randint", return_value=torch.tensor([[1], [2]]))
    with rand_draw, n_draw:
        out = FixedScaleResidualFSQ([3], codebook_layers=3, quantizer_dropout=1.0)(latent)
    out.embeddings.sum().backward()

    _assert_values_close(latent.grad, [[0.0], [0.248889]])


def test_quantizer_dropout_outside_zero_to_one_is_rejected():
    # a typo such as 1.5 would otherwise run as full dropout
    with pytest.raises(ValueError) as error:
        FixedScaleResidualFSQ([3], quantizer_dropout=1.5)

    assert str(error.value) == "quantizer_dropout must be a probability between 0 and 1, got 1.5"
