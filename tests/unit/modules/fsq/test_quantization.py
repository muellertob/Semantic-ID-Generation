import math
import pytest
import torch
from modules.fsq.quantization import FSQ, round_ste, ProjectionBlock, FSQLayer, ResidualFSQ
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
    # Test FSQ with seq_len = 4 and level_list of size 3 (latent_dim = 12)
    fsq = FSQ(dim=12, seq_len=4, level_list=[8, 6, 5])
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
            seq_len=4,
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
    n_quantizers = 4
    quantizer = ResidualFSQ(
        dim=256, level_list=[8, 6, 5], n_quantizers=n_quantizers, inner_dim=128, projection_type="mlp_1_hidden"
    )
    x = torch.randn(16, 256, requires_grad=True)
    out = quantizer(x)
    
    assert out.embeddings.shape == (16, 256)
    assert out.ids.shape == (16, n_quantizers)
    
    # Placeholder loss to check gradient flow
    loss = out.embeddings.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input"


def test_residual_fsq_metrics_and_zero_input():
    n_quantizers = 3
    quantizer = ResidualFSQ(
        dim=64, level_list=[8, 6, 5], n_quantizers=n_quantizers, inner_dim=32, projection_type="mlp_1_hidden"
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

