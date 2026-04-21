import torch
import pytest
from unittest.mock import patch
from modules.rqvae.model import RQ_VAE
from schemas.quantization import QuantizeForwardMode, QuantizeDistance

@pytest.fixture
def dummy_model():
    return RQ_VAE(
        input_dim=64,
        latent_dim=16,
        hidden_dims=[32],
        codebook_size=8,
        codebook_kmeans_init=True,
        n_quantization_layers=3,
        quantization_method=QuantizeForwardMode.STE,
        distance_mode=QuantizeDistance.L2
    )

@pytest.fixture
def dummy_data():
    return torch.randn(100, 64)

@pytest.mark.unit
def test_kmeans_init_codebooks_sets_flags(dummy_model, dummy_data):
    """
    Test that kmeans_init_codebooks initializes the quantization layers 
    properly by setting kmeans_initted flags.
    """
    for layer in dummy_model.quantization_layers:
        assert not layer.kmeans_initted
        
    dummy_model.kmeans_init_codebooks(dummy_data)
    
    for layer in dummy_model.quantization_layers:
        assert layer.kmeans_initted

@pytest.mark.unit
def test_kmeans_init_codebooks_no_grad(dummy_model, dummy_data):
    """
    Test that kmeans_init_codebooks correctly detaches everything 
    and does not build a massive computation graph.
    """
    dummy_data.requires_grad = True
    
    dummy_model.kmeans_init_codebooks(dummy_data)
    
    for name, param in dummy_model.named_parameters():
        assert param.grad is None

@patch("modules.rqvae.quantization.Quantization._kmeans_init")
@pytest.mark.unit
def test_kmeans_init_prevents_reinit_in_forward(mock_kmeans_init, dummy_model, dummy_data):
    """
    Test that a forward pass doesn't try to run kmeans again 
    if the codebooks are already marked as initialized.
    """
    for layer in dummy_model.quantization_layers:
        layer.kmeans_initted = True
        
    out = dummy_model(dummy_data)
    
    mock_kmeans_init.assert_not_called()
    assert out.loss is not None

@patch("modules.rqvae.quantization.Quantization._kmeans_init")
@pytest.mark.unit
def test_kmeans_init_is_called_when_uninitialized(mock_kmeans_init, dummy_model, dummy_data):
    """
    Test that a forward pass DOES run kmeans initialization 
    if the codebooks are not marked as initialized and do_kmeans_init is True.
    """
    # Arrange: Ensure codebooks are uninitialized
    for layer in dummy_model.quantization_layers:
        assert not layer.kmeans_initted
        assert layer.do_kmeans_init
        
    # Act: Run a standard forward pass
    out = dummy_model(dummy_data)
    
    # Assert: _kmeans_init should have been called once for each layer
    assert mock_kmeans_init.call_count == dummy_model.n_quantization_layers
    assert out.loss is not None
