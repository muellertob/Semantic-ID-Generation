import torch
from modules.fsq.model import FSQ_AutoEncoder, ResidualFSQ_AutoEncoder
from schemas.vae import VaeOutput


# ==============================================================================
# FSQ AutoEncoder (Flat Pipeline)
# ==============================================================================

def test_fsq_autoencoder_forward_and_backward():
    model = FSQ_AutoEncoder(
        input_dim=768,
        seq_len=3,
        hidden_dims=[128, 64],
        latent_dim=30,
        level_list=[8, 6, 5],
        loss_type="mse",
        normalize=True,
        projection_type="mlp_1_hidden",
        inner_dim=32
    )
    
    x = torch.randn(4, 768)
    out = model(x)
    
    assert isinstance(out, VaeOutput)
    assert out.loss is not None
    assert out.loss.requires_grad
    assert out.loss.item() >= 0
    assert out.metrics["p_unique_ids"].item() >= 0
    assert out.metrics["layer_coverages"].shape == (3,)
    assert out.metrics["layer_entropies"].shape == (3,)
    
    out.loss.backward()
    
    # Ensure gradients flow to encoder, decoder, and quantizer projection parameters
    has_encoder_grad = any(p.grad is not None for p in model.encoder.parameters())
    has_decoder_grad = any(p.grad is not None for p in model.decoder.parameters())
    has_quantizer_grad = any(p.grad is not None for p in model.quantizer.parameters())
    
    assert has_encoder_grad, "Encoder did not receive gradients"
    assert has_decoder_grad, "Decoder did not receive gradients"
    assert has_quantizer_grad, "Quantizer projections did not receive gradients"


def test_fsq_autoencoder_get_semantic_ids():
    model = FSQ_AutoEncoder(
        input_dim=768,
        seq_len=4,
        hidden_dims=[128, 64],
        latent_dim=32,
        level_list=[8, 6, 5]
    )
    
    x = torch.randn(4, 768)
    sem_ids = model.get_semantic_ids(x)
    
    assert isinstance(sem_ids, torch.Tensor)
    assert sem_ids.shape == (4, 4)


# ==============================================================================
# ResidualFSQ AutoEncoder (Residual Pipeline)
# ==============================================================================

def test_residual_fsq_autoencoder_forward_and_backward():
    model = ResidualFSQ_AutoEncoder(
        input_dim=768,
        seq_len=3,
        hidden_dims=[128, 64],
        latent_dim=30,
        level_list=[8, 6, 5],
        loss_type="mse",
        normalize=True,
        projection_type="mlp_1_hidden",
        inner_dim=32
    )
    
    x = torch.randn(4, 768)
    out = model(x)
    
    assert isinstance(out, VaeOutput)
    assert out.loss is not None
    assert out.loss.requires_grad
    assert out.loss.item() >= 0
    assert out.metrics["p_unique_ids"].item() >= 0
    assert out.metrics["layer_coverages"].shape == (3,)
    assert out.metrics["layer_entropies"].shape == (3,)
    
    out.loss.backward()
    
    # Ensure gradients flow to encoder, decoder, and quantizer projection parameters
    has_encoder_grad = any(p.grad is not None for p in model.encoder.parameters())
    has_decoder_grad = any(p.grad is not None for p in model.decoder.parameters())
    has_quantizer_grad = any(p.grad is not None for p in model.quantizer.parameters())
    
    assert has_encoder_grad, "Encoder did not receive gradients"
    assert has_decoder_grad, "Decoder did not receive gradients"
    assert has_quantizer_grad, "Quantizer projections did not receive gradients"


def test_residual_fsq_autoencoder_get_semantic_ids():
    model = ResidualFSQ_AutoEncoder(
        input_dim=768,
        seq_len=4,
        hidden_dims=[128, 64],
        latent_dim=32,
        level_list=[8, 6, 5]
    )
    
    x = torch.randn(4, 768)
    sem_ids = model.get_semantic_ids(x)
    
    assert isinstance(sem_ids, torch.Tensor)
    assert sem_ids.shape == (4, 4)

