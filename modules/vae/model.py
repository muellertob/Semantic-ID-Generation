import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from schemas.vae import VaeOutput

class NormalizeLayer(nn.Module):
    def __init__(self, dim=-1, p=2):
        super().__init__()
        self.dim = dim
        self.p = p
    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=self.p)

class QuantizedAutoEncoder(nn.Module):
    """
    Unified AutoEncoder architecture for Semantic ID Generation.
    
    This architecture handles standard continuous encoding, applies a discrete quantization
    bottleneck (e.g., FSQ, VQ, Residual-VQ), and then decodes back to the continuous space.
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        input_dim: int,
        latent_dim: int,
        loss_type: str = "mse",
        normalize: bool = True
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        
        if normalize:
            self.normalization_layer = nn.Sequential(
                nn.BatchNorm1d(input_dim, affine=False),
                NormalizeLayer(dim=-1, p=2)
            )
        else:
            self.normalization_layer = nn.Identity()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def kmeans_init_codebooks(self, data: torch.Tensor, temperature: float = 1.0) -> None:
        """Delegates K-Means initialization to the quantizer if it supports it."""
        if hasattr(self.quantizer, "kmeans_init"):
            normalized_data = self.normalization_layer(data.to(self.device).float())
            encoded = self.encoder(normalized_data)
            self.quantizer.kmeans_init(encoded, temperature=temperature)

    def get_semantic_ids(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns just the discrete IDs."""
        normalized_x = self.normalization_layer(x)
        encoded = self.encoder(normalized_x)
        quant_out = self.quantizer(encoded, **kwargs)
        return quant_out.ids

    def forward(self, x: torch.Tensor, **kwargs) -> VaeOutput:
        normalized_x = self.normalization_layer(x)
        
        encoded = self.encoder(normalized_x)

        quant_out = self.quantizer(encoded, **kwargs)
        
        x_hat = self.decoder(quant_out.embeddings)
        
        # calculate reconstruction loss
        if self.loss_type == "l1":
            recon_loss = F.l1_loss(x_hat, normalized_x)
        else:
            recon_loss = F.mse_loss(x_hat, normalized_x)
            
        # combine with optional quantization loss
        total_loss = recon_loss + quant_out.loss
        
        return VaeOutput(
            loss=total_loss,
            reconstruction_loss=recon_loss,
            quantization_loss=quant_out.loss,
            metrics=quant_out.metrics
        )

