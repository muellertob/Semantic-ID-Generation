import torch

from einops import rearrange
from .networks import Encoder, Decoder
from .quantization import Quantization
from schemas.quantization import QuantizeForwardMode, QuantizeDistance
from huggingface_hub import PyTorchModelHubMixin
from typing import List
from torch import nn
from torch import Tensor
from schemas.rq_vae import RqVaeOutput, RqVaeComputedLosses
import torch.nn.functional as F

class RQ_VAE(nn.Module, PyTorchModelHubMixin):
    """
    Residual Quantized Variational Autoencoder (RQ-VAE) for semantic ID generation.

    Supports both Straight-Through Estimation (STE) and Gumbel Softmax quantization
    methods for better gradient flow and joint training with language models.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_sim_vq: bool = False,
        n_quantization_layers: int = 3,
        commitment_weight: float = 0.25,
        quantization_method: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_mode: QuantizeDistance = QuantizeDistance.COSINE,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.codebook_size = codebook_size
        self.codebook_kmeans_init = codebook_kmeans_init
        self.codebook_sim_vq = codebook_sim_vq
        self.commitment_weight = commitment_weight
        self.quantization_method = quantization_method
        self.n_quantization_layers = n_quantization_layers
        self.distance_mode = distance_mode

        self.quantization_layers = nn.ModuleList(modules=[
            Quantization(
                latent_dim=latent_dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
                do_kmeans_init=codebook_kmeans_init,
                sim_vq=codebook_sim_vq,
                forward_mode=quantization_method,
                distance_mode=distance_mode,
            )
            for _ in range(n_quantization_layers)
        ])
        
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
        )
        
        self.decoder = Decoder(
            output_dim=input_dim,
            hidden_dims=hidden_dims[::-1],
            latent_dim=latent_dim,
        )
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def set_quantization_method(self, method: QuantizeForwardMode) -> None:
        """Change quantization method for all layers."""
        self.quantization_method = method
        for layer in self.quantization_layers:
            layer.set_quantization_method(method)



    @torch.no_grad()
    def kmeans_init_codebooks(self, data: Tensor, temperature: float = 1.0) -> None:
        """
        Initializes all quantization layers using k-means with full (or large) dataset.
        Call this before training.
        """
        x = self.encode(data.to(self.device).float())
        for layer in self.quantization_layers:
            layer._kmeans_init(x)
            emb = layer.get_item_embeddings(layer(x, temperature=temperature).ids)
            x = x - emb
        
    def get_semantic_id_single(self, x: Tensor, temperature: float = 1.0) -> Tensor:
        res = self.encode(x.unsqueeze(0))  # Add batch dim (1, ...)

        sem_ids = []
        for layer in self.quantization_layers:
            quantized = layer(res, temperature=temperature)
            id = quantized.ids.squeeze(0)  # Remove batch dim
            res = res - quantized.embeddings
            sem_ids.append(id)

        return torch.stack(sem_ids, dim=0)  # shape: (num_layers, semantic_id_dim)

    def get_semantic_ids(self, x: Tensor, temperature: float = 1.0) -> RqVaeOutput:
        res = self.encode(x)

        quantize_loss = torch.tensor(0.0, device=x.device)
        embs, residuals, sem_ids = [], [], []

        for layer in self.quantization_layers:
            residuals.append(res)
            quantized = layer(res, temperature=temperature)
            quantize_loss = quantize_loss + quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb  # Update residuals
            sem_ids.append(id)
            embs.append(emb)

        return RqVaeOutput(
            embeddings=rearrange(embs, "h b d -> h d b"),
            residuals=rearrange(residuals, "h b d -> h d b"),
            sem_ids=rearrange(sem_ids, "h b -> b h"),
            quantize_loss=quantize_loss
        )
        
    def forward(self, x, temperature: float = 1.0) -> RqVaeComputedLosses:
        res = self.encode(x)

        quantize_loss = torch.tensor(0.0, device=x.device)
        embs, residuals, sem_ids = [], [], []
        first_residual = None

        for i, layer in enumerate(self.quantization_layers):
            residuals.append(res)
            quantized = layer(res, temperature=temperature)
            quantize_loss = quantize_loss + quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb  # Update residuals
            if i == 0:
                first_residual = res
            sem_ids.append(id)
            embs.append(emb)

        final_residual = res
        embs_tensor = torch.stack(embs) # (h, b, d)

        # decode
        x_hat = self.decode(embs_tensor.sum(dim=0))

        reconstuction_loss = F.mse_loss(x_hat, x)
        rqvae_loss = quantize_loss.mean()
        loss = reconstuction_loss + rqvae_loss

        with torch.no_grad():
            sem_ids_tensor = torch.stack(sem_ids, dim=1)
            embs_norm = embs_tensor.norm(dim=-1).T  # (h, b) -> (b, h)

            # fraction of unique full-tuple IDs in the batch
            p_unique_ids = torch.unique(sem_ids_tensor, dim=0).shape[0] / sem_ids_tensor.shape[0]

            # residual norms relative to the encoder output
            input_norm = residuals[0].norm(dim=1).mean().clamp(min=1e-8)
            first_residual_norm = first_residual.norm(dim=1).mean() / input_norm
            last_residual_norm = final_residual.norm(dim=1).mean() / input_norm

            # codebook vector norms (first and last layer)
            first_centroids_norm = self.quantization_layers[0].get_codebook().norm(dim=1).mean()
            last_centroids_norm = self.quantization_layers[-1].get_codebook().norm(dim=1).mean()

            # per-layer: fraction of codebook used + Shannon entropy of assignments
            layer_coverages, layer_entropies = [], []
            for ids in sem_ids:
                unique_ids, counts = torch.unique(ids, return_counts=True)
                layer_coverages.append(unique_ids.shape[0] / self.codebook_size)
                probs = counts.float() / counts.sum()
                layer_entropies.append(-(probs * probs.log()).sum())
            layer_coverages = torch.tensor(layer_coverages, device=x.device)
            layer_entropies = torch.stack(layer_entropies)

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss,
            rqvae_loss=rqvae_loss,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            layer_coverages=layer_coverages,
            layer_entropies=layer_entropies,
            first_residual_norm=first_residual_norm,
            last_residual_norm=last_residual_norm,
            first_centroids_norm=first_centroids_norm,
            last_centroids_norm=last_centroids_norm,
        )