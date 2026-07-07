import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.cluster import KMeans

from schemas.quantization import QuantizeOutput, QuantizeForwardMode, QuantizeDistance
from utils.metrics import calculate_quantizer_metrics
from utils.seed import get_seed

class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, z: Tensor, z_hat: Tensor) -> Tensor:
        # Match TIGER's Quantization Loss:
        # Loss = ||sg[z] - z_hat||^2 + beta * ||z - sg[z_hat]||^2
        z_no_grad = z.detach()
        z_hat_no_grad = z_hat.detach()
        
        loss = F.mse_loss(z_no_grad, z_hat) + self.commitment_weight * F.mse_loss(z, z_hat_no_grad)
        return loss

class Quantization(nn.Module):
    """
    Vector Quantization module supporting both Straight-Through Estimation (STE)
    and Gumbel Softmax quantization methods.

    Args:
        latent_dim: Dimension of the latent vectors
        codebook_size: Number of vectors in the codebook
        commitment_weight: Weight for the commitment loss
        do_kmeans_init: Whether to initialize codebook with k-means
        sim_vq: Whether to use similarity-based VQ with projection layer
        forward_mode: Quantization method to use (QuantizeForwardMode enum)
        distance_mode: Distance metric to use (QuantizeDistance enum)

    Note:
        For Gumbel Softmax quantization, temperature should be passed to the forward() method.
    """
    def __init__(
        self,
        latent_dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        do_kmeans_init: bool = True,
        sim_vq: bool = False,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
    ) -> None:
        super().__init__()
        self.embed_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False) if sim_vq else nn.Identity(),
        )
        self.quantize_loss = QuantizeLoss(commitment_weight)
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights uniformly."""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_quantization_method(self, method: QuantizeForwardMode) -> None:
        """Change the quantization method."""
        self.forward_mode = method
    
    @torch.no_grad
    def _kmeans_init(self, z: Tensor):
        # flatten batch dimension
        z = z.view(-1, self.embed_dim)

        # detach to ensure no gradients
        z_np = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.codebook_size, init='k-means++', n_init='auto', max_iter=300, random_state=get_seed())
        kmeans.fit(z_np)
        
        self.embedding.weight.copy_(torch.from_numpy(kmeans.cluster_centers_).to(self.device))
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.out_proj(self.embedding(item_ids))
    
    def get_codebook(self) -> Tensor:
        return self.out_proj(self.embedding.weight)

    def _compute_distances(self, z: Tensor) -> Tensor:
        """Compute distances between input vectors and codebook entries."""
        codebook = self.get_codebook()

        if self.distance_mode == QuantizeDistance.L2:
            # Compute squared L2 distances: ||x - c||^2
            error = z.unsqueeze(1) - codebook.unsqueeze(0)
            dist = torch.sum(error**2, dim=-1)
        elif self.distance_mode == QuantizeDistance.COSINE:
            # Compute negative cosine similarity (so min distance = max similarity)
            z_norm = z / z.norm(dim=1, keepdim=True)
            codebook_norm = codebook / codebook.norm(dim=1, keepdim=True)
            dist = -(z_norm @ codebook_norm.T)
        else:
            raise ValueError(f"Unsupported distance mode: {self.distance_mode}")

        return dist

    def forward(self, z: Tensor, temperature: float = 1.0) -> QuantizeOutput:
        """
        Forward pass with unified quantization logic.

        Args:
            z: Input tensor to quantize
            temperature: Temperature for Gumbel Softmax (ignored for STE)
        """
        assert z.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(z)

        # Compute distances to codebook entries
        dist = self._compute_distances(z)

        # Get discrete assignments (used by both methods)
        _, ids = dist.detach().min(dim=1)

        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                # Gumbel Softmax quantization
                codebook = self.get_codebook()

                # Convert distances to logits (negative distances for higher probability)
                logits = -dist / temperature

                # Apply Gumbel Softmax
                soft_assignment = F.gumbel_softmax(logits, tau=temperature, hard=False)
                soft_emb = soft_assignment @ codebook
                emb_out = soft_emb

                # For loss, use the closest codebook entry (like STE) to encourage commitment
                closest_emb = self.get_item_embeddings(ids)
                loss = self.quantize_loss(z=z, z_hat=closest_emb)

            elif self.forward_mode == QuantizeForwardMode.STE:
                # Straight-Through Estimation
                closest_emb = self.get_item_embeddings(ids)
                emb_out = z + (closest_emb - z).detach()

                # Use the quantized embedding for loss
                loss = self.quantize_loss(z=z, z_hat=closest_emb)

            else:
                raise ValueError(f"Unsupported forward mode: {self.forward_mode}")
        else:
            # Evaluation mode: use hard assignment for both methods
            closest_emb = self.get_item_embeddings(ids)
            emb_out = closest_emb

            # Compute loss for compatibility
            loss = self.quantize_loss(z=z, z_hat=closest_emb)

        return QuantizeOutput(
            embeddings=emb_out,
            ids=ids,
            loss=loss,
        )

class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self, 
        codebook_layers: int, 
        latent_dim: int, 
        codebook_size: int, 
        commitment_weight: float = 0.25, 
        do_kmeans_init: bool = True, 
        sim_vq: bool = False, 
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.STE, 
        distance_mode: QuantizeDistance = QuantizeDistance.L2
    ):
        super().__init__()
        self.codebook_layers = codebook_layers
        self.codebook_size = codebook_size
        self.quantization_layers = nn.ModuleList([
            Quantization(
                latent_dim=latent_dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
                do_kmeans_init=do_kmeans_init,
                sim_vq=sim_vq,
                forward_mode=forward_mode,
                distance_mode=distance_mode,
            )
            for _ in range(codebook_layers)
        ])
        
    def set_quantization_method(self, method: QuantizeForwardMode) -> None:
        for layer in self.quantization_layers:
            layer.set_quantization_method(method)
            
    def kmeans_init(self, z: Tensor, temperature: float = 1.0) -> None:
        """Initializes all quantization layers using k-means."""
        res = z
        for layer in self.quantization_layers:
            layer._kmeans_init(res)
            # use temperature to get the right soft/hard embedding during init
            emb = layer.get_item_embeddings(layer(res, temperature=temperature).ids)
            res = res - emb

    def forward(self, x: Tensor, temperature: float = 1.0, **kwargs) -> QuantizeOutput:
        res = x
        quantize_loss = torch.tensor(0.0, device=x.device)
        embs, residuals, sem_ids = [], [], []
        first_residual = None

        for i, layer in enumerate(self.quantization_layers):
            residuals.append(res)
            quantized = layer(res, temperature=temperature)
            quantize_loss = quantize_loss + quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb  # update residuals
            if i == 0:
                first_residual = res
            sem_ids.append(id) # list of tensors containing the discrete IDs at each layer
            embs.append(emb) # list of tensors containing the quantized embeddings at each layer

        final_residual = res

        embs_tensor = torch.stack(embs) # shape: (codebook_layers, batch, latent_dim)
        
        # summing across the hierarchy dimension (dim=0) rebuilds the final continuous representation
        quantized_latent = embs_tensor.sum(dim=0) # shape: (batch, latent_dim)
        
        sem_ids_tensor = torch.stack(sem_ids, dim=1) # shape: (batch, codebook_layers)
        
        with torch.no_grad():
            codebooks = [layer.get_codebook() for layer in self.quantization_layers]
            metrics = calculate_quantizer_metrics(
                ids=sem_ids_tensor,
                codebook_size=self.codebook_size,
                input_tensor=x,
                first_residual=first_residual,
                final_residual=final_residual,
                codebooks=codebooks
            )
            
        return QuantizeOutput(
            embeddings=quantized_latent, 
            ids=sem_ids_tensor, 
            loss=quantize_loss, 
            metrics=metrics
        )