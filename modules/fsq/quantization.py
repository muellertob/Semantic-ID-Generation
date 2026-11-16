import torch
import torch.nn as nn
from typing import List

from schemas.quantization import QuantizeOutput
import torch.nn.functional as F
from utils.metrics import calculate_quantizer_metrics

def round_ste(z):
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

class FSQCore(nn.Module):
    """
    Core mathematical operations for Finite Scalar Quantization (FSQ).
    Based on the paper "Finite Scalar Quantization: VQ-VAE Made Simple" (https://arxiv.org/abs/2309.15505).
    """
    def __init__(self, level_list: List[int]):
        super().__init__()
        self.level_list = level_list
        levels = torch.tensor(self.level_list, dtype=torch.int32)
        self.register_buffer("levels", levels, persistent=False)
        basis = torch.cumprod(torch.tensor([1] + self.level_list[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("basis", basis, persistent=False)
        self.dim = len(level_list)

    @property
    def device(self) -> torch.device:
        return self.levels.device

    def bound(self, z, eps=1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self.levels - 1) * (1 - eps) / 2
        offset = torch.where(self.levels % 2 == 1, 0.0, 0.5)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset
    
    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self.levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self.levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self.levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        zhat = self._scale_and_shift(zhat)
        return (zhat * self.basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        indices = torch.unsqueeze(indices, dim=-1)
        codes_non_centered = (indices // self.basis) % self.levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        return codes

    def forward(self, z: torch.Tensor):
        zhat = self.quantize(z)
        indices = self.codes_to_indices(zhat)
        return zhat, indices

class FSQ(FSQCore):
    """
    Finite Scalar Quantization (FSQ) module with projection layer and seq chunking.
    It takes an input of dimension `len(level_list)` and maps it to a codebook.
    """
    def __init__(self, dim: int = None, codebook_layers: int = 1, level_list: List[int] = [8, 6, 5], projection_type: str = None, inner_dim: int = None):
        super().__init__(level_list)
        
        self.codebook_dim = len(self.level_list)
        self.codebook_size = 1
        for level in level_list:
            self.codebook_size *= level
        self.codebook_layers = codebook_layers
        self.chunk_dim = dim // codebook_layers if dim is not None else None
            
        if projection_type is None:
            projection_type = "linear" if self.chunk_dim is not None and self.chunk_dim != self.codebook_dim else "identity"
            
        self.project_in = ProjectionBlock(dim_in=self.chunk_dim, inner_dim=inner_dim, dim_out=self.codebook_dim, projection_type=projection_type)
        self.project_out = ProjectionBlock(dim_in=self.codebook_dim, inner_dim=inner_dim, dim_out=self.chunk_dim, projection_type=projection_type)

    def forward(self, z: torch.Tensor, **kwargs) -> QuantizeOutput:
        """
        Forward pass for FSQ.
        Args:
            z: Input tensor (batch_size, latent_dim)
        Returns:
            QuantizeOutput containing embeddings, ids, and zero loss.
        """
        batch_size = z.shape[0]
        
        # split latent dimension into sequence chunks
        z_chunked = z.view(batch_size, self.codebook_layers, -1)
        
        z_proj = self.project_in(z_chunked)
        codes = self.quantize(z_proj)
        indices = self.codes_to_indices(codes)
        out_emb = self.project_out(codes)
        
        # flatten back to the global latent dimension
        out_emb = out_emb.view(batch_size, -1)
        
        loss = torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            metrics = calculate_quantizer_metrics(
                ids=indices,
                codebook_size=self.codebook_size,
                input_tensor=z
            )
        
        return QuantizeOutput(embeddings=out_emb, ids=indices, loss=loss, metrics=metrics)

class ProjectionBlock(nn.Module):
    def __init__(self, dim_in: int, inner_dim: int, dim_out: int, projection_type: str = "mlp_1_hidden"):
        super().__init__()
        self.projection_type = projection_type
        
        if projection_type == "identity" or projection_type is None:
            self.proj = nn.Identity()
        elif projection_type == "linear":
            self.proj = nn.Linear(dim_in, dim_out, bias=True)
        elif projection_type == "mlp_1_hidden":
            self.proj = nn.Sequential(
                nn.Linear(dim_in, inner_dim, bias=False),
                nn.SiLU(),
                nn.Linear(inner_dim, dim_out, bias=False)
            )
        elif projection_type == "mlp_2_hidden":
            self.proj = nn.Sequential(
                nn.Linear(dim_in, inner_dim, bias=False),
                nn.SiLU(),
                nn.Linear(inner_dim, inner_dim, bias=False),
                nn.SiLU(),
                nn.Linear(inner_dim, dim_out, bias=False)
            )
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")

    def forward(self, x: torch.Tensor):
        return self.proj(x)

class FSQLayer(nn.Module):
    def __init__(self, dim: int, level_list: List[int], inner_dim: int = 256, projection_type: str = "mlp_1_hidden"):
        super().__init__()
        self.fsq = FSQCore(level_list)
        self.proj_in = ProjectionBlock(dim_in=dim, inner_dim=inner_dim, dim_out=self.fsq.dim, projection_type=projection_type)
        self.proj_out = ProjectionBlock(dim_in=self.fsq.dim, inner_dim=inner_dim, dim_out=dim, projection_type=projection_type)

    def forward(self, x: torch.Tensor):
        z = self.proj_in(x)
        zhat, indices = self.fsq(z)
        out = self.proj_out(zhat)
        return out, indices

class ResidualFSQ(nn.Module):
    def __init__(self, dim: int, level_list: List[int] = [8, 6, 5], codebook_layers: int = 3, inner_dim: int = 128, projection_type: str = "mlp_1_hidden"):
        super().__init__()
        self.level_list = level_list
        self.codebook_layers = codebook_layers
        
        self.codebook_size = 1
        for level in level_list:
            self.codebook_size *= level
            
        self.layers = nn.ModuleList([
            FSQLayer(dim=dim, level_list=level_list, inner_dim=inner_dim, projection_type=projection_type) 
            for _ in range(codebook_layers)
        ])
        
    def forward(self, x: torch.Tensor, **kwargs) -> QuantizeOutput:
        indices_stack = []
        residual = x
        quantized_out = 0
        first_residual = None
        
        for i, layer in enumerate(self.layers):
            q, indices = layer(residual)
            indices_stack.append(indices)
            residual = residual - q.detach()
            if i == 0:
                first_residual = residual
            quantized_out = quantized_out + q
            
        sem_ids = torch.stack(indices_stack, dim=1)
        # no quantization loss for FSQ, only reconstruction loss is used in the VAE framework
        loss = torch.tensor(0.0, device=x.device)
        final_residual = residual
        
        with torch.no_grad():
            metrics = calculate_quantizer_metrics(
                ids=sem_ids,
                codebook_size=self.codebook_size,
                input_tensor=x,
                first_residual=first_residual,
                final_residual=final_residual
            )
        
        return QuantizeOutput(embeddings=quantized_out, ids=sem_ids, loss=loss, metrics=metrics)


class FixedScaleResidualFSQ(nn.Module):
    """
    Parameter-free residual FSQ in the latent: one latent dimension per FSQ channel, no projections.

    Level i rounds the residual on the symmetric grid {2k / (L_j - 1) - 1} scaled by `L_j ** -i`, so the
    levels tile exactly: their sum is the soft-clamped input rounded to `L_j ** codebook_layers` values.

    - Residual scheme: SoundStream, Algorithm 1 (https://arxiv.org/abs/2107.03312).
    - Grid and rounding: Parker et al., Eq. 1 (https://arxiv.org/abs/2411.19842).
    - Scales and clamps: `ResidualFSQ` of lucidrains' vector-quantize-pytorch (commit 3867f60), which
      cites no paper for them.

    Unlike that implementation, the residual is not detached: the gradient of the output with respect to
    the soft-clamped input is 1 where any level is inside its hard clamp, else 0. With a detached
    residual each such level would add 1.
    """
    def __init__(self, level_list: List[int], codebook_layers: int = 3):
        super().__init__()
        self.level_list = level_list
        self.codebook_layers = codebook_layers
        self.codebook_size = 1
        for level in level_list:
            self.codebook_size *= level

        levels = torch.tensor(level_list, dtype=torch.float32)
        self.register_buffer("levels", levels, persistent=False)
        basis = torch.cumprod(torch.tensor([1] + level_list[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("basis", basis, persistent=False)
        scales = torch.stack([levels ** -i for i in range(codebook_layers)])
        self.register_buffer("scales", scales, persistent=False)
        # outer edge of the outermost grid cell
        self.register_buffer("clamp_value", 1 + 1 / (levels - 1), persistent=False)

    def forward(self, x: torch.Tensor, **kwargs) -> QuantizeOutput:
        bounded = (x / self.clamp_value).tanh() * self.clamp_value
        residual = bounded
        quantized_out = 0
        first_residual = None
        indices_stack = []

        for i, scale in enumerate(self.scales):
            u = (residual / scale).clamp(-1, 1)
            t = (self.levels - 1) * (u + 1) / 2
            # floor(t + 0.5) as in Parker et al., Eq. 1: ties go up (torch.round sends ties to the even integer)
            digit = t + (torch.floor(t + 0.5) - t).detach()
            q = (digit * 2 / (self.levels - 1) - 1) * scale
            indices_stack.append((digit.detach().round().to(torch.int32) * self.basis).sum(dim=-1))
            residual = residual - q
            if i == 0:
                first_residual = residual
            quantized_out = quantized_out + q

        ids = torch.stack(indices_stack, dim=1)
        loss = torch.tensor(0.0, device=x.device)

        with torch.no_grad():
            # input_tensor is the soft-clamped input (the input of level 1), so relative residual
            # norms are measured in the space the levels work in
            metrics = calculate_quantizer_metrics(
                ids=ids,
                codebook_size=self.codebook_size,
                input_tensor=bounded,
                first_residual=first_residual,
                final_residual=residual
            )

        return QuantizeOutput(embeddings=quantized_out, ids=ids, loss=loss, metrics=metrics)
