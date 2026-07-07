import torch
import numpy as np

class MetricAccumulator:
    """
    Accumulates retrieval metrics across multiple batches for accurate averaging,
    fully vectorized on PyTorch devices (CUDA/MPS/CPU).
    """
    def __init__(self, k_list=[1, 5, 10], num_layers=4):
        self.k_list = k_list
        self.num_layers = num_layers
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.total_recall = {k: 0.0 for k in self.k_list}
        self.total_ndcg = {k: 0.0 for k in self.k_list}
        self.hierarchical_recall = {
            f"h@{k}_slice_{i}": 0.0 
            for k in self.k_list 
            for i in range(1, self.num_layers + 1)
        }

    @torch.no_grad()
    def update(self, predictions, targets):
        """
        Update the accumulator with results from a new batch.
        
        Args:
            predictions (torch.Tensor): [Batch, Beam_Size, Codebook_Layers]
            targets (torch.Tensor): [Batch, 1, Codebook_Layers] or [Batch, Codebook_Layers]
        """
        device = predictions.device
        batch_size = targets.size(0)
        self.total_samples += batch_size

        # normalize targets to [Batch, Codebook_Layers]
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        max_k = max(self.k_list)
        
        # exact match comparison: [Batch, max_k, Codebook_Layers] vs [Batch, 1, Codebook_Layers]
        full_match = (predictions[:, :max_k, :] == targets.unsqueeze(1)).all(dim=-1) # [Batch, max_k]

        # handle potential duplicates in predictions: only keep the first match per row
        if full_match.any():
            cum_match = full_match.cumsum(dim=-1)
            full_match = full_match & (cum_match == 1)

        # precompute NDCG rank discount weights
        ranks = torch.arange(max_k, device=device).unsqueeze(0)  # [1, max_k]; 0-based ranks
        discounts = 1.0 / torch.log2(ranks.float() + 2.0)        # [1, max_k]; +2 for 1-based rank in log2

        for k in self.k_list:
            # Recall@k: exact match is within top k
            recall_k = full_match[:, :k].any(dim=-1).float().sum().item()
            self.total_recall[k] += recall_k

            # NDCG@k: sum of matching position discounts within top k
            ndcg_k = (full_match[:, :k] * discounts[:, :k]).sum(dim=-1).sum().item()
            self.total_ndcg[k] += ndcg_k

            # Hierarchical Recall@k for prefix slices
            for layer in range(1, self.num_layers + 1):
                slice_match = (predictions[:, :k, :layer] == targets[:, None, :layer]).all(dim=-1) # None equals unsqueeze(1) -> [Batch, 1, layer]
                h_recall_layer = slice_match.any(dim=-1).float().sum().item()
                self.hierarchical_recall[f"h@{k}_slice_{layer}"] += h_recall_layer

    def compute(self):
        """Returns the final averaged metrics."""
        if self.total_samples == 0:
            return {
                "recall": {k: 0.0 for k in self.k_list},
                "ndcg": {k: 0.0 for k in self.k_list},
                "hierarchical": {k: 0.0 for k in self.hierarchical_recall.keys()},
                "total_samples": 0,
            }
            
        return {
            "recall": {k: float(v / self.total_samples) for k, v in self.total_recall.items()},
            "ndcg": {k: float(v / self.total_samples) for k, v in self.total_ndcg.items()},
            "hierarchical": {k: float(v / self.total_samples) for k, v in self.hierarchical_recall.items()},
            "total_samples": self.total_samples
        }

def calculate_entropy_and_coverage(ids_or_tensor, codebook_size: int):
    """
    Calculate per-layer codebook coverage and Shannon entropy from batch assignments.
    
    Args:
        ids_or_tensor: A list of tensors of shape (batch_size,), or a single 2D tensor
                       of shape (batch_size, num_layers).
        codebook_size (int): Size of the codebook.
        
    Returns:
        coverages (torch.Tensor): Coverage fraction per layer, shape (num_layers,)
        entropies (torch.Tensor): Shannon entropy per layer, shape (num_layers,)
    """
    if isinstance(ids_or_tensor, list):
        layers = ids_or_tensor
    elif isinstance(ids_or_tensor, torch.Tensor):
        if ids_or_tensor.dim() == 2:
            layers = [ids_or_tensor[:, i] for i in range(ids_or_tensor.shape[1])]
        else:
            raise ValueError(f"Expected a 2D tensor of shape (batch_size, num_layers), got shape {ids_or_tensor.shape}")
    else:
        raise TypeError(f"Unsupported type for metric calculation: {type(ids_or_tensor)}")
        
    coverages = []
    entropies = []
    
    for ids in layers:
        unique_ids, counts = torch.unique(ids, return_counts=True)

        coverage = torch.tensor(unique_ids.numel() / codebook_size, device=ids.device)
        coverages.append(coverage)
        
        probs = counts.float() / counts.sum()
        entropy = -(probs * probs.log()).sum()
        entropies.append(entropy)
        
    return torch.stack(coverages), torch.stack(entropies)


def calculate_quantizer_metrics(
    ids: torch.Tensor,
    codebook_size: int,
    input_tensor: torch.Tensor = None,
    first_residual: torch.Tensor = None,
    final_residual: torch.Tensor = None,
    codebooks: list = None
) -> dict:
    """
    Calculate harmonized metrics for quantization models (FSQ, ResidualFSQ, RQ-VAE).
    
    Args:
        ids (torch.Tensor): Assignment IDs, shape (batch_size, num_layers) or (batch_size, seq_len)
        codebook_size (int): Size of the codebook
        input_tensor (torch.Tensor, optional): Quantizer input, shape (batch_size, ...)
        first_residual (torch.Tensor, optional): Residual after the first layer
        final_residual (torch.Tensor, optional): Residual after the final layer
        codebooks (list of torch.Tensor, optional): List of codebook parameters/tensors
        
    Returns:
        dict: Dictionary containing the calculated metrics
    """
    batch_size = ids.shape[0]
    
    # unique ID sequences ratio in the batch
    p_unique_ids = torch.tensor(
        torch.unique(ids.cpu(), dim=0).shape[0] / batch_size,
        device=ids.device
    )
    
    # layer coverages and entropies
    layer_coverages, layer_entropies = calculate_entropy_and_coverage(ids, codebook_size)
    
    metrics = {
        "p_unique_ids": p_unique_ids,
        "layer_coverages": layer_coverages,
        "layer_entropies": layer_entropies
    }
    
    # residual metrics (optional)
    if input_tensor is not None and first_residual is not None and final_residual is not None:
        # compute norms over the last dimension (latent dimension) and mean over the batch/sequence dims
        input_norm = input_tensor.norm(dim=-1).mean().clamp(min=1e-8)
        first_res_norm = first_residual.norm(dim=-1).mean()
        final_res_norm = final_residual.norm(dim=-1).mean()
        
        metrics["first_residual_norm"] = first_res_norm
        metrics["last_residual_norm"] = final_res_norm
        metrics["first_residual_rel"] = first_res_norm / input_norm
        metrics["last_residual_rel"] = final_res_norm / input_norm
        
    # codebook centroid metrics (optional)
    if codebooks is not None and len(codebooks) > 0:
        metrics["first_centroids_norm"] = codebooks[0].norm(dim=-1).mean()
        metrics["last_centroids_norm"] = codebooks[-1].norm(dim=-1).mean()
        
    return metrics