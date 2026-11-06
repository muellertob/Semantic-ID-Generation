import torch
import numpy as np

class MetricAccumulator:
    """
    Accumulates retrieval metrics over batches; compute() returns averages over all rows.

    update() optionally takes user_ids, seen_tuples (SID tuples of each row's history) and
    is_last (one flag per row), each in all update() calls or in none. compute() then adds:
      - user_ids: per-user averages (recall_user, ndcg_user, hierarchical_user), num_users,
      - is_last: averages over the users' last targets (recall_last, ndcg_last),
      - seen_tuples: seen_at_k, the share of top-k slots holding a seen item
        (with user_ids also users_with_seen_at_k).
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
        # which optional inputs the first update() received; later updates must match
        self._given = None
        # per-row values, kept for the per-user and last-target averages in compute()
        self._row_users = []
        self._row_is_last = []
        self._row_recall = {k: [] for k in self.k_list}
        self._row_ndcg = {k: [] for k in self.k_list}
        self._row_hier = {key: [] for key in self.hierarchical_recall}
        self._row_seen = {k: [] for k in self.k_list}
        self._seen_hits = {k: 0.0 for k in self.k_list}
        self._seen_slots = {k: 0 for k in self.k_list}

    @torch.no_grad()
    def update(self, predictions, targets, user_ids=None, seen_tuples=None, is_last=None):
        """
        Update the accumulator with results from a new batch.
        
        Args:
            predictions (torch.Tensor): [Batch, Beam_Size, Codebook_Layers]; Beam_Size >= max(k_list)
            targets (torch.Tensor): [Batch, 1, Codebook_Layers] or [Batch, Codebook_Layers]
            user_ids (torch.Tensor, optional): [Batch]; enables the per-user metrics
            seen_tuples (torch.Tensor, optional): [Batch, S_max, Codebook_Layers] SID tuples of all
                items in the row's history before the target, padded with -1 (negative values are
                padding); enables seen_at_k
            is_last (torch.Tensor, optional): [Batch] bool; enables the last-target metrics
        """
        max_k = max(self.k_list)
        if predictions.size(1) < max_k:
            raise ValueError(f"predictions have {predictions.size(1)} columns, need at least max(k_list) = {max_k}")
        given = {"user_ids": user_ids is not None, "seen_tuples": seen_tuples is not None, "is_last": is_last is not None}
        if self._given is None:
            self._given = given
        else:
            for name, present in given.items():
                if present != self._given[name]:
                    raise ValueError(f"{name} must be given in every update() call or in none "
                                     f"(first call: {'given' if self._given[name] else 'not given'})")

        device = predictions.device
        batch_size = targets.size(0)
        self.total_samples += batch_size

        # normalize targets to [Batch, Codebook_Layers]
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        # exact match comparison: [Batch, max_k, Codebook_Layers] vs [Batch, 1, Codebook_Layers]
        full_match = (predictions[:, :max_k, :] == targets.unsqueeze(1)).all(dim=-1) # [Batch, max_k]

        # handle potential duplicates in predictions: only keep the first match per row
        if full_match.any():
            cum_match = full_match.cumsum(dim=-1)
            full_match = full_match & (cum_match == 1)

        # precompute NDCG rank discount weights
        ranks = torch.arange(max_k, device=device).unsqueeze(0)  # [1, max_k]; 0-based ranks
        discounts = 1.0 / torch.log2(ranks.float() + 2.0)        # [1, max_k]; +2 for 1-based rank in log2

        per_user = user_ids is not None
        per_row = per_user or (is_last is not None)  # recall / NDCG rows are needed for user and last averages
        if per_user:
            self._row_users.append(user_ids.detach().cpu().long())
        if is_last is not None:
            self._row_is_last.append(is_last.detach().cpu().bool())

        for k in self.k_list:
            # Recall@k: exact match is within top k
            recall_rows = full_match[:, :k].any(dim=-1).float()
            recall_k = recall_rows.sum().item()
            self.total_recall[k] += recall_k

            # NDCG@k: sum of matching position discounts within top k
            ndcg_rows = (full_match[:, :k] * discounts[:, :k]).sum(dim=-1)
            ndcg_k = ndcg_rows.sum().item()
            self.total_ndcg[k] += ndcg_k

            if per_row:
                self._row_recall[k].append(recall_rows.cpu())
                self._row_ndcg[k].append(ndcg_rows.cpu())

            # Hierarchical Recall@k for prefix slices
            for layer in range(1, self.num_layers + 1):
                slice_match = (predictions[:, :k, :layer] == targets[:, None, :layer]).all(dim=-1) # None equals unsqueeze(1) -> [Batch, 1, layer]
                h_recall_rows = slice_match.any(dim=-1).float()
                self.hierarchical_recall[f"h@{k}_slice_{layer}"] += h_recall_rows.sum().item()
                if per_user:
                    self._row_hier[f"h@{k}_slice_{layer}"].append(h_recall_rows.cpu())

        if seen_tuples is not None:
            self._update_seen(predictions, seen_tuples, per_user)

    def _update_seen(self, predictions, seen_tuples, per_user):
        """Counts top-k slots whose full SID tuple equals one of the row's seen tuples."""
        max_k = max(self.k_list)
        batch_size = predictions.size(0)
        preds = predictions[:, :max_k, :]
        seen_tuples = seen_tuples.to(preds.device)
        seen_valid = (seen_tuples >= 0).all(dim=-1)                                  # [B, S]
        # [B, max_k, 1, L] vs [B, 1, S, L] -> [B, max_k, S]
        eq = (preds.unsqueeze(2) == seen_tuples.unsqueeze(1)).all(dim=-1)
        slot_hit = (eq & seen_valid.unsqueeze(1)).any(dim=-1)                        # [B, max_k]
        for k in self.k_list:
            self._seen_hits[k] += slot_hit[:, :k].float().sum().item()
            self._seen_slots[k] += batch_size * k
            if per_user:
                self._row_seen[k].append(slot_hit[:, :k].any(dim=-1).cpu())

    @staticmethod
    def _mean_per_user(row_values, user_index, num_users):
        """Mean over each user's rows, then mean over users."""
        sums = torch.zeros(num_users, dtype=torch.float64).scatter_add_(0, user_index, row_values.double())
        counts = torch.zeros(num_users, dtype=torch.float64).scatter_add_(0, user_index, torch.ones_like(row_values, dtype=torch.float64))
        return float((sums / counts).mean().item())

    def _extra_metrics(self):
        """User-level, last-target and seen-item metrics (only those whose inputs were provided)."""
        extra = {}
        given = self._given

        if given["user_ids"]:
            users = torch.cat(self._row_users)
            unique_users, user_index = torch.unique(users, return_inverse=True)
            num_users = unique_users.numel()
            extra["num_users"] = num_users
            extra["recall_user"] = {k: self._mean_per_user(torch.cat(self._row_recall[k]), user_index, num_users) for k in self.k_list}
            extra["ndcg_user"] = {k: self._mean_per_user(torch.cat(self._row_ndcg[k]), user_index, num_users) for k in self.k_list}
            # per-user averaged hierarchical recall, so that h@k_slice_<layers> equals recall_user[k]
            extra["hierarchical_user"] = {key: self._mean_per_user(torch.cat(rows), user_index, num_users) for key, rows in self._row_hier.items()}
            if given["seen_tuples"]:
                # share of users with at least one seen item among their top-k slots in any of their rows
                extra["users_with_seen_at_k"] = {}
                for k in self.k_list:
                    user_hit = torch.zeros(num_users, dtype=torch.float64).scatter_reduce_(
                        0, user_index, torch.cat(self._row_seen[k]).double(), reduce="amax", include_self=True)
                    extra["users_with_seen_at_k"][k] = float(user_hit.mean().item())

        if given["is_last"]:
            last = torch.cat(self._row_is_last)
            if not last.any():
                raise ValueError("is_last given but no row is marked as a user's last row")
            extra["recall_last"] = {k: float(torch.cat(self._row_recall[k])[last].double().mean().item()) for k in self.k_list}
            extra["ndcg_last"] = {k: float(torch.cat(self._row_ndcg[k])[last].double().mean().item()) for k in self.k_list}

        if given["seen_tuples"]:
            extra["seen_at_k"] = {k: float(self._seen_hits[k] / max(1, self._seen_slots[k])) for k in self.k_list}

        return extra

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
            "total_samples": self.total_samples,
            **self._extra_metrics(),
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
    
    # latent metrics (optional): size of the quantizer input
    if input_tensor is not None:
        metrics["latent_norm"] = input_tensor.norm(dim=-1).mean()

    # residual metrics (optional)
    if input_tensor is not None and first_residual is not None and final_residual is not None:
        # compute norms over the last dimension (latent dimension) and mean over the batch/sequence dims
        input_norm = metrics["latent_norm"].clamp(min=1e-8)
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