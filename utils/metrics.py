import torch
import numpy as np

class MetricAccumulator:
    """
    Accumulates retrieval metrics across multiple batches for accurate averaging,
    including hierarchical metrics for each slice of the target sequence.
    """
    def __init__(self, k_list=[1, 5, 10], num_layers=4):
        self.k_list = k_list
        self.num_layers = num_layers
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.total_recall = {k: 0.0 for k in self.k_list}
        self.total_ndcg = {k: 0.0 for k in self.k_list}
        
        # Hierarchical metrics
        self.hierarchical_recall = {f"h@{k}_slice_{i}": 0.0 for k in self.k_list for i in range(1, self.num_layers + 1)}

    def update(self, predictions, targets):
        """
        Update the accumulator with results from a new batch.
        
        Args:
            predictions (torch.Tensor): [Batch, Beam_Size, Codebook_Layers]
            targets (torch.Tensor): [Batch, 1, Codebook_Layers] or [Batch, Codebook_Layers]
        """
        batch_size = targets.size(0)
        self.total_samples += batch_size

        # normalize targets to [Batch, Codebook_Layers]
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        # convert tensors to lists of tuples for easier comparison
        targets_list = [tuple(t.tolist()) for t in targets]
        
        for i in range(batch_size):
            target_tuple = targets_list[i]
            # get top max(k) predictions
            max_k = max(self.k_list)
            pred_tuples = [tuple(p.tolist()) for p in predictions[i][:max_k]]
            
            for k in self.k_list:
                current_preds = pred_tuples[:k]
                
                # Check for exact full match
                if target_tuple in current_preds:
                    self.total_recall[k] += 1.0
                    rank = current_preds.index(target_tuple) + 1
                    self.total_ndcg[k] += 1.0 / np.log2(rank + 1)
                
                # Hierarchical checks
                for layer in range(1, self.num_layers + 1):
                    target_slice = target_tuple[:layer]
                    pred_slices = [p[:layer] for p in current_preds]
                    if target_slice in pred_slices:
                        self.hierarchical_recall[f"h@{k}_slice_{layer}"] += 1.0

    def compute(self):
        """Returns the final averaged metrics."""
        if self.total_samples == 0:
            return {
                "recall": {k: 0.0 for k in self.k_list}, 
                "ndcg": {k: 0.0 for k in self.k_list},
                "hierarchical": {k: 0.0 for k in self.hierarchical_recall.keys()}
            }
            
        return {
            "recall": {k: float(v / self.total_samples) for k, v in self.total_recall.items()},
            "ndcg": {k: float(v / self.total_samples) for k, v in self.total_ndcg.items()},
            "hierarchical": {k: float(v / self.total_samples) for k, v in self.hierarchical_recall.items()},
            "total_samples": self.total_samples
        }