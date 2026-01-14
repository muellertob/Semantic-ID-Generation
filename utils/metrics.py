import torch
import numpy as np

def calculate_recall_at_k(predictions, targets, k_list=[5, 10]):
    """
    Calculates Recall@K.
    
    Args:
        predictions (torch.Tensor): Tensor of shape [Batch, Beam_Size, Codebook_Layers] 
                                    containing generated semantic IDs.
        targets (torch.Tensor): Tensor of shape [Batch, 1, Codebook_Layers] 
                                containing ground truth semantic IDs.
        k_list (list): List of K values to calculate Recall for.
        
    Returns:
        dict: Dictionary mapping k to Recall@k value.
    """
    batch_size = targets.size(0)
    results = {k: 0.0 for k in k_list}
    
    # convert tensors to lists of tuples for easier comparison
    targets_list = [tuple(t.tolist()) for t in targets.squeeze(1)]
    
    for i in range(batch_size):
        target_tuple = targets_list[i]
        
        # get top max(k) predictions
        max_k = max(k_list)
        pred_tuples = [tuple(p.tolist()) for p in predictions[i][:max_k]]
        
        for k in k_list:
            if target_tuple in pred_tuples[:k]:
                results[k] += 1.0
                
    for k in k_list:
        results[k] /= batch_size
        
    return results

def calculate_ndcg_at_k(predictions, targets, k_list=[5, 10]):
    """
    Calculates NDCG@K.
    
    Args:
        predictions (torch.Tensor): Tensor of shape [Batch, Beam_Size, Codebook_Layers]
        targets (torch.Tensor): Tensor of shape [Batch, 1, Codebook_Layers]
        k_list (list): List of K values.
        
    Returns:
        dict: Dictionary mapping k to NDCG@k value.
    """
    batch_size = targets.size(0)
    results = {k: 0.0 for k in k_list}
    
    targets_list = [tuple(t.tolist()) for t in targets.squeeze(1)]
    
    for i in range(batch_size):
        target_tuple = targets_list[i]
        max_k = max(k_list)
        pred_tuples = [tuple(p.tolist()) for p in predictions[i][:max_k]]
        
        for k in k_list:
            current_preds = pred_tuples[:k]
            if target_tuple in current_preds:
                rank = current_preds.index(target_tuple) + 1
                dcg = 1.0 / np.log2(rank + 1)
                # IDCG is always 1.0 because we have only one relevant item
                ndcg = dcg
                results[k] += ndcg
            else:
                results[k] += 0.0
                
    for k in k_list:
        results[k] /= batch_size
        
    return results
