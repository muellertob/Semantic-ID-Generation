import logging
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os
from collections import Counter
from tqdm import tqdm

from data.loader import load_amazon_sequences
from data.sequence import SemanticIDSequenceDataset, collate_fn
from utils.metrics import MetricAccumulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_popularity(history_data, num_items):
    """
    Computes item popularity from the training set.
    
    Args:
        history_data: the loaded sequences dict containing 'train', 'eval', 'test'
        num_items: total number of items in the catalog
        
    Returns:
        List of item_ids sorted by frequency (most popular first)
    """
    logger.info("Computing Most Popular items from training set...")
    item_counts = Counter()
    
    # only look at the training set to prevent data leakage.
    train_seqs = history_data['train']['itemId']
    train_targets = history_data['train']['itemId_fut']
    
    # count occurrences in historical sequences
    for seq in train_seqs:
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        # filter out invalid item IDs and count valid ones
        item_counts.update([item for item in seq if 0 <= item < num_items])
        
    # count occurrences in targets
    for target in train_targets:
        if isinstance(target, torch.Tensor):
            target = target.item()
        if 0 <= target < num_items:
            item_counts[target] += 1
            
    # sort items by frequency
    sorted_items = [item for item, count in item_counts.most_common()]
    
    # if there are items in the catalog that never appeared in train, append them at the end
    unseen_items = set(range(num_items)) - set(sorted_items)
    sorted_items.extend(list(unseen_items))

    return sorted_items

def run_baselines(config_path, semantic_ids_path, k_list=[5, 10, 20]):
    config = OmegaConf.load(config_path)
    
    logger.info(f"Loading Semantic IDs from {semantic_ids_path}")
    if not os.path.exists(semantic_ids_path):
        raise FileNotFoundError(f"Semantic IDs file not found at {semantic_ids_path}")
        
    semids_data = torch.load(semantic_ids_path, map_location='cpu', weights_only=False)
    semantic_ids = semids_data['semantic_ids'] # [num_items, tuple_size]
    num_items = semantic_ids.shape[0]
    tuple_size = semantic_ids.shape[1]
    
    logger.info("Loading User History Data (Test Split)...")
    sequences, num_users, loaded_num_items = load_amazon_sequences(category=config.data.category)
    
    # compute most popular items from train split
    popular_item_ids = compute_popularity(sequences, num_items)
    max_k = max(k_list)
    top_k_popular_ids = popular_item_ids[:max_k]
    
    # convert popular raw IDs to their Semantic ID tuples
    top_k_popular_semids = semantic_ids[top_k_popular_ids] # shape: [max_k, tuple_size]
    
    test_dataset = SemanticIDSequenceDataset(
        history_data=sequences,
        semantic_ids=semantic_ids,
        max_len=config.seq2seq.get('max_history_len', 20),
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    random_accumulator = MetricAccumulator(k_list=k_list, num_layers=tuple_size)
    mostpop_accumulator = MetricAccumulator(k_list=k_list, num_layers=tuple_size)
    
    logger.info("Evaluating Baselines...")
    for batch in tqdm(test_loader, desc="Testing"):
        target_tuples = batch['target_tuples'] # [Batch, 1, tuple_size]
        batch_size = target_tuples.size(0)
        
        # RANDOM BASELINE PREDICTIONS
        random_raw_ids = torch.randint(0, num_items, (batch_size, max_k))
        random_predictions = semantic_ids[random_raw_ids] # shape: [Batch, max_k, tuple_size]
        
        # MOST-POPULAR BASELINE PREDICTIONS
        # every user gets the exact same top K popular items
        # expand [max_k, tuple_size] to [Batch, max_k, tuple_size]
        mostpop_predictions = top_k_popular_semids.unsqueeze(0).expand(batch_size, -1, -1)
        
        random_accumulator.update(random_predictions, target_tuples)
        mostpop_accumulator.update(mostpop_predictions, target_tuples)
        
    random_metrics = random_accumulator.compute()
    mostpop_metrics = mostpop_accumulator.compute()
    
    logger.info("="*40)
    logger.info("RANDOM BASELINE RESULTS")
    logger.info("="*40)
    for k in k_list:
        logger.info(f"Recall@{k}: {random_metrics['recall'][k]:.4f}")
        logger.info(f"NDCG@{k}:   {random_metrics['ndcg'][k]:.4f}")
    logger.info("Hierarchical Results:")
    for key, val in random_metrics['hierarchical'].items():
        logger.info(f"{key}: {val:.4f}")
    logger.info("\n")
        
    logger.info("="*40)
    logger.info("MOST-POPULAR BASELINE RESULTS")
    logger.info("="*40)
    for k in k_list:
        logger.info(f"Recall@{k}: {mostpop_metrics['recall'][k]:.4f}")
        logger.info(f"NDCG@{k}:   {mostpop_metrics['ndcg'][k]:.4f}")
    logger.info("Hierarchical Results:")
    for key, val in mostpop_metrics['hierarchical'].items():
        logger.info(f"{key}: {val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Naive Baselines")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--semids', type=str, required=True, help='Path to generated Semantic IDs (.pt file)')
    args = parser.parse_args()
    
    run_baselines(args.config, args.semids)