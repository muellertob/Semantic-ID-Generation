import logging
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os

from modules.seq2seq import TigerSeq2Seq
from data.loader import load_amazon_sequences
from data.sequence import SemanticIDSequenceDataset, collate_fn
from train_seq2seq import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_testing(config_path, semantic_ids_path, model_path):
    """
    Test TIGER Seq2Seq model on the test split.
    """
    config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # load semantic IDs
    logger.info(f"Loading Semantic IDs from {semantic_ids_path}")
    if not os.path.exists(semantic_ids_path):
        raise FileNotFoundError(f"Semantic IDs file not found at {semantic_ids_path}")
        
    semids_data = torch.load(semantic_ids_path, map_location='cpu', weights_only=False)
    semantic_ids = semids_data['semantic_ids'] # [num_items, codebook_layers]
    
    logger.info(f"Loaded Semantic IDs with shape: {semantic_ids.shape}")

    # load sequential data
    logger.info("Loading User History Data (Test Split)...")
    sequences, num_users, num_items = load_amazon_sequences(category=config.data.category)
    logger.info(f"Loaded history for {num_users} users and {num_items} items")
    
    # create dataset and dataloader for TEST split
    test_dataset = SemanticIDSequenceDataset(
        history_data=sequences,
        semantic_ids=semantic_ids,
        max_len=config.seq2seq.get('max_history_len', 20),
        mode='test'
    )
    
    num_workers = config.seq2seq.get('num_workers', 0)
    persistent_workers = (num_workers > 0)
    
    # uses smaller batch size to increase throughput and avoid OOM during beam search
    metric_batch_size = max(1, config.seq2seq.get('batch_size', 256) // 2)
    logger.info(f"Using Metrics Batch Size = {metric_batch_size}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=metric_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    
    # initialize model
    model = TigerSeq2Seq(
        codebook_layers=config.model.get('num_codebook_layers', 3),
        codebook_size=config.model.get('codebook_clusters', 256),
        user_tokens=config.seq2seq.get('user_tokens', 2000),
        d_model=config.seq2seq.get('d_model', 128),
        d_kv=config.seq2seq.get('d_kv', 64),
        d_ff=config.seq2seq.get('d_ff', 1024),
        num_layers=config.seq2seq.get('num_layers', 4),
        num_heads=config.seq2seq.get('num_heads', 6),
        dropout=config.seq2seq.get('dropout', 0.1),
        activation_fn=config.seq2seq.get('activation_fn', "relu")
    )
    model.to(device)
    
    # register the codebooks for constrained generation
    model.set_codebooks(semantic_ids.to(device))
    
    # load model checkpoint
    logger.info(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    logger.info("Starting Evaluation on Test Set...")

    # compute metrics with standard k list
    k_list = [5, 10, 20]
    avg_recall, avg_ndcg, avg_hierarchical = compute_metrics(model, test_loader, device, k_list=k_list)
    
    logger.info("Evaluation Results:")
    for k in k_list:
        logger.info(f"Recall@{k}: {avg_recall[k]:.4f}")
        logger.info(f"NDCG@{k}:   {avg_ndcg[k]:.4f}")
        
    logger.info("Hierarchical Results:")
    for key, val in avg_hierarchical.items():
        logger.info(f"{key}: {val:.4f}")
        
    return avg_recall, avg_ndcg, avg_hierarchical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TIGER Seq2Seq Model")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--semids', type=str, required=True, help='Path to generated Semantic IDs (.pt file)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    run_testing(args.config, args.semids, args.model_path)
