"""
SASRec testing script.

Evaluates a trained SASRec model on the test split using full-rank evaluation.

Usage:
    python main.py test-sasrec --config config/config_amazon_sasrec.yaml --model_path models/<id>.pt
"""
import logging
import os

import torch
from omegaconf import OmegaConf

from data.loader import load_amazon_sequences
from data.sequence import SASRecDataset
from modules.sasrec.model import SASRec
from train_sasrec import evaluate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_testing(config_path, model_path, overrides=None):
    """
    Test SASRec model on the test split.
    """
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
        
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    sc = config.sasrec
    hidden_dim = sc.hidden_dim
    num_blocks = sc.num_blocks
    num_heads = sc.num_heads
    max_seq_len = sc.max_seq_len
    dropout = sc.dropout

    # data
    sequences, num_users, num_items = load_amazon_sequences(category=config.data.category)
    logger.info(f"Loaded {num_users} users, {num_items} items")

    test_ds = SASRecDataset(sequences, num_items=num_items, max_len=max_seq_len, mode='test')

    # model
    model = SASRec(
        num_items=num_items,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)

    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"Loaded model from {model_path}")

    k_list = [1, 5, 10]
    logger.info("Starting evaluation on test set...")
    metrics = evaluate_metrics(model, test_ds, device, k_list, num_items)

    logger.info("Test Results:")
    for k in k_list:
        logger.info(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}  NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SASRec Model")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    args, overrides = parser.parse_known_args()

    run_testing(args.config, args.model_path, overrides=overrides)
