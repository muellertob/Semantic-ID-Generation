import logging
import math
import os

import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loader import load_amazon_sequences
from data.sequence import SASRecDataset, sasrec_collate_fn
from modules.sasrec.model import SASRec, sample_negatives
from utils.wandb import get_run_name, wandb_login

logger = logging.getLogger(__name__)


def evaluate_metrics(model, dataset, device, k_list, num_items):
    """
    Compute retrieval metrics (Recall@K and NDCG@K) for the given SASRecmodel and dataset.

    Args:
        model: SASRec model
        dataset: SASRecDataset (eval or test split)
        device: torch device
        k_list: list of K values for metrics
        num_items: total number of items

    Returns:
        dict with 'recall@k' and 'ndcg@k' for each k in k_list
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=sasrec_collate_fn)

    recalls = {k: 0.0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    total = 0

    with torch.no_grad():
        for batch in loader:
            item_seq = batch['item_seq'].to(device)     # [B, max_len]
            target = batch['target_item'].to(device)    # [B]
            
            B = item_seq.shape[0]
            scores = model(item_seq)  # [B, max_len, num_items+1]

            # sequences are left-padded, so the last position is the one we want to evaluate
            last_scores = scores[:, -1, :]  # [B, num_items+1]

            # exclude padding item (index 0)
            last_scores[:, 0] = float('-inf')

            # compute metrics
            for i in range(B):
                target_id = target[i].item()
                target_score = last_scores[i, target_id].item()
                rank = (last_scores[i] > target_score).sum().item() + 1

                for k in k_list:
                    if rank <= k:
                        recalls[k] += 1.0
                        ndcgs[k] += 1.0 / math.log2(rank + 1)

            total += B

    metrics = {}
    for k in k_list:
        metrics[f'recall@{k}'] = recalls[k] / total if total > 0 else 0.0
        metrics[f'ndcg@{k}'] = ndcgs[k] / total if total > 0 else 0.0

    return metrics


def run_training(config_path, resume_path=None):
    """
    Main SASRec training loop.

    Args:
        config_path: path to YAML config file
        resume_path: optional path to checkpoint for resuming
    """
    config = OmegaConf.load(config_path)
    logging.basicConfig(level=logging.INFO)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # config values
    sc = config.sasrec
    hidden_dim = sc.hidden_dim
    num_blocks = sc.num_blocks
    num_heads = sc.num_heads
    max_seq_len = sc.max_seq_len
    dropout = sc.dropout
    batch_size = sc.batch_size
    lr = sc.learning_rate
    num_epochs = sc.num_epochs
    num_workers = getattr(sc, 'num_workers', 0)

    # weights & biases
    use_wandb = config.general.get('use_wandb', False)
    if use_wandb:
        wandb_login()
        wandb.init(
            project=config.general.get('wandb_project_sasrec', 'sasrec-benchmark'),
            config=OmegaConf.to_container(config, resolve=True),
            resume="never"
        )

    model_id = get_run_name(fallback=f"sasrec-{config.data.category}")
    logger.info(f"SASRec run name: {model_id}")

    # data
    sequences, num_users, num_items = load_amazon_sequences(category=config.data.category)
    logger.info(f"Loaded {num_users} users, {num_items} items")

    train_ds = SASRecDataset(sequences, num_items=num_items, max_len=max_seq_len, mode='train')
    eval_ds = SASRecDataset(sequences, num_items=num_items, max_len=max_seq_len, mode='eval')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=sasrec_collate_fn, num_workers=num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=sasrec_collate_fn, num_workers=num_workers)

    # model
    model = SASRec(
        num_items=num_items,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if use_wandb:
        wandb.watch(model, log="all")

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # resume
    start_epoch = 0
    best_recall5 = 0.0
    patience_counter = 0
    patience = 15 

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_recall5 = checkpoint.get('best_recall5', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        logger.info(f"Resumed from epoch {start_epoch}, best recall@5={best_recall5:.4f}")

    k_list = [1, 5, 10]

    # training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            item_seq = batch['item_seq'].to(device)
            target_item = batch['target_item'].to(device)
                
            # build training targets: for each position, the positive is the next item
            pos_items = torch.zeros_like(item_seq)
            pos_items[:, :-1] = item_seq[:, 1:]
            pos_items[:, -1] = target_item
            
            # ensure that where item_seq is padding, pos_items is also padding
            pos_items[item_seq == 0] = 0

            # sample negatives
            neg_items = sample_negatives(pos_items, num_items).to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(item_seq, pos_items, neg_items)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        log_dict = {'train_loss': avg_loss, 'epoch': epoch}

        # compute eval loss
        model.eval()
        eval_loss = 0.0
        eval_batches = 0
        with torch.no_grad():
            for batch in eval_loader:
                item_seq = batch['item_seq'].to(device)
                target_item = batch['target_item'].to(device)
                pos_items = torch.zeros_like(item_seq)
                pos_items[:, :-1] = item_seq[:, 1:]
                pos_items[:, -1] = target_item
                pos_items[item_seq == 0] = 0
                neg_items = sample_negatives(pos_items, num_items).to(device)
                loss = model.compute_loss(item_seq, pos_items, neg_items)
                eval_loss += loss.item()
                eval_batches += 1
        avg_eval_loss = eval_loss / eval_batches if eval_batches > 0 else 0.0
        log_dict['eval_loss'] = avg_eval_loss

        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}")

        # compute metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_metrics(model, eval_ds, device, k_list, num_items)
            for key, val in metrics.items():
                log_dict[key] = val

            logger.info(f"Recall@5={metrics.get('recall@5', 0):.4f} | NDCG@5={metrics.get('ndcg@5', 0):.4f}")

            # early stopping check
            current_recall5 = metrics.get('recall@5', 0.0)
            if current_recall5 > best_recall5:
                best_recall5 = current_recall5
                patience_counter = 0

                if getattr(config.general, 'save_model', True):
                    os.makedirs('models', exist_ok=True)
                    save_path = f'models/{model_id}_best.pt'
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_recall5': best_recall5,
                        'patience_counter': patience_counter,
                        'config': OmegaConf.to_container(config),
                    }, save_path)
                    logger.info(f"New best Recall@5 ({current_recall5:.4f})! Model saved to {save_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement in Recall@5 for {patience_counter} evaluations.")
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered. Recall@5 hasn't improved for {patience} evaluations.")
                    if use_wandb:
                        wandb.log(log_dict)
                    break

        if use_wandb:
            wandb.log(log_dict)

    logger.info("Training Finished.")
    logger.info(f"Best Recall@5: {best_recall5:.4f}")
    if use_wandb:
        wandb.finish()
