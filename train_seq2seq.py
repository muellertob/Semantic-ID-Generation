import logging
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import os

from modules.seq2seq import TigerSeq2Seq
from data.loader import load_amazon_sequences
from data.sequence import SemanticIDSequenceDataset, collate_fn
from utils.wandb import wandb_init
from utils.model_id_generation import generate_model_id
from utils.metrics import MetricAccumulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, scheduler, epoch, metric_val, path, metric_name="recall@5"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        metric_name: metric_val,
    }, path)

def evaluate_loss(model, dataloader, device):
    """
    Compute validation loss and return a breakdown of token-level losses.
    """
    model.eval()
    total_loss = 0
    total_loss_per_token = None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Loss"):
            history_tuples = batch['history_tuples'].to(device)
            target_tuples = batch['target_tuples'].to(device)
            user_ids = batch['user_ids'].to(device)
            
            outputs = model(
                history_tuples=history_tuples,
                target_tuples=target_tuples,
                user_ids=user_ids
            )
            
            total_loss += outputs['loss'].item()
            
            if 'unreduced_loss' in outputs:
                # [Batch, SeqLen] -> sum over batch -> [SeqLen]
                batch_token_loss = outputs['unreduced_loss'].sum(dim=0).cpu()
                if total_loss_per_token is None:
                    total_loss_per_token = batch_token_loss
                else:
                    total_loss_per_token += batch_token_loss

    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_batches
    
    token_losses = {}
    if total_loss_per_token is not None:
        # average over total samples to get true token-level loss
        avg_token_loss = total_loss_per_token / num_samples
        for i, val in enumerate(avg_token_loss):
            token_losses[f"eval_loss_token_{i+1}"] = val.item()
            
    return avg_loss, token_losses

def compute_metrics(model, dataloader, device, k_list=[5, 10]):
    """
    Compute retrieval metrics using Beam Search and MetricAccumulator.
    """
    model.eval()
    accumulator = MetricAccumulator(k_list=k_list)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Metrics"):
            history_tuples = batch['history_tuples'].to(device)
            target_tuples = batch['target_tuples'].to(device)
            user_ids = batch['user_ids'].to(device)
            
            # RETRIEVAL METRICS
            # use max K for beam size
            max_k = max(k_list)
            predictions = model.beam_search(
                history_tuples=history_tuples,
                user_ids=user_ids,
                beam_size=max_k
            )
            
            # convert global IDs to raw semantic IDs by subtracting the item offsets
            predictions = predictions - model.item_offsets.view(1, 1, -1)
            
            accumulator.update(predictions, target_tuples)

    metrics_report = accumulator.compute()
    return metrics_report['recall'], metrics_report['ndcg'], metrics_report['hierarchical']

def run_training(config_path, semantic_ids_path, resume_path=None, warmup_steps_override=None):
    """
    Orchestrate TIGER Seq2Seq training.
    """
    config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # initialize wandb
    if config.general.get('use_wandb', False):
        wandb_init(config)
        
    # load semantic IDs
    logger.info(f"Loading Semantic IDs from {semantic_ids_path}")
    if not os.path.exists(semantic_ids_path):
        raise FileNotFoundError(f"Semantic IDs file not found at {semantic_ids_path}")
        
    semids_data = torch.load(semantic_ids_path, map_location='cpu', weights_only=False)
    semantic_ids = semids_data['semantic_ids'] # [num_items, codebook_layers]
    
    logger.info(f"Loaded Semantic IDs with shape: {semantic_ids.shape}")

    # load sequential data
    logger.info("Loading User History Data...")
    sequences, num_users, num_items = load_amazon_sequences(category=config.data.category)
    logger.info(f"Loaded history for {num_users} users and {num_items} items")
    
    # create datasets and dataloaders
    train_dataset = SemanticIDSequenceDataset(
        history_data=sequences,
        semantic_ids=semantic_ids,
        max_len=config.seq2seq.get('max_history_len', 20),
        mode='train'
    )
    
    eval_dataset = SemanticIDSequenceDataset(
        history_data=sequences,
        semantic_ids=semantic_ids,
        max_len=config.seq2seq.get('max_history_len', 20),
        mode='eval'
    )
    
    num_workers = config.seq2seq.get('num_workers', 0)
    persistent_workers = (num_workers > 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.seq2seq.get('batch_size', 256),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    
    # dataloader for loss evaluation
    eval_loader_loss = DataLoader(
        eval_dataset,
        batch_size=config.seq2seq.get('batch_size', 256),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    
    # dataloader for metrics evaluation
    # uses smaller batch size to increase throughput and avoid OOM during beam search
    metric_batch_size = max(1, config.seq2seq.get('batch_size', 256) // 2)
    logger.info(f"Using separate eval loaders: Loss BS={config.seq2seq.get('batch_size', 256)}, Metrics BS={metric_batch_size}")
    
    eval_loader_metrics = DataLoader(
        eval_dataset,
        batch_size=metric_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    
    # initialize model, defaulting to TIGER paper specs
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
    
    if config.general.get('use_wandb', False):
        wandb.watch(model, log="all")

    # define optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.seq2seq.get('learning_rate', 1e-3),
        weight_decay=config.seq2seq.get('weight_decay', 0.0001)
    )
    
    start_epoch = 0
    global_step = 0
    best_recall_at_5 = 0.0
    
    # RESUME CHECKPOINT
    if resume_path:
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint file not found at {resume_path}")
            
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # checkpoint saves the epoch that just finished, so we start from the next one
        start_epoch = checkpoint['epoch'] + 1
        
        # restore best recall if available
        if 'recall@5' in checkpoint:
            best_recall_at_5 = checkpoint['recall@5']
        elif 'loss' in checkpoint:
            logger.info("Old loss-based checkpoint detected. Resetting best recall to 0.0.")
            
        # estimate global step: start_epoch * steps_per_epoch
        steps_per_epoch = len(train_loader)
        global_step = start_epoch * steps_per_epoch
        
        logger.info(f"Resumed from Epoch {start_epoch}, Global Step {global_step}, Best Recall@5 {best_recall_at_5}")

    num_epochs = config.seq2seq.get('num_epochs', 2300)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    # linear warmup with cosine decay
    use_lr_scheduler = config.seq2seq.get('use_lr_scheduler', True)
    
    if use_lr_scheduler:
        if warmup_steps_override is not None:
             warmup_steps = warmup_steps_override
        else:
             warmup_steps = config.seq2seq.get('warmup_steps', 10000)
             
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        
        decay_steps = total_steps - warmup_steps
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, decay_steps), eta_min=1e-5
        )
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )
        
        # fast-forward scheduler if resuming
        if global_step > 0:
            for _ in range(global_step):
                scheduler.step()
    else:
        scheduler = None
    
    # training loop
    logger.info("Starting Training Loop...")
    num_epochs = config.seq2seq.get('num_epochs', 2300)
    
    early_stopping_patience = 7 # cycles of metrics evaluation (every 5 epochs)
    recall_no_improve = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            history_tuples = batch['history_tuples'].to(device)
            target_tuples = batch['target_tuples'].to(device)
            user_ids = batch['user_ids'].to(device)
            
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(
                history_tuples=history_tuples,
                target_tuples=target_tuples,
                user_ids=user_ids
            )
            
            loss = outputs['loss']
            
            # backward pass
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step() # step scheduler every batch
            global_step += 1
            
            total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': loss.item(), 'lr': current_lr})
            
        avg_train_loss = total_loss / len(train_loader)
        
        avg_eval_loss, eval_token_losses = evaluate_loss(model, eval_loader_loss, device)
        
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | LR: {current_lr:.6f}")
        
        wandb_log = {
            "epoch": epoch, 
            "train_loss": avg_train_loss,
            "eval_loss": avg_eval_loss,
            "learning_rate": current_lr
        }
        
        # add token-level losses
        wandb_log.update(eval_token_losses)
        
        # CHECKPOINTING & EARLY STOPPING based on retrieval metrics
        stop_training = False
        # compute retrieval metrics every 5 epochs and on the last epoch
        if not stop_training and ((epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs):
            logger.info("Computing retrieval metrics (Beam Search)...")
            avg_recall, avg_ndcg, avg_hierarchical = compute_metrics(model, eval_loader_metrics, device)
            
            logger.info(f"Recall: {avg_recall} | NDCG: {avg_ndcg}")
            
            current_recall_at_5 = avg_recall.get(5, 0.0)
            if current_recall_at_5 > best_recall_at_5:
                best_recall_at_5 = current_recall_at_5
                recall_no_improve = 0 # reset patience
                if config.general.get('save_model', True):
                    config_name = os.path.splitext(os.path.basename(config_path))[0]
                    model_id = f"{config_name}_{config.data.dataset}_{config.data.category}_best"
                    save_path = f"models/{model_id}.pt"
                    save_checkpoint(model, optimizer, scheduler, epoch, current_recall_at_5, save_path, "recall@5")
                    logger.info(f"New best Recall@5 achieved ({current_recall_at_5:.4f})! Model saved to {save_path}")
            else:
                recall_no_improve += 1
                logger.info(f"No improvement in Recall@5 for {recall_no_improve} evaluations.")
                if recall_no_improve >= early_stopping_patience:
                    logger.info(f"Early stopping triggered. Recall@5 hasn't improved for {early_stopping_patience} evaluations.")
                    stop_training = True

            # merge metrics into wandb log
            for k, v in avg_recall.items():
                wandb_log[f"recall@{k}"] = v
            for k, v in avg_ndcg.items():
                wandb_log[f"ndcg@{k}"] = v
            for k, v in avg_hierarchical.items():
                wandb_log[k] = v
        
        if config.general.get('use_wandb', False):
            wandb.log(wandb_log)
            
        if stop_training:
            break

    logger.info("Training Finished.")
    if config.general.get('use_wandb', False):
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TIGER Seq2Seq Model")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--semids', type=str, required=True, help='Path to generated Semantic IDs (.pt file)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--warmup_steps', type=int, help='Override warmup_steps from config')
    args = parser.parse_args()
    
    run_training(args.config, args.semids, args.resume, args.warmup_steps)