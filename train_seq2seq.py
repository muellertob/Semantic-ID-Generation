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
from utils.metrics import calculate_recall_at_k, calculate_ndcg_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def evaluate(model, dataloader, device, k_list=[5, 10]):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    total_recall = {k: 0.0 for k in k_list}
    total_ndcg = {k: 0.0 for k in k_list}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            history_tuples = batch['history_tuples'].to(device)
            target_tuples = batch['target_tuples'].to(device)
            user_ids = batch['user_ids'].to(device)
            
            outputs = model(
                history_tuples=history_tuples,
                target_tuples=target_tuples,
                user_ids=user_ids
            )
            
            total_loss += outputs['loss'].item()
            
            # RETRIEVAL METRICS
            # use max K for beam size
            max_k = max(k_list)
            predictions = model.beam_search(
                history_tuples=history_tuples,
                user_ids=user_ids,
                beam_size=max_k
            )
            
            recall_results = calculate_recall_at_k(predictions, target_tuples, k_list)
            ndcg_results = calculate_ndcg_at_k(predictions, target_tuples, k_list)
            
            for k in k_list:
                total_recall[k] += recall_results[k]
                total_ndcg[k] += ndcg_results[k]

    avg_loss = total_loss / len(dataloader)
    avg_recall = {k: v / len(dataloader) for k, v in total_recall.items()}
    avg_ndcg = {k: v / len(dataloader) for k, v in total_ndcg.items()}
    
    return avg_loss, avg_recall, avg_ndcg

def run_training(config_path, semantic_ids_path):
    """
    Orchestrate TIGER Seq2Seq training.
    """
    config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # initialize wandb
    if config.general.use_wandb:
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
        max_len=config.seq2seq.max_history_len,
        mode='train'
    )
    
    eval_dataset = SemanticIDSequenceDataset(
        history_data=sequences,
        semantic_ids=semantic_ids,
        max_len=config.seq2seq.max_history_len,
        mode='eval'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.seq2seq.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.seq2seq.get('num_workers', 0)
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.seq2seq.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.seq2seq.get('num_workers', 0)
    )
    
    # initialize model
    model = TigerSeq2Seq(
        codebook_layers=config.model.num_codebook_layers,
        codebook_size=config.model.codebook_clusters,
        user_tokens=config.seq2seq.user_tokens,
        d_model=config.seq2seq.d_model,
        num_layers=config.seq2seq.num_layers,
        num_heads=config.seq2seq.num_heads,
        dropout=config.seq2seq.dropout,
        activation_fn=config.seq2seq.activation_fn
    )
    model.to(device)
    
    # register the codebooks for constrained generation
    model.set_codebooks(semantic_ids.to(device))
    
    if config.general.use_wandb:
        wandb.watch(model, log="all")

    # define optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.seq2seq.learning_rate
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # training loop
    logger.info("Starting Training Loop...")
    num_epochs = config.seq2seq.num_epochs
    
    best_eval_loss = float('inf')
    early_stopping_patience = 10
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
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
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        
        # validation
        avg_eval_loss, avg_recall, avg_ndcg = evaluate(model, eval_loader, device)
        
        # update scheduler (ReduceLROnPlateau takes metric)
        scheduler.step(avg_eval_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | LR: {current_lr:.6f}")
        logger.info(f"Recall: {avg_recall} | NDCG: {avg_ndcg}")
        
        if config.general.use_wandb:
            wandb_log = {
                "epoch": epoch, 
                "train_loss": avg_train_loss,
                "eval_loss": avg_eval_loss,
                "learning_rate": current_lr
            }
            # add metrics to wandb log
            for k, v in avg_recall.items():
                wandb_log[f"recall@{k}"] = v
            for k, v in avg_ndcg.items():
                wandb_log[f"ndcg@{k}"] = v
            
            wandb.log(wandb_log)
            
        # checkpoint and early stopping
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            epochs_no_improve = 0
            if config.general.save_model:
                model_id = f"tiger_{config.data.dataset}_{config.data.category}_best"
                save_path = f"models/{model_id}.pt"
                save_checkpoint(model, optimizer, epoch, avg_eval_loss, save_path)
                logger.info(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in eval loss for {epochs_no_improve} epochs.")
            if epochs_no_improve >= early_stopping_patience:
                logger.info(f"Early stopping triggered. Eval loss hasn't improved for {early_stopping_patience} epochs.")
                break

    logger.info("Training Finished.")
    if config.general.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TIGER Seq2Seq Model")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--semids', type=str, required=True, help='Path to generated Semantic IDs (.pt file)')
    args = parser.parse_args()
    
    run_training(args.config, args.semids)