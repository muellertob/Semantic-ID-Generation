import logging
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import os
import json
from functools import partial
import uuid

from modules.recommender import TigerSeq2Seq
from data.loader import load_amazon_sequences
from data.sequence import SemanticIDSequenceDataset, collate_fn, collate_fn_with_augmentation
from utils.wandb import wandb_init, get_run_name, log_model_artifact
from utils.model_id_generation import generate_model_id
from utils.metrics import MetricAccumulator
from utils.seed import set_seed, seed_worker, get_seeded_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# enable TensorFloat-32 (TF32) on CUDA if available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metric_val, path, metric_name="recall@5"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        metric_name: metric_val,
    }, path)

def evaluate_loss(model, dataloader, device):
    """
    Compute validation loss and return a breakdown of token-level losses.
    """
    model.eval()
    total_loss = 0
    total_loss_per_token = None
    is_cuda = (device.type == 'cuda')
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Loss"):
            history_tuples = batch['history_tuples'].to(device, non_blocking=True)
            target_tuples = batch['target_tuples'].to(device, non_blocking=True)
            user_ids = batch['user_ids'].to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=is_cuda):
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

def run_training(config_path, semantic_ids_path, resume_path=None, warmup_steps_override=None, overrides=None):
    """
    Orchestrate TIGER Seq2Seq training.
    """
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, cli_conf)
        
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    seed = config.general.get('seed', 42)
    set_seed(seed)
    
    resume_optimizer = config.seq2seq.get('resume_optimizer', True)
    early_stopping = config.seq2seq.get('early_stopping', True)
    device_str = config.general.get('device', None)
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    is_cuda = (device.type == 'cuda')
    
    logger.info(f"Using device: {device}")
    
    # load semantic IDs first to get lineage info
    logger.info(f"Loading Semantic IDs from {semantic_ids_path}")
    if not os.path.exists(semantic_ids_path):
        raise FileNotFoundError(f"Semantic IDs file not found at {semantic_ids_path}")
        
    semids_data = torch.load(semantic_ids_path, map_location='cpu', weights_only=False)
    semantic_ids = semids_data['semantic_ids'] # [num_items, codebook_layers]
    
    logger.info(f"Loaded Semantic IDs with shape: {semantic_ids.shape}")
    parent_run_id = semids_data.get('parent_quantizer_run_id', None)
    parent_url = semids_data.get('parent_quantizer_run_url', None)

    # auto-extract sid_type from metadata, fallback to config general.sid_type
    if 'sid_type' in semids_data:
        extracted_sid_type = semids_data['sid_type']
    elif 'config' in semids_data and semids_data['config'].general.get('sid_type', None):
        extracted_sid_type = semids_data['config'].general.sid_type
    else:
        extracted_sid_type = "Unknown"

    # generate recommender run names (local checkpoint is static, WandB is unique)
    short_id = str(uuid.uuid4())[:8]
    if extracted_sid_type:
        recommender_run_name = f"{extracted_sid_type}_seed_{seed}"
        recommender_run_name_wandb = f"{extracted_sid_type}-seed-{seed}-{short_id}"
        group_name = extracted_sid_type
        tags = [extracted_sid_type, f"seed-{seed}"]
    else:
        recommender_run_name = generate_model_id(config)
        recommender_run_name_wandb = f"{recommender_run_name}-{short_id}"
        group_name = None
        tags = None
        
    # initialize wandb with full details and tracking overrides
    if config.general.get('use_wandb', False):
        if parent_run_id:
            OmegaConf.set_struct(config, False)
            config.general.parent_quantizer_run_id = parent_run_id
            OmegaConf.set_struct(config, True)
            
        wandb_init(
            config, 
            project=config.general.wandb_project_recommender, 
            run_name=recommender_run_name_wandb,
            group=group_name,
            job_type="seq2seq-train",
            tags=tags
        )
        
        if parent_url:
            wandb.summary["parent_quantizer_url"] = parent_url
            logger.info(f"🔗 Linked to parent quantizer run: {parent_url}")
        elif parent_run_id:
            fallback_project = config.general.get('wandb_project_quantizer', 'mt-amazon-quantizer')
            parent_url = f"https://wandb.ai/{wandb.run.entity}/{fallback_project}/runs/{parent_run_id}"
            wandb.summary["parent_quantizer_url"] = parent_url
            logger.info(f"🔗 Linked to parent quantizer run (constructed): {parent_url}")

    logger.info(f"Recommender run name: {recommender_run_name}")

    # load sequential data
    logger.info("Loading User History Data...")
    sequences, num_users, num_items = load_amazon_sequences(category=config.data.category)
    logger.info(f"Loaded history for {num_users} users and {num_items} items")
    
    # create datasets and dataloaders
    train_dataset = SemanticIDSequenceDataset(
        history_data=sequences,
        semantic_ids=semantic_ids,
        mode='train'
    )
    
    eval_dataset = SemanticIDSequenceDataset(
        history_data=sequences,
        semantic_ids=semantic_ids,
        mode='eval'
    )
    
    num_workers = config.seq2seq.get('num_workers', 0)
    persistent_workers = (num_workers > 0)
    
    use_augmentation = config.seq2seq.get('data_augmentation', False)
    max_history_len = config.seq2seq.get('max_history_len', 20)
    
    if use_augmentation:
        train_collate_fn = partial(collate_fn_with_augmentation, max_len=max_history_len)
        logger.info("Using augmented collate function with max_len truncation")
    else:
        train_collate_fn = partial(collate_fn, max_len=max_history_len)
    
    g = get_seeded_generator(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.seq2seq.get('batch_size', 256),
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=is_cuda
    )
    
    # dataloader for loss evaluation
    eval_collate_fn = partial(collate_fn, max_len=max_history_len)
    
    eval_loader_loss = DataLoader(
        eval_dataset,
        batch_size=config.seq2seq.get('batch_size', 256),
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=is_cuda
    )
    
    # dataloader for metrics evaluation
    # uses smaller batch size to increase throughput and avoid OOM during beam search
    metric_batch_size = max(1, config.seq2seq.get('batch_size', 256) // 2)
    logger.info(f"Using separate eval loaders: Loss BS={config.seq2seq.get('batch_size', 256)}, Metrics BS={metric_batch_size}")
    
    eval_loader_metrics = DataLoader(
        eval_dataset,
        batch_size=metric_batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=is_cuda
    )
    
    # initialize model, defaulting to TIGER paper specs
    model = TigerSeq2Seq(
        codebook_layers=config.seq2seq.get('codebook_layers', 3),
        codebook_size=config.seq2seq.get('codebook_size', 256),
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
    
    # compile the model
    if config.general.get('compile', False):
        if device.type == 'mps':
            logger.warning("torch.compile is not stably supported on MPS (Apple Silicon). Skipping compilation and running in eager mode.")
        else:
            logger.info("Compiling Seq2Seq model using torch.compile(dynamic=True)...")
            try:
                # dynamic=True prevents recompilation when user history lengths vary
                model = torch.compile(model, dynamic=True)
            except Exception as e:
                logger.warning(f"Failed to compile model. Falling back to eager mode. Error: {e}")
    
    if config.general.get('use_wandb', False):
        wandb.watch(model, log="all")

    # define optimizer and scheduler
    optimizer_args = {
        "lr": config.seq2seq.get('learning_rate', 1e-3),
        "weight_decay": config.seq2seq.get('weight_decay', 0.0001)
    }
    if is_cuda:
        optimizer_args["fused"] = True
        logger.info("Using fused Adam optimizer")
        
    try:
        optimizer = optim.Adam(model.parameters(), **optimizer_args)
    except Exception as e:
        if "fused" in optimizer_args:
            logger.warning(f"Failed to initialize fused Adam optimizer. Falling back to non-fused version. Error: {e}")
            optimizer_args.pop("fused")
            optimizer = optim.Adam(model.parameters(), **optimizer_args)
        else:
            raise e
    
    start_epoch = 0
    global_step = 0
    best_recall_at_5 = 0.0
    
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    # RESUME CHECKPOINT
    if resume_path:
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint file not found at {resume_path}")
            
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if resume_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("Loaded scaler state dict from checkpoint")
        
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

    # fixed lr / inverse square root decay scheduler
    use_lr_scheduler = config.seq2seq.get('use_lr_scheduler', True)
    
    if use_lr_scheduler:
        if warmup_steps_override is not None:
             warmup_steps = warmup_steps_override
        else:
             warmup_steps = config.seq2seq.get('warmup_steps', 10000)
             
        def inverse_sqrt_lambda(step):
            step = max(1, step)
            if step < warmup_steps:
                return 1.0
            return (warmup_steps / step) ** 0.5
            
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=inverse_sqrt_lambda)
        
        # fast-forward scheduler if resuming
        if global_step > 0:
            for _ in range(global_step):
                scheduler.step()
    else:
        scheduler = None
    
    # training loop
    logger.info("Starting Training Loop...")
    num_epochs = config.seq2seq.get('num_epochs', 2300)
    
    early_stopping_patience = config.seq2seq.get('early_stopping_patience', 7) # cycles of metrics evaluation
    recall_no_improve = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            history_tuples = batch['history_tuples'].to(device, non_blocking=True)
            target_tuples = batch['target_tuples'].to(device, non_blocking=True)
            user_ids = batch['user_ids'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # forward pass
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=is_cuda):
                outputs = model(
                    history_tuples=history_tuples,
                    target_tuples=target_tuples,
                    user_ids=user_ids
                )
                loss = outputs['loss']
            
            # backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
        # compute retrieval metrics every eval_epoch_interval epochs and on the last epoch
        eval_epoch_interval = config.seq2seq.get('eval_epoch_interval', 5)
        eval_delay_epochs = config.seq2seq.get('eval_delay_epochs', 0)
        if not stop_training and (epoch + 1) > eval_delay_epochs and ((epoch + 1) % eval_epoch_interval == 0 or (epoch + 1) == num_epochs):
            logger.info("Computing retrieval metrics (Beam Search)...")
            avg_recall, avg_ndcg, avg_hierarchical = compute_metrics(model, eval_loader_metrics, device)
            
            logger.info(f"Recall: {avg_recall} | NDCG: {avg_ndcg}")
            
            current_recall_at_5 = avg_recall.get(5, 0.0)
            if current_recall_at_5 > best_recall_at_5:
                best_recall_at_5 = current_recall_at_5
                recall_no_improve = 0 # reset patience
                if config.general.get('save_model', True):
                    os.makedirs("models/recommender", exist_ok=True)
                    model_id = f"{recommender_run_name}_best"
                    save_path = f"models/recommender/{model_id}.pt"
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, current_recall_at_5, save_path, "recall@5")
                    logger.info(f"New best Recall@5 achieved ({current_recall_at_5:.4f})! Model saved to {save_path}")
                    log_model_artifact(
                        model_path=save_path,
                        run_name=model_id,
                        artifact_type="tiger-recommender",
                        metadata={"epoch": epoch, "recall@5": current_recall_at_5,
                                  "rqvae_semids": semantic_ids_path},
                        config=config,
                    )
            else:
                recall_no_improve += 1
                logger.info(f"No improvement in Recall@5 for {recall_no_improve} evaluations.")
                if early_stopping and early_stopping_patience is not None and recall_no_improve >= early_stopping_patience:
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
    
    # write metadata JSON if path provided
    if config.general.get('meta_json', None):
        meta_json_path = config.general.meta_json
        os.makedirs(os.path.dirname(meta_json_path), exist_ok=True)
        with open(meta_json_path, 'w') as f:
            json.dump({
                "recommender_run_name": recommender_run_name,
                "best_model_path": "models/recommender/{}_best.pt".format(recommender_run_name)
            }, f)

    if config.general.get('use_wandb', False):
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TIGER Seq2Seq Model")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--semids', type=str, required=True, help='Path to generated Semantic IDs (.pt file)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--warmup_steps', type=int, help='Override warmup_steps from config')
    args, overrides = parser.parse_known_args()
    
    run_training(args.config, args.semids, args.resume, args.warmup_steps, overrides)