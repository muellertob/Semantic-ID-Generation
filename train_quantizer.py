"""
Training script for all quantizers (FSQ, ResidualFSQ, RQ-VAE).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import logging
import random
import numpy as np
import os
from omegaconf import OmegaConf

from schemas.quantization import QuantizeForwardMode, QuantizeDistance
from modules.rqvae.scheduler import create_temperature_scheduler
from utils.wandb import wandb_init, get_run_name, log_model_artifact
from data.factory import load_data
from modules.rqvae.model import RQ_VAE
from modules.fsq.model import FSQ_AutoEncoder, ResidualFSQ_AutoEncoder
from utils.model_id_generation import generate_model_id

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_model(config, input_dim):
    """
    Create a quantizer model based on the provided configuration.

    Returns:
        Configured quantizer model (RQ-VAE, FSQ, or ResidualFSQ).
    """
    quantizer_type = config.model.quantizer_type
    latent_dim = config.model.latent_dimension
    hidden_dims = config.model.hidden_dimensions
    codebook_layers = config.model.codebook_layers
    loss_type = getattr(config.model, 'loss_type', 'mse')
    normalize = getattr(config.data, 'normalize_data', True)

    if quantizer_type == "rqvae":
        quantization_method_str = getattr(config.model, 'quantization_method', 'ste')
        distance_method_str = getattr(config.model, 'distance_method', 'cosine')

        if quantization_method_str == "gumbel_softmax":
            quantization_method = QuantizeForwardMode.GUMBEL_SOFTMAX
        elif quantization_method_str == "ste":
            quantization_method = QuantizeForwardMode.STE
        else:
            raise ValueError(f"Unknown quantization method: {quantization_method_str}")

        if distance_method_str == "cosine":
            distance_mode = QuantizeDistance.COSINE
        elif distance_method_str == "l2":
            distance_mode = QuantizeDistance.L2
        else:
            raise ValueError(f"Unknown distance mode: {distance_method_str}")

        model = RQ_VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            codebook_size=config.model.codebook_size,
            codebook_layers=codebook_layers,
            commitment_weight=config.model.commitment_weight,
            quantization_method=quantization_method,
            distance_mode=distance_mode
        )
        logger.info(f"Created RQ-VAE model with {quantization_method_str} quantization, codebook_size={config.model.codebook_size}, layers={codebook_layers}")
    
    elif quantizer_type == "rfsq":
        projection_type = getattr(config.model, 'projection_type', 'mlp_1_hidden')
        inner_dim = getattr(config.model, 'inner_dim', 256)
        level_list = config.model.level_list
        model = ResidualFSQ_AutoEncoder(
            input_dim=input_dim,
            codebook_layers=codebook_layers,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            level_list=level_list,
            loss_type=loss_type,
            normalize=normalize,
            projection_type=projection_type,
            inner_dim=inner_dim
        )
        logger.info(f"Created ResidualFSQ AutoEncoder with layers={codebook_layers}, level_list={level_list}, projection={projection_type}")
   
    elif quantizer_type == "fsq":
        level_list = config.model.level_list
        projection_type = getattr(config.model, 'projection_type', None)
        inner_dim = getattr(config.model, 'inner_dim', None)
        model = FSQ_AutoEncoder(
            input_dim=input_dim,
            codebook_layers=codebook_layers,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            level_list=level_list,
            loss_type=loss_type,
            normalize=normalize,
            projection_type=projection_type,
            inner_dim=inner_dim
        )
        logger.info(f"Created FSQ AutoEncoder with layers={codebook_layers}, level_list={level_list}")
    
    else:
        raise ValueError(f"Unknown quantizer type: {quantizer_type}")
        
    return model

def train(model, data, optimizer, num_epochs, device, config):
    """
    Train the quantizer model for a specified number of epochs using the provided data and optimizer.
    
    Returns:
        A list of dictionaries containing training metrics for each epoch.
    """
    model.train()
    epoch_progress = tqdm(range(num_epochs), total=num_epochs, desc="Training Loop")
    results = []

    train_loader = DataLoader(data, batch_size=config.data.batch_size, shuffle=True)

    quantizer_type = config.model.quantizer_type
    is_rqvae = quantizer_type == "rqvae"

    # initialize temperature scheduler for Gumbel Softmax
    temperature_scheduler = None
    if is_rqvae:
        temperature_annealing = getattr(config.train, 'temperature_annealing', False)
        temperature_update_freq = getattr(config.train, 'temperature_update_frequency', 1)
        quantization_method_str = getattr(config.model, 'quantization_method', 'ste')
        is_gumbel_softmax = quantization_method_str == "gumbel_softmax"

        if temperature_annealing and is_gumbel_softmax:
            annealing_schedule = getattr(config.train, 'annealing_schedule', 'exponential')
            initial_temp = getattr(config.train, 'temperature', 2.0)
            min_temp = getattr(config.train, 'min_temperature', 0.1)
            decay_rate = getattr(config.train, 'temperature_decay', 0.999)

            temperature_scheduler = create_temperature_scheduler(
                schedule_type=annealing_schedule,
                initial_temperature=initial_temp,
                min_temperature=min_temp,
                decay_rate=decay_rate,
                total_steps=num_epochs // temperature_update_freq
            )
            logger.info(f"Temperature annealing enabled: {annealing_schedule} schedule")

    # sequential lr scheduler
    use_lr_scheduler = getattr(config.train, 'use_lr_scheduler', False)
    if use_lr_scheduler:
        warmup_steps = getattr(config.train, 'warmup_steps', 1000)
        total_steps = num_epochs * len(train_loader)
        
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        decay_steps = total_steps - warmup_steps
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, decay_steps), eta_min=1e-6
        )
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )
    else:
        scheduler = None

    for epoch in epoch_progress:
        total_metrics = {}

        # get current temperature for RQ-VAE
        current_temperature = 1.0 # default for STE
        if temperature_scheduler is not None:
            current_temperature = temperature_scheduler.get_temperature()
            if epoch % temperature_update_freq == 0:
                temperature_scheduler.step()

        # RQ-VAE KMeans init on epoch 0
        if is_rqvae and epoch == 0:
            kmeans_init_data = torch.Tensor(data[torch.arange(min(20000, len(data)))]).to(device, dtype=torch.float32)
            model.kmeans_init_codebooks(kmeans_init_data, temperature=current_temperature)

        for batch in train_loader:
            batch = batch.to(device).float()
            optimizer.zero_grad()
            
            if is_rqvae:
                result = model(batch, temperature=current_temperature)
            else:
                result = model(batch)
                
            result.loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            total_metrics["Loss"] = total_metrics.get("Loss", 0.0) + result.loss.item()
            total_metrics["Reconstruction Loss"] = total_metrics.get("Reconstruction Loss", 0.0) + result.reconstruction_loss.item()
            total_metrics["Quantization Loss"] = total_metrics.get("Quantization Loss", 0.0) + result.quantization_loss.item()
            
            for k, v in result.metrics.items():
                if v.numel() > 1:
                    if k not in total_metrics:
                        total_metrics[k] = v.cpu().clone()
                    else:
                        total_metrics[k] += v.cpu()
                else:
                    total_metrics[k] = total_metrics.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)

        # calculate epoch statistics
        n_batches = len(train_loader)
        epoch_stats = {"Epoch": epoch, "Learning Rate": optimizer.param_groups[0]['lr']}
        if is_rqvae and temperature_scheduler is not None:
            epoch_stats["Temperature"] = current_temperature

        wandb_stats = dict(epoch_stats)

        for k, v in total_metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() > 1:
                avg_array = v / n_batches
                epoch_stats[k] = avg_array.mean().item()
                if config.general.use_wandb:
                    for i in range(len(avg_array)):
                        if "coverage" in k:
                            wandb_stats[f"layer_{i}/coverage"] = avg_array[i].item()
                        elif "entrop" in k:
                            wandb_stats[f"layer_{i}/entropy"] = avg_array[i].item()
                        else:
                            wandb_stats[f"layer_{i}/{k}"] = avg_array[i].item()
            else:
                epoch_stats[k] = v / n_batches
                
                # Keep compatibility with old wandb chart names
                if k == "first_centroids_norm":
                    wandb_stats["First Centroids Norm"] = epoch_stats[k]
                elif k == "last_centroids_norm":
                    wandb_stats["Last Centroids Norm"] = epoch_stats[k]
                elif k == "first_residual_norm":
                    wandb_stats["First Residual Norm"] = epoch_stats[k]
                elif k == "last_residual_norm":
                    wandb_stats["Last Residual Norm"] = epoch_stats[k]
                elif k == "p_unique_ids":
                    wandb_stats["Prob Unique IDs"] = epoch_stats[k]
                else:
                    wandb_stats[k] = epoch_stats[k]

        if config.general.use_wandb:
            wandb.log(wandb_stats, step=epoch)

        epoch_progress.set_postfix(epoch_stats)
        results.append(epoch_stats)

    return results

def run_training(config_path, overrides=None):
    """
    Orchestrates the training process for the quantizer model based on the provided configuration.
    """
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    seed = getattr(config.train, 'seed', 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model_id = generate_model_id(config)
    quantizer_type = config.model.quantizer_type
    
    prefix = quantizer_type

    if config.general.use_wandb:
        wandb_init(config, project=config.general.wandb_project_quantizer)
        if wandb.run is not None:
            if not wandb.run.name.startswith(f"{prefix}-"):
                wandb.run.name = f"{prefix}-{wandb.run.name}"
        wandb.define_metric("Epoch")
        wandb.define_metric("*", step_metric="Epoch")
        # use WandB run name if provided, otherwise fallback to generated model_id
        model_id = get_run_name(fallback=model_id)
    
    logger.info(f"Model ID: {model_id}")

    # load data and create model
    data_split = getattr(config.data, 'split', 'all')
    data = load_data(config, split=data_split)
    model = create_model(config, data.shape[1])
    model.to(device)

    # setup optimizer
    optimizer_type = getattr(config.train, 'optimizer', 'adamw').lower()
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # watch model with wandb if enabled
    if config.general.use_wandb:
        wandb.watch(model, log="all")

    logger.info(f"Starting training for {quantizer_type} on {device}...")
    train_results = train(
        model=model,
        data=data,
        optimizer=optimizer,
        num_epochs=config.train.num_epochs,
        device=device,
        config=config
    )

    # save model
    os.makedirs(f"models/{quantizer_type}", exist_ok=True)
    model_path = f"models/{quantizer_type}/{model_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Training completed. Final results: {train_results[-1]}")
    logger.info(f"Model saved to: {model_path}")

    if config.general.use_wandb:
        log_model_artifact(
            model_path=model_path,
            run_name=model_id,
            artifact_type=f"{quantizer_type}-tokenizer",
            metadata={"final_loss": train_results[-1].get("Loss"), "config": OmegaConf.to_container(config, resolve=True)},
        )
        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Autoencoder Quantizer")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("overrides", nargs="*", help="Override config parameters")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_training(args.config, args.overrides)
