"""
Training script for RQ-VAE with support for temperature annealing and multiple quantization methods.
"""

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import logging
import random
import numpy as np
from omegaconf import OmegaConf
from modules.temperature_scheduler import create_temperature_scheduler
from schemas.quantization import QuantizeForwardMode
from utils.wandb import wandb_init
from data.factory import load_data
from modules.rq_vae import RQ_VAE
from utils.model_id_generation import generate_model_id

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_model(config, input_dim):
    """
    Create RQ-VAE model with configuration parameters.

    Args:
        config: OmegaConf configuration object
        input_dim: Input dimension of the data

    Returns:
        RQ_VAE: Configured model instance
    """
    # Get quantization parameters with defaults
    quantization_method_str = getattr(config.model, 'quantization_method', 'ste')

    # Convert string to enum
    if quantization_method_str == "gumbel_softmax":
        quantization_method = QuantizeForwardMode.GUMBEL_SOFTMAX
    elif quantization_method_str == "ste":
        quantization_method = QuantizeForwardMode.STE
    else:
        raise ValueError(f"Unknown quantization method: {quantization_method_str}")

    model = RQ_VAE(
        input_dim=input_dim,
        latent_dim=config.model.latent_dimension,
        hidden_dims=config.model.hidden_dimensions,
        codebook_size=config.model.codebook_clusters,
        codebook_kmeans_init=True,
        codebook_sim_vq=True,
        n_quantization_layers=config.model.num_codebook_layers,
        commitment_weight=config.model.commitment_weight,
        quantization_method=quantization_method,
    )

    logger.info(f"Created RQ-VAE model with {quantization_method_str} quantization")
    return model

def train(model, data, optimizer, scheduler, num_epochs, device, config):
    """
    Train RQ-VAE model with support for temperature annealing.

    Args:
        model: RQ-VAE model instance
        data: Training data tensor
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (currently unused but kept for compatibility)
        num_epochs: Number of training epochs
        device: Training device (CPU/GPU)
        config: Configuration object

    Returns:
        list: Training statistics for each epoch
    """
    # Note: scheduler parameter is kept for compatibility but not currently used
    model.train()

    # Get temperature annealing settings
    temperature_annealing = getattr(config.train, 'temperature_annealing', False)
    temperature_update_freq = getattr(config.train, 'temperature_update_frequency', 1)
    quantization_method_str = getattr(config.model, 'quantization_method', 'ste')

    # Convert string to enum for comparison
    is_gumbel_softmax = quantization_method_str == "gumbel_softmax"

    # Initialize temperature scheduler for Gumbel Softmax
    temperature_scheduler = None
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
        logger.info(f"Temperature annealing enabled: {annealing_schedule} schedule (update every {temperature_update_freq} epochs)")

    logger.info(f"Training with {quantization_method_str} quantization")

    epoch_progress = tqdm(range(num_epochs), total=num_epochs, desc="Training Loop")
    results = []

    train_loader = DataLoader(data, batch_size=config.data.batch_size)

    for epoch in epoch_progress:
        total_loss = 0
        total_reconstruction_loss = 0
        total_commit_loss = 0
        p_unique = 0

        # Get current temperature
        current_temperature = 1.0  # Default for STE
        if temperature_scheduler is not None:
            current_temperature = temperature_scheduler.get_temperature()
            # Update temperature according to schedule
            if epoch % temperature_update_freq == 0:
                temperature_scheduler.step()

        # Initialize codebooks on first epoch
        if epoch == 0:
            kmeans_init_data = torch.Tensor(data[torch.arange(min(20000, len(data)))]).to(device, dtype=torch.float32)
            model(kmeans_init_data, temperature=current_temperature)

        for batch in train_loader:
            batch = batch.to(device).float()
            optimizer.zero_grad()
            result = model(batch, temperature=current_temperature)
            result.loss.backward()
            optimizer.step()

            total_loss += result.loss.item()
            total_reconstruction_loss += result.reconstruction_loss.item()
            total_commit_loss += result.rqvae_loss.item()
            p_unique += result.p_unique_ids.item()

        # Calculate epoch statistics
        epoch_stats = {
            "Epoch": epoch,
            "Loss": total_loss / len(train_loader),
            "Reconstruction Loss": total_reconstruction_loss / len(train_loader),
            "RQ-VAE Loss": total_commit_loss / len(train_loader),
            "Prob Unique IDs": p_unique / len(train_loader)
        }

        # Add temperature to stats if using Gumbel Softmax
        if is_gumbel_softmax and temperature_scheduler is not None:
            epoch_stats["Temperature"] = current_temperature

        # Early stopping condition
        if p_unique / len(train_loader) >= 1:
            logger.info(f"Early stopping at epoch {epoch}: All IDs are unique")
            break

        if config.general.use_wandb:
            wandb.log(epoch_stats, step=epoch)

        epoch_progress.set_postfix(epoch_stats)
        results.append(epoch_stats)

    return results

def run_training(config_path):
    """
    Orchestrate RQ-VAE training.
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    seed = getattr(config.train, 'seed', 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model_id = generate_model_id(config)

    logger.info(f"Using device: {device}")
    logger.info(f"Model ID: {model_id}")

    # Initialize wandb if enabled
    if config.general.use_wandb:
        wandb_init(config)

    # Load data and create model
    data = load_data(config)
    model = create_model(config, data.shape[1])
    model.to(device)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.train.learning_rate,
                           weight_decay=config.train.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Watch model with wandb if enabled
    if config.general.use_wandb:
        wandb.watch(model, log="all")

    # Train model
    logger.info("Starting training...")
    train_results = train(
        model=model,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.train.num_epochs,
        device=device,
        config=config
    )

    # Save model
    torch.save(model.state_dict(), f"models/{model_id}.pt")
    logger.info(f"Training completed. Final results: {train_results[-1]}")

    if config.general.use_wandb:
        wandb.finish()