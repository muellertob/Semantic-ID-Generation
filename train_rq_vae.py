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
from modules.rqvae.scheduler import create_temperature_scheduler
from schemas.quantization import QuantizeForwardMode, QuantizeDistance
from utils.wandb import wandb_init, get_run_name, log_model_artifact
from data.factory import load_data
from modules.rqvae import RQ_VAE
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
    distance_method_str = getattr(config.model, 'distance_method', 'cosine')

    # Convert strings to enums
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
        latent_dim=config.model.latent_dimension,
        hidden_dims=config.model.hidden_dimensions,
        codebook_size=config.model.codebook_clusters,
        codebook_kmeans_init=True,
        codebook_sim_vq=False,
        n_quantization_layers=config.model.num_codebook_layers,
        commitment_weight=config.model.commitment_weight,
        quantization_method=quantization_method,
        distance_mode=distance_mode,
    )

    logger.info(f"Created RQ-VAE model with {quantization_method_str} quantization and {distance_method_str} distance")
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
        total_layer_coverages = None
        total_layer_entropies = None
        total_first_residual_norm = 0
        total_last_residual_norm = 0
        total_first_centroids_norm = 0
        total_last_centroids_norm = 0

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
            model.kmeans_init_codebooks(kmeans_init_data, temperature=current_temperature)

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

            coverages = result.layer_coverages.cpu()
            entropies = result.layer_entropies.cpu()
            if total_layer_coverages is None:
                total_layer_coverages = coverages.clone()
                total_layer_entropies = entropies.clone()
            else:
                total_layer_coverages += coverages
                total_layer_entropies += entropies
            total_first_residual_norm += result.first_residual_norm.item()
            total_last_residual_norm += result.last_residual_norm.item()
            total_first_centroids_norm += result.first_centroids_norm.item()
            total_last_centroids_norm += result.last_centroids_norm.item()

        # Calculate epoch statistics
        n_batches = len(train_loader)
        avg_layer_coverages = total_layer_coverages / n_batches
        avg_layer_entropies = total_layer_entropies / n_batches

        epoch_stats = {
            "Epoch": epoch,
            "Loss": total_loss / n_batches,
            "Reconstruction Loss": total_reconstruction_loss / n_batches,
            "RQ-VAE Loss": total_commit_loss / n_batches,
            "Prob Unique IDs": p_unique / n_batches,
            "Avg Coverage": avg_layer_coverages.mean().item(),
            "Avg Entropy": avg_layer_entropies.mean().item(),
            "First Residual Norm": total_first_residual_norm / n_batches,
            "Last Residual Norm": total_last_residual_norm / n_batches,
        }

        # Add temperature to stats if using Gumbel Softmax
        if is_gumbel_softmax and temperature_scheduler is not None:
            epoch_stats["Temperature"] = current_temperature

        # Early stopping condition
        if p_unique / n_batches >= 1:
            logger.info(f"Early stopping at epoch {epoch}: All IDs are unique")
            break

        if config.general.use_wandb:
            wandb_stats = dict(epoch_stats)
            for i in range(len(avg_layer_coverages)):
                wandb_stats[f"layer_{i}/coverage"] = avg_layer_coverages[i].item()
                wandb_stats[f"layer_{i}/entropy"] = avg_layer_entropies[i].item()
            wandb_stats["First Centroids Norm"] = total_first_centroids_norm / n_batches
            wandb_stats["Last Centroids Norm"] = total_last_centroids_norm / n_batches
            wandb.log(wandb_stats, step=epoch)

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

    # Initialize wandb if enabled
    if config.general.use_wandb:
        wandb_init(config, project=config.general.wandb_project_rqvae)

    # Use WandB run name as model ID (human-readable, traceable); fall back to
    # the hyperparam-encoded ID when WandB is disabled.
    model_id = get_run_name(fallback=generate_model_id(config))
    logger.info(f"Model ID: {model_id}")

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
    model_path = f"models/{model_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Training completed. Final results: {train_results[-1]}")
    logger.info(f"Model saved to: {model_path}")

    if config.general.use_wandb:
        log_model_artifact(
            model_path=model_path,
            run_name=model_id,
            artifact_type="rqvae-tokenizer",
            metadata={"final_loss": train_results[-1].get("Loss"),
                      "config": OmegaConf.to_container(config, resolve=True)},
        )
        wandb.finish()