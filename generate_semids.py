import torch
import argparse
from omegaconf import OmegaConf
from data.factory import load_data
from modules.rq_vae import RQ_VAE
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path, config, device, input_dim):
    """Load a trained RQ-VAE model."""
    model = RQ_VAE(
        input_dim=input_dim,
        latent_dim=config.model.latent_dimension,
        hidden_dims=config.model.hidden_dimensions,
        codebook_size=config.model.codebook_clusters,
        n_quantization_layers=config.model.num_codebook_layers,
        commitment_weight=config.model.commitment_weight,
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Mark all quantization layers as initialized to prevent k-means reinit
    for layer in model.quantization_layers:
        layer.kmeans_initted = True
    
    return model

def generate_all_semids(model, data, device, batch_size=64, temperature=1.0):
    """Generate semantic IDs for all items in the dataset."""
    model.eval()
    all_semids = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device, dtype=torch.float32)
            
            # Get semantic IDs for the batch
            output = model.get_semantic_ids(batch, temperature=temperature)
            semids = output.sem_ids  # Shape: (batch_size, num_layers)
            
            all_semids.append(semids.cpu())
    
    return torch.cat(all_semids, dim=0)

def resolve_collisions(semids, max_collisions):
    """
    Append a collision token to the semantic IDs to make them unique.
    Args:
        semids (torch.Tensor): Tensor of shape (num_items, num_layers)
        max_collisions (int): Maximum allowed collisions (vocab size of collision layer)
    Returns:
        torch.Tensor: Tensor of shape (num_items, num_layers + 1)
    """
    logger.info("Resolving collisions...")
    device = semids.device
    num_items = semids.shape[0]

    # assigns a unique index to each semid
    _, inverse_indices = torch.unique(semids, sorted=True, return_inverse=True, dim=0)
    
    # calculates the permuatation needed to group identical semids together
    perm = torch.argsort(inverse_indices)
    inverse_sorted = inverse_indices[perm]

    # counts how many items are in each group of identical semids
    _, counts = torch.unique_consecutive(inverse_sorted, return_counts=True)
    
    # Generate cumulative counts (0, 1, 2...) for each group
    group_starts = torch.cat((torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)[:-1]))
    expanded_starts = group_starts.repeat_interleave(counts)
    sorted_cumcounts = torch.arange(num_items, device=device) - expanded_starts
    
    # restore the original order
    collision_tokens = torch.zeros_like(inverse_indices)
    collision_tokens[perm] = sorted_cumcounts
    
    # check for overflow
    max_depth = collision_tokens.max().item()
    if max_depth >= max_collisions:
        raise ValueError(
            f"Collision overflow! Max depth {max_depth} exceeds limit {max_collisions}."
        )

    final_ids = torch.cat([semids, collision_tokens.unsqueeze(1)], dim=1)
    
    num_collisions = (collision_tokens > 0).sum().item()
    logger.info(f"Found {num_collisions} items with collisions.")
    logger.info(f"Max collision depth: {max_depth} (Limit: {max_collisions})")
    
    return final_ids

def run_generation(config_path, model_path, output_path, temperature=0.5, batch_size=64):
    """
    Orchestrate semantic ID generation.
    """
    # load configuration
    config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # load all items data
    data = load_data(config, split='all')
    
    logger.info(f"Loaded {len(data)} all items with dimension {data.shape[1]}")
    
    # load trained model
    logger.info(f"Loading model from: {model_path}")
    model = load_trained_model(model_path, config, device, data.shape[1])
    
    # generate semantic IDs
    logger.info("Generating semantic IDs...")
    semids = generate_all_semids(
        model, data, device, 
        batch_size=batch_size,
        temperature=temperature
    )
    
    logger.info(f"Generated raw semantic IDs shape: {semids.shape}")
    
    # RESOLVE COLLISIONS (add 4th token)
    # pass codebook_clusters as the limit for the collision token
    codebook_size = config.model.codebook_clusters
    final_semids = resolve_collisions(semids, max_collisions=codebook_size)
    
    # save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'semantic_ids': final_semids,
        'config': config,
        'num_items': len(data),
        'temperature': temperature
    }, output_path)
    
    logger.info(f"Semantic IDs saved to: {output_path}")
    
    # print some statistics
    logger.info(f"Semantic ID statistics:")
    logger.info(f"  Shape: {final_semids.shape}")
    logger.info(f"  Min ID per layer: {final_semids.min(dim=0)[0]}")
    logger.info(f"  Max ID per layer: {final_semids.max(dim=0)[0]}")
    logger.info(f"  Unique IDs per layer: {[len(torch.unique(final_semids[:, i])) for i in range(final_semids.shape[1])]}")

def main():
    parser = argparse.ArgumentParser(description="Generate semantic IDs using a trained RQ-VAE model")
    parser.add_argument('--config', type=str, default='config/config_amazon_v2_gumbel.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--output_path', type=str, default='outputs/semids.pt',
                       help='Path to save the semantic IDs')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for Gumbel Softmax (lower = sharper)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    run_generation(
        config_path=args.config,
        model_path=args.model_path,
        output_path=args.output_path,
        temperature=args.temperature,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()

