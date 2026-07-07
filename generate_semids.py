import torch
import argparse
from omegaconf import OmegaConf
from data.factory import load_data
from utils.sid_evaluation import evaluate_semids
from utils.seed import set_seed
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path, config, device, input_dim):
    """Load a trained quantizer model (FSQ or RQ-VAE)."""
    quantizer_type = getattr(config.model, 'quantizer_type', 'rqvae')
    
    if "normalize_inputs" in config.model:
        normalize = config.model.normalize_inputs
    else:
        normalize = True
    
    if quantizer_type == 'rqvae':
        from modules.rqvae.model import RQ_VAE
        model = RQ_VAE(
            input_dim=input_dim,
            latent_dim=config.model.latent_dimension,
            hidden_dims=config.model.hidden_dimensions,
            codebook_size=config.model.codebook_size,
            codebook_layers=config.model.codebook_layers,
            commitment_weight=config.model.commitment_weight,
            normalize=normalize
        )
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        
        # Mark all quantization layers as initialized to prevent k-means reinit
        for layer in model.quantizer.quantization_layers:
            layer.kmeans_initted = True
    elif quantizer_type == 'rfsq':
        from modules.fsq.model import ResidualFSQ_AutoEncoder
        projection_type = getattr(config.model, 'projection_type', 'mlp_1_hidden')
        inner_dim = getattr(config.model, 'inner_dim', 256)
        model = ResidualFSQ_AutoEncoder(
            input_dim=input_dim,
            codebook_layers=config.model.codebook_layers,
            hidden_dims=config.model.hidden_dimensions,
            latent_dim=config.model.latent_dimension,
            level_list=config.model.level_list,
            loss_type=getattr(config.model, 'loss_type', 'mse'),
            normalize=normalize,
            projection_type=projection_type,
            inner_dim=inner_dim
        )
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
    elif quantizer_type == 'fsq':
        from modules.fsq.model import FSQ_AutoEncoder
        projection_type = getattr(config.model, 'projection_type', None)
        inner_dim = getattr(config.model, 'inner_dim', None)
        model = FSQ_AutoEncoder(
            input_dim=input_dim,
            codebook_layers=config.model.codebook_layers,
            hidden_dims=config.model.hidden_dimensions,
            latent_dim=config.model.latent_dimension,
            level_list=config.model.level_list,
            loss_type=getattr(config.model, 'loss_type', 'mse'),
            normalize=normalize,
            projection_type=projection_type,
            inner_dim=inner_dim
        )
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unknown quantizer_type: {quantizer_type}")
        
    return model

def generate_all_semids(model, data, device, batch_size=64):
    """Generate semantic IDs for all items in the dataset."""
    model.eval()
    all_semids = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device, dtype=torch.float32)
            semids = model.get_semantic_ids(batch)
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

def run_generation(config_path, model_path, output_path, batch_size=64, run_eval=True, overrides=None):
    """
    Orchestrate semantic ID generation.
    """
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
        
    seed = config.general.get('seed', 42)
    set_seed(seed)
        
    logger.info(f"Resolved Configuration:\n{OmegaConf.to_yaml(config)}")
    
    generation_config = config.get('generation', {})
    if generation_config:
        batch_size = generation_config.get('batch_size', batch_size)
        
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
        batch_size=batch_size
    )
    logger.info(f"Generated raw semantic IDs shape: {semids.shape}")

    # EVALUATE SEMANTIC IDS
    if run_eval:
        evaluate_semids(semids.cpu(), config)

    # RESOLVE COLLISIONS
    # pass codebook_clusters as the limit for the collision token
    quantizer_type = getattr(config.model, 'quantizer_type', 'rqvae')
    if quantizer_type in ['fsq', 'rfsq']:
        codebook_size = 1
        for lvl in config.model.level_list:
            codebook_size *= lvl
    else:
        codebook_size = config.model.codebook_size

    final_semids = resolve_collisions(semids, max_collisions=codebook_size)
    
    # save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'semantic_ids': final_semids,
        'config': config,
        'num_items': len(data)
    }, output_path)
    logger.info(f"Semantic IDs saved to: {output_path}")
    
    logger.info(f"Semantic ID statistics:")
    logger.info(f"  Shape: {final_semids.shape}")
    logger.info(f"  Min ID per layer: {final_semids.min(dim=0)[0]}")
    logger.info(f"  Max ID per layer: {final_semids.max(dim=0)[0]}")
    logger.info(f"  Unique IDs per layer: {[len(torch.unique(final_semids[:, i])) for i in range(final_semids.shape[1])]}")

def main():
    parser = argparse.ArgumentParser(description="Generate semantic IDs using a trained quantizer model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--output_path', type=str, default='outputs/semids.pt',
                       help='Path to save the semantic IDs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing')
    parser.add_argument('--no_eval', action='store_true',
                       help='Skip evaluation metrics and plots')

    args, overrides = parser.parse_known_args()

    run_generation(
        config_path=args.config,
        model_path=args.model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        run_eval=not args.no_eval,
        overrides=overrides,
    )

if __name__ == "__main__":
    main()
