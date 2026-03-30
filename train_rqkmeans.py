"""
Orchestration for RQ-KMeans semantic ID generation.

Pipeline order (strict):
  1. Load item embeddings
  2. Fit RQKMeans and generate raw semantic IDs
  3. Evaluate on raw SIDs (BEFORE collision resolution)
  4. Resolve collisions
  5. Save output

Usage:
    python train_rqkmeans.py --config config/config_amazon_rqkmeans.yaml
"""
import logging
import os

import torch
from omegaconf import OmegaConf

from data.factory import load_data
from generate_semids import resolve_collisions
from modules.rqkmeans import RQKMeans
from utils.sid_evaluation import evaluate_semids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training(config_path: str | None, _config_override=None) -> None:
    """
    Run the full RQ-KMeans training and ID generation pipeline.

    Args:
        config_path: Path to YAML config file. May be None when _config_override
                     is provided (used in tests).
        _config_override: OmegaConf DictConfig for testing without filesystem.
    """
    if _config_override is not None:
        config = _config_override
    else:
        config = OmegaConf.load(config_path)

    embeddings = load_data(config, split="all")
    logger.info(f"Loaded {len(embeddings)} item embeddings, dim={embeddings.shape[1]}")

    # fit model and generate raw semantic IDs
    model = RQKMeans(
        n_layers=config.model.n_layers,
        n_clusters=config.model.codebook_clusters,
        n_iters=config.model.n_iters,
        normalize_residuals=config.model.normalize_residuals,
        seed=config.general.seed,
    )
    sem_ids_raw = model.fit_and_generate(embeddings)
    logger.info(f"Raw semantic IDs shape: {sem_ids_raw.shape}")

    # evaluate on raw SIDs before collision resolution
    plot_dir = getattr(config.general, "plot_dir", None)
    evaluate_semids(
        embeddings=embeddings,
        raw_semids=sem_ids_raw,
        config=config,
        plot_dir=plot_dir,
    )

    # collision resolution
    codebook_size = config.model.codebook_clusters
    sem_ids = resolve_collisions(sem_ids_raw, max_collisions=codebook_size)

    # save output
    output_path = config.general.output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({
        "semantic_ids": sem_ids,
        "config": config,
        "num_items": len(embeddings),
    }, output_path)
    logger.info(f"Semantic IDs saved to: {output_path}")

    # optionally save the fitted model
    if config.general.save_model:
        model_path = config.general.model_path
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        model.save(model_path)
        logger.info(f"RQKMeans model saved to: {model_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train RQ-KMeans and generate semantic IDs")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
