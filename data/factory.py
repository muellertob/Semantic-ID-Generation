from data.loader import load_movie_lens, load_amazon
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(config):
    """
    Load dataset based on configuration.

    Args:
        config: OmegaConf configuration object

    Returns:
        torch.Tensor: Loaded dataset

    Raises:
        NotImplementedError: For unsupported datasets
        ValueError: For unknown dataset names
    """
    if config.data.dataset == "movielens":
        data = load_movie_lens(
            category=config.data.category,
            dimension=config.data.embedding_dimension,
            train=True,
            raw=True
        )
    elif config.data.dataset == "amazon":
        data = load_amazon(
            category=config.data.category,
            normalize_data=config.data.normalize_data,
            train=True
        )
    elif config.data.dataset == "lastfm":
        raise NotImplementedError("LastFM dataset loading is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    logger.info(f"Loaded {config.data.dataset} dataset with shape: {data.shape}")
    return data
