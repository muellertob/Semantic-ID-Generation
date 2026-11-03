from data.loader import load_movie_lens, load_amazon
from data.protocols import get_split_config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(config, split='train'):
    """
    Load dataset based on configuration.

    Args:
        config: OmegaConf configuration object
        split (str): Dataset split to load ('train', 'test', 'all').

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
            train=True, # MovieLens loader ignores this parameter currently
            raw=True
        )
    elif config.data.dataset == "amazon":
        if "normalize_data" in config.data:
            raise ValueError("data.normalize_data was replaced by data.input_normalization "
                             "(True -> 'center_l2', False -> 'none')")
        data = load_amazon(
            category=config.data.category,
            split_config=get_split_config(config),
            input_normalization=config.data.input_normalization,
            split=split,
        )
    elif config.data.dataset == "lastfm":
        raise NotImplementedError("LastFM dataset loading is not implemented yet.")
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

    logger.info(f"Loaded {config.data.dataset} dataset with shape: {data.shape}")
    return data
