import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# initialized to None so that we can detect if set_seed() has not been called.
_current_seed = None

def set_seed(seed: int) -> int:
    """
    Set seed for random, numpy, and torch (CPU, CUDA, and MPS) to ensure reproducibility.
    Also configures cuDNN determinism for CUDA execution.
    """
    global _current_seed
    _current_seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    return seed

def get_seed() -> int:
    """
    Get the currently set global seed.
    If set_seed() has not been called yet, logs a warning, initializes the global seed state
    using set_seed(42), and returns 42.
    """
    global _current_seed
    if _current_seed is None:
        logger.warning("get_seed() called before set_seed() was initialized. Initializing with default seed 42.")
        set_seed(42)
    return _current_seed

def get_seeded_generator(seed: int | None = None) -> torch.Generator:
    """
    Create and return a PyTorch Generator initialized with the specified seed.
    Following the official PyTorch documentation recommendation to ensure reproducibility.
    (https://docs.pytorch.org/docs/2.12/notes/randomness.html#dataloader)
    """
    g_seed = seed if seed is not None else get_seed()
    g = torch.Generator()
    g.manual_seed(g_seed)
    return g

def seed_worker(worker_id: int):
    """
    Seed function for PyTorch DataLoader workers (subprocesses).
    Following the official PyTorch documentation recommendation to ensure reproducibility when using multiple workers.
    (https://docs.pytorch.org/docs/2.12/notes/randomness.html#dataloader)
       
    Args:
        worker_id (int): Unused argument required by PyTorch's worker_init_fn signature.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
