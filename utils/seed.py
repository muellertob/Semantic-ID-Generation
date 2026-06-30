import random
import numpy as np
import torch

def set_seed(seed: int) -> int:
    """
    Set seed for random, numpy, and torch (CPU, CUDA, and MPS) to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    return seed
