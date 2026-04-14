"""
Shared pytest configuration and fixtures for the Semantic-ID-Generation test suite.

This file is automatically loaded by pytest before any tests run. It:
  - Adds the project root to sys.path so all modules are importable
  - Defines shared fixtures reusable across unit and integration tests
"""
import sys
import os

import pytest
import torch

# Make the project root importable from any test file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Embedding fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_embeddings():
    """100 L2-normalised random embeddings of dimension 768."""
    torch.manual_seed(0)
    x = torch.randn(100, 768)
    return torch.nn.functional.normalize(x, p=2, dim=1)


@pytest.fixture
def small_embeddings():
    """50 L2-normalised random embeddings of dimension 32 (fast tests)."""
    torch.manual_seed(42)
    x = torch.randn(50, 32)
    return torch.nn.functional.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Semantic ID fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_semids():
    """Random semantic IDs: shape [100, 3], codebook_size=16."""
    torch.manual_seed(0)
    return torch.randint(0, 16, (100, 3))


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    """Minimal OmegaConf config suitable for unit tests (no wandb, no saving)."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "data": {
            "dataset": "amazon",
            "category": "beauty",
            "normalize_data": True,
        },
        "model": {
            "input_dimension": 32,
            "num_codebook_layers": 3,
            "codebook_clusters": 16,
            "quantization_method": "ste",
            "distance_method": "cosine",
        },
        "train": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_epochs": 2,
        },
        "general": {
            "use_wandb": False,
            "save_model": False,
        },
    })
