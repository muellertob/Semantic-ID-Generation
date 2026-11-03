"""
Shared fixtures for integration tests.
"""
import pytest
import torch
from unittest.mock import MagicMock
from omegaconf import OmegaConf

@pytest.fixture
def seq2seq_config(tmp_path):
    """Minimal seq2seq config dict and its path inside tmp_path."""
    config = {
        "data": {"dataset": "test", "category": "test_cat"},
        "model": {"quantizer_type": "rqvae", "codebook_layers": 2, "codebook_size": 10},
        "seq2seq": {
            "batch_size": 2, "max_history_len": 5, "user_tokens": 10,
            "d_model": 16, "d_kv": 8, "d_ff": 32, "num_layers": 1, "num_heads": 2,
            "dropout": 0.1, "activation_fn": "relu",
            "learning_rate": 0.01, "num_epochs": 2, "warmup_steps": 100,
            "num_workers": 0,
        },
        "general": {"use_wandb": False, "save_model": True},
    }
    path = str(tmp_path / "seq2seq_config.yaml")
    OmegaConf.save(OmegaConf.create(config), path)
    return config, path


@pytest.fixture
def semids_path(tmp_path, seq2seq_config):
    """Path to a saved semantic-IDs file with 20 items and 2 codebook layers, made for the dataset of seq2seq_config."""
    path = str(tmp_path / "semids.pt")
    data = seq2seq_config[0]["data"]
    stage1_config = OmegaConf.create({"data": {"dataset": data["dataset"], "category": data["category"]}, "general": {}})
    torch.save({"semantic_ids": torch.randint(0, 10, (20, 2)), "config": stage1_config}, path)
    return path


@pytest.fixture
def mock_seq2seq_model():
    """A MagicMock standing in for TigerSeq2Seq with shape-correct outputs."""
    mock = MagicMock()
    mock.return_value = {"loss": torch.tensor(1.0, requires_grad=True)}
    mock.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    mock.state_dict.return_value = {"w": torch.zeros(1)}
    mock.beam_search.return_value = torch.zeros(2, 5, 2)
    mock.item_offsets = torch.zeros(2)
    # save_checkpoint / resume unwrap torch.compile via getattr(model, '_orig_mod', model);
    # without this a MagicMock would return a fresh child mock here
    mock._orig_mod = mock
    return mock


@pytest.fixture
def mock_amazon_sequences():
    """Minimal sequences dict matching the format returned by load_amazon_sequences."""
    return {
        "train": {
            "userId": torch.arange(10),
            "itemId": [torch.tensor([1, 2, 3]) for _ in range(10)],
            "itemId_fut": torch.tensor([4] * 10),
        },
        "eval": {
            "userId": torch.arange(2),
            "itemId": [torch.tensor([1, 2]) for _ in range(2)],
            "itemId_fut": torch.tensor([3] * 2),
        },
    }
