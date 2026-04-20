"""
Shared fixtures for integration tests.
"""
import json
import pytest
import torch
from unittest.mock import MagicMock
from omegaconf import OmegaConf

@pytest.fixture
def tiny_amazon_dataset(tmp_path):
    """
    A minimal AmazonReviews instance backed by synthetic files.

    Dataset layout:
      - 20 items (1-based IDs 1..20)
      - 10 users, each with 8 distinct items  → all pass the 5-core filter (len >= 5)
      - 1 user with exactly 5 items (uid=11)  → kept (boundary: == 5)
      - 1 user with exactly 4 items (uid=12)  → filtered (boundary: < 5)
      - 1 user with 3 items (uid=99)          → filtered

    PyG's InMemoryDataset.__init__ (download / process / load) is bypassed via a
    local subclass that overrides __init__ and the raw_dir property. No real
    dataset or processed .pt file is needed.
    """
    from data.amazon_data import AmazonReviews

    split = "beauty"
    split_dir = tmp_path / "raw" / split
    split_dir.mkdir(parents=True)

    # datamaps.json — 20 items
    item2id = {f"asin{i}": i for i in range(1, 21)}
    with open(split_dir / "datamaps.json", "w") as f:
        json.dump({"item2id": item2id}, f)

    # sequential_data.txt — "user_id item1 item2 ..." (1-based item IDs)
    lines = []
    for uid in range(1, 11):
        items = [(uid * 7 + i) % 20 + 1 for i in range(8)]
        lines.append(f"{uid} " + " ".join(map(str, items)))
    lines.append("11 18 19 20 1 2")
    lines.append("12 5 6 7 8")
    lines.append("99 1 2 3")
    with open(split_dir / "sequential_data.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    # PyG defines raw_dir as a @property on the class — instance assignment is silently ignored.
    # subclass override redirects it to our tmp_path without triggering download/process.
    class _TinyDataset(AmazonReviews):
        def __init__(self, raw_dir_path, split):
            self._raw_dir_path = raw_dir_path
            self.split = split

        @property
        def raw_dir(self):
            return self._raw_dir_path

    return _TinyDataset(str(tmp_path / "raw"), split)

@pytest.fixture
def seq2seq_config(tmp_path):
    """Minimal seq2seq config dict and its path inside tmp_path."""
    config = {
        "data": {"dataset": "test", "category": "test_cat"},
        "model": {"num_codebook_layers": 2, "codebook_clusters": 10},
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
def semids_path(tmp_path):
    """Path to a saved semantic-IDs file with 20 items and 2 codebook layers."""
    path = str(tmp_path / "semids.pt")
    torch.save({"semantic_ids": torch.randint(0, 10, (20, 2))}, path)
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
