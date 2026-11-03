"""
Integration tests for the test script (test_seq2seq.py).

Verifies the meta JSON that run_testing writes, with a randomly initialised tiny model on CPU:
  - global temporal split: per-user 'metrics' and 'hierarchical', the per-target and last-target
    metrics, the counts, and the cold test split under 'cold' (counts only if it is empty)
  - leave-one-out: row-averaged 'metrics' and 'hierarchical', no gts keys

The fixture seq2seq_config is defined in tests/integration/conftest.py.
"""
import json
from unittest.mock import patch

import pytest
import torch

from data.protocols import SplitConfig


def _eval_split(user_ids, histories, targets, is_last=None):
    split = {
        "userId": torch.tensor(user_ids),
        "itemId": histories,
        "itemId_fut": torch.tensor(targets),
    }
    if is_last is not None:
        split["is_last"] = torch.tensor(is_last)
    return split


class TestTestSeq2SeqMeta:
    """Smoke test of test_seq2seq.run_testing with a randomly initialised tiny model on CPU."""

    # keys of one evaluated gts split; the cold split is stored under 'cold' with the same keys
    GTS_SPLIT_KEYS = {"metrics", "hierarchical", "hierarchical_per_target", "metrics_per_target", "metrics_last",
                      "seen_at_k", "users_with_seen_at_k", "counts"}
    LOO_KEYS = {"metrics", "hierarchical", "seen_at_k", "users_with_seen_at_k"}

    def _setup(self, tmp_path, seq2seq_config, protocol):
        from omegaconf import OmegaConf
        from modules.recommender import TigerSeq2Seq

        config, config_path = seq2seq_config
        config["data"]["split_protocol"] = protocol
        config["seq2seq"].update({"codebook_layers": 2, "codebook_size": 10})
        config["general"].update({"device": "cpu", "meta_json": str(tmp_path / "meta" / "test_meta.json")})
        OmegaConf.save(OmegaConf.create(config), config_path)

        # 2 codebook layers + the collision column (unique tuples, so the collision index is 0)
        semids = torch.stack([torch.arange(20) % 5, torch.arange(20) // 5, torch.zeros(20, dtype=torch.long)], dim=1)
        semids_file = str(tmp_path / "semids.pt")
        stage1_config = OmegaConf.create({"data": {"dataset": config["data"]["dataset"], "category": config["data"]["category"],
                                                   "split_protocol": protocol}, "general": {}})
        torch.save({"semantic_ids": semids, "config": stage1_config}, semids_file)

        s = config["seq2seq"]
        model = TigerSeq2Seq(codebook_layers=2, codebook_size=10, user_tokens=s["user_tokens"], d_model=s["d_model"],
                             d_kv=s["d_kv"], d_ff=s["d_ff"], num_layers=1, num_heads=s["num_heads"],
                             dropout=0.0, activation_fn="relu")
        model.set_codebooks(semids)
        ckpt = str(tmp_path / "model.pt")
        torch.save({"model_state_dict": model.state_dict()}, ckpt)

        if protocol == "gts":
            sequences = {
                "test": _eval_split([0, 0, 1], [[1], [1, 2], [3]], [2, 4, 5], is_last=[False, True, True]),
                "test_cold": _eval_split([2, 2], [[7], [7, 8]], [8, 9], is_last=[False, True]),
            }
        else:
            sequences = {"test": _eval_split([0, 1, 2], [[1, 2], [3], [4, 5]], [6, 7, 8])}
        return config_path, semids_file, ckpt, sequences, config["general"]["meta_json"]

    def test_gts_meta_keys(self, tmp_path, seq2seq_config):
        from test_seq2seq import run_testing
        config_path, semids_file, ckpt, sequences, meta_path = self._setup(tmp_path, seq2seq_config, "gts")
        with patch("test_seq2seq.load_amazon_sequences", return_value=(sequences, 3, 20)) as mock_load:
            run_testing(config_path, semids_file, ckpt)
        assert mock_load.call_args.args[1] == SplitConfig("gts", 0.9, 0.9)
        meta = json.load(open(meta_path))
        assert set(meta) == self.GTS_SPLIT_KEYS | {"cold"}
        assert set(meta["cold"]) == self.GTS_SPLIT_KEYS
        assert meta["counts"] == {"num_users": 2, "num_rows": 3}
        assert meta["cold"]["counts"] == {"num_users": 1, "num_rows": 2}
        assert {"Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10", "NDCG@20"} == set(meta["metrics"])
        # gts: 'metrics' and 'hierarchical' are both per-user averages, so Recall@k == h@k_slice_<last layer>
        assert meta["metrics"]["Recall@5"] == pytest.approx(meta["hierarchical"]["h@5_slice_3"])
        assert meta["metrics"]["Recall@20"] == pytest.approx(meta["hierarchical"]["h@20_slice_3"])
        assert meta["metrics_per_target"]["Recall@5"] == pytest.approx(meta["hierarchical_per_target"]["h@5_slice_3"])

    def test_gts_empty_cold_split_has_counts_only(self, tmp_path, seq2seq_config):
        from test_seq2seq import run_testing
        config_path, semids_file, ckpt, sequences, meta_path = self._setup(tmp_path, seq2seq_config, "gts")
        sequences["test_cold"] = _eval_split([], [], [], is_last=[])
        with patch("test_seq2seq.load_amazon_sequences", return_value=(sequences, 3, 20)):
            run_testing(config_path, semids_file, ckpt)
        meta = json.load(open(meta_path))
        assert meta["cold"] == {"counts": {"num_users": 0, "num_rows": 0}}

    def test_loo_meta_keys(self, tmp_path, seq2seq_config):
        from test_seq2seq import run_testing
        config_path, semids_file, ckpt, sequences, meta_path = self._setup(tmp_path, seq2seq_config, "loo")
        with patch("test_seq2seq.load_amazon_sequences", return_value=(sequences, 3, 20)):
            run_testing(config_path, semids_file, ckpt)
        meta = json.load(open(meta_path))
        assert set(meta) == self.LOO_KEYS
        # leave-one-out keeps Recall@k == h@k_slice_<last layer> (row averages)
        assert meta["metrics"]["Recall@5"] == meta["hierarchical"]["h@5_slice_3"]
