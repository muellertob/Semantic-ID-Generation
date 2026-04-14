"""
Tests for utils/metrics.py

Covers:
  - MetricAccumulator: partial match, full match, offset predictions, reset,
    ndcg computation, multi-batch accumulation
"""
import math
import pytest
import torch
from utils.metrics import MetricAccumulator


class TestMetricAccumulator:

    def test_partial_match_hierarchical_metrics(self):
        """Prediction matching first 2 of 4 layers → slice_1 and slice_2 hit, rest miss."""
        acc = MetricAccumulator(k_list=[5], num_layers=4)
        target = torch.tensor([[[10, 20, 30, 40]]])
        pred = torch.tensor([[[10, 20, 99, 99], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
        acc.update(pred, target)
        res = acc.compute()
        h = res["hierarchical"]
        assert h["h@5_slice_1"] == 1.0
        assert h["h@5_slice_2"] == 1.0
        assert h["h@5_slice_3"] == 0.0
        assert h["h@5_slice_4"] == 0.0
        assert res["recall"][5] == 0.0

    def test_full_match_all_metrics_hit(self):
        """Exact prediction → all hierarchical slices and recall are 1.0."""
        acc = MetricAccumulator(k_list=[5], num_layers=4)
        target = torch.tensor([[[10, 20, 30, 40]]])
        pred = torch.tensor([[[10, 20, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
        acc.update(pred, target)
        res = acc.compute()
        h = res["hierarchical"]
        assert h["h@5_slice_1"] == 1.0
        assert h["h@5_slice_2"] == 1.0
        assert h["h@5_slice_3"] == 1.0
        assert h["h@5_slice_4"] == 1.0
        assert res["recall"][5] == 1.0

    def test_offset_prediction_first_layer_only_matches(self):
        """Offset predictions share only the first token with target → only slice_1 hits."""
        acc = MetricAccumulator(k_list=[5], num_layers=4)
        target = torch.tensor([[[10, 20, 30, 40]]])
        pred = torch.tensor([[[10, 276, 542, 808], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
        acc.update(pred, target)
        res = acc.compute()
        h = res["hierarchical"]
        assert h["h@5_slice_1"] == 1.0
        assert h["h@5_slice_2"] == 0.0
        assert h["h@5_slice_3"] == 0.0
        assert res["recall"][5] == 0.0

    def test_reset_clears_all_state(self):
        """After reset(), accumulator behaves as if freshly initialised."""
        acc = MetricAccumulator(k_list=[5], num_layers=4)
        target = torch.tensor([[[10, 20, 30, 40]]])
        pred = torch.tensor([[[10, 20, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
        acc.update(pred, target)
        acc.reset()
        res = acc.compute()
        assert res["recall"][5] == 0.0
        assert res["total_samples"] == 0

    def test_compute_returns_expected_keys(self):
        """compute() result always contains recall, ndcg, hierarchical, total_samples."""
        acc = MetricAccumulator(k_list=[1, 5], num_layers=3)
        target = torch.tensor([[[1, 2, 3]]])
        pred = torch.zeros(1, 5, 3, dtype=torch.long)
        acc.update(pred, target)
        res = acc.compute()
        assert "recall" in res
        assert "ndcg" in res
        assert "hierarchical" in res
        assert "total_samples" in res

    def test_ndcg_rank1_hit(self):
        """Exact match at rank 1 → ndcg@k == 1.0 for all k."""
        acc = MetricAccumulator(k_list=[1, 5], num_layers=2)
        target = torch.tensor([[[3, 7]]])
        pred = torch.tensor([[[3, 7], [0, 0], [0, 0], [0, 0], [0, 0]]])
        acc.update(pred, target)
        res = acc.compute()
        expected = 1.0 / math.log2(2)
        assert res["ndcg"][1] == pytest.approx(expected)
        assert res["ndcg"][5] == pytest.approx(expected)

    def test_ndcg_rank2_hit(self):
        """Match at rank 2 → ndcg equals 1/log2(3)."""
        acc = MetricAccumulator(k_list=[5], num_layers=2)
        target = torch.tensor([[[3, 7]]])
        pred = torch.tensor([[[0, 0], [3, 7], [0, 0], [0, 0], [0, 0]]])
        acc.update(pred, target)
        res = acc.compute()
        assert res["ndcg"][5] == pytest.approx(1.0 / math.log2(3))

    def test_no_match_recall_and_ndcg_zero(self):
        """No prediction matches target → recall and ndcg are 0.0."""
        acc = MetricAccumulator(k_list=[1, 5], num_layers=2)
        target = torch.tensor([[[9, 9]]])
        pred = torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]])
        acc.update(pred, target)
        res = acc.compute()
        assert res["recall"][1] == 0.0
        assert res["recall"][5] == 0.0
        assert res["ndcg"][1] == 0.0
        assert res["ndcg"][5] == 0.0

    def test_multi_batch_accumulation(self):
        """Two update() calls accumulate correctly: 1 hit out of 2 → recall = 0.5."""
        acc = MetricAccumulator(k_list=[1], num_layers=2)

        acc.update(torch.tensor([[[1, 2]]]), torch.tensor([[[1, 2]]]))  # hit
        acc.update(torch.tensor([[[9, 9]]]), torch.tensor([[[3, 4]]]))  # miss

        res = acc.compute()
        assert res["total_samples"] == 2
        assert res["recall"][1] == pytest.approx(0.5)

    def test_2d_targets_accepted(self):
        """update() accepts targets shaped [B, Codebook_Layers] (no squeeze needed)."""
        acc = MetricAccumulator(k_list=[1], num_layers=2)
        target = torch.tensor([[5, 6]])  # 2D, not 3D
        pred = torch.tensor([[[5, 6]]])
        acc.update(pred, target)
        res = acc.compute()
        assert res["recall"][1] == 1.0
