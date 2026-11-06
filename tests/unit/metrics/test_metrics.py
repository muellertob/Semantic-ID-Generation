"""
Tests for utils/metrics.py

Covers:
  - MetricAccumulator: partial match, full match, offset predictions, reset,
    ndcg computation, multi-batch accumulation
  - MetricAccumulator with user ids / seen tuples / is_last: per-user and last-target
    averages, duplicate predictions, exact equality of recall_user and hierarchical_user,
    users_with_seen_at_k, optional inputs without user ids, consistency of the optional
    inputs across batches, too few prediction columns
"""
import math
import pytest
import torch
from utils.metrics import MetricAccumulator, calculate_entropy_and_coverage, calculate_quantizer_metrics


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


def _extended_batches():
    """
    Two users over two batches (L=2, beam 3). User 7 has rows 1-3, user 9 has row 4.

    row  user  target  predictions (top-3)        seen tuples            is_last
    1    7     (1,1)   (1,1) (2,2) (3,3)          (1,1) (9,9)            no
    2    7     (4,4)   (5,5) (6,6) (7,7)          (7,7)                  no
    3    7     (2,2)   (0,0) (1,1) (2,2)          (1,1) (0,0)            yes
    4    9     (8,8)   (8,8) (1,1) (2,2)          (5,5)                  yes
    """
    pad = [-1, -1]
    batch1 = dict(
        predictions=torch.tensor([[[1, 1], [2, 2], [3, 3]], [[5, 5], [6, 6], [7, 7]]]),
        targets=torch.tensor([[[1, 1]], [[4, 4]]]),
        user_ids=torch.tensor([7, 7]),
        seen_tuples=torch.tensor([[[1, 1], [9, 9]], [[7, 7], pad]]),
        is_last=torch.tensor([False, False]),
    )
    batch2 = dict(
        predictions=torch.tensor([[[0, 0], [1, 1], [2, 2]], [[8, 8], [1, 1], [2, 2]]]),
        targets=torch.tensor([[[2, 2]], [[8, 8]]]),
        user_ids=torch.tensor([7, 9]),
        seen_tuples=torch.tensor([[[1, 1], [0, 0], pad], [[5, 5], pad, pad]]),
        is_last=torch.tensor([True, True]),
    )
    return [batch1, batch2]


class TestMetricAccumulatorUserAndSeen:

    def _compute(self):
        acc = MetricAccumulator(k_list=[1, 3], num_layers=2)
        for b in _extended_batches():
            acc.update(**b)
        return acc.compute()

    def test_row_metrics_unchanged(self):
        """Row-averaged recall / NDCG: recall@1 = [1,0,0,1] -> 0.5, recall@3 = [1,0,1,1] -> 0.75."""
        res = self._compute()
        assert res["total_samples"] == 4
        assert res["recall"][1] == pytest.approx(0.5)
        assert res["recall"][3] == pytest.approx(0.75)
        assert res["ndcg"][3] == pytest.approx((1.0 + 0.0 + 0.5 + 1.0) / 4)

    def test_user_metrics_average_rows_per_user_first(self):
        """User 7: recall@1 = 1/3, recall@3 = 2/3, ndcg@3 = (1+0+0.5)/3; user 9: 1, 1, 1."""
        res = self._compute()
        assert res["num_users"] == 2
        assert res["recall_user"][1] == pytest.approx((1 / 3 + 1.0) / 2)
        assert res["recall_user"][3] == pytest.approx((2 / 3 + 1.0) / 2)
        assert res["ndcg_user"][3] == pytest.approx((0.5 + 1.0) / 2)

    def test_hierarchical_user_averages_rows_per_user_first(self):
        """Full-depth hierarchical recall per user equals recall_user; the row-averaged version differs."""
        res = self._compute()
        for k in (1, 3):
            assert res["hierarchical_user"][f"h@{k}_slice_2"] == pytest.approx(res["recall_user"][k])
        # slice 1 (first column only): user 7 hits at k=1 in row 1 only -> 1/3, user 9 -> 1
        assert res["hierarchical_user"]["h@1_slice_1"] == pytest.approx((1 / 3 + 1.0) / 2)
        # row average: 2 of 4 rows hit at k=1, differs from the user average (2/3)
        assert res["hierarchical"]["h@1_slice_2"] == pytest.approx(0.5)
        assert res["hierarchical_user"]["h@1_slice_2"] == pytest.approx(2 / 3)

    def test_last_metrics_use_only_flagged_rows(self):
        """Rows 3 and 4: recall@1 = (0+1)/2, recall@3 = 1, ndcg@3 = (0.5+1)/2."""
        res = self._compute()
        assert res["recall_last"][1] == pytest.approx(0.5)
        assert res["recall_last"][3] == pytest.approx(1.0)
        assert res["ndcg_last"][3] == pytest.approx(0.75)

    def test_seen_at_k_counts_slots_with_seen_tuples(self):
        """Top-3 slots holding a seen tuple per row: 1, 1, 2, 0 of 12 slots; top-1: rows 1 and 3 of 4."""
        res = self._compute()
        assert res["seen_at_k"][3] == pytest.approx(4 / 12)
        assert res["seen_at_k"][1] == pytest.approx(2 / 4)

    def test_users_with_seen_at_k(self):
        """Seen items per row in the top-3: row 1 slot 0, row 2 slot 2, row 3 slots 1 and 2, row 4 none.

        k=1: only rows 1 and 3 hit at slot 0 -> user 7 yes, user 9 no -> 0.5; k=3: user 7 yes -> 0.5.
        """
        res = self._compute()
        assert res["users_with_seen_at_k"] == {1: pytest.approx(0.5), 3: pytest.approx(0.5)}

    def test_padding_in_seen_never_matches(self):
        """-1 padded seen rows are ignored even if a prediction were all -1."""
        acc = MetricAccumulator(k_list=[1], num_layers=2)
        acc.update(torch.tensor([[[-1, -1]]]), torch.tensor([[[3, 3]]]), user_ids=torch.tensor([0]),
                   seen_tuples=torch.tensor([[[-1, -1], [-1, -1]]]), is_last=torch.tensor([True]))
        res = acc.compute()
        assert res["seen_at_k"][1] == 0.0
        assert res["users_with_seen_at_k"][1] == 0.0

    def test_no_last_row_raises(self):
        acc = MetricAccumulator(k_list=[1], num_layers=2)
        acc.update(torch.tensor([[[1, 1]]]), torch.tensor([[[1, 1]]]), user_ids=torch.tensor([0]),
                   is_last=torch.tensor([False]))
        with pytest.raises(ValueError, match="is_last"):
            acc.compute()

    def test_extra_keys_absent_without_optional_inputs(self):
        """update(predictions, targets) keeps the old report: no user / last / seen keys."""
        acc = MetricAccumulator(k_list=[1], num_layers=2)
        acc.update(torch.tensor([[[1, 1]]]), torch.tensor([[[1, 1]]]))
        res = acc.compute()
        assert set(res) == {"recall", "ndcg", "hierarchical", "total_samples"}
        assert "hierarchical_user" not in res

    def test_reset_clears_user_state(self):
        acc = MetricAccumulator(k_list=[1, 3], num_layers=2)
        for b in _extended_batches():
            acc.update(**b)
        acc.reset()
        b = _extended_batches()[1]
        acc.update(**b)
        res = acc.compute()
        assert res["num_users"] == 2
        assert res["recall_user"][3] == pytest.approx(1.0)

    def test_loo_identity_with_and_without_optional_inputs(self):
        """Passing user ids / seen tuples / is_last does not change recall, ndcg or hierarchical values."""
        g = torch.Generator().manual_seed(0)
        k_list = [1, 5, 10]
        old, new = MetricAccumulator(k_list=k_list, num_layers=4), MetricAccumulator(k_list=k_list, num_layers=4)
        for start in (0, 16):
            preds = torch.randint(0, 3, (16, 10, 4), generator=g)
            targets = torch.randint(0, 3, (16, 1, 4), generator=g)
            seen = torch.randint(0, 3, (16, 6, 4), generator=g)
            old.update(preds, targets)
            new.update(preds, targets, user_ids=torch.arange(start, start + 16), seen_tuples=seen,
                       is_last=torch.ones(16, dtype=torch.bool))
        r_old, r_new = old.compute(), new.compute()
        for key in ("recall", "ndcg", "hierarchical", "total_samples"):
            assert r_old[key] == r_new[key]
        # one row per user: user and last averages equal the row averages
        for k in k_list:
            assert r_new["recall_user"][k] == pytest.approx(r_new["recall"][k])
            assert r_new["recall_last"][k] == pytest.approx(r_new["recall"][k], abs=1e-12)
            # NDCG rows are float32 and the row average sums them in float32, so only the default tolerance holds
            assert r_new["ndcg_last"][k] == pytest.approx(r_new["ndcg"][k])

    def test_duplicate_predictions_count_first_hit_only(self):
        """Target at slots 0 and 2 of the top-3: recall 1, NDCG 1/log2(2) = 1.0 (not 1 + 1/log2(4) = 1.5)."""
        acc = MetricAccumulator(k_list=[3], num_layers=2)
        pred = torch.tensor([[[1, 1], [2, 2], [1, 1]]])
        acc.update(pred, torch.tensor([[[1, 1]]]), user_ids=torch.tensor([0]))
        res = acc.compute()
        assert res["recall_user"][3] == 1.0
        assert res["ndcg_user"][3] == pytest.approx(1.0)
        assert res["ndcg"][3] == pytest.approx(1.0)

    def test_recall_user_equals_hierarchical_user_exactly(self):
        """Bit-identical, not approx: recall_user[k] == hierarchical_user[h@k_slice_<layers>]."""
        g = torch.Generator().manual_seed(1)
        k_list = [1, 3, 5]
        acc = MetricAccumulator(k_list=k_list, num_layers=3)
        for _ in range(2):
            # 3 users drawn at random, so each user has several rows spread over both batches
            acc.update(torch.randint(0, 2, (12, 5, 3), generator=g), torch.randint(0, 2, (12, 1, 3), generator=g),
                       user_ids=torch.randint(0, 3, (12,), generator=g))
        res = acc.compute()
        assert res["num_users"] == 3
        for k in k_list:
            assert res["recall_user"][k] == res["hierarchical_user"][f"h@{k}_slice_3"]

    def test_users_with_seen_at_k_follows_k_list(self):
        """k_list=[1, 5], three users with one row each, predictions (1,1) (2,2) (3,3) (4,4) (5,5).

        User 0 has seen (4,4), at slot 3: in the top-5, not in the top-1. User 1 has seen (1,1), at slot 0:
        in both. User 2 has seen (9,9), not predicted. Share of users: k=1 -> 1/3, k=5 -> 2/3.
        """
        acc = MetricAccumulator(k_list=[1, 5], num_layers=2)
        preds = torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                              [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                              [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]])
        seen = torch.tensor([[[4, 4], [-1, -1]], [[1, 1], [-1, -1]], [[9, 9], [-1, -1]]])
        acc.update(preds, torch.full((3, 1, 2), 7), user_ids=torch.tensor([0, 1, 2]), seen_tuples=seen)
        res = acc.compute()
        assert res["users_with_seen_at_k"] == {1: pytest.approx(1 / 3), 5: pytest.approx(2 / 3)}
        assert not any(key.endswith("10") for key in res if isinstance(key, str))

    def test_is_last_without_user_ids(self):
        """is_last alone gives recall_last / ndcg_last but no per-user keys."""
        acc = MetricAccumulator(k_list=[1], num_layers=2)
        acc.update(torch.tensor([[[1, 1]], [[2, 2]]]), torch.tensor([[[1, 1]], [[3, 3]]]),
                   is_last=torch.tensor([False, True]))
        res = acc.compute()
        assert res["recall_last"][1] == 0.0  # only the second row counts and it misses
        assert not {"recall_user", "ndcg_user", "hierarchical_user", "num_users"} & set(res)

    def test_seen_tuples_without_user_ids(self):
        """seen_tuples alone gives seen_at_k but no users_with_seen_at_k."""
        acc = MetricAccumulator(k_list=[1], num_layers=2)
        acc.update(torch.tensor([[[1, 1]]]), torch.tensor([[[2, 2]]]), seen_tuples=torch.tensor([[[1, 1]]]))
        res = acc.compute()
        assert res["seen_at_k"][1] == 1.0
        assert "users_with_seen_at_k" not in res

    @pytest.mark.parametrize("first, second", [
        (dict(user_ids=torch.tensor([0])), {}),
        ({}, dict(user_ids=torch.tensor([0]))),
        (dict(is_last=torch.tensor([True])), {}),
        (dict(seen_tuples=torch.tensor([[[1, 1]]])), {}),
        ({}, dict(seen_tuples=torch.tensor([[[1, 1]]]))),
    ])
    def test_inconsistent_optional_inputs_raise(self, first, second):
        """A field given in one batch but not in the other raises, in either order."""
        acc = MetricAccumulator(k_list=[1], num_layers=2)
        pred, target = torch.tensor([[[1, 1]]]), torch.tensor([[[1, 1]]])
        acc.update(pred, target, **first)
        with pytest.raises(ValueError, match=next(iter(first or second))):
            acc.update(pred, target, **second)

    def test_too_few_prediction_columns_raise(self):
        """Beam of 3 columns with max(k_list) = 5 is an error, not silently shorter lists."""
        acc = MetricAccumulator(k_list=[1, 5], num_layers=2)
        with pytest.raises(ValueError, match="3 columns"):
            acc.update(torch.zeros(1, 3, 2, dtype=torch.long), torch.zeros(1, 1, 2, dtype=torch.long))


def test_calculate_entropy_and_coverage_tensor():
    # Batch size = 4, Layers = 2, Codebook size = 10
    # Layer 1 has 4 unique values [0, 1, 2, 3] -> coverage 0.4, entropy log(4)
    # Layer 2 has 1 unique value [5, 5, 5, 5] -> coverage 0.1, entropy 0.0
    ids = torch.tensor([[0, 5], [1, 5], [2, 5], [3, 5]])
    
    coverages, entropies = calculate_entropy_and_coverage(ids, codebook_size=10)
    
    assert torch.equal(coverages, torch.tensor([0.4, 0.1]))
    assert torch.allclose(entropies, torch.tensor([math.log(4), 0.0]))
    assert coverages.device == ids.device
    assert entropies.device == ids.device


def test_calculate_entropy_and_coverage_list():
    # Test with list of 1D tensors
    ids_list = [torch.tensor([0, 1, 2, 3]), torch.tensor([5, 5, 5, 5])]
    
    coverages, entropies = calculate_entropy_and_coverage(ids_list, codebook_size=10)
    
    assert torch.equal(coverages, torch.tensor([0.4, 0.1]))
    assert torch.allclose(entropies, torch.tensor([math.log(4), 0.0]))
    assert coverages.device == ids_list[0].device
    assert entropies.device == ids_list[0].device


def test_calculate_quantizer_metrics_basic():
    # Test basic usage with only ids and codebook_size (no residuals, no codebooks)
    ids = torch.tensor([[0, 5], [1, 5], [2, 5], [3, 5]])
    metrics = calculate_quantizer_metrics(ids, codebook_size=10)
    
    assert "p_unique_ids" in metrics
    assert "layer_coverages" in metrics
    assert "layer_entropies" in metrics
    
    # 4 unique sequence rows out of 4 batch samples -> 1.0 unique ratio
    assert metrics["p_unique_ids"].item() == 1.0
    assert torch.equal(metrics["layer_coverages"], torch.tensor([0.4, 0.1]))
    assert torch.allclose(metrics["layer_entropies"], torch.tensor([math.log(4), 0.0]))
    
    # Check that residual and centroid metrics are not present
    assert "first_residual_norm" not in metrics
    assert "last_residual_norm" not in metrics
    assert "first_residual_rel" not in metrics
    assert "last_residual_rel" not in metrics
    assert "first_centroids_norm" not in metrics
    assert "last_centroids_norm" not in metrics


def test_calculate_quantizer_metrics_with_residuals_and_codebooks():
    ids = torch.tensor([[0, 5], [1, 5], [2, 5], [3, 5]])
    input_tensor = torch.ones(4, 8) * 3.0  # L2 norm is sqrt(8 * 9) = sqrt(72) = 8.485
    first_residual = torch.ones(4, 8) * 1.5
    final_residual = torch.ones(4, 8) * 0.3
    
    # 2 layers of codebooks, shape (10, 8)
    codebooks = [
        torch.ones(10, 8) * 2.0,
        torch.ones(10, 8) * 0.5
    ]
    
    metrics = calculate_quantizer_metrics(
        ids=ids,
        codebook_size=10,
        input_tensor=input_tensor,
        first_residual=first_residual,
        final_residual=final_residual,
        codebooks=codebooks
    )
    
    assert "first_residual_norm" in metrics
    assert "last_residual_norm" in metrics
    assert "first_residual_rel" in metrics
    assert "last_residual_rel" in metrics
    assert "first_centroids_norm" in metrics
    assert "last_centroids_norm" in metrics
    
    # Calculate expected absolute norms
    sqrt_8 = math.sqrt(8)
    expected_input_norm = 3.0 * sqrt_8
    expected_first_res_norm = 1.5 * sqrt_8
    expected_final_res_norm = 0.3 * sqrt_8
    
    assert metrics["first_residual_norm"].item() == pytest.approx(expected_first_res_norm)
    assert metrics["last_residual_norm"].item() == pytest.approx(expected_final_res_norm)
    
    # Ratios
    assert metrics["first_residual_rel"].item() == pytest.approx(0.5)
    assert metrics["last_residual_rel"].item() == pytest.approx(0.1)
    
    # Codebooks centroids norms
    expected_cb0_norm = 2.0 * sqrt_8
    expected_cb1_norm = 0.5 * sqrt_8
    assert metrics["first_centroids_norm"].item() == pytest.approx(expected_cb0_norm)
    assert metrics["last_centroids_norm"].item() == pytest.approx(expected_cb1_norm)


def test_latent_norm_is_the_mean_norm_over_the_batch():
    # norms 5 and 0
    ids = torch.tensor([[0, 0], [1, 1]])
    metrics = calculate_quantizer_metrics(ids, codebook_size=2, input_tensor=torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    assert metrics["latent_norm"].item() == 2.5
