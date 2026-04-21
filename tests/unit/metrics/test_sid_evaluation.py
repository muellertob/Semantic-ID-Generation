"""
Unit tests for utils/sid_evaluation.py

Covers compute_utilisation and compute_collision_stats in isolation.
"""
import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_semids():
    """Every code used exactly once — maximum perplexity case."""
    def _make(codebook_size):
        return torch.arange(codebook_size).unsqueeze(1)
    return _make


@pytest.fixture
def single_code_semids():
    """Only code 0 used — minimum perplexity case."""
    def _make(n, n_layers, codebook_size):
        return torch.zeros(n, n_layers, dtype=torch.long)
    return _make


@pytest.fixture
def random_semids():
    def _make(n, n_layers, codebook_size, seed=0):
        torch.manual_seed(seed)
        return torch.randint(0, codebook_size, (n, n_layers))
    return _make


@pytest.fixture
def unique_tuple_semids():
    """All items have unique SID tuples — zero collision case."""
    def _make(n, n_layers=2):
        cols = [torch.arange(n)] + [torch.zeros(n, dtype=torch.long)] * (n_layers - 1)
        return torch.stack(cols, dim=1)
    return _make


@pytest.fixture
def identical_tuple_semids():
    """All items share one SID tuple — maximum collision case."""
    def _make(n, n_layers=3):
        return torch.zeros(n, n_layers, dtype=torch.long)
    return _make


# ---------------------------------------------------------------------------
# compute_utilisation
# ---------------------------------------------------------------------------

class TestComputeUtilisation:

    @pytest.fixture(autouse=True)
    def _import(self):
        from utils.sid_evaluation import compute_utilisation
        self.compute_utilisation = compute_utilisation

    @pytest.mark.parametrize("n_layers", [1, 2, 3, 5])
    def test_per_layer_list_length(self, random_semids, n_layers):
        sids = random_semids(100, n_layers, 32)
        result = self.compute_utilisation(sids, codebook_size=32)
        assert len(result["per_layer"]) == n_layers

    def test_uniform_usage_gives_max_perplexity(self, uniform_semids):
        cb = 64
        sids = uniform_semids(cb)
        layer = self.compute_utilisation(sids, codebook_size=cb)["per_layer"][0]
        assert layer["perplexity"] == pytest.approx(cb, rel=1e-3)

    def test_uniform_usage_gives_full_coverage(self, uniform_semids):
        cb = 64
        sids = uniform_semids(cb)
        layer = self.compute_utilisation(sids, codebook_size=cb)["per_layer"][0]
        assert layer["coverage"] == pytest.approx(1.0)
        assert layer["codes_used"] == cb

    def test_uniform_usage_gives_perplexity_ratio_one(self, uniform_semids):
        cb = 32
        sids = uniform_semids(cb)
        layer = self.compute_utilisation(sids, codebook_size=cb)["per_layer"][0]
        assert layer["perplexity_ratio"] == pytest.approx(1.0, rel=1e-3)

    def test_single_code_gives_min_perplexity(self, single_code_semids):
        cb = 128
        sids = single_code_semids(200, 1, cb)
        layer = self.compute_utilisation(sids, codebook_size=cb)["per_layer"][0]
        assert layer["perplexity"] == pytest.approx(1.0, rel=1e-3)
        assert layer["codes_used"] == 1

    def test_single_code_gives_min_coverage(self, single_code_semids):
        cb = 128
        sids = single_code_semids(200, 1, cb)
        layer = self.compute_utilisation(sids, codebook_size=cb)["per_layer"][0]
        assert layer["coverage"] == pytest.approx(1 / cb)

    def test_single_code_gives_min_perplexity_ratio(self, single_code_semids):
        cb = 128
        sids = single_code_semids(200, 1, cb)
        layer = self.compute_utilisation(sids, codebook_size=cb)["per_layer"][0]
        assert layer["perplexity_ratio"] == pytest.approx(1.0 / cb, rel=1e-3)

    @pytest.mark.parametrize("cb", [16, 32, 64])
    def test_perplexity_bounded_by_codebook_size(self, random_semids, cb):
        sids = random_semids(500, 3, cb)
        result = self.compute_utilisation(sids, codebook_size=cb)
        for u in result["per_layer"]:
            assert 1.0 <= u["perplexity"] <= cb + 1e-6

    @pytest.mark.parametrize("cb", [8, 32, 128])
    def test_max_perplexity_field_equals_codebook_size(self, random_semids, cb):
        sids = random_semids(100, 2, cb)
        result = self.compute_utilisation(sids, codebook_size=cb)
        for u in result["per_layer"]:
            assert u["max_perplexity"] == cb

    @pytest.mark.parametrize("cb", [16, 64])
    def test_perplexity_ratio_in_unit_interval(self, random_semids, cb):
        sids = random_semids(300, 2, cb)
        result = self.compute_utilisation(sids, codebook_size=cb)
        for u in result["per_layer"]:
            assert 0.0 <= u["perplexity_ratio"] <= 1.0 + 1e-6

    def test_layer_index_matches_position(self, random_semids):
        sids = random_semids(100, 4, 16)
        result = self.compute_utilisation(sids, codebook_size=16)
        for i, u in enumerate(result["per_layer"]):
            assert u["layer"] == i


# ---------------------------------------------------------------------------
# compute_collision_stats
# ---------------------------------------------------------------------------

class TestComputeCollisionStats:

    @pytest.fixture(autouse=True)
    def _import(self):
        from utils.sid_evaluation import compute_collision_stats
        self.compute_collision_stats = compute_collision_stats

    def test_unique_tuples_zero_collision_rate(self, unique_tuple_semids):
        sids = unique_tuple_semids(200)
        result = self.compute_collision_stats(sids)
        assert result["collision_rate"] == 0.0

    def test_unique_tuples_zero_n_collisions(self, unique_tuple_semids):
        sids = unique_tuple_semids(200)
        result = self.compute_collision_stats(sids)
        assert result["n_collisions"] == 0
        assert result["max_depth"] == 0

    def test_identical_tuples_collision_count(self, identical_tuple_semids):
        n = 100
        sids = identical_tuple_semids(n)
        result = self.compute_collision_stats(sids)
        assert result["n_collisions"] == n - 1

    def test_identical_tuples_collision_rate(self, identical_tuple_semids):
        n = 100
        sids = identical_tuple_semids(n)
        result = self.compute_collision_stats(sids)
        assert result["collision_rate"] == pytest.approx((n - 1) / n)

    def test_identical_tuples_max_depth(self, identical_tuple_semids):
        n = 100
        sids = identical_tuple_semids(n)
        result = self.compute_collision_stats(sids)
        assert result["max_depth"] == n - 1

    def test_partial_collision_count(self, unique_tuple_semids):
        unique = unique_tuple_semids(50, n_layers=3)
        shared = torch.full((50, 3), 99, dtype=torch.long)
        sids = torch.cat([unique, shared], dim=0)
        result = self.compute_collision_stats(sids)
        assert result["n_collisions"] == 49
        assert result["max_depth"] == 49

    def test_depth_distribution_length_equals_n_items(self, random_semids):
        n = 300
        sids = random_semids(n, 3, 8)
        result = self.compute_collision_stats(sids)
        assert len(result["depth_distribution"]) == n

    def test_required_keys_present(self, random_semids):
        sids = random_semids(100, 3, 16)
        result = self.compute_collision_stats(sids)
        assert {"collision_rate", "n_collisions", "max_depth",
                "depth_distribution", "mean_depth", "median_depth",
                "p90_depth", "p99_depth"}.issubset(result.keys())

    def test_no_collisions_all_depth_stats_zero(self, unique_tuple_semids):
        sids = unique_tuple_semids(50)
        result = self.compute_collision_stats(sids)
        assert result["mean_depth"] == 0.0
        assert result["median_depth"] == 0.0
        assert result["p90_depth"] == 0.0
        assert result["p99_depth"] == 0.0

    @pytest.mark.parametrize("n", [1, 2, 100, 12101])
    def test_various_dataset_sizes_do_not_raise(self, n):
        sids = torch.zeros(n, 3, dtype=torch.long)
        self.compute_collision_stats(sids)

    def test_collision_rate_in_unit_interval(self, random_semids):
        sids = random_semids(200, 3, 8)
        result = self.compute_collision_stats(sids)
        assert 0.0 <= result["collision_rate"] <= 1.0

    def test_depth_distribution_nonnegative(self, random_semids):
        sids = random_semids(100, 3, 8)
        result = self.compute_collision_stats(sids)
        assert (result["depth_distribution"] >= 0).all()
