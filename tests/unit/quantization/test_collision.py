"""
Tests for resolve_collisions in generate_semids.py

Covers:
  - No collisions: all collision tokens are 0
  - Partial collisions: tokens assigned correctly within each group
  - Overflow: ValueError raised when max depth exceeds max_collisions
"""
import pytest
import torch
from generate_semids import resolve_collisions


class TestResolveCollisions:

    def test_all_unique_collision_tokens_are_zero(self):
        """When every tuple is unique, all collision tokens should be 0."""
        sids = torch.stack([torch.arange(100),
                            torch.zeros(100, dtype=torch.long),
                            torch.zeros(100, dtype=torch.long)], dim=1)
        result = resolve_collisions(sids, max_collisions=128)
        assert result.shape == (100, 4)
        assert result[:, -1].max().item() == 0

    def test_output_shape_adds_one_layer(self):
        """resolve_collisions appends exactly one column."""
        sids = torch.randint(0, 16, (50, 3))
        result = resolve_collisions(sids, max_collisions=128)
        assert result.shape == (50, 4)

    def test_collision_group_gets_unique_tokens(self):
        """Items sharing the same tuple receive distinct collision tokens 0..n-1."""
        n = 10
        sids = torch.full((n, 3), 42, dtype=torch.long)
        result = resolve_collisions(sids, max_collisions=128)
        tokens = result[:, -1].sort().values
        assert tokens.tolist() == list(range(n))

    def test_multiple_groups_tokenised_independently(self):
        """Each collision group counts from 0 independently."""
        a = torch.full((5, 3), 1, dtype=torch.long)
        b = torch.full((3, 3), 2, dtype=torch.long)
        sids = torch.cat([a, b], dim=0)
        result = resolve_collisions(sids, max_collisions=128)
        tokens_a = result[:5, -1].sort().values
        tokens_b = result[5:, -1].sort().values
        assert tokens_a.tolist() == [0, 1, 2, 3, 4]
        assert tokens_b.tolist() == [0, 1, 2]

    def test_original_sids_preserved(self):
        """The first n_layers columns must be identical to the input."""
        sids = torch.randint(0, 32, (50, 3))
        result = resolve_collisions(sids, max_collisions=128)
        assert torch.equal(result[:, :-1], sids)

    def test_single_item(self):
        """A single item should return shape (1, n_layers+1) with collision token 0."""
        sids = torch.tensor([[5, 3, 1]])
        result = resolve_collisions(sids, max_collisions=128)
        assert result.shape == (1, 4)
        assert result[0, -1].item() == 0

    def test_overflow_raises_value_error(self):
        """Collision depth exceeding max_collisions must raise ValueError."""
        sids = torch.full((10, 3), 99, dtype=torch.long)
        with pytest.raises(ValueError, match="Collision overflow"):
            resolve_collisions(sids, max_collisions=5)

    def test_overflow_boundary_just_below_limit_passes(self):
        """max_depth == max_collisions - 1 should not raise (boundary: >=, not >)."""
        n = 5
        sids = torch.full((n, 3), 7, dtype=torch.long)
        result = resolve_collisions(sids, max_collisions=n)
        assert result[:, -1].max().item() == n - 1

    def test_overflow_boundary_at_limit_raises(self):
        """max_depth == max_collisions should raise (the >= condition)."""
        n = 5
        sids = torch.full((n + 1, 3), 7, dtype=torch.long)
        with pytest.raises(ValueError, match="Collision overflow"):
            resolve_collisions(sids, max_collisions=n)
