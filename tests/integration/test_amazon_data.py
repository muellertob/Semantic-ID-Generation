"""
Integration tests for data/amazon_data.py (AmazonReviews.train_test_split).

Uses the `tiny_amazon_dataset` fixture (synthetic files, no real download needed).
Fast — no @pytest.mark.slow required.

Real-data smoke tests (require dataset/amazon/raw/beauty/) are at the bottom,
marked @pytest.mark.slow for optional CI runs.
"""
import os
import pytest

class TestTrainTestSplit:

    def test_five_core_filter_removes_short_users(self, tiny_amazon_dataset):
        """Users with < 5 items are excluded; 10 (8-item) + 1 (5-item) users are kept."""
        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=10)
        assert len(sequences["train"]["itemId"]) == 11

    def test_five_core_filter_boundary_exactly_five_kept(self, tiny_amazon_dataset):
        """User with exactly 5 items must pass the filter (condition is len < 5, strict)."""
        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=10)
        # uid=11 has 5 items → userId column should contain 11
        user_ids = sequences["train"]["userId"].to_list()
        assert 11 in user_ids

    def test_five_core_filter_boundary_four_removed(self, tiny_amazon_dataset):
        """User with exactly 4 items must be filtered out."""
        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=10)
        user_ids = sequences["train"]["userId"].to_list()
        assert 12 not in user_ids

    def test_train_history_excludes_last_three_items(self, tiny_amazon_dataset):
        """Train history = items[:-3] for every user."""
        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=10)
        # Fixture: 10 users × 8 items → history = 5; 1 user × 5 items → history = 2
        hist_lengths = {len(hist) for hist in sequences["train"]["itemId"]}
        assert hist_lengths == {2, 5}

    def test_loo_targets_are_all_distinct(self, tiny_amazon_dataset):
        """train / eval / test future targets must be three different items."""
        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=10)
        n = len(sequences["train"]["itemId_fut"])
        for i in range(n):
            t = sequences["train"]["itemId_fut"][i]
            e = sequences["eval"]["itemId_fut"][i]
            v = sequences["test"]["itemId_fut"][i]
            assert t != e and e != v and t != v, (
                f"LOO target overlap for user {i}: train={t}, eval={e}, test={v}"
            )

    def test_eval_and_test_sequences_padded_to_max_seq_len(self, tiny_amazon_dataset):
        """eval/test sequences must be exactly max_seq_len long (padded with -1)."""
        max_seq_len = 10
        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=max_seq_len)
        for split in ("eval", "test"):
            for seq in sequences[split]["itemId"].to_list():
                assert len(seq) == max_seq_len, (
                    f"{split} sequence has length {len(seq)}, expected {max_seq_len}"
                )

    def test_padding_uses_minus_one(self, tiny_amazon_dataset):
        """Short histories are right-padded with -1 (items first, then -1s, not 0s)."""
        # max_seq_len=10 > available history → padding is applied
        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=10)
        for seq in sequences["eval"]["itemId"].to_list():
            if -1 in seq:
                first_pad = next(i for i, v in enumerate(seq) if v == -1)
                assert all(v == -1 for v in seq[first_pad:]), (
                    f"Non-pad token found after padding started: {seq}"
                )
                assert first_pad > 0, "Sequence has no real items before padding"

    def test_splits_contain_same_number_of_users(self, tiny_amazon_dataset):
        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=10)
        n_train = len(sequences["train"]["itemId"])
        n_eval = len(sequences["eval"]["itemId"])
        n_test = len(sequences["test"]["itemId"])
        assert n_train == n_eval == n_test

    def test_item_ids_are_zero_based(self, tiny_amazon_dataset):
        """_remap_ids converts 1-based file IDs to 0-based; all IDs must be in [0, num_items-1]."""
        import json
        datamaps_path = os.path.join(tiny_amazon_dataset.raw_dir, tiny_amazon_dataset.split, "datamaps.json")
        with open(datamaps_path) as f:
            num_items = len(json.load(f)["item2id"])

        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=10)
        for seq in sequences["train"]["itemId"].to_list():
            assert all(0 <= x < num_items for x in seq), f"Item ID out of [0, {num_items-1}] in train: {seq}"
        for fut in sequences["train"]["itemId_fut"].to_list():
            assert 0 <= fut < num_items, f"Future ID out of [0, {num_items-1}] in train: {fut}"

    def test_eval_window_truncates_when_history_exceeds_max_seq_len(self, tiny_amazon_dataset):
        """When a user has more items than max_seq_len, eval/test sequences are truncated, not padded."""
        max_seq_len = 3
        sequences = tiny_amazon_dataset.train_test_split(max_seq_len=max_seq_len)
        for split in ("eval", "test"):
            for seq in sequences[split]["itemId"].to_list():
                assert len(seq) == max_seq_len
                assert -1 not in seq, f"{split} sequence has unexpected padding: {seq}"

@pytest.mark.slow
def test_real_data_five_core_filtering(category="beauty"):
    root = "dataset/amazon"
    if not os.path.exists(os.path.join(root, "raw", category)):
        pytest.skip(f"Real dataset not found at {root}.")

    from data.amazon_data import AmazonReviews
    dataset = AmazonReviews(root=root, split=category)
    sequences = dataset.train_test_split(max_seq_len=20)

    train_ids = sequences["train"]["itemId"]
    assert len(train_ids) > 0
    user_total_items = [len(items) + 3 for items in train_ids]
    assert min(user_total_items) >= 5


@pytest.mark.slow
def test_real_data_loo_targets_distinct(category="beauty"):
    root = "dataset/amazon"
    if not os.path.exists(os.path.join(root, "raw", category)):
        pytest.skip(f"Real dataset not found at {root}.")

    from data.amazon_data import AmazonReviews
    dataset = AmazonReviews(root=root, split=category)
    sequences = dataset.train_test_split(max_seq_len=20)

    for i in range(min(20, len(sequences["train"]["itemId_fut"]))):
        t = sequences["train"]["itemId_fut"][i]
        e = sequences["eval"]["itemId_fut"][i]
        v = sequences["test"]["itemId_fut"][i]
        assert t != e and e != v and t != v
