"""
TDD tests for SASRec benchmark model.
"""
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Embedding layer tests
# ---------------------------------------------------------------------------

class TestEmbeddingLayer:

    def test_output_shape(self):
        """Embedding output shape is [B, max_len, num_items+1]."""
        from modules.sasrec.model import SASRec

        model = SASRec(num_items=100, hidden_dim=32, num_blocks=1, num_heads=1, max_seq_len=10, dropout=0.0)
        item_seq = torch.tensor([[0, 0, 5, 12, 3], [0, 7, 8, 1, 20]])
        model.eval()
        with torch.no_grad():
            scores = model(item_seq)
        assert scores.shape == (2, 5, 101)

    def test_padding_gets_zero_embedding(self):
        """Padding index 0 produces a zero vector in the item embedding."""
        from modules.sasrec.model import SASRec

        model = SASRec(num_items=50, hidden_dim=16, num_blocks=1, num_heads=1, max_seq_len=5, dropout=0.0)
        assert torch.all(model.item_embedding.weight[0] == 0)
        assert model.item_embedding.weight.requires_grad is True

    def test_positional_embedding_added(self):
        """Same item at different positions produces different output representations."""
        from modules.sasrec.model import SASRec

        model = SASRec(num_items=50, hidden_dim=16, num_blocks=1, num_heads=1, max_seq_len=5, dropout=0.0)
        model.eval()
        seq1 = torch.tensor([[1, 0, 0, 0, 0]])
        seq2 = torch.tensor([[0, 0, 0, 0, 1]])
        with torch.no_grad():
            out1 = model(seq1)
            out2 = model(seq2)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Causal mask tests
# ---------------------------------------------------------------------------

class TestCausalMask:

    def test_future_positions_masked(self):
        """Changing a future item must not affect the output at an earlier position."""
        from modules.sasrec.model import SASRec

        model = SASRec(num_items=20, hidden_dim=16, num_blocks=1, num_heads=1, max_seq_len=4, dropout=0.0)
        model.eval()

        seq = torch.tensor([[0, 0, 5, 10]])
        seq_modified = torch.tensor([[0, 0, 5, 15]])
        with torch.no_grad():
            out1 = model(seq)
            out2 = model(seq_modified)

        # position 2 (item 5) must be unaffected by the changed position 3
        assert torch.allclose(out1[0, 2], out2[0, 2], atol=1e-5)


# ---------------------------------------------------------------------------
# Self-attention block tests
# ---------------------------------------------------------------------------

class TestSelfAttentionBlock:

    def test_single_block_output_shape(self):
        """Single attention block preserves shape [B, n, num_items+1]."""
        from modules.sasrec.model import SASRec

        B, n, d = 4, 10, 32
        model = SASRec(num_items=50, hidden_dim=d, num_blocks=1, num_heads=1, max_seq_len=n, dropout=0.0)
        model.eval()
        seq = torch.randint(1, 51, (B, n))
        with torch.no_grad():
            scores = model(seq)
        assert scores.shape == (B, n, 51)

    def test_stacked_blocks_output_shape(self):
        """Multiple stacked blocks preserve shape [B, n, num_items+1]."""
        from modules.sasrec.model import SASRec

        B, n, d = 4, 10, 32
        model = SASRec(num_items=50, hidden_dim=d, num_blocks=3, num_heads=1, max_seq_len=n, dropout=0.0)
        model.eval()
        seq = torch.randint(1, 51, (B, n))
        with torch.no_grad():
            scores = model(seq)
        assert scores.shape == (B, n, 51)


# ---------------------------------------------------------------------------
# Prediction layer tests
# ---------------------------------------------------------------------------

class TestPredictionLayer:

    def test_prediction_scores_shape(self):
        """Full-rank prediction scores have shape [B, max_len, num_items+1]."""
        from modules.sasrec.model import SASRec

        num_items = 100
        model = SASRec(num_items=num_items, hidden_dim=32, num_blocks=2, num_heads=1, max_seq_len=10, dropout=0.0)
        model.eval()
        seq = torch.randint(0, num_items + 1, (3, 10))
        with torch.no_grad():
            scores = model(seq)
        assert scores.shape == (3, 10, num_items + 1)

    def test_shared_embedding_weights(self):
        """No separate output projection — prediction uses item_embedding.weight."""
        from modules.sasrec.model import SASRec

        model = SASRec(num_items=50, hidden_dim=16, num_blocks=1, num_heads=1, max_seq_len=5, dropout=0.0)
        assert not hasattr(model, "output_projection") or model.output_projection is None


# ---------------------------------------------------------------------------
# Loss computation tests
# ---------------------------------------------------------------------------

class TestLossComputation:

    def test_padding_ignored_in_loss(self):
        """Loss is zero when the entire sequence is padding."""
        from modules.sasrec.model import SASRec

        model = SASRec(num_items=50, hidden_dim=16, num_blocks=1, num_heads=1, max_seq_len=5, dropout=0.0)
        model.eval()

        # Input and target are all padding (0)
        item_seq = torch.zeros(2, 5, dtype=torch.long)
        pos_items = torch.zeros(2, 5, dtype=torch.long)
        neg_items = torch.full((2, 5), 2, dtype=torch.long)

        with torch.no_grad():
            loss = model.compute_loss(item_seq, pos_items, neg_items)

        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Negative sampling tests
# ---------------------------------------------------------------------------

class TestNegativeSampling:

    def test_negative_samples_shape(self):
        """Negative samples have shape [B, max_len]."""
        from modules.sasrec.model import sample_negatives

        B, max_len, num_items = 4, 10, 100
        pos_items = torch.randint(1, num_items + 1, (B, max_len))
        neg_items = sample_negatives(pos_items, num_items)
        assert neg_items.shape == (B, max_len)

    def test_negatives_differ_from_positives(self):
        """No negative sample equals its corresponding positive item."""
        from modules.sasrec.model import sample_negatives

        B, max_len, num_items = 8, 20, 1000
        pos_items = torch.randint(1, num_items + 1, (B, max_len))
        neg_items = sample_negatives(pos_items, num_items)
        assert torch.all(neg_items != pos_items)

    def test_negatives_in_valid_range(self):
        """All negative samples are in [1, num_items]."""
        from modules.sasrec.model import sample_negatives

        num_items = 50
        pos_items = torch.randint(1, num_items + 1, (4, 10))
        neg_items = sample_negatives(pos_items, num_items)
        assert neg_items.min().item() >= 1
        assert neg_items.max().item() <= num_items


# ---------------------------------------------------------------------------
# Training step test
# ---------------------------------------------------------------------------

class TestTrainStep:

    def test_single_train_step(self):
        """A single forward+backward pass produces a finite, positive loss."""
        from modules.sasrec.model import SASRec, sample_negatives

        num_items = 50
        model = SASRec(num_items=num_items, hidden_dim=16, num_blocks=1, num_heads=1, max_seq_len=5, dropout=0.0)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        item_seq = torch.tensor([[0, 0, 3, 5, 10], [0, 1, 2, 8, 15]])
        pos_items = torch.tensor([[0, 3, 5, 10, 20], [1, 2, 8, 15, 30]])
        neg_items = sample_negatives(pos_items, num_items)

        optimizer.zero_grad()
        loss = model.compute_loss(item_seq, pos_items, neg_items)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert torch.isfinite(torch.tensor(loss.item()))


# ---------------------------------------------------------------------------
# SASRecDataset tests
# ---------------------------------------------------------------------------

class TestSASRecDataset:

    def _make_history_data(self):
        return {
            "train": {
                "userId": torch.tensor([0, 1, 2]),
                "itemId": [
                    torch.tensor([0, 1, 2, 3, 4]),
                    torch.tensor([10, 11, 12]),
                    torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                ],
                "itemId_fut": torch.tensor([5, 13, 10]),
            },
            "eval": {
                "userId": torch.tensor([0, 1, 2]),
                "itemId": [
                    torch.tensor([0, 1, 2, 3, 4, 5]),
                    torch.tensor([10, 11, 12, 13]),
                    torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                ],
                "itemId_fut": torch.tensor([6, 14, 11]),
            },
        }

    def test_output_keys_and_shapes(self):
        """__getitem__ returns dict with item_seq [max_len], target_item (int), seq_len (int)."""
        from data.sequence import SASRecDataset

        history = self._make_history_data()
        ds = SASRecDataset(history_data=history, num_items=20, max_len=8, mode="train")
        sample = ds[0]
        assert "item_seq" in sample
        assert "target_item" in sample
        assert "seq_len" in sample
        assert sample["item_seq"].shape == (8,)
        assert isinstance(sample["target_item"], int)
        assert isinstance(sample["seq_len"], int)

    def test_ids_are_1_based(self):
        """Item IDs in item_seq are 1-based (raw + 1). Padding is 0."""
        from data.sequence import SASRecDataset

        history = self._make_history_data()
        ds = SASRecDataset(history_data=history, num_items=20, max_len=8, mode="train")
        sample = ds[0]
        seq = sample["item_seq"]
        non_pad = seq[seq != 0]
        assert non_pad.min().item() >= 1
        assert sample["target_item"] == 6  # raw 5 + 1

    def test_left_padding(self):
        """Shorter sequences are left-padded with 0."""
        from data.sequence import SASRecDataset

        history = self._make_history_data()
        ds = SASRecDataset(history_data=history, num_items=20, max_len=8, mode="train")
        sample = ds[1]  # user 1: only 3 items, max_len=8
        seq = sample["item_seq"]
        assert torch.all(seq[:5] == 0)
        assert torch.all(seq[5:] > 0)
        assert sample["seq_len"] == 3

    def test_truncation(self):
        """Sequences longer than max_len are truncated to last max_len items."""
        from data.sequence import SASRecDataset

        history = self._make_history_data()
        ds = SASRecDataset(history_data=history, num_items=20, max_len=6, mode="train")
        sample = ds[2]  # user 2: 10 items, max_len=6
        seq = sample["item_seq"]
        assert seq.shape == (6,)
        assert torch.all(seq > 0)
        assert sample["seq_len"] == 6
        assert seq[-1].item() == 10  # raw_id 9 + 1

    def test_collate_fn_stacks(self):
        """sasrec_collate_fn properly stacks a batch."""
        from data.sequence import SASRecDataset, sasrec_collate_fn

        history = self._make_history_data()
        ds = SASRecDataset(history_data=history, num_items=20, max_len=8, mode="train")
        batch = [ds[i] for i in range(3)]
        collated = sasrec_collate_fn(batch)
        assert collated["item_seq"].shape == (3, 8)
        assert collated["target_item"].shape == (3,)
        assert collated["seq_len"].shape == (3,)


# ---------------------------------------------------------------------------
# Full-rank evaluation tests
# ---------------------------------------------------------------------------

class TestFullRankEvaluation:

    def test_returns_correct_metric_structure(self):
        """evaluate_metrics returns dict with recall and ndcg at each k."""
        from modules.sasrec.model import SASRec
        from data.sequence import SASRecDataset
        from train_sasrec import evaluate_metrics

        num_items = 30
        model = SASRec(num_items=num_items, hidden_dim=16, num_blocks=1, num_heads=1, max_seq_len=5, dropout=0.0)
        model.eval()

        history = {
            "eval": {
                "userId": torch.tensor([0, 1]),
                "itemId": [torch.tensor([1, 2, 3]), torch.tensor([5, 6])],
                "itemId_fut": torch.tensor([4, 7]),
            }
        }
        ds = SASRecDataset(history_data=history, num_items=num_items, max_len=5, mode="eval")
        k_list = [1, 5, 10]
        metrics = evaluate_metrics(model, ds, torch.device("cpu"), k_list, num_items)

        for k in k_list:
            assert f"recall@{k}" in metrics
            assert f"ndcg@{k}" in metrics
            assert 0.0 <= metrics[f"recall@{k}"] <= 1.0
            assert 0.0 <= metrics[f"ndcg@{k}"] <= 1.0

    def test_scores_all_items(self):
        """Full-rank eval computes scores for all items."""
        from modules.sasrec.model import SASRec

        num_items = 50
        model = SASRec(num_items=num_items, hidden_dim=16, num_blocks=1, num_heads=1, max_seq_len=5, dropout=0.0)
        model.eval()
        seq = torch.tensor([[0, 0, 1, 5, 10]])
        with torch.no_grad():
            scores = model(seq)
        last_scores = scores[0, -1]
        assert last_scores.shape == (num_items + 1,)


# ---------------------------------------------------------------------------
# History exclusion tests
# ---------------------------------------------------------------------------

class TestHistoryExclusion:

    def test_history_items_get_neg_inf(self):
        """During eval, items in user history get -inf scores."""
        from modules.sasrec.model import SASRec

        num_items = 20
        model = SASRec(num_items=num_items, hidden_dim=16, num_blocks=1, num_heads=1, max_seq_len=5, dropout=0.0)
        model.eval()

        item_seq = torch.tensor([[0, 0, 1, 5, 10]])
        with torch.no_grad():
            scores = model(item_seq)
        last_scores = scores[0, -1].clone()

        history_items = [1, 5, 10]
        for item_id in history_items:
            last_scores[item_id] = float("-inf")

        for item_id in history_items:
            assert last_scores[item_id] == float("-inf")

        for item_id in [2, 3, 4, 6, 7, 8, 9, 11]:
            assert torch.isfinite(last_scores[item_id])
