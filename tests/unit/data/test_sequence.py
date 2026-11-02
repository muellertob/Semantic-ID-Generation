"""
Tests for SemanticIDSequenceDataset / collate_fn: full histories, the one place where the
model input is cut to max_len, and the evaluation-only fields 'seen_tuples' and 'is_last'.
"""
import torch
from data.sequence import SemanticIDSequenceDataset, collate_fn

# 6 items with 2-layer SIDs: item i -> (i, 10 + i)
SEMANTIC_IDS = torch.stack([torch.arange(6), torch.arange(6) + 10], dim=1)


def _loo_data():
    return {"eval": {
        "userId": torch.tensor([3, 4]),
        "itemId": [[0, 1], [2, 3, 4]],
        "itemId_fut": torch.tensor([5, 5]),
    }}


def _gts_data():
    return {"test": {
        "userId": torch.tensor([3, 3, 4]),
        "itemId": [[0], [0, 1], [2, 3, 4, 0]],
        "itemId_fut": torch.tensor([1, 2, 5]),
        "is_last": torch.tensor([False, True, True]),
    }}


def test_history_and_seen_are_the_full_history():
    ds = SemanticIDSequenceDataset(_loo_data(), SEMANTIC_IDS, mode="eval")
    item = ds[1]
    assert torch.equal(item["history_tuples"], SEMANTIC_IDS[[2, 3, 4]])
    assert torch.equal(item["seen_tuples"], SEMANTIC_IDS[[2, 3, 4]])
    assert torch.equal(item["target_tuples"], SEMANTIC_IDS[[5]])
    assert item["user_id"].item() == 4


def test_without_is_last_every_row_counts_as_last():
    ds = SemanticIDSequenceDataset(_loo_data(), SEMANTIC_IDS, mode="eval")
    assert ds[0]["is_last"] is True
    batch = collate_fn([ds[0], ds[1]])
    assert batch["seen_tuples"].shape == (2, 3, 2)
    assert torch.equal(batch["seen_tuples"][0, 2], torch.tensor([-1, -1]))
    assert batch["is_last"].dtype == torch.bool and batch["is_last"].all()


def test_max_len_cuts_the_history_but_not_seen():
    ds = SemanticIDSequenceDataset(_gts_data(), SEMANTIC_IDS, mode="test")
    batch = collate_fn([ds[i] for i in range(3)], max_len=1)
    assert batch["history_tuples"].shape[1] == 1
    assert torch.equal(batch["history_tuples"][2, 0], SEMANTIC_IDS[0])
    assert batch["seen_tuples"].shape == (3, 4, 2)
    assert torch.equal(batch["seen_tuples"][2], SEMANTIC_IDS[[2, 3, 4, 0]])
    assert torch.equal(batch["seen_tuples"][0, 1:], torch.full((3, 2), -1))
    assert torch.equal(batch["is_last"], torch.tensor([False, True, True]))
    assert torch.equal(batch["user_ids"], torch.tensor([3, 3, 4]))


def test_train_mode_has_no_eval_fields():
    data = {"train": {"userId": torch.tensor([0]), "itemId": [[0, 1]], "itemId_fut": torch.tensor([2])}}
    item = SemanticIDSequenceDataset(data, SEMANTIC_IDS, mode="train")[0]
    assert "seen_tuples" not in item and "is_last" not in item
    assert set(collate_fn([item])) == {"history_tuples", "target_tuples", "user_ids"}


def _batch(histories, mode="train"):
    n = len(histories)
    data = {mode: {"userId": torch.arange(n), "itemId": histories, "itemId_fut": torch.full((n,), 5)}}
    ds = SemanticIDSequenceDataset(data, SEMANTIC_IDS, mode=mode)
    return [ds[i] for i in range(n)]


def test_collate_keeps_the_last_max_len_items_and_pads_short_rows():
    out = collate_fn(_batch([[0, 1, 2, 3, 4], [1, 2]]), max_len=3)["history_tuples"]
    assert out.shape == (2, 3, 2)
    assert torch.equal(out[0], SEMANTIC_IDS[[2, 3, 4]])
    assert torch.equal(out[1, :2], SEMANTIC_IDS[[1, 2]])
    assert torch.equal(out[1, 2], torch.tensor([-1, -1]))


def test_collate_without_max_len_keeps_the_full_history():
    out = collate_fn(_batch([[0, 1, 2, 3, 4], [1, 2]]))["history_tuples"]
    assert out.shape == (2, 5, 2)


def test_accepts_tensor_rows():
    ds = SemanticIDSequenceDataset(
        {"train": {"userId": torch.tensor([0]), "itemId": [torch.tensor([0, 1])],
                   "itemId_fut": torch.tensor([2])}}, SEMANTIC_IDS, mode="train")
    assert torch.equal(ds[0]["history_tuples"], SEMANTIC_IDS[[0, 1]])
