"""
Integration tests for data/loader.py (load_amazon_sequences, load_amazon) on a synthetic
Amazon category in a temporary working directory, plus real-data smoke tests (slow).
"""
import json
import os
import pickle

import pytest
import torch

from data.loader import load_amazon, load_amazon_sequences, load_amazon_split
from data.protocols import SplitConfig

DAY = 86_400
PERM = torch.tensor([3, 7, 0, 9, 2, 5, 8, 1, 4, 6])  # P5 item p -> PERM[p - 1]
# (P5 item, day). Under GTS with q 0.75 / 0.75: t_val = day 4, t_test = day 21; warm items
# (an interaction up to day 4) are 1-5, cold items 6-8. Leave-one-out ignores the days.
USERS = {
    1: [(1, 1), (2, 2), (3, 3), (4, 10), (5, 50), (6, 51)],
    2: [(2, 1), (3, 2), (4, 3), (5, 4), (6, 20), (7, 21), (8, 22)],
    3: [(1, 1), (2, 2)],
}
LOO = SplitConfig("loo")
GTS = SplitConfig("gts", 0.75, 0.75)


@pytest.fixture
def category(tmp_path, monkeypatch):
    raw = tmp_path / "dataset/amazon/raw/synth"
    raw.mkdir(parents=True)
    with open(raw / "sequential_data.txt", "w") as f:
        for u, inter in USERS.items():
            f.write(" ".join([str(u)] + [str(p) for p, _ in inter]) + "\n")
    json.dump({"id2user": {str(u): f"R{u}" for u in USERS},
               "id2item": {str(p): f"A{p}" for p in range(1, 11)}}, open(raw / "datamaps.json", "w"))
    reviews = [{"reviewerID": f"R{u}", "asin": f"A{p}", "unixReviewTime": d * DAY}
               for u, inter in USERS.items() for p, d in inter]
    pickle.dump({"train": reviews[0::3], "val": reviews[1::3], "test": reviews[2::3]},
                open(raw / "review_splits.pkl", "wb"))

    processed = tmp_path / "dataset/amazon/processed"
    processed.mkdir()
    x = torch.arange(10, dtype=torch.float).unsqueeze(1) * torch.ones(1, 3)
    # an older cache also holds the (padded) history, which is not read
    data = {"item": {"id_permutation": PERM, "x": x, "is_train": torch.arange(10) < 8},
            ("user", "rated", "item"): {"history": "unused"}}
    torch.save((data, None, None), processed / "data_synth.pt")
    monkeypatch.chdir(tmp_path)
    return "synth"


def m(p):
    return PERM[p - 1].item()


def test_loo_sequences(category):
    sequences, num_users, num_items = load_amazon_sequences(category, LOO)
    assert (num_users, num_items) == (2, 10)  # user 3 has fewer than 5 items
    assert set(sequences) == {"train", "eval", "test"}
    assert sequences["train"]["userId"].tolist() == [1, 2]
    assert sequences["test"]["itemId"][1] == [m(p) for p in (2, 3, 4, 5, 6, 7)]
    assert sequences["test"]["itemId_fut"].tolist() == [m(6), m(8)]
    assert sequences["eval"]["itemId_fut"].tolist() == [m(5), m(7)]
    assert sequences["train"]["itemId"] == [[m(1), m(2), m(3)], [m(2), m(3), m(4), m(5)]]


def gts_rows(sequences, name):
    h = sequences[name]
    return [(u, items, t) + ((h["is_last"][i].item(),) if "is_last" in h else ())
            for i, (u, items, t) in enumerate(zip(h["userId"].tolist(), h["itemId"],
                                                  h["itemId_fut"].tolist()))]


def test_gts_sequences(category):
    sequences, num_users, num_items = load_amazon_sequences(category, GTS)
    assert (num_users, num_items) == (2, 10)  # user 3 has fewer than 5 items
    assert set(sequences) == {"train", "eval", "test", "test_cold"}
    assert gts_rows(sequences, "train") == [(1, [m(1), m(2)], m(3)), (2, [m(2), m(3), m(4)], m(5))]
    # item 4 is warm through user 2 (day 3); user 2 has only cold items in (day 4, day 21]
    assert gts_rows(sequences, "eval") == [(1, [m(1), m(2), m(3)], m(4))]
    assert gts_rows(sequences, "test") == [(1, [m(1), m(2), m(3), m(4)], m(5), True)]
    assert gts_rows(sequences, "test_cold") == [
        (1, [m(1), m(2), m(3), m(4), m(5)], m(6), True),
        (2, [m(2), m(3), m(4), m(5), m(6), m(7)], m(8), True),
    ]


def test_gts_cutoffs_come_from_the_review_times(category):
    split = load_amazon_split(category, GTS)
    assert (split["t_val"], split["t_test"]) == (4 * DAY, 21 * DAY)


def test_split_is_cached_per_protocol(category):
    load_amazon_sequences(category, LOO)
    load_amazon_sequences(category, GTS)
    names = sorted(f for f in os.listdir("dataset/amazon/processed") if f.startswith("split_"))
    assert names == ["split_synth_gts_q0.75-0.75.pt", "split_synth_loo.pt"]


def test_loo_items_use_is_train_mask(category):
    train = load_amazon(category, LOO, normalize_data=False, split="train")
    test = load_amazon(category, LOO, normalize_data=False, split="test")
    assert len(train) == 8 and len(test) == 2


def test_gts_items_use_stage1_mask(category):
    # column 0 of the fixture embeddings is the model item id
    warm = load_amazon(category, GTS, normalize_data=False, split="train")[:, 0].long().tolist()
    cold = load_amazon(category, GTS, normalize_data=False, split="test")[:, 0].long().tolist()
    assert sorted(warm) == sorted(m(p) for p in range(1, 6))
    assert sorted(cold) == sorted(set(range(10)) - set(warm))
    assert len(load_amazon(category, GTS, normalize_data=False, split="all")) == 10


def test_invalid_item_split_raises(category):
    with pytest.raises(ValueError, match="Invalid split"):
        load_amazon(category, LOO, split="val")


@pytest.mark.slow
@pytest.mark.parametrize("category", ["beauty", "toys"])
def test_real_data_loo_rows(category):
    if not os.path.exists(f"dataset/amazon/raw/{category}"):
        pytest.skip("Real dataset not found.")
    sequences, num_users, num_items = load_amazon_sequences(category, LOO)
    train, eval_, test = (sequences[s] for s in ("train", "eval", "test"))
    assert len(train["userId"]) == len(eval_["userId"]) == len(test["userId"]) > 0
    assert min(len(h) for h in train["itemId"]) >= 2
    for i in range(min(50, len(train["userId"]))):
        # the targets of the three splits are the last three items of the user
        assert test["itemId"][i][:-1] == eval_["itemId"][i]
        assert test["itemId"][i][-1] == eval_["itemId_fut"][i]
        assert eval_["itemId"][i][-1] == train["itemId_fut"][i]


@pytest.mark.slow
def test_real_data_gts_toys():
    """Regression test: the GTS split of Toys (q 0.9 / 0.9) used for the thesis results."""
    if not os.path.exists("dataset/amazon/raw/toys"):
        pytest.skip("Real dataset not found.")
    split = load_amazon_split("toys", SplitConfig("gts", 0.9, 0.9))
    # 2014-01-12 and 2014-03-20, 00:00 UTC
    assert (split["t_val"], split["t_test"]) == (1389484800, 1395273600)
    assert split["stats"] == {
        "users": 19412, "interactions": 167597, "train_rows": 17406, "eval_rows": 4948,
        "test_rows": 12988, "test_cold_rows": 2970, "test_users": 4814, "test_cold_users": 1484,
        "warm_targets": 12988, "cold_targets": 2970, "stage1_items": 11645,
    }
