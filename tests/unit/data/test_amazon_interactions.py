"""
Unit tests for data/amazon_interactions.py on a tiny synthetic P5 folder
(sequential_data.txt, datamaps.json, review_splits.pkl).
"""
import json
import pickle

import pytest
import torch

from data.amazon_interactions import read_amazon_interactions

DAY = 86_400
PERM = torch.tensor([3, 7, 0, 9, 2, 5, 8, 1, 4, 6])  # P5 item p -> PERM[p - 1]

# user -> [(P5 item, day)] in file order
USERS = {
    1: [(1, 1), (2, 2), (3, 2), (4, 5), (5, 9)],
    2: [(6, 3), (1, 4), (2, 7)],
}


def write_raw(folder, users):
    with open(folder / "sequential_data.txt", "w") as f:
        for u, inter in users.items():
            f.write(" ".join([str(u)] + [str(p) for p, _ in inter]) + "\n")
    maps = {"id2user": {str(u): f"R{u}" for u in users},
            "id2item": {str(p): f"A{p}" for p in range(1, 11)}}
    json.dump(maps, open(folder / "datamaps.json", "w"))
    reviews = [{"reviewerID": f"R{u}", "asin": f"A{p}", "unixReviewTime": day * DAY}
               for u, inter in users.items() for p, day in inter]
    # every review in exactly one of train/val/test
    pickle.dump({"train": reviews[0::3], "val": reviews[1::3], "test": reviews[2::3]},
                open(folder / "review_splits.pkl", "wb"))


@pytest.fixture
def raw_dir(tmp_path):
    write_raw(tmp_path, USERS)
    return str(tmp_path)


def test_items_are_permuted_and_times_looked_up(raw_dir):
    got = read_amazon_interactions(raw_dir, PERM)
    assert got == [
        (1, [PERM[p - 1].item() for p in (1, 2, 3, 4, 5)], [d * DAY for d in (1, 2, 2, 5, 9)]),
        (2, [PERM[p - 1].item() for p in (6, 1, 2)], [d * DAY for d in (3, 4, 7)]),
    ]


def test_all_users_are_returned_in_file_order(raw_dir):
    assert [u for u, _, _ in read_amazon_interactions(raw_dir, PERM)] == [1, 2]


def test_without_times_only_sequential_data_is_needed(tmp_path):
    (tmp_path / "sequential_data.txt").write_text("4 1 2 3\n")
    assert read_amazon_interactions(str(tmp_path), PERM, with_times=False) == [
        (4, [PERM[0].item(), PERM[1].item(), PERM[2].item()], None)]


def test_non_chronological_file_raises(tmp_path):
    write_raw(tmp_path, {1: [(1, 5), (2, 3), (3, 6)]})
    with pytest.raises(ValueError, match="not chronological"):
        read_amazon_interactions(str(tmp_path), PERM)
