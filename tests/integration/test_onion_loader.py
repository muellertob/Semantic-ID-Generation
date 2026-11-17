"""
Integration tests for the Music4All-Onion functions of data/loader.py on a tiny synthetic
dataset in a temporary working directory. All times are 10:00:00 on the given day of
August 2019; the subset is SMALL (k = 2).

    user  events (track, day)         track  feature
    1     A 1, B 2, A 3, B 4, C 20    A      [1, 2]
    2     B 1, A 2, B 3, A 4, C 21    B      [3, 4]
                                      C      [5, 6]

Item ids A = 0, B = 1, C = 2; the 2-core keeps everything. GTS with q 0.75 / 0.75 on the
10 sorted times: t_test = day 4, t_val = day 3. Warm items (an event up to day 3): A, B;
cold: C. Train rows: user 1 history [0, 1] -> target 0, user 2 history [1, 0] -> target 1.
"""
import bz2
import hashlib
import os
import shutil

import pytest
import torch

from data.loader import (ONION_FORMAT_VERSION, ONION_SUBSET, load_onion, load_onion_sequences,
                         load_onion_split, onion_data_path)
from data.protocols import SplitConfig

SMALL = {**ONION_SUBSET, "k": 2}
GTS = SplitConfig("gts", 0.75, 0.75)
RAW = "dataset/music4all_onion/raw"
PROCESSED = "dataset/music4all_onion/processed"

# unix seconds (UTC) of 10:00:00 on day d of August 2019
D1, D2, D3, D4, D5, D20, D21 = (1564653600, 1564740000, 1564826400, 1564912800, 1564999200,
                                1566295200, 1566381600)
EVENTS = [(1, "A", 1), (1, "B", 2), (1, "A", 3), (1, "B", 4), (1, "C", 20),
          (2, "B", 1), (2, "A", 2), (2, "B", 3), (2, "A", 4), (2, "C", 21)]
EXPECTED_SEQUENCES = [(1, [0, 1, 0, 1, 2], [D1, D2, D3, D4, D20]),
                      (2, [1, 0, 1, 0, 2], [D1, D2, D3, D4, D21])]
MD5 = hashlib.md5(repr(EXPECTED_SEQUENCES).encode()).hexdigest()


def write_events(events):
    lines = ["user_id\ttrack_id\ttimestamp"] + [
        f"{user}\t{track}\t2019-08-{day:02d} 10:00:00" for user, track, day in events]
    with open(f"{RAW}/userid_trackid_timestamp.tsv.bz2", "wb") as f:
        f.write(bz2.compress("".join(line + "\r\n" for line in lines).encode()))


@pytest.fixture
def onion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(RAW)
    write_events(EVENTS)
    lines = ["id\t0\t1", "A\t1\t2", "B\t3\t4", "C\t5\t6"]
    with open(f"{RAW}/id_maest.tsv.bz2", "wb") as f:
        f.write(bz2.compress("".join(line + "\n" for line in lines).encode()))


def test_subset_constant_holds_the_decided_window_core_and_feature():
    # a changed value would silently give another dataset
    assert ONION_SUBSET == {"start": "2019-08-01 00:00:00", "end": "2019-09-30 23:59:59",
                            "k": 5, "features": "id_maest.tsv.bz2"}


def test_data_file_name_encodes_window_k_and_feature():
    # another subset must never read this file
    assert onion_data_path(ONION_SUBSET) == (
        "dataset/music4all_onion/processed/data_onion_2019-08-01_2019-09-30_k5_maest.pt")


def test_first_call_builds_the_data_file_from_the_raw_files(onion):
    # everything a later run needs is in this one file
    load_onion_split(GTS, subset=SMALL)
    data = torch.load(onion_data_path(SMALL), weights_only=False)
    x = data.pop("x")  # a dict holding a tensor cannot be compared with ==
    assert data == {"track_ids": ["A", "B", "C"], "sequences": EXPECTED_SEQUENCES,
                    "sequences_md5": MD5, "meta": {"format_version": ONION_FORMAT_VERSION, **SMALL}}
    assert torch.equal(x, torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32))


def test_later_calls_do_not_read_the_raw_files(onion):
    # the raw event file is large and not on the training server
    first = load_onion_split(GTS, subset=SMALL)
    shutil.rmtree(RAW)
    second = load_onion_split(GTS, subset=SMALL)
    assert (second["stats"], second["history"]["train"]["userId"].tolist(),
            second["history"]["train"]["itemId"], second["history"]["train"]["itemId_fut"].tolist()
            ) == (first["stats"], [1, 2], first["history"]["train"]["itemId"], [0, 1])


def test_data_file_with_another_format_version_raises(onion):
    # a file built under older reader rules must not be used
    load_onion_split(GTS, subset=SMALL)
    data = torch.load(onion_data_path(SMALL), weights_only=False)
    data["meta"]["format_version"] = 0
    torch.save(data, onion_data_path(SMALL))
    with pytest.raises(ValueError, match="does not match"):
        load_onion_split(GTS, subset=SMALL)


def test_leave_one_out_protocol_raises(onion):
    # Onion has no 95/5 item split, only the global temporal split is defined
    with pytest.raises(ValueError, match="only the global temporal split"):
        load_onion_split(SplitConfig("loo"), subset=SMALL)


def test_split_file_name_carries_the_sequence_md5(onion):
    # ties the cached split to the sequences it was built from
    load_onion_split(GTS, subset=SMALL)
    assert sorted(os.listdir(PROCESSED)) == [
        "data_onion_2019-08-01_2019-09-30_k2_maest.pt",
        f"split_onion_2019-08-01_2019-09-30_k2_{MD5[:8]}_gts_q0.75-0.75.pt"]


def test_changed_sequences_never_reuse_the_old_split(onion):
    # item ids and item count stay the same, so the split meta alone would accept the old split
    first = load_onion_split(GTS, subset=SMALL)
    os.remove(onion_data_path(SMALL))
    write_events(EVENTS + [(1, "A", 5)])
    second = load_onion_split(GTS, subset=SMALL)
    changed = [(1, [0, 1, 0, 1, 0, 2], [D1, D2, D3, D4, D5, D20]), EXPECTED_SEQUENCES[1]]
    changed_md5 = hashlib.md5(repr(changed).encode()).hexdigest()
    assert (first["stats"]["interactions"], second["stats"]["interactions"]) == (10, 11)
    assert sorted(os.listdir(PROCESSED)) == sorted([
        "data_onion_2019-08-01_2019-09-30_k2_maest.pt",
        f"split_onion_2019-08-01_2019-09-30_k2_{MD5[:8]}_gts_q0.75-0.75.pt",
        f"split_onion_2019-08-01_2019-09-30_k2_{changed_md5[:8]}_gts_q0.75-0.75.pt"])


def test_train_returns_the_features_of_the_warm_items(onion):
    # Stage 1 is fitted on the warm items only; C is first heard after t_val
    x = load_onion(GTS, "none", "train", subset=SMALL)
    assert torch.equal(x, torch.tensor([[1, 2], [3, 4]], dtype=torch.float32))


def test_test_returns_the_features_of_the_cold_items(onion):
    # the complement of the warm items
    x = load_onion(GTS, "none", "test", subset=SMALL)
    assert torch.equal(x, torch.tensor([[5, 6]], dtype=torch.float32))


def test_all_returns_every_item_in_item_id_order(onion):
    # SID generation needs a code for every item, row i for item i
    x = load_onion(GTS, "none", "all", subset=SMALL)
    assert torch.equal(x, torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32))


def test_normalization_uses_the_statistics_of_the_warm_items(onion):
    # warm mean [2, 3]; no statistics from items first heard after t_val (all three: [3, 4])
    x = load_onion(GTS, "center", "all", subset=SMALL)
    assert torch.equal(x, torch.tensor([[-1, -1], [1, 1], [3, 3]], dtype=torch.float32))


def test_sequences_return_the_gts_train_rows_with_user_and_item_counts(onion):
    # the Stage-2 scripts read exactly this tuple, as for Amazon
    sequences, num_users, num_items = load_onion_sequences(GTS, subset=SMALL)
    train = sequences["train"]
    assert (num_users, num_items, train["userId"].tolist(), train["itemId"],
            train["itemId_fut"].tolist()) == (2, 3, [1, 2], [[0, 1], [1, 0]], [0, 1])
