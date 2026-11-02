"""
Unit tests for data/protocols.py: SplitConfig, the leave-one-out and the global temporal
split (GTS) builders and the SASRec protocol guard.

GTS fixture: with q_test = q_val = 0.75 the cutoffs are day 9 (t_val) and day 16 (t_test);
item ids go through a non-trivial permutation.

Users (P5 item @ day):
  1: 1@1 2@2 3@3 4@4 5@5                                  only training
  2: 1@2 2@3 3@6 6@8 7@9                                  only training
  3: 1@1 2@2 8@3 4@6 | 5@12 6@13 | 7@22 9@23 3@24         all three windows
  4: 5@12 6@14 7@16 | 4@25 9@26                           starts in validation
  5: 1@30 2@31 9@32 10@33 3@34                            starts in test
  6: 1@2 2@4 3@5 4@9 | 5@15                               one validation target
  7: 1@3 2@4 3@5 4@6 | 10@14                              validation item is cold
  8: 9@1 9@2 9@3 9@4 (4 items: dropped)
Warm items (interaction <= day 9): 1-8; cold: 9, 10.
"""
import pytest
import torch
from omegaconf import OmegaConf

from data.protocols import (SplitConfig, build_gts_split, build_loo_split, check_sasrec_protocol,
                            get_split_config)

DAY = 86_400
PERM = torch.tensor([3, 7, 0, 9, 2, 5, 8, 1, 4, 6])  # P5 item p -> PERM[p - 1]

USERS = {
    1: [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
    2: [(1, 2), (2, 3), (3, 6), (6, 8), (7, 9)],
    3: [(1, 1), (2, 2), (8, 3), (4, 6), (5, 12), (6, 13), (7, 22), (9, 23), (3, 24)],
    4: [(5, 12), (6, 14), (7, 16), (4, 25), (9, 26)],
    5: [(1, 30), (2, 31), (9, 32), (10, 33), (3, 34)],
    6: [(1, 2), (2, 4), (3, 5), (4, 9), (5, 15)],
    7: [(1, 3), (2, 4), (3, 5), (4, 6), (10, 14)],
    8: [(9, 1), (9, 2), (9, 3), (9, 4)],
}


def m(p):
    return PERM[p - 1].item()


def sequences():
    return [(u, [m(p) for p, _ in inter], [d * DAY for _, d in inter]) for u, inter in USERS.items()]


@pytest.fixture(scope="module")
def split():
    return build_gts_split(sequences(), num_items=10, q_test=0.75, q_val=0.75)


def rows(split, name):
    """Rows of a split as (user, history, target, is_last) tuples."""
    h = split["history"][name]
    return [(h["userId"][i].item(), h["itemId"][i], h["itemId_fut"][i].item(),
             h["is_last"][i].item() if "is_last" in h else None) for i in range(len(h["userId"]))]


class TestSplitConfig:
    def test_defaults(self):
        assert get_split_config(OmegaConf.create({"data": {}})) == SplitConfig("loo")
        assert get_split_config(OmegaConf.create({"data": {"split_protocol": "gts"}})) == \
            SplitConfig("gts", 0.9, 0.9)

    def test_gts_values_are_read(self):
        cfg = OmegaConf.create({"data": {"split_protocol": "gts", "gts_q_test": 0.8, "gts_q_val": 0.7}})
        assert get_split_config(cfg) == SplitConfig("gts", 0.8, 0.7)

    def test_q_values_ignored_under_loo(self):
        cfg = OmegaConf.create({"data": {"split_protocol": "loo", "gts_q_test": 0.5}})
        assert get_split_config(cfg) == SplitConfig("loo")

    def test_split_names(self):
        assert SplitConfig("loo").split_names == ("train", "eval", "test")
        assert SplitConfig("gts", 0.9, 0.9).split_names == ("train", "eval", "test", "test_cold")

    @pytest.mark.parametrize("data", [
        {"split_protocol": "random"},
        {"split_protocol": "gts", "gts_q_test": 1.0},
        {"split_protocol": "gts", "gts_q_val": 0.0},
        {"split_protocol": "gts", "gts_q_val": -0.1},
    ])
    def test_invalid_values_raise(self, data):
        with pytest.raises(ValueError):
            get_split_config(OmegaConf.create({"data": data}))

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            SplitConfig("loo").protocol = "gts"


class TestLeaveOneOut:
    def test_targets_and_histories(self):
        split = build_loo_split([(7, [10, 11, 12, 13, 14, 15], None)])
        h = split["history"]
        assert h["train"]["itemId"] == [[10, 11, 12]] and h["train"]["itemId_fut"].tolist() == [13]
        assert h["eval"]["itemId"] == [[10, 11, 12, 13]] and h["eval"]["itemId_fut"].tolist() == [14]
        assert h["test"]["itemId"] == [[10, 11, 12, 13, 14]] and h["test"]["itemId_fut"].tolist() == [15]
        for rows_ in h.values():
            assert rows_["userId"].tolist() == [7]
            assert "is_last" not in rows_

    def test_histories_are_not_truncated_or_padded(self):
        items = list(range(40))
        h = build_loo_split([(1, items, None)])["history"]
        assert h["test"]["itemId"][0] == items[:-1]
        assert h["eval"]["itemId"][0] == items[:-2]

    def test_users_with_fewer_than_five_items_are_dropped(self):
        seqs = [(1, [0, 1, 2, 3], None), (2, [0, 1, 2, 3, 4], None), (3, [0, 1, 2], None)]
        split = build_loo_split(seqs)
        assert split["history"]["train"]["userId"].tolist() == [2]
        assert split["history"]["train"]["itemId"] == [[0, 1]]
        assert split["stats"] == {"users": 1, "interactions": 5,
                                  "train_rows": 1, "eval_rows": 1, "test_rows": 1}

    def test_user_order_is_input_order(self):
        seqs = [(u, list(range(6)), None) for u in (5, 2, 9)]
        assert build_loo_split(seqs)["history"]["test"]["userId"].tolist() == [5, 2, 9]

    def test_formats(self):
        h = build_loo_split([(1, list(range(6)), None)])["history"]
        for rows_ in h.values():
            assert rows_["userId"].dtype == torch.long and rows_["itemId_fut"].dtype == torch.long
            assert isinstance(rows_["itemId"], list)


class TestGtsCutoffsAndWarmItems:
    def test_cutoffs(self, split):
        assert split["protocol"] == "gts"
        assert split["t_val"] == 9 * DAY
        assert split["t_test"] == 16 * DAY
        assert type(split["t_val"]) is int and type(split["t_test"]) is int

    def test_validation_cutoff_includes_the_test_cutoff(self):
        # t_val is the q_val quantile of the times up to and including t_test:
        # times 1..10, q 0.5 -> t_test = time[4] = 5; times <= 5 -> t_val = time[2] = 3
        times = [d * DAY for d in range(1, 11)]
        split = build_gts_split([(1, list(range(10)), times)], num_items=10, q_test=0.5, q_val=0.5)
        assert (split["t_test"], split["t_val"]) == (5 * DAY, 3 * DAY)

    def test_warm_and_cold_items(self, split):
        warm = {m(p) for p in range(1, 9)}
        assert set(torch.nonzero(split["stage1_items"]).flatten().tolist()) == warm
        assert split["stage1_items"].shape == (10,)

    def test_short_users_are_dropped(self, split):
        # user 8 would make item 9 warm (day <= 9) if it were not dropped
        assert not split["stage1_items"][m(9)]
        assert split["stats"]["users"] == 7
        assert 8 not in split["history"]["train"]["userId"].tolist()

    def test_missing_times_raise(self):
        with pytest.raises(ValueError, match="times"):
            build_gts_split([(1, list(range(6)), None)], num_items=6, q_test=0.9, q_val=0.9)


class TestGtsTrain:
    def test_rows(self, split):
        got = {u: (h, t) for u, h, t, _ in rows(split, "train")}
        assert got == {
            1: ([m(1), m(2), m(3), m(4)], m(5)),
            2: ([m(1), m(2), m(3), m(6)], m(7)),
            3: ([m(1), m(2), m(8)], m(4)),
            6: ([m(1), m(2), m(3)], m(4)),
            7: ([m(1), m(2), m(3)], m(4)),
        }

    def test_excludes_everything_after_t_val(self, split):
        late = {u: {m(p) for p, d in inter if d > 9} for u, inter in USERS.items()}
        for u, h, t, _ in rows(split, "train"):
            assert t not in late[u] and not set(h) & late[u]

    def test_users_without_interaction_up_to_t_val_have_no_row(self, split):
        users = split["history"]["train"]["userId"].tolist()
        assert 4 not in users and 5 not in users

    def test_user_with_one_interaction_up_to_t_val_has_no_row(self):
        # a train row needs an input and a target, so one interaction up to t_val is not enough.
        # user 1: days 1..10, user 2: days 1, 20..23; q 0.5 -> t_test = day 7, t_val = day 3,
        # so user 2 has only day 1 up to t_val
        sequences = [(1, list(range(10)), [d * DAY for d in range(1, 11)]),
                     (2, [0, 10, 11, 12, 13], [d * DAY for d in (1, 20, 21, 22, 23)])]
        split = build_gts_split(sequences, num_items=14, q_test=0.5, q_val=0.5)
        assert (split["t_test"], split["t_val"]) == (7 * DAY, 3 * DAY)
        assert split["history"]["train"]["userId"].tolist() == [1]


class TestGtsEval:
    def test_rows(self, split):
        got = {u: (h, t) for u, h, t, _ in rows(split, "eval")}
        assert got == {
            # last warm target in (day 9, 16]; the history is complete
            3: ([m(1), m(2), m(8), m(4), m(5)], m(6)),
            # starts in the validation period
            4: ([m(5), m(6)], m(7)),
            6: ([m(1), m(2), m(3), m(4)], m(5)),
        }

    def test_cold_only_window_gives_no_row(self, split):
        assert 7 not in split["history"]["eval"]["userId"].tolist()

    def test_one_row_per_user(self, split):
        users = split["history"]["eval"]["userId"].tolist()
        assert len(users) == len(set(users))

    def test_no_is_last(self, split):
        assert "is_last" not in split["history"]["eval"]


class TestGtsTest:
    def test_warm_rows(self, split):
        assert rows(split, "test") == [
            (3, [m(1), m(2), m(8), m(4), m(5), m(6)], m(7), False),
            (3, [m(1), m(2), m(8), m(4), m(5), m(6), m(7), m(9)], m(3), True),
            (4, [m(5), m(6), m(7)], m(4), True),
            # the first interaction (1) is input only
            (5, [m(1)], m(2), False),
            (5, [m(1), m(2), m(9), m(10)], m(3), True),
        ]

    def test_cold_rows(self, split):
        assert rows(split, "test_cold") == [
            (3, [m(1), m(2), m(8), m(4), m(5), m(6), m(7)], m(9), True),
            (4, [m(5), m(6), m(7), m(4)], m(9), True),
            (5, [m(1), m(2)], m(9), False),
            (5, [m(1), m(2), m(9)], m(10), True),
        ]

    def test_targets_are_warm_or_cold(self, split):
        warm = split["stage1_items"]
        assert warm[split["history"]["test"]["itemId_fut"]].all()
        assert not warm[split["history"]["test_cold"]["itemId_fut"]].any()

    def test_one_is_last_per_user(self, split):
        for name in ("test", "test_cold"):
            h = split["history"][name]
            users = h["userId"][h["is_last"]].tolist()
            assert sorted(users) == sorted(set(h["userId"].tolist()))


class TestGtsInvariants:
    @pytest.mark.parametrize("name", ["eval", "test", "test_cold"])
    def test_history_is_everything_before_the_target(self, split, name):
        full = {u: [m(p) for p, _ in inter] for u, inter in USERS.items()}
        for u, hist, target, _ in rows(split, name):
            items = full[u]
            assert hist == items[:items.index(target)]

    def test_formats(self, split):
        for name in ("train", "eval", "test", "test_cold"):
            h = split["history"][name]
            assert h["userId"].dtype == torch.long and h["itemId_fut"].dtype == torch.long
            assert isinstance(h["itemId"], list)
        assert split["history"]["test"]["is_last"].dtype == torch.bool

    def test_stats(self, split):
        s = split["stats"]
        assert (s["users"], s["interactions"]) == (7, 39)
        assert (s["train_rows"], s["eval_rows"], s["test_rows"], s["test_cold_rows"]) == (5, 3, 5, 4)
        assert (s["test_users"], s["test_cold_users"], s["stage1_items"]) == (3, 3, 8)


def cfg(**data):
    return OmegaConf.create({"data": data})


GTS = dict(split_protocol="gts", gts_q_test=0.9, gts_q_val=0.9)


class TestSasrecGuard:
    def test_accepts_loo_and_default(self):
        check_sasrec_protocol(cfg(split_protocol="loo"))
        check_sasrec_protocol(cfg(category="toys"))

    def test_rejects_gts(self):
        with pytest.raises(NotImplementedError, match="leave-one-out"):
            check_sasrec_protocol(cfg(**GTS))
