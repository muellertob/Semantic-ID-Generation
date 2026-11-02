"""
Evaluation protocols: how the interactions of all users are split into the Stage-2
(recommender) train / eval / test rows. The functions are pure and independent of any
dataset; data/split_cache.py stores their result.

Row layout of every split (all protocols):
    userId       LongTensor (N)
    itemId       list of N lists: the user's full chronological history before the target,
                 unpadded and untruncated (also the set of items the user has seen; the
                 collate function in data/sequence.py cuts the model input)
    itemId_fut   LongTensor (N): the target item
    is_last      BoolTensor (N), only in splits with several rows per user (gts 'test',
                 'test_cold'): marks the user's last row
"""
from dataclasses import dataclass

import torch

PROTOCOLS = ("loo", "gts")
LOO_SPLITS = ("train", "eval", "test")
GTS_SPLITS = ("train", "eval", "test", "test_cold")
MIN_INTERACTIONS = 5


@dataclass(frozen=True)
class SplitConfig:
    """
    Evaluation protocol of a run. q_test / q_val are the cutoff quantiles of 'gts' and
    None under 'loo'.
    """
    protocol: str
    q_test: float | None = None
    q_val: float | None = None

    def __post_init__(self):
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"Invalid split protocol: {self.protocol!r}. Must be one of {PROTOCOLS}.")
        if self.protocol == "loo":
            if self.q_test is not None or self.q_val is not None:
                raise ValueError("The leave-one-out protocol has no cutoff quantiles.")
            return
        for name, q in (("q_test", self.q_test), ("q_val", self.q_val)):
            if q is None or not 0 < q < 1:
                raise ValueError(f"Invalid {name}: {q!r}. Must be in (0, 1) under 'gts'.")

    @property
    def split_names(self) -> tuple[str, ...]:
        return GTS_SPLITS if self.protocol == "gts" else LOO_SPLITS

    def describe(self) -> str:
        if self.protocol == "gts":
            return f"'gts' (q_test={self.q_test}, q_val={self.q_val})"
        return repr(self.protocol)


def get_split_config(config) -> SplitConfig:
    """
    The evaluation protocol of a config: data.split_protocol ('loo' by default) and, under
    'gts', the cutoff quantiles data.gts_q_test / data.gts_q_val (0.9 by default). The only
    place that reads these keys.
    """
    data_config = config.data
    protocol = data_config.get("split_protocol", "loo")
    if protocol == "gts":
        return SplitConfig(protocol,
                           data_config.get("gts_q_test", 0.9),
                           data_config.get("gts_q_val", 0.9))
    return SplitConfig(protocol)


def check_sasrec_protocol(config):
    """Raise a NotImplementedError unless the config uses the leave-one-out split."""
    split_config = get_split_config(config)
    if split_config.protocol != "loo":
        raise NotImplementedError(
            f"SASRec supports only the leave-one-out split protocol, got "
            f"data.split_protocol={split_config.protocol!r}.")


def _pack(cols: dict) -> dict:
    """Column lists of one split to its row layout."""
    rows = {
        "userId": torch.tensor(cols["userId"], dtype=torch.long),
        "itemId": cols["itemId"],
        "itemId_fut": torch.tensor(cols["itemId_fut"], dtype=torch.long),
    }
    if "is_last" in cols:
        rows["is_last"] = torch.tensor(cols["is_last"], dtype=torch.bool)
    return rows


def build_loo_split(sequences) -> dict:
    """
    Build the leave-one-out split: per user, 'train' predicts the third-last item from the
    items before it, 'eval' the second-last and 'test' the last item.

    Args:
        sequences: list of (user_id, items, times) with model item ids in chronological
            order; times is not used and may be None. Users with fewer than
            MIN_INTERACTIONS items are dropped.

    Returns:
        dict with protocol, history (train, eval, test in the row layout of the module
        docstring, one row per user in input order) and stats.
    """
    cols = {sp: {"userId": [], "itemId": [], "itemId_fut": []} for sp in LOO_SPLITS}
    n_users = n_interactions = 0
    for user, items, _ in sequences:
        if len(items) < MIN_INTERACTIONS:
            continue
        n_users += 1
        n_interactions += len(items)
        for sp, target in (("train", -3), ("eval", -2), ("test", -1)):
            cols[sp]["userId"].append(user)
            cols[sp]["itemId"].append(items[:target])
            cols[sp]["itemId_fut"].append(items[target])

    stats = {"users": n_users, "interactions": n_interactions}
    stats.update({f"{sp}_rows": n_users for sp in LOO_SPLITS})
    return {
        "protocol": "loo",
        "history": {sp: _pack(c) for sp, c in cols.items()},
        "stats": stats,
    }


def _target_rows(items, times, is_target_item, lo, hi):
    """
    Candidate target positions of one user in the window lo < t <= hi.

    A position is a candidate if its time is in the window, its item passes
    is_target_item and at least one interaction precedes it (the model needs an
    input), so a user's first interaction is never a target.

    Returns:
        list of (history, target item) in time order; the history is all items before the target.
    """
    rows = []
    # range starts from 1 to ensure history of at least one item
    for k in range(1, len(items)):
        in_window = lo < times[k] <= hi
        if in_window and is_target_item(items[k]):
            history = items[:k]
            rows.append((history, items[k]))
    return rows


def build_gts_split(sequences, num_items, q_test, q_val) -> dict:
    """
    Build the global temporal split.

    The cutoffs are global: t_test is the q_test quantile of all interaction times, t_val
    the q_val quantile of the times up to t_test (lower quantiles, so both are actual
    interaction times). Items with at least one interaction up to t_val are warm; Stage 1
    (the SID quantizer) is fitted on the warm items only.

    Splits:
        train      one row per user: the interactions up to t_val, the last one as target
        eval       one row per user: the last warm target in (t_val, t_test]
        test       one row per warm target after t_test, with is_last
        test_cold  one row per cold target after t_test, with is_last
    A target needs at least one earlier interaction as input. A user with only one
    interaction up to t_val therefore gets no train row; an item seen only by such users is
    warm (part of Stage 1 and of the test targets) but absent from the train rows.

    Args:
        sequences: list of (user_id, items, times) with model item ids and unix times in
            chronological order. Users with fewer than MIN_INTERACTIONS items are dropped.
        num_items: Number of items (size of the item embedding matrix).
        q_test, q_val: Quantiles of the sorted interaction times for the cutoffs.

    Returns:
        dict with protocol, t_val, t_test (unix seconds), stage1_items (BoolTensor over
        items), history (train, eval, test, test_cold in the row layout of the module
        docstring) and stats.
    """
    users = [(user, items, ts) for user, items, ts in sequences if len(items) >= MIN_INTERACTIONS]
    if any(ts is None for _, _, ts in users):
        raise ValueError("The global temporal split needs interaction times")

    # all interactions of all users as two flat tensors
    flat_items = torch.tensor([i for _, items, _ in users for i in items], dtype=torch.long)
    flat_times = torch.tensor([t for _, _, ts in users for t in ts], dtype=torch.long)

    all_times = flat_times.sort().values
    n = all_times.numel()
    t_test = int(all_times[int(q_test * (n - 1))]) # int rounds down, lower quantile method
    pre = all_times[all_times <= t_test]
    t_val = int(pre[int(q_val * (pre.numel() - 1))])

    # warm items: at least one interaction up to t_val
    stage1 = torch.zeros(num_items, dtype=torch.bool)
    stage1[flat_items[flat_times <= t_val]] = True
    is_warm = stage1.tolist()

    cols = {sp: {"userId": [], "itemId": [], "itemId_fut": []} for sp in GTS_SPLITS}
    for sp in ("test", "test_cold"):
        cols[sp]["is_last"] = []

    for user, items, ts in users:
        # train: everything up to t_val, last item is the target
        seq = [i for i, t in zip(items, ts) if t <= t_val]
        if len(seq) >= 2:
            cols["train"]["userId"].append(user)
            cols["train"]["itemId"].append(seq[:-1])
            cols["train"]["itemId_fut"].append(seq[-1])

        # validation: one row per user, the last warm target in (t_val, t_test]
        rows = _target_rows(items, ts, lambda i: is_warm[i], t_val, t_test)
        if rows:
            history, target = rows[-1]
            cols["eval"]["userId"].append(user)
            cols["eval"]["itemId"].append(history)
            cols["eval"]["itemId_fut"].append(target)

        # test: one row per warm target after t_test; test_cold: same for cold items
        for sp, is_target in (("test", lambda i: is_warm[i]),
                              ("test_cold", lambda i: not is_warm[i])):
            rows = _target_rows(items, ts, is_target, t_test, float("inf"))
            for r, (history, target) in enumerate(rows):
                cols[sp]["userId"].append(user)
                cols[sp]["itemId"].append(history)
                cols[sp]["itemId_fut"].append(target)
                cols[sp]["is_last"].append(r == len(rows) - 1)

    stats = {
        "users": len(users),
        "interactions": n,
        "train_rows": len(cols["train"]["userId"]),
        "eval_rows": len(cols["eval"]["userId"]),
        "test_rows": len(cols["test"]["userId"]),
        "test_cold_rows": len(cols["test_cold"]["userId"]),
        "test_users": len(set(cols["test"]["userId"])),
        "test_cold_users": len(set(cols["test_cold"]["userId"])),
        "warm_targets": len(cols["test"]["userId"]),
        "cold_targets": len(cols["test_cold"]["userId"]),
        "stage1_items": int(stage1.sum()),
    }
    return {
        "protocol": "gts",
        "t_val": t_val,
        "t_test": t_test,
        "stage1_items": stage1,
        "history": {sp: _pack(c) for sp, c in cols.items()},
        "stats": stats,
    }
