"""
Reader for the Amazon review data in the P5 file format (Amazon Beauty / Sports / Toys).

Specific to this format: sequential_data.txt holds one line per user ("user_id item_id ...")
in chronological order, but without timestamps; review_splits.pkl has the review times
(whole days) and datamaps.json maps the P5 ids to user and item names.
"""
import json
import os
import pickle


def read_amazon_interactions(raw_dir, id_permutation, with_times=True):
    """
    Read the interactions of every user in the order of the file.

    Args:
        raw_dir: Folder with sequential_data.txt, datamaps.json and review_splits.pkl.
        id_permutation: Tensor mapping the 0-based P5 item id to the model item id.
        with_times: Look up the review time of every interaction (needs datamaps.json and
            review_splits.pkl). If False, times is None.

    The file order is the chronological order; items of the same day stay in file order.

    Returns:
        list of (P5 user id, model item ids, unix times or None) per user. Raises a
        ValueError if the times of a user are not non-decreasing.
    """
    perm = id_permutation.tolist()

    time = id2user = id2item = None
    if with_times:
        with open(os.path.join(raw_dir, "review_splits.pkl"), "rb") as f:
            reviews = pickle.load(f)
        with open(os.path.join(raw_dir, "datamaps.json")) as f:
            maps = json.load(f)
        # every review appears in exactly one of the three splits
        time = {(r["reviewerID"], r["asin"]): r["unixReviewTime"]
                for sp in ("train", "val", "test") for r in reviews[sp]}
        id2user, id2item = maps["id2user"], maps["id2item"]

    sequences = []
    with open(os.path.join(raw_dir, "sequential_data.txt")) as f:
        for line in f:
            user, *p5_items = line.split()
            items = [perm[int(i) - 1] for i in p5_items]
            times = None
            if with_times:
                times = [time[(id2user[user], id2item[i])] for i in p5_items]
                if any(a > b for a, b in zip(times, times[1:])):
                    raise ValueError(f"User {user}: sequential_data.txt is not chronological")
            sequences.append((int(user), items, times))
    return sequences
