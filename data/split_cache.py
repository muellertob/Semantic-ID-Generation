"""
Cache of the Stage-2 split of a dataset (data/protocols.py) on disk.

One file per (dataset, protocol, cutoff quantiles). The file stores the split together with the
parameters and inputs it depends on; a file that does not match the current code, data or
arguments raises an error instead of being used. A dataset only needs a reader that yields
its (user_id, items, times) sequences and a cache folder.
"""
import hashlib
import logging
import os

import torch

from data.protocols import SplitConfig, build_gts_split, build_loo_split

logger = logging.getLogger(__name__)

# increase when the split rules or the stored layout change
FORMAT_VERSION = 2


def split_cache_path(processed_dir: str, name: str, split_config: SplitConfig) -> str:
    """split_<name>_<protocol>[_q<q_test>-<q_val>].pt in processed_dir."""
    suffix = f"_q{split_config.q_test}-{split_config.q_val}" if split_config.protocol == "gts" else ""
    return os.path.join(processed_dir, f"split_{name}_{split_config.protocol}{suffix}.pt")


def split_meta(id_permutation, num_items, split_config: SplitConfig) -> dict:
    """Parameters and inputs a split depends on, stored with it and checked on load."""
    perm = id_permutation.detach().cpu().contiguous()
    return {
        "format_version": FORMAT_VERSION,
        "protocol": split_config.protocol,
        "q_test": split_config.q_test,
        "q_val": split_config.q_val,
        "num_items": num_items,
        "perm_md5": hashlib.md5(perm.numpy().tobytes()).hexdigest(),
    }


def check_split_meta(split, expected, path):
    """
    Raise a ValueError naming every field of the split's meta that differs from the
    expected meta. A split without meta counts as stale.
    """
    meta = split.get("meta")
    if meta is None:
        raise ValueError(f"Split {path} has no meta (stale cache). "
                         f"Delete the file so it is rebuilt.")
    mismatches = [f"{k}: file has {meta.get(k)!r}, expected {v!r}"
                  for k, v in expected.items() if meta.get(k) != v]
    if mismatches:
        raise ValueError(f"Split {path} does not match the current code or data "
                         f"({'; '.join(mismatches)}). Delete the file so it is rebuilt.")


def save_atomic(obj, path):
    """torch.save to a temporary file in the target folder, then replace the target."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_or_build_split(name, split_config, processed_dir, read_sequences, id_permutation, num_items) -> dict:
    """
    Load the cached split of a dataset, building and caching it if missing.

    Args:
        name: Dataset name (part of the file name).
        split_config: SplitConfig of the protocol.
        processed_dir: Cache folder.
        read_sequences: Callable returning the (user_id, items, times) sequences with model
            item ids; only called when the split has to be built.
        id_permutation: Tensor mapping the dataset's item ids to the model item ids.
        num_items: Number of items (size of the item embedding matrix).

    Returns:
        dict of build_loo_split / build_gts_split plus meta (see split_meta). A cached file
        whose meta differs from the current code version, data or arguments raises a
        ValueError instead of being used.
    """
    path = split_cache_path(processed_dir, name, split_config)
    meta = split_meta(id_permutation, num_items, split_config)

    if os.path.exists(path):
        split = torch.load(path, weights_only=False)
        check_split_meta(split, meta, path)
        return split

    sequences = read_sequences()
    if split_config.protocol == "gts":
        split = build_gts_split(sequences, num_items, split_config.q_test, split_config.q_val)
    else:
        split = build_loo_split(sequences)
    split["meta"] = meta
    save_atomic(split, path)
    logger.info(f"Built {split_config.describe()} split for {name}; stats: {split['stats']}")
    return split
