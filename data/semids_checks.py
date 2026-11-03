"""
Checks that a Semantic ID file fits the recommender run it is used in.

A semids file (generate_semids.py, train_rqkmeans.py) stores the Stage-1 config it was
produced with (semids_data['config']). Stage 2 maps every item id of the dataset to a row
of semantic_ids, so SIDs of another dataset or category would be used silently whenever the
item ids happen to be in range, and a quantizer fitted under another split protocol was
fitted on another set of items. These checks stop such runs at startup.
"""
from data.protocols import get_split_config


def _stage1_config(semids_data):
    """The Stage-1 config stored in a semids file; raises a ValueError if it is missing."""
    stage1_config = semids_data.get("config")
    if stage1_config is None or "data" not in stage1_config:
        raise ValueError("The Semantic ID file stores no Stage-1 config, so it cannot be verified.")
    return stage1_config


def check_semids_dataset(semids_data, config, num_items):
    """
    Raise a ValueError unless the SIDs belong to the dataset of the recommender config.

    Compares data.dataset and data.category of the Stage-1 config stored in the semids file
    with the recommender config, and the number of SID rows with num_items of the loaded
    dataset.
    """
    stage1_config = _stage1_config(semids_data)

    mismatches = []
    for key in ("dataset", "category"):
        sid_value = stage1_config.data.get(key)
        rec_value = config.data.get(key)
        if sid_value != rec_value:
            mismatches.append(f"data.{key}: SIDs {sid_value!r} vs recommender config {rec_value!r}")

    num_rows = semids_data["semantic_ids"].shape[0]
    if num_rows != num_items:
        mismatches.append(f"SID rows {num_rows} vs {num_items} items in the dataset")

    if mismatches:
        raise ValueError("Semantic IDs do not match the dataset of this run: " + "; ".join(mismatches) + ".")


def check_semids_protocol(semids_data, config):
    """
    Raise a ValueError unless the SIDs were produced under the split protocol of the
    recommender config (under 'gts' including the cutoff quantiles), as read from the
    Stage-1 config stored in the semids file.
    """
    sid_split = get_split_config(_stage1_config(semids_data))
    rec_split = get_split_config(config)
    if sid_split != rec_split:
        raise ValueError(
            f"Semantic IDs were trained under split protocol {sid_split.describe()}, "
            f"but the recommender config uses {rec_split.describe()}. "
            f"The quantizer must be trained under the same protocol.")


def check_semids(semids_data, config, num_items):
    """Run all checks of a semids file against the recommender config and the loaded dataset."""
    check_semids_dataset(semids_data, config, num_items)
    check_semids_protocol(semids_data, config)
