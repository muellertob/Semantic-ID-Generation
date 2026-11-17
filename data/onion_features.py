"""
Reader for the item features of Music4All-Onion (numeric id_<feature>.tsv.bz2 files, e.g.
id_maest.tsv.bz2).

Specific to these files: tab-separated, header "id" followed by the d column names, then one
row per track with the track id and its d values, in no particular order.
"""
import bz2

import torch


def read_onion_features(features_path, track_ids):
    """
    Read the feature vectors of the given tracks, in the order of track_ids.

    Args:
        features_path: Path of a numeric Onion feature file.
        track_ids: Track ids in model item id order (see read_onion_interactions).

    Returns:
        float32 tensor (len(track_ids), d); row i is the feature vector of track_ids[i], d the
        number of columns after "id" in the header. Tracks of the file that are not in
        track_ids are skipped. Raises a ValueError if any row of the file has another number
        of values than the header, if a track id appears twice in the file or if a track of
        track_ids has no row.
    """
    position = {track: i for i, track in enumerate(track_ids)}
    found = [False] * len(track_ids)
    seen = set()
    with bz2.open(features_path, "rt") as f:
        d = len(next(f).rstrip("\r\n").split("\t")) - 1
        features = torch.zeros(len(track_ids), d, dtype=torch.float32)
        for line in f:
            track, *values = line.rstrip("\r\n").split("\t")
            if len(values) != d:
                raise ValueError(f"{features_path}: track {track} has {len(values)} values, "
                                 f"the header names {d}")
            if track in seen:
                raise ValueError(f"{features_path}: track {track} appears twice")
            seen.add(track)
            i = position.get(track)
            if i is not None:
                features[i] = torch.tensor([float(v) for v in values], dtype=torch.float32)
                found[i] = True

    missing = found.count(False)
    if missing:
        raise ValueError(f"{features_path}: {missing} of {len(track_ids)} tracks have no feature row")
    return features
