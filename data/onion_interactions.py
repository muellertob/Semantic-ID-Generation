"""
Reader for the listening events of Music4All-Onion (userid_trackid_timestamp.tsv.bz2).

Specific to this file: tab-separated with CRLF line ends and the header user_id, track_id,
timestamp; one row per listening event, in no particular order; timestamps are strings
"YYYY-MM-DD HH:MM:SS" without a time zone.
"""
import bz2
from collections import Counter
from datetime import datetime, timezone

HEADER = ["user_id", "track_id", "timestamp"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _check_bound(name, bound):
    """Raise a ValueError unless bound is a timestamp exactly as the file stores them."""
    try:
        valid = datetime.strptime(bound, TIME_FORMAT).strftime(TIME_FORMAT) == bound
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a timestamp 'YYYY-MM-DD HH:MM:SS', got {bound!r}")


def _unix(timestamp):
    """Unix seconds of a stored timestamp, read as UTC."""
    return int(datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc).timestamp())


def _k_core(pairs, k):
    """
    The largest subset of the distinct (user, track) pairs in which every user has at least
    k tracks and every track at least k users. Removing a pair can push another user or
    track below k, so the filter repeats until nothing changes.
    """
    while True:
        n_tracks = Counter(user for user, _ in pairs)
        n_users = Counter(track for _, track in pairs)
        kept = {(user, track) for user, track in pairs if n_tracks[user] >= k and n_users[track] >= k}
        if len(kept) == len(pairs):
            return kept
        pairs = kept


def read_onion_interactions(events_path, start, end, k):
    """
    Read the listening events of a time window and keep the k-core of its users and tracks.

    Args:
        events_path: Path of userid_trackid_timestamp.tsv.bz2.
        start, end: First and last timestamp of the window, both inclusive, in the stored
            format "YYYY-MM-DD HH:MM:SS". They are compared with the stored strings; for
            this fixed-width format the string order is the time order.
        k: Users need at least k distinct tracks and tracks at least k distinct users,
            counted inside the window (see _k_core).

    Every event of a surviving (user, track) pair is kept, repeated listens and identical
    rows included. Timestamps are read as UTC. A user's events are ordered by time; events
    of the same second by track id.

    Returns:
        (sequences, track_ids). track_ids are the surviving track ids in ascending order;
        the model item id of a track is its position in this list. sequences is a list of
        (user id, model item ids, unix times) per surviving user, in ascending user id
        order. Raises a ValueError if a bound is not a full timestamp, if the file has
        another header or if no event survives.
    """
    _check_bound("start", start)
    _check_bound("end", end)

    events = []  # (user, track, unix time) of the rows inside the window
    with bz2.open(events_path, "rt", newline="") as f:
        header = next(f).rstrip("\r\n").split("\t")
        if header != HEADER:
            raise ValueError(f"{events_path}: expected the header {HEADER}, got {header}")
        for line in f:
            user, track, timestamp = line.rstrip("\r\n").split("\t")
            if start <= timestamp <= end:
                events.append((int(user), track, _unix(timestamp)))

    pairs = _k_core({(user, track) for user, track, _ in events}, k)
    events = [e for e in events if (e[0], e[1]) in pairs]
    if not events:
        raise ValueError(f"{events_path}: no event from {start} to {end} survives the {k}-core")

    track_ids = sorted({track for _, track in pairs})
    item_id = {track: i for i, track in enumerate(track_ids)}
    by_user = {}
    for user, track, t in events:
        by_user.setdefault(user, []).append((t, item_id[track]))

    sequences = []
    for user in sorted(by_user):
        # item ids follow the track id order, so the tuple order is time, then track id
        rows = sorted(by_user[user])
        sequences.append((user, [item for _, item in rows], [t for t, _ in rows]))
    return sequences, track_ids
