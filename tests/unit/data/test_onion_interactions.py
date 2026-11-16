"""
Unit tests for data/onion_interactions.py on tiny synthetic event files (real .tsv.bz2, tab-separated,
CRLF line ends, header user_id, track_id, timestamp). Default window 2019-08-01 00:00:00 to
2019-09-30 23:59:59, k = 2.

BLOCK is the smallest 2-core (2 users x 2 tracks); user 2 hears C before A, so time order and track
order differ. Every fixture is BLOCK plus or with a change.

    user  track  timestamp            unix
    1     A      2019-08-05 10:00:00  1564999200 (T5)
    1     C      2019-08-06 10:00:00  1565085600 (T6)
    2     C      2019-08-07 10:00:00  1565172000 (T7)
    2     A      2019-08-08 10:00:00  1565258400 (T8)

BLOCK result: track_ids [A, C]; item ids are the positions in that list (A = 0, C = 1).

Further times used by single tests:
    2019-08-09 10:00:00  1565344800 (T9)
    2019-08-01 00:00:00  1564617600 (FIRST_SECOND, the window start)
    2019-09-30 23:59:59  1569887999 (LAST_SECOND, the window end)
"""
import bz2

import pytest

from data.onion_interactions import read_onion_interactions

T5, T6, T7, T8, T9 = 1564999200, 1565085600, 1565172000, 1565258400, 1565344800
FIRST_SECOND, LAST_SECOND = 1564617600, 1569887999
AUG_9 = "2019-08-09 10:00:00"
AUG_10 = "2019-08-10 10:00:00"
AUG_11 = "2019-08-11 10:00:00"

BLOCK = [
    (1, "A", "2019-08-05 10:00:00"),
    (1, "C", "2019-08-06 10:00:00"),
    (2, "C", "2019-08-07 10:00:00"),
    (2, "A", "2019-08-08 10:00:00"),
]
USER_1 = (1, [0, 1], [T5, T6])
USER_2 = (2, [1, 0], [T7, T8])
EXPECTED = ([USER_1, USER_2], ["A", "C"])


def read(tmp_path, rows, header="user_id\ttrack_id\ttimestamp", end="2019-09-30 23:59:59", k=2):
    path = tmp_path / "events.tsv.bz2"
    lines = [header] + [f"{user}\t{track}\t{timestamp}" for user, track, timestamp in rows]
    path.write_bytes(bz2.compress("".join(line + "\r\n" for line in lines).encode()))
    return read_onion_interactions(path, "2019-08-01 00:00:00", end, k)


def test_timestamps_become_unix_seconds_read_as_utc(tmp_path):
    # a local-time reading would shift every value by the offset of the machine
    assert read(tmp_path, BLOCK) == EXPECTED


def test_window_start_is_inclusive(tmp_path):
    # with an exclusive start the event is lost and user 1 falls below two tracks
    rows = [(1, "A", "2019-08-01 00:00:00")] + BLOCK[1:]
    assert read(tmp_path, rows) == ([(1, [0, 1], [FIRST_SECOND, T6]), USER_2], ["A", "C"])


def test_event_before_window_start_is_dropped(tmp_path):
    assert read(tmp_path, BLOCK + [(1, "A", "2019-07-31 23:59:59")]) == EXPECTED


def test_window_end_is_inclusive_with_crlf_line_ends(tmp_path):
    # the timestamp is the last field of a CRLF line; an unstripped \r would drop the last second
    rows = BLOCK[:3] + [(2, "A", "2019-09-30 23:59:59")]
    assert read(tmp_path, rows) == ([USER_1, (2, [1, 0], [T7, LAST_SECOND])], ["A", "C"])


def test_event_after_window_end_is_dropped(tmp_path):
    assert read(tmp_path, BLOCK + [(2, "A", "2019-10-01 00:00:00")]) == EXPECTED


def test_row_with_invalid_1970_timestamp_is_dropped_without_error(tmp_path):
    # the raw file has such a row; it lies outside the window and needs no special case
    assert read(tmp_path, BLOCK + [(1, "A", "1970-01-01 01:00:36")]) == EXPECTED


def test_bound_that_is_not_a_full_timestamp_raises(tmp_path):
    # a date-only end sorts before every event of that day and would silently cut the last day
    with pytest.raises(ValueError, match="end must be a timestamp"):
        read(tmp_path, BLOCK, end="2019-09-30")


def test_file_with_another_header_raises(tmp_path):
    # the play-count file lies next to the event file and shares its first two columns
    with pytest.raises(ValueError, match="expected the header"):
        read(tmp_path, BLOCK, header="user_id\ttrack_id\tcount")


def test_row_order_of_the_file_does_not_change_the_result(tmp_path):
    # the raw file is not in time order
    assert read(tmp_path, BLOCK[::-1]) == EXPECTED


def test_users_are_ordered_by_numeric_id(tmp_path):
    # a text sort would put 10 before 2
    rows = [(10, track, timestamp) for _, track, timestamp in BLOCK[:2]] + BLOCK[2:]
    assert read(tmp_path, rows) == ([USER_2, (10, [0, 1], [T5, T6])], ["A", "C"])


def test_equal_timestamps_of_one_user_are_ordered_by_track_id(tmp_path):
    # the order of tied rows in the file is arbitrary, so ties break by track id
    rows = [(1, "C", "2019-08-05 10:00:00"), (1, "A", "2019-08-05 10:00:00")] + BLOCK[2:]
    assert read(tmp_path, rows) == ([(1, [0, 1], [T5, T5]), USER_2], ["A", "C"])


def test_relistens_of_a_surviving_pair_stay(tmp_path):
    # next-item prediction keeps every event of a surviving pair
    assert read(tmp_path, BLOCK + [(1, "A", AUG_9)]) == (
        [(1, [0, 1, 0], [T5, T6, T9]), USER_2], ["A", "C"])


def test_identical_rows_are_two_events(tmp_path):
    # no deduplication: the raw file has such rows and the event count includes them
    assert read(tmp_path, BLOCK + [(1, "A", "2019-08-05 10:00:00")]) == (
        [(1, [0, 0, 1], [T5, T5, T6]), USER_2], ["A", "C"])


def test_user_threshold_counts_distinct_tracks_not_events(tmp_path):
    # three events on one track are one distinct track, below k
    rows = BLOCK + [(3, "A", AUG_9), (3, "A", AUG_10), (3, "A", AUG_11)]
    assert read(tmp_path, rows) == EXPECTED


def test_track_threshold_counts_distinct_users_and_ids_have_no_gap(tmp_path):
    # B has one distinct user and goes, its user stays; B sorts between A and C, yet C keeps id 1
    rows = BLOCK + [(1, "B", AUG_9), (1, "B", AUG_10), (1, "B", AUG_11)]
    assert read(tmp_path, rows) == EXPECTED


def test_core_is_counted_inside_the_window(tmp_path):
    # user 3 has one track inside the window; a count over the whole file would keep them
    rows = BLOCK + [(3, "A", AUG_9), (3, "C", "2019-10-01 00:00:00")]
    assert read(tmp_path, rows) == EXPECTED


def test_filter_repeats_until_nothing_changes(tmp_path):
    # user 4 goes, then D, then user 3; a single pass leaves user 3 in
    rows = BLOCK + [(3, "A", AUG_9), (3, "D", AUG_10), (4, "D", AUG_11)]
    assert read(tmp_path, rows) == EXPECTED


def test_empty_result_raises(tmp_path):
    # every count in BLOCK is 2, below k = 3; an empty dataset means a wrong file or window
    with pytest.raises(ValueError, match="survives the 3-core"):
        read(tmp_path, BLOCK, k=3)
