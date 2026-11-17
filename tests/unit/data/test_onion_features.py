"""
Unit tests for data/onion_features.py on tiny synthetic feature files (real .tsv.bz2, tab-separated,
LF line ends, header id, 0, 1, so d = 2).

FEAT lists the tracks in an order that differs from the sorted track ids:

    id  0    1
    C   5.0  6.0
    B   3.0  4.0
    A   1.0  2.0
"""
import bz2

import pytest
import torch

from data.onion_features import read_onion_features

FEAT = [("C", "5.0", "6.0"), ("B", "3.0", "4.0"), ("A", "1.0", "2.0")]


def read(tmp_path, rows, track_ids):
    path = tmp_path / "features.tsv.bz2"
    lines = ["id\t0\t1"] + ["\t".join(row) for row in rows]
    path.write_bytes(bz2.compress("".join(line + "\n" for line in lines).encode()))
    return read_onion_features(path, track_ids)


def test_row_i_holds_the_feature_of_the_ith_track(tmp_path):
    # the file lists C first; reading in file order would give C's vector to A
    expected = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    assert torch.equal(read(tmp_path, FEAT, ["A", "B", "C"]), expected)


def test_tracks_outside_track_ids_are_skipped_without_a_gap(tmp_path):
    # B lies between A and C in the file but is not an item, so C moves up to row 1
    expected = torch.tensor([[1, 2], [5, 6]], dtype=torch.float32)
    assert torch.equal(read(tmp_path, FEAT, ["A", "C"]), expected)


def test_result_is_float32(tmp_path):
    # Python floats would give float64 and twice the memory
    assert read(tmp_path, FEAT, ["A", "B", "C"]).dtype == torch.float32


def test_exponent_notation_and_signs_are_read_as_numbers(tmp_path):
    # both values are exact in float32
    expected = torch.tensor([[-0.25, 15.0]], dtype=torch.float32)
    assert torch.equal(read(tmp_path, [("A", "-2.5e-01", "1.5E+01")], ["A"]), expected)


def test_track_without_feature_row_raises_and_counts_missing(tmp_path):
    # a zero vector or a dropped row would silently create a fake item or shift later ids
    with pytest.raises(ValueError, match="1 of 2"):
        read(tmp_path, FEAT, ["A", "D"])


def test_track_id_twice_in_file_raises(tmp_path):
    # which of the two vectors is right is unknown, so the last one must not win silently
    with pytest.raises(ValueError, match="appears twice"):
        read(tmp_path, FEAT + [("A", "7.0", "8.0")], ["A", "B", "C"])


def test_row_with_other_number_of_values_than_header_raises(tmp_path):
    # a cut row would mix up the dimensions or fail later with an unclear shape error
    with pytest.raises(ValueError, match="1 values"):
        read(tmp_path, [("C", "5.0", "6.0"), ("B", "3.0"), ("A", "1.0", "2.0")], ["A", "B", "C"])
