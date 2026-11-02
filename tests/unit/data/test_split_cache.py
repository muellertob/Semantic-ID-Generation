"""
Unit tests for data/split_cache.py: file names, meta checks, atomic saving and the
build-once behaviour for both protocols.
"""
import hashlib
from unittest.mock import Mock

import pytest
import torch

from data.protocols import SplitConfig
from data.split_cache import (FORMAT_VERSION, load_or_build_split, save_atomic, split_cache_path)

PERM = torch.tensor([3, 7, 0, 9, 2, 5, 8, 1, 4, 6])
LOO = SplitConfig("loo")
GTS = SplitConfig("gts", 0.75, 0.75)
DAY = 86_400

# user, items, times (days); one user is too short
SEQUENCES = [
    (1, [0, 1, 2, 3, 4, 5], [d * DAY for d in (1, 2, 3, 4, 12, 13)]),
    (2, [1, 2, 3, 4, 5, 6, 7], [d * DAY for d in (1, 2, 3, 4, 20, 21, 22)]),
    (3, [0, 1], [DAY, 2 * DAY]),
]


def load(tmp_path, split_config=LOO, reader=None, perm=PERM, num_items=10):
    reader = reader or Mock(return_value=SEQUENCES)
    return load_or_build_split("synth", split_config, str(tmp_path), reader, perm, num_items)


class TestPath:
    def test_loo_has_no_quantiles(self):
        assert split_cache_path("d", "toys", LOO) == "d/split_toys_loo.pt"

    def test_gts_has_quantiles(self):
        assert split_cache_path("d", "toys", SplitConfig("gts", 0.9, 0.8)) == "d/split_toys_gts_q0.9-0.8.pt"


@pytest.mark.parametrize("split_config", [LOO, GTS], ids=["loo", "gts"])
class TestCache:
    def _path(self, tmp_path, split_config):
        return split_cache_path(str(tmp_path), "synth", split_config)

    def test_built_once_then_loaded(self, tmp_path, split_config):
        reader = Mock(return_value=SEQUENCES)
        built = load(tmp_path, split_config, reader)
        cached = load(tmp_path, split_config, reader)
        reader.assert_called_once_with()
        assert cached["meta"] == built["meta"] and cached["stats"] == built["stats"]
        assert cached["history"]["test"]["itemId"] == built["history"]["test"]["itemId"]

    def test_meta_round_trip(self, tmp_path, split_config):
        split = load(tmp_path, split_config)
        assert split["meta"] == {
            "format_version": FORMAT_VERSION, "protocol": split_config.protocol,
            "q_test": split_config.q_test, "q_val": split_config.q_val, "num_items": 10,
            "perm_md5": hashlib.md5(PERM.numpy().tobytes()).hexdigest()}
        assert split["protocol"] == split_config.protocol

    def _tamper(self, tmp_path, split_config, **changes):
        load(tmp_path, split_config)
        path = self._path(tmp_path, split_config)
        split = torch.load(path, weights_only=False)
        split["meta"].update(changes)
        torch.save(split, path)

    @pytest.mark.parametrize("field,value", [
        ("format_version", FORMAT_VERSION + 1), ("protocol", "other"), ("num_items", 11),
        ("perm_md5", "0" * 32)])
    def test_each_mismatch_raises(self, tmp_path, split_config, field, value):
        self._tamper(tmp_path, split_config, **{field: value})
        with pytest.raises(ValueError, match=field) as err:
            load(tmp_path, split_config)
        assert "Delete the file" in str(err.value)

    def test_all_mismatching_fields_are_named(self, tmp_path, split_config):
        self._tamper(tmp_path, split_config, format_version=0, num_items=11)
        with pytest.raises(ValueError) as err:
            load(tmp_path, split_config)
        assert "format_version" in str(err.value) and "num_items" in str(err.value)

    def test_changed_permutation_raises(self, tmp_path, split_config):
        load(tmp_path, split_config)
        with pytest.raises(ValueError, match="perm_md5"):
            load(tmp_path, split_config, perm=PERM.flip(0))

    def test_missing_meta_raises(self, tmp_path, split_config):
        load(tmp_path, split_config)
        path = self._path(tmp_path, split_config)
        split = torch.load(path, weights_only=False)
        del split["meta"]
        torch.save(split, path)
        with pytest.raises(ValueError, match="no meta"):
            load(tmp_path, split_config)

    def test_atomic_save_leaves_no_temp_file(self, tmp_path, split_config):
        load(tmp_path, split_config)
        assert [p.name for p in tmp_path.iterdir()] == [self._path(tmp_path, split_config).rsplit("/", 1)[1]]


def test_protocols_use_separate_files(tmp_path):
    load(tmp_path, LOO)
    load(tmp_path, GTS)
    load(tmp_path, SplitConfig("gts", 0.5, 0.5))
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "split_synth_gts_q0.5-0.5.pt", "split_synth_gts_q0.75-0.75.pt", "split_synth_loo.pt"]


def test_loo_and_gts_content(tmp_path):
    loo = load(tmp_path, LOO)
    assert loo["history"]["test"]["userId"].tolist() == [1, 2]
    gts = load(tmp_path, GTS)
    assert gts["stage1_items"].shape == (10,) and "test_cold" in gts["history"]


def test_failed_save_keeps_target_and_removes_temp(tmp_path, monkeypatch):
    target = tmp_path / "x.pt"
    save_atomic({"a": 1}, str(target))

    def broken_save(obj, path):
        open(path, "wb").write(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(RuntimeError):
        save_atomic({"a": 2}, str(target))
    monkeypatch.undo()
    assert torch.load(target, weights_only=False) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.pt"]
