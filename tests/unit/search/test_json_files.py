"""Tests for search/json_files.py on the local file system; NFS failures are simulated by mocking os.link."""
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from search import json_files
from search.json_files import create_json, replace_json

DATA = {"spec": {"dataset": "toys"}, "context": {"host": "gpu1"}}


def _files(folder):
    return sorted(path.name for path in folder.iterdir())


def test_create_json_writes_the_data(tmp_path):
    create_json(tmp_path / "trial.json", DATA)

    assert (_files(tmp_path), json.loads((tmp_path / "trial.json").read_text())) == (["trial.json"], DATA)


def test_create_json_rejects_an_existing_file_and_leaves_it_unchanged(tmp_path):
    (tmp_path / "trial.json").write_text('{"earlier": "run"}')

    with pytest.raises(FileExistsError):
        create_json(tmp_path / "trial.json", DATA)

    assert (_files(tmp_path), (tmp_path / "trial.json").read_text()) == (["trial.json"], '{"earlier": "run"}')


def test_create_json_with_data_that_is_not_json_leaves_no_file(tmp_path):
    with pytest.raises(TypeError):
        create_json(tmp_path / "trial.json", {"host": Path("/gpu1")})

    assert _files(tmp_path) == []


def test_a_link_that_succeeds_but_reports_an_error_counts_as_created(tmp_path, monkeypatch):
    # NFS: the server made the link, but its reply was lost and the retried call reports an error;
    # the unique tmp file then has 2 links
    real_link = os.link

    def link_then_fail(src, dst):
        real_link(src, dst)
        raise OSError("lost reply")
    monkeypatch.setattr(json_files.os, "link", Mock(side_effect=link_then_fail))

    create_json(tmp_path / "trial.json", DATA)

    assert (_files(tmp_path), json.loads((tmp_path / "trial.json").read_text())) == (["trial.json"], DATA)


def test_a_link_that_fails_without_linking_raises_and_leaves_no_file(tmp_path, monkeypatch):
    # the tmp file keeps 1 link, so this is a real failure, not a lost reply
    monkeypatch.setattr(json_files.os, "link", Mock(side_effect=PermissionError("denied")))

    with pytest.raises(PermissionError):
        create_json(tmp_path / "trial.json", DATA)

    assert _files(tmp_path) == []


@pytest.mark.parametrize("write", [create_json, replace_json])
def test_a_write_that_fails_halfway_leaves_no_file(tmp_path, monkeypatch, write):
    # e.g. the NFS quota runs full after the first bytes
    real_write_text = Path.write_text

    def write_part_then_fail(self, text):
        real_write_text(self, text[:5])
        raise OSError("Disk quota exceeded")
    monkeypatch.setattr(Path, "write_text", write_part_then_fail)

    with pytest.raises(OSError, match="Disk quota exceeded"):
        write(tmp_path / "trial.json", DATA)

    assert _files(tmp_path) == []


@pytest.mark.parametrize("write", [create_json, replace_json])
def test_another_process_tmp_file_for_the_same_target_is_not_touched(tmp_path, write):
    # a writer on another host, between its write and its link or replace
    other = tmp_path / "trial.json.otherhost.123.tmp"
    other.write_text('{"other": "writer"}')

    write(tmp_path / "trial.json", DATA)

    assert (_files(tmp_path), other.read_text(), json.loads((tmp_path / "trial.json").read_text())) == (
        ["trial.json", "trial.json.otherhost.123.tmp"], '{"other": "writer"}', DATA)


def test_replace_json_overwrites_an_existing_file(tmp_path):
    (tmp_path / "trial.json").write_text('{"earlier": "run"}')

    replace_json(tmp_path / "trial.json", DATA)

    assert (_files(tmp_path), json.loads((tmp_path / "trial.json").read_text())) == (["trial.json"], DATA)
