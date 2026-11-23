"""
Tests for search/config_io.py: a resolved stage config is the base config with exactly the given overrides,
and an override that matches no key of the base is rejected instead of being added unused.
"""
import pytest
from omegaconf import OmegaConf

from search.config_io import write_resolved_config

# | key          | base       | override     | resolved     |
# |--------------|------------|--------------|--------------|
# | model.beta   | 6.0        | 0.25         | 0.25         |
# | model.levels | [8, 6, 6]  | [4, 4, 4, 4] | [4, 4, 4, 4] |
# | train.lr     | 0.01       | -            | 0.01         |
BASE_YAML = """\
model:
  beta: 6.0
  levels: [8, 6, 6]
train:
  lr: 0.01
"""


@pytest.fixture
def base(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(BASE_YAML)
    return path


def _load(path):
    return OmegaConf.to_container(OmegaConf.load(path))


def test_resolved_config_is_the_base_with_the_overrides(base, tmp_path):
    # dest lies in a folder that does not exist yet, as configs/ in a fresh trial folder
    dest = tmp_path / "trial" / "configs" / "stage1.yaml"
    write_resolved_config(base, {"model.beta": 0.25, "model.levels": [4, 4, 4, 4]}, dest)
    assert _load(dest) == {"model": {"beta": 0.25, "levels": [4, 4, 4, 4]}, "train": {"lr": 0.01}}


def test_write_returns_the_destination_path(base, tmp_path):
    # the stages pass this path on to the stage function
    dest = tmp_path / "out.yaml"
    assert write_resolved_config(base, {"model.beta": 0.25}, dest) == dest


def test_base_file_is_left_unchanged(base, tmp_path):
    write_resolved_config(base, {"model.beta": 0.25}, tmp_path / "out.yaml")
    assert base.read_text() == BASE_YAML


def test_override_key_missing_from_the_base_is_rejected(base, tmp_path):
    # "betaa" is a typo of "beta": added silently, the searched value would never reach the model
    dest = tmp_path / "out.yaml"
    with pytest.raises(ValueError) as excinfo:
        write_resolved_config(base, {"model.betaa": 0.25}, dest)
    assert str(excinfo.value) == f"override keys not in the base config {base}: ['model.betaa']"
    assert not dest.exists()


def test_allow_new_admits_the_named_new_key(base, tmp_path):
    # the driver's general.* keys are not in every base config
    dest = tmp_path / "out.yaml"
    write_resolved_config(base, {"general.meta_json": "m.json"}, dest, allow_new={"general.meta_json"})
    assert _load(dest) == {"model": {"beta": 6.0, "levels": [8, 6, 6]}, "train": {"lr": 0.01},
                           "general": {"meta_json": "m.json"}}
