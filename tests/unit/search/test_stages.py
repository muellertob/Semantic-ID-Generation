"""
Tests for search/stages.py: each stage gets the base config plus the spec's overrides plus the driver's keys
(seed, paths, compile) of that stage, the stage functions are called with those configs, and a stage that
leaves no meta file is an error. The stage functions are mocked; each mock writes the meta file the real
function writes to general.meta_json.
"""
import dataclasses
import json
from pathlib import Path
from unittest.mock import call, patch

import pytest
from omegaconf import OmegaConf

from search import stages
from search.spec import TrialSpec

# Stage-1 base used for every quantizer in these tests (patched into BASE_CONFIGS):
# | key              | base  | after stage1() with SPEC and compile=False            |
# |------------------|-------|-------------------------------------------------------|
# | model.beta       | 6.0   | 0.25 (spec override)                                  |
# | general.seed     | 10    | 42 (stage1_seed; stage2_seed is 7, so a swap shows)   |
# | general.use_wandb| False | False (not a driver key, kept)                        |
# | general.compile  | -     | False (the worker's choice per card)                  |
STAGE1_BASE_YAML = """\
model:
  beta: 6.0
general:
  seed: 10
  use_wandb: false
"""

SPEC = TrialSpec(
    dataset="toys", quantizer="rqvae", phase="A",
    stage1_overrides={"model.beta": 0.25}, stage2_overrides={},
    stage1_seed=42, stage2_seed=7, semids_from=None, max_steps=100_000, run_test=False,
    source={"study": "toys.A.rqvae", "trial_number": 7, "retry_of": None},
)
QUANTIZER_META = {"model_path": "quantizer.pt", "wandb_run_id": None, "wandb_run_url": None}
GENERATE_META = {"collision_rate": 0.01}

# the real table, typed out by hand so a swapped pair shows up
EXPECTED_BASE_CONFIGS = {
    ("toys", "rqvae"): ("config/rq-vae/amazon_toy_tiger_gts.yaml", "config/recommender/amazon_toy_grid_gts.yaml"),
    ("toys", "rqkmeans"): ("config/rq-kmeans/amazon_toy_rqkmeans_gts.yaml",
                           "config/recommender/amazon_toy_grid_gts.yaml"),
    ("toys", "fsq"): ("config/fsq/amazon_toy_gts.yaml", "config/recommender/amazon_toy_grid_fsq_gts.yaml"),
    ("toys", "rfsq"): ("config/fsq/amazon_toy_rfsq_fixed_gts.yaml", "config/recommender/amazon_toy_grid_fsq_gts.yaml"),
    ("music4all_onion", "rqvae"): ("config/rq-vae/onion_tiger_gts.yaml", "config/recommender/onion_grid_gts.yaml"),
    ("music4all_onion", "rqkmeans"): ("config/rq-kmeans/onion_rqkmeans_gts.yaml",
                                      "config/recommender/onion_grid_gts.yaml"),
    ("music4all_onion", "fsq"): ("config/fsq/onion_gts.yaml", "config/recommender/onion_grid_fsq_gts.yaml"),
}


def _writes_meta(meta):
    """Side effect of a mocked stage function: write `meta` to the config's general.meta_json, as the real one does."""
    def side_effect(config_path, *args, **kwargs):
        path = Path(OmegaConf.load(config_path).general.meta_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta))
    return side_effect


def _load(path):
    return OmegaConf.to_container(OmegaConf.load(path))


@pytest.fixture
def stage1_mocks(tmp_path, monkeypatch):
    """Patches the base configs to a small YAML and mocks the three Stage-1 functions; returns (folder, mocks)."""
    base = tmp_path / "stage1_base.yaml"
    base.write_text(STAGE1_BASE_YAML)
    monkeypatch.setattr(stages, "BASE_CONFIGS", {("toys", "rqvae"): (str(base), "unused.yaml"),
                                                 ("toys", "rqkmeans"): (str(base), "unused.yaml")})
    with patch.object(stages.train_quantizer, "run_training", side_effect=_writes_meta(QUANTIZER_META)) as quantizer, \
            patch.object(stages.train_rqkmeans, "run_training", side_effect=_writes_meta(QUANTIZER_META)) as rqkmeans, \
            patch.object(stages.generate_semids, "run_generation", side_effect=_writes_meta(GENERATE_META)) as generate:
        yield tmp_path / "trial_0007", {"quantizer": quantizer, "rqkmeans": rqkmeans, "generate": generate}


def _expected_stage1_config(folder, meta_name):
    return {
        "model": {"beta": 0.25},
        "general": {"seed": 42, "use_wandb": False, "model_path": f"{folder}/checkpoints/quantizer.pt",
                    "meta_json": f"{folder}/meta/{meta_name}", "wandb_dir": f"{folder}/wandb", "compile": False},
    }


# ---- base configs ---------------------------------------------------------------------------------------------

def test_base_config_table_is_the_reference_pairing():
    # FSQ and RFSQ need the Stage-2 config with codebook_size 288; the others the one with 256
    assert stages.BASE_CONFIGS == EXPECTED_BASE_CONFIGS


@pytest.mark.parametrize("dataset, quantizer", list(EXPECTED_BASE_CONFIGS))
def test_every_base_config_file_exists(dataset, quantizer):
    stage1_base, stage2_base = stages.base_configs(dataset, quantizer)
    assert (stage1_base.is_file(), stage2_base.is_file()) == (True, True)


def test_pair_without_base_configs_is_rejected():
    # there is no RFSQ config for Music4All-Onion yet
    with pytest.raises(ValueError) as excinfo:
        stages.base_configs("music4all_onion", "rfsq")
    assert str(excinfo.value) == "no base configs for dataset 'music4all_onion' and quantizer 'rfsq'"


# ---- Stage 1 --------------------------------------------------------------------------------------------------

def test_neural_quantizer_trains_on_base_plus_overrides_plus_driver_keys(stage1_mocks):
    folder, mocks = stage1_mocks
    stages.stage1(SPEC, folder, compile=False)
    assert mocks["quantizer"].call_args_list == [call(str(folder / "configs" / "stage1.yaml"))]
    assert _load(folder / "configs" / "stage1.yaml") == _expected_stage1_config(folder, "quantizer.json")


def test_generate_uses_the_training_config_with_its_own_meta_file(stage1_mocks):
    # the SIDs must come from the config the quantizer was trained with
    folder, mocks = stage1_mocks
    stages.stage1(SPEC, folder, compile=False)
    assert mocks["generate"].call_args_list == [call(str(folder / "configs" / "generate.yaml"),
                                                     model_path=str(folder / "checkpoints" / "quantizer.pt"),
                                                     output_path=str(folder / "semids.pt"))]
    assert _load(folder / "configs" / "generate.yaml") == _expected_stage1_config(folder, "generate.json")


def test_neural_stage1_returns_the_semids_path_and_both_metas(stage1_mocks):
    folder, _ = stage1_mocks
    result = stages.stage1(SPEC, folder, compile=False)
    assert result == stages.Stage1Result(folder / "semids.pt", QUANTIZER_META, GENERATE_META)


def test_rqkmeans_writes_its_semids_while_fitting_and_skips_generate(stage1_mocks):
    folder, mocks = stage1_mocks
    stages.stage1(dataclasses.replace(SPEC, quantizer="rqkmeans"), folder, compile=False)
    expected = _expected_stage1_config(folder, "quantizer.json")
    expected["general"]["output_path"] = f"{folder}/semids.pt"
    assert mocks["rqkmeans"].call_args_list == [call(str(folder / "configs" / "stage1.yaml"))]
    assert _load(folder / "configs" / "stage1.yaml") == expected
    assert mocks["generate"].call_count == 0


def test_rqkmeans_stage1_has_no_generate_meta(stage1_mocks):
    # None in trial.json marks a stage that never ran, unlike an empty meta
    folder, _ = stage1_mocks
    result = stages.stage1(dataclasses.replace(SPEC, quantizer="rqkmeans"), folder, compile=False)
    assert result == stages.Stage1Result(folder / "semids.pt", QUANTIZER_META, None)


def test_stage1_with_semids_from_is_rejected(stage1_mocks):
    # Phase B and Stage-2 reruns reuse a trial's SIDs and must never fit new ones
    folder, mocks = stage1_mocks
    spec = dataclasses.replace(SPEC, stage1_overrides={}, semids_from="/search/toys/A-rqvae/trial_0003")
    with pytest.raises(ValueError) as excinfo:
        stages.stage1(spec, folder, compile=False)
    assert str(excinfo.value) == "Stage 1 does not run when semids_from is set (/search/toys/A-rqvae/trial_0003)"
    assert [m.call_count for m in mocks.values()] == [0, 0, 0]


def test_stage1_override_of_a_driver_key_is_rejected(stage1_mocks):
    # the driver would overwrite it, so trial.json would state a seed that never ran
    folder, mocks = stage1_mocks
    with pytest.raises(ValueError) as excinfo:
        stages.stage1(dataclasses.replace(SPEC, stage1_overrides={"general.seed": 1}), folder, compile=False)
    assert str(excinfo.value) == "stage1_overrides must not set driver keys: ['general.seed']"
    assert [m.call_count for m in mocks.values()] == [0, 0, 0]


@pytest.mark.parametrize("silent, meta_name", [("quantizer", "quantizer.json"), ("generate", "generate.json")])
def test_stage1_function_that_writes_no_meta_is_an_error(stage1_mocks, silent, meta_name):
    # a stage without its meta must not count as done
    folder, mocks = stage1_mocks
    mocks[silent].side_effect = None
    with pytest.raises(RuntimeError) as excinfo:
        stages.stage1(SPEC, folder, compile=False)
    assert str(excinfo.value) == f"the stage finished without writing its meta file {folder / 'meta' / meta_name}"
