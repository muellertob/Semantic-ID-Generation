"""
Tests for search/stages.py: each stage gets the base config plus the spec's overrides plus the driver's keys
(seed, paths, compile, W&B names) of that stage, the recommender and the test read the right SIDs, the stage
functions are called with those configs, and a stage that leaves no meta file is an error. The stage functions are
mocked; each mock writes the meta file the real function writes to general.meta_json.
"""
import dataclasses
import json
from pathlib import Path
from unittest.mock import call, patch

import pytest
from omegaconf import OmegaConf

from search import stages
from search.spec import TrialSpec

# Quantizer base used for every quantizer in these tests (patched into BASE_CONFIGS):
# | key                | base  | after quantizer() with SPEC and compile=False                       |
# |--------------------|-------|---------------------------------------------------------------------|
# | model.beta         | 6.0   | 0.25 (spec override)                                                |
# | general.seed       | 10    | 42 (quantizer_seed; recommender_seed is 7, so a swap shows)         |
# | general.use_wandb  | False | False (not a driver key, kept)                                      |
# | general.compile    | -     | False (the worker's choice per card)                                |
# | general.model_path | -     | <f>/checkpoints/quantizer.pt (inside the trial folder)              |
# | general.meta_json  | -     | <f>/meta/quantizer.json (generate: meta/generate.json)              |
# | general.wandb_dir  | -     | <f>/wandb                                                           |
# | general.wandb_*    | -     | project sid-search-toys-quantizer, group A-rqvae (the study folder), |
# |                    |       | name A-rqvae-trial_0007, tags [A, rqvae, eager]                     |
QUANTIZER_BASE_YAML = """\
model:
  beta: 6.0
general:
  seed: 10
  use_wandb: false
"""

SPEC = TrialSpec(
    dataset="toys", quantizer="rqvae", phase="A",
    quantizer_overrides={"model.beta": 0.25}, recommender_overrides={"seq2seq.dropout": 0.2},
    quantizer_seed=42, recommender_seed=7, semids_from=None, max_steps=100_000, run_test=False,
    source={"study": "toys.A.rqvae", "trial_number": 7, "retry_of": None},
)
# Recommender base (patched into BASE_CONFIGS):
# | key                           | base  | after recommender_config() with SPEC and compile=False           |
# |-------------------------------|-------|-------------------------------------------------------------------|
# | seq2seq.dropout               | 0.1   | 0.2 (spec override)                                               |
# | general.seed                  | 10    | 7 (recommender_seed)                                              |
# | general.compile               | True  | False (the card's choice wins over the base)                      |
# | general.use_wandb             | False | False (not a driver key, kept)                                    |
# | general.run_name              | -     | "recommender" (so the checkpoint is checkpoints/recommender_best.pt) |
# | general.recommender_model_dir | -     | <f>/checkpoints (inside the trial folder)                         |
# | general.meta_json             | -     | <f>/meta/recommender.json (test: meta/test.json)                  |
# | general.wandb_dir             | -     | <f>/wandb                                                         |
# | general.wandb_*               | -     | as for the quantizer, project sid-search-toys-recommender         |
RECOMMENDER_BASE_YAML = """\
seq2seq:
  dropout: 0.1
general:
  seed: 10
  compile: true
  use_wandb: false
"""
SOURCE_TRIAL = "/search/toys/A-rqvae/trial_0003"
TEST_META = {"metrics": {"ndcg@10": 0.02}}

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
def quantizer_mocks(tmp_path, monkeypatch):
    """Patches the base configs to a small YAML and mocks the three quantizer functions; returns (folder, mocks)."""
    base = tmp_path / "quantizer_base.yaml"
    base.write_text(QUANTIZER_BASE_YAML)
    monkeypatch.setattr(stages, "BASE_CONFIGS", {("toys", "rqvae"): (str(base), "unused.yaml"),
                                                 ("toys", "rqkmeans"): (str(base), "unused.yaml")})
    with patch.object(stages.train_quantizer, "run_training", side_effect=_writes_meta(QUANTIZER_META)) as quantizer, \
            patch.object(stages.train_rqkmeans, "run_training", side_effect=_writes_meta(QUANTIZER_META)) as rqkmeans, \
            patch.object(stages.generate_semids, "run_generation", side_effect=_writes_meta(GENERATE_META)) as generate:
        yield tmp_path / "A-rqvae" / "trial_0007", {"train": quantizer, "rqkmeans": rqkmeans, "generate": generate}


def _expected_quantizer_config(folder, meta_name):
    return {
        "model": {"beta": 0.25},
        "general": {"seed": 42, "use_wandb": False, "model_path": f"{folder}/checkpoints/quantizer.pt",
                    "meta_json": f"{folder}/meta/{meta_name}", "wandb_dir": f"{folder}/wandb", "compile": False,
                    "wandb_project": "sid-search-toys-quantizer", "wandb_group": "A-rqvae",
                    "wandb_run_name": "A-rqvae-trial_0007", "wandb_tags": ["A", "rqvae", "eager"]},
    }


# ---- base configs ---------------------------------------------------------------------------------------------

def test_base_config_table_is_the_reference_pairing():
    # FSQ and RFSQ need the recommender config with codebook_size 288; the others the one with 256
    assert stages.BASE_CONFIGS == EXPECTED_BASE_CONFIGS


@pytest.mark.parametrize("dataset, quantizer", list(EXPECTED_BASE_CONFIGS))
def test_every_base_config_file_exists(dataset, quantizer):
    quantizer_base, recommender_base = stages.base_configs(dataset, quantizer)
    assert (quantizer_base.is_file(), recommender_base.is_file()) == (True, True)


def test_pair_without_base_configs_is_rejected():
    # there is no RFSQ config for Music4All-Onion yet
    with pytest.raises(ValueError) as excinfo:
        stages.base_configs("music4all_onion", "rfsq")
    assert str(excinfo.value) == "no base configs for dataset 'music4all_onion' and quantizer 'rfsq'"


# ---- quantizer stage ------------------------------------------------------------------------------------------

def test_neural_quantizer_trains_on_base_plus_overrides_plus_driver_keys(quantizer_mocks):
    folder, mocks = quantizer_mocks
    stages.quantizer(SPEC, folder, compile=False)
    assert mocks["train"].call_args_list == [call(str(folder / "configs" / "quantizer.yaml"))]
    assert _load(folder / "configs" / "quantizer.yaml") == _expected_quantizer_config(folder, "quantizer.json")


def test_generate_uses_the_training_config_with_its_own_meta_file(quantizer_mocks):
    # the SIDs must come from the config the quantizer was trained with
    folder, mocks = quantizer_mocks
    stages.quantizer(SPEC, folder, compile=False)
    assert mocks["generate"].call_args_list == [call(str(folder / "configs" / "generate.yaml"),
                                                     model_path=str(folder / "checkpoints" / "quantizer.pt"),
                                                     output_path=str(folder / "semids.pt"))]
    assert _load(folder / "configs" / "generate.yaml") == _expected_quantizer_config(folder, "generate.json")


def test_neural_quantizer_returns_the_semids_path_and_both_metas(quantizer_mocks):
    folder, _ = quantizer_mocks
    result = stages.quantizer(SPEC, folder, compile=False)
    assert result == stages.QuantizerResult(folder / "semids.pt", QUANTIZER_META, GENERATE_META)


def test_rqkmeans_writes_its_semids_while_fitting_and_skips_generate(quantizer_mocks):
    folder, mocks = quantizer_mocks
    stages.quantizer(dataclasses.replace(SPEC, quantizer="rqkmeans"), folder, compile=False)
    expected = _expected_quantizer_config(folder, "quantizer.json")
    expected["general"]["output_path"] = f"{folder}/semids.pt"
    expected["general"]["wandb_tags"] = ["A", "rqkmeans", "eager"]  # the tag follows the spec
    assert mocks["rqkmeans"].call_args_list == [call(str(folder / "configs" / "quantizer.yaml"))]
    assert _load(folder / "configs" / "quantizer.yaml") == expected
    assert mocks["generate"].call_count == 0


def test_rqkmeans_quantizer_has_no_generate_meta(quantizer_mocks):
    # None in trial.json marks a stage that never ran, unlike an empty meta
    folder, _ = quantizer_mocks
    result = stages.quantizer(dataclasses.replace(SPEC, quantizer="rqkmeans"), folder, compile=False)
    assert result == stages.QuantizerResult(folder / "semids.pt", QUANTIZER_META, None)


def test_quantizer_with_semids_from_is_rejected(quantizer_mocks):
    # Phase B and recommender-only reruns reuse a trial's SIDs and must never fit new ones
    folder, mocks = quantizer_mocks
    spec = dataclasses.replace(SPEC, quantizer_overrides={}, semids_from=SOURCE_TRIAL)
    with pytest.raises(ValueError) as excinfo:
        stages.quantizer(spec, folder, compile=False)
    assert str(excinfo.value) == f"the quantizer stage does not run when semids_from is set ({SOURCE_TRIAL})"
    assert [m.call_count for m in mocks.values()] == [0, 0, 0]


def test_quantizer_override_of_a_driver_key_is_rejected(quantizer_mocks):
    # the driver would overwrite it, so trial.json would state a seed that never ran
    folder, mocks = quantizer_mocks
    with pytest.raises(ValueError) as excinfo:
        stages.quantizer(dataclasses.replace(SPEC, quantizer_overrides={"general.seed": 1}), folder, compile=False)
    assert str(excinfo.value) == "quantizer_overrides must not set driver keys: ['general.seed']"
    assert [m.call_count for m in mocks.values()] == [0, 0, 0]


@pytest.mark.parametrize("silent, meta_name", [("train", "quantizer.json"), ("generate", "generate.json")])
def test_a_quantizer_stage_call_that_writes_no_meta_is_an_error(quantizer_mocks, silent, meta_name):
    # a stage without its meta must not count as done
    folder, mocks = quantizer_mocks
    mocks[silent].side_effect = None
    with pytest.raises(RuntimeError) as excinfo:
        stages.quantizer(SPEC, folder, compile=False)
    assert str(excinfo.value) == f"the stage finished without writing its meta file {folder / 'meta' / meta_name}"


# ---- recommender stage and test ------------------------------------------------------------------------------

def _writes_recommender_meta(config_path, *args, **kwargs):
    """Side effect of the mocked train_seq2seq.run_training: the recommender meta the real one writes."""
    general = OmegaConf.load(config_path).general
    meta = {"recommender_run_name": general.run_name,
            "best_model_path": f"{general.recommender_model_dir}/{general.run_name}_best.pt"}
    Path(general.meta_json).parent.mkdir(parents=True, exist_ok=True)
    Path(general.meta_json).write_text(json.dumps(meta))


def _on_validation(record):
    return False


def _recommender(spec, folder):
    return stages.recommender(spec, folder, stages.recommender_config(spec, folder, compile=False), _on_validation)


@pytest.fixture
def recommender_mocks(tmp_path, monkeypatch):
    """Patches the recommender base config to a small YAML and mocks training and testing; returns (folder, mocks)."""
    base = tmp_path / "recommender_base.yaml"
    base.write_text(RECOMMENDER_BASE_YAML)
    monkeypatch.setattr(stages, "BASE_CONFIGS", {("toys", "rqvae"): ("unused.yaml", str(base))})
    with patch.object(stages.train_seq2seq, "run_training", side_effect=_writes_recommender_meta) as train, \
            patch.object(stages.test_seq2seq, "run_testing", side_effect=_writes_meta(TEST_META)) as test:
        yield tmp_path / "A-rqvae" / "trial_0007", {"train": train, "test": test}


def _expected_recommender_config(folder, meta_name):
    return {
        "seq2seq": {"dropout": 0.2},
        "general": {"seed": 7, "compile": False, "use_wandb": False, "meta_json": f"{folder}/meta/{meta_name}",
                    "run_name": "recommender", "recommender_model_dir": f"{folder}/checkpoints",
                    "wandb_dir": f"{folder}/wandb", "wandb_project": "sid-search-toys-recommender",
                    "wandb_group": "A-rqvae", "wandb_run_name": "A-rqvae-trial_0007",
                    "wandb_tags": ["A", "rqvae", "eager"]},
    }


def test_recommender_trains_on_base_plus_overrides_plus_driver_keys(recommender_mocks):
    folder, mocks = recommender_mocks
    _recommender(SPEC, folder)
    assert mocks["train"].call_args_list == [call(str(folder / "configs" / "recommender.yaml"),
                                                  str(folder / "semids.pt"),
                                                  on_validation=_on_validation)]
    assert _load(folder / "configs" / "recommender.yaml") == _expected_recommender_config(folder, "recommender.json")


def test_recommender_with_semids_from_trains_on_the_source_trials_semids(recommender_mocks):
    # Phase B and recommender-only reruns train on the SIDs of the trial they start from
    folder, mocks = recommender_mocks
    spec = dataclasses.replace(SPEC, quantizer_overrides={}, semids_from=SOURCE_TRIAL)
    _recommender(spec, folder)
    assert mocks["train"].call_args_list == [call(str(folder / "configs" / "recommender.yaml"),
                                                  f"{SOURCE_TRIAL}/semids.pt", on_validation=_on_validation)]


def test_recommender_returns_the_best_checkpoint_named_in_the_recommender_meta(recommender_mocks):
    folder, _ = recommender_mocks
    result = _recommender(SPEC, folder)
    best = f"{folder}/checkpoints/recommender_best.pt"
    assert result == stages.RecommenderResult(Path(best), {"recommender_run_name": "recommender",
                                                           "best_model_path": best})


def test_recommender_override_of_a_driver_key_is_rejected(recommender_mocks):
    # the driver would overwrite it, so trial.json would state a seed that never ran
    folder, mocks = recommender_mocks
    spec = dataclasses.replace(SPEC, recommender_overrides={"general.seed": 1})
    with pytest.raises(ValueError) as excinfo:
        _recommender(spec, folder)
    assert str(excinfo.value) == "recommender_overrides must not set driver keys: ['general.seed']"
    assert [m.call_count for m in mocks.values()] == [0, 0]


def test_a_compiled_recommender_is_tagged_compile(recommender_mocks):
    folder, _ = recommender_mocks
    expected = _expected_recommender_config(folder, "recommender.json")
    expected["general"].update(compile=True, wandb_tags=["A", "rqvae", "compile"])

    assert _load(stages.recommender_config(SPEC, folder, compile=True)) == expected


def test_a_rerun_is_grouped_by_its_source_trial(recommender_mocks):
    # the seeds of one candidate stand together in W&B; the phase tag tells confirmation from final runs
    folder, _ = recommender_mocks
    rerun = folder.parent.parent / "rerun" / "A-rqvae-t0007" / "seed_7"

    expected = _expected_recommender_config(rerun, "recommender.json")
    expected["general"].update(wandb_group="A-rqvae-t0007", wandb_run_name="A-rqvae-t0007-seed_7",
                               wandb_tags=["confirm", "rqvae", "eager"])

    spec = dataclasses.replace(SPEC, phase="confirm")

    assert _load(stages.recommender_config(spec, rerun, compile=False)) == expected


@pytest.mark.parametrize("key", ["general.wandb_group", "general.wandb_run_name", "general.wandb_tags",
                                 "general.wandb_project"])
def test_recommender_override_of_a_wandb_name_is_rejected(recommender_mocks, key):
    # the driver names the run; a spec cannot claim another name than the one W&B shows
    folder, _ = recommender_mocks
    with pytest.raises(ValueError) as excinfo:
        stages.recommender_config(dataclasses.replace(SPEC, recommender_overrides={key: "x"}), folder, compile=False)
    assert str(excinfo.value) == f"recommender_overrides must not set driver keys: ['{key}']"


def test_test_evaluates_the_best_checkpoint_with_the_trained_config(recommender_mocks):
    # the model is tested exactly as it was trained; only the meta file differs
    folder, mocks = recommender_mocks
    result = _recommender(SPEC, folder)
    stages.test(SPEC, folder, result.best_checkpoint)
    assert mocks["test"].call_args_list == [call(str(folder / "configs" / "test.yaml"), str(folder / "semids.pt"),
                                                 model_path=f"{folder}/checkpoints/recommender_best.pt")]
    assert _load(folder / "configs" / "test.yaml") == _expected_recommender_config(folder, "test.json")


def test_test_with_semids_from_reads_the_source_trials_semids(recommender_mocks):
    # a recommender-only rerun is tested on the SIDs it was trained on
    folder, mocks = recommender_mocks
    spec = dataclasses.replace(SPEC, quantizer_overrides={}, semids_from=SOURCE_TRIAL)
    result = _recommender(spec, folder)
    stages.test(spec, folder, result.best_checkpoint)
    assert mocks["test"].call_args_list == [call(str(folder / "configs" / "test.yaml"), f"{SOURCE_TRIAL}/semids.pt",
                                                 model_path=f"{folder}/checkpoints/recommender_best.pt")]


def test_test_returns_the_test_meta(recommender_mocks):
    folder, _ = recommender_mocks
    result = _recommender(SPEC, folder)
    assert stages.test(SPEC, folder, result.best_checkpoint) == TEST_META


def test_recommender_function_that_writes_no_meta_is_an_error(recommender_mocks):
    # a stage without its meta must not count as done
    folder, mocks = recommender_mocks
    mocks["train"].side_effect = None
    with pytest.raises(RuntimeError) as excinfo:
        _recommender(SPEC, folder)
    meta = folder / 'meta' / 'recommender.json'
    assert str(excinfo.value) == f"the stage finished without writing its meta file {meta}"


def test_test_function_that_writes_no_meta_is_an_error(recommender_mocks):
    folder, mocks = recommender_mocks
    result = _recommender(SPEC, folder)
    mocks["test"].side_effect = None
    with pytest.raises(RuntimeError) as excinfo:
        stages.test(SPEC, folder, result.best_checkpoint)
    assert str(excinfo.value) == f"the stage finished without writing its meta file {folder / 'meta' / 'test.json'}"
