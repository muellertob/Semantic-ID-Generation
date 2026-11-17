"""
Integration tests for the experiment runner (run_experiments.py): the run configs it writes per variant.
"""
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest

import run_experiments
from run_experiments import (GTS_EXPERIMENTS, LOO_EXPERIMENTS, ONION_GTS_EXPERIMENTS, prepare_generate_task,
                             prepare_quantizer_task, run_recommender_task, select_experiments)

MISSING = "MISSING"


def flatten(config):
    return {f"{section}.{key}": value for section, keys in config.items() for key, value in keys.items()}


def flat_diff(toys, onion):
    """{key: (toys value, onion value)} for every flat key whose value differs or that only one dict has."""
    return {key: (toys.get(key, MISSING), onion.get(key, MISSING))
            for key in toys.keys() | onion.keys() if toys.get(key, MISSING) != onion.get(key, MISSING)}


def config_diff(toys_path, onion_path):
    """flat_diff of two yaml files."""
    return flat_diff(flatten(OmegaConf.to_container(OmegaConf.load(toys_path))),
                     flatten(OmegaConf.to_container(OmegaConf.load(onion_path))))


@pytest.fixture
def written_config(tmp_path, monkeypatch):
    """Writes the seed-42 run config of a GTS variant into tmp_path and returns it as a flat dict.

    `stage` is "quantizer" (default), "generate" or "recommender".
    """
    for name, sub in [("OUTPUTS_DIR", "out"), ("MODELS_DIR", "models"),
                      ("EXPERIMENT_DIR", "out/experiments"), ("CONFIGS_DIR", "out/experiments/configs")]:
        monkeypatch.setattr(run_experiments, name, str(tmp_path / sub))

    def write(variant, stage="quantizer", experiments=GTS_EXPERIMENTS):
        exp = next(e for e in experiments if e["name"] == variant)
        if stage == "quantizer":
            cmd, _, _, _ = prepare_quantizer_task(exp, 42, dry_run=False)
            config_path = cmd[-1]
        elif stage == "generate":
            cmd, _ = prepare_generate_task(exp, 42, "quantizer.pt", "semids.pt")
            config_path = cmd[cmd.index("--config") + 1]
        else:
            # stop before the test step: the train command is the first one run
            with patch("run_experiments.run_command", side_effect=StopIteration) as run_command:
                with pytest.raises(StopIteration):
                    run_recommender_task(exp, 42, "semids.pt", dry_run=False)
            config_path = run_command.call_args.args[0][run_command.call_args.args[0].index("--config") + 1]
        config = OmegaConf.to_container(OmegaConf.load(config_path))
        return flatten(config)
    return write


@pytest.mark.parametrize("variant, commit_all_levels", [("RQ_VAE", False), ("RQ_VAE_cal", True)])
def test_gts_rqvae_variants_set_commit_all_levels(written_config, variant, commit_all_levels):
    # RQ_VAE keeps the residual update of the GRID and RQ-VAE-Recommender code; RQ_VAE_cal enables commit_all_levels
    config = written_config(variant)
    assert (config["model.commit_all_levels"], config["general.seed"]) == (commit_all_levels, 42)


def test_rqvae_cal_differs_from_rqvae_only_in_the_option_and_its_paths(written_config):
    # a paired comparison: nothing else may change with the variant
    old, new = written_config("RQ_VAE"), written_config("RQ_VAE_cal")
    assert set(old) == set(new)
    assert {key for key in old if old[key] != new[key]} == {
        "model.commit_all_levels", "general.model_path", "general.meta_json"}


def test_rqvae_center_differs_from_rqvae_only_in_the_input_options_and_its_paths(written_config):
    # centered input without the model-internal BatchNorm + L2 is the only change to the baseline
    old, new = written_config("RQ_VAE"), written_config("RQ_VAE_center")
    assert set(old) == set(new)
    assert {key for key in old if old[key] != new[key]} == {
        "data.input_normalization", "model.normalize_inputs", "general.model_path", "general.meta_json"}


def test_rqkmeans_std_differs_from_rqkmeans_only_in_the_input_normalization_and_its_paths(written_config):
    # RQ-KMeans also writes general.output_path (its SIDs folder is named after the variant)
    old, new = written_config("RQ_KMeans"), written_config("RQ_KMeans_std")
    assert set(old) == set(new)
    assert {key for key in old if old[key] != new[key]} == {
        "data.input_normalization", "general.model_path", "general.meta_json", "general.output_path"}


def test_rqvae_center_generate_config_carries_the_variant_overrides(written_config):
    # SIDs must be generated with the input setting the quantizer was trained with
    config = written_config("RQ_VAE_center", "generate")
    assert (config["data.input_normalization"], config["model.normalize_inputs"]) == ("center", False)


def test_rqvae_center_generate_config_has_the_data_and_model_sections_of_the_quantizer_config(written_config):
    train, generate = written_config("RQ_VAE_center"), written_config("RQ_VAE_center", "generate")
    section = lambda config, name: {k: v for k, v in config.items() if k.startswith(name + ".")}
    assert (section(generate, "data"), section(generate, "model")) == (section(train, "data"), section(train, "model"))


def test_rqvae_generate_config_keeps_the_base_input_settings(written_config):
    config = written_config("RQ_VAE", "generate")
    assert (config["data.input_normalization"], config["model.normalize_inputs"]) == ("none", True)


@pytest.mark.parametrize("variant", ["RQ_VAE_cal", "FSQ_20k", "RFSQ_Fixed_866_qd05"])
def test_recommender_train_config_names_the_run_after_the_variant(written_config, variant):
    # each variant shares the sid_type with its base (RQ_VAE, FSQ, R-FSQ_Fixed), so only the variant name keeps
    # their checkpoints apart
    assert written_config(variant, "recommender")["general.run_name"] == f"{variant}_seed_42"


@pytest.mark.parametrize("variant, base", [("RQ_VAE_20k", "RQ_VAE"), ("RQ_VAE_cal_20k", "RQ_VAE_cal"),
                                           ("RQ_VAE_center_20k", "RQ_VAE_center"), ("RQ_VAE_b025_20k", "RQ_VAE_b025"),
                                           ("RQ_VAE_cal_b2_20k", "RQ_VAE_cal_b2"), ("FSQ_20k", "FSQ"),
                                           ("Residual_FSQ_Simple_HighDim_20k", "Residual_FSQ_Simple_HighDim"),
                                           ("Residual_FSQ_Simple_LowDim_20k", "Residual_FSQ_Simple_LowDim"),
                                           ("Residual_FSQ_Complex_20k", "Residual_FSQ_Complex")])
def test_20k_variants_differ_from_their_base_only_in_the_epochs_and_paths(written_config, variant, base):
    # paired with the 2000-epoch runs of the same seed: the Stage-1 length is the only change
    old, new = written_config(base), written_config(variant)
    assert set(old) == set(new)
    assert {key for key in old if old[key] != new[key]} == {
        "train.num_epochs", "general.model_path", "general.meta_json"}
    assert new["train.num_epochs"] == 20000


@pytest.mark.parametrize("variant, base, beta", [("RQ_VAE_b025", "RQ_VAE", 0.25), ("RQ_VAE_cal_b2", "RQ_VAE_cal", 2.0)])
def test_beta_variants_differ_from_their_base_only_in_beta_and_paths(written_config, variant, base, beta):
    # b025: TIGER's printed 0.25 instead of 6; cal_b2: 6 split over the 3 levels that reach the encoder
    old, new = written_config(base), written_config(variant)
    assert set(old) == set(new)
    assert {key for key in old if old[key] != new[key]} == {"model.beta", "general.model_path", "general.meta_json"}
    assert new["model.beta"] == beta


def test_rqvae_cal_b025_20k_differs_from_rqvae_b025_20k_only_in_commit_all_levels_and_paths(written_config):
    # TIGER's printed loss sums the commitment over all levels with beta 0.25; its pair is the default beta 0.25, 20k
    old, new = written_config("RQ_VAE_b025_20k"), written_config("RQ_VAE_cal_b025_20k")
    assert set(old) == set(new)
    assert {key for key in old if old[key] != new[key]} == {
        "model.commit_all_levels", "general.model_path", "general.meta_json"}
    assert new["model.commit_all_levels"] is True


@pytest.mark.parametrize("variant, base", [("RQ_VAE_b025_lr001_20k", "RQ_VAE_b025_20k"),
                                           ("RQ_VAE_cal_b025_lr001_20k", "RQ_VAE_cal_b025_20k")])
def test_lr001_variants_differ_from_their_base_only_in_the_learning_rate_and_paths(written_config, variant, base):
    # paired with the lr 1e-3 runs of the same seed: Adagrad's learning rate is the only change
    old, new = written_config(base), written_config(variant)
    assert set(old) == set(new)
    assert {key for key in old if old[key] != new[key]} == {
        "train.learning_rate", "general.model_path", "general.meta_json"}
    assert (old["train.learning_rate"], new["train.learning_rate"]) == (0.001, 0.01)


@pytest.mark.parametrize("grid, level_list", [("866", [8, 6, 6]), ("4444", [4, 4, 4, 4]), ("2x8", [2] * 8)])
@pytest.mark.parametrize("suffix", ["", "_qd05", "_qd1"])
def test_rfsq_fixed_variants_write_the_grid_of_their_name_with_one_latent_dimension_per_channel(
        written_config, grid, level_list, suffix):
    # the quantizer has no projection, so the latent dimension is the number of channels: 3, 4 and 8
    config = written_config(f"RFSQ_Fixed_{grid}{suffix}")
    assert (config["model.level_list"], config["model.latent_dimension"]) == (level_list, len(level_list))


@pytest.mark.parametrize("suffix, quantizer_dropout", [("", 0.0), ("_qd05", 0.5), ("_qd1", 1.0)])
@pytest.mark.parametrize("grid", ["866", "4444", "2x8"])
def test_rfsq_fixed_variants_write_the_dropout_probability_of_their_suffix(
        written_config, grid, suffix, quantizer_dropout):
    # no suffix trains without quantizer dropout; qd05 and qd1 stand for the probabilities 0.5 and 1
    assert written_config(f"RFSQ_Fixed_{grid}{suffix}")["model.quantizer_dropout"] == quantizer_dropout


@pytest.mark.parametrize("variant, base", [
    ("RFSQ_Fixed_866_qd05", "RFSQ_Fixed_866"), ("RFSQ_Fixed_866_qd1", "RFSQ_Fixed_866"),
    ("RFSQ_Fixed_4444_qd05", "RFSQ_Fixed_4444"), ("RFSQ_Fixed_4444_qd1", "RFSQ_Fixed_4444"),
    ("RFSQ_Fixed_2x8_qd05", "RFSQ_Fixed_2x8"), ("RFSQ_Fixed_2x8_qd1", "RFSQ_Fixed_2x8")])
def test_rfsq_fixed_dropout_variants_differ_from_their_grid_only_in_the_dropout_and_paths(
        written_config, variant, base):
    # a paired comparison: nothing else may change
    old, new = written_config(base), written_config(variant)
    assert set(old) == set(new)
    assert {key for key in old if old[key] != new[key]} == {
        "model.quantizer_dropout", "general.model_path", "general.meta_json"}


@pytest.mark.parametrize("variant", ["RFSQ_Fixed_4444", "RFSQ_Fixed_2x8"])
def test_rfsq_fixed_grids_differ_from_866_only_in_the_grid_latent_dimension_and_paths(written_config, variant):
    # the grids are compared at equal quantizer settings
    old, new = written_config("RFSQ_Fixed_866"), written_config(variant)
    assert set(old) == set(new)
    assert {key for key in old if old[key] != new[key]} == {
        "model.level_list", "model.latent_dimension", "general.model_path", "general.meta_json"}


@pytest.mark.parametrize("variant, codebook_size", [("RFSQ_Fixed_866", 288), ("RFSQ_Fixed_4444", 256),
                                                    ("RFSQ_Fixed_2x8", 256)])
def test_rfsq_fixed_recommender_config_has_as_many_values_per_level_as_the_grid(
        written_config, variant, codebook_size):
    # 8 * 6 * 6 = 288 and 4^4 = 2^8 = 256; Semantic IDs outside the Stage-2 vocabulary are rejected at the start of
    # training, and a vocabulary that is too large would leave tokens unused
    assert written_config(variant, "recommender")["seq2seq.codebook_size"] == codebook_size


def test_rfsq_fixed_generate_config_has_the_data_and_model_sections_of_the_quantizer_config(written_config):
    # the Semantic IDs must be generated with the grid the quantizer was trained with
    train, generate = written_config("RFSQ_Fixed_4444_qd05"), written_config("RFSQ_Fixed_4444_qd05", "generate")
    section = lambda config, name: {k: v for k, v in config.items() if k.startswith(name + ".")}
    assert (section(generate, "data"), section(generate, "model")) == (section(train, "data"), section(train, "model"))


DATASET_DIFF = {"data.dataset": ("amazon", "music4all_onion"), "data.category": ("toys", MISSING)}
QUANTIZER_DIFF = {**DATASET_DIFF, "general.wandb_project_quantizer": ("mt-amazon-quantizer", "mt-onion-quantizer")}
RECOMMENDER_DIFF = {**DATASET_DIFF, "seq2seq.num_epochs": (1471, 1961), "seq2seq.eval_delay_epochs": (150, 200),
                    "general.wandb_project_recommender": ("mt-amazon-recommender", "mt-onion-recommender"),
                    "general.wandb_project_quantizer": (MISSING, "mt-onion-quantizer")}


def test_onion_rqkmeans_config_differs_from_toys_only_in_the_dataset():
    # every Stage-1 setting stays the Toys reference; Onion has no category
    assert config_diff("config/rq-kmeans/amazon_toy_rqkmeans_gts.yaml",
                       "config/rq-kmeans/onion_rqkmeans_gts.yaml") == DATASET_DIFF


def test_onion_fsq_config_differs_from_toys_only_in_the_dataset_and_wandb_project():
    # Onion runs must not mix into the Toys quantizer curves
    assert config_diff("config/fsq/amazon_toy_gts.yaml", "config/fsq/onion_gts.yaml") == QUANTIZER_DIFF


def test_onion_rqvae_config_differs_from_toys_only_in_the_dataset_and_wandb_project():
    # same reason as for FSQ
    assert config_diff("config/rq-vae/amazon_toy_tiger_gts.yaml", "config/rq-vae/onion_tiger_gts.yaml") == QUANTIZER_DIFF


def test_onion_recommender_config_differs_from_toys_only_in_dataset_budget_and_wandb_projects():
    # 100k steps at 51 steps per epoch, first evaluation after 10,200 steps; the quantizer project links the parent run
    assert config_diff("config/recommender/amazon_toy_grid_gts.yaml",
                       "config/recommender/onion_grid_gts.yaml") == RECOMMENDER_DIFF


def test_onion_fsq_recommender_config_differs_from_toys_in_the_same_keys():
    # like the Toys pair, it differs from the other recommender file only in the codebook size
    assert config_diff("config/recommender/amazon_toy_grid_fsq_gts.yaml",
                       "config/recommender/onion_grid_fsq_gts.yaml") == RECOMMENDER_DIFF


def test_onion_experiments_are_the_toys_reference_variants_on_the_onion_configs():
    # the Stage-1 defaults live in one place, the Toys variants
    toys = {e["name"]: e for e in GTS_EXPERIMENTS}
    assert [e["name"] for e in ONION_GTS_EXPERIMENTS] == ["RQ_VAE_cal_b025_lr001_20k", "FSQ_20k", "RQ_KMeans"]
    assert ONION_GTS_EXPERIMENTS == [
        {**toys["RQ_VAE_cal_b025_lr001_20k"], "quant_config": "config/rq-vae/onion_tiger_gts.yaml",
         "recommender_config": "config/recommender/onion_grid_gts.yaml"},
        {**toys["FSQ_20k"], "quant_config": "config/fsq/onion_gts.yaml",
         "recommender_config": "config/recommender/onion_grid_fsq_gts.yaml"},
        {**toys["RQ_KMeans"], "quant_config": "config/rq-kmeans/onion_rqkmeans_gts.yaml",
         "recommender_config": "config/recommender/onion_grid_gts.yaml"},
    ]


@pytest.mark.parametrize("variant, expected_diff", [("RQ_VAE_cal_b025_lr001_20k", QUANTIZER_DIFF),
                                                    ("FSQ_20k", QUANTIZER_DIFF), ("RQ_KMeans", DATASET_DIFF)])
def test_written_onion_quantizer_config_differs_from_toys_only_in_the_dataset_and_wandb_project(
        written_config, variant, expected_diff):
    # what actually runs: the variant's overrides reach the Onion run
    toys = written_config(variant)
    onion = written_config(variant, experiments=ONION_GTS_EXPERIMENTS)
    assert flat_diff(toys, onion) == expected_diff


def test_select_experiments_returns_the_onion_list_for_the_onion_dataset():
    # the new option reaches the new list
    assert select_experiments("music4all_onion", "gts", None) is ONION_GTS_EXPERIMENTS


def test_select_experiments_keeps_the_toys_lists():
    # the default runs do not change
    assert select_experiments("toys", "gts", None) is GTS_EXPERIMENTS
    assert select_experiments("toys", "loo", None) is LOO_EXPERIMENTS


def test_select_experiments_rejects_the_onion_dataset_under_leave_one_out():
    # Onion has only the global temporal split
    with pytest.raises(ValueError, match="only under --protocol gts"):
        select_experiments("music4all_onion", "loo", None)


def test_select_experiments_filters_variants_case_insensitively_in_the_given_order():
    by_name = {e["name"]: e for e in ONION_GTS_EXPERIMENTS}
    assert select_experiments("music4all_onion", "gts", "rq_kmeans,FSQ_20k") == [by_name["RQ_KMeans"], by_name["FSQ_20k"]]


def test_select_experiments_rejects_a_toys_only_variant_for_the_onion_dataset():
    # a Toys-only variant must not run with Toys configs in an Onion output folder
    with pytest.raises(ValueError, match=r"unknown \['RQ_VAE'\].*RQ_VAE_cal_b025_lr001_20k"):
        select_experiments("music4all_onion", "gts", "RQ_VAE")
