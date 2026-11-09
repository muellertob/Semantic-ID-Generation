"""
Integration tests for the experiment runner (run_experiments.py): the run configs it writes per variant.
"""
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest

import run_experiments
from run_experiments import GTS_EXPERIMENTS, prepare_generate_task, prepare_quantizer_task, run_recommender_task


@pytest.fixture
def written_config(tmp_path, monkeypatch):
    """Writes the seed-42 run config of a GTS variant into tmp_path and returns it as a flat dict.

    `stage` is "quantizer" (default), "generate" or "recommender".
    """
    for name, sub in [("OUTPUTS_DIR", "out"), ("MODELS_DIR", "models"),
                      ("EXPERIMENT_DIR", "out/experiments"), ("CONFIGS_DIR", "out/experiments/configs")]:
        monkeypatch.setattr(run_experiments, name, str(tmp_path / sub))

    def write(variant, stage="quantizer"):
        exp = next(e for e in GTS_EXPERIMENTS if e["name"] == variant)
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
        return {f"{section}.{key}": value for section, keys in config.items() for key, value in keys.items()}
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


@pytest.mark.parametrize("variant", ["RQ_VAE_cal", "FSQ_20k"])
def test_recommender_train_config_names_the_run_after_the_variant(written_config, variant):
    # each variant shares the sid_type with its base (RQ_VAE, FSQ), so only the variant name keeps their checkpoints apart
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
