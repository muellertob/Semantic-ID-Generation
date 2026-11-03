"""
Integration tests for the experiment runner (run_experiments.py): the run configs it writes per variant.
"""
from omegaconf import OmegaConf
import pytest

import run_experiments
from run_experiments import GTS_EXPERIMENTS, prepare_quantizer_task


@pytest.fixture
def written_config(tmp_path, monkeypatch):
    """Writes the seed-42 quantizer run config of a GTS variant into tmp_path and returns it as a flat dict."""
    for name, sub in [("OUTPUTS_DIR", "out"), ("MODELS_DIR", "models"),
                      ("EXPERIMENT_DIR", "out/experiments"), ("CONFIGS_DIR", "out/experiments/configs")]:
        monkeypatch.setattr(run_experiments, name, str(tmp_path / sub))

    def write(variant):
        exp = next(e for e in GTS_EXPERIMENTS if e["name"] == variant)
        cmd, _, _, _ = prepare_quantizer_task(exp, 42, dry_run=False)
        config = OmegaConf.to_container(OmegaConf.load(cmd[-1]))
        return {f"{section}.{key}": value for section, keys in config.items() for key, value in keys.items()}
    return write


@pytest.mark.parametrize("variant, commit_all_levels", [("RQ_VAE", False), ("RQ_VAE_cal", True)])
def test_gts_rqvae_variants_set_commit_all_levels(written_config, variant, commit_all_levels):
    # RQ_VAE keeps TIGER's residual update; RQ_VAE_cal enables commit_all_levels
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
