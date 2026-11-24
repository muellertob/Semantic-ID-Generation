import pytest
from unittest.mock import MagicMock, call, patch
from omegaconf import OmegaConf
from utils.wandb import log_model_artifact, prefix_run_name, wandb_init

@patch("utils.wandb.wandb")
def test_log_model_artifact_no_run(mock_wandb):
    # If wandb.run is None, it should do nothing
    mock_wandb.run = None
    log_model_artifact("dummy.pt", "dummy_run", "dummy_type")
    mock_wandb.Artifact.assert_not_called()
    mock_wandb.log_artifact.assert_not_called()

@patch("utils.wandb.wandb")
def test_log_model_artifact_default_false_no_config(mock_wandb):
    # If wandb.run is active, but config is not passed, it defaults to False (no logging)
    mock_wandb.run = MagicMock()
    log_model_artifact("dummy.pt", "dummy_run", "dummy_type")
    mock_wandb.Artifact.assert_not_called()
    mock_wandb.log_artifact.assert_not_called()

@patch("utils.wandb.wandb")
def test_log_model_artifact_config_false(mock_wandb):
    # If wandb_log_artifacts is False in config, it should not log
    mock_wandb.run = MagicMock()
    config = OmegaConf.create({
        "general": {
            "wandb_log_artifacts": False
        }
    })
    log_model_artifact("dummy.pt", "dummy_run", "dummy_type", config=config)
    mock_wandb.Artifact.assert_not_called()
    mock_wandb.log_artifact.assert_not_called()

@patch("utils.wandb.wandb")
def test_log_model_artifact_config_missing_param(mock_wandb):
    # If wandb_log_artifacts is omitted in config, it should not log (defaults to False)
    mock_wandb.run = MagicMock()
    config = OmegaConf.create({
        "general": {
            "use_wandb": True
        }
    })
    log_model_artifact("dummy.pt", "dummy_run", "dummy_type", config=config)
    mock_wandb.Artifact.assert_not_called()
    mock_wandb.log_artifact.assert_not_called()

@patch("utils.wandb.wandb")
def test_log_model_artifact_config_true(mock_wandb):
    # If wandb_log_artifacts is True in config, it should log
    mock_wandb.run = MagicMock()
    config = OmegaConf.create({
        "general": {
            "wandb_log_artifacts": True
        }
    })
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact

    log_model_artifact("dummy.pt", "dummy_run", "dummy_type", config=config)

    mock_wandb.Artifact.assert_called_once_with(
        name="dummy_run",
        type="dummy_type",
        metadata={},
    )
    mock_artifact.add_file.assert_called_once_with("dummy.pt")
    mock_wandb.log_artifact.assert_called_once_with(mock_artifact)


# ---- run names from the config (search driver) ---------------------------------------------------------------

def _config(tmp_path, **names):
    return OmegaConf.create({"general": {"wandb_entity": "team", "wandb_dir": str(tmp_path), **names}})


def _init_call(config, project, name, group, tags):
    return call(project=project, name=name, group=group, job_type="seq2seq-train", tags=tags, entity="team",
                config=OmegaConf.to_container(config, resolve=True), resume="never", dir=config.general.wandb_dir)


# | keys set in the config     | project                | name               | group    | tags     |
# |----------------------------|------------------------|--------------------|----------|----------|
# | none                       | caller's               | caller's           | caller's | caller's |
# | project, name, group, tags | sid-search-toys-stage2 | A-rqvae-trial_0007 | A-rqvae  | config's |
# | group only                 | caller's               | caller's           | A-rqvae  | caller's |
@patch("utils.wandb.wandb")
def test_without_name_keys_the_callers_names_are_used(mock_wandb, tmp_path):
    # run_experiments.py and the existing configs set none of the keys, so their runs keep their names
    config = _config(tmp_path)

    wandb_init(config, "proj", run_name="RQ-VAE-seed-42-ab12", group="RQ-VAE", job_type="seq2seq-train",
               tags=["RQ-VAE", "seed-42"])

    assert mock_wandb.init.call_args_list == [_init_call(config, "proj", "RQ-VAE-seed-42-ab12", "RQ-VAE",
                                                         ["RQ-VAE", "seed-42"])]


@patch("utils.wandb.wandb")
def test_name_keys_in_the_config_win_over_the_callers_names(mock_wandb, tmp_path):
    config = _config(tmp_path, wandb_project="sid-search-toys-stage2", wandb_run_name="A-rqvae-trial_0007",
                     wandb_group="A-rqvae", wandb_tags=["A", "rqvae", "compile"])

    wandb_init(config, "proj", run_name="RQ-VAE-seed-42-ab12", group="RQ-VAE", job_type="seq2seq-train",
               tags=["RQ-VAE", "seed-42"])

    assert mock_wandb.init.call_args_list == [_init_call(config, "sid-search-toys-stage2",
                                                         "A-rqvae-trial_0007", "A-rqvae",
                                                         ["A", "rqvae", "compile"])]


@patch("utils.wandb.wandb")
def test_each_name_key_is_taken_on_its_own(mock_wandb, tmp_path):
    config = _config(tmp_path, wandb_group="A-rqvae")

    wandb_init(config, "proj", run_name="RQ-VAE-seed-42-ab12", group="RQ-VAE", job_type="seq2seq-train",
               tags=["RQ-VAE", "seed-42"])

    assert mock_wandb.init.call_args_list == [_init_call(config, "proj", "RQ-VAE-seed-42-ab12", "A-rqvae",
                                                         ["RQ-VAE", "seed-42"])]


# | run name before         | wandb_run_name in config | run name after        | why               |
# |-------------------------|--------------------------|-----------------------|-------------------|
# | sunny-energy-67 (W&B's) | -                        | rqvae-sunny-energy-67 | today's behaviour |
# | rqvae-sunny             | -                        | rqvae-sunny           | already prefixed  |
# | A-rqvae-trial_0007      | A-rqvae-trial_0007       | A-rqvae-trial_0007    | the driver's name |
@pytest.mark.parametrize("name, names, expected", [
    ("sunny-energy-67", {}, "rqvae-sunny-energy-67"),
    ("rqvae-sunny", {}, "rqvae-sunny"),
    ("A-rqvae-trial_0007", {"wandb_run_name": "A-rqvae-trial_0007"}, "A-rqvae-trial_0007"),
], ids=["generated", "already-prefixed", "named-by-config"])
@patch("utils.wandb.wandb")
def test_the_quantizer_prefix_is_added_only_to_a_generated_name(mock_wandb, tmp_path, name, names, expected):
    mock_wandb.run = MagicMock()
    mock_wandb.run.name = name

    prefix_run_name(_config(tmp_path, **names), "rqvae")

    assert mock_wandb.run.name == expected
