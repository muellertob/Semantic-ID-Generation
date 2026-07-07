import pytest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
from utils.wandb import log_model_artifact

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
