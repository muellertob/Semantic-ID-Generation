import pytest
from unittest.mock import patch
from omegaconf import OmegaConf
import torch
import os

def test_seq2seq_training_calls_set_seed(tmp_path):
    import train_seq2seq
    
    config = {
        "data": {"dataset": "amazon", "category": "beauty"},
        "seq2seq": {
            "d_model": 16, "num_layers": 1, "num_heads": 2,
            "learning_rate": 0.001, "batch_size": 2,
            "resume_optimizer": False, "early_stopping": False
        },
        "general": {"seed": 99, "use_wandb": False}
    }
    config_path = str(tmp_path / "seq2seq_config.yaml")
    OmegaConf.save(OmegaConf.create(config), config_path)
    
    semids_path = str(tmp_path / "semids.pt")
    # Write a dummy file to satisfy os.path.exists
    with open(semids_path, "w") as f:
        f.write("dummy")
    
    # We patch torch.load (which is the first file operation after seed setup)
    # to raise a custom exception to interrupt training right after seeding.
    with patch("train_seq2seq.set_seed") as mock_set_seed, \
         patch("train_seq2seq.torch.load", side_effect=ValueError("StopAfterSeeding")):
        
        with pytest.raises(ValueError, match="StopAfterSeeding"):
            train_seq2seq.run_training(config_path, semids_path)
            
        mock_set_seed.assert_called_with(99)

def test_quantizer_training_calls_set_seed(tmp_path):
    import train_quantizer
    
    config = {
        "train": {},
        "model": {"quantizer_type": "fsq"},
        "data": {"dataset": "amazon", "normalize_data": False},
        "general": {"seed": 123, "use_wandb": False}
    }
    config_path = str(tmp_path / "quantizer_config.yaml")
    OmegaConf.save(OmegaConf.create(config), config_path)
    
    with patch("train_quantizer.set_seed") as mock_set_seed, \
         patch("train_quantizer.load_data", side_effect=ValueError("StopAfterSeeding")):
         
        with pytest.raises(ValueError, match="StopAfterSeeding"):
            train_quantizer.run_training(config_path)
            
        mock_set_seed.assert_called_with(123)

def test_sasrec_training_calls_set_seed(tmp_path):
    import train_sasrec
    
    config = {
        "sasrec": {
            "hidden_dim": 8, "num_blocks": 1, "num_heads": 1, "dropout": 0.1,
            "max_seq_len": 5, "learning_rate": 0.001, "batch_size": 2, "num_epochs": 1
        },
        "data": {"dataset": "amazon", "category": "beauty"},
        "general": {"seed": 456, "use_wandb": False}
    }
    config_path = str(tmp_path / "sasrec_config.yaml")
    OmegaConf.save(OmegaConf.create(config), config_path)
    
    with patch("train_sasrec.set_seed") as mock_set_seed, \
         patch("train_sasrec.load_amazon_sequences", side_effect=ValueError("StopAfterSeeding")):
         
        with pytest.raises(ValueError, match="StopAfterSeeding"):
            train_sasrec.run_training(config_path)
            
        mock_set_seed.assert_called_with(456)

def test_rqkmeans_training_calls_set_seed(tmp_path):
    import train_rqkmeans
    
    config = {
        "model": {"codebook_layers": 2, "codebook_size": 4, "n_iters": 5, "normalize_residuals": False},
        "general": {"seed": 789},
        "data": {"dataset": "amazon", "normalize_data": False}
    }
    config_path = str(tmp_path / "rqkmeans_config.yaml")
    OmegaConf.save(OmegaConf.create(config), config_path)
    
    with patch("train_rqkmeans.set_seed") as mock_set_seed, \
         patch("train_rqkmeans.load_data", side_effect=ValueError("StopAfterSeeding")):
         
        with pytest.raises(ValueError, match="StopAfterSeeding"):
            train_rqkmeans.run_training(config_path)
            
        mock_set_seed.assert_called_with(789)
