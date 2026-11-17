"""
Unit tests for data/factory.py: load_data and load_sequences.
"""
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from data.factory import load_data, load_sequences
from data.protocols import SplitConfig

# Quantiles 0.8 / 0.7 are not the defaults (0.9), so a dispatch that ignores
# the config's split settings produces a different SplitConfig and fails.
ONION = {"data": {"dataset": "music4all_onion", "input_normalization": "none",
                  "split_protocol": "gts", "gts_q_test": 0.8, "gts_q_val": 0.7}}
AMAZON = {"data": {**ONION["data"], "dataset": "amazon", "category": "toys"}}
GTS_SPLIT = SplitConfig("gts", 0.8, 0.7)


def test_removed_normalize_data_key_raises_before_loading():
    config = OmegaConf.create({"data": {"dataset": "amazon", "category": "toys", "normalize_data": True}})
    with patch("data.factory.load_amazon") as load_amazon:
        with pytest.raises(ValueError, match="input_normalization"):
            load_data(config)
    load_amazon.assert_not_called()


def test_load_data_sends_onion_config_to_load_onion_with_the_run_split_settings():
    # Stage 1 and SID generation read the items through load_data.
    config = OmegaConf.create(ONION)
    items = torch.zeros(3, 2)
    with patch("data.factory.load_onion", return_value=items) as load_onion, \
            patch("data.factory.load_amazon") as load_amazon:
        result = load_data(config, split="all")
    load_onion.assert_called_once_with(split_config=GTS_SPLIT, input_normalization="none", split="all")
    load_amazon.assert_not_called()
    assert result is items


def test_load_data_still_sends_amazon_config_to_load_amazon():
    # The new Onion branch must not change the Toys runs.
    config = OmegaConf.create(AMAZON)
    items = torch.zeros(3, 2)
    with patch("data.factory.load_amazon", return_value=items) as load_amazon, \
            patch("data.factory.load_onion") as load_onion:
        result = load_data(config, split="all")
    load_amazon.assert_called_once_with(category="toys", split_config=GTS_SPLIT,
                                        input_normalization="none", split="all")
    load_onion.assert_not_called()
    assert result is items


def test_load_sequences_sends_onion_config_to_load_onion_sequences():
    # The Stage-2 scripts get the Onion rows without a category.
    config = OmegaConf.create(ONION)
    with patch("data.factory.load_onion_sequences", return_value=("seqs", 2, 3)) as load_onion_sequences, \
            patch("data.factory.load_amazon_sequences") as load_amazon_sequences:
        result = load_sequences(config)
    assert result == ("seqs", 2, 3)
    load_onion_sequences.assert_called_once_with(GTS_SPLIT)
    load_amazon_sequences.assert_not_called()


def test_load_sequences_sends_amazon_config_to_load_amazon_sequences():
    # Same call the Stage-2 scripts made before the dispatch existed.
    config = OmegaConf.create(AMAZON)
    with patch("data.factory.load_amazon_sequences", return_value=("seqs", 2, 3)) as load_amazon_sequences, \
            patch("data.factory.load_onion_sequences") as load_onion_sequences:
        result = load_sequences(config)
    assert result == ("seqs", 2, 3)
    load_amazon_sequences.assert_called_once_with("toys", GTS_SPLIT)
    load_onion_sequences.assert_not_called()


def test_load_sequences_raises_for_unknown_dataset_instead_of_falling_back():
    # A typo in a config must not silently load Amazon data.
    config = OmegaConf.create({"data": {"dataset": "lastfm"}})
    with patch("data.factory.load_amazon_sequences") as load_amazon_sequences, \
            patch("data.factory.load_onion_sequences") as load_onion_sequences:
        with pytest.raises(ValueError, match="lastfm"):
            load_sequences(config)
    load_amazon_sequences.assert_not_called()
    load_onion_sequences.assert_not_called()
