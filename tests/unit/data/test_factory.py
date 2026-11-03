"""
Unit tests for data/factory.py: load_data.
"""
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from data.factory import load_data


def test_removed_normalize_data_key_raises_before_loading():
    config = OmegaConf.create({"data": {"dataset": "amazon", "category": "toys", "normalize_data": True}})
    with patch("data.factory.load_amazon") as load_amazon:
        with pytest.raises(ValueError, match="input_normalization"):
            load_data(config)
    load_amazon.assert_not_called()
