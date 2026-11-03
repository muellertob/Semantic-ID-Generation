"""
Unit tests for data/semids_checks.py: a semids file must belong to the dataset and split
protocol of the recommender config.
"""
import pytest
import torch
from omegaconf import OmegaConf

from data.semids_checks import check_semids, check_semids_dataset, check_semids_protocol


def semids(num_rows=10, **data):
    data = {"dataset": "amazon", "category": "toys", **data}
    return {"semantic_ids": torch.zeros(num_rows, 4, dtype=torch.long),
            "config": OmegaConf.create({"data": data})}


def cfg(**data):
    return OmegaConf.create({"data": {"dataset": "amazon", "category": "toys", **data}})


# semids files without a usable Stage-1 config: no config at all, or a config without a data block
NO_STAGE1_CONFIG = pytest.mark.parametrize("semids_data", [
    {"semantic_ids": torch.zeros(10, 4)},
    {"semantic_ids": torch.zeros(10, 4), "config": OmegaConf.create({})},
], ids=["no_config", "no_data_block"])


class TestCheckSemidsDataset:
    def test_matching_file_passes(self):
        check_semids_dataset(semids(), cfg(), num_items=10)

    def test_other_category_raises(self):
        # same row count as the dataset, so only the category check can fail
        with pytest.raises(ValueError, match="data.category: SIDs 'beauty' vs recommender config 'toys'"):
            check_semids_dataset(semids(num_rows=10, category="beauty"), cfg(), num_items=10)

    def test_other_dataset_raises(self):
        with pytest.raises(ValueError, match="data.dataset"):
            check_semids_dataset(semids(dataset="movielens"), cfg(), num_items=10)

    def test_row_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="SID rows 12 vs 10 items"):
            check_semids_dataset(semids(num_rows=12), cfg(), num_items=10)

    def test_all_mismatches_are_named(self):
        with pytest.raises(ValueError) as err:
            check_semids_dataset(semids(num_rows=12, category="beauty"), cfg(), num_items=10)
        assert "data.category" in str(err.value) and "SID rows 12" in str(err.value)

    @NO_STAGE1_CONFIG
    def test_missing_stage1_config_raises(self, semids_data):
        with pytest.raises(ValueError, match="no Stage-1 config"):
            check_semids_dataset(semids_data, cfg(), num_items=10)


GTS = dict(split_protocol="gts", gts_q_test=0.9, gts_q_val=0.9)


class TestCheckSemidsProtocol:
    def test_loo_semids_with_loo_config_ok(self):
        check_semids_protocol(semids(), cfg(split_protocol="loo"))
        check_semids_protocol(semids(split_protocol="loo"), cfg())

    def test_gts_semids_with_gts_config_ok(self):
        check_semids_protocol(semids(**GTS), cfg(**GTS))

    def test_q_values_ignored_under_loo(self):
        check_semids_protocol(semids(split_protocol="loo", gts_q_test=0.5), cfg(gts_q_val=0.7))

    def test_loo_semids_with_gts_config_raises(self):
        with pytest.raises(ValueError, match="same protocol") as err:
            check_semids_protocol(semids(), cfg(**GTS))
        assert "'loo'" in str(err.value) and "'gts'" in str(err.value)

    def test_gts_semids_with_loo_config_raises(self):
        with pytest.raises(ValueError, match="same protocol"):
            check_semids_protocol(semids(**GTS), cfg(split_protocol="loo"))

    @pytest.mark.parametrize("key,value", [("gts_q_test", 0.8), ("gts_q_val", 0.8)])
    def test_gts_semids_with_other_q_raises(self, key, value):
        other = dict(GTS, **{key: value})
        with pytest.raises(ValueError, match="same protocol") as err:
            check_semids_protocol(semids(**GTS), cfg(**other))
        assert "0.8" in str(err.value) and "0.9" in str(err.value)

    @NO_STAGE1_CONFIG
    def test_missing_stage1_config_raises(self, semids_data):
        with pytest.raises(ValueError, match="no Stage-1 config"):
            check_semids_protocol(semids_data, cfg())


class TestCheckSemids:
    def test_runs_dataset_and_protocol_checks(self):
        check_semids(semids(), cfg(), num_items=10)
        with pytest.raises(ValueError, match="data.category"):
            check_semids(semids(category="beauty"), cfg(), num_items=10)
        with pytest.raises(ValueError, match="split protocol"):
            check_semids(semids(), cfg(**GTS), num_items=10)
