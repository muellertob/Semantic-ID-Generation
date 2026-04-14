"""
Unit tests for data/preprocessing.py — sequence_df_to_tensor_dict.

encode_text_embeddings is not tested: it is a thin wrapper around SentenceTransformer.encode
"""
import torch
import polars as pl
import pytest

from data.preprocessing import sequence_df_to_tensor_dict


def _fixed_df(n=3, seq_len=4):
    """DataFrame where every 'itemId' list has the same length."""
    return pl.DataFrame({
        "userId": list(range(1, n + 1)),
        "itemId": [[i * seq_len + j for j in range(seq_len)] for i in range(n)],
        "itemId_fut": list(range(100, 100 + n)),
    })


def _variable_df():
    """DataFrame where 'itemId' lists have different lengths."""
    return pl.DataFrame({
        "userId": [1, 2],
        "itemId": [[10, 20], [30, 40, 50]],
        "itemId_fut": [99, 88],
    })


class TestSequenceDfToTensorDict:

    def test_fixed_length_sequences_become_tensor(self):
        result = sequence_df_to_tensor_dict(_fixed_df(n=3, seq_len=4), ["itemId"])
        assert isinstance(result["itemId"], torch.Tensor)

    def test_fixed_length_tensor_shape_is_N_by_seq_len(self):
        result = sequence_df_to_tensor_dict(_fixed_df(n=3, seq_len=4), ["itemId"])
        assert result["itemId"].shape == (3, 4)

    def test_fixed_length_tensor_values_correct(self):
        df = pl.DataFrame({
            "userId": [1, 2],
            "itemId": [[10, 20, 30], [40, 50, 60]],
            "itemId_fut": [99, 88],
        })
        result = sequence_df_to_tensor_dict(df, ["itemId"])
        assert result["itemId"].tolist() == [[10, 20, 30], [40, 50, 60]]

    def test_variable_length_sequences_become_list(self):
        result = sequence_df_to_tensor_dict(_variable_df(), ["itemId"])
        assert isinstance(result["itemId"], list)

    def test_variable_length_list_preserves_values(self):
        result = sequence_df_to_tensor_dict(_variable_df(), ["itemId"])
        assert result["itemId"][0] == [10, 20]
        assert result["itemId"][1] == [30, 40, 50]

    def test_fut_column_is_tensor(self):
        result = sequence_df_to_tensor_dict(_fixed_df(), ["itemId"])
        assert isinstance(result["itemId_fut"], torch.Tensor)

    def test_fut_column_values_correct(self):
        result = sequence_df_to_tensor_dict(_fixed_df(n=3), ["itemId"])
        assert result["itemId_fut"].flatten().tolist() == [100, 101, 102]

    def test_user_id_is_tensor(self):
        result = sequence_df_to_tensor_dict(_fixed_df(), ["itemId"])
        assert isinstance(result["userId"], torch.Tensor)

    def test_user_id_values_correct(self):
        result = sequence_df_to_tensor_dict(_fixed_df(n=3), ["itemId"])
        assert result["userId"].flatten().tolist() == [1, 2, 3]

    def test_output_contains_exactly_expected_keys(self):
        result = sequence_df_to_tensor_dict(_fixed_df(), ["itemId"])
        assert set(result.keys()) == {"itemId", "itemId_fut", "userId"}
