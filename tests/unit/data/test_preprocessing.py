"""
Unit tests for data/preprocessing.py.

TestSequenceDfToTensorDict  — tests for sequence_df_to_tensor_dict, covering fixed-length vs variable-length
TestMaskedMeanPool          — tests for the extracted _masked_mean_pool helper.
TestEncodeTextEmbeddings    — tests for encode_text_embeddings with mocked
                              AutoTokenizer / T5EncoderModel so no real model
                              download is required.
"""
import pytest
import torch
import polars as pl
from unittest.mock import patch, MagicMock

from data.preprocessing import sequence_df_to_tensor_dict, _masked_mean_pool, encode_text_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_texts():
    return ["Beauty product with great scent.", "Moisturizing cream for dry skin."]


@pytest.fixture
def mock_t5_components():
    """
    Patches AutoTokenizer.from_pretrained and T5EncoderModel.from_pretrained so
    no real model is loaded.  Yields (mock_tokenizer_cls, mock_encoder_cls) while
    the patches are active.

    The mock encoder returns hidden states of shape (n_texts, 128, 768) filled
    with 3.0 — giving a mean-pooled norm of sqrt(768 * 9) ≈ 83, clearly != 1.0.
    The mock tokenizer returns all-ones attention masks so every token contributes.
    """
    SEQ_LEN, HIDDEN = 128, 768

    def tokenizer_side_effect(batch, **kwargs):
        n = len(batch)
        out = MagicMock()
        out.to.return_value = {
            "input_ids":      torch.ones(n, SEQ_LEN, dtype=torch.long),
            "attention_mask": torch.ones(n, SEQ_LEN, dtype=torch.long),
        }
        return out

    def encoder_side_effect(input_ids, attention_mask):
        n = input_ids.shape[0]
        mock_out = MagicMock()
        mock_out.last_hidden_state = torch.full((n, SEQ_LEN, HIDDEN), 3.0)
        return mock_out

    mock_tokenizer = MagicMock(side_effect=tokenizer_side_effect)
    mock_encoder   = MagicMock(side_effect=encoder_side_effect)
    mock_encoder.to.return_value   = mock_encoder
    mock_encoder.eval.return_value = mock_encoder

    with patch("data.preprocessing.AutoTokenizer.from_pretrained",  return_value=mock_tokenizer), \
         patch("data.preprocessing.T5EncoderModel.from_pretrained", return_value=mock_encoder):
        yield mock_tokenizer, mock_encoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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


class TestMaskedMeanPool:
    """Tests for the _masked_mean_pool helper function."""

    def test_single_real_token_returns_that_embedding(self):
        """With one real token and no padding, output equals that token."""
        hidden = torch.tensor([[[1.0, 2.0, 3.0]]])   # (1, 1, 3)
        mask   = torch.ones(1, 1, dtype=torch.long)
        result = _masked_mean_pool(hidden, mask)
        assert torch.allclose(result, torch.tensor([[1.0, 2.0, 3.0]]))

    def test_padding_tokens_excluded_from_mean(self):
        """Padding positions (mask=0) must not contribute to the mean."""
        # token 0: real value [1, 2]; token 1: padding [100, 200]
        hidden = torch.tensor([[[1.0, 2.0], [100.0, 200.0]]])  # (1, 2, 2)
        mask   = torch.tensor([[1, 0]])
        result = _masked_mean_pool(hidden, mask)
        assert torch.allclose(result, torch.tensor([[1.0, 2.0]]))

    def test_all_real_tokens_are_averaged(self):
        """When no padding, result is plain mean of all token embeddings."""
        hidden = torch.tensor([[[0.0, 0.0], [2.0, 4.0]]])  # (1, 2, 2)
        mask   = torch.ones(1, 2, dtype=torch.long)
        result = _masked_mean_pool(hidden, mask)
        assert torch.allclose(result, torch.tensor([[1.0, 2.0]]))

    def test_batch_rows_are_pooled_independently(self):
        """Each row in the batch uses its own mask."""
        hidden = torch.tensor([
            [[1.0], [2.0], [3.0]],   # row 0: 3 tokens
            [[4.0], [5.0], [6.0]],   # row 1: 3 tokens
        ])  # (2, 3, 1)
        # row 0: first 2 real → mean = 1.5; row 1: first 1 real → mean = 4.0
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        result = _masked_mean_pool(hidden, mask)
        assert torch.allclose(result, torch.tensor([[1.5], [4.0]]))

    def test_all_padding_does_not_raise(self):
        """All-zero mask must not cause division by zero (clamp guard)."""
        hidden = torch.zeros(1, 3, 4)
        mask   = torch.zeros(1, 3, dtype=torch.long)
        result = _masked_mean_pool(hidden, mask)
        assert result.shape == (1, 4)
        assert not torch.isnan(result).any()

    def test_output_shape_is_batch_by_hidden_dim(self):
        """Output shape must be (B, D) for input (B, S, D)."""
        B, S, D = 5, 10, 32
        hidden = torch.randn(B, S, D)
        mask   = torch.ones(B, S, dtype=torch.long)
        assert _masked_mean_pool(hidden, mask).shape == (B, D)


class TestEncodeTextEmbeddings:
    """Tests for encode_text_embeddings with mocked T5 components.

    The mock encoder returns constant hidden states filled with 3.0 so that
    the mean-pooled norm is sqrt(768 * 9) ≈ 83 — clearly not 1.0.  This lets
    us verify that no normalisation is silently applied inside the function.
    """

    # --- output properties --------------------------------------------------

    def test_output_shape_is_N_by_768(self, two_texts, mock_t5_components):
        result = encode_text_embeddings(two_texts)
        assert result.shape == (len(two_texts), 768)

    def test_output_is_cpu_tensor(self, two_texts, mock_t5_components):
        result = encode_text_embeddings(two_texts)
        assert result.device == torch.device("cpu")

    def test_output_dtype_is_float32(self, two_texts, mock_t5_components):
        result = encode_text_embeddings(two_texts)
        assert result.dtype == torch.float32

    def test_output_norms_are_not_unit(self, two_texts, mock_t5_components):
        """Key contract: embeddings must NOT be L2-normalised to unit sphere.

        The SentenceTransformer pipeline runs a 3_Normalize module, producing
        norm = 1.0 for every row. This implementation bypasses that module by
        loading T5EncoderModel directly, so norms must differ from 1.0.
        """
        result = encode_text_embeddings(two_texts)
        norms = result.norm(dim=1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=0.05), (
            f"Embeddings appear unit-normalised (norms={norms.tolist()}). "
            "The 3_Normalize step must be bypassed."
        )

    # --- model / tokenizer identity ----------------------------------------

    def test_loads_sentence_t5_base(self, two_texts, mock_t5_components):
        _, mock_encoder_cls = mock_t5_components
        with patch("data.preprocessing.AutoTokenizer.from_pretrained") as mock_tok_cls, \
             patch("data.preprocessing.T5EncoderModel.from_pretrained", return_value=mock_encoder_cls):
            mock_tok_cls.return_value = MagicMock(
                side_effect=lambda batch, **kw: MagicMock(
                    to=lambda d: {
                        "input_ids":      torch.ones(len(batch), 128, dtype=torch.long),
                        "attention_mask": torch.ones(len(batch), 128, dtype=torch.long),
                    }
                )
            )
            encode_text_embeddings(two_texts)
            mock_tok_cls.assert_called_once_with("sentence-transformers/sentence-t5-base")

    # --- tokenizer settings ------------------------------------------------

    def test_tokenizer_padding_is_max_length(self, two_texts, mock_t5_components):
        mock_tokenizer, _ = mock_t5_components
        encode_text_embeddings(two_texts)
        _, kwargs = mock_tokenizer.call_args
        assert kwargs.get("padding") == "max_length"

    def test_tokenizer_truncation_is_true(self, two_texts, mock_t5_components):
        mock_tokenizer, _ = mock_t5_components
        encode_text_embeddings(two_texts)
        _, kwargs = mock_tokenizer.call_args
        assert kwargs.get("truncation") is True

    def test_tokenizer_add_special_tokens_is_true(self, two_texts, mock_t5_components):
        mock_tokenizer, _ = mock_t5_components
        encode_text_embeddings(two_texts)
        _, kwargs = mock_tokenizer.call_args
        assert kwargs.get("add_special_tokens") is True

    def test_tokenizer_return_tensors_is_pt(self, two_texts, mock_t5_components):
        mock_tokenizer, _ = mock_t5_components
        encode_text_embeddings(two_texts)
        _, kwargs = mock_tokenizer.call_args
        assert kwargs.get("return_tensors") == "pt"

    # --- batching -----------------------------------------------------------

    def test_more_texts_than_batch_size_produces_correct_shape(self, mock_t5_components):
        """N=70 texts (> internal batch size of 64) must yield shape (70, 768)."""
        texts = [f"item description {i}" for i in range(70)]
        result = encode_text_embeddings(texts)
        assert result.shape == (70, 768)

    def test_single_text_produces_shape_1_by_768(self, mock_t5_components):
        result = encode_text_embeddings(["single product description"])
        assert result.shape == (1, 768)


# ---------------------------------------------------------------------------
# Slow integration test (requires real model download)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_model_norms_are_not_unit():
    """With the real sentence-t5-base model loaded via T5EncoderModel (not
    SentenceTransformer), output norms must NOT be 1.0.

    The old pipeline ran sentence-t5-xl via SentenceTransformer, whose 3_Normalize
    module forces every row to norm=1.0.  The new pipeline loads T5EncoderModel
    directly, bypassing that module, so norms must deviate clearly from 1.0.
    We do not assert a specific target norm — the exact value depends on text
    content and model version — only that normalisation was not silently applied.
    """
    result = encode_text_embeddings([
        "Title: Moisturizing Cream; Brand: CeraVe; Categories: Beauty; Price: 12.99;",
        "Title: Lip Balm; Brand: Burt's Bees; Categories: Beauty; Price: 3.49;",
    ])
    norms = result.norm(dim=1)
    assert not torch.isnan(result).any(), "Output contains NaN values."
    assert not torch.allclose(norms, torch.ones_like(norms), atol=0.05), (
        f"Embeddings appear unit-normalised (norms={norms.tolist()}). "
        "The 3_Normalize step must be bypassed."
    )
