"""
Shared preprocessing utilities used across dataset classes.
"""
import torch
import polars as pl
from sentence_transformers import SentenceTransformer
from typing import List

from data.schemas import FUT_SUFFIX


def encode_text_embeddings(text_feat, model=None) -> torch.Tensor:
    """Encode a list of text strings into sentence embeddings via sentence-t5-xl."""
    print("Encoding text features...")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if model is None:
        model = SentenceTransformer('sentence-transformers/sentence-t5-xl', device=device)
    return model.encode(
        sentences=text_feat, show_progress_bar=True,
        convert_to_tensor=True, batch_size=8,
    ).cpu()


def sequence_df_to_tensor_dict(df: pl.DataFrame, seq_cols: List[str]) -> dict:
    """
    Packs user interaction histories from a DataFrame into a tensor dict.

    Returns:
        col          Padded histories as (N, seq_len) tensor; list[list[int]] if lengths vary.
        col + '_fut' Leave-one-out target items as (N,) tensor.
        'userId'     User identifiers as (N,) tensor.
    """
    out = {}
    for col in seq_cols:
        lengths = df[col].list.len()
        if lengths.min() == lengths.max():
            out[col] = torch.tensor(df[col].to_list())
        else:
            out[col] = df[col].to_list()
        out[col + FUT_SUFFIX] = torch.tensor(df[col + FUT_SUFFIX].to_list())
    out["userId"] = torch.tensor(df["userId"].to_list())
    return out
