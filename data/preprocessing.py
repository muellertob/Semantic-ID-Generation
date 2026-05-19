"""
Shared preprocessing utilities used across dataset classes.
"""
import torch
import polars as pl
from transformers import AutoTokenizer, T5EncoderModel
from tqdm import tqdm
from typing import List

from data.schemas import FUT_SUFFIX


def _masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding positions.

    Args:
        hidden:          (B, seq_len, D) last hidden states from the encoder.
        attention_mask:  (B, seq_len)    1 for real tokens, 0 for padding.

    Returns:
        (B, D) mean embedding per sequence.
    """
    mask = attention_mask.unsqueeze(-1).float()          # (B, seq_len, 1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def encode_text_embeddings(text_feat) -> torch.Tensor:
    """Encode text strings into raw (unnormalized) embeddings via sentence-t5-base."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model_name = "sentence-transformers/sentence-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device).eval()

    texts = list(text_feat)
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 64), desc="Encoding text features"):
            batch = texts[i : i + 64]
            encoded = tokenizer(
                batch,
                max_length=128, # sufficient as item descriptions are not included 
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(device)
            hidden = encoder(**encoded).last_hidden_state          # (B, seq_len, 768)
            embeddings = _masked_mean_pool(hidden, encoded["attention_mask"])
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


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
