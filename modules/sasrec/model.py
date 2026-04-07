"""
SASRec: Self-Attentive Sequential Recommendation.

Implementation of the paper architecture:
- Decoder-only Transformer on raw item ID sequences
- Pre-norm residual pattern with causal self-attention
- Shared item embedding for prediction (Eq. 6)
- BCE loss with per-position negative sampling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_negatives(pos_items: torch.Tensor, num_items: int) -> torch.Tensor:
    """Sample one negative item per position, ensuring it differs from the positive."""
    neg_items = torch.randint(1, num_items + 1, pos_items.shape, device=pos_items.device)
    collisions = neg_items == pos_items
    while collisions.any():
        neg_items[collisions] = torch.randint(1, num_items + 1, (collisions.sum(),), device=pos_items.device)
        collisions = neg_items == pos_items
    return neg_items


class PointWiseFFN(nn.Module):
    """Two-layer position-wise feed-forward network (Section III-C)."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))


class SASRecBlock(nn.Module):
    """Single self-attention block with pre-norm residual pattern."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = PointWiseFFN(hidden_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))
        return x


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.

    Input: left-padded sequences of 1-based item IDs (0 = padding).
    Output: per-position scores over all items via shared embedding dot product.
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # stacked self-attention blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        # final LayerNorm after last block
        self.final_norm = nn.LayerNorm(hidden_dim)

    def encode(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_seq: [B, max_len] 1-based item IDs (0 = padding)

        Returns:
            x: [B, max_len, hidden_dim] hidden states at each position
        """
        B, L = item_seq.shape
        device = item_seq.device

        # causal mask [L, L]: -inf for future positions, 0 elsewhere
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask.bool(), float('-inf'))

        # padding mask [B, L, 1]: 1 for real tokens, 0 for padding
        pad_mask = (item_seq != 0).float().unsqueeze(-1)

        # embedding
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.item_embedding(item_seq) + self.position_embedding(positions)
        x = self.emb_dropout(x)
        x = x * pad_mask  # zero out padding before first block

        # apply self-attention blocks, re-zeroing padding after each
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
            x = x * pad_mask

        return self.final_norm(x)

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_seq: [B, max_len] 1-based item IDs (0 = padding)

        Returns:
            scores: [B, max_len, num_items+1] prediction scores at each position
        """
        return self.encode(item_seq) @ self.item_embedding.weight.T

    def compute_loss(
        self,
        item_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss per non-padding position."""
        x = self.encode(item_seq)  # [B, L, D]

        # direct dot product for pos/neg only — avoids full matrix multiplication
        pos_emb = self.item_embedding(pos_items)   # [B, L, D]
        neg_emb = self.item_embedding(neg_items)   # [B, L, D]
        pos_scores = (x * pos_emb).sum(-1)         # [B, L]
        neg_scores = (x * neg_emb).sum(-1)         # [B, L]

        # BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores), reduction='none')
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores), reduction='none')

        # derive padding mask from pos_items
        pad_mask = (pos_items != 0).float()
        loss = (pos_loss + neg_loss) * pad_mask

        # average over valid positions
        num_valid = pad_mask.sum()
        if num_valid > 0:
            return loss.sum() / num_valid
        return torch.tensor(0.0, requires_grad=True, device=item_seq.device)
