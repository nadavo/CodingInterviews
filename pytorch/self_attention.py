from torch import Tensor, nn
from torch.nn import functional as F
from math import sqrt
import torch


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, attn_dim: int, project_output: bool = True, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, attn_dim, bias=False) if project_output else None
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def create_mask(seq_len: int, padding_mask: Tensor = None, is_bidirectional: bool = False) -> Tensor:
        mask = None
        if not is_bidirectional:
            # We want to mask out the upper triangle of the scores matrix so the tokens can't attend to future tokens.
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)  # (S, S)

        if padding_mask is not None:
            # Assume size (B, S) where 1 is real token and 0 is pad token
            pad_mask_bool = padding_mask.bool().logical_not().unsqueeze(1)  # (B, S) -> (B, 1, S)
            if mask is None:
                mask = pad_mask_bool
            else:
                mask = mask | pad_mask_bool  # (S, S) | (B, 1, S) -> (B, S, S)
        return mask

    @staticmethod
    def scaled_dot_product(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_bidirectional: bool = False,
        dropout: float = 0.0,
        padding_mask: Tensor = None,
    ) -> Tensor:  # (B, S, A) -> (B, S, A)
        seq_len, d = q.size(-2), q.size(-1)
        scores = (q @ k.transpose(-2, -1)) / sqrt(d)  # (B, S, A) x (B, A, S) -> (B, S, S)

        mask = SelfAttention.create_mask(seq_len, padding_mask, is_bidirectional)
        if mask is not None:
            # Insert -inf in masked positions so softmax on scores will result in 0 in those positions
            scores.masked_fill_(mask.to(scores.device), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, dropout, True)
        return attn_weights @ v

    def forward(
        self, x: Tensor, is_bidirectional: bool = False, padding_mask: Tensor = None
    ) -> Tensor:  # (B, S, E) -> (B, S, A)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # (B, S, E) -> (B, S, A)
        output = self.scaled_dot_product(
            q, k, v, is_bidirectional, self.dropout.p, padding_mask
        )  # (B, S, A) -> (B, S, A)
        return output if self.o_proj is None else self.dropout(self.o_proj(output))  # (B, S, A) -> (B, S, A)
