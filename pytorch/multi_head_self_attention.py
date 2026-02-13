from .self_attention import SelfAttention
from torch import Tensor, nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be an integer of at least 1")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_q_heads ({num_heads})")

        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([SelfAttention(embed_dim, self.head_dim, False, dropout)] * num_heads)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, is_bidirectional: bool = False) -> Tensor:
        outputs = torch.cat([h(x, is_bidirectional) for h in self.heads], dim=-1)
        return self.dropout(self.o_proj(outputs))


class BatchedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads ({num_heads}) must be an integer of at least 1")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_q_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, is_bidirectional: bool = False) -> Tensor:  # (B, S, E) -> (B, S, E)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        batch_size, seq_len, attn_dim = q.size()
        for t in (q, k, v):
            t = t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )  # (B, S, E) -> (B, S, nh, hd) -> (B, nh, S, hd)
        outputs = (
            SelfAttention.scaled_dot_product(q, k, v, is_bidirectional, self.dropout.p)
            .transpose(1, 2)
            .view(batch_size, seq_len, attn_dim)
        )  # (B, nh, S, hd) -> (B, S, nh, hd) -> (B, S, E)
        return self.dropout(self.o_proj(outputs))  # (B, S, E) -> (B, S, E)
