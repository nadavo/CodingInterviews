from .self_attention import SelfAttention
from torch import Tensor, nn


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, num_groups: int = 1, dropout: float = 0.0):
        super().__init__()
        if num_heads < 1 or num_groups < 1:
            raise ValueError(f"num_heads ({num_heads}) and num_groups ({num_groups}) must be an integer of at least 1")
        if num_heads < num_groups:
            raise ValueError(f"num_groups ({num_groups}) must be larger than num_heads ({num_heads})")
        if num_heads % num_groups != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_groups ({num_groups})")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_q_heads ({num_heads})")

        self.num_q_heads = num_heads
        self.num_kv_heads = num_groups
        self.head_dim = embed_dim // num_heads
        self.num_repeat = self.num_q_heads // self.num_kv_heads  # num repetitions of KV to be distributed to all Q
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # hQ = hd * nQ = embed_dim
        self.k_proj = nn.Linear(embed_dim, self.head_dim * self.num_kv_heads, bias=False)  # hKV = hd * nKV
        self.v_proj = nn.Linear(embed_dim, self.head_dim * self.num_kv_heads, bias=False)  # hKV = hd * nKV
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, is_bidirectional: bool = False) -> Tensor:  # (B, S, E) -> (B, S, E)
        batch_size, seq_len = x.size(0), x.size(1)
        q = (
            self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        )  # (B, S, E) -> (B, S, nQ, hd) -> (B, nQ, S, hd)
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .repeat_interleave(self.num_repeat, dim=1)
        )  # (B, S, E) -> (B, S, nKV, hd) -> (B, nKV, S, hd) x num_repeat on nKV dim
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .repeat_interleave(self.num_repeat, dim=1)
        )  # (B, S, E) -> (B, S, nKV, hd) -> (B, nKV, S, hd) x num_repeat on nKV dim
        outputs = (
            SelfAttention.scaled_dot_product(q, k, v, is_bidirectional, self.dropout.p)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )  # (B, nQ, S, hd) -> (B, S, nQ, hd) -> (B, S, E)
        return self.dropout(self.o_proj(outputs))  # (B, S, E) -> (B, S, E)
