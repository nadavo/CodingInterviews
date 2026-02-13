from torch import Tensor, nn
import torch
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads ({num_heads}) must be an integer of at least 1")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_q_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Simple linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def feature_map(x: Tensor) -> Tensor:
        """
        The feature map phi(x) = elu(x) + 1.
        This is a simple non-negative function to approximate the softmax kernel.
        """
        return F.elu(x) + 1

    def forward(self, x: Tensor, is_bidirectional: bool = False, padding_mask: Tensor = None) -> Tensor:
        """
        x: Input tensor of shape (batch_size, seq_len, embed_dim)
        padding_mask: (batch_size, seq_len) tensor, 1 for pad, 0 for real
        """
        b_s, s_l, e_d = x.size()
        n_h = self.num_heads
        h_d = self.head_dim  # 'h_d' is the head dimension, (d_k and d_v)

        # 1. Project inputs to Q, K, V + Reshape for multi-head attention
        # (b_s, s_l, e_d) -> (b_s, s_l, n_h, h_d) -> (b_s, n_h, s_l, h_d)
        q = self.q_proj(x).view(b_s, s_l, n_h, h_d).transpose(1, 2)
        k = self.k_proj(x).view(b_s, s_l, n_h, h_d).transpose(1, 2)
        v = self.v_proj(x).view(b_s, s_l, n_h, h_d).transpose(1, 2)

        # 2. Apply feature map PHI
        q_prime = self.feature_map(q)
        k_prime = self.feature_map(k)

        # Handle padding by zeroing out padded tokens in K' and V
        if padding_mask is not None:
            # (b, n) -> (b, 1, n, 1) to broadcast over heads and dims
            mask = padding_mask.view(b_s, 1, n_h, 1).bool()
            k_prime.masked_fill_(mask, 0.0)
            v.masked_fill(mask, 0.0)

        if is_bidirectional:
            # 3. The (Bidirectional) "Linear Attention" trick

            # 3a. Compute (K'^T V) first.
            # k_prime shape: (b, h, n, d)
            # We need to transpose the last two dims: (b, h, d, n)
            # v shape: (b, h, n, d)
            # Result (kv_context) shape: (b, h, d, d)
            kv_context = k_prime.transpose(-2, -1) @ v

            # 3b. Compute the numerator N_i = Q'_i * (K'^T V)
            # q_prime shape: (b, h, n, d)
            # kv_context shape: (b, h, d, d)
            # Result (numerator) shape: (b, h, n, d)
            numerator = q_prime @ kv_context

            # 3c. Compute the denominator for normalization
            # Sum k_prime over the sequence length dimension (n)
            # k_prime shape: (b, h, n, d) -> k_sum shape: (b, h, d)
            k_sum_per_head = torch.sum(k_prime, dim=2)

            # We need to multiply Q' by the sum of K'
            # q_prime shape: (b, h, n, d)
            # k_sum_per_head shape: (b, h, d)
            # To use matmul, we unsqueeze k_sum: (b, h, d) -> (b, h, d, 1)
            # Result (denominator) shape: (b, h, n, 1)
            denominator = q_prime @ k_sum_per_head.unsqueeze(-1)

        else:
            # 3. The (Causal) "Linear Attention" trick

            # 3a. Compute the cumulative sum of (K'^T V)
            # We need the outer product (k_j' * v_j^T) for each token j.
            # k_prime.unsqueeze(-1) is (b, h, n, d, 1)
            # v.unsqueeze(-2) is (b, h, n, 1, d)
            # kv_outer_prod is (b, h, n, d, d) -> a *sequence* of (d, d) matrices
            kv_outer_prod = k_prime.unsqueeze(-1) @ v.unsqueeze(-2)

            # Now, compute the cumulative sum along the sequence dim 'n'
            # This is our state S_i from the explanation
            # kv_context_prefix shape: (b, h, n, d, d)
            kv_context_prefix = torch.cumsum(kv_outer_prod, dim=2)

            # 3b. Compute the numerator N_i = Q'_i * S_i
            # q_prime.unsqueeze(-2) is (b, h, n, 1, d)
            # kv_context_prefix is (b, h, n, d, d)
            # numerator shape: (b, h, n, 1, d)
            numerator = q_prime.unsqueeze(-2) @ kv_context_prefix
            # Squeeze to (b, h, n, d)
            numerator = numerator.squeeze(-2)

            # 3c. Compute the cumulative sum of K'
            # This is our state Z_i from the explanation
            # k_sum_prefix shape: (b, h, n, d)
            k_sum_prefix = torch.cumsum(k_prime, dim=2)

            # 3d. Compute the denominator D_i = Q'_i * Z_i
            # q_prime.unsqueeze(-2) is (b, h, n, 1, d)
            # k_sum_prefix.unsqueeze(-1) is (b, h, n, d, 1)
            # denominator shape: (b, h, n, 1, 1)
            denominator = q_prime.unsqueeze(-2) @ k_sum_prefix.unsqueeze(-1)
            # Squeeze to (b, h, n, 1)
            denominator = denominator.squeeze(-1)

        # 4. Compute final output
        # Add epsilon for numerical stability
        out = numerator / (denominator + 1e-6)

        # 5. Reshape and project out
        # (b, h, n, d) -> (b, n, h, d) -> (b, n, e)
        out = out.transpose(1, 2).contiguous().view(b_s, n_h, e_d)
        return self.o_proj(out)
