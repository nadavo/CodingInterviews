from torch import Tensor, nn
import torch
from typing import Tuple


class PositionalEmbeddings(nn.Module):
    def __init__(self, context_length: int, embed_dim: int, base: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_length = context_length

        # Create a positional encoding matrix of shape (max_len, d_model)
        embedding = torch.zeros(context_length, embed_dim)

        # Create a vector of positions (0, 1, 2, ..., max_len - 1)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)  # shape (max_len, 1)

        # Create the division term for the sin/cos functions
        # This is (10000^(2i / d_model))
        # (d_model // 2) because we have pairs of sin/cos
        double_i = torch.arange(0, embed_dim, 2, dtype=torch.float)
        div_term = base ** (double_i / embed_dim)
        full_term = position / div_term

        # Apply sin to even indices (0, 2, 4, ...)
        embedding[:, 0::2] = torch.sin(full_term)
        # Apply cos to odd indices (1, 3, 5, ...)
        embedding[:, 1::2] = torch.cos(full_term)

        # Add a batch dimension so it can be broadcasted: (1, max_len, d_model)
        embedding = embedding.unsqueeze(0)

        # Register 'embedding' as a buffer. This means it's part of the model's
        # state, but it's not a parameter that should be updated by the optimizer.
        self.register_buffer("embedding", embedding)

    def forward(self, x: Tensor) -> Tensor:  # (B, S) -> (B, S, E)
        seq_len = x.size(-1)
        return self.embedding[:, :seq_len, :]


# TODO: Verify understanding and correctness
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Create the 'theta' frequencies
        # The dimension 'dim' must be even
        assert dim % 2 == 0

        # Calculate theta_i = 10000^(-2(i-1)/dim) for i = 1, 2, ..., dim/2
        # Shape: (dim / 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # Now, create a tensor for all positions up to max_seq_len
        # Shape: (max_seq_len)
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)

        # Calculate the outer product of positions and frequencies
        # Shape: (max_seq_len, dim / 2)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        # 'freqs' now contains the m*theta_i values.
        # We need to duplicate this for the sin and cos components.
        # Shape: (max_seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Pre-compute and cache the cos and sin values
        # We use register_buffer so these tensors are part of the model's state,
        # (e.g., moved to GPU with .to(device)), but are not model parameters.
        # Shape: (1, 1, max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: Tensor, seq_len: int = None) -> Tuple[Tensor, Tensor]:
        # x shape: (batch_size, num_heads, seq_len, head_dim)
        # This module is typically just used to fetch the cached values.
        # We slice the cache up to the current sequence length.
        if seq_len is None:
            seq_len = x.size(2)

        cos = self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype, device=x.device)

        return RotaryPositionalEmbeddings.apply_rotary_pos_emb(x, cos, sin)

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        """Rotates half the hidden dims of the input."""
        # Split the last dimension into two halves
        # x1 = [..., 0:dim/2], x2 = [..., dim/2:dim]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        # Concatenate them in swapped order with x2 negated
        # This creates [-x2, x1]
        return torch.cat((-x2, x1), dim=-1, device=x.device)

    @staticmethod
    def apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Applies rotary positional embedding to the input tensor 'x'.

        Args:
            x (torch.Tensor): Input tensor (Q or K).
                            Shape: (batch, n_heads, seq_len, head_dim)
            cos (torch.Tensor): Cached cosine values.
                                Shape: (1, 1, seq_len, head_dim)
            sin (torch.Tensor): Cached sine values.
                                Shape: (1, 1, seq_len, head_dim)
        """

        # This is the implementation of the formula:
        # x' = x * cos + rotate_half(x) * sin
        #
        # Let's break it down:
        # x = [x1, x2] (where x1 is the first half, x2 is the second half)
        # cos = [c1, c1] (cos values duplicated)
        # sin = [s1, s1] (sin values duplicated)
        # rotate_half(x) = [-x2, x1]
        #
        # x * cos          = [x1*c1, x2*c1]
        # rotate_half(x) * sin = [-x2*s1, x1*s1]
        #
        # Adding them together:
        # [x1*c1 - x2*s1, x2*c1 + x1*s1]
        #
        # This is exactly the complex number rotation:
        # x1' = x1*cos - x2*sin
        # x2' = x2*cos + x1*sin

        x_rotated = (x * cos) + (RotaryPositionalEmbeddings.rotate_half(x) * sin)
        return x_rotated
