from torch import nn, Tensor
from torch.nn import functional as F
from typing import Callable, Optional
import torch


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    @staticmethod
    def root_mean_square(x: Tensor, eps: float = 1e-8) -> float:
        return torch.sqrt(eps + torch.mean(torch.pow(x, 2), dim=-1, keepdim=True))

    def forward(self, x: Tensor) -> Tensor:
        rms = self.root_mean_square(x, self.eps)
        return (x / rms) * self.scale


class FeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        activation_fn: Callable = F.gelu,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if hidden_dim is None:
            hidden_dim = output_dim * 4
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        hidden_dim: int = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if hidden_dim is None:
            hidden_dim = int(2 * output_dim / 3)
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias)
        self.i_proj = nn.Linear(input_dim, hidden_dim, bias)
        self.o_proj = nn.Linear(hidden_dim, output_dim, bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))
        info = self.i_proj(x)
        output = self.o_proj(gate * info)
        return self.dropout(output)
