from torch import Tensor, nn
import torch
from math import sqrt


class LoRA(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = None, r: int = 1, alpha: int = 1, dropout: float = 0.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.W_A = nn.Parameter(torch.randn(input_dim, r, dtype=torch.float) / sqrt(float(r)))  # i_d -> r
        self.W_B = nn.Parameter(torch.zeros(r, output_dim, dtype=torch.float))  # r -> o_d
        self.r = r
        self.alpha = alpha
        self.scale = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # return ((self.dropout(x) @ self.W_A) @ self.W_B) * self.scale
        h = self.dropout(x @ self.W_A)
        return (h @ self.W_B) * self.scale  # x -> scale * (dropout(x @ A) @ B)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 1, alpha: int = 1, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRA(linear.in_features, linear.out_features, r, alpha, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.lora(x)

    @staticmethod
    def apply_lora_to_model(model: nn.Module, r: int = 1, alpha: int = 1, dropout: float = 0.0):
        """
        Recursively replaces all nn.Linear layers in a model with LinearWithLoRA.

        Args:
            model (nn.Module): The model to modify.
            r (int): The rank of the LoRA decomposition.
            alpha (int): The LoRA scaling factor.
            dropout (float): The dropout probability for LoRA.
        """
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Linear):
                # Replace the nn.Linear layer with LinearWithLoRA
                print(f"Applying LoRA to Linear layer: {name}")
                setattr(model, name, LinearWithLoRA(module, r, alpha, dropout))
            else:
                # Recurse into child modules
                LinearWithLoRA.apply_lora_to_model(module, r, alpha, dropout)
