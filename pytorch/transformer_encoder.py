from .transformer_blocks import TransformerBlock
from dataclasses import dataclass
from torch import LongTensor, arange, device, long, nn, FloatTensor
from torch.nn import functional as F
from typing import Tuple


@dataclass
class TransformerConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_blocks: int
    num_heads: int
    dropout: float


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embedding_dim
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embedding_dim, config.num_heads, config.dropout, True
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(
        self, seq_t: LongTensor, targets: LongTensor = None
    ) -> Tuple[FloatTensor, float]:
        seq_t_embedding = self.token_embedding(seq_t)  # (B, S) -> (B, S, E)
        seq_t_pos = arange(0, seq_t.size(-1), dtype=long, device=device)  # (1) -> (S)
        seq_t_pos_embedding = self.positional_embedding(seq_t_pos)  # (S) -> (S, E)

        x = self.dropout(seq_t_embedding + seq_t_pos_embedding)  # (B, S, E)

        for b in self.blocks:
            x = b(x)

        x = self.layer_norm(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:  # Training
            targets_flat = targets.view(-1)
            logits_flat = logits.view(-1, logits.size(-1))
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)

        return logits, loss
