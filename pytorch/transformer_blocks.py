from typing import Dict, TypedDict
from .feed_forward import RMSNorm, SwiGLUFeedForward
from .positional_embedding import PositionalEmbeddings
from .grouped_query_attention import GroupedQueryAttention
from torch import Tensor, nn, FloatTensor
from torch.nn import functional as F


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        dropout: float = 0.0,
        positional_embedding_cls=PositionalEmbeddings,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = positional_embedding_cls(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # (B, S) -> (B, S, E)
        tok_x = self.token_embedding(x)
        pos_x = self.position_embedding(x)
        return self.dropout(tok_x + pos_x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int, dropout: float = 0.0, is_encoder: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attention = GroupedQueryAttention(embed_dim, num_heads, num_groups, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ff = SwiGLUFeedForward(embed_dim, dropout=dropout)
        self.is_encoder = is_encoder
        # TODO: Implement Cross-Attention for Encoder-Decoder Transformers
        # if not is_decoder:
        #     self.norm_cross = RMSNorm(embed_dim)
        #     self.cross_attention = ...

    def forward(self, x: Tensor, encoder_output: Dict[str, Tensor] = None) -> Tensor:
        x = x + self.attention(self.norm1(x), self.is_encoder)
        # TODO: Implement Cross-Attention for Encoder-Decoder Transformers
        # if not self.is_encoder:
        #     x = x + self.cross_attention(self.norm_cross(x), encoder_output, not self.is_encoder)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerLMHeadOutput(TypedDict):
    loss: FloatTensor
    logits: FloatTensor


class TransformerLMHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self, x: Tensor, targets: Tensor = None
    ) -> Dict[str, FloatTensor]:  # output_layer (B, S, E) -> logits (B, S, V)
        x = self.norm(x)

        if targets is None:
            # Inference - only need last token logits for generation
            last_token = x[:, -1, :]
            logits = self.lm_head(last_token)
            loss = None
        else:
            # Training
            logits = self.lm_head(x)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)

        return TransformerLMHeadOutput({"loss": loss, "logits": logits})
