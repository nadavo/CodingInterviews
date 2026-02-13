from .transformer_blocks import TransformerBlock, TransformerEmbedding, TransformerLMHead, TransformerLMHeadOutput
from torch import Tensor, LongTensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer, AdamW
from typing import TypedDict
import torch
import inspect


class TransformerEncoderConfig(TypedDict):
    vocab_size: int
    context_length: int
    embed_dim: int
    num_blocks: int
    num_heads: int
    num_groups: int
    dropout: float


class TransformerDecoderConfig(TransformerEncoderConfig):
    stop_token: int


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        num_groups: int,
        dropout: float = 0.0,
        stop_token: int = None,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, context_length, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, num_groups, dropout, False)] * num_blocks)
        self.lm_head = TransformerLMHead(embed_dim, vocab_size)
        self.stop_token = stop_token
        # init all weights
        self.apply(self._init_weights)

    def forward(self, x: Tensor, targets: Tensor = None) -> TransformerLMHeadOutput:
        device = x.device
        batch_size, curr_seq_len = x.size()
        if curr_seq_len > self.embedding.context_length:
            raise ValueError(f"curr_seq_len {curr_seq_len} is larger than context_length {self.context_length}")

        x = self.embedding(x)

        for b in self.blocks:
            x = b(x)

        return self.lm_head(x, targets)

    @torch.no_grad()
    def generate(
        self, tokens: LongTensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None
    ) -> LongTensor:
        for _ in range(max_new_tokens):
            curr_tokens = (
                tokens
                if tokens.size(1) <= self.embedding.context_length
                else tokens[:, -self.embedding.context_length :]
            )  # truncate context if larger than model's context_length

            logits = self(curr_tokens)["logits"] / temperature  # get next_token scores
            # Top K sampling - limit possible tokens to K of those with highest scores
            if top_k is not None and top_k < logits.size(-1):
                top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
                # top_k_mask = (
                #     logits < top_k_logits[:, [-1]]
                # )  # verify correct indexing in top_k_logits
                # logits[top_k_mask] = float("-inf")
                top_k_mask = torch.ones_like(logits) * float("-inf")  # full mask of -inf
                top_k_mask[:, top_k_indices] = 0.0  # exclude top_k token indices from mask
                logits = logits + top_k_mask  # add -inf to logits only in masked positions
            probs = F.softmax(logits, dim=-1)  # convert scores to probabilities

            next_token = torch.multinomial(probs, num_samples=1)  # sample tokens according to probabilities
            tokens = torch.cat((tokens, next_token), dim=1)  # add generated token to sequence
            if next_token == self.stop_token:  # stop if next_token is special stop_token
                break

        return tokens

    def configure_adamw_optimizer(self, weight_decay: float, learning_rate: float, device_type: str) -> Optimizer:
        # start with all of the candidate parameters and filter out those that do not require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Only parameters that is 2D or higher will be weight decayed.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = {"params": list(), "weight_decay": weight_decay}
        no_decay_params = {"params": list(), "weight_decay": 0.0}
        for n, p in param_dict.items():
            if p.dim() < 2:
                no_decay_params["params"].append(p)
            else:
                decay_params["params"].append(p)
        num_decay_params = sum(p.numel() for p in decay_params["params"])
        print(f"num decayed parameter tensors: {len(decay_params['params'])}, with {num_decay_params:,} parameters")
        num_nodecay_params = sum(p.numel() for p in no_decay_params["params"])
        print(
            f"num non-decayed parameter tensors: {len(no_decay_params['params'])}, with {num_nodecay_params:,} parameters"
        )
        optim_groups = [decay_params, no_decay_params]

        # Create AdamW optimizer and use the fused version if it is available
        extra_args = dict()
        use_fused = device_type == "cuda" and "fused" in inspect.signature(AdamW).parameters
        if use_fused:
            extra_args["fused"] = use_fused
        print(f"using fused AdamW: {use_fused}")

        optimizer = AdamW(optim_groups, lr=learning_rate, **extra_args)

        return optimizer

    def get_num_params(self, non_embedding: bool = True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= (
                self.embedding.token_embedding.weight.numel()
                + self.embedding.position_embedding.embedding.weight.numel()
            )
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
