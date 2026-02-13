from typing import Iterable, List
from tiktoken import encoding_for_model


class GPTTokenizer:
    def __init__(self, model_name: str = "gpt-oss-20b", stop_token_str: str = "<|endoftext|>"):
        self.tokenizer = encoding_for_model(model_name)
        self.stop_token_str = stop_token_str

    def get_vocab_size(self) -> int:
        return self.tokenizer.max_token_value + 1

    def get_stop_token_id(self) -> int:
        return self.tokenize(self.stop_token_str)[0]

    def tokenize(self, texts: str | Iterable[str]) -> List[int] | List[List[int]]:
        if isinstance(texts, str):
            return self.tokenizer.encode(texts, allowed_special={self.stop_token_str})
        elif isinstance(texts, Iterable):
            return self.tokenizer.encode_batch(texts, allowed_special={self.stop_token_str})

    def decode(self, tokens: List[int] | Iterable[List[int]]) -> str | List[str]:
        if isinstance(tokens, list) and isinstance(tokens[0], int):
            return self.tokenizer.decode(tokens)
        elif isinstance(tokens, Iterable) and isinstance(tokens[0], list):
            return self.tokenizer.decode_batch(tokens)
