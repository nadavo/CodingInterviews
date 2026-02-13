from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from torch import tensor
from torch import cuda

MAX_SEQ_LEN = 512  # Example maximum sequence length
batch_size = 32  # Example batch size


class NextTokenDataset(Dataset):
    def __init__(self, tokenized_text: List[int], context_size: int = MAX_SEQ_LEN - 1):
        self.data = tensor(tokenized_text, requires_grad=False)
        self.context_size = context_size

    def __len__(self) -> int:
        return len(self.data) - self.context_size

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.data[idx : idx + self.context_size], self.data[idx + 1 : idx + self.context_size + 1]


# train_dataset = NextTokenDataset(train_data_tokens)
# val_dataset = NextTokenDataset(val_data_tokens)

# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=cuda.is_available())
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=cuda.is_available())
