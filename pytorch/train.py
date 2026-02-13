from .transformer_decoder import TransformerDecoder
from .gpt_tokenizer import GPTTokenizer
from torch import tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch
from tqdm.auto import tqdm


def generate_completion(
    transformer: TransformerDecoder, tokenizer: GPTTokenizer, prompt: str, device: str, max_new_tokens: int = 2
) -> str:
    print(prompt)
    prompt_tokens = tokenizer.tokenize(prompt)
    generated_tokens = transformer.generate(
        tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device), max_new_tokens
    )
    generated_str = tokenizer.decode(generated_tokens.squeeze().tolist())
    print(generated_str)
    return generated_tokens, generated_str


@torch.no_grad()
def calc_dataset_loss(
    model: TransformerDecoder, dataloader: DataLoader, batch_size: int, device: str, split: str = "validation"
) -> float:
    model.eval()
    losses = list()
    num_batches = len(dataloader.dataset) // batch_size
    with tqdm(enumerate(dataloader), desc="Batch", total=num_batches) as batch_pbar:
        for i, batch in batch_pbar:
            x, y = batch
            output = model(x.to(device), y.to(device))

            batch_loss = output["loss"].mean().item()
            batch_pbar.set_postfix(loss=f"{batch_loss:.8f}")
            losses.append(batch_loss)

    loss = tensor(losses).mean().item()
    model.train()
    print(f"{split.capitalize()} Loss: {loss:.6f}")
    return loss


def training_loop(
    model: TransformerDecoder,
    optimizer: Optimizer,
    train_dataloader: DataLoader,
    num_epochs: int,
    batch_size: int,
    device: str,
    val_dataloader: DataLoader = None,
    eval_every_epoch: bool = False,
) -> float:
    if val_dataloader is not None:
        print("Pre-training Eval")
        val_loss = calc_dataset_loss(model, val_dataloader, batch_size, device, "validation")

    train_losses = list()
    model.train()
    num_batches = len(train_dataloader.dataset) // batch_size
    with tqdm(range(num_epochs), desc="Epoch") as epoch_pbar:
        for e in epoch_pbar:
            e_losses = list()
            with tqdm(enumerate(train_dataloader), desc="Batch", total=num_batches) as batch_pbar:
                for i, batch in batch_pbar:
                    x, y = batch
                    output = model(x.to(device), y.to(device))
                    loss = output["loss"]

                    batch_loss = loss.mean().item()
                    batch_pbar.set_postfix(loss=f"{batch_loss:.8f}")
                    e_losses.append(batch_loss)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epoch_loss = tensor(e_losses).mean().item()
            epoch_pbar.set_postfix(loss=f"{epoch_loss:.8f}")
            print(f"Epoch {e + 1} loss: {epoch_loss}")
            train_losses.extend(e_losses)
            if val_dataloader is not None and eval_every_epoch:
                e_val_loss = calc_dataset_loss(model, val_dataloader, batch_size, device, "validation")

    train_loss = tensor(train_losses).mean().item()
    print(f"Training loss: {train_loss}")
    if val_dataloader is not None:
        print("Post-training Eval")
        val_loss = calc_dataset_loss(model, val_dataloader, batch_size, device, "validation")
    else:
        val_loss = None

    return train_loss, val_loss
