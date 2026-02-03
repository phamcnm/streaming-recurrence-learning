import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils import format_aux

vocab_size = 3  # excludes 0 as blank
blank_id = 0
input_dim = vocab_size + 1
output_dim = input_dim


class RepeatPreviousKDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=vocab_size, k=1):
        assert k >= 0
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.k = k

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(1, self.vocab_size + 1, (self.seq_len,), dtype=torch.long)
        y = torch.full((self.seq_len,), blank_id, dtype=torch.long)
        if self.k < self.seq_len:
            y[self.k:] = x[: self.seq_len - self.k]
        return x, y


def train(model, seq_len, vocab_size=vocab_size, k=1, num_samples=10_000, num_epochs=10, act_loss_coeff=0.01):
    dataset = RepeatPreviousKDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
        k=k,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        epoch_aux_accumulator = None
        num_batches = 0
        for x, y in loader:
            x, y = x.transpose(0, 1), y.transpose(0, 1)
            logits, hidden, aux = model(x)
            T, B, V = logits.shape
            loss = loss_fn(logits.reshape(T * B, V), y.reshape(T * B))
            if isinstance(aux, dict) and aux.get("ponder_cost") is not None:
                loss = loss + act_loss_coeff * aux["ponder_cost"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            summary = aux.get("summary") if isinstance(aux, dict) else aux
            if isinstance(summary, list):
                if epoch_aux_accumulator is None:
                    epoch_aux_accumulator = [0.0] * len(summary)
                for i, val in enumerate(summary):
                    epoch_aux_accumulator[i] += val
                num_batches += 1
        if epoch_aux_accumulator is not None and num_batches > 0:
            avg_aux = [val / num_batches for val in epoch_aux_accumulator]
            print(f"Epoch {epoch}: avg_aux = {format_aux(avg_aux)}")

    return model


@torch.no_grad()
def evaluate(model, seq_len, vocab_size=vocab_size, k=1, n=10, show_results=True):
    success = 0
    for _ in range(n):
        x = torch.randint(1, vocab_size + 1, (seq_len,))
        y = torch.full((seq_len,), blank_id, dtype=torch.long)
        if k < seq_len:
            y[k:] = x[: seq_len - k]
        x_input = x.view(seq_len, 1)
        logits, hidden, aux = model(x_input)
        preds = logits.argmax(-1).squeeze(1)
        ok = torch.equal(preds, y)
        if show_results:
            print(f"ponder={format_aux(aux)} | x={x.tolist()} | preds={preds.tolist()} | target={y.tolist()}")
        success += int(ok)
    return success


TASK = {
    "name": "repeat_previous",
    "train": train,
    "evaluate": evaluate,
    "input_dim": input_dim,
    "output_dim": output_dim,
}


def build_task(vocab_size, k=3):
    input_dim = vocab_size + 1
    return {
        "name": "repeat_previous",
        "train": train,
        "evaluate": evaluate,
        "input_dim": input_dim,
        "output_dim": input_dim,
        "train_kwargs": {"vocab_size": vocab_size, "k": k},
        "eval_kwargs": {"vocab_size": vocab_size, "k": k},
    }
