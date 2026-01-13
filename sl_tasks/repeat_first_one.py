import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sl_batch import run_experiment
from utils import format_aux

vocab_size = 2  # excludes 0 as blank
input_dim = vocab_size + 1
output_dim = input_dim


class RepeatFirstOneDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(1, self.vocab_size + 1, (self.seq_len,), dtype=torch.long)
        y = x[0]
        return x, y


def train(model, seq_len, vocab_size=vocab_size, act_loss_coeff=0.01):
    dataset = RepeatFirstOneDataset(num_samples=10_000, seq_len=seq_len, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        epoch_aux_accumulator = None
        num_batches = 0
        for x, y in loader:
            x = x.transpose(0, 1)
            logits, hidden, aux = model(x)
            logits_T = logits[-1]
            loss = loss_fn(logits_T, y)
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
def evaluate(model, seq_len, vocab_size=vocab_size, n=10, show_results=True):
    success = 0
    for _ in range(n):
        x = torch.randint(1, vocab_size + 1, (seq_len,))
        x_input = x.view(seq_len, 1)
        logits, hidden, aux = model(x_input)
        pred = logits[-1].argmax(-1).item()
        target = x[0].item()
        if show_results:
            print(f"ponder={format_aux(aux)} | x={x.tolist()} | pred={pred} | target={target}")
        success += int(pred) == int(target)
    return success


TASK = {
    "name": "repeat_first_one",
    "train": train,
    "evaluate": evaluate,
    "input_dim": input_dim,
    "output_dim": output_dim,
}

def build_task(vocab_size):
    input_dim = vocab_size + 1
    return {
        "name": "repeat_first_one",
        "train": train,
        "evaluate": evaluate,
        "input_dim": input_dim,
        "output_dim": input_dim,
        "train_kwargs": {"vocab_size": vocab_size},
        "eval_kwargs": {"vocab_size": vocab_size},
    }


if __name__ == "__main__":
    run_experiment(
        train, evaluate, input_dim, output_dim=output_dim, seq_lens=[5], rnn_list=['lru', 'glru'], rnn_mode='sequential',
    )
