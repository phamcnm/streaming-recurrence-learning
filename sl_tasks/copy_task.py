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

vocab_size = 5
S = 3
blank_id = 0
delim_id = 1
input_dim = vocab_size + 1
output_dim = input_dim
class CopyingTaskDataset(Dataset):
    def __init__(self, num_samples=10000, T=1000, vocab_size=vocab_size, S=S):
        assert vocab_size >= 3, "Need >=3 tokens (blank, delimiter, symbols)"
        self.num_samples, self.T, self.S = num_samples, T, S
        self.vocab_size = vocab_size
        self.blank, self.delim = blank_id, delim_id
        self.symbol_low, self.symbol_high = 2, vocab_size   # inclusive
        self.seq_len = T + 2 * S                                # T+20 when S=10

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        u = torch.randint(
            self.symbol_low, self.symbol_high + 1, (self.S,), dtype=torch.long
        )  # symbols to remember
        x = torch.full((self.seq_len,), self.blank, dtype=torch.long)
        x[:self.S] = u                                          # prefix: symbols
        x[self.S + self.T - 1] = self.delim                     # delimiter
        y = torch.full((self.seq_len,), self.blank, dtype=torch.long)
        y[-self.S:] = u                                         # target suffix
        return x, y
    
def train(model, seq_len, vocab_size=vocab_size, S=S, num_samples=10_000, num_epochs=10, act_loss_coeff=0.01):
    T = seq_len
    batch_size = 32
    dataset = CopyingTaskDataset(num_samples=num_samples, T=T, vocab_size=vocab_size, S=S)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        epoch_aux_accumulator = None
        num_batches = 0
        for x, y in loader:
            x, y = x.transpose(0, 1), y.transpose(0, 1)
            logits, hidden, aux = model(x)
            L, B, V = logits.shape
            loss = loss_fn(logits.reshape(L*B, V), y.reshape(L*B))
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
def evaluate(model, seq_len, vocab_size=vocab_size, S=S, n=10, show_results=True):
    T = seq_len
    success = 0
    for _ in range(n):
        L = T + 2 * S
        u = torch.randint(2, vocab_size + 1, (S,), dtype=torch.long)
        x = torch.full((L,), blank_id, dtype=torch.long)
        x[:S] = u
        x[S + T - 1] = delim_id
        y = torch.full((L,), blank_id, dtype=torch.long)
        y[-S:] = u
        logits, hidden, aux = model(x.unsqueeze(1))      # (L, 1, V)
        preds = logits.argmax(-1).squeeze(1)
        ok = torch.equal(preds, y)
        if show_results:
            print(f"ponder={format_aux(aux)} | x={x.tolist()} | preds={preds.tolist()} | y={y.tolist()}")
        success += int(ok)
    return success

TASK = {
    "name": "copy",
    "train": train,
    "evaluate": evaluate,
    "input_dim": input_dim,
    "output_dim": output_dim,
}

def build_task(vocab_size, S=S):
    input_dim = vocab_size + 1
    return {
        "name": "copy",
        "train": train,
        "evaluate": evaluate,
        "input_dim": input_dim,
        "output_dim": input_dim,
        "train_kwargs": {"vocab_size": vocab_size, "S": S},
        "eval_kwargs": {"vocab_size": vocab_size, "S": S},
    }

if __name__ == "__main__":
    run_experiment(
        train, evaluate, input_dim, output_dim=output_dim, seq_lens=[10], rnn_list=['lru', 'glru', 'gru'], rnn_mode='sequential',
    )
