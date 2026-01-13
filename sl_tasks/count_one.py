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

vocab_size = 3
input_dim = vocab_size + 1
output_dim = 1
def sampler(seq_len, vocab_size=vocab_size):
    """
    Sample (x, q, target)
    first, q and target are sampled uniformly
    then, x is constructed with such q and target
    """
    q = torch.randint(1, vocab_size + 1, (1,), dtype=torch.long).item()
    target = torch.randint(0, seq_len + 1, (1,), dtype=torch.long).item()
    x = torch.empty(seq_len, dtype=torch.long)
    if vocab_size == 1:
        target = seq_len
        x.fill_(q)
    else:
        non_q = torch.randint(1, vocab_size, (seq_len,), dtype=torch.long)
        non_q[non_q >= q] += 1
        x[:] = non_q
        if target > 0:
            idx = torch.randperm(seq_len)[:target]
            x[idx] = q
    return x, q, target

class CountRecallDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        x, q, target = sampler(seq_len=self.seq_len, vocab_size=self.vocab_size)
        q_tensor = torch.tensor([q], dtype=torch.long)
        x_input = torch.cat([x, q_tensor], dim=0)  # (seq_len + 1,)
        y = torch.tensor(float(target)).repeat(self.seq_len + 1)
        return x_input, y
    
def train(model, seq_len, vocab_size=vocab_size, act_loss_coeff=0.01):
    dataset = CountRecallDataset(num_samples=10000, seq_len=seq_len, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(10):
        epoch_aux_accumulator = None
        num_batches = 0
        for x, y in loader:
            x, y = x.transpose(0, 1), y.transpose(0, 1)  # (B, T) -> (T, B)
            preds, hidden, aux = model(x)  # (T, B, 1)
            pred_T = preds[-1, :, 0]
            y_T = y[-1]
            loss = loss_fn(pred_T, y_T)
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
    errors = []
    for _ in range(n):
        x, q, target = sampler(seq_len=seq_len, vocab_size=vocab_size)
        x_seq = x.view(seq_len, 1)
        q_tensor = torch.tensor([[q]], dtype=torch.long)
        x_input = torch.cat([x_seq, q_tensor], dim=0)
        preds, hidden, aux = model(x_input)
        pred = preds[-1, 0, 0].item()
        errors.append(abs(pred - target))
        if show_results:
            print(f"ponder={format_aux(aux)} | x={x.tolist()} | q={q} | target={target} | pred={pred}")
    return sum(errors) / n

TASK = {
    "name": "count_one",
    "train": train,
    "evaluate": evaluate,
    "input_dim": input_dim,
    "output_dim": output_dim,
}

def build_task(vocab_size):
    input_dim = vocab_size + 1
    return {
        "name": "count_one",
        "train": train,
        "evaluate": evaluate,
        "input_dim": input_dim,
        "output_dim": 1,
        "train_kwargs": {"vocab_size": vocab_size},
        "eval_kwargs": {"vocab_size": vocab_size},
    }

if __name__ == "__main__":
    run_experiment(train, evaluate, input_dim, 
        output_dim=1, seq_lens=[10], rnn_list=['gru'], rnn_mode='sequential'
    )
