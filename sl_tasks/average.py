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
output_dim = 1
def sampler(seq_len, vocab_size=vocab_size):
    """
    Sample (x, target_avg)
    first, target is sampled uniformly from possible sums
    then, x is constructed with such target
    """
    assert vocab_size >= 1
    min_sum = 1 * seq_len
    max_sum = vocab_size * seq_len
    target_sum = torch.randint(min_sum, max_sum + 1, (1,), dtype=torch.long).item()
    x = torch.empty(seq_len, dtype=torch.long)
    remaining_sum = target_sum
    remaining_positions = seq_len
    for t in range(seq_len):
        remaining_positions -= 1
        min_val = max(1, remaining_sum - vocab_size * remaining_positions)
        max_val = min(vocab_size, remaining_sum - 1 * remaining_positions)
        v = torch.randint(min_val, max_val + 1, (1,), dtype=torch.long).item()
        x[t] = v
        remaining_sum -= v
    assert remaining_sum == 0
    assert 1 <= x.min() and x.max() <= vocab_size
    assert x.sum().item() == target_sum
    target_avg = target_sum / seq_len
    return x, target_avg

class AvgSequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        x, target_avg = sampler(seq_len=self.seq_len, vocab_size=self.vocab_size)
        y = torch.zeros(self.seq_len, dtype=torch.float)
        y[-1] = float(target_avg)
        return x, y
    
def train(
    model,
    seq_len,
    vocab_size=vocab_size,
    num_samples=10000,
    num_epochs=10,
    act_loss_coeff=0.01,
    return_loss_history=False,
):
    dataset = AvgSequenceDataset(num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    loss_history = []
    for epoch in range(num_epochs):
        epoch_aux_accumulator = None
        num_batches = 0
        epoch_loss_sum = 0.0
        for x, y in loader:
            x, y = x.transpose(0, 1), y.transpose(0, 1)  # (B, T) -> (T, B)
            preds, hidden, aux = model(x)
            preds = preds.squeeze(-1)  # (T, B, 1) -> (T, B)
            loss = loss_fn(preds, y)
            if isinstance(aux, dict) and aux.get("ponder_cost") is not None:
                loss = loss + act_loss_coeff * aux["ponder_cost"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.item())
            num_batches += 1

            summary = aux.get("summary") if isinstance(aux, dict) else aux
            if isinstance(summary, list):
                if epoch_aux_accumulator is None:
                    epoch_aux_accumulator = [0.0] * len(summary)
                for i, val in enumerate(summary):
                    epoch_aux_accumulator[i] += val
        if epoch_aux_accumulator is not None and num_batches > 0:
            avg_aux = [val / num_batches for val in epoch_aux_accumulator]
            print(f"Epoch {epoch}: avg_aux = {format_aux(avg_aux)}")
        if num_batches > 0:
            loss_history.append(epoch_loss_sum / num_batches)
    if return_loss_history:
        return model, loss_history
    return model

@torch.no_grad()
def evaluate(model, seq_len, vocab_size=vocab_size, n=10, show_results=True):
    errors = []
    for _ in range(n):
        x, target_avg = sampler(seq_len=seq_len, vocab_size=vocab_size)
        x_input = x.view(seq_len, 1)
        preds, hidden, aux = model(x_input)
        preds = preds.squeeze(-1).squeeze(-1)
        preds = torch.round(preds * 10) / 10.0
        preds_disp = [float(f"{v:.1f}") for v in preds.tolist()]
        y = torch.zeros(seq_len, dtype=torch.float)
        y[-1] = float(target_avg)
        pred_T = preds[-1].item()
        target_T = y[-1].item()
        errors.append(abs(pred_T - target_T))
        if show_results:
            print(f"ponder={format_aux(aux)} | x={x.tolist()} | pred={preds_disp} | target={y.tolist()} | final={pred_T:.3f}/{target_T:.3f}")
    return sum(errors) / n

TASK = {
    "name": "average",
    "train": train,
    "evaluate": evaluate,
    "input_dim": input_dim,
    "output_dim": output_dim,
}

def build_task(vocab_size):
    input_dim = vocab_size + 1
    return {
        "name": "average",
        "train": train,
        "evaluate": evaluate,
        "input_dim": input_dim,
        "output_dim": 1,
        "train_kwargs": {"vocab_size": vocab_size},
        "eval_kwargs": {"vocab_size": vocab_size},
    }
