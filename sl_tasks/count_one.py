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
    Sample (x, q, target_pct)
    first, q and target are sampled uniformly
    then, x is constructed with such target
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
    target_pct = target / seq_len
    return x, q, target_pct

class CountRecallDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        x, q, target_pct = sampler(seq_len=self.seq_len, vocab_size=self.vocab_size)
        q_tensor = torch.tensor([q], dtype=torch.long)
        x_input = torch.cat([x, q_tensor], dim=0)  # (seq_len + 1,)
        y = torch.tensor(float(target_pct)).repeat(self.seq_len + 1)
        return x_input, y
    
def train(
    model,
    seq_len,
    vocab_size=vocab_size,
    num_samples=10000,
    num_epochs=10,
    act_loss_coeff=0.01,
    return_loss_history=False,
    val_split=0.2,
    val_seed=1234,
):
    dataset = CountRecallDataset(num_samples=num_samples, seq_len=seq_len, vocab_size=vocab_size)
    import random
    import numpy as np
    rng_state = torch.random.get_rng_state()
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch.manual_seed(val_seed)
    random.seed(val_seed)
    np.random.seed(val_seed)
    samples = [dataset[i] for i in range(num_samples)]
    torch.random.set_rng_state(rng_state)
    random.setstate(py_state)
    np.random.set_state(np_state)
    split_idx = int((1 - val_split) * num_samples)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    loader = DataLoader(train_samples, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=32, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    loss_history = []
    val_loss_history = []
    for epoch in range(num_epochs):
        epoch_aux_accumulator = None
        num_batches = 0
        epoch_loss_sum = 0.0
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
        if val_loader is not None and len(val_samples) > 0:
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.transpose(0, 1), y.transpose(0, 1)
                    preds, hidden, aux = model(x)
                    pred_T = preds[-1, :, 0]
                    y_T = y[-1]
                    val_loss = loss_fn(pred_T, y_T)
                    if isinstance(aux, dict) and aux.get("ponder_cost") is not None:
                        val_loss = val_loss + act_loss_coeff * aux["ponder_cost"]
                    val_loss_sum += float(val_loss.item())
                    val_batches += 1
            val_loss_history.append(val_loss_sum / val_batches if val_batches > 0 else None)
        else:
            val_loss_history.append(None)
    if return_loss_history:
        return model, loss_history, val_loss_history
    return model

@torch.no_grad()
def evaluate(model, seq_len, vocab_size=vocab_size, n=10, show_results=True):
    errors = []
    for _ in range(n):
        x, q, target_pct = sampler(seq_len=seq_len, vocab_size=vocab_size)
        x_seq = x.view(seq_len, 1)
        q_tensor = torch.tensor([[q]], dtype=torch.long)
        x_input = torch.cat([x_seq, q_tensor], dim=0)
        preds, hidden, aux = model(x_input)
        pred = preds[-1, 0, 0].item()
        errors.append(abs(pred - target_pct))
        if show_results:
            print(f"ponder={format_aux(aux)} | x={x.tolist()} | q={q} | target={target_pct} | pred={pred}")
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
