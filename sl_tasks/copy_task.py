import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sl_batch import run_experiment
from utils import format_aux

vocab_size = 2
S = 1
blank_id = 0
delim_id = 1
num_classes = vocab_size + 2  # blank + delim + symbols
input_dim = num_classes
output_dim = input_dim


class CopyingTaskDataset(Dataset):
    def __init__(
        self,
        num_samples=10000,
        T=1000,
        vocab_size=vocab_size,
        S=S,
        one_hot=True,
        variable_k=False,
    ):
        self.num_samples, self.T, self.S = num_samples, T, S
        self.vocab_size = vocab_size
        self.blank, self.delim = blank_id, delim_id
        self.symbol_low, self.symbol_high = 2, vocab_size + 1   # inclusive
        self.seq_len = T + 2 * S
        self.num_classes = vocab_size + 2
        self.one_hot = one_hot
        self.variable_k = variable_k

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        k_sample = None
        if self.variable_k:
            k_min = max(1, int(0.75 * self.T))
            k_sample = torch.randint(k_min, self.T + 1, (1,), dtype=torch.long).item()
        u = torch.randint(self.symbol_low, self.symbol_high + 1, (self.S,), dtype=torch.long)
        x = torch.full((self.seq_len,), self.blank, dtype=torch.long)
        x[:self.S] = u
        if self.variable_k:
            x[self.S + k_sample - 1] = self.delim
        else:
            x[self.S + self.T - 1] = self.delim
        if self.one_hot:
            x = F.one_hot(x, num_classes=self.num_classes).float()
        y = torch.full((self.seq_len,), self.blank, dtype=torch.long)
        if self.variable_k:
            y[self.S + k_sample : self.S + k_sample + self.S] = u
        else:
            y[-self.S:] = u
        return x, y


class CopyingTaskOneHotDataset(CopyingTaskDataset):
    def __init__(self, num_samples=10000, T=1000, vocab_size=vocab_size, S=S):
        super().__init__(num_samples=num_samples, T=T, vocab_size=vocab_size, S=S, one_hot=True)


class CopyingTaskTokenDataset(CopyingTaskDataset):
    def __init__(self, num_samples=10000, T=1000, vocab_size=vocab_size, S=S):
        super().__init__(num_samples=num_samples, T=T, vocab_size=vocab_size, S=S, one_hot=False)


class CopyingTaskOneHotVariableKDataset(CopyingTaskDataset):
    def __init__(self, num_samples=10000, T=1000, vocab_size=vocab_size, S=S):
        super().__init__(
            num_samples=num_samples,
            T=T,
            vocab_size=vocab_size,
            S=S,
            one_hot=True,
            variable_k=True,
        )

def train(
    model,
    seq_len,
    vocab_size=vocab_size,
    S=S,
    one_hot=True,
    variable_k=False,
    num_samples=10_000,
    num_epochs=10,
    act_loss_coeff=0.01,
    return_loss_history=False,
    val_split=0.2,
    val_seed=1234,
):
    T = seq_len
    batch_size = 32
    dataset = CopyingTaskDataset(
        num_samples=num_samples,
        T=T,
        vocab_size=vocab_size,
        S=S,
        one_hot=one_hot,
        variable_k=variable_k,
    )
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
    loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    loss_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(num_epochs):
        epoch_aux_accumulator = None
        num_batches = 0
        epoch_loss_sum = 0.0
        for x, y in loader:
            x, y = x.transpose(0, 1), y.transpose(0, 1)   # x: (L,B,V), y: (L,B)
            logits, hidden, aux = model(x)
            L, B, V = logits.shape
            loss = loss_fn(logits.reshape(L*B, V), y.reshape(L*B))
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
                    logits, hidden, aux = model(x)
                    L, B, V = logits.shape
                    val_loss = loss_fn(logits.reshape(L*B, V), y.reshape(L*B))
                    if isinstance(aux, dict) and aux.get("ponder_cost") is not None:
                        val_loss = val_loss + act_loss_coeff * aux["ponder_cost"]
                    val_loss_sum += float(val_loss.item())
                    val_batches += 1
            val_loss_history.append(val_loss_sum / val_batches if val_batches > 0 else None)
            val_acc = evaluate(
                model,
                seq_len,
                vocab_size=vocab_size,
                S=S,
                one_hot=one_hot,
                variable_k=variable_k,
                show_results=False,
                loader=val_loader,
            )
            val_acc_history.append(val_acc)
        else:
            val_loss_history.append(None)
            val_acc_history.append(None)
    if return_loss_history:
        return model, loss_history, val_loss_history, val_acc_history
    return model

@torch.no_grad()
def evaluate(
    model,
    seq_len,
    vocab_size=vocab_size,
    S=S,
    one_hot=True,
    variable_k=False,
    n=20,
    show_results=True,
    loader=None,
):
    if loader is not None:
        success = 0
        total = 0
        for x, y in loader:
            x, y = x.transpose(0, 1), y.transpose(0, 1)
            logits, hidden, aux = model(x)
            preds = logits.argmax(-1)
            if preds.shape != y.shape:
                preds = preds.view_as(y)
            ok = (preds == y).all(dim=0)
            success += ok.sum().item()
            total += y.shape[1]
        return success / total if total > 0 else 0.0
    T = seq_len
    success = 0
    for _ in range(n):
        L = T + 2 * S
        k_sample = None
        if variable_k:
            k_min = max(1, int(0.75 * T))
            k_sample = torch.randint(k_min, T + 1, (1,), dtype=torch.long).item()
        u = torch.randint(2, vocab_size + 2, (S,), dtype=torch.long)
        x = torch.full((L,), blank_id, dtype=torch.long)
        x[:S] = u
        if variable_k:
            x[S + k_sample - 1] = delim_id
        else:
            x[S + T - 1] = delim_id
        y = torch.full((L,), blank_id, dtype=torch.long)
        if variable_k:
            y[S + k_sample : S + k_sample + S] = u
        else:
            y[-S:] = u
        if one_hot:
            x = F.one_hot(x, num_classes=vocab_size + 2).float()
            logits, hidden, aux = model(x.unsqueeze(1))      # (L, 1, V)
        else:
            logits, hidden, aux = model(x.unsqueeze(1))      # (L, 1)
        preds = logits.argmax(-1).squeeze(1)
        ok = torch.equal(preds, y)
        if show_results:
            x_tokens = x.argmax(-1).tolist() if one_hot else x.tolist()
            preds_list = preds.tolist()
            y_list = y.tolist()
            head = S
            if variable_k:
                delim_idx = S + k_sample - 1
            else:
                delim_idx = S + T - 1
            window_end = min(L, delim_idx + S + 1)
            x_view = x_tokens[:head] + ["..."] + x_tokens[delim_idx:window_end]
            preds_view = preds_list[:head] + ["..."] + preds_list[delim_idx:window_end]
            y_view = y_list[:head] + ["..."] + y_list[delim_idx:window_end]
            print(f"ponder={format_aux(aux)} | x={x_view} | preds={preds_view} | y={y_view}")
        success += int(ok)
    return success/n

TASK = {
    "name": "copy",
    "train": train,
    "evaluate": evaluate,
    "input_dim": input_dim,
    "output_dim": output_dim,
    "input_is_one_hot": True,
}

def build_task(vocab_size, S=S, one_hot=True, variable_k=False):
    input_dim = vocab_size + 2
    return {
        "name": "copy",
        "train": train,
        "evaluate": evaluate,
        "input_dim": input_dim,
        "output_dim": input_dim,
        "input_is_one_hot": one_hot,
        "train_kwargs": {"vocab_size": vocab_size, "S": S, "one_hot": one_hot, "variable_k": variable_k},
        "eval_kwargs": {"vocab_size": vocab_size, "S": S, "one_hot": one_hot, "variable_k": variable_k},
    }

if __name__ == "__main__":
    run_experiment(
        train, evaluate, input_dim, output_dim=output_dim, seq_lens=[10],
        rnn_list=['lru', 'glru', 'gru'], rnn_mode='sequential',
    )
