import torch
import torch.nn as nn

class DropinGRU(nn.Module):
    def __init__(self, in_features, out_features, state_features=None, bias=True, **kwargs):
        super().__init__()
        if state_features is None:
            state_features = out_features
        self.gru = nn.GRU(in_features, state_features, bias=bias)
        self.proj = nn.Identity() if state_features == out_features else nn.Linear(state_features, out_features, bias=False)

    def forward(self, x, hidden=None, done=None, **kwargs):
        seq_len, batch_size, _ = x.shape
        device, dtype = x.device, x.dtype

        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.gru.hidden_size, device=device, dtype=dtype)
        elif hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        if done is None:
            y, hidden = self.gru(x, hidden)
        else:
            outs = []
            for t in range(seq_len):
                y_t, hidden = self.gru(x[t:t + 1], hidden)
                outs.append(y_t)
                hidden = hidden * (1.0 - done[t]).to(hidden.dtype).view(1, batch_size, 1)
            y = torch.cat(outs, dim=0)

        y = self.proj(y)
        return y, hidden.squeeze(0), None
