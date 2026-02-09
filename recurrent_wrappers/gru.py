import torch
import torch.nn as nn

class GRUWrapper(nn.Module):
    def __init__(self, embed_dim, hidden_dim, use_layernorm=False, **kwargs):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x, hidden=None, done=None, **kwargs):
        x = self.ln1(x) if self.use_layernorm else x
        T, B, _ = x.shape
        device, dtype = x.device, x.dtype

        if hidden is None:
            hidden = torch.zeros(1, B, self.gru.hidden_size, device=device, dtype=dtype)
        hidden = hidden if hidden.dim() == 3 else hidden.unsqueeze(0)

        if done is None:
            y, hidden = self.gru(x, hidden)
            y = self.ln2(y) if self.use_layernorm else y
            return y, hidden, None

        outs = []
        for t in range(T):
            y_t, hidden = self.gru(x[t:t+1], hidden)
            outs.append(y_t)
            hidden = hidden * (1.0 - done[t]).to(hidden.dtype).view(1, B, 1)

        y = torch.cat(outs, dim=0)                        # (T,B,H)
        y = self.ln2(y) if self.use_layernorm else y
        return y, hidden.squeeze(0), None
