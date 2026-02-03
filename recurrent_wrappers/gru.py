import torch
import torch.nn as nn

class GRUWrapper(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layernorm=False):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim)
        self.layernorm = layernorm
        if layernorm:
            self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x, h=None, done=None):
        x = self.ln1(x) if self.layernorm else x
        T, B, _ = x.shape
        device, dtype = x.device, x.dtype

        if h is None:
            h = torch.zeros(1, B, self.gru.hidden_size, device=device, dtype=dtype)
        h = h if h.dim() == 3 else h.unsqueeze(0) 

        if done is None:
            y, h = self.gru(x, h)
            y = self.ln2(y) if self.layernorm else y
            return y, h, None

        outs = []
        for t in range(T):
            y_t, h = self.gru(x[t:t+1], h)
            outs.append(y_t)
            h = h * (1.0 - done[t]).to(h.dtype).view(1, B, 1)

        y = torch.cat(outs, dim=0)                        # (T,B,H)
        y = self.ln2(y) if self.layernorm else y
        return y, h.squeeze(0), None
