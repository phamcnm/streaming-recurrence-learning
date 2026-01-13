import torch.nn as nn

class GRUWrapper(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layernorm=False):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim)
        self.layernorm = layernorm
        if layernorm:
            self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x, h=None):
        x = self.ln1(x) if self.layernorm else x
        x, h = self.gru(x)         # (T, B, H)
        x = self.ln2(x) if self.layernorm else x
        return x, h, None
