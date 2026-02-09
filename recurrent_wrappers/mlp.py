import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, use_layernorm=False, **kwargs):
        super().__init__()
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(hidden_dim)

    def forward(self, x, h=None, **kwargs):
        x = self.ln1(x) if self.use_layernorm else x
        x = F.relu(x)
        x = self.fc(x)
        x = self.ln2(x) if self.use_layernorm else x
        x = F.relu(x)
        return x, None, None
