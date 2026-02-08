import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layernorm=False):
        super().__init__()
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.layernorm = layernorm
        if layernorm: self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(hidden_dim)

    def forward(self, x, h=None):
        x = self.ln1(x) if self.layernorm else x
        x = F.relu(x)
        x = self.fc(x)
        x = self.ln2(x) if self.layernorm else x
        x = F.relu(x)
        return x, None, None
