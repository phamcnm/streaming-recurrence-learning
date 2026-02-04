import torch.nn as nn

class DropinLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, **kwargs):
        return self.linear(x), None, None
