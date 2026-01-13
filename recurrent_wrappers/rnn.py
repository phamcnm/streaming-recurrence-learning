import torch.nn as nn

class RNNWrapper(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layernorm=False):
        super().__init__()
        self.rnn = nn.RNN(input_size=embed_dim,
            hidden_size=hidden_dim,
            nonlinearity="tanh",
            batch_first=False)
        self.layernorm = layernorm
        if layernorm:
            self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x, h=None):
        x = self.ln1(x) if self.layernorm else x
        x, h = self.rnn(x)        # (T, B, H)
        x = self.ln2(x) if self.layernorm else x
        return x, h, None
