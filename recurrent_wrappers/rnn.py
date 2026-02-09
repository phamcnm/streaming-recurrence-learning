import torch.nn as nn

class RNNWrapper(nn.Module):
    def __init__(self, embed_dim, hidden_dim, use_layernorm=False, **kwargs):
        super().__init__()
        self.rnn = nn.RNN(input_size=embed_dim,
            hidden_size=hidden_dim,
            nonlinearity="tanh",
            batch_first=False)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x, hidden=None, done=None, **kwargs):
        x = self.ln1(x) if self.use_layernorm else x
        x, hidden = self.rnn(x)        # (T, B, H)
        x = self.ln2(x) if self.use_layernorm else x
        return x, hidden, None
