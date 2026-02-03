import torch.nn as nn
import torch.nn.functional as F
import torch, math
from typing import Optional
from recurrent_units.mapping import LRU_MAPPING

class GatedMLP(nn.Module):
    def __init__(self, d_model, gated_mlp_expansion=3, dropout=0.0, activation=F.gelu, bias=False):
        super().__init__()

        expanded_dim = int(gated_mlp_expansion * d_model)

        self.combined_proj = nn.Linear(d_model, 2 * expanded_dim, bias=bias)

        self.gate_norm = nn.LayerNorm(expanded_dim, elementwise_affine=False)
        self.value_norm = nn.LayerNorm(expanded_dim, elementwise_affine=False)
        
        self.contract = nn.Linear(expanded_dim, d_model, bias=bias)
        self.contract_norm = nn.LayerNorm(expanded_dim, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = activation

    def forward(self, x):
        # Get combined projection and split into gate and value parts
        combined = self.combined_proj(x)
        pre_gate, value = torch.chunk(combined, 2, dim=-1)
        
        # Apply layer normalization to each part
        pre_gate = self.gate_norm(pre_gate)
        value = self.value_norm(value)
        
        # Element-wise multiplication of gate and value
        out = self.activation(pre_gate) * value

        # Project back to original dimension
        out = self.contract_norm(out)
        out = self.activation(out)
        out = self.contract(out)
        out = self.dropout(out)

        return out

class GatedRecurrence(nn.Module):
    """Handles the core recurrent computation"""
    def __init__(self, recurrent_class, d_model, d_state, dropout=0.0, activation=F.gelu, bias=False, inner_loops: int = 1):
        super().__init__()

        # Gating mechanism
        self.gating_proj = nn.Linear(d_model, d_state, bias=bias)
        self.post_norm_gate = nn.LayerNorm(d_state, elementwise_affine=False)
        self.contract = nn.Linear(d_state, d_model, bias=bias)
        self.contract_norm = nn.LayerNorm(d_state, elementwise_affine=False)

        # Recurrent processing
        self.rnn_post_norm = nn.LayerNorm(d_state, elementwise_affine=False)
        self.recurrent_unit = self.rnn = LRU_MAPPING[recurrent_class](
            in_features=d_model,
            out_features=d_state,
            state_features=d_state,
            rmin=0.9,
            rmax=0.999,
            mode='sequential',)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Activation function
        self.activation = activation
        # Inner loops for the recurrent unit updates
        self.inner_loops = int(inner_loops)

    def forward(self, x, hidden=None, done=None, apply_change=True, inner_loops: Optional[int] = None):

        residual = x

        # Process through recurrent unit
        loops = self.inner_loops if inner_loops is None else int(inner_loops)
        x, next_hidden, aux = self.recurrent_unit.forward(x, hidden, done=done, inner_loops=loops)
        if isinstance(aux, tuple):
            summary = [round(a.float().mean().item(), 1) for a in aux[2]]
            if len(aux) > 3:
                aux = {"summary": summary, "ponder_cost": aux[3]}
            else:
                aux = summary
        x = self.rnn_post_norm(x)

        # Apply output projection and dropout
        gate = self.activation(self.post_norm_gate(self.gating_proj(residual)))
        x = x * gate

        # Contract the output
        x = self.contract_norm(x)
        x = self.activation(x)
        x = self.contract(x)
        x = self.dropout(x)
        
        return x, next_hidden, aux

class ResidualBlock(nn.Module):
    """Residual block with recurrent layer, convolution, and MLP"""
    def __init__(self, recurrent_class=None, d_model=64, d_state=128,
                 dropout=0.0, activation=F.gelu, gated_mlp_expansion=3, bias=False, inner_loops: int = 1):
        super().__init__()
        
        # Gated RNN
        self.recurrent = GatedRecurrence(
            recurrent_class=recurrent_class,
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            activation=activation,
            bias=bias,
            inner_loops=inner_loops
        )
        
        # Gated MLP
        self.gated_mlp = GatedMLP(
            d_model=d_model,
            gated_mlp_expansion=gated_mlp_expansion,
            dropout=dropout,
            activation=activation,
            bias=bias
        )
        
        self.rnn_norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.mlp_norm = nn.LayerNorm(d_model, elementwise_affine=True)


    def forward(self, x, hidden=None, done=None, apply_change=True, inner_loops: Optional[int] = None):

        # Apply the gated RNN
        rnn_out, next_hidden, aux = self.recurrent(self.rnn_norm(x), hidden, done=done, apply_change=apply_change, inner_loops=inner_loops)
        x = rnn_out + x

        # Apply the gated MLP
        x = self.gated_mlp(self.mlp_norm(x)) + x
        
        return x, next_hidden, aux

    def reset_state(self):
        self.recurrent.recurrent_unit.reset_state()

class Memora(nn.Module):
    def __init__(self, recurrent_unit='lru',
                 d_model=64, d_state=128, num_layers=1, dropout=0.0,
                 activation=F.gelu, gated_mlp_expansion=2, bias=False, inner_loops: int = 1,
                 mode='sequential', ponder_eps=0.1, ponder_n=4, layernorm=True,):
        super().__init__()
        if recurrent_unit is None:
            raise ValueError("recurrent_unit must be specified")
        # self.input_proj = nn.Linear(input_size, d_model, bias=bias)
        self.input_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.inner_loops = int(inner_loops)
        # Create the residual blocks
        self.block = ResidualBlock(
            recurrent_class=recurrent_unit,
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            activation=activation,
            gated_mlp_expansion=gated_mlp_expansion,
            bias=bias,
            inner_loops=self.inner_loops
        )

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        self.num_layers = num_layers

        # Apply custom initialization
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('contract.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=1.0/math.sqrt(2 * self.num_layers * p.shape[1]))

    def reset_state(self):
        for block in self.blocks:
            block.reset_state()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0/math.sqrt(module.in_features))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, hidden=None, done=None, apply_change=True, inner_loops: Optional[int] = None):
        # x = self.input_proj(x)  # [seq_len, batch_size, hidden_size]
        x = self.input_norm(x)

        # Process through all recurrent blocks sequentially
        loops = self.inner_loops if inner_loops is None else int(inner_loops)
        x, hidden, aux = self.block(x, hidden=hidden, done=done, apply_change=apply_change, inner_loops=loops)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x, hidden, aux