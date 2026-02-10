import torch.nn as nn
from recurrent_units.mapping import LRU_MAPPING
import torch.nn.functional as F


class BestNetBilinear(nn.Module):
    # LN -> skip -> Activation -> RNN -> LN -> Activation -> Linear x Linear -> LN -> add skip
    def __init__(
        self, 
        recurrent_unit='lru', 
        d_model=64, 
        d_state=64, 
        rmin=0.0, 
        rmax=1.0, 
        mode='sequential', 
        ponder_eps=0.1, 
        ponder_n=4, 
        use_layernorm=True,
        use_nonlinearity=True,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_state)
            self.ln3_left = nn.LayerNorm(d_model, elementwise_affine=False)
            self.ln3_right = nn.LayerNorm(d_model, elementwise_affine=False)
            self.ln4 = nn.LayerNorm(d_model)
        self.use_nonlinearity = use_nonlinearity
        
        self.rnn = LRU_MAPPING[recurrent_unit](
            in_features=d_model,
            out_features=d_state,
            state_features=d_state,
            rmin=rmin,
            rmax=rmax,
            mode=mode,
            # epsilon=ponder_eps,
            # max_steps=ponder_n,
        )
        self.mlp_left = nn.Linear(d_state, d_model)
        self.mlp_right = nn.Linear(d_state, d_model)
        self._last_seq_grad = None

    def forward(self, x, hidden=None, done=None, track_seq_grad=False, track_activity=False, **kwargs):
        x = self.ln1(x) if self.use_layernorm else x
        skip = x.clone()  # [seq_len, batch, d_model]
        if self.use_nonlinearity:
            x = F.leaky_relu(x)
        
        if track_seq_grad:
            x.retain_grad()
            x.register_hook(self._save_seq_grad)

        x, new_hidden, aux = self.rnn.forward(x, hidden=hidden, done=done, **kwargs)
        
        if isinstance(aux, tuple):
            summary = [round(a.float().mean().item(), 1) for a in aux[2]]
            if len(aux) > 3:
                aux = {"summary": summary, "ponder_cost": aux[3]}
            else:
                aux = summary
        
        x = self.ln2(x) if self.use_layernorm else x
        if self.use_nonlinearity:
            x = F.leaky_relu(x)
        rnn_out = x if track_activity else None
        pre_bilinear = x
        x_left = self.mlp_left(pre_bilinear)
        x_left = self.ln3_left(x_left) if self.use_layernorm else x_left
        x_right = self.mlp_right(pre_bilinear)
        x_right = self.ln3_right(x_right) if self.use_layernorm else x_right
        x = x_left * x_right
        x = self.ln4(x) if self.use_layernorm else x
        # x = F.leaky_relu(x)
        mlp_out = x if track_activity else None
        x = x + skip

        if track_activity:
            activity = {"rnn_out": rnn_out, "mlp_out": mlp_out}
            if aux is None:
                aux = activity
            elif isinstance(aux, dict):
                aux = {**aux, **activity}
            else:
                aux = {"aux": aux, **activity}
        
        return x, new_hidden, aux

    def _save_seq_grad(self, grad):
        # grad shape: [seq_len, batch, d_state]
        self._last_seq_grad = grad.detach().norm(dim=(1, 2))
    
    def reset_state(self):
        pass
