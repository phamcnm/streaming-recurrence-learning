import torch.nn as nn
from recurrent_units.mapping import LRU_MAPPING
import torch.nn.functional as F


class SimpleLRUWrapper(nn.Module):
    # RNN -> Activation -> MLP -> LN
    def __init__(
        self, 
        recurrent_unit='lru', 
        d_model=64, 
        d_state=64, 
        rmin=0.9, 
        rmax=0.999, 
        mode='sequential', 
        ponder_eps=0.1, 
        ponder_n=4, 
        layernorm=True,
    ):
        super().__init__()
        self.mode = mode
        self.layernorm = layernorm
        if layernorm:
            self.ln3 = nn.LayerNorm(d_model)
        
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
        self.mlp = nn.Linear(d_state, d_model)
        self._last_seq_grad = None

    def forward(self, x, hidden=None, done=None, track_seq_grad=False, **kwargs):
        if track_seq_grad:
            x.retain_grad()
            x.register_hook(self._save_seq_grad)

        x, new_hidden, aux = self.rnn.forward(x, hidden, done=done, **kwargs)
        if isinstance(aux, tuple):
            summary = [round(a.float().mean().item(), 1) for a in aux[2]]
            if len(aux) > 3:
                aux = {"summary": summary, "ponder_cost": aux[3]}
            else:
                aux = summary
        
        x = self.mlp(x)
        x = F.leaky_relu(x)
        x = self.ln3(x) if self.layernorm else x
        
        return x, new_hidden, aux

    def _save_seq_grad(self, grad):
        # grad shape: [seq_len, batch, d_state]
        self._last_seq_grad = grad.detach().norm(dim=(1, 2))
    
    def reset_state(self):
        pass
