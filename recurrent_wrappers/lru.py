import torch.nn as nn
from recurrent_units.mapping import LRU_MAPPING
import torch.nn.functional as F


class LRUWrapper(nn.Module):
    # LN -> skip -> Activation -> RNN -> LN -> Activation -> MLP -> LN -> add skip
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
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_state)
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

    def forward(self, x, hidden=None, done=None, **kwargs):
        x = self.ln1(x) if self.layernorm else x
        skip = x.clone()  # [seq_len, batch, d_model]
        x = F.leaky_relu(x)
        
        x, new_hidden, aux = self.rnn.forward(x, hidden, done=done, **kwargs)
        if isinstance(aux, tuple):
            summary = [round(a.float().mean().item(), 1) for a in aux[2]]
            if len(aux) > 3:
                aux = {"summary": summary, "ponder_cost": aux[3]}
            else:
                aux = summary
        
        x = self.ln2(x) if self.layernorm else x
        x = F.leaky_relu(x)
        x = self.mlp(x)
        x = self.ln3(x) if self.layernorm else x
        x = x + skip
        
        return x, new_hidden, aux
    
    def reset_state(self):
        pass