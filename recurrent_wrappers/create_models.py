from recurrent_wrappers.mlp import MLP
from recurrent_wrappers.gru import GRUWrapper
from recurrent_wrappers.rnn import RNNWrapper
from recurrent_wrappers.lru import LRUWrapper
from recurrent_units.mapping import LRU_MAPPING

def create_model(name, embed_dim, hidden_dim, rnn_mode='act', layernorm=False):
    # return the rnn component and output dim
    if name == "mlp":
        return MLP(embed_dim, hidden_dim, layernorm=layernorm), hidden_dim
    if name == "rnn":
        return RNNWrapper(embed_dim, hidden_dim, layernorm=layernorm), hidden_dim
    elif name == "gru":
        return GRUWrapper(embed_dim, hidden_dim, layernorm=layernorm), hidden_dim
    elif name in LRU_MAPPING:
        return LRUWrapper(
            recurrent_unit=name,
            d_model=embed_dim,
            d_state=hidden_dim,
            mode=rnn_mode,
            layernorm=layernorm,
        ), embed_dim
    else:
        raise ValueError(name)
