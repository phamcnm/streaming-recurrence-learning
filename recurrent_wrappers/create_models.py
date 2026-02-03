from recurrent_wrappers.mlp import MLP
from recurrent_wrappers.gru import GRUWrapper
from recurrent_wrappers.rnn import RNNWrapper
from recurrent_wrappers.simple import SimpleLRUWrapper
from recurrent_wrappers.mynet import MyNet
from recurrent_wrappers.bestnet import BestNet
from recurrent_wrappers.simba import SimbaWrapper
from recurrent_wrappers.memora import Memora
from recurrent_units.mapping import LRU_MAPPING

WRAPPERS = (SimpleLRUWrapper, SimbaWrapper, MyNet, BestNet)
arch_map = {'simple': SimpleLRUWrapper, 'mynet': MyNet, 'bestnet': BestNet, 'simba': SimbaWrapper, 'memora': Memora}

def create_model(name, embed_dim, hidden_dim, arch='mynet', rnn_mode='act', layernorm=False):
    # return the rnn component and output dim
    if name == "mlp":
        return MLP(embed_dim, hidden_dim, layernorm=layernorm), hidden_dim
    if name == "rnn":
        return RNNWrapper(embed_dim, hidden_dim, layernorm=layernorm), hidden_dim
    elif name == "gru":
        return GRUWrapper(embed_dim, hidden_dim, layernorm=layernorm), hidden_dim
    elif name in LRU_MAPPING:
        arch_wrapper = arch_map[arch]
        return arch_wrapper(
            recurrent_unit=name,
            d_model=embed_dim,
            d_state=hidden_dim,
            mode=rnn_mode,
            layernorm=layernorm,
        ), embed_dim
    else:
        raise ValueError(name)
