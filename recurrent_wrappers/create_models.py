from recurrent_wrappers.mlp import MLP
from recurrent_wrappers.gru import GRUWrapper
from recurrent_wrappers.mynet_glu import MyNetGLU
from recurrent_wrappers.rnn import RNNWrapper
from recurrent_wrappers.simple import SimpleLRUWrapper
from recurrent_wrappers.mynet import MyNet
from recurrent_wrappers.bestnet import BestNet
from recurrent_wrappers.simba import SimbaWrapper
from recurrent_wrappers.memora import Memora
from recurrent_wrappers.bestnet_bilinear import BestNetBilinear
from recurrent_units.mapping import LRU_MAPPING

WRAPPERS = (SimpleLRUWrapper, SimbaWrapper, MyNet, BestNet, MyNetGLU, BestNetBilinear)
arch_map = {
    'simple':      SimpleLRUWrapper,
    'mynet':       MyNet,
    'bestnet':     BestNet,
    'simba':       SimbaWrapper,
    'memora':      Memora,
    'mynet_glu':   MyNetGLU,
    'bestnet_bilinear': BestNetBilinear,
}
default_arch = 'bestnet'

def create_model(name, embed_dim, hidden_dim, arch=default_arch, rnn_mode='sequential', use_layernorm=False):
    # return the rnn component and output dim
    if arch == 'none':
        if name == "mlp":
            return MLP(embed_dim, hidden_dim, use_layernorm=use_layernorm), hidden_dim
        elif name == "rnn":
            return RNNWrapper(embed_dim, hidden_dim, use_layernorm=use_layernorm), hidden_dim
        elif name == "gru":
            return GRUWrapper(embed_dim, hidden_dim, use_layernorm=use_layernorm), hidden_dim
        else:
            if name in LRU_MAPPING:
                arch = default_arch
            else:
                raise ValueError(name)
    if arch != 'none':
        arch_wrapper = arch_map[arch]
        use_nonlinearity = name not in ['gru', 'rnn']
        return arch_wrapper(
            recurrent_unit=name,
            d_model=embed_dim,
            d_state=hidden_dim,
            mode=rnn_mode,
            use_layernorm=use_layernorm,
            use_nonlinearity=use_nonlinearity,
        ), embed_dim
