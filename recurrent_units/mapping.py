from recurrent_units.gated_lru_real_bptt import GatedRealLRUBPTT
from recurrent_units.gated_lru_real_bptt_minimal import GatedRealLRUBPTTMin
from recurrent_units.real_lru_bptt import RealLRUBPTT

LRU_MAPPING = {
    'glru': GatedRealLRUBPTT,
    'lru': RealLRUBPTT,
    'glru_min': GatedRealLRUBPTTMin
}
