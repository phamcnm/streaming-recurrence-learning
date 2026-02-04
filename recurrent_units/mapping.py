from recurrent_units.gated_lru_real_bptt import GatedRealLRUBPTT
from recurrent_units.gated_lru_real_bptt_minimal import GatedRealLRUBPTTMin
from recurrent_units.real_lru_bptt import RealLRUBPTT
from recurrent_units.gated_lru_real_rtrl import GatedRealLRURTRL
from recurrent_units.real_lru_rtrl import RealLRURTRL
from recurrent_units.dropin_linear import DropinLinear

LRU_MAPPING = {
    'lin': DropinLinear,
    'glru': GatedRealLRUBPTT,
    'lru': RealLRUBPTT,
    'glru_min': GatedRealLRUBPTTMin,
    'lru_rtrl': RealLRURTRL,
    'glru_rtrl': GatedRealLRURTRL
}
