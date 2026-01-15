# models/__init__.py

from .minkunet import MinkUNetSparse
from .minkunet_attention import MinkUNetSparseAttention
from .minkunet_attention import MinkUNetSparseAttentionNoEnc, MinkUNetSparseAttentionNoFlash, MinkUNetSparseAttentionNoFlashEnc

MODEL_REGISTRY = {

    ## this is base model with sparse attention
    "attn_base": MinkUNetSparseAttention,

    ## varitions of the sparse attention module
    "attn_noenc" : MinkUNetSparseAttentionNoEnc,
    "attn_noflash":  MinkUNetSparseAttentionNoFlash,
    "attn_noflashenc": MinkUNetSparseAttentionNoFlashEnc,

    "base" : MinkUNetSparse,
}
