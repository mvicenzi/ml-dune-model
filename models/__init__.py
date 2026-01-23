# models/__init__.py

from .minkunet import MinkUNetSparse
from .minkunet_attention import MinkUNetSparseAttention, MinkUNetSparseAttention125
from .minkunet_attention import MinkUNetSparseAttentionNoEnc, MinkUNetSparseAttentionNoFlash, MinkUNetSparseAttentionNoFlash125, MinkUNetSparseAttentionNoFlashEnc

MODEL_REGISTRY = {

    ## this is base model with sparse attention
    "attn_base": MinkUNetSparseAttention,
    "attn_base125": MinkUNetSparseAttention125,

    ## varitions of the sparse attention module
    "attn_noenc" : MinkUNetSparseAttentionNoEnc,
    "attn_noflash":  MinkUNetSparseAttentionNoFlash,
    "attn_noflash125":  MinkUNetSparseAttentionNoFlash125,
    "attn_noflashenc": MinkUNetSparseAttentionNoFlashEnc,

    "base" : MinkUNetSparse,
}
