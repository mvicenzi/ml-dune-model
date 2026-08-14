# models/__init__.py
"""
Model registry for DUNE neutrino detector classifiers and backbones.

Backbone naming convention:
- Sparse variants (Voxels → Voxels): MinkUNetSparseAttention*
- Dense variants  (Tensor → Tensor): MinkUNetSparseAttention*Dense
  Dense = sparse backbone wrapped with DenseInput / DenseOutput boundary layers.

- Backbones: pure feature extractors, return [B, 64, H, W] dense features (Dense) or Voxels (Sparse)
- Classifiers: backbone + classification head, return [B, 4] class logits
"""

import inspect

# ============ Sparse backbone classes (Voxels → Voxels) ============
from .minkunet import MinkUNetSparse
from .minkunet_attention import (
    MinkUNetSparseAttention,
    MinkUNetSparseAttentionMAE,
    MinkUNetSparseAttentionNoEnc,
    MinkUNetSparseAttentionNoFlash,
    MinkUNetSparseAttentionNoFlashEnc,
)

# ============ Dense backbone classes (Tensor → Tensor) ============
from .minkunet_attention import (
    MinkUNetSparseAttentionDense,
    MinkUNetSparseAttentionNoEncDense,
    MinkUNetSparseAttentionNoFlashDense,
    MinkUNetSparseAttentionNoFlashEncDense,
)

# ============ Classifier wrapper classes (backbone + head for supervised training) ============
from .minkunet import MinkUNetSparseClassifier
from .minkunet_attention import (
    MinkUNetSparseAttentionClassifier,
    MinkUNetSparseAttentionNoEncClassifier,
    MinkUNetSparseAttentionNoFlashClassifier,
    MinkUNetSparseAttentionNoFlashEncClassifier,
)

# ============ MODEL_REGISTRY (classifiers for backward compatibility with training.py) ============
MODEL_REGISTRY = {
    # Backbone with sparse attention + classification head
    "attn_default":     MinkUNetSparseAttentionClassifier,

    # Variants of sparse attention module + classification head
    "attn_noenc":       MinkUNetSparseAttentionNoEncClassifier,
    "attn_noflash":     MinkUNetSparseAttentionNoFlashClassifier,
    "attn_noflashenc":  MinkUNetSparseAttentionNoFlashEncClassifier,

    # Backbone without attention + classification head
    "base":             MinkUNetSparseClassifier,
}

# ============ BACKBONE_REGISTRY (exposed for DINO and other self-supervised methods) ============
# Points to sparse variants (Voxels → Voxels).
BACKBONE_REGISTRY = {
    # Backbone with sparse attention
    "attn_default":     MinkUNetSparseAttention,

    # MAE-augmented backbone (inject mask tokens at every skip in the decoder)
    "attn_mae":         MinkUNetSparseAttentionMAE,

    # Variants of sparse attention module
    "attn_noenc":       MinkUNetSparseAttentionNoEnc,
    "attn_noflash":     MinkUNetSparseAttentionNoFlash,
    "attn_noflashenc":  MinkUNetSparseAttentionNoFlashEnc,

    # Backbone without attention
    "base":             MinkUNetSparse,
}


# ============ Backbone construction ============

def backbone_kwargs(backbone_cls, **requested):
    """Check that `backbone_cls` accepts every requested constructor flag.

    inspect.signature() on the leaf class does not see through `def __init__(self, **kw)`,
    which every backbone variant uses to forward to its base — the architecture flags are
    named only on MinkUNetSparseAttention.__init__, so a leaf-only check reports that the
    backbone accepts nothing and the flags fall back to their hardcoded defaults. Merging
    each class's own parameters across the MRO recovers the true accepted set.

    A flag the backbone cannot honour raises rather than being dropped: a silently ignored
    flag trains a different network than the config describes.
    """
    accepted = set()
    for cls in backbone_cls.__mro__:
        if "__init__" in cls.__dict__:
            accepted.update(inspect.signature(cls.__init__).parameters)

    unsupported = sorted(set(requested) - accepted)
    if unsupported:
        raise ValueError(
            f"backbone {backbone_cls.__name__} does not accept "
            f"{', '.join(unsupported)} — drop the key from the config, or choose a "
            f"backbone that supports it"
        )
    return dict(requested)
