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

# ============ Sparse backbone classes (Voxels → Voxels) ============
from .minkunet import MinkUNetSparse
from .minkunet_attention import (
    MinkUNetSparseAttention,
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
# Points to Dense variants (Tensor → Tensor) to keep the current dense dataloader working.
# Switch to sparse variants once the sparse pipeline is in place.
BACKBONE_REGISTRY = {
    # Backbone with sparse attention
    "attn_default":     MinkUNetSparseAttentionDense,

    # Variants of sparse attention module
    "attn_noenc":       MinkUNetSparseAttentionNoEncDense,
    "attn_noflash":     MinkUNetSparseAttentionNoFlashDense,
    "attn_noflashenc":  MinkUNetSparseAttentionNoFlashEncDense,

    # Backbone without attention
    "base":             MinkUNetSparse,
}
