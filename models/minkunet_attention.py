# ----------------------------- U-Res + Attention -----------------------------
# Architecture overview:
#   Encoder:   Sparse residual blocks + sparse pooling (efficient on empty space)
#   Bottleneck: Self-attention at small resolution (global context)
#   Decoder:   Sparse upsampling (transposed convolutions) + skip connections + residual blocks
#   Head:      Dense global pooling and classification
#
# The backbone is split into three composable modules:
#
#   FromDense                       Dense Tensor  →  Voxels
#   MinkUNetSparseAttentionCore     Voxels        →  Voxels   (all learnable layers)
#   ToDense                         Voxels        →  Dense Tensor
#
# MinkUNetSparseAttention composes all three for the original Dense → Dense interface.
# Use MinkUNetSparseAttentionCore directly when input already arrives as Voxels
# (e.g. from APASparseDataset), avoiding the from_dense overhead entirely.

from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.transforms import cat
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

from .blocks import (
    ConvBlock2D, ConvTrBlock2D,
    ResidualSparseBlock2D,
    BottleneckSparseAttention2D,
    FromDense, ToDense
)


# ---------------------------------------------------------------------------
# Core backbone: Voxels → Voxels
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionCore(nn.Module):
    """
    Core U-ResNet backbone operating entirely in sparse Voxels space.

    Architecture:
    - Initial conv at full resolution                      [B, 1,  500×500]
    - Encoder stage 1: strided conv + residual block       [B, 32, 500×500 → 250×250]
    - Encoder stage 2: strided conv + residual block       [B, 32→64, 250×250 → 125×125]
    - Bottleneck: sparse self-attention at 125×125         [B, 64,  125×125]
    - Decoder stage 1: transposed conv + skip + residual   [B, 64,  250×250]
    - Decoder stage 2: transposed conv + skip + residual   [B, 64,  500×500]
    - Final 1×1 conv (feature refinement)                  [B, 64,  500×500]

    Interface: Voxels  →  Voxels
    """

    def __init__(self, *,
                 spatial_encoding: bool = True,
                 flash_attention:  bool = True,
                 encoding_dim:     int  = 32,
                 encoding_range:   float = 125.0):
        super().__init__()

        # ---- Initial convolution (full resolution feature extraction) ----
        self.conv0  = ConvBlock2D(1, 32, kernel_size=3, stride=1)   # [B,1,500,500] → [B,32,500,500]

        # ---- Encoder (2 stages) ----
        self.conv1  = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # 500×500 → 250×250
        self.block1 = ResidualSparseBlock2D(32, 32, kernel_size=3)

        self.conv2  = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # 250×250 → 125×125
        self.block2 = ResidualSparseBlock2D(32, 64, kernel_size=3)  # ch 32 → 64

        # ---- Bottleneck: sparse attention at 125×125 ----
        self.bottleneck = BottleneckSparseAttention2D(
            channels=64, attn_channels=128, heads=4,
            encoding=spatial_encoding, flash=flash_attention,
            encoding_range=encoding_range, encoding_channels=encoding_dim,
        )

        # ---- Decoder (2 stages) ----
        self.convtr5 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)    # 125×125 → 250×250
        self.block6  = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3) # merge skip1

        self.convtr7 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)    # 250×250 → 500×500
        self.block8  = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3) # merge skip0

        # ---- Final 1×1 conv (feature refinement) ----
        self.final = SparseConv2d(64, 64, kernel_size=1, bias=True)

    def forward(self, xs: Voxels) -> Voxels:
        """
        Input:  Voxels — sparse representation of [B, 1, 500, 500]
        Output: Voxels — sparse feature map        [B, 64, 500, 500]
        """
        # ---- Encoder ----
        out      = self.conv0(xs)           # [B, 1,  500×500] → [B, 32, 500×500]
        out_p1   = out                      # skip: [B, 32, 500×500]

        out      = self.conv1(out_p1)       # → [B, 32, 250×250]
        out      = self.block1(out)
        out_b1p2 = out                      # skip: [B, 32, 250×250]

        out      = self.conv2(out_b1p2)     # → [B, 32, 125×125]
        out      = self.block2(out)         # → [B, 64, 125×125]

        # ---- Bottleneck ----
        out = self.bottleneck(out)          # [B, 64, 125×125] (attention)

        # ---- Decoder ----
        out = self.convtr5(out, out_b1p2)   # → [B, 64, 250×250]
        out = cat(out, out_b1p2)            # → [B, 96, 250×250]
        out = self.block6(out)              # → [B, 64, 250×250]

        out = self.convtr7(out, out_p1)     # → [B, 64, 500×500]
        out = cat(out, out_p1)              # → [B, 96, 500×500]
        out = self.block8(out)              # → [B, 64, 500×500]

        return self.final(out)              # → [B, 64, 500×500]




# ---------------------------------------------------------------------------
# Full backbone wrapper (backward compatible): Dense Tensor → Dense Tensor
# ---------------------------------------------------------------------------

class MinkUNetSparseAttention(nn.Module):
    """
    Backbone: U-ResNet-style sparse model with attention bottleneck.
    Dense Tensor → Dense Tensor (original, backward-compatible interface).

    Composes:
        self.input  = FromDense   (Dense → Sparse)
        self.core   = MinkUNetSparseAttentionCore     (Sparse → Sparse)
        self.output = ToDense  (Sparse → Dense)

    For pipelines where data arrives as Voxels, use self.core directly and
    call self.output manually with the true batch_size.

    Returns: [B, 64, 500, 500] dense feature map (no classification head)
    """

    def __init__(self, *,
                 spatial_encoding: bool  = True,
                 flash_attention:  bool  = True,
                 encoding_dim:     int   = 32,
                 encoding_range:   float = 125.0,
                 **kwargs):
        super().__init__()
        self.input  = FromDense()
        self.core   = MinkUNetSparseAttentionCore(
            spatial_encoding=spatial_encoding,
            flash_attention=flash_attention,
            encoding_dim=encoding_dim,
            encoding_range=encoding_range,
        )
        self.output = ToDense()

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:  [B, 1, 500, 500] dense tensor
        Output: [B, 64, 500, 500] dense feature map
        """
        B  = x.shape[0]
        xs = self.input(x)
        xs = self.core(xs)
        return self.output(xs, B)


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionClassifier(nn.Module):
    """
    Supervised classification wrapper: MinkUNetSparseAttention backbone + head.
    Returns: [B, n_classes] log-probabilities
    """

    def __init__(self, n_classes: int = 4, **backbone_kwargs):
        super().__init__()
        self.backbone = MinkUNetSparseAttention(**backbone_kwargs)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)             # [B, 64, 500, 500]
        logits   = self.head(features)          # [B, n_classes]
        return F.log_softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Backbone variants (different attention settings)
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionNoEnc(MinkUNetSparseAttention):
    """Variant: spatial positional encoding disabled."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=True, **kwargs)


class MinkUNetSparseAttentionNoFlash(MinkUNetSparseAttention):
    """Variant: flash attention disabled."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=True, flash_attention=False, **kwargs)


class MinkUNetSparseAttentionNoFlashEnc(MinkUNetSparseAttention):
    """Variant: flash attention and spatial encoding both disabled."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=False, **kwargs)


# ---------------------------------------------------------------------------
# Classifier wrappers for the variants
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionNoEncClassifier(MinkUNetSparseAttentionClassifier):
    def __init__(self, n_classes: int = 4):
        super().__init__(n_classes=n_classes, spatial_encoding=False, flash_attention=True)


class MinkUNetSparseAttentionNoFlashClassifier(MinkUNetSparseAttentionClassifier):
    def __init__(self, n_classes: int = 4):
        super().__init__(n_classes=n_classes, spatial_encoding=True, flash_attention=False)


class MinkUNetSparseAttentionNoFlashEncClassifier(MinkUNetSparseAttentionClassifier):
    def __init__(self, n_classes: int = 4):
        super().__init__(n_classes=n_classes, spatial_encoding=False, flash_attention=False)
