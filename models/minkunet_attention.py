# ----------------------------- U-Res + Attention -----------------------------
# Architecture overview:
#   Encoder:   Sparse residual blocks + sparse pooling (efficient on empty space)
#   Bottleneck: Self-attention at small resolution (global context)
#   Decoder:   Sparse upsampling (transposed convolutions) + skip connections + residual blocks
#   Head:      Dense global pooling and classification

from torch import Tensor
import torch
import torch.nn.functional as F    # Functional layer calls (stateless)
import torch.nn as nn              # Neural network base classes

# --- WarpConvNet specific imports for sparse convolutional ops ---
from warpconvnet.geometry.types.voxels import Voxels                    # Sparse voxel data structure
from warpconvnet.nn.functional.transforms import cat                    # Concatenate sparse voxel features
from warpconvnet.nn.modules.sparse_conv import SparseConv2d             # 2D sparse convolution

from .blocks import (
    ConvBlock2D, ConvTrBlock2D, 
    ResidualSparseBlock2D, 
    BottleneckSparseAttention2D
    )

# ---------------------------------------------------------------------------
# Full network: Sparse encoder + dense attention bottleneck + sparse decoder
# ---------------------------------------------------------------------------

class MinkUNetSparseAttention(nn.Module):
    """
    Backbone: U-ResNet-style sparse model with attention bottleneck.

    Architecture:
    - Initial conv at full resolution
    - Encoder: strided convolutions + residual blocks (2 stages)
    - Bottleneck: sparse self-attention at 125×125 resolution (global context)
    - Decoder: transposed convolutions + skip connections + residual blocks (2 stages)
    - Final: feature refinement (1×1 sparse conv)

    Returns: [B, 64, 500, 500] dense feature map (no classification head)
    """
    def __init__(self, *,
                 spatial_encoding: bool = True,
                 flash_attention: bool = True,
                 encoding_dim: int = 32,
                 encoding_range: float = 125.0,
                 **kwargs,):
        super().__init__()

        # ---- Initial convolution (full resolution feature extraction) ----
        self.conv0 = ConvBlock2D(1, 32, kernel_size=3, stride=1)  # [B,1,500,500] → [B,32,500,500]

        # ---- Encoder (2 stages) ----
        # Stage 1: 500×500 to 250×250
        self.conv1 = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # Spatial downsample
        self.block1 = ResidualSparseBlock2D(32, 32, kernel_size=3) # Channel stays 32

        # Stage 2: 250×250 to 125×125
        self.conv2 = ConvBlock2D(32, 32, kernel_size=2, stride=2)  # Spatial downsample
        self.block2 = ResidualSparseBlock2D(32, 64, kernel_size=3) # Channel projection 32 to 64

        # ---- Bottleneck (attention at 125×125) ----
        # Global context at 125×125 resolution (15625 spatial tokens)
        self.bottleneck = BottleneckSparseAttention2D(channels=64, attn_channels=128, heads=4,
                                                      encoding=spatial_encoding, flash=flash_attention,
                                                      encoding_range=encoding_range, encoding_channels=encoding_dim)

        # ---- Decoder (2 stages, symmetric to encoder) ----
        # Stage 1: 125×125 to 250×250
        self.convtr5 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)   # Upsample
        self.block6 = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3) # Merge skip1, process

        # Stage 2: 250×250 to 500×500 (full resolution)
        self.convtr7 = ConvTrBlock2D(64, 64, kernel_size=2, stride=2)   # Upsample
        self.block8 = ResidualSparseBlock2D(64 + 32, 64, kernel_size=3) # Merge skip0, process

        # ---- Final projection (feature refinement, no classification head) ----
        self.final = SparseConv2d(64, 64, kernel_size=1, bias=True)  # Feature refinement

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the backbone.
        Input: [B, 1, 500, 500] dense tensor
        Output: [B, 64, 500, 500] dense feature map
        """
        # Convert dense input image to sparse voxel representation
        xs = Voxels.from_dense(x)

        # ============ ENCODER ============

        # Initial convolution at full resolution
        out = self.conv0(xs)                    # [B,1,500,500] to [B,32,500,500]
        out_p1 = out                            # Skip connection for final decoder stage

        # Stage 1: 500×500 to 250×250
        out = self.conv1(out_p1)                # Downsample spatially
        out = self.block1(out)                  # Residual processing
        out_b1p2 = out                          # Skip connection [B,32,250,250]

        # Stage 2: 250×250 to 125×125
        out = self.conv2(out_b1p2)              # Downsample spatially
        out = self.block2(out)                  # Residual + channel projection 32 to 64
                                                # Result: [B,64,125,125]

        # ============ BOTTLENECK (Sparse Attention at 125×125) ============
        out = self.bottleneck(out)              # [B,64,125,125] -> [B,128,125,125] (attention) -> [B,64,125,125]
        # ============ DECODER ============

        # Stage 1: 125×125 to 250×250
        out = self.convtr5(out, out_b1p2)       # Upsample, guided by skip geometry
        out = cat(out, out_b1p2)                # [B,64,250,250] + [B,32,250,250] = [B,96,250,250]
        out = self.block6(out)                  # Process to [B,64,250,250]

        # Stage 2: 250×250 to 500×500 (full resolution)
        out = self.convtr7(out, out_p1)         # Upsample
        out = cat(out, out_p1)                  # [B,64,500,500] + [B,32,500,500] = [B,96,500,500]
        out = self.block8(out)                  # Process to [B,64,500,500]

        # ============ FINAL PROJECTION ============
        out = self.final(out)                   # Feature refinement [B,64,500,500]

        # Convert to dense and return features (no head)
        spatial_shape = (x.shape[2], x.shape[3])
        out_dense = out.to_dense(channel_dim=1, spatial_shape=spatial_shape)  # [B,64,H,W]

        # https://github.com/NVlabs/WarpConvNet/issues/23
        # to_dense() infers batch_size via bincount on batch indices (offsets)
        # if the last sample is empty image, it gets dropped by from_dense() 
        # because it contributes no voxels; to_dense() then returns B' < B
        # Pad back with zeros.
        B = x.shape[0]
        if out_dense.shape[0] < B:
            pad = out_dense.new_zeros(B - out_dense.shape[0], *out_dense.shape[1:])
            out_dense = torch.cat([out_dense, pad], dim=0)

        return out_dense


class MinkUNetSparseAttentionClassifier(nn.Module):
    """
    Supervised classification wrapper: MinkUNetSparseAttention backbone + classification head.

    Wraps the backbone and adds a classification head for supervised training.
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
        """Forward pass: backbone → features → global pool → classify."""
        features = self.backbone(x)            # [B, 64, 500, 500]
        logits = self.head(features)           # [B, n_classes]
        return F.log_softmax(logits, dim=1)    # [B, n_classes] log-probabilities
    

class MinkUNetSparseAttentionNoEnc(MinkUNetSparseAttention):
    """Backbone variant: no spatial encoding"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=True, **kwargs)

class MinkUNetSparseAttentionNoFlash(MinkUNetSparseAttention):
    """Backbone variant: no flash attention"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=True, flash_attention=False, **kwargs)

class MinkUNetSparseAttentionNoFlashEnc(MinkUNetSparseAttention):
    """Backbone variant: no flash attention, no spatial encoding"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=False, **kwargs)


# ============ Classifier wrappers for supervised training ============

class MinkUNetSparseAttentionNoEncClassifier(MinkUNetSparseAttentionClassifier):
    """Classifier: MinkUNetSparseAttentionNoEnc backbone + head"""
    def __init__(self, n_classes: int = 4):
        super().__init__(n_classes=n_classes, spatial_encoding=False, flash_attention=True)

class MinkUNetSparseAttentionNoFlashClassifier(MinkUNetSparseAttentionClassifier):
    """Classifier: MinkUNetSparseAttentionNoFlash backbone + head"""
    def __init__(self, n_classes: int = 4):
        super().__init__(n_classes=n_classes, spatial_encoding=True, flash_attention=False)

class MinkUNetSparseAttentionNoFlashEncClassifier(MinkUNetSparseAttentionClassifier):
    """Classifier: MinkUNetSparseAttentionNoFlashEnc backbone + head"""
    def __init__(self, n_classes: int = 4):
        super().__init__(n_classes=n_classes, spatial_encoding=False, flash_attention=False)
