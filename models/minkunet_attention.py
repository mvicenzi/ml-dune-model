# ----------------------------- U-Res + Attention -----------------------------
# Architecture overview:
#   Encoder:   Sparse residual blocks + sparse pooling (efficient on empty space)
#   Bottleneck: Self-attention at small resolution (global context)
#   Decoder:   Sparse upsampling (transposed convolutions) + skip connections + residual blocks
#   Head:      Dense global pooling and classification

import torch
from torch import Tensor
import torch.nn.functional as F    # Functional layer calls (stateless)
import torch.nn as nn              # Neural network base classes

# --- WarpConvNet specific imports for sparse convolutional ops ---
from warpconvnet.geometry.types.voxels import Voxels                    # Sparse voxel data structure
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.nn.functional.transforms import cat                    # Concatenate sparse voxel features
from warpconvnet.nn.modules.sparse_conv import SparseConv2d             # 2D sparse convolution

from .blocks import (
    ConvBlock2D, ConvTrBlock2D,
    ResidualSparseBlock2D,
    BottleneckSparseAttention2D,
    DenseInput, DenseOutput,
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

    Input/output: Voxels → Voxels (fully sparse, no dense materialisation)
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

    def forward(self, xs: Voxels) -> Voxels:

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

        return out

# ---------------------------------------------------------------------------
# MAE backbone subclass
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionMAE(MinkUNetSparseAttention):
    """
    MAE-augmented backbone: encoder identical to base; decoder optionally emits
    features at masked coordinates via skip augmentation at every skip level.

    Both DINO student and teacher are instances of this class:
      - teacher calls forward(xs)                    -> base behaviour (no injection)
      - student calls forward(xs, masked_coords=...) -> injects mask tokens at skips

    In the default behaviour, the skips already guide the upsampling, telling the 
    decoder where to place features.  The MAE augmentation "activates" also 
    the masked positions by injecting a learnable mask_token at those coordinates.
    --> this makes the decoder predict features at those positions

    Same state_dict shape on student and teacher -> EMA update works unchanged.
    Teacher's copy of mask_token is EMA-updated but never read; harmless.
    """
    def __init__(self, **kw):
        """
        Args:
            **kw: Passed through to MinkUNetSparseAttention.__init__.
        """
        super().__init__(**kw)

        # Channel counts come from the base class layer definitions:
        #   out_p1   = conv0  = ConvBlock2D(1, 32, ...)             
        #   out_b1p2 = block1 = ResidualSparseBlock2D(32, 32, ...)  
        self.mask_token_full = nn.Parameter(torch.zeros(32))  # injected at out_p1
        self.mask_token_half = nn.Parameter(torch.zeros(32))  # injected at out_b1p2
        nn.init.trunc_normal_(self.mask_token_full, std=0.02)
        nn.init.trunc_normal_(self.mask_token_half, std=0.02)

    def _augment_skip_with_masked(self,
        skip: Voxels,
        masked_coords_per_batch: list[Tensor],
        mask_token: Tensor,
    ) -> Voxels:
        """
        Return a new "skip" Voxels object with the masked coordinates injected:
         - Features at non-masked positions are copied from skip;
         - features at masked positions are filled with mask_token.

        Offsets are recomputed to reflect the enlarged per-item counts.
        """
        device = skip.coordinate_tensor.device

        B = len(masked_coords_per_batch)  # images in this back batch
        C = skip.feature_tensor.shape[1]  # channels in the skip features 
        coord_dim = skip.coordinate_tensor.shape[1] # coordinate dim (2D)

        new_coords_list: list[Tensor] = []
        new_feats_list:  list[Tensor] = []

        # for each image in the batch
        for b in range(B):
            s = int(skip.offsets[b])
            e = int(skip.offsets[b + 1])
            kc = skip.coordinate_tensor[s:e]   # [N_active, 2]
            kf = skip.feature_tensor[s:e]      # [N_active, C]
            mc = masked_coords_per_batch[b]    # [N_masked, 2]

            # if no masked positions, carry on
            if mc.shape[0] == 0:
                new_coords_list.append(kc)
                new_feats_list.append(kf)
            else: 
                # set the same mask_token vector for all masked positions
                # this is just a fancier way than defaulting all of them to 0
                mf = mask_token.unsqueeze(0).expand(mc.shape[0], -1)
                # add the masked positions/features to the skip's original ones
                new_coords_list.append(torch.cat([kc, mc], dim=0))
                new_feats_list.append(torch.cat([kf, mf], dim=0))

        # recompute offsets for the new skip
        counts = torch.tensor(
            [c.shape[0] for c in new_coords_list], dtype=torch.int64, device=device
        )
        new_offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=device),
            counts.cumsum(0),
        ])
        new_coords = (torch.cat(new_coords_list, dim=0) if new_coords_list
                      else skip.coordinate_tensor.new_zeros(0, coord_dim))
        new_feats  = (torch.cat(new_feats_list, dim=0) if new_feats_list
                      else skip.feature_tensor.new_zeros(0, C))

        # preserve the skip's tensor_stride so WarpConvNet's stride checks pass
        # (e.g. the half-res skip has tensor_stride=(2,2); losing it breaks convtr).
        ts = skip.batched_coordinates.tensor_stride
        return Voxels(
            batched_coordinates=IntCoords(new_coords, offsets=new_offsets, tensor_stride=ts),
            batched_features=CatFeatures(new_feats, offsets=new_offsets),
            offsets=new_offsets,
        )

    @staticmethod
    def _project_masked_to_skip(
        masked_coords_per_batch: list[Tensor],
        skip: Voxels,
        stride: int,
    ) -> list[Tensor]:
        """
        Project full-res masked coords to the resolution of a lower-res skip.
        This means applying the same downsampling process and return only 
        the coords NOT already present in that skip (to avoid duplicates).
        """
        B = len(masked_coords_per_batch) # number of images in the batch
        coord_dim = skip.coordinate_tensor.shape[1] # coordinate dim (2D)

        # we want to create a unique key for each (x,y) pair
        # W must exceed the max x-coord so that y*W+x is a unique.
        W = 1
        if skip.coordinate_tensor.shape[0] > 0:
            W = int(skip.coordinate_tensor[:, 0].max().item()) + 2 

        result: list[Tensor] = []
        # for each image in the bacth
        for b in range(B):

            mc = masked_coords_per_batch[b]
            # if no masked positions, carry on
            if mc.shape[0] == 0:
                result.append(mc.new_zeros(0, coord_dim))
                continue

            # downsample the masked coordinates to the skip's resolution
            # matches striding convention from WarpConvNet's convolutions
            mc_low = torch.div(mc, stride, rounding_mode="floor").to(mc.dtype)
            mc_low = torch.unique(mc_low, dim=0)

            # find the skip coordinates for this image
            # basically: we want to remove any masked position that already exists
            ss = int(skip.offsets[b])
            se = int(skip.offsets[b + 1])
            skip_coords = skip.coordinate_tensor[ss:se]

            if skip_coords.shape[0] > 0:
                # encode 2D coords as flat integers
                skip_keys = skip_coords[:, 1].long() * W + skip_coords[:, 0].long()
                mc_keys   = mc_low[:, 1].long()      * W + mc_low[:, 0].long()
                # binary-search each projected masked coord into the sorted skip keys
                sk_sorted, _ = skip_keys.sort()
                pos = torch.searchsorted(sk_sorted, mc_keys).clamp(max=sk_sorted.shape[0] - 1)
                # keep only coords NOT already in the skip
                mc_low = mc_low[sk_sorted[pos] != mc_keys]

            result.append(mc_low)
        return result

    def forward(
        self,
        xs: Voxels,
        masked_coords: list[Tensor] | None = None,
    ) -> Voxels:
        """
        Args:
            xs:            Input Voxels (kept voxels only for the student;
                           full input for the teacher).
            masked_coords: List of B tensors [N_masked_b, 2] at full resolution.
                           Pass None (or omit) to run the unmodified base decoder.
        """
        # ===== ENCODER (identical to base) =====
        out = self.conv0(xs)
        out_p1 = out                               # full-res skip  (32 ch)

        out = self.conv1(out_p1)
        out = self.block1(out)
        out_b1p2 = out                             # half-res skip  (32 ch)

        out = self.conv2(out_b1p2)
        out = self.block2(out)
        out = self.bottleneck(out)

        # ===== DECODER =====
        # student is provided with masked_coords; teacher is not (None)
        if masked_coords is not None:
            # -- Stage 1: 125 -> 250, inject at half-res skip --
            # need to track the masked coordinates at the half-res skip level,
            # which means replicating the downsampling process...
            # then add the masked coordinates with maks_token features to the skip
            masked_half = self._project_masked_to_skip(masked_coords, out_b1p2, stride=2)
            skip_b1p2_aug = self._augment_skip_with_masked(out_b1p2, masked_half, self.mask_token_half)

            out = self.convtr5(out, skip_b1p2_aug) # Upsample, guided by augmented skip geometry
            out = cat(out, skip_b1p2_aug)
            out = self.block6(out)

            # -- Stage 2: 250->500, inject at full-res skip --
            # add the masked coordinates with mask_token features to the skip
            skip_p1_aug = self._augment_skip_with_masked(out_p1, masked_coords, self.mask_token_full)

            out = self.convtr7(out, skip_p1_aug) # Upsample, guided by augmented skip geometry
            out = cat(out, skip_p1_aug)
            out = self.block8(out)
        else:
            out = self.convtr5(out, out_b1p2)
            out = cat(out, out_b1p2)
            out = self.block6(out)

            out = self.convtr7(out, out_p1)
            out = cat(out, out_p1)
            out = self.block8(out)

        return self.final(out)


# ---------------------------------------------------------------------------
# Dense-interface wrapper and classifier heads
# ---------------------------------------------------------------------------

class MinkUNetSparseAttentionDense(MinkUNetSparseAttention):
    """
    Dense-interface wrapper around MinkUNetSparseAttention.

    Attaches DenseInput (Tensor → Voxels) and DenseOutput (Voxels → Tensor)
    around the sparse backbone, preserving a Tensor → Tensor interface for use
    with dense dataloaders or the supervised classifier head.

    Input/output: [B, 1, H, W] Tensor → [B, 64, H, W] Tensor
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.from_dense = DenseInput()
        self.to_dense   = DenseOutput()

    def forward(self, x: Tensor) -> Tensor:
        xs  = self.from_dense(x)
        out = super().forward(xs)
        return self.to_dense(out, reference=x)


class MinkUNetSparseAttentionClassifier(nn.Module):
    """
    Supervised classification wrapper: MinkUNetSparseAttentionDense backbone + classification head.

    Wraps the backbone and adds a classification head for supervised training.
    Returns: [B, n_classes] log-probabilities
    """
    def __init__(self, n_classes: int = 4, **backbone_kwargs):
        super().__init__()
        self.backbone = MinkUNetSparseAttentionDense(**backbone_kwargs)
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


# ============ Sparse (Voxels -> Voxels) variants ============

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

# ============ Dense (Tensor -> Tensor) variants ============

class MinkUNetSparseAttentionNoEncDense(MinkUNetSparseAttentionDense):
    """Dense-interface variant: no spatial encoding"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=False, flash_attention=True, **kwargs)

class MinkUNetSparseAttentionNoFlashDense(MinkUNetSparseAttentionDense):
    """Dense-interface variant: no flash attention"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, spatial_encoding=True, flash_attention=False, **kwargs)

class MinkUNetSparseAttentionNoFlashEncDense(MinkUNetSparseAttentionDense):
    """Dense-interface variant: no flash attention, no spatial encoding"""
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
