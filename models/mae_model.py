# models/mae_model.py

import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

from .minkunet_attention import MinkUNetSparseAttentionCore


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def voxels_to_device(vox: Voxels, device: torch.device) -> Voxels:
    """
    Move a Voxels object to the target device.

    offsets always stays on CPU (WarpConvNet CSR requirement);
    coordinate_tensor and feature_tensor are moved to device.
    """
    coords  = vox.coordinate_tensor.to(device)
    feats   = vox.feature_tensor.to(device)
    offsets = vox.offsets   # keep on CPU
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


# ---------------------------------------------------------------------------
# Sparse global average pooling
# ---------------------------------------------------------------------------

def sparse_global_avg_pool(vox: Voxels) -> Tensor:
    """
    Compute the mean feature vector for each batch item.

    Returns
    -------
    Tensor of shape [B, C] on the same device as vox.feature_tensor.
    """
    feats   = vox.feature_tensor   # (N_total, C)
    offsets = vox.offsets          # (B+1,), CPU
    B       = len(offsets) - 1
    C       = feats.shape[1]
    device  = feats.device

    counts    = (offsets[1:] - offsets[:-1]).to(device=device, dtype=torch.float32)   # (B,)
    batch_idx = torch.repeat_interleave(
        torch.arange(B, device=device),
        counts.long(),
    )  # (N_total,)

    pooled = torch.zeros(B, C, device=device, dtype=feats.dtype)
    pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(feats), feats)
    pooled = pooled / counts.unsqueeze(1).clamp(min=1.0)
    return pooled   # [B, C]


# ---------------------------------------------------------------------------
# MAE model
# ---------------------------------------------------------------------------

class SparseMAEModel(nn.Module):
    """
    Sparse Masked Auto-Encoder for DUNE wire-plane data.

    Components
    ----------
    backbone       : MinkUNetSparseAttentionCore  (Voxels[1 ch] → Voxels[64 ch])
    charge_head    : 1×1 SparseConv2d(64→1)       SSL reconstruction head
    nu_flavor_head : sparse global avg-pool → Linear(64, n_classes)  SFT head

    Usage
    -----
    SSL training:
        pred   = model.forward_ssl(masked_voxels)
        loss   = F.l1_loss(pred.feature_tensor[mask_bool],
                           original.feature_tensor[mask_bool])

    SFT training (backbone frozen):
        model.freeze_backbone()
        logits = model.forward_sft(voxels)
        loss   = F.cross_entropy(logits[valid], labels[valid])
        model.unfreeze_backbone()
    """

    def __init__(
        self,
        n_classes: int = 3,
        spatial_encoding: bool = True,
        flash_attention:  bool = True,
        encoding_dim:     int  = 32,
        encoding_range:   float = 125.0,
    ):
        super().__init__()

        self.backbone = MinkUNetSparseAttentionCore(
            spatial_encoding=spatial_encoding,
            flash_attention=flash_attention,
            encoding_dim=encoding_dim,
            encoding_range=encoding_range,
        )

        # SSL head: 64 → 1 feature channel (charge reconstruction)
        self.charge_head = SparseConv2d(64, 1, kernel_size=1, bias=True)

        # SFT head: global pool → class logits
        self.nu_flavor_head = nn.Linear(64, n_classes)

    # ------------------------------------------------------------------ #

    def forward_ssl(self, masked_voxels: Voxels) -> Voxels:
        """
        Forward pass for SSL (self-supervised) training.

        Input  : masked Voxels with 1 feature channel
        Output : Voxels with 1 feature channel (predicted charge amplitude)
        """
        feats = self.backbone(masked_voxels)    # Voxels [64 ch]
        return self.charge_head(feats)          # Voxels [1 ch]

    def forward_sft(self, voxels: Voxels) -> Tensor:
        """
        Forward pass for SFT (supervised fine-tuning).

        Backbone runs under torch.no_grad() for memory efficiency.
        Call freeze_backbone() before this to also prevent grad accumulation.

        Input  : Voxels with 1 feature channel
        Output : [B, n_classes] logits
        """
        with torch.no_grad():
            feats = self.backbone(voxels)       # Voxels [64 ch]
        pooled = sparse_global_avg_pool(feats)  # [B, 64]  (detached)
        return self.nu_flavor_head(pooled)      # [B, n_classes]

    # ------------------------------------------------------------------ #

    def freeze_backbone(self):
        """Prevent backbone parameters from accumulating gradients."""
        self.backbone.requires_grad_(False)

    def unfreeze_backbone(self):
        """Re-enable gradient flow through backbone."""
        self.backbone.requires_grad_(True)
