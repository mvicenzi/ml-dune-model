# models/mae_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Feature replacement helper
# ---------------------------------------------------------------------------

def _replace_features(vox: Voxels, new_feats: Tensor) -> Voxels:
    """Return a new Voxels with the same coords/offsets but replaced features."""
    offsets = vox.offsets  # keep on CPU
    return Voxels(
        batched_coordinates=IntCoords(vox.coordinate_tensor, offsets=offsets),
        batched_features=CatFeatures(new_feats, offsets=offsets),
        offsets=offsets,
    )


# ---------------------------------------------------------------------------
# Charge normalization helpers
# ---------------------------------------------------------------------------

def log1p_voxels(vox: Voxels) -> Voxels:
    """Apply log1p to all feature values.  Maps raw ADC → log(ADC+1).

    Compresses the ~350× ADC dynamic range to ~10×, making reconstruction
    targets uniform across the charge spectrum.  Pair with expm1_voxels to
    recover raw ADC for visualization or downstream use.
    """
    return _replace_features(vox, torch.log1p(vox.feature_tensor))


def expm1_voxels(vox: Voxels) -> Voxels:
    """Apply expm1 to all feature values.  Inverts log1p_voxels: log(ADC+1) → ADC."""
    return _replace_features(vox, torch.expm1(vox.feature_tensor))


# ---------------------------------------------------------------------------
# Sparse CNN classification head
# ---------------------------------------------------------------------------

class SparseCNNHead(nn.Module):
    """
    Sparse CNN classification head.

    Three stride-2 3×3 sparse conv layers with expanding channels (64→128→256),
    followed by a 1×1 conv at 256 ch, sparse global average pooling, and a
    linear classifier.  Mirrors the standard CNN pattern of doubling channels
    while halving spatial resolution at each stride-2 stage.

    Each stride-2 conv halves the coordinate grid, so after three stages the
    pooling operates over ~N/64 voxels — more spatially discriminative.

    The architecture is identical for both use cases; only in_ch differs:
      in_ch=64  — applied to backbone features
      in_ch=1   — applied to raw charge directly

    Parameters
    ----------
    in_ch     : input feature channels (64 for backbone output, 1 for raw charge)
    n_classes : number of output classes
    """

    def __init__(self, in_ch: int = 64, n_classes: int = 3):
        super().__init__()
        self.conv1 = SparseConv2d(in_ch,  64,  kernel_size=3, stride=2, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = SparseConv2d(64,  128, kernel_size=3, stride=2, bias=False)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = SparseConv2d(128, 256, kernel_size=3, stride=2, bias=False)
        self.bn3   = nn.BatchNorm1d(256)
        self.conv4 = SparseConv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.bn4   = nn.BatchNorm1d(256)
        self.fc    = nn.Linear(256, n_classes)

    def forward(self, vox: Voxels) -> Tensor:
        vox    = self.conv1(vox)
        vox    = _replace_features(vox, F.relu(self.bn1(vox.feature_tensor)))
        vox    = self.conv2(vox)
        vox    = _replace_features(vox, F.relu(self.bn2(vox.feature_tensor)))
        vox    = self.conv3(vox)
        vox    = _replace_features(vox, F.relu(self.bn3(vox.feature_tensor)))
        vox    = self.conv4(vox)
        vox    = _replace_features(vox, F.relu(self.bn4(vox.feature_tensor)))
        pooled = sparse_global_avg_pool(vox)   # [B, 256]
        return self.fc(pooled)                  # [B, n_classes]


# ---------------------------------------------------------------------------
# MAE model
# ---------------------------------------------------------------------------

class SparseMAEModel(nn.Module):
    """
    Sparse Masked Auto-Encoder for DUNE wire-plane data.

    Components
    ----------
    backbone           : MinkUNetSparseAttentionCore  (Voxels[1 ch] → Voxels[64 ch])
    charge_head        : 1×1 SparseConv2d(64→1)       SSL reconstruction head
    nu_flavor_head     : SparseCNNHead(in_ch=64)       SFT head on backbone features
    ref_nu_flavor_head : SparseCNNHead(in_ch=1)        SFT reference on raw charge

    Usage
    -----
    SSL training:
        pred   = model.forward_ssl(masked_voxels)
        loss   = F.l1_loss(pred.feature_tensor[mask_bool],
                           original.feature_tensor[mask_bool])

    SFT training (backbone frozen):
        model.freeze_backbone()
        logits     = model.forward_sft(voxels)      # uses backbone features
        logits_ref = model.forward_sft_ref(voxels)  # uses raw 1-ch charge
        loss   = F.cross_entropy(logits[valid], labels[valid])
        model.unfreeze_backbone()
    """

    def __init__(
        self,
        n_classes: int = 3,
        spatial_encoding: bool = True,
        flash_attention:  bool = True,
        encoding_dim:     int  = 32,
        encoding_range:   float = 300.0,
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

        # SFT head on backbone features (64 ch)
        self.nu_flavor_head = SparseCNNHead(in_ch=64, n_classes=n_classes)

        # Reference SFT head on raw 1-ch charge (no backbone)
        self.ref_nu_flavor_head = SparseCNNHead(in_ch=1, n_classes=n_classes)

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
        Forward pass for SFT using backbone features.

        Backbone runs under torch.no_grad() for memory efficiency.
        Call freeze_backbone() before this to also prevent grad accumulation.

        Input  : Voxels with 1 feature channel
        Output : [B, n_classes] logits
        """
        with torch.no_grad():
            feats = self.backbone(voxels)   # Voxels [64 ch]
        return self.nu_flavor_head(feats)   # [B, n_classes]

    def forward_sft_ref(self, voxels: Voxels) -> Tensor:
        """
        Forward pass for reference SFT using raw 1-ch charge (no backbone).

        Input  : Voxels with 1 feature channel
        Output : [B, n_classes] logits
        """
        return self.ref_nu_flavor_head(voxels)  # [B, n_classes]

    # ------------------------------------------------------------------ #

    def freeze_backbone(self):
        """Freeze backbone weights and BatchNorm running stats for SFT training.

        requires_grad_(False) stops weight updates, but BatchNorm1d still
        updates running_mean/running_var in train() mode regardless of grads.
        Calling .eval() on the backbone prevents that stat drift, so the
        saved checkpoint has running stats calibrated to SSL data, not SFT data.
        """
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def unfreeze_backbone(self):
        """Re-enable gradient flow and restore BatchNorm to training mode."""
        self.backbone.requires_grad_(True)
        self.backbone.train()

    def reset_sft_head(self):
        """Re-initialize both SFT heads to random weights.

        Call at the start of each SFT evaluation so both heads probe
        the current backbone features and raw charge from a clean slate,
        giving an unbiased comparison at each SSL checkpoint.
        """
        for m in self.nu_flavor_head.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for m in self.ref_nu_flavor_head.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
