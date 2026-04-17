"""Activity-aware spatial cropping for DINO student augmentation."""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures


@dataclass
class CropConfig:
    """Parameters for activity-aware multi-crop augmentation on sparse DUNE images."""

    # Number of global and local crops per image
    n_global: int = 2
    n_local: int = 4

    # Crop scale as fraction of total image area
    global_scale: Tuple[float, float] = (0.4, 1.0)
    local_scale: Tuple[float, float] = (0.05, 0.2)

    # Width-to-height aspect ratio range (log-uniform sampling)
    aspect_ratio: Tuple[float, float] = (3 / 4, 4 / 3)

    # Gaussian blur sigma (px) applied to the binary activity map to build heatmap
    blur_sigma_px: float = 10.0

    # Exponent applied to heatmap before sampling; >1 emphasises hotspots
    heatmap_power: float = 1.0

    # Minimum active pixels inside a crop box for it to be accepted
    min_active_pixels: int = 10

    # Maximum attempts to find a valid crop before falling back to centred crop
    max_attempts: int = 50

    # Image spatial dimensions (must match dataset; 1500 ticks × 1050 wires for DUNE)
    image_h: int = 1500
    image_w: int = 1050


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_crop_wh(
    img_w: int,
    img_h: int,
    scale_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
) -> Tuple[int, int]:
    """Sample crop (w, h) from area-scale and log-uniform aspect ratio."""
    area = img_w * img_h
    scale = random.uniform(*scale_range)
    target_area = scale * area
    log_ar_min = math.log(aspect_range[0])
    log_ar_max = math.log(aspect_range[1])
    aspect = math.exp(random.uniform(log_ar_min, log_ar_max))
    crop_w = int(round(math.sqrt(target_area * aspect)))
    crop_h = int(round(math.sqrt(target_area / aspect)))
    crop_w = max(1, min(crop_w, img_w))
    crop_h = max(1, min(crop_h, img_h))
    return crop_w, crop_h


def _propose_box(
    anchor_xy: Tuple[int, int],
    crop_w: int,
    crop_h: int,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """Return (left, top, right, bottom) centred at anchor, clamped to image."""
    ax, ay = anchor_xy
    left = int(round(ax - crop_w / 2))
    top = int(round(ay - crop_h / 2))
    left = max(0, min(left, img_w - crop_w))
    top = max(0, min(top, img_h - crop_h))
    return left, top, left + crop_w, top + crop_h


def _fallback_centre_box(
    img_w: int,
    img_h: int,
    scale_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
) -> Tuple[int, int, int, int]:
    crop_w, crop_h = _sample_crop_wh(img_w, img_h, scale_range, aspect_range)
    left = (img_w - crop_w) // 2
    top = (img_h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h


# ---------------------------------------------------------------------------
# SparseCropper
# ---------------------------------------------------------------------------

class SparseCropper:
    """
    Activity-aware multi-crop augmentation for batched sparse Voxels.

    For each image in a batch:
      - Builds a binary activity array from the voxel coordinates.
      - Applies a Gaussian blur to create a sampling heatmap.
      - Samples n_global + n_local crop boxes biased toward active regions.
      - Returns one sub-Voxels object per crop.

    Anchor-point sampling and hit-checking are done on GPU to avoid expensive
    CPU transfers of the dense activity/heatmap arrays.

    Important: selected voxels keep their original (x, y) coordinates (no
    translation to a crop-local origin). This allows spatial intersection
    between crops to be found by coordinate matching on the backbone outputs,
    and it is safe because sparse convolutions compute positions as coord // stride
    regardless of the coordinate origin.
    """

    def __init__(self, config: CropConfig):
        self.cfg = config
        self._n_crops = config.n_global + config.n_local
        self._init_blur_kernel(config.blur_sigma_px)

    def _init_blur_kernel(self, sigma: float):
        """Pre-compute a separable Gaussian kernel for GPU blurring."""
        if sigma <= 0:
            self._blur_kernel_h = None
            self._blur_kernel_v = None
            self._blur_pad = 0
            return
        radius = int(3 * sigma + 0.5)
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k1d = torch.exp(-0.5 * (x / sigma) ** 2)
        k1d = k1d / k1d.sum()
        self._blur_kernel_h = k1d.reshape(1, 1, 1, -1)
        self._blur_kernel_v = k1d.reshape(1, 1, -1, 1)
        self._blur_pad = radius

    def _gpu_blur(self, A_batch: Tensor) -> Tensor:
        """Gaussian blur a [B, H, W] activity tensor on GPU (separable conv)."""
        if self._blur_kernel_h is None:
            return A_batch.clone()
        x = A_batch.unsqueeze(1)  # [B, 1, H, W]
        kh = self._blur_kernel_h.to(x.device, x.dtype)
        kv = self._blur_kernel_v.to(x.device, x.dtype)
        x = F.pad(x, (self._blur_pad, self._blur_pad, 0, 0), mode="reflect")
        x = F.conv2d(x, kh)
        x = F.pad(x, (0, 0, self._blur_pad, self._blur_pad), mode="reflect")
        x = F.conv2d(x, kv)
        return x.squeeze(1)  # [B, H, W]

    def __call__(self, voxels: Voxels) -> List[Voxels]:
        """
        Args:
            voxels: Batched Voxels (full image, B items) — teacher's view.

        Returns:
            crops: List of n_crops Voxels objects (each batched over B images).
                   Voxels retain their original (x, y) coordinates, so spatial
                   intersection between crops can be found by coordinate matching
                   on the backbone outputs.
        """
        cfg = self.cfg
        B = len(voxels.offsets) - 1
        device = voxels.coordinate_tensor.device

        # --- Build activity array on GPU and blur to get heatmap -------------
        A_batch = torch.zeros(B, cfg.image_h, cfg.image_w, device=device)
        for b in range(B):
            start = int(voxels.offsets[b])
            end   = int(voxels.offsets[b + 1])
            if end > start:
                c = voxels.coordinate_tensor[start:end]
                cx = c[:, 0].long().clamp(0, cfg.image_w - 1)
                cy = c[:, 1].long().clamp(0, cfg.image_h - 1)
                A_batch[b, cy, cx] = 1.0

        H_batch = self._gpu_blur(A_batch)                  # [B, H, W] on GPU
        del A_batch                                         # no longer needed

        # --- Pre-sample all anchor points on GPU via multinomial -------------
        # This replaces per-attempt np.random.choice (which recomputes a 1.575M
        # cumsum each call).  One multinomial call per image is ~150× faster.
        Hp = H_batch.clamp(min=0)
        if cfg.heatmap_power != 1.0:
            Hp = Hp ** cfg.heatmap_power
        Hp_flat = Hp.view(B, -1)                            # [B, H*W]

        # For images with zero activity, fall back to uniform sampling
        row_sums = Hp_flat.sum(dim=1, keepdim=True)
        uniform = torch.ones(1, Hp_flat.shape[1], device=device) / Hp_flat.shape[1]
        Hp_flat = torch.where(row_sums > 1e-12, Hp_flat, uniform)

        n_samples = self._n_crops * cfg.max_attempts
        # anchor_flat[b, i] is a flat index into the H×W grid
        anchor_flat = torch.multinomial(Hp_flat, n_samples, replacement=True)  # [B, n_samples]
        anchor_x = (anchor_flat % cfg.image_w).cpu()        # small transfer
        anchor_y = (anchor_flat // cfg.image_w).cpu()
        del H_batch, Hp, Hp_flat, anchor_flat

        # --- Per-image, per-crop: propose boxes and filter voxels ------------
        crop_coords  = [[] for _ in range(self._n_crops)]
        crop_feats   = [[] for _ in range(self._n_crops)]
        crop_counts  = [[] for _ in range(self._n_crops)]

        for b in range(B):
            start = int(voxels.offsets[b])
            end   = int(voxels.offsets[b + 1])

            coords_b = voxels.coordinate_tensor[start:end]  # [N, 2] on GPU
            feats_b  = voxels.feature_tensor[start:end]      # [N, C]
            N = end - start

            sample_cursor = 0  # walks through pre-sampled anchors for this image

            for k in range(self._n_crops):
                scale_range = cfg.global_scale if k < cfg.n_global else cfg.local_scale

                if N == 0:
                    kidx = torch.zeros(0, dtype=torch.long, device=device)
                else:
                    found = False
                    for _ in range(cfg.max_attempts):
                        ax = int(anchor_x[b, sample_cursor])
                        ay = int(anchor_y[b, sample_cursor])
                        sample_cursor += 1

                        crop_w, crop_h = _sample_crop_wh(
                            cfg.image_w, cfg.image_h, scale_range, cfg.aspect_ratio,
                        )
                        box = _propose_box((ax, ay), crop_w, crop_h, cfg.image_w, cfg.image_h)
                        left, top, right, bottom = box

                        # Hit-check directly on sparse coords (avoids dense CPU array)
                        mask = (
                            (coords_b[:, 0] >= left)  &
                            (coords_b[:, 0] <  right) &
                            (coords_b[:, 1] >= top)   &
                            (coords_b[:, 1] <  bottom)
                        )
                        n_hits = mask.sum().item()
                        if n_hits >= cfg.min_active_pixels:
                            kidx = mask.nonzero(as_tuple=False).squeeze(1)
                            found = True
                            break

                    if not found:
                        box = _fallback_centre_box(
                            cfg.image_w, cfg.image_h, scale_range, cfg.aspect_ratio,
                        )
                        left, top, right, bottom = box
                        mask = (
                            (coords_b[:, 0] >= left)  &
                            (coords_b[:, 0] <  right) &
                            (coords_b[:, 1] >= top)   &
                            (coords_b[:, 1] <  bottom)
                        )
                        kidx = mask.nonzero(as_tuple=False).squeeze(1)
                        if kidx.numel() == 0:
                            kidx = torch.zeros(1, dtype=torch.long, device=device)

                if kidx.numel() > 0:
                    crop_coords[k].append(coords_b[kidx])
                    crop_feats[k].append(feats_b[kidx])
                    crop_counts[k].append(kidx.shape[0])
                else:
                    crop_counts[k].append(0)

        # --- Assemble one Voxels object per crop ----------------------------
        crops = []
        for k in range(self._n_crops):
            counts_k = torch.tensor(crop_counts[k], dtype=torch.int64)
            offsets_k = torch.cat([torch.zeros(1, dtype=torch.int64), counts_k.cumsum(0)])

            if crop_coords[k]:
                new_coords = torch.cat(crop_coords[k], dim=0)
                new_feats  = torch.cat(crop_feats[k],  dim=0)
            else:
                new_coords = voxels.coordinate_tensor.new_zeros(0, voxels.coordinate_tensor.shape[1])
                new_feats  = voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1])

            crop_voxels = Voxels(
                batched_coordinates=IntCoords(new_coords, offsets=offsets_k),
                batched_features=CatFeatures(new_feats, offsets=offsets_k),
                offsets=offsets_k,
            )
            crops.append(crop_voxels)

        return crops
