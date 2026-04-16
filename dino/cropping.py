"""Activity-aware spatial cropping for DINO student augmentation."""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
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
# Internal helpers (ported from wc_dino2/dinov2/data/augmentations.py)
# ---------------------------------------------------------------------------

def _build_activity_array(
    coords: Tensor,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """
    Build a float32 binary activity array of shape [image_h, image_w].

    coords: [N, 2] integer tensor with col-0 = x (channel axis) and col-1 = y (tick axis).
    Active positions are set to 1.0; all others remain 0.0.
    """
    A = np.zeros((image_h, image_w), dtype=np.float32)
    if coords.shape[0] == 0:
        return A
    xy = coords.cpu().numpy()
    xs = xy[:, 0].astype(int)
    ys = xy[:, 1].astype(int)
    # Clamp to valid bounds (safety guard)
    xs = np.clip(xs, 0, image_w - 1)
    ys = np.clip(ys, 0, image_h - 1)
    A[ys, xs] = 1.0
    return A


def _gaussian_blur_np(A: np.ndarray, sigma_px: float) -> np.ndarray:
    """Apply a Gaussian blur via scipy to a 2-D float array."""
    if sigma_px <= 0:
        return A.copy()
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(A, sigma=sigma_px)


def _sample_anchor(H: np.ndarray, power: float) -> Tuple[int, int]:
    """Sample (x, y) from heatmap proportional to H**power."""
    Hp = np.clip(H, 0.0, None)
    if power != 1.0:
        Hp = Hp ** power
    total = float(Hp.sum())
    h, w = Hp.shape
    if total <= 1e-12:
        return random.randrange(w), random.randrange(h)
    p = (Hp / total).ravel()
    idx = int(np.random.choice(p.size, p=p))
    y, x = divmod(idx, w)
    return int(x), int(y)


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


def _box_has_enough_hits(
    A: np.ndarray,
    box: Tuple[int, int, int, int],
    min_active: int,
) -> bool:
    left, top, right, bottom = box
    patch = A[top:bottom, left:right]
    return patch.size > 0 and int((patch > 0).sum()) >= min_active


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

    def _sample_box_for_image(
        self,
        A: np.ndarray,
        H: np.ndarray,
        scale_range: Tuple[float, float],
    ) -> Tuple[int, int, int, int]:
        cfg = self.cfg
        for _ in range(cfg.max_attempts):
            anchor = _sample_anchor(H, cfg.heatmap_power)
            crop_w, crop_h = _sample_crop_wh(cfg.image_w, cfg.image_h, scale_range, cfg.aspect_ratio)
            box = _propose_box(anchor, crop_w, crop_h, cfg.image_w, cfg.image_h)
            if _box_has_enough_hits(A, box, cfg.min_active_pixels):
                return box
        # Fallback to centred crop
        return _fallback_centre_box(cfg.image_w, cfg.image_h, scale_range, cfg.aspect_ratio)

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

        # --- Build activity arrays and heatmaps on GPU in one batch ----------
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

        # Single transfer to CPU for the sampling loop
        A_np = A_batch.cpu().numpy()                        # one GPU→CPU sync
        H_np = H_batch.cpu().numpy()

        # --- Per-image, per-crop: sample boxes and filter voxels -------------
        crop_coords  = [[] for _ in range(self._n_crops)]
        crop_feats   = [[] for _ in range(self._n_crops)]
        crop_counts  = [[] for _ in range(self._n_crops)]

        for b in range(B):
            start = int(voxels.offsets[b])
            end   = int(voxels.offsets[b + 1])

            coords_b = voxels.coordinate_tensor[start:end]  # [N, 2]
            feats_b  = voxels.feature_tensor[start:end]      # [N, C]
            N = end - start

            A = A_np[b]
            H = H_np[b]

            for k in range(self._n_crops):
                if k < cfg.n_global:
                    scale_range = cfg.global_scale
                else:
                    scale_range = cfg.local_scale

                if N == 0:
                    kidx = torch.zeros(0, dtype=torch.long, device=device)
                else:
                    box = self._sample_box_for_image(A, H, scale_range)
                    left, top, right, bottom = box

                    xy = coords_b
                    mask = (
                        (xy[:, 0] >= left)  &
                        (xy[:, 0] <  right) &
                        (xy[:, 1] >= top)   &
                        (xy[:, 1] <  bottom)
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
