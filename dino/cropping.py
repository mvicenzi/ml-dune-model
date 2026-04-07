"""Activity-aware spatial cropping for DINO student augmentation."""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
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

    # Image spatial dimensions (must match dataset; 500×500 for DUNE)
    image_h: int = 500
    image_w: int = 500


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
      - Returns one sub-Voxels object per crop, plus kept_indices aligned to
        the teacher's full-image voxel slice — the same contract as
        SparseVoxelMasker.

    Important: selected voxels keep their original (x, y) coordinates (no
    translation to a crop-local origin). This is required so that kept_indices
    directly index the teacher's voxel slice without any offset correction, and
    it is safe because sparse convolutions compute positions as coord // stride
    regardless of the coordinate origin.
    """

    def __init__(self, config: CropConfig):
        self.cfg = config
        self._n_crops = config.n_global + config.n_local

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

    def __call__(self, voxels: Voxels) -> Tuple[List[Voxels], List[List[Tensor]], List[List[Tensor]]]:
        """
        Args:
            voxels: Batched Voxels (full image, B items) — teacher's view.

        Returns:
            crops:        List of n_crops Voxels objects (each batched over B images).
            kept_indices: List of n_crops lists; kept_indices[k][b] indexes into the
                          per-batch-item slice [offsets[b]:offsets[b+1]] of voxels.
            teacher_luts: List of n_global lists; teacher_luts[g][b] is a tensor of
                          size N_b mapping full-image voxel index → local position
                          within teacher crop g (or -1 if absent). Pre-built here so
                          that intersection matching in the training loop is a single
                          O(N) gather instead of O(N log N) isin + searchsorted.
        """
        cfg = self.cfg
        B = len(voxels.offsets) - 1
        device = voxels.coordinate_tensor.device

        # Per-crop accumulators: coords, feats, indices, and offset counts
        crop_coords  = [[] for _ in range(self._n_crops)]
        crop_feats   = [[] for _ in range(self._n_crops)]
        crop_kidx    = [[] for _ in range(self._n_crops)]
        crop_counts  = [[] for _ in range(self._n_crops)]

        for b in range(B):
            start = int(voxels.offsets[b])
            end   = int(voxels.offsets[b + 1])

            coords_b = voxels.coordinate_tensor[start:end]  # [N, 2]
            feats_b  = voxels.feature_tensor[start:end]     # [N, C]
            N = end - start

            # Build activity array and heatmap once per image
            A = _build_activity_array(coords_b, cfg.image_h, cfg.image_w)
            H = _gaussian_blur_np(A, cfg.blur_sigma_px)

            # Sample one box per crop
            for k in range(self._n_crops):
                if k < cfg.n_global:
                    scale_range = cfg.global_scale
                else:
                    scale_range = cfg.local_scale

                if N == 0:
                    # Empty image: keep zero voxels for this crop
                    kidx = torch.zeros(0, dtype=torch.long, device=device)
                else:
                    box = self._sample_box_for_image(A, H, scale_range)
                    left, top, right, bottom = box

                    # Select voxels inside the box
                    xy = coords_b  # [N, 2], col 0 = x, col 1 = y
                    mask = (
                        (xy[:, 0] >= left)  &
                        (xy[:, 0] <  right) &
                        (xy[:, 1] >= top)   &
                        (xy[:, 1] <  bottom)
                    )
                    kidx = mask.nonzero(as_tuple=False).squeeze(1)

                    if kidx.numel() == 0:
                        # Fallback: include at least the nearest voxel so the
                        # batch item is never empty (avoids BatchNorm edge case)
                        kidx = torch.zeros(1, dtype=torch.long, device=device)

                crop_kidx[k].append(kidx)

                if kidx.numel() > 0:
                    crop_coords[k].append(coords_b[kidx])
                    crop_feats[k].append(feats_b[kidx])
                    crop_counts[k].append(kidx.shape[0])
                else:
                    crop_counts[k].append(0)

        # Assemble one Voxels object per crop
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

        # Build lookup tables for global (teacher) crops: for each teacher crop g
        # and batch item b, map full-image voxel index → local position in the crop.
        teacher_luts = []
        for g in range(cfg.n_global):
            luts_g = []
            for b in range(B):
                N_b = int(voxels.offsets[b + 1] - voxels.offsets[b])
                lut = torch.full((N_b,), -1, dtype=torch.long, device=device)
                T_g_b = crop_kidx[g][b]
                if T_g_b.numel() > 0:
                    lut[T_g_b] = torch.arange(T_g_b.numel(), device=device)
                luts_g.append(lut)
            teacher_luts.append(luts_g)

        return crops, crop_kidx, teacher_luts
