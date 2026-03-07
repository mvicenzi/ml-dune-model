# models/sparse_masking.py

import math
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures


def sparse_block_mask(
    voxels: Voxels,
    masking_frac: float,
    win_ch: int,
    win_tick: int,
) -> tuple[Voxels, torch.Tensor]:
    """
    Block-mask a fraction of active voxels and their spatial neighbours.

    For each batch item independently:
      1. Randomly sample ceil(masking_frac × N_i) voxels as seeds.
      2. Mark all voxels within [±win_ch, ±win_tick] of any seed as masked.
      3. Zero the feature values at masked positions (coordinates unchanged).

    Parameters
    ----------
    voxels      : batched Voxels; coordinates are (channel, tick) int32
    masking_frac: fraction of active voxels used as seeds, in [0, 1]
    win_ch      : half-window radius in the channel direction
    win_tick    : half-window radius in the tick direction

    Returns
    -------
    masked_voxels : Voxels
        Same coordinate structure as input; features zeroed at masked positions.
    mask_bool : BoolTensor, shape (N_total,)
        True at every position that was masked.
    """
    coords  = voxels.coordinate_tensor   # (N_total, 2)  int32, on device
    feats   = voxels.feature_tensor      # (N_total, C)  float, on device
    offsets = voxels.offsets             # (B+1,)        int64, on CPU

    device    = feats.device
    N_total   = feats.shape[0]
    mask_bool = torch.zeros(N_total, dtype=torch.bool, device=device)

    B = len(offsets) - 1
    for i in range(B):
        start = int(offsets[i].item())
        end   = int(offsets[i + 1].item())
        N_i   = end - start
        if N_i == 0:
            continue

        n_seeds = math.ceil(masking_frac * N_i)
        if n_seeds == 0:
            continue

        coords_i  = coords[start:end]                        # (N_i, 2)
        seed_idx  = torch.randperm(N_i, device=device)[:n_seeds]
        seeds     = coords_i[seed_idx]                       # (n_seeds, 2)

        # Vectorised window check:
        #   diff[n, s, d] = coords_i[n, d] - seeds[s, d]
        diff   = coords_i.unsqueeze(1) - seeds.unsqueeze(0)  # (N_i, n_seeds, 2)
        in_win = (diff[:, :, 0].abs() <= win_ch) & \
                 (diff[:, :, 1].abs() <= win_tick)            # (N_i, n_seeds)
        mask_bool[start:end] = in_win.any(dim=1)             # (N_i,)

    # Clone features and zero out masked positions; keep coordinates intact.
    new_feats = feats.clone()
    new_feats[mask_bool] = 0.0

    masked_voxels = Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(new_feats, offsets=offsets),
        offsets=offsets,
    )
    return masked_voxels, mask_bool
