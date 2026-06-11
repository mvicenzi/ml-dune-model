"""Sparse-aware masking strategies for DINO student augmentation."""

import math
from typing import List, Tuple

import torch
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures


class SparseVoxelMasker:
    """
    Masks active voxels for the DINO student by removing entries from a Voxels object.

    For each image in a batch:
    1. Randomly select a fraction of active voxels to keep (1 - mask_ratio)
    2. Returns: 
        - reduced student Voxels
        - (x, y) coordinates of the masked voxels per batch item

    The masked_coords can be used to inject learnable mask tokens at the dropped positions,
    so the student can predict teacher features there.
    
    NOTE: Student/teacher alignment in the loss continues to use
    coordinate intersection via match_and_gather -- no index bookkeeping needed.
    """

    def __init__(self, mask_ratio: float = 0.5, seed: int = None):
        """
        Args:
            mask_ratio: Fraction of active voxels to mask (0.0 to 1.0)
            seed: Optional random seed for reproducibility
        """
        self.mask_ratio = mask_ratio
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, voxels: Voxels) -> Tuple[Voxels, List[torch.Tensor]]:
        """
        Apply masking to a batched Voxels object.

        Args:
            voxels: Batched Voxels with batch_size B

        Returns:
            student_voxels:          Voxels with ~(1 - mask_ratio) of the original voxels
            masked_coords_per_batch: List of B tensors, each [N_masked_b, 2] holding
                                     the (x, y) integer coords of voxels dropped from
                                     that image. Same dtype/device as input coords.
        """
        
        #number of images in the batch
        B = len(voxels.offsets) - 1 

        device = voxels.coordinate_tensor.device

        # number of coordinate dimensions (2D or 3D)
        coord_dim = voxels.coordinate_tensor.shape[1]

        masked_coords_per_batch = []
        coords_list = []
        feats_list  = []

        # for each image in the batch
        for b in range(B):

            # find start/end indices of the voxels for image b
            # and count them 
            start = int(voxels.offsets[b])
            end   = int(voxels.offsets[b + 1])
            N     = end - start

            if N == 0:
                # empty image: append empty entries to ALL lists so that
                # student_voxels.offsets stays length B+1 and aligns with
                # masked_coords_per_batch (which is always length B).
                masked_coords_per_batch.append(
                    voxels.coordinate_tensor.new_zeros(0, coord_dim)
                )
                coords_list.append(voxels.coordinate_tensor.new_zeros(0, coord_dim))
                feats_list.append(voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1]))
                continue

            # max(1, ...) guarantees at least one voxel is always kept
            n_keep   = max(1, N - int(N * self.mask_ratio))

            # generate a random permutation of the voxel indices, then 
            # keep the first n_keep and drop the the rest.
            # sort the indices to preserve spatial order.
            perm     = torch.randperm(N, device=device)
            keep     = perm[:n_keep].sort().values    
            masked   = perm[n_keep:].sort().values

            masked_coords_per_batch.append(voxels.coordinate_tensor[start:end][masked])
            coords_list.append(voxels.coordinate_tensor[start:end][keep])
            feats_list.append(voxels.feature_tensor[start:end][keep])

        # number of surviving voxels per batch item
        n_kept      = [c.shape[0] for c in coords_list] # 
        counts      = torch.tensor(n_kept, dtype=torch.int64, device=device)
        
        # offsets for the new batched Voxels object
        new_offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=device),
            counts.cumsum(0),
        ])

        new_coords = (torch.cat(coords_list, dim=0) if coords_list
                      else voxels.coordinate_tensor.new_zeros(0, coord_dim))
        new_feats  = (torch.cat(feats_list,  dim=0) if feats_list
                      else voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1]))

        # package new Voxels object
        student_voxels = Voxels(
            batched_coordinates=IntCoords(new_coords, offsets=new_offsets),
            batched_features=CatFeatures(new_feats, offsets=new_offsets),
            offsets=new_offsets,
        )

        return student_voxels, masked_coords_per_batch


class SparseBlockMasker:
    """
    Block-masks active voxels for the DINO student by removing entire spatial regions.

    For each image in the batch:
    1. Estimate K block centers needed to cover mask_ratio of voxels using the
       geometric coverage formula: E[fraction] = 1 - (1 - p)^K, where
       p = min(block_area, N) / N is the expected per-block coverage rate.
    2. Sample K random centers from active voxels (no replacement).
    3. Mask all voxels within [±win_ch, ±win_tick] of any center in one vectorised op.
    4. Remove masked voxels from the student input; return their coordinates.

    Drop-in replacement for SparseVoxelMasker — same (Voxels) → (Voxels, List[coords])
    interface, so match_and_gather, encode_student, and the loss need no changes.

    Because blocks may overlap, the actual masked fraction varies around mask_ratio.
    """

    def __init__(self, mask_ratio: float = 0.5, win_ch: int = 5, win_tick: int = 5):
        """
        Args:
            mask_ratio: Target fraction of active voxels to mask (0.0 to 1.0).
            win_ch:     Half-window radius in the channel direction (voxels).
            win_tick:   Half-window radius in the tick direction (voxels).
        """
        self.mask_ratio  = mask_ratio
        self.win_ch      = win_ch
        self.win_tick    = win_tick
        self._block_area = (2 * win_ch + 1) * (2 * win_tick + 1)
        # Adaptive estimate of the effective per-block coverage rate p.
        # Initialised to None (falls back to the analytic formula on the first call)
        # then updated via EMA from observed coverage — no extra GPU syncs needed.
        self._p_eff: float = None

    def __call__(self, voxels: Voxels) -> Tuple[Voxels, List[torch.Tensor]]:
        """
        Apply block masking to a batched Voxels object.

        Args:
            voxels: Batched Voxels with batch_size B.

        Returns:
            student_voxels:          Voxels with block-masked voxels removed.
            masked_coords_per_batch: List of B tensors, each [N_masked_b, 2] holding
                                     the (channel, tick) coords of removed voxels.
        """
        # Reset per-call so the EMA from one crop type (e.g. sparse global crop)
        # does not contaminate K estimates for a different crop type (dense local
        # crop) in the next call.  The EMA still converges within a call across
        # batch elements, which is when it is actually useful.
        self._p_eff = None

        B         = len(voxels.offsets) - 1
        device    = voxels.coordinate_tensor.device
        coord_dim = voxels.coordinate_tensor.shape[1]

        masked_coords_per_batch = []
        coords_list = []
        feats_list  = []

        for b in range(B):
            start = int(voxels.offsets[b])
            end   = int(voxels.offsets[b + 1])
            N     = end - start

            if N == 0:
                # empty image: append empty entries to ALL lists so that
                # student_voxels.offsets stays length B+1 and aligns with
                # masked_coords_per_batch (which is always length B).
                masked_coords_per_batch.append(
                    voxels.coordinate_tensor.new_zeros(0, coord_dim)
                )
                coords_list.append(voxels.coordinate_tensor.new_zeros(0, coord_dim))
                feats_list.append(voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1]))
                continue

            coords_i = voxels.coordinate_tensor[start:end]  # (N, 2)
            feats_i  = voxels.feature_tensor[start:end]     # (N, C)

            # Estimate how many block centers K are needed to cover mask_ratio of voxels.
            # Geometric coverage model: E[covered] = 1 - (1-p)^K.
            # p is the effective per-block coverage rate: ideally block_area/N, but for
            # sparse data the active voxels per window are far fewer than block_area, so
            # we learn p from observed coverage via EMA (all CPU math, no extra GPU sync).
            p_formula = min(self._block_area, N) / N
            p = self._p_eff if self._p_eff is not None else p_formula
            p = max(1e-6, min(p, 1.0 - 1e-6))
            K = min(
                math.ceil(math.log(1.0 - self.mask_ratio) / math.log(1.0 - p)),
                N - 1,
            )

            center_idx = torch.randperm(N, device=device)[:K]
            centers    = coords_i[center_idx]                           # (K, 2)
            diff       = coords_i.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, 2)
            mask_bool  = (
                (diff[..., 0].abs() <= self.win_ch) &
                (diff[..., 1].abs() <= self.win_tick)
            ).any(dim=1)                                                # (N,) — no sync

            # Guard: guarantee at least one voxel survives (rare with K ≤ N-1).
            if mask_bool.all():
                mask_bool[torch.randint(0, N, (1,), device=device)] = False

            keep   = (~mask_bool).nonzero(as_tuple=False).squeeze(1)
            masked = mask_bool.nonzero(as_tuple=False).squeeze(1)

            # Update EMA of effective p from observed coverage (keep.shape[0] is a
            # Python int after nonzero — no extra GPU sync).
            actual = (N - keep.shape[0]) / N
            if K > 0 and 0.0 < actual < 1.0:
                p_measured = 1.0 - (1.0 - actual) ** (1.0 / K)
                self._p_eff = (p_measured if self._p_eff is None
                               else 0.9 * self._p_eff + 0.1 * p_measured)

            masked_coords_per_batch.append(coords_i[masked])
            coords_list.append(coords_i[keep])
            feats_list.append(feats_i[keep])

        n_kept      = [c.shape[0] for c in coords_list]
        counts      = torch.tensor(n_kept, dtype=torch.int64, device=device)
        new_offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=device),
            counts.cumsum(0),
        ])

        new_coords = (torch.cat(coords_list, dim=0) if coords_list
                      else voxels.coordinate_tensor.new_zeros(0, coord_dim))
        new_feats  = (torch.cat(feats_list,  dim=0) if feats_list
                      else voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1]))

        student_voxels = Voxels(
            batched_coordinates=IntCoords(new_coords, offsets=new_offsets),
            batched_features=CatFeatures(new_feats, offsets=new_offsets),
            offsets=new_offsets,
        )

        return student_voxels, masked_coords_per_batch
