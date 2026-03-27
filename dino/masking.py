"""Sparse-aware pixel masking for DINO student augmentation."""

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
    2. Return a reduced student Voxels and the kept indices per batch item

    The kept_indices are needed by the loss to align student and teacher features:
    teacher_out.feature_tensor[t_start:t_end][kept_indices[b]] gives the teacher
    features at exactly the positions the student processed.
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
            student_voxels: Voxels with ~(1 - mask_ratio) of the original active voxels
            kept_indices:   List of B tensors; kept_indices[b] indexes into the
                            per-batch-item slice of voxels (i.e. offsets[b]:offsets[b+1])
        """
        B = len(voxels.offsets) - 1
        device = voxels.coordinate_tensor.device

        kept_indices = []
        coords_list  = []
        feats_list   = []

        for b in range(B):

            # find pixels in this image
            start = int(voxels.offsets[b])
            end   = int(voxels.offsets[b + 1])
            N     = end - start

            if N == 0:
                kept_indices.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            n_keep = max(1, N - int(N * self.mask_ratio))
            perm   = torch.randperm(N, device=device)
            keep   = perm[:n_keep].sort().values   # sorted to preserve spatial order

            kept_indices.append(keep)
            coords_list.append(voxels.coordinate_tensor[start:end][keep])
            feats_list.append(voxels.feature_tensor[start:end][keep])

        counts      = torch.tensor([k.shape[0] for k in kept_indices], dtype=torch.int64)
        new_offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])

        new_coords = torch.cat(coords_list, dim=0) if coords_list else voxels.coordinate_tensor.new_zeros(0, voxels.coordinate_tensor.shape[1])
        new_feats  = torch.cat(feats_list,  dim=0) if feats_list  else voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1])

        student_voxels = Voxels(
            batched_coordinates=IntCoords(new_coords, offsets=new_offsets),
            batched_features=CatFeatures(new_feats, offsets=new_offsets),
            offsets=new_offsets,
        )

        return student_voxels, kept_indices
