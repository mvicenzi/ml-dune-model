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
                # empty image: nothing to mask; append empty tensors to keep
                # masked_coords_per_batch length-B and offsets consistent.
                masked_coords_per_batch.append(
                    voxels.coordinate_tensor.new_zeros(0, coord_dim)
                )
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
