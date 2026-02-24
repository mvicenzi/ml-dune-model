# loader/collate.py

import torch
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures


def voxels_collate_fn(batch):
    """
    Collate a list of single-item Voxels objects into one batched Voxels.

    Each input Voxels must have batch_size == 1.  The resulting Voxels has
    batch_size == len(batch) with offsets [0, N_0, N_0+N_1, ..., sum(N_i)].

    Usage with DataLoader:
        from loader.collate import voxels_collate_fn
        loader = DataLoader(dataset, batch_size=8, collate_fn=voxels_collate_fn)
    """
    all_coords = [v.coordinate_tensor for v in batch]
    all_feats  = [v.feature_tensor for v in batch]

    counts  = torch.tensor([c.shape[0] for c in all_coords], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, dim=0)])

    coords_cat = torch.cat(all_coords, dim=0)
    feats_cat  = torch.cat(all_feats,  dim=0)

    return Voxels(
        batched_coordinates=IntCoords(coords_cat, offsets=offsets),
        batched_features=CatFeatures(feats_cat, offsets=offsets),
        offsets=offsets,
    )
