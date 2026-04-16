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


def voxels_label_collate_fn(batch):
    """
    Collate a list of (Voxels, int) tuples into (batched_Voxels, LongTensor).

    Used with APASparseMetaDataset for supervised fine-tuning DataLoaders.

    Usage:
        from loader.collate import voxels_label_collate_fn
        loader = DataLoader(dataset, batch_size=8, collate_fn=voxels_label_collate_fn)
    """
    voxels_list, labels = zip(*batch)
    return voxels_collate_fn(list(voxels_list)), torch.tensor(labels, dtype=torch.long)


def voxels_meta_collate_fn(batch):
    """
    Collate a list of (Voxels, meta_dict) tuples into (batched_Voxels, meta_dict).

    Used with APASparseMetaDataset(return_full_metadata=True). Numeric dict fields
    are stacked into batched tensors; `event_key` stays a list of strings.

    Expected dict keys (as produced by APASparseMetaDataset._read_full_metadata):
        label, nu_pdg, nu_ccnc, nu_intType  →  LongTensor[B]
        nu_energy                           →  FloatTensor[B]
        vertex_xyz                          →  FloatTensor[B, 3]
        event_key                           →  list[str] of length B

    Usage:
        from loader.collate import voxels_meta_collate_fn
        loader = DataLoader(dataset, batch_size=8, collate_fn=voxels_meta_collate_fn)
    """
    voxels_list, metas = zip(*batch)
    batched_voxels = voxels_collate_fn(list(voxels_list))

    out = {
        "label":      torch.tensor([m["label"]      for m in metas], dtype=torch.long),
        "nu_pdg":     torch.tensor([m["nu_pdg"]     for m in metas], dtype=torch.long),
        "nu_ccnc":    torch.tensor([m["nu_ccnc"]    for m in metas], dtype=torch.long),
        "nu_intType": torch.tensor([m["nu_intType"] for m in metas], dtype=torch.long),
        "nu_energy":  torch.tensor([m["nu_energy"]  for m in metas], dtype=torch.float32),
        "vertex_xyz": torch.stack([m["vertex_xyz"]  for m in metas], dim=0),
        "event_key":  [m["event_key"] for m in metas],
    }
    return batched_voxels, out
