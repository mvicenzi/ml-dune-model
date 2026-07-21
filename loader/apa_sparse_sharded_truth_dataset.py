# loader/apa_sparse_sharded_truth_dataset.py

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures


class APASparseShardedTruthDataset(Dataset):
    """
    Map-style Dataset over truth-annotated shard files produced by
    loader/create_truth_shards.py.

    Returns (Voxels, meta_dict) per sample, behaviorally identical to
    APASparseMetaDataset(return_full_metadata=True, return_pixel_truth=True).
    Compatible with voxels_meta_collate_fn and extract_features._run_loader
    without any changes to those callers.

    Shards are loaded lazily and cached in memory on first access.
    Use DataLoader with num_workers=0 to avoid per-worker cache duplication
    (each shard is then loaded at most once per process).

    Attributes:
        apa  (int): APA number baked into the shard set at creation time.
        view (str): Wire-plane view baked into the shard set at creation time.
    """

    def __init__(self, root_dir: str) -> None:
        root = Path(root_dir)

        meta_path = root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {root_dir}")
        with open(meta_path) as fp:
            meta = json.load(fp)

        self.apa:  int = int(meta["apa"])
        self.view: str = str(meta["view"])

        self.shards: List[Path] = sorted(root.glob("shard_*.h5"))
        if not self.shards:
            raise RuntimeError(f"No shard_*.h5 files found in {root_dir}")

        # Build global index: (shard_idx, local_idx_within_shard) for every sample.
        # Opening each shard briefly here only reads /offsets — no pixel data yet.
        self._index: List[Tuple[int, int]] = []
        for si, shard_path in enumerate(self.shards):
            with h5py.File(shard_path, "r") as f:
                n_images = len(f["offsets"]) - 1
            for li in range(n_images):
                self._index.append((si, li))

        self._shard_cache: Dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._index[idx]
        shard = self._get_shard(shard_idx)
        return self._make_item(shard, local_idx)

    # ------------------------------------------------------------------
    # Shard loading (lazy, cached per shard)
    # ------------------------------------------------------------------

    def _get_shard(self, shard_idx: int) -> dict:
        if shard_idx not in self._shard_cache:
            self._shard_cache[shard_idx] = self._load_shard(self.shards[shard_idx])
        return self._shard_cache[shard_idx]

    def _load_shard(self, path: Path) -> dict:
        """Read one shard file into memory. One file open per shard."""
        with h5py.File(path, "r") as f:
            return {
                "coords":     torch.from_numpy(f["coords"][()]).to(torch.int32),
                "features":   torch.from_numpy(f["features"][()]).to(torch.float32),
                "offsets":    torch.from_numpy(f["offsets"][()]).to(torch.int64),
                "labels":     f["labels"][()],        # (N,) int32
                "nu_pdg":     f["nu_pdg"][()],         # (N,) int32
                "nu_ccnc":    f["nu_ccnc"][()],        # (N,) int32
                "nu_intType": f["nu_intType"][()],     # (N,) int32
                "nu_energy":  f["nu_energy"][()],      # (N,) float32
                "vertex_xyz": f["vertex_xyz"][()],     # (N, 3) float32
                "event_key":  f["event_key"][()],      # (N,) bytes
                "pid_labels": f["pid_labels"][()],     # (N_pix,) int32
            }

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _make_item(self, shard: dict, local_idx: int):
        s = int(shard["offsets"][local_idx])
        e = int(shard["offsets"][local_idx + 1])

        coords = shard["coords"][s:e]     # (N, 2) int32 tensor
        feats  = shard["features"][s:e]   # (N, 1) float32 tensor
        n      = e - s
        offsets = torch.tensor([0, n], dtype=torch.int64)

        voxels = Voxels(
            batched_coordinates=IntCoords(coords, offsets=offsets),
            batched_features=CatFeatures(feats, offsets=offsets),
            offsets=offsets,
        )

        ev_key = shard["event_key"][local_idx]
        if isinstance(ev_key, (bytes, np.bytes_)):
            ev_key = ev_key.decode("utf-8")

        meta = {
            "label":      int(shard["labels"][local_idx]),
            "nu_pdg":     int(shard["nu_pdg"][local_idx]),
            "nu_ccnc":    int(shard["nu_ccnc"][local_idx]),
            "nu_intType": int(shard["nu_intType"][local_idx]),
            "nu_energy":  float(shard["nu_energy"][local_idx]),
            "vertex_xyz": torch.tensor(shard["vertex_xyz"][local_idx], dtype=torch.float32),
            "event_key":  ev_key,
            "pid_labels": shard["pid_labels"][s:e].astype(np.int32),
        }

        return voxels, meta
