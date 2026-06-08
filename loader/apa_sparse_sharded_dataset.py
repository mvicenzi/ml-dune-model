# loader/apa_sparse_sharded_dataset.py

import json
import h5py
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import IterableDataset

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures


class APASparseShardedDataset(IterableDataset):
    """
    IterableDataset over pre-sharded HDF5 files produced by loader/create_shards.py.

    Each shard holds ~shard_size samples in a flat layout (/coords, /features,
    /offsets).  At training time a shuffle buffer spanning multiple shards
    approximates per-epoch reshuffling while keeping IO to one file read per
    buffer-refill step.

    Each item yielded is a batched Voxels of exactly batch_size samples.
    Use with DataLoader(batch_size=None, ...) to disable PyTorch auto-batching.

    With num_workers > 0 the shard list is split round-robin across workers so
    every sample is seen exactly once per epoch and workers never duplicate work.
    Each worker maintains its own independent buffer.

    Samples that don't fill a complete final batch are dropped (implicit drop_last).
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 100,
        buffer_size: int = 3000,
    ):
        self.root_dir   = Path(root_dir)
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.shards: List[Path] = sorted(self.root_dir.glob("shard_*.h5"))
        if not self.shards:
            raise RuntimeError(f"No shard_*.h5 files found in {root_dir}")

        meta_path = self.root_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_samples = int(meta["n_samples"])
        else:
            # fallback: assume full shards (last may be partial, this is approximate)
            n_samples = len(self.shards) * 1000
            print(f"Warning: metadata.json not found in {root_dir}, __len__ is approximate")

        self._n_batches = n_samples // batch_size

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Approximate number of batches per epoch (used by DataLoader and schedulers)."""
        return self._n_batches

    def __iter__(self) -> Iterator[Voxels]:
        # Split shards round-robin across DataLoader workers.
        worker_info = torch.utils.data.get_worker_info()
        shards = self.shards
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        # Shuffle shard order for this epoch.
        order  = torch.randperm(len(shards)).tolist()
        shards = [shards[i] for i in order]

        # Buffer: list of (coords, feats) tuples, one entry per sample.
        buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        shard_idx = 0

        def fill_buffer() -> None:
            nonlocal shard_idx
            while len(buffer) < self.buffer_size and shard_idx < len(shards):
                buffer.extend(self._load_shard(shards[shard_idx]))
                shard_idx += 1

        fill_buffer()

        while len(buffer) >= self.batch_size:
            # Pick batch_size samples at random from the buffer, keep the rest.
            perm          = torch.randperm(len(buffer)).tolist()
            batch_indices = perm[: self.batch_size]
            keep_indices  = perm[self.batch_size :]
            batch  = [buffer[i] for i in batch_indices]
            buffer[:] = [buffer[i] for i in keep_indices]
            fill_buffer()
            yield self._make_batch(batch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_shard(self, path: Path) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Read one shard file and return a list of (coords, feats) per sample."""
        with h5py.File(path, "r") as f:
            coords  = torch.from_numpy(f["coords"][()]).to(torch.int32)    # (N_total, 2)
            feats   = torch.from_numpy(f["features"][()]).to(torch.float32) # (N_total, 1)
            offsets = torch.from_numpy(f["offsets"][()]).to(torch.int64)   # (B+1,)

        samples = []
        for i in range(len(offsets) - 1):
            s, e = int(offsets[i]), int(offsets[i + 1])
            samples.append((coords[s:e].clone(), feats[s:e].clone()))
        return samples

    def _make_batch(self, samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Voxels:
        """Assemble a list of (coords, feats) tuples into a single batched Voxels."""
        all_coords, all_feats = zip(*samples)
        counts  = torch.tensor([c.shape[0] for c in all_coords], dtype=torch.int64)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, dim=0)])
        coords_cat = torch.cat(all_coords, dim=0)
        feats_cat  = torch.cat(all_feats,  dim=0)
        return Voxels(
            batched_coordinates=IntCoords(coords_cat, offsets=offsets),
            batched_features=CatFeatures(feats_cat, offsets=offsets),
            offsets=offsets,
        )
