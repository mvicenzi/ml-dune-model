# loader/apa_sparse_sharded_dataset.py

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

# Per-image event-truth datasets (int / float / special-cased below).
_EVENT_INT_KEYS   = ("labels", "nu_pdg", "nu_ccnc", "nu_intType")
_EVENT_FLOAT_KEYS = ("nu_energy",)
# Per-pixel truth datasets (CSR-aligned with /coords via /offsets).
_PIXEL_KEYS = ("pixel_labels", "pixel_energyfrac", "pixel_trackid", "pixel_truth_q")


class APASparseShardedDataset(IterableDataset):
    """
    IterableDataset over pre-sharded HDF5 files produced by loader/create_shards.py.

    Each shard holds ~shard_size samples in a flat layout (/coords, /features,
    /offsets, plus whatever truth tiers the shards were created with).  At
    training time a shuffle buffer spanning multiple shards approximates
    per-epoch reshuffling while keeping IO to one file read per buffer-refill
    step.

    Event-level truth is returned automatically when the shards carry it.
    Per-pixel tiers are opt-in (return_pixel_truth / return_extra_truth,
    same API as APASparseMetaDataset / APAPackedDataset) because HDF5
    datasets present in a shard are decompressed on every read — training
    should leave them off even on full-truth shard sets.

    Each item yielded is a tuple (voxels, meta):
        voxels: batched Voxels of exactly batch_size samples
        meta:   dict in the same format as voxels_meta_collate_fn:
                    label/nu_pdg/nu_ccnc/nu_intType → LongTensor[B]
                    nu_energy                       → FloatTensor[B]
                    vertex_xyz                      → FloatTensor[B, 3]
                    event_key                       → list[str]
                    pixel_labels / pixel_*          → list of B np.ndarray[N_i]
                {} when the shards carry no truth datasets (legacy reco shards).

    Use with DataLoader(batch_size=None, ...) to disable PyTorch auto-batching.
    Training loops that only need pixels just unpack and drop the meta.

    shuffle=True  (training):    shard order reshuffled each epoch + buffer
                                 shuffle; samples dropped into random batches.
    shuffle=False (diagnostics): deterministic shard order (sorted) and
                                 in-shard sample order — identical event
                                 sequence on every pass, e.g. for comparing
                                 features across checkpoints.

    With num_workers > 0 the shard list is split round-robin across workers so
    every sample is seen exactly once per epoch and workers never duplicate work.
    Each worker maintains its own independent buffer.

    Samples that don't fill a complete final batch are dropped (implicit drop_last).

    Attributes:
        apa  (int|None):  APA number from metadata.json (None if absent).
        view (str|None):  wire-plane view from metadata.json (None if absent).
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 100,
        buffer_size: int = 3000,
        shuffle: bool = True,
        return_pixel_truth: bool = False,
        return_extra_truth: bool = False,
    ):
        self.root_dir   = Path(root_dir)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        if return_extra_truth:
            return_pixel_truth = True
        self.return_pixel_truth = return_pixel_truth
        self.return_extra_truth = return_extra_truth

        self.shards: List[Path] = sorted(self.root_dir.glob("shard_*.h5"))
        if not self.shards:
            raise RuntimeError(f"No shard_*.h5 files found in {root_dir}")

        meta_path = self.root_dir / "metadata.json"
        self.apa: Optional[int] = None
        self.view: Optional[str] = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_samples = int(meta["n_samples"])
            self.apa  = int(meta["apa"]) if "apa" in meta else None
            self.view = str(meta["view"]) if "view" in meta else None
        else:
            # fallback: assume full shards (last may be partial, this is approximate)
            n_samples = len(self.shards) * 1000
            print(f"Warning: metadata.json not found in {root_dir}, __len__ is approximate")

        self._n_batches = n_samples // batch_size

        # Discover which truth datasets the shards actually carry, then keep
        # only the requested tiers: per-pixel truth in the shards is NOT free
        # to read (unlike the packed .npz, HDF5 datasets present in a shard
        # would be decompressed on every epoch), so it is opt-in like on
        # APASparseMetaDataset / APAPackedDataset. Event truth (a few scalars
        # per event) is always returned when present.
        with h5py.File(self.shards[0], "r") as f:
            self._has_event_truth = "labels" in f
            available = tuple(k for k in _PIXEL_KEYS if k in f)
        wanted = []
        if self.return_pixel_truth:
            wanted.append("pixel_labels")
        if self.return_extra_truth:
            wanted += ["pixel_energyfrac", "pixel_trackid", "pixel_truth_q"]
        missing = [k for k in wanted if k not in available]
        if missing:
            raise ValueError(
                f"{root_dir}: shards have no {missing}; recreate with the "
                f"matching create_shards.py --with_pixel_truth/--with_extra_truth flags."
            )
        self._pixel_keys: Tuple[str, ...] = tuple(wanted)

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of batches per epoch (used by DataLoader and schedulers)."""
        return self._n_batches

    def __iter__(self) -> Iterator[Tuple[Voxels, dict]]:
        # Split shards round-robin across DataLoader workers.
        worker_info = torch.utils.data.get_worker_info()
        shards = self.shards
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        if self.shuffle:
            # Shuffle shard order for this epoch.
            order  = torch.randperm(len(shards)).tolist()
            shards = [shards[i] for i in order]

        # Buffer: list of (coords, feats, meta_row) tuples, one entry per sample.
        buffer: List[tuple] = []
        shard_idx = 0

        def fill_buffer() -> None:
            nonlocal shard_idx
            while len(buffer) < self.buffer_size and shard_idx < len(shards):
                buffer.extend(self._load_shard(shards[shard_idx]))
                shard_idx += 1

        fill_buffer()

        while len(buffer) >= self.batch_size:
            if self.shuffle:
                # Pick batch_size samples at random from the buffer, keep the rest.
                perm          = torch.randperm(len(buffer)).tolist()
                batch_indices = perm[: self.batch_size]
                keep_indices  = perm[self.batch_size :]
                batch  = [buffer[i] for i in batch_indices]
                buffer[:] = [buffer[i] for i in keep_indices]
            else:
                # Deterministic: consume the buffer front in order.
                batch  = buffer[: self.batch_size]
                buffer[:] = buffer[self.batch_size :]
            fill_buffer()
            yield self._make_batch(batch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_shard(self, path: Path) -> List[tuple]:
        """Read one shard file and return a list of (coords, feats, meta_row) per sample."""
        with h5py.File(path, "r") as f:
            coords  = torch.from_numpy(f["coords"][()]).to(torch.int32)    # (N_total, 2)
            feats   = torch.from_numpy(f["features"][()]).to(torch.float32) # (N_total, 1)
            offsets = torch.from_numpy(f["offsets"][()]).to(torch.int64)   # (B+1,)

            event = {}
            if self._has_event_truth:
                for k in _EVENT_INT_KEYS + _EVENT_FLOAT_KEYS:
                    event[k] = f[k][()]
                event["vertex_xyz"] = f["vertex_xyz"][()]
                event["event_key"]  = f["event_key"][()]
            pixel = {k: f[k][()] for k in self._pixel_keys}

        samples = []
        for i in range(len(offsets) - 1):
            s, e = int(offsets[i]), int(offsets[i + 1])

            meta_row = None
            if self._has_event_truth or pixel:
                meta_row = {}
                if self._has_event_truth:
                    for k in _EVENT_INT_KEYS:
                        meta_row[k[:-1] if k == "labels" else k] = int(event[k][i])
                    meta_row["nu_energy"]  = float(event["nu_energy"][i])
                    meta_row["vertex_xyz"] = event["vertex_xyz"][i]
                    ev_key = event["event_key"][i]
                    if isinstance(ev_key, (bytes, np.bytes_)):
                        ev_key = ev_key.decode("utf-8")
                    meta_row["event_key"] = ev_key
                for k, arr in pixel.items():
                    meta_row[k] = arr[s:e]

            samples.append((coords[s:e].clone(), feats[s:e].clone(), meta_row))
        return samples

    def _make_batch(self, samples: List[tuple]) -> Tuple[Voxels, dict]:
        """Assemble per-sample tuples into (batched Voxels, collated meta dict)."""
        all_coords, all_feats, meta_rows = zip(*samples)
        counts  = torch.tensor([c.shape[0] for c in all_coords], dtype=torch.int64)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, dim=0)])
        coords_cat = torch.cat(all_coords, dim=0)
        feats_cat  = torch.cat(all_feats,  dim=0)
        voxels = Voxels(
            batched_coordinates=IntCoords(coords_cat, offsets=offsets),
            batched_features=CatFeatures(feats_cat, offsets=offsets),
            offsets=offsets,
        )

        if meta_rows[0] is None:
            return voxels, {}

        meta = {}
        if self._has_event_truth:
            meta["label"]      = torch.tensor([m["label"]      for m in meta_rows], dtype=torch.long)
            meta["nu_pdg"]     = torch.tensor([m["nu_pdg"]     for m in meta_rows], dtype=torch.long)
            meta["nu_ccnc"]    = torch.tensor([m["nu_ccnc"]    for m in meta_rows], dtype=torch.long)
            meta["nu_intType"] = torch.tensor([m["nu_intType"] for m in meta_rows], dtype=torch.long)
            meta["nu_energy"]  = torch.tensor([m["nu_energy"]  for m in meta_rows], dtype=torch.float32)
            meta["vertex_xyz"] = torch.tensor(
                np.stack([m["vertex_xyz"] for m in meta_rows]), dtype=torch.float32)
            meta["event_key"]  = [m["event_key"] for m in meta_rows]
        for k in self._pixel_keys:
            meta[k] = [m[k] for m in meta_rows]
        return voxels, meta
