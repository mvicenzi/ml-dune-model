# loader/apa_sparse_dataset.py

import h5py
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

from loader.apa_dataset import APASampleIndex


class APASparseDataset(Dataset):
    """
    Dataset that:
      - scans recursively for sparse HDF5 files ending with anode<APA>.h5
      - treats each /<group>/frame_rebinned_reco as an independent sparse sample
      - selects only one view (U, V, or W) by filtering on the channel coordinate
      - returns a single-item Voxels object per sample
      - caches the index to disk

    Expected sparse HDF5 structure per group:
        /<group>/frame_rebinned_reco/coords    (N, 2) int32  — (channel, tick)
        /<group>/frame_rebinned_reco/features  (N,)   float32
    """

    VIEW_RANGES = {
        "U": (0, 800),
        "V": (800, 1600),
        "W": (1600, 2650),
    }

    FRAME_NAME = "frame_rebinned_reco"

    def __init__(
        self,
        rootdir: Union[str, Path],
        apa: int,
        view: str,
        use_cache: bool = True,
        cache_dir: Union[str, Path] = "./data",
    ):
        self.rootdir = Path(rootdir)
        self.apa = int(apa)

        self.view = view.upper()
        if self.view not in self.VIEW_RANGES:
            raise ValueError(f"view must be one of {list(self.VIEW_RANGES)}, got {view}")

        self.ch_start, self.ch_end = self.VIEW_RANGES[self.view]

        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.cache_file = (
            self.cache_dir
            / f"APASparseDataset_APA{self.apa}_view{self.view}_cache.pt"
        )

        self.samples: List[APASampleIndex] = self._scan()
        if not self.samples:
            raise RuntimeError(
                f"No sparse samples found under {self.rootdir} for APA {self.apa}"
            )

    # -------------------------
    # cache helpers
    # -------------------------

    def _save_index_pt(self, cache_file: Path, samples: List[APASampleIndex]):
        data = [(str(s.path), s.group) for s in samples]
        torch.save(data, cache_file)

    def _load_index_pt(self, cache_file: Path) -> List[APASampleIndex]:
        data = torch.load(cache_file, map_location="cpu")
        return [APASampleIndex(path=Path(p), group=g) for p, g in data]

    # -------------------------
    # scanning
    # -------------------------

    def _scan(self) -> List[APASampleIndex]:
        samples: List[APASampleIndex] = []

        if self.use_cache and self.cache_file.exists():
            print(f"Loading dataset index from cache: {self.cache_file}")
            return self._load_index_pt(self.cache_file)

        print(f"Cache does not exist: {self.cache_file} -- generating new one!")
        pattern = f"*anode{self.apa}.h5"

        for fp in self.rootdir.rglob(pattern):
            if not fp.is_file():
                continue

            try:
                with h5py.File(fp, "r") as f:
                    for group in f.keys():
                        grp = f[group]
                        if not isinstance(grp, h5py.Group):
                            continue
                        if self.FRAME_NAME not in grp:
                            continue
                        frame_entry = grp[self.FRAME_NAME]
                        # Accept only sparse format: subgroup with coords and features
                        if (
                            isinstance(frame_entry, h5py.Group)
                            and "coords" in frame_entry
                            and "features" in frame_entry
                        ):
                            samples.append(APASampleIndex(path=fp, group=group))
            except OSError as e:
                print(f"Warning: could not open {fp}: {e}")

        # stable ordering
        samples.sort(key=lambda s: (str(s.path), int(s.group)))

        if self.use_cache:
            print(f"Saving dataset index to cache: {self.cache_file}")
            self._save_index_pt(self.cache_file, samples)

        return samples

    # -------------------------
    # Dataset API
    # -------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Voxels:
        s = self.samples[idx]

        with h5py.File(s.path, "r") as f:
            frame = f[s.group][self.FRAME_NAME]
            coords = torch.from_numpy(frame["coords"][()]).to(torch.int32)   # (N, 2)
            feats  = torch.from_numpy(frame["features"][()]).to(torch.float32)  # (N,)

        # select view: keep only active pixels in [ch_start, ch_end)
        mask = (coords[:, 0] >= self.ch_start) & (coords[:, 0] < self.ch_end)
        coords = coords[mask].clone()
        feats  = feats[mask]

        # rebase channel coordinate to 0 for this view
        coords[:, 0] -= self.ch_start

        # features must be (N, C) for Voxels
        feats = feats.unsqueeze(1)

        # single-item batch offsets
        n = coords.shape[0]
        offsets = torch.tensor([0, n], dtype=torch.int64)

        return Voxels(
            batched_coordinates=IntCoords(coords, offsets=offsets),
            batched_features=CatFeatures(feats, offsets=offsets),
            offsets=offsets,
        )
