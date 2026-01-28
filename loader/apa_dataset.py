# loader/apa_dataset.py

import h5py
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class APASampleIndex:
    path: Path
    group: str   # "1", "2", ...


class APAImageDataset(Dataset):
    """
    Dataset that:
      - scans recursively for HDF5 files ending with anode<APA>.h5
      - treats each /<group>/frame_rebinned_reco as an independent image
      - selects only one view (U, V, or W)
      - caches the index to disk
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
            / f"APAImageDataset_APA{self.apa}_view{self.view}_cache.pt"
        )

        self.samples: List[APASampleIndex] = self._scan()
        if not self.samples:
            raise RuntimeError(
                f"No samples found under {self.rootdir} for APA {self.apa}"
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
                        if (
                            isinstance(grp, h5py.Group)
                            and self.FRAME_NAME in grp
                        ):
                            samples.append(
                                APASampleIndex(path=fp, group=group)
                            )
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

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        with h5py.File(s.path, "r") as f:
            frame = f[s.group][self.FRAME_NAME][()]  # (channels, time ticks)

        # select view
        view_data = frame[self.ch_start:self.ch_end, :]

        # to torch tensor
        x = torch.from_numpy(view_data.copy()).to(torch.float32)

        # (channels, time ticks) -> (1,channels, time ticks)
        x = x.unsqueeze(0)

        return x
