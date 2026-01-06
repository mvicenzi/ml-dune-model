# loader/dataset.py
import zlib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

@dataclass(frozen=True)
class SampleIndex:
    path: Path
    label: int

class DUNEImageDataset(Dataset):
    """
    Dataset that:
      - scans recursively for .gz files
      - assigns neutrinos label based on .info file
      - loads 3 views per file, returns only one view (img_index)
    """
    def __init__(
        self,
        rootdir: Union[str, Path],
        class_names: Sequence[str] = [ "numu", "nue", "nutau", "NC" ],
        view_index: int = 0,
        allowed_ext: Tuple[str, ...] = (".gz",),
        use_cache: bool = True,
        cache_dir: Union[str, Path] = "./data",
    ):
        
        self.rootdir = Path(rootdir)
        self.view_index = int(view_index)
        if self.view_index not in (0, 1, 2):
            raise ValueError(f"view_index must be 0,1,2 but got {view_index}")

        self.allowed_ext = allowed_ext
        self.use_cache = use_cache

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / f"DUNEImageDataset_view{self.view_index}_cache.pkl"

        self.class_names = list(class_names)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.class_names)}
        self.idx_to_class: Dict[int, str] = {i: c for c, i in self.class_to_idx.items()}

        self.samples: List[SampleIndex] = self._scan()
        if not self.samples:
            raise RuntimeError(f"No samples found under {self.rootdir} with extensions {self.allowed_ext}")
        
        self.labels = torch.empty(len(self.samples), dtype=torch.long)
        for i, sample in enumerate(self.samples):
            self.labels[i] = sample.label 

    def _save_index_pt(self, cache_file, samples):
        data = [(str(s.path), int(s.label)) for s in samples]
        torch.save(data, cache_file)

    def _load_index_pt(self, cache_file):
        data = torch.load(cache_file, map_location="cpu")
        return [SampleIndex(path=Path(p), label=l) for p, l in data]

    def _assign_label(self, fp: Path) -> int:
        """
        Assigns label based on .info file associated with .gz file.
        The .info file is expected to be in the same directory as the .gz file.
        """
        filename = fp.stem # without .gz
        info_file = fp.parent / f"{filename}.info"
        if not info_file.exists():
            raise RuntimeError(f"Missing .info file for {fp} at {info_file}")
        
        nupdg = None
        with open(info_file, "r") as f:
            info = f.readlines()
            nupdg = int(info[7].strip())
        
        label = None
        if nupdg == 1:
            label = self.class_to_idx.get("NC")
        elif nupdg == 12 or nupdg == -12:
            label = self.class_to_idx.get("nue")
        elif nupdg == 14 or nupdg == -14:
            label = self.class_to_idx.get("numu")
        elif nupdg == 16 or nupdg == -16:
            label = self.class_to_idx.get("nutau")
        else:
            print(f"Warning: unexpected nupdg {nupdg} in {info_file}")

        return label

    def _scan(self) -> List[SampleIndex]:
        """
        Scans recursively for .gz files + assign labels based on top-level directory.
        """
        samples: List[SampleIndex] = []

        if self.use_cache and self.cache_file.exists():
            print(f"Loading dataset index from cache: {self.cache_file}")
            samples = self._load_index_pt(self.cache_file)
        else:
            print(f"Cache does not exist: {self.cache_file} -- generating new one!")
            # Find .gz files that live under any prodgenie* directory
            for fp in self.rootdir.glob("prodgenie*/**/*.gz"):
                if not fp.is_file():
                    continue
                if fp.suffix not in self.allowed_ext:
                    continue

                label = self._assign_label(fp)
                samples.append(SampleIndex(path=fp, label=label))

        samples.sort(key=lambda s: str(s.path))
        if self.use_cache and not self.cache_file.exists():
            print(f"Saving dataset index to cache: {self.cache_file}")
            self._save_index_pt(self.cache_file, samples)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):

        s = self.samples[idx]

        imgs: List[np.ndarray] = []

        with open(s.path, "rb") as f:
            compressed = f.read()

            raw = zlib.decompress(compressed)
            arr = np.frombuffer(raw, dtype=np.uint8)

            expected = int(np.prod((3, 500, 500)))
            if arr.size != expected:
                raise ValueError(f"{s.path}: expected {expected} values, got {arr.size}")

            arr = arr.reshape((3, 500, 500))  # (3, 500, 500)
            imgs = [arr[i] for i in range(3)]
    
        if len(imgs) != 3:
            raise ValueError(f"{s.path} did not yield 3 images (got {len(imgs)})")

        # select only one view
        x = imgs[self.view_index]

        # add channel dim if your model expects (C,H,W)
        x = torch.from_numpy(x.copy()).to(torch.float32)  # (500,500)
        x = x.unsqueeze(0)  # (1,500,500)

        y = int(s.label)
        return x, y
