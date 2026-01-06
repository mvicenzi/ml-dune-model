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
      - assigns neutrinos label based on top-level directory
      - loads 3 views per file, returns only one view (img_index)
    """
    def __init__(
        self,
        rootdir: Union[str, Path],
        class_names: Sequence[str] = [ "nu", "nue", "nutau" ],
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

    def _save_index_pt(self, cache_file, samples):
        data = [(str(s.path), int(s.label)) for s in samples]
        torch.save(data, cache_file)

    def _load_index_pt(self, cache_file):
        data = torch.load(cache_file, map_location="cpu")
        return [SampleIndex(path=Path(p), label=l) for p, l in data]

    def _scan(self) -> List[SampleIndex]:
        """
        Scans recursively for .gz files + assign labels based on top-level directory.
        """
        samples: List[SampleIndex] = []

        if self.use_cache and self.cache_file.exists():
            print(f"Loading dataset index from cache: {self.cache_file}")
            samples =self._load_index_pt(self.cache_file)
        else:
            # Find .gz files that live under any prodgenie* directory
            for fp in self.rootdir.glob("prodgenie*/**/*.gz"):
                if not fp.is_file():
                    continue
                if fp.suffix not in self.allowed_ext:
                    continue

                rel = fp.relative_to(self.rootdir)

                # rel.parts[0] should be prodgenie_dunevd_..._<classname>
                prodgenie_dir = rel.parts[0]
                classname = prodgenie_dir.split("_")[-1]

                if classname not in self.class_to_idx:
                    raise RuntimeError(f"Unknown classname '{classname}' from {prodgenie_dir} (file={fp})")

                label = self.class_to_idx[classname]
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
