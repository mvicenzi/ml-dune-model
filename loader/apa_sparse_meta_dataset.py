# loader/apa_sparse_meta_dataset.py

import warnings
import h5py
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from warpconvnet.geometry.types.voxels import Voxels

from loader.apa_sparse_dataset import APASparseDataset

# Maps (nu_pdg, nu_ccnc) → class index.
# nu_ccnc == 1 means neutral-current (any flavour), checked first.
#   0 → numuCC  (nu_pdg=14, nu_ccnc=0)
#   1 → nueCC   (nu_pdg=12, nu_ccnc=0)
#   2 → NC      (nu_ccnc=1)
#  -1 → skip    (anything else, e.g. nu_tau CC)
CLASS_NAMES = ["numuCC", "nueCC", "NC"]


def _classify(nu_pdg: int, nu_ccnc: int) -> int:
    if nu_ccnc == 1:
        return 2          # NC — any flavour
    if nu_pdg == 14 and nu_ccnc == 0:
        return 0          # numuCC
    if nu_pdg == 12 and nu_ccnc == 0:
        return 1          # nueCC
    return -1             # skip (e.g. nu_tau CC)


class APASparseMetaDataset(APASparseDataset):
    """
    Extends APASparseDataset to also load per-event truth labels from the
    co-located metadata HDF5 file.

    Each __getitem__ returns (voxels: Voxels, label: int) where label is:
        0  numuCC  (nu_pdg=14, nu_ccnc=0)
        1  nueCC   (nu_pdg=12, nu_ccnc=0)
        2  NC      (nu_ccnc=1)
       -1  unknown / skip (metadata missing or unmapped pdg)

    Metadata file co-location: given
        {dir}/{basename}_pixeldata-anode{N}.h5
    the metadata file is
        {dir}/{basename}_metadata.h5
    and carries the HDF5 structure:
        /{group}/metadata  — numpy structured array (1,) with fields
                             nu_pdg, nu_ccnc, nu_intType, nu_energy,
                             nu_vertex_{x,y,z}
    """

    def __init__(
        self,
        rootdir: Union[str, Path],
        apa: int,
        view: str,
        use_cache: bool = True,
        cache_dir: Union[str, Path] = "./data",
        view_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        frame_name: str = "frame_rebinned_reco",
    ):
        super().__init__(
            rootdir=rootdir,
            apa=apa,
            view=view,
            use_cache=use_cache,
            cache_dir=cache_dir,
            view_ranges=view_ranges,
            frame_name=frame_name,
        )
        self._warned_missing: set = set()

    def __getitem__(self, idx: int) -> tuple[Voxels, int]:
        voxels = super().__getitem__(idx)

        s = self.samples[idx]
        label = self._read_label(s.path, s.group)
        return voxels, label

    def _read_label(self, pixeldata_path: Path, group: str) -> int:
        # Derive metadata file path from pixeldata filename
        suffix = f"_pixeldata-anode{self.apa}.h5"
        if not pixeldata_path.name.endswith(suffix):
            # Older naming: "..._anode3.h5" without "_pixeldata" prefix.
            # Metadata lives next to the pixeldata file with the _metadata suffix.
            basename = pixeldata_path.stem   # drop .h5
            metadata_path = pixeldata_path.parent / f"{basename}_metadata.h5"
        else:
            basename = pixeldata_path.name[: -len(suffix)]
            metadata_path = pixeldata_path.parent / f"{basename}_metadata.h5"

        if not metadata_path.exists():
            if str(metadata_path) not in self._warned_missing:
                warnings.warn(f"Metadata file not found: {metadata_path}")
                self._warned_missing.add(str(metadata_path))
            return -1

        try:
            with h5py.File(metadata_path, "r") as f:
                row = f[group]["metadata"][0]
                nu_pdg  = int(row["nu_pdg"])
                nu_ccnc = int(row["nu_ccnc"])
            return _classify(nu_pdg, nu_ccnc)
        except Exception as e:
            if str(metadata_path) not in self._warned_missing:
                warnings.warn(f"Could not read metadata from {metadata_path}[{group}]: {e}")
                self._warned_missing.add(str(metadata_path))
            return -1
