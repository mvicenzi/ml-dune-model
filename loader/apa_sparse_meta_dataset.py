# loader/apa_sparse_meta_dataset.py

import warnings
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from loader.apa_sparse_dataset import APASparseDataset

# Maps (nu_pdg, nu_ccnc) → class index.
# nu_ccnc == 1 means neutral-current (any flavour), checked first.
# Neutrinos and antineutrinos share a class (|nu_pdg|).
#   0 → numuCC  (nu_pdg=±14, nu_ccnc=0)
#   1 → nueCC   (nu_pdg=±12, nu_ccnc=0)
#   2 → NC      (nu_ccnc=1)
#  -1 → skip    (anything else, e.g. nu_tau CC)
CLASS_NAMES = ["numuCC", "nueCC", "NC"]


def _classify(nu_pdg: int, nu_ccnc: int) -> int:
    if nu_ccnc == 1:
        return 2          # NC — any flavour
    if abs(nu_pdg) == 14 and nu_ccnc == 0:
        return 0          # numuCC (nu or anti-nu)
    if abs(nu_pdg) == 12 and nu_ccnc == 0:
        return 1          # nueCC (nu or anti-nu)
    return -1             # skip (e.g. nu_tau CC)


class APASparseMetaDataset(APASparseDataset):
    """
    Extends APASparseDataset to also load truth information. This is the
    default dataset for training scripts and diagnostics; the bare
    APASparseDataset (voxels only) is kept for low-level inspection.

    __getitem__ always returns (voxels: Voxels, meta: dict). The event-level
    truth (from the co-located metadata HDF5 file) is always present — one
    small file open per event; consumers that only need pixels (e.g. SSL
    training) simply ignore it. The per-pixel tiers are opt-in flags because
    they cost an extra pixeldata read plus an O(N_pixels) alignment per event.

    Event-level meta keys (always present):
        label:      int            class index from `_classify`
                                   (0=numuCC, 1=nueCC, 2=NC, -1=skip)
        nu_pdg:     int
        nu_ccnc:    int
        nu_intType: int
        nu_energy:  float
        vertex_xyz: Tensor[3]      float32, detector coords
        event_key:  str            f"{pixeldata_path.name}:{group}" for traceability
    All numeric fields default to -1 / 0.0 when metadata is missing; the
    caller can detect this via label == -1.

    - return_pixel_truth=True:
        Also adds to the meta dict:
            pixel_labels: np.ndarray[N_pixels] int8
                Per-pixel class label from frame_label_1st (produced by
                data/scripts/classify_pixels_sparse.py), aligned to the order
                of pixels in the returned Voxels.  Taxonomy:
                0=Background/no-truth, 1=Track, 2=Shower, 3=Michel,
                4=DeltaRay, 5=Blip, 6=Other.

    - return_extra_truth=True (requires return_pixel_truth=True):
        Also adds, same alignment and 0/0.0 fill for pixels without truth:
            pixel_energyfrac: np.ndarray[N_pixels] float32  (frame_energyfrac_1st)
            pixel_trackid:    np.ndarray[N_pixels] int32    (frame_trackid_1st,
                signed: negative = G4-dropped secondary of parent abs(id))
            pixel_truth_q:    np.ndarray[N_pixels] float32  (frame_total_numelectrons)

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
        datadir: Union[str, Path],
        apa: int,
        view: str,
        use_cache: bool = True,
        cache_dir: Union[str, Path] = "./data",
        view_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        frame_name: str = "frame_rebinned_reco",
        return_pixel_truth: bool = False,
        return_extra_truth: bool = False,
    ):
        super().__init__(
            datadir=datadir,
            apa=apa,
            view=view,
            use_cache=use_cache,
            cache_dir=cache_dir,
            view_ranges=view_ranges,
            frame_name=frame_name,
        )
        self.return_pixel_truth = return_pixel_truth
        self.return_extra_truth = return_extra_truth
        self._warned_missing: set = set()

    def __getitem__(self, idx: int):
        voxels = super().__getitem__(idx)

        s = self.samples[idx]
        meta = self._read_event_truth(s.path, s.group)
        if self.return_pixel_truth:
            meta.update(self._read_pixel_truth_arrays(s.path, s.group))
        return voxels, meta

    # ------------------------------------------------------------------
    # Metadata path helpers
    # ------------------------------------------------------------------

    def _metadata_path(self, pixeldata_path: Path) -> Path:
        suffix = f"_pixeldata-anode{self.apa}.h5"
        if not pixeldata_path.name.endswith(suffix):
            # Older naming: "..._anode3.h5" without "_pixeldata" prefix.
            basename = pixeldata_path.stem   # drop .h5
        else:
            basename = pixeldata_path.name[: -len(suffix)]
        return pixeldata_path.parent / f"{basename}_metadata.h5"

    def _warn_once(self, metadata_path: Path, msg: str) -> None:
        key = str(metadata_path)
        if key not in self._warned_missing:
            warnings.warn(msg)
            self._warned_missing.add(key)

    # ------------------------------------------------------------------
    # Event-truth reader
    # ------------------------------------------------------------------

    def _unknown_metadata(self, event_key: str) -> dict:
        """Sentinel metadata dict used when the metadata file is missing/unreadable."""
        return {
            "label":      -1,
            "nu_pdg":     0,
            "nu_ccnc":    -1,
            "nu_intType": -1,
            "nu_energy":  0.0,
            "vertex_xyz": torch.zeros(3, dtype=torch.float32),
            "event_key":  event_key,
        }

    def _read_event_truth(self, pixeldata_path: Path, group: str) -> dict:
        event_key = f"{pixeldata_path.name}:{group}"
        metadata_path = self._metadata_path(pixeldata_path)

        if not metadata_path.exists():
            self._warn_once(metadata_path, f"Metadata file not found: {metadata_path}")
            return self._unknown_metadata(event_key)

        try:
            with h5py.File(metadata_path, "r") as f:
                row = f[group]["metadata"][0]
                nu_pdg     = int(row["nu_pdg"])
                nu_ccnc    = int(row["nu_ccnc"])
                nu_intType = int(row["nu_intType"])
                nu_energy  = float(row["nu_energy"])
                vx = float(row["nu_vertex_x"])
                vy = float(row["nu_vertex_y"])
                vz = float(row["nu_vertex_z"])
        except Exception as e:
            self._warn_once(metadata_path, f"Could not read metadata from {metadata_path}[{group}]: {e}")
            return self._unknown_metadata(event_key)

        return {
            "label":      _classify(nu_pdg, nu_ccnc),
            "nu_pdg":     nu_pdg,
            "nu_ccnc":    nu_ccnc,
            "nu_intType": nu_intType,
            "nu_energy":  nu_energy,
            "vertex_xyz": torch.tensor([vx, vy, vz], dtype=torch.float32),
            "event_key":  event_key,
        }

    # ------------------------------------------------------------------
    # Pixel-level truth reader
    # ------------------------------------------------------------------

    # Extra per-pixel truth frames: meta key -> (HDF5 frame name, output dtype).
    # Track ids can be positive or negative (negative = G4-dropped secondary
    # of parent abs(id)); both are kept as-is.
    _EXTRA_TRUTH_FRAMES = {
        "pixel_energyfrac": ("frame_energyfrac_1st",     np.float32),
        "pixel_trackid":    ("frame_trackid_1st",        np.int32),
        "pixel_truth_q":    ("frame_total_numelectrons", np.float32),
    }

    def _empty_pixel_truth(self, n: int) -> dict:
        """All-fill pixel-truth dict of length n (label 0 = Background/no-truth)."""
        out = {"pixel_labels": np.zeros(n, dtype=np.int8)}
        if self.return_extra_truth:
            for key, (_, dtype) in self._EXTRA_TRUTH_FRAMES.items():
                out[key] = np.zeros(n, dtype=dtype)
        return out

    def _read_pixel_truth_arrays(self, pixeldata_path: Path, group: str) -> dict:
        """
        Read per-pixel truth frames (frame_label_1st and, with
        return_extra_truth, energyfrac/trackid/total_numelectrons) and
        return arrays aligned to the reco pixel order produced by
        APASparseDataset.__getitem__ for the same (path, group).

        Reco pixels with no truth hit carry the fill value (0 / 0.0).
        """
        try:
            with h5py.File(pixeldata_path, "r") as f:
                g = f[group]
                reco_coords = g[self.frame_name]["coords"][()]   # (N, 2) int32

                mask_reco = ((reco_coords[:, 0] >= self.ch_start) &
                             (reco_coords[:, 0] < self.ch_end))
                n_view = int(mask_reco.sum())

                if "frame_label_1st" not in g:
                    self._warn_once(
                        pixeldata_path,
                        f"frame_label_1st not found in {pixeldata_path}[{group}]; "
                        f"pixel truth defaults to 0 (Background). Pre-2026-06-11 "
                        f"productions (frame_pid_*) are no longer supported.",
                    )
                    return self._empty_pixel_truth(n_view)

                label_coords = g["frame_label_1st"]["coords"][()]    # (M, 2) int32
                label_feats  = g["frame_label_1st"]["features"][()]  # (M,)   int8

                extra_raw = {}
                if self.return_extra_truth:
                    for key, (frame, _) in self._EXTRA_TRUTH_FRAMES.items():
                        if frame in g:
                            extra_raw[key] = (g[frame]["coords"][()],
                                              g[frame]["features"][()])
                        else:
                            self._warn_once(
                                pixeldata_path,
                                f"{frame} not found in {pixeldata_path}; "
                                f"{key} defaults to 0.",
                            )
        except Exception as e:
            self._warn_once(
                pixeldata_path,
                f"Could not read pixel truth from {pixeldata_path}[{group}]: {e}",
            )
            return self._empty_pixel_truth(0)

        # Filter to this view's channel range (same logic as APASparseDataset)
        reco_view = reco_coords[mask_reco]
        mask_lbl  = ((label_coords[:, 0] >= self.ch_start) &
                     (label_coords[:, 0] < self.ch_end))
        lbl_view_coords = label_coords[mask_lbl]
        lbl_view_feats  = label_feats[mask_lbl]

        # Map each reco pixel to its row in the (view-filtered) truth frame.
        row_lookup = {
            (int(c[0]), int(c[1])): i for i, c in enumerate(lbl_view_coords)
        }
        rows = np.array(
            [row_lookup.get((int(c[0]), int(c[1])), -1) for c in reco_view],
            dtype=np.int64,
        )
        has = rows >= 0

        out = self._empty_pixel_truth(len(reco_view))
        out["pixel_labels"][has] = lbl_view_feats[rows[has]].astype(np.int8)

        for key, (coords_f, feats_f) in extra_raw.items():
            dtype = self._EXTRA_TRUTH_FRAMES[key][1]
            m = ((coords_f[:, 0] >= self.ch_start) &
                 (coords_f[:, 0] < self.ch_end))
            f_view_coords = coords_f[m]
            f_view_feats  = feats_f[m]

            # All truth frames share coords by construction (classify script
            # reuses frame_trackid coords) — reuse the label lookup when true.
            if (f_view_coords.shape == lbl_view_coords.shape
                    and np.array_equal(f_view_coords, lbl_view_coords)):
                f_rows, f_has = rows, has
            else:
                self._warn_once(
                    pixeldata_path,
                    f"{self._EXTRA_TRUTH_FRAMES[key][0]} coords differ from "
                    f"frame_label_1st in {pixeldata_path}; using per-frame lookup.",
                )
                lk = {(int(c[0]), int(c[1])): i
                      for i, c in enumerate(f_view_coords)}
                f_rows = np.array(
                    [lk.get((int(c[0]), int(c[1])), -1) for c in reco_view],
                    dtype=np.int64,
                )
                f_has = f_rows >= 0

            vals = f_view_feats[f_rows[f_has]]
            if np.issubdtype(dtype, np.integer) and vals.dtype.kind == "f":
                vals = np.rint(vals)
            out[key][f_has] = vals.astype(dtype)

        return out
