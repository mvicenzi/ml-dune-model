"""The extracted feature file, and the raw-charge input built from it.

Two things live here, because the second is derived from the first and from
nothing else:

  `Features` / `load_features`   one `extract_features` .npz, with the
                                pixel<->event mapping resolved.
  `raw_charge`              the raw-charge input: `[channel, tick,
                                log_charge]` per pixel, which is the model's own
                                input and nothing else. Scored through the
                                identical split, pool and head, so a feat - raw
                                delta means "what the backbone added".

It carries no learned weights, so it doubles as a self-check: two checkpoints
extracted over the same events must produce the same score from it.

Features are consumed from disk and never recomputed.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Per-pixel truth channels the probes may use, and the extraction flag that
# writes them. `pixel_labels` needs --pixel_truth; the rest --extra_truth.
PIXEL_TRUTH_KEYS = ("pixel_labels", "pixel_energyfrac", "pixel_trackid", "pixel_truth_q")


# Loaded feature files, keyed by (resolved path, source). Entries are held for
# the life of the process and are several GB each, so only a runner scoring one
# file with several probes benefits; a single-probe process loads one and exits.
#
# Capped because an entry is ~7 GB: two is enough for the one case that wants a
# second (probe_knn_pid scores student and teacher together), and the cap keeps a
# caller that walks several files from accumulating them all in memory.
_FEATURE_CACHE: dict = {}
_FEATURE_CACHE_MAX = 2


# Wall time of the load, printed once per cache miss. On by default because the
# load is the single largest line item in a probe job and is not attributable from
# the outside: `run_probes.run_stage` reports `[Ns total for <stage>]`, which wraps
# this call, while each probe's own `[Ns]` starts after it.
#
# The cost is dominated by neither I/O nor decompression -- on one worker a 581.9 s
# load read its bytes in 9.0 s (777 MB/s) and inflated them in 69.9 s; the rest is
# allocator work and page faults for the ~9 GB resident set. So this number tracks
# memory pressure on the slot, not the file. Set PROBE_LOAD_TIMING=0 to silence.
_LOAD_TIMING = os.environ.get("PROBE_LOAD_TIMING", "1") != "0"


# ---------------------------------------------------------------------------
# Feature file
# ---------------------------------------------------------------------------

@dataclass
class Features:
    """One extracted feature file, with the pixel↔event mapping resolved."""
    path: Path
    source: str                      # "student" or "teacher"
    feat: np.ndarray                 # [N_pix, D] float16, as stored on disk
    positions: np.ndarray            # [N_pix, 2] int32  (channel, tick)
    charges: np.ndarray              # [N_pix] float32   raw ADC, pre-normalization
    offsets: np.ndarray              # [n_events + 1] int64
    pixel_event: np.ndarray          # [N_pix] int64     event index per pixel
    labels: np.ndarray               # [n_events] int64  flavor, -1 = unknown
    event_key: np.ndarray            # [n_events] str
    vertex_xyz: np.ndarray           # [n_events, 3] float64  true vertex, cm
    truth: dict = field(default_factory=dict)   # pixel truth channels present
    provenance: dict = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_pixels(self) -> int:
        return len(self.feat)

    def has(self, *keys) -> bool:
        return all(k in self.truth for k in keys)

    def require(self, *keys) -> None:
        missing = [k for k in keys if k not in self.truth]
        if missing:
            raise SystemExit(
                f"{self.path.name} lacks per-pixel truth {missing}. Re-extract with "
                f"`--pixel_truth` (pixel_labels) and `--extra_truth` "
                f"(pixel_energyfrac / pixel_trackid / pixel_truth_q)."
            )


def load_features(npz_path, source: str = "student") -> Features:
    """Read an extract_features .npz. `source` picks student or teacher features.

    Features keep their on-disk dtype (float16). A full-array upcast would cost
    ~2.8 GB on a 10 M-pixel file for no benefit: every consumer indexes a pool
    or one event first, so the cast happens on the small slice instead.

    The result is cached per (file, source) for the life of the process, so a
    runner that scores several probes in one process reads and inflates the file
    once instead of once per probe. The file is ~7 GB of compressed npz and no
    probe mutates what it returns, so the copies would be identical anyway.
    """
    if source not in ("student", "teacher"):
        raise ValueError(f"source must be 'student' or 'teacher', got {source!r}")
    path = Path(npz_path)

    cache_key = (str(path.resolve()), source)
    hit = _FEATURE_CACHE.get(cache_key)
    if hit is not None:
        return hit

    t_start = time.time()

    d = np.load(path, allow_pickle=True)
    key = f"{source}_features"
    if key not in d.files:
        raise SystemExit(f"{path.name} has no {key!r} (present: {sorted(d.files)})")

    feat = d[key]
    offsets = d["offsets"].astype(np.int64)
    counts = np.diff(offsets)
    pixel_event = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    if len(pixel_event) != len(feat):
        raise SystemExit(
            f"{path.name}: offsets cover {len(pixel_event)} pixels but {key} has "
            f"{len(feat)} rows — the file is inconsistent."
        )

    truth = {k: d[k] for k in PIXEL_TRUTH_KEYS if k in d.files}
    for k, arr in truth.items():
        if len(arr) != len(feat):
            raise SystemExit(
                f"{path.name}: {k} has {len(arr)} entries but {len(feat)} pixels."
            )

    # Provenance scalars, written by newer extractions. Absent in older files;
    # the raw-charge input falls back to a parameter-free transform (see
    # `raw_charge`).
    prov = {}
    for k in ("epoch", "backbone_name", "encoding_range", "feature_dim", "apa", "view",
              "use_log_transform", "feat_min_val", "feat_max_val",
              "backbone_kwargs_applied", "extraction_source"):
        if k in d.files:
            v = d[k]
            prov[k] = v.item() if getattr(v, "ndim", 1) == 0 else v.tolist()

    fx = Features(
        path=path, source=source, feat=feat,
        positions=d["positions"].astype(np.int32),
        charges=np.asarray(d["charges"]).reshape(-1).astype(np.float32),
        offsets=offsets, pixel_event=pixel_event,
        labels=d["labels"].astype(np.int64),
        event_key=d["event_key"],
        vertex_xyz=d["vertex_xyz"].astype(np.float64),
        truth=truth, provenance=prov,
    )

    if _LOAD_TIMING:
        print(f"  [load_features {time.time() - t_start:.0f}s]", flush=True)
    # Drop the oldest rather than grow: holding a stale 7 GB entry alongside the
    # one in use is what would push a 16 GB slot over.
    while len(_FEATURE_CACHE) >= _FEATURE_CACHE_MAX:
        del _FEATURE_CACHE[next(iter(_FEATURE_CACHE))]
    _FEATURE_CACHE[cache_key] = fx
    return fx


# ---------------------------------------------------------------------------
# Raw charge
# ---------------------------------------------------------------------------

def log_charge(charge: np.ndarray, min_val: Optional[float] = None,
               max_val: Optional[float] = None) -> np.ndarray:
    """Compress raw ADC charge to ~[-1, 1].

    With `min_val`/`max_val` this is `dino.transforms.FeatureLogTransform`, i.e.
    exactly what the backbone was fed during training — so the charge column of
    the raw-charge input is the model's real input, not an approximation of it.
    Older feature files carry no transform parameters; callers then omit them and
    get a parameter-free log10(1 + q), which is monotone in charge and adequate
    here (the heads standardize their inputs anyway) but not bit-comparable to a
    run scored with the trained transform.
    """
    q = np.clip(np.asarray(charge, dtype=np.float64), 0.0, None)
    if min_val is None or max_val is None:
        return np.log10(1.0 + q)
    y0 = np.log10(min_val)
    y1 = np.log10(max_val + min_val)
    return 2.0 * (np.log10(q + min_val) - y0) / (y1 - y0) - 1.0


def raw_charge_kind(fx: Features) -> str:
    """Which charge transform the raw-charge input uses: `trained` or `log10_1p`.

    Recorded with every result and checked by merge.py, because the two are
    different raw-charge inputs: a `trained` delta is not comparable with a
    `log10_1p` one even for the same checkpoint.
    """
    prov = fx.provenance
    if prov.get("use_log_transform") and "feat_min_val" in prov:
        return "trained"
    return "log10_1p"


def raw_charge(fx: Features) -> np.ndarray:
    """The raw-charge input: `[channel, tick, log_charge]` per pixel.

    Depends on the extraction and the charge transform, never on the checkpoint
    weights — which is what makes it the number to beat rather than a second
    model.
    """
    if raw_charge_kind(fx) == "trained":
        prov = fx.provenance
        lq = log_charge(fx.charges, prov["feat_min_val"], prov["feat_max_val"])
    else:
        print(f"  [warn] {fx.path.name} carries no charge-transform provenance; "
              f"raw-charge input uses log10(1+q), so its feat-raw deltas are NOT "
              f"comparable with runs extracted after provenance was added. "
              f"Re-extract to fix.")
        lq = log_charge(fx.charges)
    return np.stack([fx.positions[:, 0].astype(np.float64),
                     fx.positions[:, 1].astype(np.float64),
                     lq], axis=1).astype(np.float32)
