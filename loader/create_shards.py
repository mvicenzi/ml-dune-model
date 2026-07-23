"""
Create pre-sharded HDF5 files from an APASparseDataset.

Each shard file uses a flat layout (not one group per event) so the pixel
payload can be loaded with exactly 3 HDF5 reads regardless of shard size.
Event-level truth is always stored alongside (it is a few scalars per event);
per-pixel truth tiers are opt-in flags. The reader (APASparseShardedDataset)
detects and returns whatever truth datasets are present automatically.

HDF5 layout per shard file:
    /coords       (N_pix, 2)  int32   -- channel/tick coords (view-rebased)
    /features     (N_pix, 1)  float32 -- pixel ADC values
    /offsets      (N_img+1,)  int64   -- CSR pixel offsets (shared by pixel truth)
    /labels       (N_img,)    int32   -- event class: 0=numuCC 1=nueCC 2=NC -1=unknown
    /nu_pdg       (N_img,)    int32
    /nu_ccnc      (N_img,)    int32
    /nu_intType   (N_img,)    int32
    /nu_energy    (N_img,)    float32
    /vertex_xyz   (N_img, 3)  float32
    /event_key    (N_img,)    bytes   -- UTF-8 "{file}:{group}" traceability string
  with --with_pixel_truth additionally:
    /pixel_labels (N_pix,)    int8    -- per-pixel class label (0=Background/no-truth,
                                         1=Track 2=Shower 3=Michel 4=DeltaRay 5=Blip
                                         6=Other); same CSR as /coords
  with --with_extra_truth additionally:
    /pixel_energyfrac (N_pix,) float32 -- truth-overlap score (frame_energyfrac_1st)
    /pixel_trackid    (N_pix,) int32   -- truth track id, signed (frame_trackid_1st)
    /pixel_truth_q    (N_pix,) float32 -- truth charge (frame_total_numelectrons)

A metadata.json alongside records apa, view, n_samples, shard_size, seed and
which truth tiers are present.

Two use cases:
  - Training shards: shard the FULL dataset (default) — omit --n_shards.
  - Diagnostics shards: a small fixed random subset reused across checkpoint
    extractions — pass --n_shards (e.g. 10) and --with_pixel_truth.

Usage:
    python -m loader.create_shards \\
        --datadir /path/to/dataset \\
        --apa 0 --view W \\
        --outdir /path/to/shards \\
        --cache_dir /path/to/cache \\
        [--shard_size 1000] [--seed 42] [--n_shards 10]
        [--with_pixel_truth] [--with_extra_truth]

The script skips existing shard files, so it can be safely re-run to resume
an interrupted run.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from loader.apa_sparse_meta_dataset import APASparseMetaDataset

LABEL_FORMAT = "classes7_v1"
CLASS_NAMES = ["Background", "Track", "Shower", "Michel", "DeltaRay", "Blip", "Other"]

EXTRA_TRUTH_KEYS = {
    "pixel_energyfrac": np.float32,
    "pixel_trackid":    np.int32,
    "pixel_truth_q":    np.float32,
}


def create_shards(
    datadir: str,
    apa: int,
    view: str,
    outdir: str,
    cache_dir: str,
    shard_size: int = 1000,
    seed: int = 42,
    n_shards: int = None,
    with_pixel_truth: bool = False,
    with_extra_truth: bool = False,
) -> None:
    if with_extra_truth:
        with_pixel_truth = True

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = APASparseMetaDataset(
        datadir=datadir,
        apa=apa,
        view=view,
        use_cache=True,
        cache_dir=cache_dir,
        return_pixel_truth=with_pixel_truth,
        return_extra_truth=with_extra_truth,
    )
    n_total = len(dataset)
    print(f"Dataset size : {n_total}")

    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(n_total, generator=rng).tolist()

    if n_shards is not None:
        # Diagnostics mode: fixed random subset, reused across extractions.
        indices = indices[: n_shards * shard_size]
    n_samples = len(indices)

    shards = [indices[i : i + shard_size] for i in range(0, n_samples, shard_size)]
    print(f"Sharding     : {n_samples} samples → {len(shards)} x ~{shard_size} → {outdir}")

    # Write metadata.json before the loop so a partial run is still usable.
    meta_out = {
        "n_samples":    n_samples,
        "n_shards":     len(shards),
        "shard_size":   shard_size,
        "apa":          apa,
        "view":         view,
        "seed":         seed,
        "pixel_truth":  with_pixel_truth,
        "extra_truth":  with_extra_truth,
        "label_format": LABEL_FORMAT,
        "class_names":  CLASS_NAMES,
    }
    with open(outdir / "metadata.json", "w") as fp:
        json.dump(meta_out, fp, indent=2)

    for shard_idx, shard_indices in enumerate(shards):
        outpath = outdir / f"shard_{shard_idx:05d}.h5"
        if outpath.exists():
            print(f"  [{shard_idx + 1}/{len(shards)}] skip (exists): {outpath.name}")
            continue

        coords_list, feats_list, pixlbl_list = [], [], []
        extra_lists = {k: [] for k in EXTRA_TRUTH_KEYS} if with_extra_truth else {}
        offsets = [0]
        labels, nu_pdg, nu_ccnc, nu_intType, nu_energy = [], [], [], [], []
        vertex_xyz, event_keys = [], []

        for i, idx in enumerate(shard_indices):
            voxels, meta = dataset[idx]

            c = voxels.coordinate_tensor.numpy()   # (N, 2) int32
            f = voxels.feature_tensor.numpy()      # (N, 1) float32

            coords_list.append(c)
            feats_list.append(f)
            offsets.append(offsets[-1] + len(c))

            if with_pixel_truth:
                pixlbl = meta["pixel_labels"]      # (N,) int8 np.ndarray
                assert len(pixlbl) == len(c), (
                    f"pixel_labels length {len(pixlbl)} != coords length {len(c)} "
                    f"for dataset index {idx}"
                )
                pixlbl_list.append(pixlbl)
                for k, lst in extra_lists.items():
                    lst.append(meta[k])

            labels.append(meta["label"])
            nu_pdg.append(meta["nu_pdg"])
            nu_ccnc.append(meta["nu_ccnc"])
            nu_intType.append(meta["nu_intType"])
            nu_energy.append(float(meta["nu_energy"]))
            vertex_xyz.append(meta["vertex_xyz"].numpy())   # (3,) float32
            event_keys.append(meta["event_key"].encode("utf-8"))

            if (i + 1) % 200 == 0 or (i + 1) == len(shard_indices):
                print(f"    {i + 1}/{len(shard_indices)}")

        coords_cat  = np.concatenate(coords_list, axis=0).astype(np.int32)
        feats_cat   = np.concatenate(feats_list,  axis=0).astype(np.float32)
        offsets_arr = np.array(offsets,           dtype=np.int64)
        vxyz_arr    = np.array(vertex_xyz,        dtype=np.float32)   # (N_img, 3)

        with h5py.File(outpath, "w") as hf:
            hf.create_dataset("coords",     data=coords_cat,                    compression="gzip", compression_opts=6)
            hf.create_dataset("features",   data=feats_cat,                     compression="gzip", compression_opts=6)
            hf.create_dataset("offsets",    data=offsets_arr)
            hf.create_dataset("labels",     data=np.array(labels,     dtype=np.int32))
            hf.create_dataset("nu_pdg",     data=np.array(nu_pdg,     dtype=np.int32))
            hf.create_dataset("nu_ccnc",    data=np.array(nu_ccnc,    dtype=np.int32))
            hf.create_dataset("nu_intType", data=np.array(nu_intType, dtype=np.int32))
            hf.create_dataset("nu_energy",  data=np.array(nu_energy,  dtype=np.float32))
            hf.create_dataset("vertex_xyz", data=vxyz_arr)
            hf.create_dataset("event_key",  data=np.array(event_keys, dtype=object),
                               dtype=h5py.special_dtype(vlen=bytes))
            if with_pixel_truth:
                pixlbl_cat = np.concatenate(pixlbl_list, axis=0).astype(np.int8)
                hf.create_dataset("pixel_labels", data=pixlbl_cat,              compression="gzip", compression_opts=6)
            for k, lst in extra_lists.items():
                hf.create_dataset(k, data=np.concatenate(lst, axis=0).astype(EXTRA_TRUTH_KEYS[k]),
                                  compression="gzip", compression_opts=6)

        print(f"  [{shard_idx + 1}/{len(shards)}] wrote {outpath.name}  "
              f"({coords_cat.shape[0]} pixels, {len(offsets) - 1} images)")

    print(f"\nDone. Wrote {len(shards)} shards + metadata.json to {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-shard an APASparseDataset into flat HDF5 shard files."
    )
    parser.add_argument("--datadir",    required=True,           help="Root directory of source HDF5 files")
    parser.add_argument("--apa",        type=int, required=True, help="APA number")
    parser.add_argument("--view",       default="W",             help="Wire-plane view (U/V/W)")
    parser.add_argument("--outdir",     required=True,           help="Output directory for shard files")
    parser.add_argument("--cache_dir",  required=True,           help="Dataset index cache directory")
    parser.add_argument("--shard_size", type=int, default=1000,  help="Images per shard (default: 1000)")
    parser.add_argument("--seed",       type=int, default=42,    help="Random seed for the shuffle (default: 42)")
    parser.add_argument("--n_shards",   type=int, default=None,
                        help="Only shard a random subset of n_shards*shard_size samples "
                             "(diagnostics mode); default: shard the full dataset")
    parser.add_argument("--with_pixel_truth", action="store_true",
                        help="Also store per-pixel class labels (pixel_labels)")
    parser.add_argument("--with_extra_truth", action="store_true",
                        help="Also store pixel_energyfrac/pixel_trackid/pixel_truth_q "
                             "(implies --with_pixel_truth)")
    args = parser.parse_args()

    create_shards(
        datadir=args.datadir,
        apa=args.apa,
        view=args.view,
        outdir=args.outdir,
        cache_dir=args.cache_dir,
        shard_size=args.shard_size,
        seed=args.seed,
        n_shards=args.n_shards,
        with_pixel_truth=args.with_pixel_truth,
        with_extra_truth=args.with_extra_truth,
    )


if __name__ == "__main__":
    main()
