"""
Create truth-annotated shard files for DINO diagnostics.

Unlike the training shards (create_shards.py), each shard stores full
event-level truth and per-pixel PID labels alongside the sparse pixel data.
The shard set is intentionally small (5-10 files, ~1000 images each) and is
designed to be reused across all checkpoint extractions for a given dataset,
so KNN/KNN-PID comparisons across epochs are on identical images.

Pixel data comes from APASparseMetaDataset.__getitem__ verbatim, so any
log-transform or normalization applied by the dataset is preserved correctly.

HDF5 layout per shard file:
    /coords       (N_pix, 2)  int32   -- channel/tick coords (view-rebased, same as training)
    /features     (N_pix, 1)  float32 -- pixel ADC values
    /offsets      (N_img+1,)  int64   -- CSR pixel offsets (shared by /pid_labels)
    /labels       (N_img,)    int32   -- class index: 0=numuCC 1=nueCC 2=NC -1=unknown
    /nu_pdg       (N_img,)    int32
    /nu_ccnc      (N_img,)    int32
    /nu_intType   (N_img,)    int32
    /nu_energy    (N_img,)    float32
    /vertex_xyz   (N_img, 3)  float32
    /event_key    (N_img,)    bytes    -- UTF-8 encoded "{file}:{group}" traceability string
    /pid_labels   (N_pix,)    int32   -- per-pixel PDG code (0=no truth); same CSR as /coords

A metadata.json alongside records apa, view, n_samples, shard_size, and seed.
APASparseShardedTruthDataset reads this to assert dataset/checkpoint compatibility.

Usage:
    python -m loader.create_truth_shards \\
        --datadir /path/to/dataset \\
        --apa 0 --view W \\
        --outdir /path/to/truth_shards \\
        --cache_dir /path/to/cache \\
        [--n_shards 10] [--shard_size 1000] [--seed 42]
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from loader.apa_sparse_meta_dataset import APASparseMetaDataset


def create_truth_shards(
    datadir: str,
    apa: int,
    view: str,
    outdir: str,
    cache_dir: str,
    n_shards: int = 10,
    shard_size: int = 1000,
    seed: int = 42,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = APASparseMetaDataset(
        datadir=datadir,
        apa=apa,
        view=view,
        use_cache=True,
        cache_dir=cache_dir,
        return_full_metadata=True,
        return_pixel_truth=True,
    )
    n_total = n_shards * shard_size
    print(f"Dataset size : {len(dataset)}")
    print(f"Sampling     : {n_total} images ({n_shards} shards x {shard_size})")

    rng = torch.Generator()
    rng.manual_seed(seed)
    all_indices = torch.randperm(len(dataset), generator=rng)[:n_total].tolist()

    # Write metadata.json before the loop so a partial run is still usable.
    meta_out = {
        "n_samples":  n_total,
        "n_shards":   n_shards,
        "shard_size": shard_size,
        "apa":        apa,
        "view":       view,
        "seed":       seed,
    }
    with open(outdir / "metadata.json", "w") as fp:
        json.dump(meta_out, fp, indent=2)

    for shard_idx in range(n_shards):
        outpath = outdir / f"shard_{shard_idx:05d}.h5"
        if outpath.exists():
            print(f"  [{shard_idx + 1}/{n_shards}] skip (exists): {outpath.name}")
            continue

        shard_indices = all_indices[shard_idx * shard_size : (shard_idx + 1) * shard_size]

        coords_list, feats_list, pid_list = [], [], []
        offsets = [0]
        labels, nu_pdg, nu_ccnc, nu_intType, nu_energy = [], [], [], [], []
        vertex_xyz, event_keys = [], []

        for i, idx in enumerate(shard_indices):
            voxels, meta = dataset[idx]

            c   = voxels.coordinate_tensor.numpy()   # (N, 2) int32
            f   = voxels.feature_tensor.numpy()      # (N, 1) float32
            pid = meta["pid_labels"]                 # (N,)   int32 np.ndarray

            assert len(pid) == len(c), (
                f"pid_labels length {len(pid)} != coords length {len(c)} for dataset index {idx}"
            )

            coords_list.append(c)
            feats_list.append(f)
            pid_list.append(pid)
            offsets.append(offsets[-1] + len(c))

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
        pid_cat     = np.concatenate(pid_list,    axis=0).astype(np.int32)
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
            hf.create_dataset("pid_labels", data=pid_cat,                       compression="gzip", compression_opts=6)
            hf.create_dataset("event_key",  data=np.array(event_keys, dtype=object),
                               dtype=h5py.special_dtype(vlen=bytes))

        print(f"  [{shard_idx + 1}/{n_shards}] wrote {outpath.name}  "
              f"({coords_cat.shape[0]} pixels, {len(offsets) - 1} images)")

    meta_out = {
        "n_samples":  n_total,
        "n_shards":   n_shards,
        "shard_size": shard_size,
        "apa":        apa,
        "view":       view,
        "seed":       seed,
    }
    with open(outdir / "metadata.json", "w") as fp:
        json.dump(meta_out, fp, indent=2)

    print(f"\nDone. Wrote {n_shards} shards + metadata.json to {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create truth-annotated shard files for DINO diagnostics."
    )
    parser.add_argument("--datadir",    required=True,           help="Root directory of source HDF5 files")
    parser.add_argument("--apa",        type=int, required=True, help="APA number")
    parser.add_argument("--view",       default="W",             help="Wire-plane view (U/V/W)")
    parser.add_argument("--outdir",     required=True,           help="Output directory for truth shard files")
    parser.add_argument("--cache_dir",  required=True,           help="Dataset index cache directory")
    parser.add_argument("--n_shards",   type=int, default=10,    help="Number of shard files (default: 10)")
    parser.add_argument("--shard_size", type=int, default=1000,  help="Images per shard (default: 1000)")
    parser.add_argument("--seed",       type=int, default=42,    help="Random seed for image sampling (default: 42)")
    args = parser.parse_args()

    create_truth_shards(
        datadir=args.datadir,
        apa=args.apa,
        view=args.view,
        outdir=args.outdir,
        cache_dir=args.cache_dir,
        n_shards=args.n_shards,
        shard_size=args.shard_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
