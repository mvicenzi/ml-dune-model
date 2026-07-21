"""
Create pre-sharded HDF5 files from an existing APASparseDataset.

Each shard file uses a flat layout (not one group per event) so the entire
shard can be loaded with exactly 3 HDF5 reads regardless of shard size:

    /coords   (N_total, 2) int32   -- wire/tick coords, all samples concatenated
    /features (N_total, 1) float32 -- ADC values
    /offsets  (B+1,)      int64   -- cumulative pixel counts; offsets[i]:offsets[i+1]
                                     gives the slice for sample i (same role as groups)

A metadata.json is written alongside the shards with n_samples, shard_size, etc.

Usage:
    python -m loader.create_shards \\
        --datadir /path/to/dataset \\
        --apa 0 --view W \\
        --outdir /path/to/shards \\
        --cache_dir /path/to/cache \\
        --shard_size 1000

The script skips existing shard files, so it can be safely re-run to resume
an interrupted run.
"""

import argparse
import json
from pathlib import Path

import h5py
import torch

from loader.apa_sparse_dataset import APASparseDataset


def create_shards(
    datadir: str,
    apa: int,
    view: str,
    outdir: str,
    cache_dir: str,
    shard_size: int = 1000,
    seed: int = 42,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = APASparseDataset(
        datadir=datadir,
        apa=apa,
        view=view,
        use_cache=True,
        cache_dir=cache_dir,
    )
    n_samples = len(dataset)
    print(f"Total samples : {n_samples}")

    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(n_samples, generator=rng).tolist()

    shards = [indices[i : i + shard_size] for i in range(0, n_samples, shard_size)]
    n_shards = len(shards)
    print(f"Shards        : {n_shards} x ~{shard_size} samples → {outdir}")

    for shard_idx, shard_indices in enumerate(shards):
        outpath = outdir / f"shard_{shard_idx:05d}.h5"
        if outpath.exists():
            print(f"  [{shard_idx+1}/{n_shards}] skip (exists): {outpath.name}")
            continue

        samples = [dataset[i] for i in shard_indices]

        all_coords = [s.coordinate_tensor for s in samples]
        all_feats  = [s.feature_tensor   for s in samples]
        counts  = torch.tensor([c.shape[0] for c in all_coords], dtype=torch.int64)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, dim=0)])
        coords_cat = torch.cat(all_coords, dim=0)
        feats_cat  = torch.cat(all_feats,  dim=0)

        with h5py.File(outpath, "w") as f:
            f.create_dataset("coords",   data=coords_cat.numpy(), compression="gzip", compression_opts=6)
            f.create_dataset("features", data=feats_cat.numpy(),  compression="gzip", compression_opts=6)
            f.create_dataset("offsets",  data=offsets.numpy())

        if (shard_idx + 1) % 10 == 0 or shard_idx + 1 == n_shards:
            print(f"  [{shard_idx+1}/{n_shards}] wrote {outpath.name}")

    meta = {
        "n_samples":  n_samples,
        "shard_size": shard_size,
        "n_shards":   n_shards,
        "apa":        apa,
        "view":       view,
        "seed":       seed,
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Wrote {n_shards} shards + metadata.json to {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-shard an APASparseDataset into HDF5 shard files.")
    parser.add_argument("--datadir",    required=True,        help="Root directory of source HDF5 files")
    parser.add_argument("--apa",        type=int, required=True, help="APA number")
    parser.add_argument("--view",       default="W",          help="Wire-plane view (U/V/W)")
    parser.add_argument("--outdir",     required=True,        help="Output directory for shard files")
    parser.add_argument("--shard_size", type=int, default=1000, help="Samples per shard (default: 1000)")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed for initial shuffle")
    parser.add_argument("--cache_dir",  required=True,        help="Dataset index cache directory")
    args = parser.parse_args()

    create_shards(
        datadir=args.datadir,
        apa=args.apa,
        view=args.view,
        outdir=args.outdir,
        cache_dir=args.cache_dir,
        shard_size=args.shard_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
