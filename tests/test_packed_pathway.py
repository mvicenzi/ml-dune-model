"""
End-to-end verification of the packed/sharded data pathways on a small slice
of the rich-truth production. Standalone script (no pytest), same style as
the other tests/ scripts.

Checks:
  1. Pixel-truth alignment: APASparseMetaDataset pixel arrays are aligned to
     the voxels, correctly typed, and match a direct h5py re-read.
  2. Pack <-> raw equivalence: every packed sample is torch.equal to the raw
     dataset sample at the same index (pack preserves index order).
  3. Shard <-> pack equivalence: the shard stream (shuffle=False) replays the
     seed-42 creation permutation of the same events.
  4. Clone-fix regression: two epochs of in-place FeatureLogTransform must
     not mutate the packed dataset's shared RAM buffer.
  5. Sharded-reader truth knobs: tiers are opt-in and fail loudly when the
     shards lack a requested tier.

Usage (any small directory of the production works; one event dir = seconds):
    python -m tests.test_packed_pathway \\
        --datadir /gpfs01/lbne/users/bnayak/cffm-data/prod-jay-100k-truth-2026-06-11/nominal/out_monte-carlo-016908-000001_361085_52_1_20260610T051553Z \\
        --workdir /tmp/$USER/packed_pathway_test
"""

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from loader.apa_sparse_dataset import APASparseDataset
from loader.apa_sparse_meta_dataset import APASparseMetaDataset
from loader.apa_packed_dataset import APAPackedDataset
from loader.apa_sparse_sharded_dataset import APASparseShardedDataset
from loader.pack_dataset import pack_dataset
from loader.create_shards import create_shards
from loader.collate import voxels_meta_collate_fn
from dino.transforms import FeatureLogTransform

APA, VIEW, SEED = 0, "W", 42
_failures = []


def check(name: str, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        _failures.append(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--datadir", required=True,
                    help="Small slice of the production (e.g. one event dir)")
    ap.add_argument("--workdir", default="./packed_pathway_test",
                    help="Scratch dir for pack/shards/cache (wiped)")
    args = ap.parse_args()

    work = Path(args.workdir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cache = str(work / "cache")

    print(f"datadir : {args.datadir}")
    print(f"workdir : {work}\n")

    # ---- datasets and artifacts built once, shared by the checks ----------
    meta_ds = APASparseMetaDataset(args.datadir, apa=APA, view=VIEW,
                                   use_cache=False, cache_dir=cache,
                                   return_pixel_truth=True,
                                   return_extra_truth=True)
    raw_ds = APASparseDataset(args.datadir, apa=APA, view=VIEW,
                              use_cache=False, cache_dir=cache)
    n = len(meta_ds)
    print(f"events  : {n}\n")

    pack_path = work / "pack.npz"
    pack_dataset(datadir=args.datadir, out_path=str(pack_path), apa=APA,
                 view=VIEW, cache_dir=cache, num_workers=0, log_every=10**9)
    shards_dir = work / "shards"
    create_shards(datadir=args.datadir, apa=APA, view=VIEW,
                  outdir=str(shards_dir), cache_dir=cache,
                  shard_size=max(2, n // 2), with_extra_truth=True,
                  num_workers=0)
    print()

    # ---- 1. pixel-truth alignment -----------------------------------------
    def truth_alignment():
        for i in range(n):
            vox, meta = meta_ds[i]
            n_vox = vox.coordinate_tensor.shape[0]
            pl = meta["pixel_labels"]
            assert pl.shape == (n_vox,) and pl.dtype == np.int8, (pl.shape, pl.dtype)
            assert set(np.unique(pl)) <= set(range(7)), np.unique(pl)
            for k, dt in (("pixel_energyfrac", np.float32),
                          ("pixel_trackid", np.int32),
                          ("pixel_truth_q", np.float32)):
                assert meta[k].shape == (n_vox,) and meta[k].dtype == dt, k
        # direct h5py cross-check on event 0 (independent of the loader code)
        s = meta_ds.samples[0]
        with h5py.File(s.path, "r") as f:
            g = f[s.group]
            reco = g[meta_ds.frame_name]["coords"][()]
            lc = g["frame_label_1st"]["coords"][()]
            lv = g["frame_label_1st"]["features"][()]
        m = (reco[:, 0] >= meta_ds.ch_start) & (reco[:, 0] < meta_ds.ch_end)
        lk = {(int(c[0]), int(c[1])): int(v) for c, v in zip(lc, lv)
              if meta_ds.ch_start <= c[0] < meta_ds.ch_end}
        expected = np.array([lk.get((int(c[0]), int(c[1])), 0)
                             for c in reco[m]], dtype=np.int8)
        _, meta0 = meta_ds[0]
        assert np.array_equal(expected, meta0["pixel_labels"])

    check("pixel-truth alignment + h5py cross-check", truth_alignment)

    # ---- 2. pack <-> raw ---------------------------------------------------
    pack = APAPackedDataset(pack_path, return_extra_truth=True)

    def pack_vs_raw():
        assert len(pack) == n
        for i in range(n):
            pv, pm = pack[i]
            rv = raw_ds[i]
            _, rm = meta_ds[i]
            assert torch.equal(pv.coordinate_tensor, rv.coordinate_tensor), i
            assert torch.equal(pv.feature_tensor, rv.feature_tensor), i
            assert pm["label"] == rm["label"] and pm["event_key"] == rm["event_key"], i
            assert np.array_equal(pm["pixel_labels"], rm["pixel_labels"]), i
            assert np.array_equal(pm["pixel_trackid"], rm["pixel_trackid"]), i
        # signed track ids must survive the round trip
        assert any(int((pack[i][1]["pixel_trackid"] < 0).sum()) > 0 for i in range(n))

    check("pack <-> raw exact equivalence (incl. signed trackids)", pack_vs_raw)

    # ---- 3. shard <-> pack -------------------------------------------------
    def shard_vs_pack():
        rng = torch.Generator()
        rng.manual_seed(SEED)
        perm = torch.randperm(n, generator=rng).tolist()
        shard_size = max(2, n // 2)
        n_kept = (n // shard_size) * shard_size  # trailing partial shard exists too
        reader = APASparseShardedDataset(str(shards_dir), batch_size=1,
                                         shuffle=False, return_extra_truth=True)
        for j, (vox, meta) in enumerate(reader):
            pv, pm = pack[perm[j]]
            assert torch.equal(vox.coordinate_tensor, pv.coordinate_tensor), j
            assert torch.equal(vox.feature_tensor, pv.feature_tensor), j
            assert meta["event_key"][0] == pm["event_key"], j
            assert np.array_equal(meta["pixel_labels"][0], pm["pixel_labels"]), j

    check("shard stream replays creation permutation of pack content", shard_vs_pack)

    # ---- 4. clone-fix regression -------------------------------------------
    def clone_fix():
        tf = FeatureLogTransform(3.75, 83861.2)
        before = pack.feats.clone()
        dl = DataLoader(pack, batch_size=max(2, n // 2), shuffle=False,
                        collate_fn=voxels_meta_collate_fn)
        for _epoch in range(2):
            for xs, _ in dl:
                tf(xs)
        assert torch.equal(pack.feats, before), "shared buffer was mutated!"

    check("clone fix: 2 normalization passes leave the pack buffer intact", clone_fix)

    # ---- 5. sharded-reader truth knobs --------------------------------------
    def truth_knobs():
        r0 = APASparseShardedDataset(str(shards_dir), batch_size=2, shuffle=False)
        _, m0 = next(iter(r0))
        assert "label" in m0 and "pixel_labels" not in m0  # default: event truth only
        r1 = APASparseShardedDataset(str(shards_dir), batch_size=2, shuffle=False,
                                     return_pixel_truth=True)
        _, m1 = next(iter(r1))
        assert "pixel_labels" in m1 and "pixel_truth_q" not in m1
        # requesting a tier the shards lack must raise
        lean_dir = work / "shards_lean"
        create_shards(datadir=args.datadir, apa=APA, view=VIEW,
                      outdir=str(lean_dir), cache_dir=cache,
                      shard_size=max(2, n // 2), num_workers=0)
        try:
            APASparseShardedDataset(str(lean_dir), batch_size=2,
                                    return_pixel_truth=True)
            raise AssertionError("missing tier did not raise")
        except ValueError:
            pass

    check("sharded-reader truth knobs (opt-in, loud on missing tier)", truth_knobs)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
