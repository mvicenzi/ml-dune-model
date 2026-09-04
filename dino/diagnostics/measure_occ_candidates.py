"""
measure_occ_candidates.py
─────────────────────────
How many occupancy candidates does region masking actually produce on real events?

The mae objective scores occupancy over a candidate set enumerated from the wiped cells:
every pixel of every wiped cell, mapped to the stride the occupancy head reads at. That
set is what has to be injected into the decoder, so its size is what decides whether a
batch fits in memory -- and it depends on the event, not on anything we can read off the
config. `occ_max_neg` and `occ_neg_per_pos` are the caps that bound it, and this is the
measurement they should be set from.

What it prints, over a sample of real events at the configured cell size:

  positives  candidates whose cell held charge -- the pixels the model must find.
             Never capped, so this is the floor on what every image costs.
  negatives  candidates whose cell was empty. These are the memory driver and the only
             thing the caps touch.
  ratio      negatives per positive, uncapped. Sets what `occ_neg_per_pos` would mean.

Read the per-image DISTRIBUTION, not the mean: the caps exist for the tail, and it is the
busiest event in a batch that decides whether the run fits. A sensible `occ_max_neg` sits
somewhere around the upper percentiles -- high enough that a typical image is untouched,
low enough that the worst one is bounded.

The masking is random, so the numbers move a little run to run; take the percentiles as
approximate. Nothing here is written to disk and nothing trains -- it reads events,
masks them, counts, and prints.

Run (needs the data, not a GPU):
    python -u dino/diagnostics/measure_occ_candidates.py --n_events=500
    python -u dino/diagnostics/measure_occ_candidates.py --cell_w=105 --cell_h=100
"""

from __future__ import annotations

import os
import sys

import fire
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dino.config import DINOConfig
from dino.masking import SparseRegionMasker
from dino.transforms import FeatureLogTransform
from loader.apa_sparse_meta_dataset import APASparseMetaDataset
from loader.collate import voxels_meta_collate_fn
from loader.splits import Subset


def percentiles(values: list, qs=(0, 50, 90, 99, 100)) -> str:
    """p0 is included deliberately: the caps are set from the upper tail, but an image
    that yields NO wiped cell yields no charge target either, and only the bottom of the
    distribution shows whether that happens."""
    t = torch.tensor(values, dtype=torch.float64).sort().values
    out = []
    for q in qs:
        i = min(int(round(q / 100.0 * (t.numel() - 1))), t.numel() - 1)
        label = "max" if q == 100 else ("min" if q == 0 else f"p{q}")
        out.append(f"{label}={t[i].item():,.0f}")
    return "  ".join(out)


def main(
    n_events: int = 500,
    batch_size: int = 20,
    cell_w: int = None,
    cell_h: int = None,
    flavor: str = "wipe",
    wipe_max: float = None,
    cand_stride: int = 2,
    num_workers: int = 4,
    device: str = "cpu",
):
    """
    Args:
        n_events:    how many events to measure over.
        batch_size:  events per masker call; does not change the per-image counts.
        cell_w/h:    cell size to measure. Defaults to the config's.
        flavor:      "wipe" or "randomize".
        wipe_max:    ceiling on the masked voxel fraction. Defaults to the config's.
        cand_stride: the stride the occupancy head reads at.
        device:      "cpu" is enough; the masker is plain torch.
    """
    cfg = DINOConfig()
    cell_w = cell_w or cfg.mask_region_cell_w
    cell_h = cell_h or cfg.mask_region_cell_h
    wipe_max = cfg.mask_region_wipe_max if wipe_max is None else wipe_max

    print("Measuring occupancy candidates")
    print(f"  canvas       = {cfg.image_w} x {cfg.image_h}")
    print(f"  cell         = {cell_w} x {cell_h}  "
          f"({cfg.image_w // cell_w} x {cfg.image_h // cell_h} grid)")
    print(f"  flavor       = {flavor}   wipe_max = {wipe_max}")
    print(f"  cand_stride  = {cand_stride}  "
          f"({(cell_w // cand_stride) * (cell_h // cand_stride):,} candidates per wiped cell)")
    print(f"  events       = {n_events}")
    print()

    dataset = APASparseMetaDataset(
        datadir=cfg.datadir, apa=cfg.apa, view=cfg.view,
        use_cache=True, cache_dir=cfg.cache_dir,
    )
    loader = DataLoader(
        Subset(dataset, list(range(min(n_events, len(dataset))))),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=voxels_meta_collate_fn,
    )
    normalizer = FeatureLogTransform(cfg.feat_min_val, cfg.feat_max_val)
    masker = SparseRegionMasker(
        image_w=cfg.image_w, image_h=cfg.image_h, cell_w=cell_w, cell_h=cell_h,
        flavor=flavor, wipe_max=wipe_max, r1=cfg.mask_region_r1, r2=cfg.mask_region_r2,
        return_masked_feats=True, build_candidates=True, cand_stride=cand_stride,
    )

    n_active, n_masked, n_pos, n_neg, n_cells = [], [], [], [], []
    seen = 0
    with torch.no_grad():
        for xs in loader:
            xs = normalizer(xs.to(device))
            _, masked_coords, _, cand, occ = masker(xs)
            for b in range(len(xs.offsets) - 1):
                total = int(xs.offsets[b + 1]) - int(xs.offsets[b])
                if total == 0:
                    continue
                pos = int(occ[b].sum())
                n_active.append(total)
                n_masked.append(masked_coords[b].shape[0])
                n_pos.append(pos)
                n_neg.append(int(occ[b].numel()) - pos)
                per_cell = (cell_w // cand_stride) * (cell_h // cand_stride)
                n_cells.append(int(cand[b].shape[0]) / per_cell)
                seen += 1
            if seen >= n_events:
                break

    if not seen:
        print("no events measured")
        return 1

    print(f"Measured {seen} events\n")
    print(f"  active voxels     {percentiles(n_active)}")
    print(f"  masked voxels     {percentiles(n_masked)}")
    print(f"  wiped cells       {percentiles(n_cells)}")
    print()
    print(f"  POSITIVES         {percentiles(n_pos)}")
    print(f"  NEGATIVES         {percentiles(n_neg)}")
    print()
    ratios = [n / max(p, 1) for n, p in zip(n_neg, n_pos)]
    print(f"  negatives/positive {percentiles(ratios)}")
    n_empty = sum(1 for c in n_cells if c == 0)
    if n_empty:
        print(f"\n  WARNING: {n_empty} of {seen} events produced no wiped cell at all, so"
              f"\n  they contribute no charge target. Lower mask_region_cell_* or raise"
              f"\n  mask_region_wipe_max if this is more than a rounding tail.")
    masked_frac = [m / a for m, a in zip(n_masked, n_active)]
    print(f"  masked fraction    {'  '.join(f'p{q}={torch.tensor(masked_frac).sort().values[min(int(round(q / 100.0 * (len(masked_frac) - 1))), len(masked_frac) - 1)].item():.3f}' for q in (50, 90, 100))}")
    print()
    print("Set occ_max_neg from the NEGATIVES distribution: a value near the upper")
    print("percentiles leaves typical events untouched while bounding the worst one.")
    print("Set occ_neg_per_pos instead (or as well) to fix the class balance directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(fire.Fire(main))
