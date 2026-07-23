# Loader

Dataset classes and preprocessing scripts for the DUNE event data.

# Dataset classes

| File | Class / script | Status | Format | Truth | Dataset |
|---|---|---|---|---|---|
| `apa_sparse_meta_dataset.py` | `APASparseMetaDataset` | default | sparse | yes — event truth always; per-pixel opt-in (`return_pixel_truth`), rich per-pixel opt-in (`return_extra_truth`) | APA productions |
| `apa_sparse_sharded_dataset.py` | `APASparseShardedDataset` | grid-optimized | sparse | yes — auto-detects and returns whatever tiers the shards carry | make shards with `create_shards.py` |
| `apa_sparse_dataset.py` | `APASparseDataset` | legacy | sparse | no | APA productions |
| `apa_dataset.py` | `APAImageDataset` | legacy | dense | no | early APA productions (dense) |
| `dataset.py` | `DUNEImageDataset` | legacy | dense | yes — event labels only (from `.info` files) | DUNE CVN dataset (`.gz` image files) |

All non-legacy classes return `(Voxels, meta)`; training loops unpack and
ignore `meta`, diagnostics consume it. The grid-optimized containers exist
because reading ~10k small HDF5 files per epoch is slow on GPFS — per-sample
content is identical to `APASparseMetaDataset`.

Truth tiers: event-level truth is always returned (a cheap co-located
metadata read). The per-pixel tiers are opt-in — on `APASparseMetaDataset`
via the constructor flags `return_pixel_truth=True` (class labels) and
`return_extra_truth=True` (energyfrac / trackid / truth charge); for shards
they are baked in at creation time via `create_shards.py --with_pixel_truth`
/ `--with_extra_truth` (plus `--n_shards N` for a small fixed diagnostics
subset), and `APASparseShardedDataset` then auto-detects what is present.

## Productions

Locations are given for both clusters.

`prod-jay-100k-truth-2026-06-11` — **DEFAULT**
- SDCC: `/gpfs01/lbne/users/bnayak/cffm-data/prod-jay-100k-truth-2026-06-11`
- WCWC: `/srv/data/1/nitish/cffm-data/prod-jay-100k-truth-2026-06-11`
- Events: ~200k (`nominal/` νμ flux + `nueswap/` νe-swap flux)
- Truth: full (event metadata, per-pixel labels / energyfrac / trackid / charge, MC genealogy map)

`prod-jay-1M-truth-2026-06-11` — supported
- SDCC: ?
- WCWC: ?
- Events: ~1M (`nominal/` νμ flux + `nueswap/` νe-swap flux)
- Truth: event metadata only

`prod-jay-100k-truth-2026-02-27` - not supported (old truth labels)
- SDCC: `/gpfs01/lbne/users/fm/cffm-data/prod-jay-100k-truth-2026-02-27`
- WCWC: `/nfs/data/1/yuhw/cffm-data/prod-jay-100k-truth-2026-02-27`
- Events: ~100k (νμ flux)
- Truth: event metadata + per-pixel raw PDG

`prod-jay-1M-2026-02-27` — supported
- SDCC: `/gpfs01/lbne/users/fm/cffm-data/prod-jay-1M-2026-02-27`
- WCWC: `/nfs/data/1/yuhw/cffm-data/prod-jay-1M-truth-2026-02-27`
- Events: ~1M (νμ flux)
- Truth: event metadata only

Early APA production (dense, `frame_rebinned_reco`) — legacy
- SDCC: ?
- WCWC: `/nfs/data/1/mvicenzi/apa-test-data/gzip2`
- Used by `APAImageDataset`

DUNE CVN dataset (`.gz` image files) — legacy
- SDCC: ?
- WCWC: `/nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd`
- Used by `DUNEImageDataset`


