# Build a Sparse Masked Auto-Encoder (MAE) Training Workflow

## Read first — existing code you will build on

Before writing any code, read these files in full:

- `models/minkunet_attention.py` — `MinkUNetSparseAttentionCore` is the backbone
  (Voxels → Voxels, 64 output feature channels)
- `models/blocks.py` — `ConvBlock2D`, `ResidualSparseBlock2D`, `BottleneckSparseAttention2D`
- `loader/apa_sparse_dataset.py` — `APASparseDataset`; `__getitem__` returns a
  **single-item** `Voxels` with `offsets = [0, N]`
- `loader/collate.py` — `voxels_collate_fn` already exists (batches plain Voxels)
- `training.py` — reference script (fire.Fire CLI, MetricsMonitor, AdamW + StepLR,
  checkpoint loop)
- `metrics_monitor.py` — the `MetricsMonitor` class

## Data layout (confirmed on disk)

Root: `/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27`

```
{run}/{subrun}/{event}/out_{basename}/
    {basename}_pixeldata-anode{N}.h5   ← sparse wire data
    {basename}_metadata.h5             ← per-event truth labels
```

**Pixeldata HDF5** — group `/{g}/frame_rebinned_reco`:
- `coords`   (N, 2) int32  — `[:, 0]` = channel, `[:, 1]` = tick
- `features` (N,)   float32 — charge amplitude

**Metadata HDF5** — group `/{g}/metadata`:
- numpy structured array, shape (1,)
- fields: `nu_pdg` (int32), `nu_ccnc` (int32), `nu_intType` (int32),
  `nu_energy` (float64), `nu_vertex_{x,y,z}` (float64)
- Classification targets (3 classes, derived from `nu_pdg` + `nu_ccnc`):

  | index | label  | condition                      |
  |-------|--------|--------------------------------|
  | 0     | numuCC | `nu_pdg == 14` and `nu_ccnc == 0` |
  | 1     | nueCC  | `nu_pdg == 12` and `nu_ccnc == 0` |
  | 2     | NC     | `nu_ccnc == 1`                 |

  Anything else (e.g. ν_τ CC) → `label = -1` (skip during SFT training).

---

## Task 1 — Sparse block masking: `models/sparse_masking.py`

```python
def sparse_block_mask(
    voxels: Voxels,
    masking_frac: float,   # fraction of active voxels used as seeds, range [0, 1]
    win_ch: int,           # half-window size in the channel direction
    win_tick: int,         # half-window size in the tick direction
) -> tuple[Voxels, torch.BoolTensor]:
```

**Algorithm** — process each batch item independently:

1. For batch item `i` (rows `offsets[i]:offsets[i+1]` in `feature_tensor`),
   randomly select `ceil(masking_frac × N_i)` voxels as seeds via `torch.randperm`.
2. For each seed at `(ch, tick)`, mark all active voxels in the rectangular window
   `[ch − win_ch, ch + win_ch] × [tick − win_tick, tick + win_tick]` as masked.
   Use vectorised broadcasting over all `N_i` voxels per seed for efficiency.
3. Build `masked_voxels` by cloning the input Voxels and zeroing `feature_tensor`
   at all masked positions. **Keep the coordinate structure identical to the
   input** — do not drop masked voxels from the COO set.
4. Build `mask_bool`: `BoolTensor` of shape `(N_total,)` aligned with
   `voxels.feature_tensor`, `True` where masked.

Return `(masked_voxels, mask_bool)`.

The reconstruction target is `voxels.feature_tensor[mask_bool]` directly —
no `from_dense` round-trip needed since data never leaves COO format.

---

## Task 2 — Dataset with labels: `loader/apa_sparse_meta_dataset.py`

Subclass `APASparseDataset` as `APASparseMetaDataset`:

- In `__getitem__`, after loading the Voxels, also load the metadata file.
  Given pixeldata path `{dir}/{basename}_pixeldata-anode{N}.h5`, derive:
  `metadata_path = path.parent / path.name.replace(f"_pixeldata-anode{self.apa}.h5", "_metadata.h5")`
  Read `f[group]["metadata"][0]` and extract both `nu_pdg` and `nu_ccnc`.
- Derive the class label from both `nu_pdg` and `nu_ccnc`:
  - `nu_pdg == 14` and `nu_ccnc == 0` → class 0 (numuCC)
  - `nu_pdg == 12` and `nu_ccnc == 0` → class 1 (nueCC)
  - `nu_ccnc == 1`                    → class 2 (NC)
  - anything else                     → `label = -1` (skip)
  If the metadata file is missing, warn once and return `label = -1`.
- Return `(voxels: Voxels, label: int)`.

---

## Task 3 — Model: `models/mae_model.py`

```python
class SparseMAEModel(nn.Module):
    """
    backbone       : MinkUNetSparseAttentionCore  (Voxels[1ch]  → Voxels[64ch])
    charge_head    : SparseConv2d(64, 1, kernel_size=1)          (SSL head)
    nu_flavor_head : sparse global avg-pool → nn.Linear(64, n_classes)  (SFT head)
    """
```

Helper (move here, not in tests):
```python
def sparse_global_avg_pool(vox: Voxels) -> Tensor:
    # scatter-add over batch index, divide by per-batch count
```

Methods:
- `forward_ssl(masked_voxels: Voxels) -> Voxels` — backbone + charge_head;
  output Voxels has 1 feature channel.
- `forward_sft(voxels: Voxels) -> Tensor` — backbone (no grad) + global pool +
  linear; returns `[B, n_classes]` logits. Backbone is frozen by the caller
  before this is called.
- `freeze_backbone()` / `unfreeze_backbone()` — toggle `backbone.requires_grad_`.

---

## Task 4 — Batched collation: `loader/collate.py`

`loader/collate.py` already exists and contains `voxels_collate_fn` which
batches a list of single-item `Voxels` into one batched `Voxels` (concatenates
COO tensors, recomputes offsets).  **Do not rewrite it.**

Use `voxels_collate_fn` as-is for the SSL DataLoader.

For the SFT DataLoader, `APASparseMetaDataset.__getitem__` returns
`(Voxels, int)` tuples, which `voxels_collate_fn` cannot handle.
Add one new function to `loader/collate.py`:

```python
def voxels_label_collate_fn(batch):
    """Collate a list of (Voxels, int) tuples."""
    voxels_list, labels = zip(*batch)
    return voxels_collate_fn(list(voxels_list)), torch.tensor(labels, dtype=torch.long)
```

---

## Task 5 — Training script: `scripts/train_mae.py`

Follow the structure of `training.py` (fire.Fire, MetricsMonitor, checkpoint loop).

### CLI parameters

```python
def main(
    data_root       = "/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27",
    apa             = 3,
    view            = "W",
    batch_size      = 16,
    epochs          = 20,
    lr              = 1e-3,
    scheduler_step  = 10,
    gamma           = 0.7,
    n_ssl           = 4,       # SSL batches per interleave cycle
    n_sft           = 1,       # SFT batches per interleave cycle
    masking_frac    = 0.3,
    win_ch          = 10,
    win_tick        = 20,
    n_classes       = 3,
    device          = "cuda",
    metrics_dir     = "./metrics",
    checkpoints_dir = "./checkpoints",
    save_every      = 5,
):
```

### DataLoaders

```python
ssl_loader = DataLoader(APASparseDataset(...),     batch_size=batch_size,
                        shuffle=True, collate_fn=voxels_collate_fn)
sft_loader = DataLoader(APASparseMetaDataset(...), batch_size=batch_size,
                        shuffle=True, collate_fn=voxels_label_collate_fn)
```

Both iterate over the same root; keep two independent cycling iterators
(`itertools.cycle`).

### Optimizers (two separate, no cross-contamination)

```python
opt_ssl = AdamW(list(model.backbone.parameters()) +
                list(model.charge_head.parameters()), lr=lr)
opt_sft = AdamW(model.nu_flavor_head.parameters(), lr=lr)
```

### Interleaved loop (per epoch, per global step)

```
for each global step in epoch:

    # ── SSL phase ──────────────────────────────────────────────
    for _ in range(n_ssl):
        vox = next(ssl_iter)                          # Voxels, shape (B, N, 1)
        masked, mask_bool = sparse_block_mask(vox, masking_frac, win_ch, win_tick)
        pred = model.forward_ssl(masked)              # Voxels, shape (B, N, 1)
        ssl_loss = F.l1_loss(pred.feature_tensor[mask_bool],
                             vox.feature_tensor[mask_bool])
        opt_ssl.zero_grad(); ssl_loss.backward(); opt_ssl.step()

    # ── SFT phase ──────────────────────────────────────────────
    model.freeze_backbone()
    for _ in range(n_sft):
        vox, labels = next(sft_iter)                  # Voxels + LongTensor[B]
        valid = labels >= 0
        if not valid.any():
            continue
        logits = model.forward_sft(vox)               # [B, n_classes]
        sft_loss = F.cross_entropy(logits[valid], labels[valid])
        opt_sft.zero_grad(); sft_loss.backward(); opt_sft.step()
    model.unfreeze_backbone()
```

Note: `forward_sft` calls backbone inside `torch.no_grad()` internally; the
`freeze_backbone()` call here prevents opt_ssl from accumulating stale grads.

### Monitoring — print and save per epoch

1. **SSL**: running-mean L1 loss over all SSL batches in the epoch.
2. **SFT**: running-mean cross-entropy loss over all SFT batches.
3. **SFT confusion matrix** (n_classes × n_classes, raw counts) — printed as a
   formatted table at epoch end.
4. **SFT per-class efficiency and purity** table:

   | class  | condition               | efficiency (recall) | purity (precision) |
   |--------|-------------------------|---------------------|--------------------|
   | numuCC | nu_pdg=14, nu_ccnc=0    | TP/(TP+FN)          | TP/(TP+FP)         |
   | nueCC  | nu_pdg=12, nu_ccnc=0    | …                   | …                  |
   | NC     | nu_ccnc=1               | …                   | …                  |

   Derive both from the confusion matrix; handle zero-division gracefully.

Save checkpoints as `{checkpoints_dir}/mae_epoch{E}.pt` containing
`{"epoch": E, "model": model.state_dict()}`.
