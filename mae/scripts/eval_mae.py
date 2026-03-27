"""
eval_mae.py — Evaluation script for trained Sparse MAE models.

Modules
-------
  1  (ssl) : Test SSL — per-event voxel sizes, mean L1 loss, per-event visualizations.
  2  (sft) : Test SFT — overall accuracy, macro-averaged accuracy, confusion matrix,
             per-class efficiency & purity.

To add a new module:
  1. Define a function  _module_<name>(model, dataset, device, cfg)
  2. Add an entry to MODULE_REGISTRY at the bottom of this file.

Usage
-----
  python scripts/eval_mae.py --checkpoint=checkpoints/mae_epoch10.pt
  python scripts/eval_mae.py --checkpoint=... --modules=1,2 --n_samples=500
  python scripts/eval_mae.py --checkpoint=... --modules=sft --n_samples=2000
"""

import sys
import math
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
import warp as wp
from torch.utils.data import DataLoader, Subset
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.mae_model import SparseMAEModel, voxels_to_device, log1p_voxels, expm1_voxels
from models.sparse_masking import sparse_block_mask
from loader.apa_sparse_meta_dataset import APASparseMetaDataset, CLASS_NAMES
from loader.collate import voxels_label_collate_fn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _least_occupied_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    best_idx, best_free = 0, 0
    for i in range(n):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free, best_idx = free, i
    dev = torch.device(f"cuda:{best_idx}")
    print(f"Selected {dev}  ({best_free / 2**30:.1f} GB free of {n} GPU{'s' if n > 1 else ''})")
    return dev


def _sparse_to_dense(coords: torch.Tensor, feats: torch.Tensor):
    """Sparse (N,2) int coords + (N,1) float feats → dense (H,W) numpy array, or None."""
    if len(coords) == 0:
        return None
    ch, tk = coords[:, 0], coords[:, 1]
    ch_min, tk_min = int(ch.min()), int(tk.min())
    H = int(ch.max()) - ch_min + 1
    W = int(tk.max()) - tk_min + 1
    grid = torch.zeros(H, W)
    grid[ch - ch_min, tk - tk_min] = feats[:, 0]
    return grid.numpy()


def _slice_single_event(vox: Voxels, evt: int) -> Voxels:
    """Extract one event from a batched Voxels object as a fresh single-event Voxels."""
    start = int(vox.offsets[evt].item())
    end   = int(vox.offsets[evt + 1].item())
    offsets = torch.tensor([0, end - start])
    coords  = vox.coordinate_tensor[start:end].cpu()
    feats   = vox.feature_tensor[start:end].cpu()
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


def _visualize_event(orig_vox, masked_vox, pred_vox, event_id: int, out_path: Path) -> None:
    """Save a 3-panel PNG (original | masked | reconstructed) for a single event."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [viz] matplotlib not available — skipping")
        return

    end = int(orig_vox.offsets[1].item())
    if end == 0:
        return

    coords       = orig_vox.coordinate_tensor[:end].cpu().int()
    orig_dense   = _sparse_to_dense(coords, orig_vox.feature_tensor[:end].cpu())
    masked_dense = _sparse_to_dense(coords, masked_vox.feature_tensor[:end].cpu())
    pred_dense   = _sparse_to_dense(coords, pred_vox.feature_tensor[:end].cpu())

    if orig_dense is None:
        return

    vmax = float(orig_dense.max()) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, img, title in zip(
        axes,
        [orig_dense, masked_dense, pred_dense],
        ["Original (unmasked)", "Masked input", "Reconstructed output"],
    ):
        im = ax.imshow(img, aspect="auto", origin="lower", vmin=0.0, vmax=vmax, cmap="cubehelix_r")
        ax.set_title(title)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Channel")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"SSL Reconstruction — Event {event_id}", fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [viz] Saved: {out_path}")


def _print_confusion(cm: torch.Tensor, class_names: list[str]) -> None:
    n = len(class_names)
    width = max(len(c) for c in class_names) + 2
    header = " " * (width + 2) + "  ".join(f"{c:>{width}}" for c in class_names)
    print(f"\n  Confusion matrix  (rows = true, cols = predicted)")
    print(f"  {header}")
    for i, name in enumerate(class_names):
        row = f"  {name:>{width}}  " + "  ".join(f"{cm[i, j].item():>{width}d}" for j in range(n))
        print(row)


def _print_class_metrics(cm: torch.Tensor, class_names: list[str]) -> None:
    n = len(class_names)
    print(f"\n  Per-class metrics:")
    print(f"  {'class':>10}  {'N_true':>8}  {'efficiency':>12}  {'purity':>10}")
    effs = []
    for i in range(n):
        tp     = cm[i, i].item()
        n_true = cm[i, :].sum().item()
        n_pred = cm[:, i].sum().item()
        eff = tp / n_true if n_true > 0 else float("nan")
        pur = tp / n_pred if n_pred > 0 else float("nan")
        print(f"  {class_names[i]:>10}  {n_true:>8.0f}  {eff:>12.4f}  {pur:>10.4f}")
        if not math.isnan(eff):
            effs.append(eff)
    macro = sum(effs) / len(effs) if effs else float("nan")
    print(f"\n  Macro-averaged efficiency (equal weight per class): {macro:.4f}")


# ---------------------------------------------------------------------------
# Module 1: SSL evaluation
# ---------------------------------------------------------------------------

def _module_ssl(model, dataset, device, cfg: dict) -> None:
    n_samples    = cfg["n_samples"]
    n_viz        = cfg["n_viz"]
    batch_size   = cfg["batch_size"]
    masking_frac = cfg["masking_frac"]
    win_ch       = cfg["win_ch"]
    win_tick     = cfg["win_tick"]
    viz_dir      = Path(cfg["viz_dir"])

    print(f"\n{'='*60}")
    print(f"Module 1 — SSL evaluation")
    print(f"  samples={n_samples}  n_viz={n_viz}  masking_frac={masking_frac}"
          f"  win_ch={win_ch}  win_tick={win_tick}")

    n_use = min(n_samples, len(dataset))
    # subset = Subset(dataset, torch.randperm(len(dataset))[:n_use].tolist())
    subset = Subset(dataset, list(range(n_use)))  # deterministic subset for reproducibility
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=voxels_label_collate_fn, num_workers=0)

    model.eval()
    l1_losses  = []
    n_events   = 0
    n_viz_done = 0

    with torch.no_grad():
        for batch_idx, (vox_cpu, _labels) in enumerate(loader):
            vox_raw = voxels_to_device(vox_cpu, device)
            if vox_raw.feature_tensor.shape[0] == 0:
                continue

            # Normalize to log space — must match training preprocessing.
            vox    = log1p_voxels(vox_raw)
            masked, mask_bool = sparse_block_mask(vox, masking_frac, win_ch, win_tick)
            pred   = model.forward_ssl(masked)

            n_input  = vox.feature_tensor.shape[0]
            n_masked = int(mask_bool.sum())
            n_output = pred.feature_tensor.shape[0]
            B        = len(vox.offsets) - 1
            n_events += B

            if mask_bool.any():
                loss = F.l1_loss(
                    pred.feature_tensor[mask_bool],
                    vox.feature_tensor[mask_bool],
                )
                l1_losses.append(loss.item())

            # Print detailed sizes once (first non-empty batch)
            if batch_idx == 0:
                mask_pct = 100.0 * n_masked / max(n_input, 1)
                print(f"\n  [sizes] first batch (B={B} events)")
                print(f"    input  voxels : {n_input:>8d}  ({n_input/B:>7.1f} avg/event)")
                print(f"    masked voxels : {n_masked:>8d}  ({n_masked/B:>7.1f} avg/event)"
                      f"  [{mask_pct:.1f}% of input]")
                print(f"    output voxels : {n_output:>8d}  ({n_output/B:>7.1f} avg/event)")

            # Per-event visualization — convert back to raw ADC for display.
            for evt in range(B):
                if n_viz_done >= n_viz:
                    break
                if int(vox.offsets[evt + 1].item()) - int(vox.offsets[evt].item()) == 0:
                    continue
                orig_evt   = expm1_voxels(_slice_single_event(vox, evt))
                masked_evt = expm1_voxels(_slice_single_event(masked,  evt))
                pred_evt   = expm1_voxels(_slice_single_event(pred,    evt))
                event_id   = batch_idx * batch_size + evt
                _visualize_event(orig_evt, masked_evt, pred_evt, event_id,
                                 viz_dir / f"eval_ssl_event{event_id:04d}.png")
                n_viz_done += 1

    mean_l1 = sum(l1_losses) / len(l1_losses) if l1_losses else float("nan")
    print(f"\n  Evaluated {n_events} events")
    print(f"  Mean L1 loss (masked voxels, log1p space): {mean_l1:.6f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Module 2: SFT evaluation
# ---------------------------------------------------------------------------

def _module_sft(model, dataset, device, cfg: dict) -> None:
    n_samples  = cfg["n_samples"]
    batch_size = cfg["batch_size"]
    n_classes  = cfg["n_classes"]

    print(f"\n{'='*60}")
    print(f"Module 2 — SFT evaluation  (samples={n_samples})")

    n_use = min(n_samples, len(dataset))
    subset = Subset(dataset, torch.randperm(len(dataset))[:n_use].tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=voxels_label_collate_fn, num_workers=0)

    model.eval()
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    n_skipped = 0

    with torch.no_grad():
        for vox_cpu, labels in loader:
            vox    = log1p_voxels(voxels_to_device(vox_cpu, device))
            labels = labels.to(device)

            valid = labels >= 0
            n_skipped += int((~valid).sum())
            if not valid.any() or vox.feature_tensor.shape[0] == 0:
                continue

            logits = model.forward_sft(vox)
            preds  = logits[valid].argmax(dim=1).cpu()
            trues  = labels[valid].cpu()
            for t, p in zip(trues.tolist(), preds.tolist()):
                if 0 <= t < n_classes and 0 <= p < n_classes:
                    confusion[t, p] += 1

    total   = int(confusion.sum())
    correct = int(confusion.diagonal().sum())
    overall_acc = 100.0 * correct / total if total > 0 else float("nan")

    # Macro-averaged accuracy: each class weighted equally (rare classes count more)
    per_class_acc = [
        confusion[i, i].item() / int(confusion[i].sum())
        for i in range(n_classes)
        if int(confusion[i].sum()) > 0
    ]
    macro_acc = 100.0 * sum(per_class_acc) / len(per_class_acc) if per_class_acc else float("nan")

    print(f"\n  {total} events scored  |  {n_skipped} skipped / unlabelled")
    print(f"  Overall accuracy      : {overall_acc:.2f}%  ({correct}/{total})")
    print(f"  Macro-avg accuracy    : {macro_acc:.2f}%  (equal weight per class)")

    names = CLASS_NAMES[:n_classes]
    _print_confusion(confusion, names)
    _print_class_metrics(confusion, names)
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Module registry — add new modules here
# ---------------------------------------------------------------------------
#
# Each entry:  "canonical_name": (display_name, function)
# The function signature must be:  fn(model, dataset, device, cfg: dict) -> None
# cfg contains all CLI parameters as a dict.
#
MODULE_REGISTRY: dict[str, tuple[str, callable]] = {
    "ssl": ("SSL reconstruction", _module_ssl),
    "sft": ("SFT classification", _module_sft),
}

# Numeric aliases
_MODULE_ALIASES: dict[str, str] = {"1": "ssl", "2": "sft"}


def _parse_modules(modules_str: str) -> list[str]:
    keys = [k.strip().lower() for k in str(modules_str).split(",")]
    result, seen = [], set()
    for k in keys:
        canonical = _MODULE_ALIASES.get(k, k)
        if canonical not in MODULE_REGISTRY:
            available = list(MODULE_REGISTRY.keys()) + list(_MODULE_ALIASES.keys())
            raise ValueError(f"Unknown module '{k}'.  Available: {available}")
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_root    = "/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27",
    checkpoint   = None,          # required: path to .pt checkpoint
    apa          = 0,
    view         = "W",
    modules      = "1",           # comma-separated: "1", "1,2", "ssl,sft"
    n_samples    = 64,          # events to evaluate per module
    n_viz        = 5,             # SSL module: number of events to visualize
    batch_size   = 64,
    masking_frac = 0.01,
    win_ch       = 3,
    win_tick     = 5,
    n_classes    = 3,
    device       = "cuda",
    viz_dir      = "./viz_eval",
):
    """Evaluate a trained SparseMAEModel checkpoint on selected modules."""
    if checkpoint is None:
        raise ValueError(
            "--checkpoint is required.\n"
            "  Example: python scripts/eval_mae.py --checkpoint=checkpoints/mae_epoch10.pt"
        )

    module_list = _parse_modules(modules)
    print(f"Modules to run: {[MODULE_REGISTRY[m][0] for m in module_list]}")

    # ── Device ────────────────────────────────────────────────────────────
    if device == "cuda":
        device_t = _least_occupied_cuda_device()
    else:
        device_t = torch.device(device)
    if device_t.type == "cuda":
        torch.cuda.set_device(device_t)
    wp.init()

    # ── Load checkpoint ───────────────────────────────────────────────────
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device_t)
    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    epoch_info = raw.get("epoch", "?") if isinstance(raw, dict) else "?"

    model = SparseMAEModel(n_classes=n_classes).to(device_t)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded: {checkpoint_path}  (epoch {epoch_info})")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = APASparseMetaDataset(
        data_root, apa=apa, view=view, frame_name="frame_rebinned_reco",
    )
    print(f"Dataset: {len(dataset)} total samples")

    # ── Config bundle passed to every module ──────────────────────────────
    cfg = dict(
        n_samples=n_samples, n_viz=n_viz, batch_size=batch_size,
        masking_frac=masking_frac, win_ch=win_ch, win_tick=win_tick,
        n_classes=n_classes, viz_dir=viz_dir,
    )

    # ── Dispatch ──────────────────────────────────────────────────────────
    for mod in module_list:
        _, fn = MODULE_REGISTRY[mod]
        fn(model, dataset, device_t, cfg)


if __name__ == "__main__":
    fire.Fire(main)
