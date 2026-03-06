"""
train_mae.py — Sparse Masked Auto-Encoder training: SSL epochs + SFT epochs.

Architecture
------------
  Backbone  : MinkUNetSparseAttentionCore  (Voxels[1ch] → Voxels[64ch])
  SSL head  : SparseConv2d(64, 1)          charge reconstruction
  SFT head  : Linear(64, n_classes)        neutrino-flavour classification

Training loop
-------------
  For each SSL epoch:
    1. One full pass through ssl_dataset (backbone + SSL head updated).
    2. n_sft_epochs_per_ssl_epoch full passes through sft_dataset
       (SFT head only updated, backbone frozen).
  The SFT head and its optimizer persist across SSL epochs.

  After each SSL epoch a PNG is saved comparing:
    original (unmasked) | masked input | reconstructed output
  for the first batch of that epoch.

SFT classes
-----------
  0  numuCC  (nu_pdg=14, nu_ccnc=0)
  1  nueCC   (nu_pdg=12, nu_ccnc=0)
  2  NC      (nu_ccnc=1)
 -1  skip

Usage
-----
  python scripts/train_mae.py               # defaults
  python scripts/train_mae.py --epochs=50 --batch_size=32
"""

import sys
import math
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR

# ---------------------------------------------------------------------------
# GPU selection helper
# ---------------------------------------------------------------------------

def _least_occupied_cuda_device() -> torch.device:
    """Return the CUDA device with the most free VRAM, or CPU if none available."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    best_idx, best_free = 0, 0
    for i in range(n):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free, best_idx = free, i
    dev = torch.device(f"cuda:{best_idx}")
    print(f"Selected {dev}  ({best_free / 2**30:.1f} GB free"
          f" of {n} GPU{'s' if n > 1 else ''})")
    return dev


# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.mae_model import SparseMAEModel, voxels_to_device
from models.sparse_masking import sparse_block_mask
from loader.apa_sparse_dataset import APASparseDataset
from loader.apa_sparse_meta_dataset import APASparseMetaDataset, CLASS_NAMES
from loader.collate import voxels_collate_fn, voxels_label_collate_fn
from metrics_monitor import MetricsMonitor


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_confusion(cm: torch.Tensor, class_names: list[str]) -> None:
    n = len(class_names)
    width = max(len(n) for n in class_names) + 2
    header = " " * (width + 2) + "  ".join(f"{n:>{width}}" for n in class_names)
    print(f"\n  Confusion matrix  (rows = true, cols = predicted)")
    print(f"  {header}")
    for i, name in enumerate(class_names):
        row = f"  {name:>{width}}  " + "  ".join(f"{cm[i, j].item():>{width}d}" for j in range(n))
        print(row)


def _print_class_metrics(cm: torch.Tensor, class_names: list[str]) -> None:
    n = len(class_names)
    print(f"\n  Per-class metrics:")
    print(f"  {'class':>10}  {'efficiency':>12}  {'purity':>10}")
    for i in range(n):
        tp = cm[i, i].item()
        fn = cm[i, :].sum().item() - tp
        fp = cm[:, i].sum().item() - tp
        eff = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        pur = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        print(f"  {class_names[i]:>10}  {eff:>12.4f}  {pur:>10.4f}")


# ---------------------------------------------------------------------------
# SSL Visualization
# ---------------------------------------------------------------------------

def _sparse_to_dense(coords: torch.Tensor, feats: torch.Tensor) -> "np.ndarray | None":
    """
    Convert sparse (N, 2) int coords and (N, 1) float feats to a dense 2-D array.

    coords[:, 0] = channel,  coords[:, 1] = tick.
    Returns float32 numpy array of shape (H, W), or None if empty.
    """
    if len(coords) == 0:
        return None
    ch = coords[:, 0]
    tk = coords[:, 1]
    ch_min, ch_max = int(ch.min()), int(ch.max())
    tk_min, tk_max = int(tk.min()), int(tk.max())
    H = ch_max - ch_min + 1
    W = tk_max - tk_min + 1
    grid = torch.zeros(H, W)
    grid[ch - ch_min, tk - tk_min] = feats[:, 0]
    return grid.numpy()


def _visualize_ssl(original_vox, masked_vox, pred_vox, epoch: int, viz_dir: Path) -> None:
    """
    Save a 3-panel PNG: original (unmasked) | masked input | reconstructed output.

    Uses the first event in the batch.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [viz] matplotlib not available — skipping")
        return

    offsets = original_vox.offsets
    if len(offsets) < 2:
        return
    end = int(offsets[1].item())
    if end == 0:
        return

    # All three share the same coordinate structure (masking/conv preserve coords).
    coords = original_vox.coordinate_tensor[:end].cpu().int()
    orig_dense   = _sparse_to_dense(coords, original_vox.feature_tensor[:end].cpu())
    masked_dense = _sparse_to_dense(coords, masked_vox.feature_tensor[:end].cpu())
    pred_dense   = _sparse_to_dense(coords, pred_vox.feature_tensor[:end].cpu())

    if orig_dense is None:
        return

    vmax = float(orig_dense.max()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Original (unmasked)", "Masked input", "Reconstructed output"]
    for ax, img, title in zip(axes, [orig_dense, masked_dense, pred_dense], titles):
        im = ax.imshow(img, aspect="auto", origin="lower", vmin=0.0, vmax=vmax, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Channel")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"SSL Reconstruction — Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / f"ssl_viz_epoch{epoch:04d}.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [viz] Saved: {out_path}")


# ---------------------------------------------------------------------------
# One SSL epoch
# ---------------------------------------------------------------------------

def _train_ssl_epoch(
    model, ssl_loader, opt_ssl,
    device, masking_frac, win_ch, win_tick,
    epoch, monitor, viz_dir: Path,
):
    model.train()
    ssl_losses = []
    # Store raw CPU batch from first step for visualization (no gradient risk).
    viz_vox_cpu = None

    for global_step, vox_cpu in enumerate(ssl_loader):
        vox = voxels_to_device(vox_cpu, device)

        monitor.on_batch_begin()

        if vox.feature_tensor.shape[0] == 0:
            monitor.on_batch_end(global_step, 0.0, 0)
            continue

        masked, mask_bool = sparse_block_mask(vox, masking_frac, win_ch, win_tick)
        pred = model.forward_ssl(masked)

        # Capture first valid batch for end-of-epoch visualization.
        if viz_vox_cpu is None:
            viz_vox_cpu = vox_cpu

        if mask_bool.any():
            loss = F.l1_loss(
                pred.feature_tensor[mask_bool],
                vox.feature_tensor[mask_bool],
            )
            opt_ssl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.backbone.parameters()) + list(model.charge_head.parameters()),
                max_norm=1.0,
            )
            opt_ssl.step()
            ssl_losses.append(loss.item())
            monitor.on_batch_end(global_step, loss.item(), int(mask_bool.sum()))
        else:
            monitor.on_batch_end(global_step, 0.0, 0)

        if (global_step + 1) % 50 == 0:
            ssl_mean = sum(ssl_losses) / len(ssl_losses) if ssl_losses else float("nan")
            print(
                f"  [SSL] Epoch {epoch}  step [{global_step + 1}/{len(ssl_loader)}]"
                f"  loss={ssl_mean:.4f}"
            )

    # ── Visualization (end-of-epoch, model eval, no grad) ─────────────────
    if viz_vox_cpu is not None:
        model.eval()
        with torch.no_grad():
            vox_viz   = voxels_to_device(viz_vox_cpu, device)
            if vox_viz.feature_tensor.shape[0] > 0:
                masked_viz, _ = sparse_block_mask(vox_viz, masking_frac, win_ch, win_tick)
                pred_viz      = model.forward_ssl(masked_viz)
                _visualize_ssl(vox_viz, masked_viz, pred_viz, epoch, viz_dir)
        model.train()

    return ssl_losses


# ---------------------------------------------------------------------------
# One SFT epoch
# ---------------------------------------------------------------------------

def _train_sft_epoch(
    model, sft_loader, opt_sft,
    device, n_classes, epoch, sft_epoch,
):
    model.freeze_backbone()
    sft_losses = []
    confusion  = torch.zeros(n_classes, n_classes, dtype=torch.long)

    for step, (vox_sft_cpu, labels) in enumerate(sft_loader):
        vox_sft = voxels_to_device(vox_sft_cpu, device)
        labels  = labels.to(device)

        valid = labels >= 0
        if not valid.any():
            continue
        if vox_sft.feature_tensor.shape[0] == 0:
            continue

        logits   = model.forward_sft(vox_sft)
        sft_loss = F.cross_entropy(logits[valid], labels[valid])
        opt_sft.zero_grad()
        sft_loss.backward()
        opt_sft.step()
        sft_losses.append(sft_loss.item())

        preds = logits[valid].argmax(dim=1).cpu()
        trues = labels[valid].cpu()
        for t, p in zip(trues.tolist(), preds.tolist()):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                confusion[t, p] += 1

        if (step + 1) % 50 == 0:
            sft_mean = sum(sft_losses) / len(sft_losses) if sft_losses else float("nan")
            total    = int(confusion.sum())
            correct  = int(confusion.diagonal().sum())
            acc      = 100.0 * correct / total if total > 0 else float("nan")
            print(
                f"  [SFT] SSL-epoch {epoch}  SFT-epoch {sft_epoch}"
                f"  step [{step + 1}/{len(sft_loader)}]"
                f"  loss={sft_mean:.4f}  acc={acc:.1f}%"
            )

    model.unfreeze_backbone()
    return sft_losses, confusion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_root                  = "/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27",
    apa                        = 0,
    view                       = "W",
    batch_size                 = 16,
    epochs                     = 2,
    lr                         = 1e-3,
    scheduler_step             = 10,
    gamma                      = 0.7,
    n_sft_epochs_per_ssl_epoch = 3,    # full SFT epochs per SSL epoch
    masking_frac               = 0.3,
    win_ch                     = 10,
    win_tick                   = 20,
    n_classes                  = 3,
    ssl_subset_frac            = 1.0,  # fraction of SSL dataset to use
    sft_subset_frac            = 1.0,  # fraction of SFT dataset to use
    num_workers                = 0,    # set >0 only if warp is initialised in workers
    device                     = "cuda",
    metrics_dir                = "./metrics",
    checkpoints_dir            = "./checkpoints",
    save_every                 = 5,
    viz_dir                    = "./viz",
):
    """Sparse MAE training: one SSL epoch → n_sft_epochs_per_ssl_epoch SFT epochs, repeated."""
    # Resolve device BEFORE wp.init() so Warp/CuPy establish their CUDA context
    # on the correct GPU.  torch.cuda.set_device() must be called first so that
    # CuPy's raw-kernel compiler targets the same device as our tensors.
    if device == "cuda":
        device = _least_occupied_cuda_device()
    else:
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    wp.init()
    torch.manual_seed(42)

    print(f"Device: {device}")

    # ── Datasets & DataLoaders ────────────────────────────────────────────
    ssl_dataset = APASparseDataset(
        data_root, apa=apa, view=view, frame_name="frame_rebinned_reco",
    )
    sft_dataset = APASparseMetaDataset(
        data_root, apa=apa, view=view, frame_name="frame_rebinned_reco",
    )

    if ssl_subset_frac < 1.0:
        n_ssl_use   = max(1, int(len(ssl_dataset) * ssl_subset_frac))
        ssl_dataset = Subset(ssl_dataset, torch.randperm(len(ssl_dataset))[:n_ssl_use])
        print(f"ssl_subset_frac={ssl_subset_frac}: using {n_ssl_use} SSL samples")

    if sft_subset_frac < 1.0:
        n_sft_use   = max(1, int(len(sft_dataset) * sft_subset_frac))
        sft_dataset = Subset(sft_dataset, torch.randperm(len(sft_dataset))[:n_sft_use])
        print(f"sft_subset_frac={sft_subset_frac}: using {n_sft_use} SFT samples")

    ssl_loader = DataLoader(
        ssl_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=voxels_collate_fn, num_workers=num_workers,
    )
    sft_loader = DataLoader(
        sft_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=voxels_label_collate_fn, num_workers=num_workers,
    )

    print(f"SSL dataset: {len(ssl_dataset)} samples  |  SFT dataset: {len(sft_dataset)} samples")
    print(f"n_sft_epochs_per_ssl_epoch={n_sft_epochs_per_ssl_epoch}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = SparseMAEModel(n_classes=n_classes).to(device)

    # ── Optimizers ────────────────────────────────────────────────────────
    opt_ssl = optim.AdamW(
        list(model.backbone.parameters()) + list(model.charge_head.parameters()),
        lr=lr,
    )
    opt_sft = optim.AdamW(model.nu_flavor_head.parameters(), lr=lr)
    sched_ssl = StepLR(opt_ssl, step_size=scheduler_step, gamma=gamma)

    # ── Metrics ───────────────────────────────────────────────────────────
    monitor = MetricsMonitor("sparse_mae", save_dir=metrics_dir)
    monitor.on_train_begin(
        model,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        scheduler_step_size=scheduler_step,
        gamma=gamma,
    )

    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(exist_ok=True)
    viz_dir = Path(viz_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"SSL Epoch {epoch}/{epochs}")

        monitor.on_epoch_begin(epoch)

        # ── SSL epoch ─────────────────────────────────────────────────────
        ssl_losses = _train_ssl_epoch(
            model, ssl_loader, opt_ssl,
            device, masking_frac, win_ch, win_tick,
            epoch, monitor, viz_dir,
        )
        ssl_mean = sum(ssl_losses) / len(ssl_losses) if ssl_losses else float("nan")
        print(f"  SSL epoch {epoch} done  |  mean L1={ssl_mean:.4f}")
        sched_ssl.step()

        # ── SFT epochs ────────────────────────────────────────────────────
        all_sft_losses = []
        confusion      = torch.zeros(n_classes, n_classes, dtype=torch.long)

        for sft_epoch in range(1, n_sft_epochs_per_ssl_epoch + 1):
            sft_losses, conf_ep = _train_sft_epoch(
                model, sft_loader, opt_sft,
                device, n_classes, epoch, sft_epoch,
            )
            all_sft_losses.extend(sft_losses)
            confusion += conf_ep

            sft_mean_ep = sum(sft_losses) / len(sft_losses) if sft_losses else float("nan")
            total_ep    = int(conf_ep.sum())
            correct_ep  = int(conf_ep.diagonal().sum())
            acc_ep      = 100.0 * correct_ep / total_ep if total_ep > 0 else float("nan")
            print(
                f"  SFT epoch {sft_epoch}/{n_sft_epochs_per_ssl_epoch}"
                f"  |  CE={sft_mean_ep:.4f}  |  acc={acc_ep:.1f}%"
            )

        sft_mean    = sum(all_sft_losses) / len(all_sft_losses) if all_sft_losses else float("nan")
        total_sft   = int(confusion.sum())
        sft_correct = int(confusion.diagonal().sum())
        sft_acc     = 100.0 * sft_correct / total_sft if total_sft > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"Epoch {epoch:3d}  |  SSL L1={ssl_mean:.4f}  |  "
              f"SFT CE={sft_mean:.4f}  |  SFT acc={sft_acc:.1f}%")
        _print_confusion(confusion, CLASS_NAMES)
        _print_class_metrics(confusion, CLASS_NAMES)
        print(f"{'='*60}\n")

        monitor.on_validation_begin(epoch)
        monitor.on_validation_end()
        monitor.on_epoch_end(epoch, sft_mean if not math.isnan(sft_mean) else 0.0, sft_acc)

        if epoch % save_every == 0 or epoch == epochs:
            ckpt = checkpoints_dir / f"mae_epoch{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt)
            monitor.save()
            print(f"Checkpoint saved: {ckpt}")

    monitor.print_summary()
    monitor.save()


if __name__ == "__main__":
    fire.Fire(main)
