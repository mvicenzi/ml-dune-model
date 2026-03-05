"""
train_mae.py — Sparse Masked Auto-Encoder training with interleaved SSL + SFT.

Architecture
------------
  Backbone  : MinkUNetSparseAttentionCore  (Voxels[1ch] → Voxels[64ch])
  SSL head  : SparseConv2d(64, 1)          charge reconstruction
  SFT head  : Linear(64, n_classes)        neutrino-flavour classification

Training loop
-------------
  Each global step: n_ssl SSL batches (backbone + SSL head updated),
                    n_sft SFT batches (SFT head only updated, backbone frozen).

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
import itertools
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
import torch.optim as optim
import warp as wp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

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
# One epoch of interleaved SSL + SFT
# ---------------------------------------------------------------------------

def _train_epoch(
    model, ssl_iter, sft_iter, opt_ssl, opt_sft,
    device, n_ssl, n_sft, masking_frac, win_ch, win_tick,
    n_classes, epoch, steps_per_epoch, monitor,
):
    model.train()

    ssl_losses  = []
    sft_losses  = []
    confusion   = torch.zeros(n_classes, n_classes, dtype=torch.long)
    global_step = 0

    for step in range(steps_per_epoch):

        # ── SSL phase ──────────────────────────────────────────────────────
        for k in range(n_ssl):
            vox = voxels_to_device(next(ssl_iter), device)

            monitor.on_batch_begin()

            # Skip batches where every event has 0 active voxels —
            # WarpConvNet's kernel map builder crashes on empty coordinate tensors.
            if vox.feature_tensor.shape[0] == 0:
                monitor.on_batch_end(global_step, 0.0, 0)
                global_step += 1
                continue

            masked, mask_bool = sparse_block_mask(vox, masking_frac, win_ch, win_tick)

            pred  = model.forward_ssl(masked)

            # L1 loss over masked voxels only
            if mask_bool.any():
                loss = F.l1_loss(
                    pred.feature_tensor[mask_bool],
                    vox.feature_tensor[mask_bool],
                )
                opt_ssl.zero_grad()
                loss.backward()
                opt_ssl.step()
                ssl_losses.append(loss.item())
                monitor.on_batch_end(global_step, loss.item(), int(mask_bool.sum()))
            else:
                monitor.on_batch_end(global_step, 0.0, 0)

            global_step += 1

        # ── SFT phase ──────────────────────────────────────────────────────
        model.freeze_backbone()
        for _ in range(n_sft):
            vox_sft, labels = next(sft_iter)
            vox_sft = voxels_to_device(vox_sft, device)
            labels  = labels.to(device)

            valid = labels >= 0
            if not valid.any():
                continue

            # Skip if no active voxels across the whole batch
            if vox_sft.feature_tensor.shape[0] == 0:
                continue

            logits   = model.forward_sft(vox_sft)          # [B, n_classes]
            sft_loss = F.cross_entropy(logits[valid], labels[valid])
            opt_sft.zero_grad()
            sft_loss.backward()
            opt_sft.step()
            sft_losses.append(sft_loss.item())

            # Accumulate confusion matrix
            preds = logits[valid].argmax(dim=1).cpu()
            trues = labels[valid].cpu()
            for t, p in zip(trues.tolist(), preds.tolist()):
                if 0 <= t < n_classes and 0 <= p < n_classes:
                    confusion[t, p] += 1

        model.unfreeze_backbone()

        # Progress print every 50 global steps
        if (step + 1) % 50 == 0:
            ssl_mean = sum(ssl_losses) / len(ssl_losses) if ssl_losses else float("nan")
            sft_mean = sum(sft_losses) / len(sft_losses) if sft_losses else float("nan")
            print(
                f"  Epoch {epoch}  step [{step + 1}/{steps_per_epoch}]"
                f"  SSL={ssl_mean:.4f}  SFT={sft_mean:.4f}"
            )

    return ssl_losses, sft_losses, confusion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_root       = "/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27",
    apa             = 0,
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
    num_workers     = 0,       # set >0 only if warp is initialised in workers
    device          = "cuda",
    metrics_dir     = "./metrics",
    checkpoints_dir = "./checkpoints",
    save_every      = 5,
):
    """Sparse MAE training: interleaved SSL (charge reconstruction) + SFT (nu flavour)."""
    wp.init()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print(f"Device: {device}")

    # ── Datasets & DataLoaders ────────────────────────────────────────────
    ssl_dataset = APASparseDataset(
        data_root, apa=apa, view=view, frame_name="frame_rebinned_reco",
    )
    sft_dataset = APASparseMetaDataset(
        data_root, apa=apa, view=view, frame_name="frame_rebinned_reco",
    )

    ssl_loader = DataLoader(
        ssl_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=voxels_collate_fn, num_workers=num_workers,
    )
    sft_loader = DataLoader(
        sft_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=voxels_label_collate_fn, num_workers=num_workers,
    )

    ssl_iter = itertools.cycle(ssl_loader)
    sft_iter = itertools.cycle(sft_loader)

    # Steps per epoch = full pass through SSL data, grouped into cycles of n_ssl
    steps_per_epoch = max(1, len(ssl_loader) // n_ssl)
    print(f"SSL dataset: {len(ssl_dataset)} samples  |  "
          f"SFT dataset: {len(sft_dataset)} samples")
    print(f"Steps per epoch: {steps_per_epoch}  "
          f"(n_ssl={n_ssl}, n_sft={n_sft})")

    # ── Model ─────────────────────────────────────────────────────────────
    model = SparseMAEModel(n_classes=n_classes).to(device)

    # ── Optimizers ────────────────────────────────────────────────────────
    # opt_ssl updates backbone + charge_head; opt_sft updates only SFT head.
    opt_ssl = optim.AdamW(
        list(model.backbone.parameters()) + list(model.charge_head.parameters()),
        lr=lr,
    )
    opt_sft = optim.AdamW(model.nu_flavor_head.parameters(), lr=lr)

    sched_ssl = StepLR(opt_ssl, step_size=scheduler_step, gamma=gamma)
    sched_sft = StepLR(opt_sft, step_size=scheduler_step, gamma=gamma)

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

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        monitor.on_epoch_begin(epoch)

        ssl_losses, sft_losses, confusion = _train_epoch(
            model, ssl_iter, sft_iter, opt_ssl, opt_sft,
            device, n_ssl, n_sft, masking_frac, win_ch, win_tick,
            n_classes, epoch, steps_per_epoch, monitor,
        )

        ssl_mean = sum(ssl_losses) / len(ssl_losses) if ssl_losses else float("nan")
        sft_mean = sum(sft_losses) / len(sft_losses) if sft_losses else float("nan")

        total_sft = int(confusion.sum())
        sft_correct = int(confusion.diagonal().sum())
        sft_acc = 100.0 * sft_correct / total_sft if total_sft > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"Epoch {epoch:3d}  |  SSL L1={ssl_mean:.4f}  |  "
              f"SFT CE={sft_mean:.4f}  |  SFT acc={sft_acc:.1f}%")
        _print_confusion(confusion, CLASS_NAMES)
        _print_class_metrics(confusion, CLASS_NAMES)
        print(f"{'='*60}\n")

        sched_ssl.step()
        sched_sft.step()

        # Use SFT mean CE as the "test loss" for MetricsMonitor bookkeeping
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
