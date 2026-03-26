"""
train_mae.py — Sparse Masked Auto-Encoder training: SSL epochs + SFT epochs.

Architecture
------------
  Backbone  : MinkUNetSparseAttentionCore  (Voxels[1ch] → Voxels[64ch])
  SSL head  : SparseConv2d(64, 1)          charge reconstruction
  SFT head  : SparseCNNHead(in_ch=64)      neutrino-flavour classification on backbone features
  Ref head  : SparseCNNHead(in_ch=1)       same architecture, applied to raw charge (no backbone)

Training loop
-------------
  For each SSL epoch:
    1. One full pass through ssl_dataset (backbone + SSL head updated).
    2. Both SFT heads are reset to random weights.
    3. n_sft_epochs_per_ssl_epoch full passes through sft_dataset,
       training both heads in parallel (backbone frozen).
  Both heads and their optimizers are recreated fresh each SSL epoch
  for an unbiased comparison of SSL features vs. raw charge.

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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.mae_model import SparseMAEModel, voxels_to_device, log1p_voxels, expm1_voxels
from models.sparse_masking import sparse_block_mask
from loader.apa_sparse_dataset import APASparseDataset
from loader.apa_sparse_meta_dataset import APASparseMetaDataset, CLASS_NAMES
from loader.collate import voxels_collate_fn, voxels_label_collate_fn
from metrics_monitor import MetricsMonitor


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss for multi-class classification.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Reduces to cross-entropy when gamma=0.  gamma=2 is a standard default
    that strongly down-weights easy (high-confidence correct) examples and
    concentrates training on rare / hard classes such as nueCC.
    """
    ce  = F.cross_entropy(logits, targets, reduction="none")   # (N,)
    pt  = torch.exp(-ce)                                        # confidence on correct class
    return ((1.0 - pt) ** gamma * ce).mean()


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
        im = ax.imshow(img, aspect="auto", origin="lower", vmin=0.0, vmax=vmax, cmap="cubehelix_r")
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
    viz_batch: int = 0,
):
    model.train()
    ssl_losses = []
    # Store raw CPU batch for end-of-epoch visualization.
    # Targets viz_batch; falls back to batch 0 if viz_batch is out of range.
    viz_vox_cpu  = None   # fallback: batch 0
    viz_vox_target = None # desired: batch viz_batch

    for global_step, vox_cpu in enumerate(ssl_loader):
        # Capture batch 0 as fallback and the target batch if reached.
        if global_step == 0 and vox_cpu is not None:
            viz_vox_cpu = vox_cpu
        if global_step == viz_batch and vox_cpu is not None:
            viz_vox_target = vox_cpu

        # Normalize charge: log(ADC+1) compresses 350× dynamic range to ~10×.
        vox = log1p_voxels(voxels_to_device(vox_cpu, device))

        monitor.on_batch_begin()

        if vox.feature_tensor.shape[0] == 0:
            monitor.on_batch_end(global_step, 0.0, 0)
            continue

        masked, mask_bool = sparse_block_mask(vox, masking_frac, win_ch, win_tick)
        if global_step == 0 and epoch == 1:
            print(f"  [mask] effective masking rate: {mask_bool.float().mean():.1%}")
        pred = model.forward_ssl(masked)

        if mask_bool.any():
            # Loss on ALL active voxels; masked positions upweighted so their
            # total contribution equals that of unmasked positions.
            n_total  = pred.feature_tensor.shape[0]
            n_masked = int(mask_bool.sum())
            per_voxel = F.l1_loss(pred.feature_tensor, vox.feature_tensor,
                                   reduction="none")[:, 0]   # [N]
            weights = torch.ones(n_total, device=device, dtype=per_voxel.dtype)
            weights[mask_bool] = n_total / n_masked
            loss = (per_voxel * weights).mean()
            opt_ssl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.backbone.parameters()) + list(model.charge_head.parameters()),
                max_norm=1.0,
            )
            opt_ssl.step()
            ssl_losses.append(loss.item())
            monitor.on_batch_end(global_step, loss.item(), n_masked)
        else:
            monitor.on_batch_end(global_step, 0.0, 0)

        if (global_step + 1) % 50 == 0:
            ssl_mean = sum(ssl_losses) / len(ssl_losses) if ssl_losses else float("nan")
            print(
                f"  [SSL] Epoch {epoch}  step [{global_step + 1}/{len(ssl_loader)}]"
                f"  loss={ssl_mean:.4f}"
            )

    # ── Visualization (end-of-epoch, model eval, no grad) ─────────────────
    # Use the target batch; fall back to batch 0 if it was out of range.
    if viz_vox_target is None and viz_batch != 0:
        print(f"  [viz] batch {viz_batch} not reached — falling back to batch 0")
    viz_vox_cpu = viz_vox_target if viz_vox_target is not None else viz_vox_cpu
    if viz_vox_cpu is not None:
        model.eval()
        with torch.no_grad():
            vox_viz_raw = voxels_to_device(viz_vox_cpu, device)
            if vox_viz_raw.feature_tensor.shape[0] > 0:
                vox_viz_log      = log1p_voxels(vox_viz_raw)
                masked_viz_log, _ = sparse_block_mask(vox_viz_log, masking_frac, win_ch, win_tick)
                pred_viz_log     = model.forward_ssl(masked_viz_log)
                # Convert log space → raw ADC for human-readable display.
                masked_viz_raw   = expm1_voxels(masked_viz_log)
                pred_viz_raw     = expm1_voxels(pred_viz_log)
                _visualize_ssl(vox_viz_raw, masked_viz_raw, pred_viz_raw, epoch, viz_dir)
        model.train()

    return ssl_losses


# ---------------------------------------------------------------------------
# One SFT epoch (trains both SSL-feature head and raw-charge reference head)
# ---------------------------------------------------------------------------

def _train_sft_epoch(
    model, sft_loader, opt_sft, opt_ref,
    device, n_classes, epoch, sft_epoch,
    focal_gamma: float = 2.0,
):
    model.freeze_backbone()
    sft_losses, ref_losses = [], []
    confusion_sft = torch.zeros(n_classes, n_classes, dtype=torch.long)
    confusion_ref = torch.zeros(n_classes, n_classes, dtype=torch.long)

    for step, (vox_sft_cpu, labels) in enumerate(sft_loader):
        vox_sft = log1p_voxels(voxels_to_device(vox_sft_cpu, device))
        labels  = labels.to(device)

        valid = labels >= 0
        if not valid.any() or vox_sft.feature_tensor.shape[0] == 0:
            continue

        trues = labels[valid].cpu()

        # ── SSL feature head ──────────────────────────────────────────────
        logits = model.forward_sft(vox_sft)
        # Guard: strided sparse convs can drop empty batch items from offsets,
        # making logits.shape[0] < labels.shape[0].  Skip the batch if so.
        if logits.shape[0] != labels.shape[0]:
            continue
        sft_loss = focal_loss(logits[valid], labels[valid], gamma=focal_gamma)
        opt_sft.zero_grad()
        sft_loss.backward()
        opt_sft.step()
        sft_losses.append(sft_loss.item())
        preds = logits[valid].argmax(dim=1).cpu()
        for t, p in zip(trues.tolist(), preds.tolist()):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                confusion_sft[t, p] += 1

        # ── Raw-charge reference head ─────────────────────────────────────
        logits_ref = model.forward_sft_ref(vox_sft)
        if logits_ref.shape[0] != labels.shape[0]:
            continue
        ref_loss = focal_loss(logits_ref[valid], labels[valid], gamma=focal_gamma)
        opt_ref.zero_grad()
        ref_loss.backward()
        opt_ref.step()
        ref_losses.append(ref_loss.item())
        preds_ref = logits_ref[valid].argmax(dim=1).cpu()
        for t, p in zip(trues.tolist(), preds_ref.tolist()):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                confusion_ref[t, p] += 1

        if (step + 1) % 50 == 0:
            sft_mean  = sum(sft_losses) / len(sft_losses) if sft_losses else float("nan")
            ref_mean  = sum(ref_losses)  / len(ref_losses)  if ref_losses  else float("nan")
            total_s   = int(confusion_sft.sum())
            acc_s     = 100.0 * int(confusion_sft.diagonal().sum()) / total_s if total_s > 0 else float("nan")
            total_r   = int(confusion_ref.sum())
            acc_r     = 100.0 * int(confusion_ref.diagonal().sum()) / total_r if total_r > 0 else float("nan")
            print(
                f"  [SFT] SSL-epoch {epoch}  SFT-epoch {sft_epoch}"
                f"  step [{step + 1}/{len(sft_loader)}]"
                f"  SSL-feat: loss={sft_mean:.4f} acc={acc_s:.1f}%"
                f"  | raw-charge: loss={ref_mean:.4f} acc={acc_r:.1f}%"
            )

    model.unfreeze_backbone()
    return sft_losses, confusion_sft, ref_losses, confusion_ref


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_root                  = "/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27",
    apa                        = 0,
    view                       = "W",
    batch_size                 = 64,
    epochs                     = 2,
    lr                         = 1e-3,
    scheduler_step             = 10,
    gamma                      = 0.7,
    n_sft_epochs_per_ssl_epoch = 3,    # full SFT epochs per SSL epoch
    masking_frac               = 0.01,
    win_ch                     = 3,
    win_tick                   = 5,
    n_classes                  = 3,
    focal_gamma                = 2.0,  # focal loss gamma; 0 = plain cross-entropy
    ssl_subset_frac            = 1.0,  # fraction of SSL dataset to use
    sft_subset_frac            = 1.0,  # fraction of SFT dataset to use
    num_workers                = 0,    # set >0 only if warp is initialised in workers
    device                     = "cuda",
    metrics_dir                = "./metrics",
    checkpoints_dir            = "./checkpoints",
    save_every                 = 5,
    viz_dir                    = "./viz",
    viz_batch                  = 0,      # which batch to visualize (0-indexed); 0 if out of range
    resume                     = None,   # path to checkpoint to resume from
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
        # ssl_dataset = Subset(ssl_dataset, list(range(n_ssl_use)))  # deterministic subset for reproducibility
        print(f"ssl_subset_frac={ssl_subset_frac}: using {n_ssl_use} SSL samples")

    if sft_subset_frac < 1.0:
        n_sft_use   = max(1, int(len(sft_dataset) * sft_subset_frac))
        sft_dataset = Subset(sft_dataset, torch.randperm(len(sft_dataset))[:n_sft_use])
        # sft_dataset = Subset(sft_dataset, list(range(n_sft_use)))  # deterministic subset for reproducibility
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
    sched_ssl = StepLR(opt_ssl, step_size=scheduler_step, gamma=gamma)
    # opt_sft and opt_ref are recreated each SSL epoch after head reset (see loop below)

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_epoch = 1
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "opt_ssl" in ckpt:
            opt_ssl.load_state_dict(ckpt["opt_ssl"])
        if "sched_ssl" in ckpt:
            sched_ssl.load_state_dict(ckpt["sched_ssl"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {resume}  (epoch {start_epoch - 1} → continuing from {start_epoch})")

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
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*60}")
        print(f"SSL Epoch {epoch}/{epochs}")

        monitor.on_epoch_begin(epoch)

        # ── SSL epoch ─────────────────────────────────────────────────────
        ssl_losses = _train_ssl_epoch(
            model, ssl_loader, opt_ssl,
            device, masking_frac, win_ch, win_tick,
            epoch, monitor, viz_dir,
            viz_batch=viz_batch,
        )
        ssl_mean = sum(ssl_losses) / len(ssl_losses) if ssl_losses else float("nan")
        print(f"  SSL epoch {epoch} done  |  mean L1={ssl_mean:.4f}")
        sched_ssl.step()

        # ── SFT epochs ────────────────────────────────────────────────────
        # Reset both heads + recreate optimizers for a fair per-epoch comparison.
        model.reset_sft_head()
        opt_sft = optim.AdamW(model.nu_flavor_head.parameters(),     lr=lr)
        opt_ref = optim.AdamW(model.ref_nu_flavor_head.parameters(), lr=lr)

        all_sft_losses, all_ref_losses = [], []
        confusion_sft = torch.zeros(n_classes, n_classes, dtype=torch.long)
        confusion_ref = torch.zeros(n_classes, n_classes, dtype=torch.long)

        for sft_epoch in range(1, n_sft_epochs_per_ssl_epoch + 1):
            sft_losses, conf_sft_ep, ref_losses, conf_ref_ep = _train_sft_epoch(
                model, sft_loader, opt_sft, opt_ref,
                device, n_classes, epoch, sft_epoch,
                focal_gamma=focal_gamma,
            )
            all_sft_losses.extend(sft_losses)
            all_ref_losses.extend(ref_losses)
            confusion_sft += conf_sft_ep
            confusion_ref += conf_ref_ep

            sft_mean_ep = sum(sft_losses) / len(sft_losses) if sft_losses else float("nan")
            ref_mean_ep = sum(ref_losses)  / len(ref_losses)  if ref_losses  else float("nan")
            total_s     = int(conf_sft_ep.sum())
            acc_s       = 100.0 * int(conf_sft_ep.diagonal().sum()) / total_s if total_s > 0 else float("nan")
            total_r     = int(conf_ref_ep.sum())
            acc_r       = 100.0 * int(conf_ref_ep.diagonal().sum()) / total_r if total_r > 0 else float("nan")
            print(
                f"  SFT epoch {sft_epoch}/{n_sft_epochs_per_ssl_epoch}"
                f"  |  SSL-feat: CE={sft_mean_ep:.4f} acc={acc_s:.1f}%"
                f"  |  raw-charge: CE={ref_mean_ep:.4f} acc={acc_r:.1f}%"
            )

        sft_mean  = sum(all_sft_losses) / len(all_sft_losses) if all_sft_losses else float("nan")
        ref_mean  = sum(all_ref_losses)  / len(all_ref_losses)  if all_ref_losses  else float("nan")
        total_sft = int(confusion_sft.sum())
        sft_acc   = 100.0 * int(confusion_sft.diagonal().sum()) / total_sft if total_sft > 0 else 0.0
        total_ref = int(confusion_ref.sum())
        ref_acc   = 100.0 * int(confusion_ref.diagonal().sum()) / total_ref if total_ref > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"Epoch {epoch:3d}  |  SSL L1={ssl_mean:.4f}")
        print(f"  SSL features  :  CE={sft_mean:.4f}  acc={sft_acc:.1f}%")
        print(f"  Raw charge ref:  CE={ref_mean:.4f}  acc={ref_acc:.1f}%")
        print(f"\n  [SSL features]")
        _print_confusion(confusion_sft, CLASS_NAMES)
        _print_class_metrics(confusion_sft, CLASS_NAMES)
        print(f"\n  [Raw charge reference]")
        _print_confusion(confusion_ref, CLASS_NAMES)
        _print_class_metrics(confusion_ref, CLASS_NAMES)
        print(f"{'='*60}\n")

        monitor.on_validation_begin(epoch)
        monitor.on_validation_end()
        monitor.on_epoch_end(epoch, sft_mean if not math.isnan(sft_mean) else 0.0, sft_acc)

        if epoch % save_every == 0 or epoch == epochs:
            ckpt = checkpoints_dir / f"mae_epoch{epoch}.pt"
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "opt_ssl":  opt_ssl.state_dict(),
                "sched_ssl": sched_ssl.state_dict(),
            }, ckpt)
            monitor.save()
            print(f"Checkpoint saved: {ckpt}")

    monitor.print_summary()
    monitor.save()


if __name__ == "__main__":
    fire.Fire(main)
