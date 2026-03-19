"""
DINO training script for DUNE sparse UNet backbone.

Usage:
    python dino/train_dino.py --epochs=100 --batch_size=16 --backbone_name=attn_default
    python dino/train_dino.py --epochs=2 --batch_size=4 --test_mode=True --debug=True
"""

import fire
import inspect
import json
import sys
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

from loader.dataset import DUNEImageDataset
from loader.splits import train_val_split, Subset

from .config import DINOConfig
from .masking import SparseVoxelMasker
from .loss import PixelDINOLoss
from .scheduler import CosineScheduler
from .model import DINODuneModel
from .debug import DINODebugger


@torch.no_grad()
def validate_epoch(model, val_loader, masker, loss_fn, device):
    """
    Compute mean DINO loss on the validation set.

    Student runs in eval mode (no dropout / batchnorm stochasticity) with the
    same random masking as during training. Teacher is always in eval mode.
    No gradients are computed. Model is restored to train mode before returning.

    Returns:
        mean validation loss (0.0 if val_loader is empty)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for data, _ in val_loader:
        data = data.to(device)
        x_student, mask = masker(data)
        teacher_feats = model.teacher(data)
        student_feats = model.student(x_student)
        loss = loss_fn(student_feats, teacher_feats, mask, data)
        total_loss += loss.item()
        n_batches += 1

    model.train()  # restores student; teacher stays eval via DINODuneModel.train()
    return total_loss / n_batches if n_batches > 0 else 0.0


def main(
    backbone_name: str = "attn_default",
    epochs: int = 100,
    batch_size: int = 50,
    lr: float = 1e-4,
    mask_ratio: float = 0.5,
    loss_type: str = "cosine",
    center_momentum: float = 0.9,
    use_centering: bool = True,
    teacher_temp: float = 1.0,
    student_temp: float = 1.0,
    momentum_start: float = 0.996,
    momentum_end: float = 0.9999,
    weight_decay: float = 0.04,
    weight_decay_end: float = 0.4,
    warmup_epochs: int = 5,
    output_dir: str = "./dino_checkpoints",
    save_every: int = 10,
    device: str = "cuda",
    debug: bool = False,
    debug_every: int = 100,
    debug_dir: str = "./dino_debug",
    run_name: str = "",
    test_mode: bool = False,
    num_workers: int = 4,
):
    """
    DINO training loop for DUNE detector.

    Args:
        backbone_name: Model architecture ("attn_default", "base", etc.)
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        lr: Base learning rate
        mask_ratio: Fraction of active pixels to mask
        loss_type: "cosine", "mse", or "dino"
        center_momentum: EMA decay for the teacher center buffer
        use_centering: subtract running center from teacher features before loss
        teacher_temp: teacher softmax temperature (only used for "dino")
        student_temp: student softmax temperature (only used for "dino")
        momentum_start: Initial EMA momentum
        momentum_end: Final EMA momentum
        weight_decay: L2 regularization
        weight_decay_end: Final weight decay (cosine annealed)
        warmup_epochs: Linear warmup duration
        output_dir: Where to save checkpoints
        save_every: Save checkpoint every N epochs
        device: "cuda" or "cpu"
        debug: Enable debugging and history logging
        debug_every: Log scalars / stats / grad norms every N batches
        debug_dir: Base directory for debug outputs
        run_name: Optional label; outputs go to debug_dir/run_name/ if set
        test_mode: Use small subset for quick smoke tests
        num_workers: Number of dataloader workers
    """
    # ============ Setup ============
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # If a run name is given, nest outputs under debug_dir/run_name/ and output_dir/run_name/
    if run_name:
        debug_dir = f"{debug_dir}/{run_name}"
        output_dir = f"{output_dir}/{run_name}"

    # Build config
    cfg = DINOConfig(
        backbone_name=backbone_name,
        mask_ratio=mask_ratio,
        loss_type=loss_type,
        center_momentum=center_momentum,
        use_centering=use_centering,
        teacher_temp=teacher_temp,
        student_temp=student_temp,
        momentum_start=momentum_start,
        momentum_end=momentum_end,
        lr=lr,
        weight_decay=weight_decay,
        weight_decay_end=weight_decay_end,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        save_every=save_every,
        debug=debug,
        debug_every=debug_every,
        debug_dir=debug_dir,
        run_name=run_name,
        num_workers=num_workers,
    )

    print(f"Device: {device}")
    print(f"Config: backbone={cfg.backbone_name}, mask_ratio={cfg.mask_ratio}, "
          f"lr={cfg.lr}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, warmup_epochs={cfg.warmup_epochs}, "
          f" momentum_start={cfg.momentum_start}, momentum_end={cfg.momentum_end}")
    print(f'Loss: type={cfg.loss_type}, center_momentum={cfg.center_momentum}, '
          f'use_centering={cfg.use_centering}, teacher_temp={cfg.teacher_temp}, '
          f'student_temp={cfg.student_temp}')

    # ============ Data ============
    print("\nLoading dataset...")
    dataset = DUNEImageDataset(
        rootdir=cfg.rootdir,
        class_names=["numu", "nue", "nutau", "NC"],
        view_index=cfg.view_index,
        use_cache=True,
    )

    if test_mode:
        n_subset = 100000
        print(f"TEST MODE: using {n_subset} samples")
        subset_indices = torch.randperm(len(dataset))[:n_subset]
        dataset = Subset(dataset, subset_indices)

    train_ds, val_ds, train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, use_cache=False)

    train_set = set(train_idx.tolist())
    val_set   = set(val_idx.tolist())
    overlap = train_set & val_set
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")
    print(f"Overlap: {len(overlap)} samples")                    # should be 0
    print(f"Union covers full dataset: {len(train_set | val_set) == len(dataset)}")  # should be True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    epoch_len = len(train_loader)
    total_iters = epochs * epoch_len
    print(f"Total training iterations: {total_iters} (epochs={epochs}, epoch_len={epoch_len})")

    # ============ Model, optimizer, loss ============
    print("\nBuilding model...")
    model = DINODuneModel(backbone_name=backbone_name).to(device)
    optimizer = optim.AdamW(model.student.parameters(), lr=lr, weight_decay=weight_decay)

    masker = SparseVoxelMasker(mask_ratio=mask_ratio)
    loss_fn = PixelDINOLoss(
        loss_type=cfg.loss_type,
        center_momentum=cfg.center_momentum,
        use_centering=cfg.use_centering,
        teacher_temp=cfg.teacher_temp,
        student_temp=cfg.student_temp,
    ).to(device)

    # ============ Schedulers ============
    warmup_iters = min(warmup_epochs * epoch_len, int(0.2 * total_iters))
    print(f"Schedules: total_iters={total_iters}, warmup_iters={warmup_iters}")

    lr_schedule = CosineScheduler(
        base_value=cfg.lr,
        final_value=cfg.min_lr,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
    )
    wd_schedule = CosineScheduler(
        base_value=cfg.weight_decay,
        final_value=cfg.weight_decay_end,
        total_iters=total_iters,
    )
    momentum_schedule = CosineScheduler(
        base_value=cfg.momentum_start,
        final_value=cfg.momentum_end,
        total_iters=total_iters,
    )

    # ============ Checkpointing ============
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============ Debugging ============
    debugger = DINODebugger(cfg, enabled=True)
    debugger.log_config(cfg)

    # ============ Training loop ============
    print("\nStarting training...")
    first_batch = True

    for epoch in range(1, epochs + 1):
        model.train()

        for batch_idx, (data, _) in enumerate(train_loader):
            iteration = (epoch - 1) * epoch_len + batch_idx
            data = data.to(device)

            # Warn about empty images (all-zero pixels) — these can cause warpconvnet
            # to silently drop batch entries due to bincount trailing-zero truncation.
            empty = (data.view(data.shape[0], -1) == 0).all(dim=1)
            if empty.any():
                empty_idx = empty.nonzero(as_tuple=True)[0].tolist()
                print(f"WARNING: epoch {epoch}, batch {batch_idx}: "
                      f"{len(empty_idx)} empty image(s) at batch positions {empty_idx}")

            # Apply schedules
            lr_val = lr_schedule[iteration]
            wd_val = wd_schedule[iteration]
            mom_val = momentum_schedule[iteration]

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_val
                param_group["weight_decay"] = wd_val

            # Forward + backward
            optimizer.zero_grad()
            loss_val, s_feats, t_feats, mask_fwd = model.forward_backward(data, masker, loss_fn)
            optimizer.step()

            # EMA teacher update
            model.update_teacher(mom_val)

            # Centering: update teacher center for next iteration
            loss_fn.update_center(t_feats, data)
            debugger.log_center_stats(iteration, loss_fn)

            # Scalar logging
            n_valid = (~mask_fwd & (data != 0)).sum().item()
            debugger.log_batch(epoch, batch_idx, iteration, loss_val, n_valid, lr_val, mom_val)

            # Gradient norms per backbone module (.grad still populated before next zero_grad)
            debugger.log_gradient_norms(iteration, model.student)

            # Representation-quality statistics (variance, covariance, norm)
            debugger.log_feature_stats(iteration, s_feats, t_feats, mask_fwd, data)

            # First batch: log tensor shapes
            if first_batch:
                debugger.log_shapes(data, data, mask_fwd, s_feats, t_feats)
                first_batch = False

            # Periodically persist histories to disk
            debugger.maybe_save_histories(iteration)

            # Explicitly free feature tensors: each is ~1.9 GB on GPU.
            # Dropping them here lets CUDA reclaim memory before the next forward pass.
            del s_feats, t_feats

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(f"[{epoch}/{epochs}] iter {iteration}: loss={loss_val:.6f}, "
                      f"lr={lr_val:.2e}, mom={mom_val:.6f}")

        # Validation
        val_loss = validate_epoch(model, val_loader, masker, loss_fn, device)
        print(f"[{epoch}/{epochs}] val_loss={val_loss:.6f}")
        debugger.log_val_epoch(epoch, iteration, val_loss)

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "student": model.student.state_dict(),
                "teacher": model.teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
            }
            ckpt_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    debugger.save_histories()

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")


def from_config(
    config_path: str,
    run_name: str = "",
    device: str = "cuda",
    test_mode: bool = False,
):
    """
    Start training from a saved run_config.json file.

    Loads training parameters from a previously saved run_config.json (e.g. from
    ./dino_debug/<run_name>/run_config.json).  Any JSON field that does not match
    a parameter of main() is silently ignored, so old configs with stale or missing
    keys work without errors — missing fields fall back to main()'s defaults.

    The `run_name`, `device`, and `test_mode` arguments override the corresponding
    values from the config file.  Providing a new run_name is recommended when
    re-running a config so outputs don't overwrite the original run.

    Args:
        config_path: Path to the run_config.json file
        run_name: Override run name (determines output sub-directories)
        device: Override device ("cuda" or "cpu")
        test_mode: Override test_mode flag
    """
    with open(config_path) as f:
        raw = json.load(f)

    # Discover which parameters main() accepts (names + defaults)
    sig = inspect.signature(main)
    valid_params = set(sig.parameters)

    # Build kwargs: only keep JSON keys that main() understands
    kwargs = {k: v for k, v in raw.items() if k in valid_params}

    # The JSON stores debug_dir as the fully-nested path (debug_dir/run_name).
    # main() will re-append run_name, so we strip the suffix here to avoid
    # double-nesting.  We use the original run_name from the JSON for this,
    # before any CLI override is applied.
    orig_run_name = kwargs.get("run_name", "")
    stored_debug_dir = kwargs.get("debug_dir", "")
    if orig_run_name and stored_debug_dir.endswith("/" + orig_run_name):
        kwargs["debug_dir"] = stored_debug_dir[: -len("/" + orig_run_name)]

    # CLI-level overrides always win
    if run_name:
        kwargs["run_name"] = run_name
    kwargs["device"] = device
    kwargs["test_mode"] = test_mode

    main(**kwargs)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "from_config":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        fire.Fire(from_config)
    else:
        fire.Fire(main)
