"""Debug utilities: logging and visualization for DINO training."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DINODebugger:
    """
    Conditional debugging and visualization for DINO training.

    Features (only active if cfg.debug=True):
    - File logging: shapes, loss, momentum, lr to training.log
    - PNG visualizations:
      - mask_sample_iter{N}.png: original image, masked image, binary mask
      - features_iter{N}.png: first 4 teacher vs student feature channels
      - loss_curve.png: running loss history
    """

    def __init__(self, cfg, enabled: bool = True):
        """
        Initialize debugger.

        Args:
            cfg: DINOConfig instance
            enabled: if False, all methods become no-ops
        """
        self.enabled = enabled and cfg.debug
        self.debug_every = cfg.debug_every
        self.debug_dir = Path(cfg.debug_dir) if self.enabled else None
        self.logger = None
        self.loss_history = [] if self.enabled else None

        if not self.enabled:
            return

        if not HAS_MATPLOTLIB:
            print("WARNING: matplotlib not available, PNG visualizations disabled")

        # Create debug directory
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logger
        self.logger = logging.getLogger("dino_debug")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.debug_dir / "training.log")
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_config(self, cfg):
        """Log config summary at training start."""
        if not self.enabled or self.logger is None:
            return
        self.logger.info(f"Config: backbone={cfg.backbone_name}, "
                        f"mask_ratio={cfg.mask_ratio}, loss_type={cfg.loss_type}, "
                        f"lr={cfg.lr}, epochs={cfg.epochs}")

    def log_shapes(self, x, x_student, mask, s_feats, t_feats):
        """Log tensor shapes on first batch."""
        if not self.enabled or self.logger is None:
            return
        self.logger.info(
            f"Shapes: x={tuple(x.shape)}, x_student={tuple(x_student.shape)}, "
            f"mask={tuple(mask.shape)}, s_feats={tuple(s_feats.shape)}, "
            f"t_feats={tuple(t_feats.shape)}"
        )

    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        iteration: int,
        loss: float,
        n_valid: int,
        lr: float,
        momentum: float,
    ):
        """Log per-batch scalar information."""
        if not self.enabled or self.logger is None:
            return
        self.logger.info(
            f"[epoch {epoch:3d} batch {batch_idx:4d} iter {iteration:6d}] "
            f"loss={loss:.6f} n_valid={n_valid} lr={lr:.2e} momentum={momentum:.6f}"
        )
        if self.loss_history is not None:
            self.loss_history.append(loss)

    def maybe_save_visuals(
        self,
        iteration: int,
        x: Tensor,
        x_student: Tensor,
        mask: Tensor,
        s_feats: Tensor,
        t_feats: Tensor,
    ):
        """
        Conditionally save PNG visualizations every `debug_every` iterations.

        Args:
            iteration: current iteration number
            x: original dense image [B, 1, H, W]
            x_student: masked dense image [B, 1, H, W]
            mask: boolean mask [B, H, W]
            s_feats: student features [B, D, H, W]
            t_feats: teacher features [B, D, H, W]
        """
        if not self.enabled or not HAS_MATPLOTLIB:
            return
        if iteration % self.debug_every != 0:
            return

        try:
            # Visualize masks
            if iteration < 10:
                self._save_mask_viz(iteration, x, x_student, mask)
            # Visualize features
            self._save_feature_viz(iteration, x, s_feats, t_feats)
            # Update loss curve
            self._save_loss_curve(iteration)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving visualizations: {e}")

    def _save_mask_viz(self, iteration: int, x: Tensor, x_student: Tensor, mask: Tensor):
        """Save 3-panel visualization: original | masked | binary mask."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Sample from batch (index 0); transpose so wire=x, time tick=y
        x_np = x[0, 0].cpu().detach().numpy().T          # [W, H]
        x_student_np = x_student[0, 0].cpu().detach().numpy().T  # [W, H]
        mask_np = mask[0].cpu().detach().numpy().T        # [W, H]

        im0 = axes[0].imshow(x_np, interpolation="none", cmap="twilight")
        axes[0].set_title("Original Image")
        axes[0].set_xlabel("Wire")
        axes[0].set_ylabel("Time tick")
        fig.colorbar(im0, ax=axes[0], label="ADC")

        im1 = axes[1].imshow(x_student_np, interpolation="none", cmap="twilight")
        axes[1].set_title("Masked Image (Student Input)")
        axes[1].set_xlabel("Wire")
        axes[1].set_ylabel("Time tick")
        fig.colorbar(im1, ax=axes[1], label="ADC")

        im2 = axes[2].imshow(mask_np, interpolation="none", cmap="binary")
        axes[2].set_title("Mask (True = Masked)")
        axes[2].set_xlabel("Wire")
        axes[2].set_ylabel("Time tick")
        fig.colorbar(im2, ax=axes[2], label="Masked")

        plt.tight_layout()
        save_path = self.debug_dir / f"mask_sample_iter{iteration:06d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

    def _save_feature_viz(self, iteration: int, x: Tensor, s_feats: Tensor, t_feats: Tensor):
        """Save 2×4 grid: first 4 teacher channels (top) vs first 4 student channels (bottom).

        Inactive pixel positions (where original image is zero) are shown as NaN,
        so only active detector pixels are colored — matches post_training_visualize.py style.
        One shared colorbar per row.
        """
        n_ch = 4
        fig, axes = plt.subplots(2, n_ch, figsize=(n_ch * 4, 10), constrained_layout=True)

        # Active pixel mask from original image: inactive → NaN (batch index 0)
        active = (x[0, 0].cpu().detach().numpy() != 0)  # [H, W]

        s_np = s_feats[0].cpu().detach().numpy()  # [D, H, W]
        t_np = t_feats[0].cpu().detach().numpy()  # [D, H, W]

        last_t = last_s = None

        for i in range(n_ch):
            # Teacher channels (top row)
            ch_t = t_np[i % t_np.shape[0]]
            img_t = np.full_like(ch_t, np.nan, dtype=float)
            img_t[active] = ch_t[active]
            last_t = axes[0, i].imshow(img_t.T, interpolation="none", cmap="viridis")
            axes[0, i].set_title(f"Teacher Ch{i}")
            axes[0, i].set_xlabel("Wire")
            axes[0, i].set_ylabel("Time tick")

            # Student channels (bottom row)
            ch_s = s_np[i % s_np.shape[0]]
            img_s = np.full_like(ch_s, np.nan, dtype=float)
            img_s[active] = ch_s[active]
            last_s = axes[1, i].imshow(img_s.T, interpolation="none", cmap="viridis")
            axes[1, i].set_title(f"Student Ch{i}")
            axes[1, i].set_xlabel("Wire")
            axes[1, i].set_ylabel("Time tick")

        # One shared colorbar per row
        if last_t is not None:
            fig.colorbar(last_t, ax=axes[0, :], label="Activation", fraction=0.02, pad=0.02)
        if last_s is not None:
            fig.colorbar(last_s, ax=axes[1, :], label="Activation", fraction=0.02, pad=0.02)

        fig.suptitle(f"Feature Maps — iter {iteration}", fontsize=14)
        save_path = self.debug_dir / f"features_iter{iteration:06d}.png"
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _save_loss_curve(self, iteration: int):
        """Update and save running loss curve."""
        if not self.loss_history or len(self.loss_history) == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.loss_history, linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss (Running)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.debug_dir / "loss_curve.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
