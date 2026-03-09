"""Debug utilities: logging and history tracking for DINO training."""

import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor


class DINODebugger:
    """
    Conditional debugging for DINO training.

    Features (only active if cfg.debug=True):

    File logging (training.log):
    - Config summary at startup
    - Tensor shapes on first batch
    - Per-batch scalars: loss, n_valid, lr, momentum  [every batch]
    - Feature statistics: variance, covariance, L2 norm  [every debug_every]
    - Gradient norms per backbone module group  [every debug_every]

    History file (histories.json):
    - loss:   [float, ...]                      per-batch train loss
    - val:    {iter: [...], loss: [...]}         per-epoch val loss
    - stats:  {iter: [...], s_var: [...], ...}  feature statistics
    - grad:   {module: {iter: [...], norm: [...]}, ...}
    """

    def __init__(self, cfg, enabled: bool = True):
        self.enabled = enabled and cfg.debug
        self.debug_every = cfg.debug_every
        self.debug_dir = Path(cfg.debug_dir) if self.enabled else None
        self.logger = None
        self.loss_history = [] if self.enabled else None

        # Histories for offline plotting
        self.stats_history = (
            {"iter": [],
             "s_norm_min": [], "s_norm_max": [], "s_norm_median": [],
             "t_norm_min": [], "t_norm_max": [], "t_norm_median": [],
             "s_cov_mat": [], "t_cov_mat": []}
            if self.enabled else None
        )
        # grad_history: module_group -> {"iter": [...], "norm": [...]}
        self.grad_history = {} if self.enabled else None
        # val_history: iteration index at end of each epoch -> val loss
        self.val_history = {"iter": [], "loss": []} if self.enabled else None

        if not self.enabled:
            return

        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("dino_debug")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.debug_dir / "training.log")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.logger.addHandler(handler)

        # Overwrite any stale histories from a previous run immediately
        self.save_histories()

    # ------------------------------------------------------------------
    # Startup / one-time methods
    # ------------------------------------------------------------------

    def log_config(self, cfg):
        """Log config summary and save run_config.json for experiment tracking."""
        if not self.enabled or self.logger is None:
            return
        self.logger.info(
            f"Config: backbone={cfg.backbone_name}, mask_ratio={cfg.mask_ratio}, "
            f"loss_type={cfg.loss_type}, lr={cfg.lr}, epochs={cfg.epochs}"
        )
        config_dict = {
            "timestamp": datetime.now().isoformat(),
            "run_name": getattr(cfg, "run_name", ""),
            **dataclasses.asdict(cfg),
        }
        with open(self.debug_dir / "run_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def log_shapes(self, x, x_student, mask, s_feats, t_feats):
        """Log tensor shapes on first batch."""
        if not self.enabled or self.logger is None:
            return
        self.logger.info(
            f"Shapes: x={tuple(x.shape)}, x_student={tuple(x_student.shape)}, "
            f"mask={tuple(mask.shape)}, s_feats={tuple(s_feats.shape)}, "
            f"t_feats={tuple(t_feats.shape)}"
        )

    # ------------------------------------------------------------------
    # Per-batch logging
    # ------------------------------------------------------------------

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
        """Log per-batch scalar information (every batch)."""
        if not self.enabled or self.logger is None:
            return
        self.logger.info(
            f"[epoch {epoch:3d} batch {batch_idx:4d} iter {iteration:6d}] "
            f"loss={loss:.6f} n_valid={n_valid} lr={lr:.2e} momentum={momentum:.6f}"
        )
        if self.loss_history is not None:
            self.loss_history.append(loss)

    def log_val_epoch(self, epoch: int, iteration: int, val_loss: float):
        """
        Record end-of-epoch validation loss.

        The iteration passed should be the last training iteration of that epoch so
        the val point lands at the right x position on the shared train/val plot.
        """
        if not self.enabled or self.logger is None:
            return
        self.logger.info(
            f"[epoch {epoch:3d} iter {iteration:6d}] VAL_LOSS: {val_loss:.6f}"
        )
        self.val_history["iter"].append(iteration)
        self.val_history["loss"].append(val_loss)

    def save_histories(self):
        """
        Persist all in-memory histories to histories.json for offline analysis/plotting.

        JSON structure:
          loss:   [float, ...]                      per-batch train loss
          val:    {iter: [...], loss: [...]}         per-epoch val loss
          stats:  {iter: [...], s_var: [...], ...}  feature statistics
          grad:   {module: {iter: [...], norm: [...]}, ...}
        """
        if not self.enabled:
            return
        data = {
            "loss":  self.loss_history or [],
            "val":   self.val_history  or {},
            "stats": self.stats_history or {},
            "grad":  self.grad_history  or {},
        }
        try:
            with open(self.debug_dir / "histories.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving histories: {e}")

    def log_feature_stats(
        self,
        iteration: int,
        s_feats: Tensor,
        t_feats: Tensor,
        mask: Tensor,
        x: Tensor,
    ):
        """
        Compute and log representation-quality statistics at valid pixels.

        Valid pixels = active (non-zero in original image) AND unmasked
        (positions the student actually processed).

        Metrics:
        - Mean feature variance across channels: low → dimensional collapse
        - Participation ratio (effective rank): low → few dominant channels, rest collapsed
        - Mean |off-diagonal covariance|: high → redundant / correlated features
        - Mean L2 feature norm: high → potential divergence

        Computed for both student and teacher. Runs every `debug_every` iterations.
        Full covariance matrices (64×64) are saved to history for offline heatmap plotting.
        """
        if not self.enabled or self.logger is None:
            return
        if iteration % self.debug_every != 0:
            return

        with torch.no_grad():
            active = (x.squeeze(1) != 0)           # [B, H, W]
            valid = active & (~mask)                # [B, H, W]

            # [N_valid, D] — use float32 for numerical stability
            # first permute [B, D, H, W] to [B, H, W, D] 
            # then apply valid mask [B, H, W] and flatten to [N_valid, D]
            s_flat = s_feats.detach().permute(0, 2, 3, 1)[valid].float()
            t_flat = t_feats.detach().permute(0, 2, 3, 1)[valid].float()

            if s_flat.shape[0] < 2:
                return

            # make [N_valid, D] into [D, N_valid] with .T
            s_cov_mat = torch.cov(s_flat.T)   # [D, D]
            t_cov_mat = torch.cov(t_flat.T)

            # L2 norm of the feature vector at each valid pixel [N_valid]
            s_norms = s_flat.norm(dim=-1)
            t_norms = t_flat.norm(dim=-1)

        self.logger.info(
            f"[iter {iteration:6d}] FEAT_STATS: "
            f"s_norm={s_norms.mean():.4f} t_norm={t_norms.mean():.4f}"
        )

        h = self.stats_history
        h["iter"].append(iteration)
        h["s_norm_min"].append(s_norms.min().item())
        h["s_norm_max"].append(s_norms.max().item())
        h["s_norm_median"].append(s_norms.median().item())
        h["t_norm_min"].append(t_norms.min().item())
        h["t_norm_max"].append(t_norms.max().item())
        h["t_norm_median"].append(t_norms.median().item())
        h["s_cov_mat"].append(s_cov_mat.cpu().tolist())
        h["t_cov_mat"].append(t_cov_mat.cpu().tolist())

    def log_gradient_norms(self, iteration: int, student: torch.nn.Module):
        """
        Log gradient norms for each top-level backbone module.

        After loss.backward() the .grad tensors are populated; this method groups
        named parameters by their first path token (e.g. 'conv0', 'bottleneck',
        'block6') and logs the mean gradient norm per group.

        Useful for diagnosing vanishing / exploding gradients at different network
        depths. Runs every `debug_every` iterations.
        """
        if not self.enabled or self.logger is None:
            return
        if iteration % self.debug_every != 0:
            return

        has_grad = False
        for name, param in student.named_parameters():
            if param.grad is None:
                continue
            has_grad = True
            group = name.split(".")[0]
            suffix = ".".join(name.split(".")[1:]) or name
            norm = param.grad.detach().norm().item()
            entry = self.grad_history.setdefault(group, {}).setdefault(suffix, {"iter": [], "norm": []})
            entry["iter"].append(iteration)
            entry["norm"].append(norm)

        if not has_grad:
            return

        msg = "  ".join(
            f"{g}.{s}={data['norm'][-1]:.3e}"
            for g, params in sorted(self.grad_history.items())
            for s, data in params.items()
            if data["iter"] and data["iter"][-1] == iteration
        )
        self.logger.info(f"[iter {iteration:6d}] GRAD_NORMS: {msg}")

    def maybe_save_histories(self, iteration: int):
        """Persist histories to disk every `debug_every` iterations."""
        if not self.enabled:
            return
        if iteration % self.debug_every == 0:
            try:
                self.save_histories()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error saving histories: {e}")
