"""Per-pixel DINO-style loss for self-supervised training."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels


class PixelDINOLoss(nn.Module):
    """
    Per-pixel DINO-style knowledge distillation loss.

    Compares feature vectors pixel-by-pixel between student and teacher at unmasked
    active positions. Teacher (frozen, EMA updated) provides soft targets; student
    (trainable) learns to match these targets.

    Implemented on dense backbone outputs [B, D, H, W]. Loss computed only at positions
    where the student actually computed features (unmasked active pixels), not at
    structural zeros from sparse→dense conversion.

    Three loss modes:
    - "cosine": 1 - cosine_similarity (works well with normalized features)
    - "mse":    mean squared error (for unnormalized features)
    - "dino":   cross-entropy between softmax(teacher/tau_t) and log_softmax(student/tau_s);
                treats the D-dim feature vector as logits, creating per-dimension competition
                that prevents dimensional collapse
    """

    def __init__(
        self,
        loss_type: str = "cosine",
        center_momentum: float = 0.9,
        use_centering: bool = True,
        teacher_temp: float = 1.0,
        student_temp: float = 1.0,
        normalize_features: bool = True,
        use_cov_penalty: bool = False,
        cov_penalty_weight: float = 1e-3,
    ):
        """
        Args:
            loss_type:           "cosine", "mse", or "dino"
            center_momentum:     EMA decay for the teacher center buffer (default 0.9)
            use_centering:       if True, subtract running center from teacher features before
                                 computing the loss; the center buffer is always updated regardless
            teacher_temp:        softmax temperature for teacher logits (only used for "dino")
            student_temp:        softmax temperature for student logits (only used for "dino")
            normalize_features:  if True, L2-normalise student and teacher features before the
                                 loss; set to False when a projection head is used (the head's
                                 internal L2 norm already normalises the features)
            use_cov_penalty:     if True, add a VICReg-style covariance decorrelation penalty on
                                 student features to prevent dimensional collapse
            cov_penalty_weight:  scalar weight for the covariance penalty term (default 1e-3)
        """
        super().__init__()
        assert loss_type in ("cosine", "mse", "dino"), f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type
        self.center_momentum = center_momentum
        self.use_centering = use_centering
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.normalize_features = normalize_features
        self.use_cov_penalty = use_cov_penalty
        self.cov_penalty_weight = cov_penalty_weight
        # Lazily initialized on first forward call once feature dim D is known.
        # register_buffer ensures it moves with .to(device) and is saved in checkpoints.
        self.register_buffer("center", None)

    def forward(
        self,
        student_out: Voxels,          # sparse student output (after head if present)
        student_backbone: Voxels,     # raw backbone output (before head); used for cov penalty
        teacher_out: Voxels,          # sparse teacher output (should be detached)
        kept_indices: List[Tensor],   # per-batch-item indices mapping student→teacher positions
    ) -> Tensor:
        """
        Compute per-voxel DINO loss at the positions the student processed.

        For each batch item b, kept_indices[b] maps student voxel i to the
        corresponding teacher voxel: teacher[t_start + kept_indices[b][i]].

        Args:
            student_out:      Student output (Voxels) — head output if head present, else backbone
            student_backbone: Raw backbone output (Voxels) — always 64-dim; used for cov penalty
            teacher_out:      Teacher backbone output (Voxels), pre-detached
            kept_indices:     List of B index tensors from SparseVoxelMasker

        Returns:
            Scalar loss value
        """
        B      = len(student_out.offsets) - 1
        device = student_out.feature_tensor.device

        # Align student and teacher features using kept_indices
        s_parts, t_parts = [], []
        for b in range(B):
            # for each image, find pixels in student/teacher
            t_start = int(teacher_out.offsets[b])
            t_end   = int(teacher_out.offsets[b + 1])
            s_start = int(student_out.offsets[b])
            s_end   = int(student_out.offsets[b + 1])

            t_b = teacher_out.feature_tensor[t_start:t_end]           # [N_teacher, D]
            s_b = student_out.feature_tensor[s_start:s_end]           # [N_student, D]

            # select teacher pixels only if originnaly kept for the student 
            t_at_student = t_b[kept_indices[b].to(device)]            # [N_student, D]

            s_parts.append(s_b)
            t_parts.append(t_at_student)

        s = torch.cat(s_parts, dim=0)  # [N_total, D]
        t = torch.cat(t_parts, dim=0)  # [N_total, D]

        # Per-image voxel counts (from student offsets, which reflect unmasked voxels)
        counts = (student_out.offsets[1:] - student_out.offsets[:-1]).float().to(device)  # [B]

        # Lazy-initialize center on first forward call
        if self.center is None:
            self.center = torch.zeros(
                t.shape[-1],
                device=t.device,
                dtype=t.dtype,
            )

        # For dino loss: L2-normalize to unit sphere so scale-invariant
        ## TEMPORARY: only for legacy norm before centering
        #if self.loss_type == "dino":
        #    s = F.normalize(s, dim=-1)
        #    t = F.normalize(t, dim=-1)

        # Optionally subtract running mean from teacher to remove the dominant direction
        if self.use_centering:
            t = t - self.center

        # For dino loss: L2-normalize to unit sphere so scale-invariant.
        # normalize_features=False skips this when a projection head is used
        # (the head's internal L2 norm already puts features on the sphere).
        if self.loss_type == "dino" and self.normalize_features:
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)

        # teacher_entropy_px, student_entropy_px, and kl_px are only set in the dino
        # branch; initialize to None so they have a defined value for the other loss types.
        teacher_entropy_px = None
        student_entropy_px = None
        kl_px = None

        # Compute per-pixel loss [N_valid]
        if self.loss_type == "cosine":
            loss = 1.0 - F.cosine_similarity(s, t, dim=-1)
        elif self.loss_type == "mse":
            loss = F.mse_loss(s, t, reduction="none").mean(dim=-1)
        else:  # dino
            # Cross-entropy H(P_t, P_s) = H(P_t) + KL(P_t || P_s).
            # Both student and teacher are treated as raw logits over D dimensions.
            t_prob = F.softmax(t / self.teacher_temp, dim=-1)      # [N_valid, D]
            s_logp = F.log_softmax(s / self.student_temp, dim=-1)  # [N_valid, D]
            t_logp = F.log_softmax(t / self.teacher_temp, dim=-1)  # [N_valid, D]
            loss = -(t_prob * s_logp).sum(dim=-1)                  # H(P_t, P_s) [N_valid]

            # decompose loss into teacher entropy and KL divergence for diagnostics:
            s_prob = F.softmax(s / self.student_temp, dim=-1)      # [N_valid, D]
            teacher_entropy_px = -(t_prob * t_logp).sum(dim=-1)    # H(P_t)      [N_valid]
            student_entropy_px = -(s_prob * s_logp).sum(dim=-1)    # H(P_s)      [N_valid]
            kl_px = loss - teacher_entropy_px                      # KL(P_t|P_s)[N_valid]

        # Optional covariance decorrelation penalty on raw backbone features (64-dim).
        # Computed before the projection head and before L2-normalization so the
        # penalty sees the actual feature scale and covariance structure.
        cov_penalty = None
        if self.use_cov_penalty:
            cov_penalty = self._cov_penalty(student_backbone.feature_tensor)

        # Two-stage reduction: sum per image via scatter, divide by count, then mean.
        # Mirrors DINOv2: sum(loss * mask) / mask.sum() per image, then .mean()
        student_counts = (student_out.offsets[1:] - student_out.offsets[:-1]).to(device)
        batch_idx = torch.repeat_interleave(torch.arange(B, device=device), student_counts)
        per_image_loss = torch.zeros(B, device=loss.device, dtype=loss.dtype)
        per_image_loss.scatter_add_(0, batch_idx, loss)
        per_image_loss = per_image_loss / counts.clamp(min=1.0)
        scalar_loss = per_image_loss[counts > 0].mean()

        if self.use_cov_penalty:
            scalar_loss = scalar_loss + self.cov_penalty_weight * cov_penalty

        cov_penalty_item = cov_penalty.item() if cov_penalty is not None else None

        if teacher_entropy_px is not None:
            per_image_t_ent = torch.zeros(B, device=loss.device, dtype=loss.dtype)
            per_image_t_ent.scatter_add_(0, batch_idx, teacher_entropy_px)
            t_ent = (per_image_t_ent / counts.clamp(min=1.0))[counts > 0].mean().item()

            per_image_s_ent = torch.zeros(B, device=loss.device, dtype=loss.dtype)
            per_image_s_ent.scatter_add_(0, batch_idx, student_entropy_px)
            s_ent = (per_image_s_ent / counts.clamp(min=1.0))[counts > 0].mean().item()

            per_image_kl = torch.zeros(B, device=loss.device, dtype=loss.dtype)
            per_image_kl.scatter_add_(0, batch_idx, kl_px)
            kl = (per_image_kl / counts.clamp(min=1.0))[counts > 0].mean().item()
        else:
            t_ent = None
            s_ent = None
            kl = None

        return scalar_loss, t_ent, s_ent, kl, cov_penalty_item

    def _cov_penalty(self, s: Tensor) -> Tensor:
        """
        VICReg-style covariance decorrelation penalty.

        Penalizes off-diagonal entries of the feature covariance matrix, pushing
        each pair of dimensions to be uncorrelated and thus spread information
        across the full feature space.

        Args:
            s: Raw backbone features [N, D] (valid pixels only, before head and L2-norm)

        Returns:
            Scalar penalty: sum of squared off-diagonal covariance entries, divided by D
        """
        N, D = s.shape
        if N < 2:
            return s.new_tensor(0.0)
        z = s - s.mean(dim=0)                   # center features
        C = (z.T @ z) / (N - 1)                 # [D, D] covariance matrix
        # penalize only off-diagonal entries
        off_diag_sq = C.pow(2).sum() - C.diagonal().pow(2).sum()
        return off_diag_sq / D

    @torch.no_grad()
    def update_center(self, teacher_out: Voxels) -> None:
        """
        Update the running center with the EMA of teacher features at active positions.

        Should be called once per training batch, AFTER the loss backward and optimizer
        step. Must NOT be called during validation to avoid shifting the baseline with
        eval-mode teacher outputs.

        Args:
            teacher_out: Teacher backbone output (Voxels); all voxels are active by definition
        """
        teacher_flat = teacher_out.feature_tensor  # [N_active, D]

        if teacher_flat.shape[0] == 0:
            return

        # For dino loss: normalize to unit sphere before computing the center,
        # consistent with the normalization applied in forward()
        ### TEMPORARY: only for legacy norm before centering
        #if self.loss_type == "dino":
        #    teacher_flat = F.normalize(teacher_flat, dim=-1)

        batch_mean = teacher_flat.mean(dim=0)  # [D]

        if self.center is None:
            self.center = batch_mean.clone()
        else:
            self.center = self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_mean
