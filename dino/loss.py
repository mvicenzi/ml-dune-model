"""Per-pixel DINO-style loss for self-supervised training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    ):
        """
        Args:
            loss_type:        "cosine", "mse", or "dino"
            center_momentum:  EMA decay for the teacher center buffer (default 0.9)
            use_centering:    if True, subtract running center from teacher features before
                              computing the loss; the center buffer is always updated regardless
            teacher_temp:     softmax temperature for teacher logits (only used for "dino")
            student_temp:     softmax temperature for student logits (only used for "dino")
        """
        super().__init__()
        assert loss_type in ("cosine", "mse", "dino"), f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type
        self.center_momentum = center_momentum
        self.use_centering = use_centering
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        # Lazily initialized on first forward call once feature dim D is known.
        # register_buffer ensures it moves with .to(device) and is saved in checkpoints.
        self.register_buffer("center", None)

    def forward(
        self,
        student_feats: Tensor,  # [B, D, H, W] student output
        teacher_feats: Tensor,  # [B, D, H, W] teacher output (should be detached)
        mask: Tensor,           # [B, H, W] bool, True = masked (student didn't see)
        original_x: Tensor,     # [B, 1, H, W] original (unmodified) image
    ) -> Tensor:
        """
        Compute pixel-level DINO loss at unmasked active positions.

        Args:
            student_feats: Student backbone output [B, D, H, W]
            teacher_feats: Teacher backbone output [B, D, H, W], pre-detached
            mask: Boolean mask [B, H, W], True where pixels were masked from student
            original_x: Original image [B, 1, H, W] to identify active pixels

        Returns:
            Scalar loss value
        """
        # Identify active pixels (originally non-zero in the image)
        active = (original_x.squeeze(1) != 0)  # [B, H, W]

        # Valid positions: active AND not masked (where student computed real features)
        valid = active & (~mask)  # [B, H, W]

        counts = valid.sum(dim=(1, 2)).float()  # [B] valid pixels per image

        # Handle empty case
        if counts.sum() == 0:
            return torch.tensor(0.0, device=student_feats.device, dtype=student_feats.dtype)

        # Flatten to valid pixels only — keeps heavy ops (normalize, softmax) efficient
        # for sparse images where N_valid << B*H*W
        s = student_feats.permute(0, 2, 3, 1)[valid]           # [N_valid, D]
        t = teacher_feats.permute(0, 2, 3, 1)[valid]           # [N_valid, D]

        # Lazy-initialize center on first forward call
        if self.center is None:
            self.center = torch.zeros(
                t.shape[-1],
                device=t.device,
                dtype=t.dtype,
            )

        # Optionally subtract running mean from teacher to remove the dominant direction
        if self.use_centering:
            t = t - self.center

        # For dino loss: L2-normalize to unit sphere so scale-invariant
        if self.loss_type == "dino":
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)

        # Compute per-pixel loss [N_valid]
        if self.loss_type == "cosine":
            loss = 1.0 - F.cosine_similarity(s, t, dim=-1)
        elif self.loss_type == "mse":
            loss = F.mse_loss(s, t, reduction="none").mean(dim=-1)
        else:  # dino
            # Cross-entropy between sharpened teacher distribution and student log-probs.
            # Both student and teacher are treated as raw logits over D dimensions.
            t_prob = F.softmax(t / self.teacher_temp, dim=-1)      # [N_valid, D]
            s_logp = F.log_softmax(s / self.student_temp, dim=-1)  # [N_valid, D]
            loss = -(t_prob * s_logp).sum(dim=-1)                  # [N_valid]

        # Two-stage reduction: sum per image via scatter, divide by count, then mean.
        # Mirrors DINOv2: sum(loss * mask) / mask.sum() per image, then .mean()
        B = student_feats.shape[0]
        batch_idx = torch.where(valid)[0]  # [N_valid] — image index for each valid pixel
        per_image_loss = torch.zeros(B, device=loss.device, dtype=loss.dtype)
        per_image_loss.scatter_add_(0, batch_idx, loss)
        per_image_loss = per_image_loss / counts.clamp(min=1.0)
        return per_image_loss[counts > 0].mean()

    @torch.no_grad()
    def update_center(self, teacher_feats: Tensor, original_x: Tensor) -> None:
        """
        Update the running center with the EMA of teacher features at active positions.

        Should be called once per training batch, AFTER the loss backward and optimizer
        step. Must NOT be called during validation to avoid shifting the baseline with
        eval-mode teacher outputs.

        Args:
            teacher_feats: Teacher backbone output [B, D, H, W], detached
            original_x: Original (unmasked) image [B, 1, H, W] to identify active pixels
        """
        active = (original_x.squeeze(1) != 0)  # [B, H, W]
        teacher_flat = teacher_feats.permute(0, 2, 3, 1)[active]  # [N_active, D]

        if teacher_flat.shape[0] == 0:
            return

        # for each feature, take mean over all active pixels in the batch
        # result is [D]
        batch_mean = teacher_flat.mean(dim=0)

        if self.center is None:
            self.center = batch_mean.clone()
        else:
            self.center = self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_mean
