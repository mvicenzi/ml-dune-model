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

    Two loss modes:
    - "cosine": 1 - cosine_similarity (works well with normalized features)
    - "mse": mean squared error (for unnormalized features)
    """

    def __init__(self, loss_type: str = "cosine", center_momentum: float = 0.9):
        """
        Args:
            loss_type: "cosine" or "mse"
            center_momentum: EMA decay for teacher center (default 0.9, same as DINOv2)
        """
        super().__init__()
        assert loss_type in ("cosine", "mse"), f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type
        self.center_momentum = center_momentum
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

        # Reshape features to [B*H*W, D] and select valid positions
        # Permute to [B, H, W, D] then gather at valid mask positions
        student_flat = student_feats.permute(0, 2, 3, 1)[valid]  # [N_valid, D]
        teacher_flat = teacher_feats.permute(0, 2, 3, 1)[valid].detach()

        # Handle empty case
        if student_flat.shape[0] == 0:
            return torch.tensor(0.0, device=student_feats.device, dtype=student_feats.dtype)

        # Lazy-initialize center on first forward call
        if self.center is None:
            self.center = torch.zeros(
                teacher_flat.shape[-1],
                device=teacher_flat.device,
                dtype=teacher_flat.dtype,
            )

        # Teacher centering: remove the running mean direction before loss computation
        teacher_flat = teacher_flat - self.center

        # Compute loss
        if self.loss_type == "cosine":
            # Cosine loss: 1 - similarity (similarity ranges [-1, 1], we want it close to 1)
            loss = 1.0 - F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        else:  # mse
            # MSE between unnormalized features
            loss = F.mse_loss(student_flat, teacher_flat, reduction="none").mean(dim=-1)

        return loss.mean()

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
