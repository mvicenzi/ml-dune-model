"""DINO training model: student + teacher with EMA update."""

import torch
import torch.nn as nn
from torch import Tensor

from models import BACKBONE_REGISTRY
from models.blocks import DenseInput


class DINODuneModel(nn.Module):
    """
    DINO-style teacher/student framework for DUNE backbone.

    - Student: trainable backbone, receives masked input
    - Teacher: frozen backbone, receives full input, EMA-updated from student
    - Both share the same architecture; training updates only student

    Key: teacher parameters are NOT updated by gradients, only by explicit EMA update.
    """

    def __init__(self, backbone_name: str = "attn_default"):
        """
        Initialize student and teacher backbones.

        Args:
            backbone_name: Key into BACKBONE_REGISTRY (e.g., "attn_default", "base")
        """
        super().__init__()

        ## FIXME FIXME: TEMPORARY
        self.from_dense = DenseInput()

        # Instantiate both backbones (sparse: Voxels → Voxels)
        backbone_cls = BACKBONE_REGISTRY[backbone_name]
        print("Initializing STUDENT backbone:")
        self.student = backbone_cls()
        print("Initializing TEACHER backbone:")
        self.teacher = backbone_cls()

        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())

        # Freeze teacher: no gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Teacher always in eval mode (batchnorm, dropout, etc.)
        self.teacher.eval()

    def train(self, mode: bool = True):
        """Override train() to keep teacher in eval mode."""
        super().train(mode)
        self.teacher.eval()  # Teacher always stays in eval
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """
        EMA update: teacher = momentum * teacher + (1 - momentum) * student

        Args:
            momentum: EMA momentum (typically 0.996 → 0.9999 over training)
        """
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data.mul_(momentum).add_((1.0 - momentum) * s_param.data)

    def forward_backward(self, x: Tensor, masker, loss_fn):
        """
        Forward pass and backward update.

        Args:
            x: [B, 1, H, W] dense image tensor (from the dense dataloader)
            masker: SparseVoxelMasker instance
            loss_fn: PixelDINOLoss instance

        Returns:
            loss_value:      scalar loss
            teacher_entropy: H(P_t) per batch (dino loss only, else None)
            kl:              KL(P_t||P_s) per batch (dino loss only, else None)
            cov_penalty:     covariance penalty (if enabled, else None)
            student_out:     student Voxels output
            teacher_out:     teacher Voxels output
        """
        # FIXME FIXME: TEMPORARY
        xs = self.from_dense(x)

        # Masking on Voxels: returns reduced student Voxels + kept_indices
        xs_student, kept_indices = masker(xs)

        # Teacher forward (full Voxels, frozen, no grad)
        with torch.no_grad():
            teacher_out = self.teacher(xs)

        # Student forward (masked Voxels, trainable)
        student_out = self.student(xs_student)

        # Compute loss and backprop
        loss, teacher_entropy, kl, cov_penalty = loss_fn(student_out, teacher_out, kept_indices)
        loss.backward()

        return loss.item(), teacher_entropy, kl, cov_penalty, student_out, teacher_out
