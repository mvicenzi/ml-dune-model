"""DINO training model: student + teacher with EMA update."""

import torch
import torch.nn as nn
from torch import Tensor

from models import BACKBONE_REGISTRY


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

        # Instantiate both backbones
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
            x: [B, 1, 500, 500] dense image
            masker: SparseVoxelMasker instance
            loss_fn: PixelDINOLoss instance

        Returns:
            loss_value: scalar loss
            student_feats: student output (detached)
            teacher_feats: teacher output (from no_grad context)
            mask: mask applied to student input
        """
        # Apply masking to create student input
        x_student, mask = masker(x)

        # Teacher forward (full image, frozen, no grad)
        with torch.no_grad():
            teacher_feats = self.teacher(x)  # [B, D, H, W]

        # Student forward (masked image, trainable)
        student_feats = self.student(x_student)  # [B, D, H, W]

        # Compute loss and backprop
        loss, teacher_entropy, kl = loss_fn(student_feats, teacher_feats, mask, x)
        loss.backward()

        return loss.item(), teacher_entropy, kl, student_feats.detach(), teacher_feats.detach(), mask
