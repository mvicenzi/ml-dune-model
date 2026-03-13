"""DINO training model: student + teacher with EMA update."""

import torch
import torch.nn as nn
from torch import Tensor

from models import BACKBONE_REGISTRY
from .head import DINOHead


class DINODuneModel(nn.Module):
    """
    DINO-style teacher/student framework for DUNE backbone.

    - Student: trainable backbone, receives masked input
    - Teacher: frozen backbone, receives full input, EMA-updated from student
    - Both share the same architecture; training updates only student

    Key: teacher parameters are NOT updated by gradients, only by explicit EMA update.
    """

    def __init__(
        self,
        backbone_name: str = "attn_default",
        use_proj_head: bool = False,
        proj_in_dim: int = 64,
        proj_out_dim: int = 256,
        proj_hidden_dim: int = 256,
        proj_bottleneck_dim: int = 128,
        proj_nlayers: int = 2,
    ):
        """
        Initialize student and teacher backbones, and optionally projection heads.

        Args:
            backbone_name:       Key into BACKBONE_REGISTRY (e.g., "attn_default", "base")
            use_proj_head:       If True, attach a DINOHead to both student and teacher.
                                 The head maps [B, D, H, W] → [B, out_dim, H, W] per pixel,
                                 expanding the prototype space seen by the loss.
            proj_in_dim:         Backbone output channels; must match feature_dim in config.
            proj_out_dim:        Prototype dimension output by the head (>> proj_in_dim).
            proj_hidden_dim:     Hidden layer width inside the head MLP.
            proj_bottleneck_dim: MLP output dim before the weight-norm projection layer.
            proj_nlayers:        Number of Linear→GELU blocks in the head MLP.
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

        # Optional projection heads (student trainable, teacher frozen + EMA-updated)
        if use_proj_head:
            print(f"Initializing projection heads: {proj_in_dim} → {proj_out_dim} "
                  f"(hidden={proj_hidden_dim}, bottleneck={proj_bottleneck_dim}, nlayers={proj_nlayers})")
            self.student_head = DINOHead(
                in_dim=proj_in_dim,
                out_dim=proj_out_dim,
                hidden_dim=proj_hidden_dim,
                bottleneck_dim=proj_bottleneck_dim,
                nlayers=proj_nlayers,
            )
            self.teacher_head = DINOHead(
                in_dim=proj_in_dim,
                out_dim=proj_out_dim,
                hidden_dim=proj_hidden_dim,
                bottleneck_dim=proj_bottleneck_dim,
                nlayers=proj_nlayers,
            )
            self.teacher_head.load_state_dict(self.student_head.state_dict())
            for p in self.teacher_head.parameters():
                p.requires_grad = False
            self.teacher_head.eval()
        else:
            self.student_head = None
            self.teacher_head = None

    def train(self, mode: bool = True):
        """Override train() to keep teacher (and teacher head) in eval mode."""
        super().train(mode)
        self.teacher.eval()
        if self.teacher_head is not None:
            self.teacher_head.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """
        EMA update: teacher = momentum * teacher + (1 - momentum) * student.
        Applied to both backbone and projection head (if present).

        Args:
            momentum: EMA momentum (typically 0.996 → 0.9999 over training)
        """
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data.mul_(momentum).add_((1.0 - momentum) * s_param.data)
        if self.student_head is not None:
            for s_param, t_param in zip(
                self.student_head.parameters(), self.teacher_head.parameters()
            ):
                t_param.data.mul_(momentum).add_((1.0 - momentum) * s_param.data)

    @staticmethod
    def _apply_head(head: DINOHead, feats: Tensor) -> Tensor:
        """
        Apply a projection head to a spatial feature map.

        Permutes [B, D, H, W] → [B, H, W, D] so nn.Linear operates on the
        feature dimension, then permutes back to [B, out_dim, H, W].

        Args:
            head:  DINOHead instance
            feats: [B, D, H, W] backbone output

        Returns:
            [B, out_dim, H, W] projected features
        """
        x = feats.permute(0, 2, 3, 1)   # [B, H, W, D]
        x = head(x)                      # [B, H, W, out_dim]
        return x.permute(0, 3, 1, 2)    # [B, out_dim, H, W]

    @torch.no_grad()
    def encode_teacher(self, x: Tensor) -> Tensor:
        """Teacher forward pass (backbone + optional head), no gradients."""
        feats = self.teacher(x)
        if self.teacher_head is not None:
            feats = self._apply_head(self.teacher_head, feats)
        return feats

    def encode_student(self, x: Tensor) -> Tensor:
        """Student forward pass (backbone + optional head)."""
        feats = self.student(x)
        if self.student_head is not None:
            feats = self._apply_head(self.student_head, feats)
        return feats

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
            teacher_feats = self.encode_teacher(x)   # [B, D, H, W] or [B, out_dim, H, W]

        # Student forward (masked image, trainable)
        student_feats = self.encode_student(x_student)  # same shape as teacher_feats

        # Compute loss and backprop
        loss = loss_fn(student_feats, teacher_feats, mask, x)
        loss.backward()

        return loss.item(), student_feats.detach(), teacher_feats.detach(), mask
