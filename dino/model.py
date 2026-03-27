"""DINO training model: student + teacher with EMA update."""

import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels

from models import BACKBONE_REGISTRY
from models.blocks import DenseInput
from .projhead import DINOProjectionHead


class DINODuneModel(nn.Module):
    """
    DINO-style teacher/student framework for DUNE backbone.

    - Student: trainable backbone (+ optional projection head), receives masked input
    - Teacher: frozen backbone (+ optional projection head), receives full input, EMA-updated from student
    - Both share the same architecture; training updates only student

    Key: teacher parameters are NOT updated by gradients, only by explicit EMA update.
    """

    def __init__(
        self,
        backbone_name: str = "attn_default",
        use_proj_head: bool = False,
        proj_head_hidden_dim: int = 256,
        proj_head_output_dim: int = 256,
        proj_head_n_layers: int = 4,
    ):
        """
        Args:
            backbone_name:        Key into BACKBONE_REGISTRY (e.g., "attn_default", "base")
            use_proj_head:        Attach a DINO MLP projection head after the backbone
            proj_head_hidden_dim: Inner MLP width (DINO paper uses 2048)
            proj_head_output_dim: Output dimension of the final FC layer
            proj_head_n_layers:   Number of MLP layers before the final FC
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

        # Optional projection heads — teacher head is the EMA of the student head,
        # exactly as in the original DINO (head is part of the full student network).
        if use_proj_head:
            in_dim = 64  # backbone output channels — must match architecture
            self.student_head = DINOProjectionHead(in_dim, proj_head_hidden_dim, proj_head_output_dim, proj_head_n_layers)
            self.teacher_head = DINOProjectionHead(in_dim, proj_head_hidden_dim, proj_head_output_dim, proj_head_n_layers)
            self.teacher_head.load_state_dict(self.student_head.state_dict())
        else:
            self.student_head = None
            self.teacher_head = None

        # Freeze teacher backbone and head: no gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        if self.teacher_head is not None:
            for p in self.teacher_head.parameters():
                p.requires_grad = False

        # Teacher always in eval mode (batchnorm, dropout, etc.)
        self.teacher.eval()
        if self.teacher_head is not None:
            self.teacher_head.eval()

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
        EMA update: teacher = momentum * teacher + (1 - momentum) * student

        Covers both backbone and projection head (if present), mirroring the
        original DINO where the full student (backbone + head) is momentum-encoded.

        Args:
            momentum: EMA momentum (typically 0.996 → 0.9999 over training)
        """
        pairs = list(zip(self.student.parameters(), self.teacher.parameters()))
        if self.student_head is not None:
            pairs += list(zip(self.student_head.parameters(), self.teacher_head.parameters()))
        for s_param, t_param in pairs:
            t_param.data.mul_(momentum).add_((1.0 - momentum) * s_param.data)

    def encode_teacher(self, xs: Voxels) -> Voxels:
        """Run the teacher backbone (+ head if present). Always called in no_grad context."""
        out = self.teacher(xs)
        if self.teacher_head is not None:
            out = self.teacher_head(out)
        return out

    def encode_student(self, xs: Voxels) -> tuple[Voxels, Voxels]:
        """
        Run the student backbone (+ head if present).

        Returns:
            backbone_out: raw 64-dim backbone output (before head)
            final_out:    head output if head present, else same as backbone_out
        """
        backbone_out = self.student(xs)
        if self.student_head is not None:
            return backbone_out, self.student_head(backbone_out)
        return backbone_out, backbone_out

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
            student_entropy: H(P_s) per batch (dino loss only, else None)
            kl:              KL(P_t||P_s) per batch (dino loss only, else None)
            cov_penalty:     covariance penalty (if enabled, else None)
            student_out:     student Voxels output (after head if present)
            teacher_out:     teacher Voxels output (after head if present)
        """
        # FIXME FIXME: TEMPORARY
        xs = self.from_dense(x)

        # Masking on Voxels: returns reduced student Voxels + kept_indices
        xs_student, kept_indices = masker(xs)

        # Teacher forward (full Voxels, frozen, no grad)
        with torch.no_grad():
            teacher_out = self.encode_teacher(xs)

        # Student forward (masked Voxels, trainable)
        student_backbone_out, student_out = self.encode_student(xs_student)

        # Compute loss and backprop
        loss, teacher_entropy, student_entropy, kl, cov_penalty = loss_fn(student_out, student_backbone_out, teacher_out, kept_indices)
        loss.backward()

        return loss.item(), teacher_entropy, student_entropy, kl, cov_penalty, student_out, teacher_out
