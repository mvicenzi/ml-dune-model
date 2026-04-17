"""DINO training model: student + teacher with EMA update."""

import inspect
import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels

from models import BACKBONE_REGISTRY
from .projhead import DINOProjectionHead


def match_and_gather(
    s_out: Voxels,
    s_backbone: Voxels,
    t_out: Voxels,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Match student and teacher output voxels by spatial coordinates and
    return pre-aligned feature tensors ready for the loss.

    Fully vectorized — no Python loop over batch items.  Batch items are kept
    separate by encoding the batch index into the flat coordinate key:
    ``batch_idx * HW + y * W + x``.  One sort + one searchsorted handles
    all batch items at once.

    Args:
        s_out:      Student head output (Voxels).
        s_backbone: Student backbone output (Voxels), same coordinates as s_out.
        t_out:      Teacher head output (Voxels).

    Returns:
        s_feats:    [N_matched, D_head]  student head features at intersection
        s_bb_feats: [N_matched, D_bb]    student backbone features at intersection
        t_feats:    [N_matched, D_head]  teacher head features at intersection
        counts:     [B] int64 per-image matched voxel counts
    """
    B = len(s_out.offsets) - 1
    device = s_out.feature_tensor.device
    N_s = s_out.coordinate_tensor.shape[0]
    N_t = t_out.coordinate_tensor.shape[0]

    # Early exit when either side is empty
    if N_s == 0 or N_t == 0:
        D_head = s_out.feature_tensor.shape[1]
        D_bb   = s_backbone.feature_tensor.shape[1]
        return (s_out.feature_tensor.new_zeros(0, D_head),
                s_backbone.feature_tensor.new_zeros(0, D_bb),
                t_out.feature_tensor.new_zeros(0, D_head),
                torch.zeros(B, dtype=torch.int64, device=device))

    # --- Build unique flat keys: batch_idx * HW + y * W + x ---------------
    # W must exceed the max x-coord so that y * W + x is unique per pixel.
    # HW = (max_y + 1) * W is the key range per batch item; prepending
    # batch_idx * HW guarantees no collisions across batch items.
    W = max(int(s_out.coordinate_tensor[:, 0].max().item()),
            int(t_out.coordinate_tensor[:, 0].max().item())) + 1
    HW = (max(int(s_out.coordinate_tensor[:, 1].max().item()),
              int(t_out.coordinate_tensor[:, 1].max().item())) + 1) * W

    # Expand per-item counts into per-voxel batch indices
    s_counts = (s_out.offsets[1:] - s_out.offsets[:-1]).to(device)
    t_counts = (t_out.offsets[1:] - t_out.offsets[:-1]).to(device)
    s_batch = torch.repeat_interleave(torch.arange(B, device=device), s_counts)
    t_batch = torch.repeat_interleave(torch.arange(B, device=device), t_counts)

    # Batch-prefixed flat keys — prevents cross-batch matching
    s_keys = s_batch * HW + s_out.coordinate_tensor[:, 1].long() * W + s_out.coordinate_tensor[:, 0].long()
    t_keys = t_batch * HW + t_out.coordinate_tensor[:, 1].long() * W + t_out.coordinate_tensor[:, 0].long()

    # --- Sort teacher keys, binary-search student keys into them -----------
    t_sorted, t_order = t_keys.sort()
    pos = torch.searchsorted(t_sorted, s_keys)
    pos = pos.clamp(max=N_t - 1)           # clamp for safe indexing
    valid = t_sorted[pos] == s_keys         # exact match required

    # --- Gather matched features ------------------------------------------
    # s_idx: global indices of matched student voxels in the flat feature tensor
    # t_idx: corresponding teacher indices (via t_order to undo the sort)
    s_idx = valid.nonzero(as_tuple=False).squeeze(1)
    t_idx = t_order[pos[valid]]

    s_feats    = s_out.feature_tensor[s_idx]
    s_bb_feats = s_backbone.feature_tensor[s_idx]
    t_feats    = t_out.feature_tensor[t_idx]

    # --- Per-image matched counts via scatter_add -------------------------
    counts = torch.zeros(B, dtype=torch.int64, device=device)
    if s_idx.numel() > 0:
        matched_batch = s_batch[s_idx]
        counts.scatter_add_(0, matched_batch, torch.ones_like(matched_batch, dtype=torch.int64))

    return s_feats, s_bb_feats, t_feats, counts


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
        encoding_range: float = 125.0,
    ):
        """
        Args:
            backbone_name:        Key into BACKBONE_REGISTRY (e.g., "attn_default", "base")
            use_proj_head:        Attach a DINO MLP projection head after the backbone
            proj_head_hidden_dim: Inner MLP width (DINO paper uses 2048)
            proj_head_output_dim: Output dimension of the final FC layer
            proj_head_n_layers:   Number of MLP layers before the final FC
            encoding_range:       Sinusoidal positional encoding range passed to the backbone
        """
        super().__init__()

        # Instantiate both backbones (sparse: Voxels → Voxels)
        # Only pass encoding_range to backbones that accept it (attn_* variants).
        backbone_cls = BACKBONE_REGISTRY[backbone_name]
        backbone_kwargs = {}
        if "encoding_range" in inspect.signature(backbone_cls.__init__).parameters:
            backbone_kwargs["encoding_range"] = encoding_range
        print("Initializing STUDENT backbone:")
        self.student = backbone_cls(**backbone_kwargs)
        print("Initializing TEACHER backbone:")
        self.teacher = backbone_cls(**backbone_kwargs)

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
        backbone_out = self.teacher(xs)
        if self.teacher_head is not None:
            return backbone_out, self.teacher_head(backbone_out)
        return backbone_out, backbone_out

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

    def forward_backward_crops(self, xs: Voxels, cropper, loss_fn):
        """
        Forward pass and backward update using activity-aware multi-crop augmentation.

        Follows the standard DINO multi-crop strategy:
          - Teacher sees only the n_global global crops (large views).
          - Student sees all crops (global + local).
          - Loss is computed for every (student_k, teacher_g) pair where k != g,
            restricted to voxels that fall inside both crops (spatial intersection).

        Spatial intersection is found by coordinate matching on the backbone
        outputs — no index tracking or LUTs from the cropper are needed.

        Args:
            xs:      batched Voxels (from the sparse dataloader, already on device)
            cropper: SparseCropper instance (provides n_global via cropper.cfg.n_global)
            loss_fn: PixelDINOLoss instance

        Returns:
            loss_value:           mean scalar loss across all (student, teacher) pairs
            teacher_entropy, student_entropy, kl, cov_penalty: averaged diagnostics
            student_backbone_out: backbone output for the last student crop (logging)
            teacher_backbone_out: backbone output for the first teacher global crop (logging)
            student_out:          head output for the last student crop (logging)
            teacher_out:          head output for the first teacher global crop (logging)
        """
        n_global = cropper.cfg.n_global

        crops = cropper(xs)
        n_crops = len(crops)

        # Teacher forward — global crops only, no gradient
        with torch.no_grad():
            teacher_encoded = [self.encode_teacher(crops[g]) for g in range(n_global)]

        total_loss = None
        sum_t_ent = sum_s_ent = sum_kl = sum_cov = sum_var = 0.0
        n_metric = n_pairs = 0

        for k in range(n_crops):
            student_backbone_k, student_out_k = self.encode_student(crops[k])

            for g in range(n_global):
                if k == g:
                    continue  # skip same-view pairs (matches original DINO)

                teacher_backbone_g, teacher_out_g = teacher_encoded[g]

                # Spatial intersection via coordinate matching
                s_feats, s_bb_feats, t_feats, counts = match_and_gather(
                    student_out_k, student_backbone_k, teacher_out_g,
                )

                loss_kg, t_ent, s_ent, kl, cov, var = loss_fn(
                    s_feats, s_bb_feats, t_feats, counts,
                )

                total_loss = loss_kg if total_loss is None else total_loss + loss_kg
                n_pairs += 1

                if t_ent is not None:
                    sum_t_ent += t_ent
                    sum_s_ent += s_ent
                    sum_kl    += kl
                    n_metric  += 1
                if cov is not None:
                    sum_cov += cov
                if var is not None:
                    sum_var += var

        total_loss = total_loss / n_pairs
        total_loss.backward()

        avg_t_ent = sum_t_ent / n_metric if n_metric > 0 else None
        avg_s_ent = sum_s_ent / n_metric if n_metric > 0 else None
        avg_kl    = sum_kl    / n_metric if n_metric > 0 else None
        avg_cov   = sum_cov   / n_pairs  if sum_cov != 0.0 else None
        avg_var   = sum_var   / n_pairs  if sum_var != 0.0 else None

        # Logging: last student crop, first teacher global crop
        teacher_backbone_log, teacher_out_log = teacher_encoded[0]
        return (
            total_loss.item(),
            avg_t_ent, avg_s_ent, avg_kl, avg_cov, avg_var,
            student_backbone_k, teacher_backbone_log,
            student_out_k, teacher_out_log,
        )

    def forward_backward(self, xs: Voxels, masker, loss_fn):
        """
        Forward pass and backward update.

        Args:
            xs: batched Voxels (from the sparse dataloader, already on device)
            masker: SparseVoxelMasker instance
            loss_fn: PixelDINOLoss instance

        Returns:
            loss_value:      scalar loss
            teacher_entropy: H(P_t) per batch (dino loss only, else None)
            student_entropy: H(P_s) per batch (dino loss only, else None)
            kl:              KL(P_t||P_s) per batch (dino loss only, else None)
            cov_penalty:          covariance penalty (if enabled, else None)
            student_backbone_out: raw 64-dim backbone output (before head)
            student_out:          student Voxels output (after head if present)
            teacher_out:          teacher Voxels output (after head if present)
        """
        # Masking on Voxels: returns reduced student Voxels (kept_indices no longer needed)
        xs_student, _ = masker(xs)

        # Teacher forward (full Voxels, frozen, no grad)
        with torch.no_grad():
            teacher_backbone_out, teacher_out = self.encode_teacher(xs)

        # Student forward (masked Voxels, trainable)
        student_backbone_out, student_out = self.encode_student(xs_student)

        # Match student/teacher by coordinates and compute loss
        s_feats, s_bb_feats, t_feats, counts = match_and_gather(
            student_out, student_backbone_out, teacher_out,
        )
        loss, teacher_entropy, student_entropy, kl, cov_penalty, var_penalty = loss_fn(
            s_feats, s_bb_feats, t_feats, counts,
        )
        loss.backward()

        return loss.item(), teacher_entropy, student_entropy, kl, cov_penalty, var_penalty, student_backbone_out, teacher_backbone_out, student_out, teacher_out
