"""DINO training model: student + teacher with EMA update."""

import inspect
import torch
import torch.nn as nn

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

from models import BACKBONE_REGISTRY
from .projhead import DINOProjectionHead


def _filter_voxels(voxels: Voxels, local_indices: list) -> Voxels:
    """
    Return a new Voxels containing only the selected voxels per batch item.

    Args:
        voxels:        Source Voxels (batched).
        local_indices: List of B 1-D tensors; local_indices[b] selects rows from
                       batch item b's slice of voxels.
    """
    B = len(voxels.offsets) - 1
    new_coords_list, new_feats_list, counts = [], [], []
    for b in range(B):
        start = int(voxels.offsets[b])
        end   = int(voxels.offsets[b + 1])
        idx   = local_indices[b]
        new_coords_list.append(voxels.coordinate_tensor[start:end][idx])
        new_feats_list.append(voxels.feature_tensor[start:end][idx])
        counts.append(idx.shape[0])

    counts_t = torch.tensor(counts, dtype=torch.int64)
    offsets  = torch.cat([torch.zeros(1, dtype=torch.int64), counts_t.cumsum(0)])

    if any(c > 0 for c in counts):
        new_coords = torch.cat(new_coords_list, dim=0)
        new_feats  = torch.cat(new_feats_list,  dim=0)
    else:
        new_coords = voxels.coordinate_tensor.new_zeros(0, voxels.coordinate_tensor.shape[1])
        new_feats  = voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1])

    return Voxels(
        batched_coordinates=IntCoords(new_coords, offsets=offsets),
        batched_features=CatFeatures(new_feats, offsets=offsets),
        offsets=offsets,
    )


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

        For each pair the intersection is found by matching full-image voxel indices
        returned by the cropper.  A filtered student Voxels (intersection voxels only)
        and the corresponding local indices into the teacher crop are passed to loss_fn,
        keeping PixelDINOLoss.forward unchanged.

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
        B        = len(xs.offsets) - 1
        device   = xs.coordinate_tensor.device

        # Produce all crops, their full-image voxel indices, and teacher LUTs
        crops, kept_indices_list, teacher_luts = cropper(xs)
        n_crops = len(crops)

        # Teacher forward — global crops only, no gradient
        with torch.no_grad():
            teacher_encoded = [self.encode_teacher(crops[g]) for g in range(n_global)]
            # teacher_encoded[g] = (backbone_out_g, head_out_g)

        total_loss = None
        sum_t_ent = sum_s_ent = sum_kl = sum_cov = sum_var = 0.0
        n_metric = n_pairs = 0

        for k in range(n_crops):
            student_backbone_k, student_out_k = self.encode_student(crops[k])

            for g in range(n_global):
                if k == g:
                    continue  # skip same-view pairs (matches original DINO)

                teacher_backbone_g, teacher_out_g = teacher_encoded[g]

                # ------------------------------------------------------------------
                # Spatial intersection via LUT gather: look up each student
                # voxel in the teacher crop's pre-built lookup table.  O(N)
                # gather replaces O(N log N) isin + searchsorted.
                # ------------------------------------------------------------------
                s_local_idx_list = []
                t_local_idx_list = []

                for b in range(B):
                    S_k_b = kept_indices_list[k][b]
                    lut = teacher_luts[g][b]

                    if S_k_b.numel() == 0:
                        empty = torch.zeros(0, dtype=torch.long, device=device)
                        s_local_idx_list.append(empty)
                        t_local_idx_list.append(empty)
                        continue

                    t_local_raw = lut[S_k_b]          # -1 where not in teacher crop
                    valid = t_local_raw >= 0
                    s_local = valid.nonzero(as_tuple=False).squeeze(1)

                    if s_local.numel() == 0:
                        empty = torch.zeros(0, dtype=torch.long, device=device)
                        s_local_idx_list.append(empty)
                        t_local_idx_list.append(empty)
                        continue

                    t_local = t_local_raw[valid]
                    s_local_idx_list.append(s_local)
                    t_local_idx_list.append(t_local)

                # Build student Voxels restricted to intersection voxels
                student_out_kg      = _filter_voxels(student_out_k,      s_local_idx_list)
                student_backbone_kg = _filter_voxels(student_backbone_k, s_local_idx_list)

                loss_kg, t_ent, s_ent, kl, cov, var = loss_fn(
                    student_out_kg, student_backbone_kg, teacher_out_g, t_local_idx_list
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
        # Masking on Voxels: returns reduced student Voxels + kept_indices
        xs_student, kept_indices = masker(xs)

        # Teacher forward (full Voxels, frozen, no grad)
        with torch.no_grad():
            teacher_backbone_out, teacher_out = self.encode_teacher(xs)

        # Student forward (masked Voxels, trainable)
        student_backbone_out, student_out = self.encode_student(xs_student)

        # Compute loss and backprop
        loss, teacher_entropy, student_entropy, kl, cov_penalty, var_penalty = loss_fn(student_out, student_backbone_out, teacher_out, kept_indices)
        loss.backward()

        return loss.item(), teacher_entropy, student_entropy, kl, cov_penalty, var_penalty, student_backbone_out, teacher_backbone_out, student_out, teacher_out
