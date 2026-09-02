"""DINO training model: student + teacher with EMA update."""

import contextlib
import inspect
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels

from models import BACKBONE_REGISTRY, backbone_kwargs
from .debug import NULL_TIMER
from .projhead import DINOProjectionHead


def match_and_gather(
    s_out: Voxels,
    s_backbone: Voxels,
    t_out: Voxels,
    masked_coords_per_batch=None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """
    Match student and teacher output voxels by spatial coordinates and
    return pre-aligned feature tensors ready for the loss.

    Each voxel gets a flat integer key `b * S + y * W + x`. The batch index in the
    key keeps images apart, so one sort and one searchsorted over the whole batch
    do the work of B independent per-image intersections. Keys are unique (a
    coordinate appears once per image), so the sort has no ties to break.

    Coordinates are intrinsic to the backbone output -- no LUTs or index
    bookkeeping from the cropper are needed.

    Args:
        s_out:      Student head output (Voxels).
        s_backbone: Student backbone output (Voxels), same coordinates as s_out.
        t_out:      Teacher head output (Voxels).

    Returns:
        s_feats:    [N_matched, D_head]  student head features at intersection
        s_bb_feats: [N_matched, D_bb]    student backbone features at intersection
        t_feats:    [N_matched, D_head]  teacher head features at intersection
        counts:     [B] int64 per-image matched voxel counts
        is_masked:  [N_matched] bool — True for positions that were masked in the
                    student input (i.e. the backbone received a mask token there);
                    None when masked_coords_per_batch is not provided, and also
                    when nothing matched at all
    """
    B = len(s_out.offsets) - 1
    device = s_out.feature_tensor.device

    s_coords = s_out.coordinate_tensor
    t_coords = t_out.coordinate_tensor

    def _empty():
        s_feats    = s_out.feature_tensor.new_zeros(0, s_out.feature_tensor.shape[1])
        s_bb_feats = s_backbone.feature_tensor.new_zeros(0, s_backbone.feature_tensor.shape[1])
        t_feats    = t_out.feature_tensor.new_zeros(0, t_out.feature_tensor.shape[1])
        counts     = torch.zeros(B, dtype=torch.int64, device=device)
        return s_feats, s_bb_feats, t_feats, counts, None

    # Nothing on one side means nothing to intersect. Also guards the clamp below,
    # which needs at least one teacher key to clamp against.
    if s_coords.shape[0] == 0 or t_coords.shape[0] == 0:
        return _empty()

    # Key strides: W spans x, S spans one image's key space. Both are taken across
    # student and teacher so the two agree on the encoding.
    W = int(max(s_coords[:, 0].max().item(), t_coords[:, 0].max().item())) + 1
    H = int(max(s_coords[:, 1].max().item(), t_coords[:, 1].max().item())) + 1
    S = H * W

    def _keys(coords, offsets):
        counts = (offsets[1:] - offsets[:-1]).to(device)
        b_idx = torch.repeat_interleave(torch.arange(B, device=device), counts)
        return b_idx * S + coords[:, 1].long() * W + coords[:, 0].long(), b_idx

    s_keys, b_s = _keys(s_coords, s_out.offsets)
    t_keys, _   = _keys(t_coords, t_out.offsets)

    # Sorting the whole teacher key array keeps t_order pointing at rows of the flat
    # feature tensor, so matched teacher indices need no per-image offset shift.
    t_sorted, t_order = t_keys.sort()
    pos = torch.searchsorted(t_sorted, s_keys).clamp(max=t_sorted.shape[0] - 1)
    valid = t_sorted[pos] == s_keys

    # Ascending global index == image order, then within-image order: the flat
    # tensors are batch-major, so this is the order a per-image loop would produce.
    s_idx = valid.nonzero(as_tuple=False).squeeze(1)
    if s_idx.numel() == 0:
        return _empty()
    t_idx = t_order[pos[s_idx]]

    counts = torch.bincount(b_s[s_idx], minlength=B)

    s_feats    = s_out.feature_tensor[s_idx]
    s_bb_feats = s_backbone.feature_tensor[s_idx]  # same coords as s_out
    t_feats    = t_out.feature_tensor[t_idx]

    # Tag each matched student position as masked or not: same key trick, with the
    # per-image masked coords concatenated into one array first.
    is_masked = None
    if masked_coords_per_batch is not None:
        m_per_image = list(masked_coords_per_batch)[:B]
        m_counts = torch.tensor([m.shape[0] for m in m_per_image],
                                dtype=torch.int64, device=device)
        if int(m_counts.sum()) > 0:
            m_coords = torch.cat(m_per_image, dim=0)
            b_m = torch.repeat_interleave(torch.arange(B, device=device), m_counts)
            m_keys = b_m * S + m_coords[:, 1].long() * W + m_coords[:, 0].long()
            m_sorted, _ = m_keys.sort()
            matched_keys = s_keys[s_idx]
            mp = torch.searchsorted(m_sorted, matched_keys).clamp(max=m_sorted.shape[0] - 1)
            is_masked = m_sorted[mp] == matched_keys
        else:
            is_masked = torch.zeros(s_idx.shape[0], dtype=torch.bool, device=device)

    return s_feats, s_bb_feats, t_feats, counts, is_masked


class DINODuneModel(nn.Module):
    """
    DINO-style teacher/student framework for DUNE backbone.

    - Student: trainable backbone (+ optional projection head), receives masked/cropped input
    - Teacher: frozen backbone (+ optional projection head), receives full input
    - Both share the same architecture; training updates only student
    - Teacher updates via EMA of student parameters (momentum schedule)

    """
    def __init__(
        self,
        backbone_name: str = "attn_default",
        use_proj_head: bool = False,
        proj_head_hidden_dim: int = 256,
        proj_head_output_dim: int = 256,
        proj_head_n_layers: int = 4,
        encoding_range: float = 125.0,
        encoding_dim: int = 32,
    ):
        """
        Args:
            backbone_name:        Key into BACKBONE_REGISTRY (e.g., "attn_default", "base")
            use_proj_head:        Attach a DINO MLP projection head after the backbone
            proj_head_hidden_dim: Inner MLP width (DINO paper uses 2048)
            proj_head_output_dim: Output dimension of the final FC layer
            proj_head_n_layers:   Number of MLP layers before the final FC
            encoding_range:       Sinusoidal positional encoding range passed to the backbone
            encoding_dim:         Number of sinusoidal encoding channels per spatial axis
        """
        super().__init__()

        # Instantiate both backbones (sparse: Voxels -> Voxels)
        backbone_cls = BACKBONE_REGISTRY[backbone_name]
        kwargs = backbone_kwargs(
            backbone_cls,
            encoding_range=encoding_range,
            encoding_dim=encoding_dim,
        )

        # Detect whether this backbone supports masked_coords injection (MAE backbones).
        self._student_accepts_masked_coords = (
            "masked_coords" in inspect.signature(backbone_cls.forward).parameters
        )

        print("Initializing STUDENT backbone:")
        self.student = backbone_cls(**kwargs)
        print("Initializing TEACHER backbone:")
        self.teacher = backbone_cls(**kwargs)

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
        """ 
            Run the teacher backbone (+ head if present). 
            Always called in no_grad context.
        """
        backbone_out = self.teacher(xs)
        if self.teacher_head is not None:
            return backbone_out, self.teacher_head(backbone_out)
        return backbone_out, backbone_out

    def encode_student(
        self,
        xs: Voxels,
        masked_coords: Optional[list] = None,
    ) -> tuple[Voxels, Voxels]:
        """
        Run the student backbone (+ head if present).

        Args:
            xs:            Student input Voxels (kept voxels only when masking).
            masked_coords: List of B tensors [N_masked_b, 2] for MAE injection.
                           Passed through to the backbone only when not None.

        Returns:
            backbone_out: raw 64-dim backbone output (before head)
            final_out:    head output if head present, else same as backbone_out
        """

        if masked_coords is not None and self._student_accepts_masked_coords:
            backbone_out = self.student(xs, masked_coords=masked_coords)
        else:
            backbone_out = self.student(xs)

        if self.student_head is not None:
            return backbone_out, self.student_head(backbone_out)
        return backbone_out, backbone_out

    # ----------------------- main forward/backward pass for training -------------------
    def forward_backward(
        self,
        xs: Voxels,
        cropper,
        masker,
        loss_fn,
        use_cropping: bool = False,
        use_masking: bool = True,
        timer=None,
    ):
        """
        Unified forward pass and backward update.

        Views are always treated as a list:
          - Cropping enabled:  SparseCropper produces n_global + n_local views.
          - Cropping disabled: the original full-image batch is the single view.

        Masking (random pixel dropout) is applied independently to each student
        view when enabled; teacher views are never masked.

        Loss is summed over all (student_k, teacher_g) pairs. Same-index pairs
        (k == g) are skipped only when multiple views exist; with a single view
        the masking pair (k=0, g=0) is the only valid pair and must be kept.

        A masked student view is not a copy of the teacher's: the student sees
        mask tokens where the teacher sees charge, so the same-index pair carries
        the masked-prediction signal at that view's scale and is kept. With
        crop_n_global=1 the global view's only candidate pair is (0, 0), so
        skipping it would encode that view and then discard it.

        Args:
            xs:           batched Voxels (from the sparse dataloader, on device)
            cropper:      SparseCropper instance, or None when use_cropping=False
            masker:       SparseVoxelMasker instance (used only when use_masking=True)
            loss_fn:      PixelDINOLoss instance
            use_cropping: enable activity-aware multi-crop augmentation
            use_masking:  enable random pixel dropout on student views
            timer:        StageTimer collecting the per-stage GPU breakdown

        Returns:
            loss_value:           mean scalar loss across all (student, teacher) pairs
            teacher_entropy, student_entropy, kl, cov_penalty, var_penalty: averaged diagnostics
            student_backbone_out: backbone output of the last student view (for logging)
            teacher_backbone_out: backbone output of the first teacher global view (for logging)
            student_out:          head output of the last student view (for logging)
            teacher_out:          head output of the first teacher global view (for logging)
        """
        # ── 1. Generate views ──────────────────────────────────────────────
        # view are always treated as a list: 
        # either multiple crops or a single full-image view
        timer = timer or NULL_TIMER
        if use_cropping:
            with timer.stage("crop"):
                all_views = cropper(xs)
            n_global  = cropper.cfg.n_global
        else:
            all_views = [xs]
            n_global  = 1
        n_crops = len(all_views)

        # A masked same-index pair only differs from a self-comparison when the
        # backbone puts mask tokens back at the dropped coordinates: the student
        # then has to predict the teacher's features there. A backbone that just
        # drops the masked voxels gives back the teacher's own view minus some
        # points, so its same-index pair stays skipped.
        keep_same_index = use_masking and self._student_accepts_masked_coords

        # ── 2. Teacher: encode global views, frozen, no gradient ───────────
        with torch.no_grad(), timer.stage("teach"):
            teacher_encoded = [self.encode_teacher(all_views[g]) for g in range(n_global)]

        # ── 3. Student: encode all views (optionally masked), compute loss ─
        total_loss = None
        sum_t_ent = sum_s_ent = sum_kl = sum_cov = sum_var = 0.0
        sum_loss_masked = sum_loss_unmasked = 0.0
        n_metric = n_split = 0

        # Decide the (student view, teacher view) pairs before running any of them.
        # Each student view gets its own backward so that DistributedDataParallel sees
        # one forward per backward -- its reducer fires per forward, and feeding it
        # several forwards before a single backward reduces some parameters more than
        # once (measured: gradients off by more than the ranks differ from each other).
        # Knowing the schedule up front is what lets the loop mark the final backward,
        # which is the one DDP must synchronise on.
        pair_plan = []
        for k in range(n_crops):
            gs = [g for g in range(n_global)
                  if not (k == g and n_crops > 1 and not keep_same_index)]
            if gs:
                pair_plan.append((k, gs))
        n_pairs = sum(len(gs) for _, gs in pair_plan)

        # for each student view
        for plan_idx, (k, teacher_indices) in enumerate(pair_plan):
            if use_masking:
                with timer.stage("mask"):
                    view_k_masked, masked_coords_k = masker(all_views[k])
            else:
                view_k_masked, masked_coords_k = all_views[k], None

            # execute the model, returning backbone and head outputs
            with timer.stage("stud"):
                student_backbone_k, student_out_k = self.encode_student(
                    view_k_masked, masked_coords=masked_coords_k,
                )

            # for each teacher global this view pairs with
            view_loss = None
            for g in teacher_indices:
                teacher_backbone_g, teacher_out_g = teacher_encoded[g]

                # returns features for each matching voxels across views
                # shape is [N_matched, D] --> D differs for backbone vs head
                # returing student backbone feature for optional cov/var penalties
                with timer.stage("match"):
                    s_feats, s_bb_feats, t_feats, counts, is_masked = match_and_gather(
                        student_out_k, student_backbone_k, teacher_out_g,
                        masked_coords_per_batch=masked_coords_k,
                    )

                # compute the loss for these views
                with timer.stage("loss"):
                    loss_kg, t_ent, s_ent, kl, cov, var, loss_masked_kg, loss_unmasked_kg = loss_fn(
                        s_feats, s_bb_feats, t_feats, counts, is_masked=is_masked,
                    )

                # accumulate within this view; the backward happens once below, so a
                # view used by several teacher views still needs only one graph pass
                view_loss = loss_kg if view_loss is None else view_loss + loss_kg
                total_loss = (loss_kg.detach() if total_loss is None
                              else total_loss + loss_kg.detach())

                # accumulate entropy metrics for logging (averaging)
                if t_ent is not None:
                    sum_t_ent += t_ent
                    sum_s_ent += s_ent
                    sum_kl    += kl
                    n_metric  += 1

                # accumulate covariance/variance penalty metrics (averaging)
                if cov is not None:
                    sum_cov += cov
                if var is not None:
                    sum_var += var

                # accumulate masked/unmasked split losses (diagnostics only)
                if loss_masked_kg is not None:
                    sum_loss_masked   += loss_masked_kg
                    n_split += 1
                if loss_unmasked_kg is not None:
                    sum_loss_unmasked += loss_unmasked_kg

            # Backward for this view. Scaling by n_pairs here makes the accumulated
            # gradient identical to one backward on the mean over all pairs.
            # Every view but the last runs under no_sync(), so DDP all-reduces once
            # per step rather than once per view -- the standard accumulation idiom.
            is_last = plan_idx == len(pair_plan) - 1
            with timer.stage("bwd"):
                with contextlib.ExitStack() as stack:
                    if not is_last:
                        for module in (self.student, self.student_head):
                            if hasattr(module, "no_sync"):
                                stack.enter_context(module.no_sync())
                    (view_loss / n_pairs).backward()

        total_loss = total_loss / n_pairs

        avg_t_ent = sum_t_ent / n_metric if n_metric > 0 else None
        avg_s_ent = sum_s_ent / n_metric if n_metric > 0 else None
        avg_kl    = sum_kl    / n_metric if n_metric > 0 else None
        avg_cov   = sum_cov   / n_pairs  if sum_cov != 0.0 else None
        avg_var   = sum_var   / n_pairs  if sum_var != 0.0 else None
        avg_loss_masked   = sum_loss_masked   / n_split if n_split > 0 else None
        avg_loss_unmasked = sum_loss_unmasked / n_split if n_split > 0 else None

        # logging: last student view, first teacher global view
        # teacher outputs also used for centering update
        teacher_backbone_log, teacher_out_log = teacher_encoded[0]
        return (
            total_loss.item(), # loss
            # entropy metrics for logging
            avg_t_ent, avg_s_ent, avg_kl, avg_cov, avg_var,
            # masked/unmasked loss split for logging
            avg_loss_masked, avg_loss_unmasked,
            # student/teacher backbone features for logging
            student_backbone_k, teacher_backbone_log,
            # student/teacher head features for logging
            student_out_k, teacher_out_log,
        )
