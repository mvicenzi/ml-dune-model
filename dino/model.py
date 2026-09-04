"""DINO training model: student + teacher with EMA update."""

import contextlib
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels

from models import BACKBONE_REGISTRY, backbone_kwargs
from .config import objective_injects
from .loss import charge_loss, occupancy_loss
from .debug import NULL_TIMER
from .projhead import DINOProjectionHead


@dataclass
class ForwardStats:
    """Everything one `forward_backward` pass reports back to the training loop.

    Named rather than positional because which quantities exist depends on the
    configuration: a run without a teacher has no entropies or KL, a run without
    masking has no masked/unmasked split. Fields that do not apply are None, and a
    reader that wants one asks for it by name instead of counting tuple positions.

    The Voxels fields are the last student view and the first teacher global view,
    kept for logging; the training loop drops the whole object once it has logged,
    which is what releases them before the next forward.
    """

    loss: float
    n_pairs: int
    t_ent: Optional[float] = None
    s_ent: Optional[float] = None
    kl: Optional[float] = None
    cov: Optional[float] = None
    var: Optional[float] = None
    loss_masked: Optional[float] = None
    loss_unmasked: Optional[float] = None
    loss_charge: Optional[float] = None
    loss_occ: Optional[float] = None
    s_backbone: Optional[Voxels] = None
    t_backbone: Optional[Voxels] = None
    s_out: Optional[Voxels] = None
    t_out: Optional[Voxels] = None


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


def gather_at_coords(vox: Voxels, coords_per_batch, targets_per_batch=None):
    """Read a one-channel prediction at a requested set of coordinates, per image.

    `vox` is a head output whose coordinates are a superset of the requested ones (they
    were injected, so they are there). Matching is by the same batch-unique flat key
    `match_and_gather` uses, rather than by row position: nothing in the sparse
    convolution API promises that rows come back in the order they went in.

    A requested coordinate that is somehow missing from the output is skipped along with
    its target, so predictions and targets stay aligned and the per-image count reflects
    what was actually matched -- which is what the loss divides by.

    Returns (pred [M], target [M] or None, counts [B]).
    """
    device = vox.feature_tensor.device
    B = len(vox.offsets) - 1
    req = [c for c in list(coords_per_batch)[:B]]
    n_req = torch.tensor([c.shape[0] for c in req], dtype=torch.int64, device=device)
    out_coords = vox.coordinate_tensor

    if out_coords.shape[0] == 0 or int(n_req.sum()) == 0:
        empty = vox.feature_tensor.new_zeros(0)
        return empty, (empty if targets_per_batch is not None else None), \
            torch.zeros(B, dtype=torch.int64, device=device)

    req_coords = torch.cat(req, dim=0).to(device)
    req_b = torch.repeat_interleave(torch.arange(B, device=device), n_req)

    # key strides span both operands so no coordinate can wrap into another's key
    W = int(max(int(out_coords[:, 0].max()), int(req_coords[:, 0].max()))) + 1
    H = int(max(int(out_coords[:, 1].max()), int(req_coords[:, 1].max()))) + 1
    S = H * W
    o_counts = (vox.offsets[1:] - vox.offsets[:-1]).to(device)
    o_b = torch.repeat_interleave(torch.arange(B, device=device), o_counts)
    o_keys = o_b * S + out_coords[:, 1].long() * W + out_coords[:, 0].long()
    r_keys = req_b * S + req_coords[:, 1].long() * W + req_coords[:, 0].long()

    o_sorted, o_order = o_keys.sort()
    pos = torch.searchsorted(o_sorted, r_keys).clamp(max=o_sorted.shape[0] - 1)
    hit = o_sorted[pos] == r_keys

    pred = vox.feature_tensor.reshape(-1)[o_order[pos[hit]]]
    counts = torch.bincount(req_b[hit], minlength=B)
    target = None
    if targets_per_batch is not None:
        tgt = torch.cat([t.reshape(-1) for t in list(targets_per_batch)[:B]], dim=0).to(device)
        target = tgt[hit]
    return pred, target, counts


def occupancy_target(coords_per_batch, active_per_batch, stride: int, device):
    """Was any pixel in this cell active before masking?

    `active_per_batch` is the ORIGINAL active set (kept plus removed) at full resolution;
    coordinates are floor-divided to the candidate scale and membership is tested there,
    so a half-resolution cell counts as occupied if it contained any active pixel. The
    masked structure is absent from what the candidates were grown from, so nothing here
    peeks at the answer.
    """
    out = []
    for cand, act in zip(coords_per_batch, active_per_batch):
        if cand.shape[0] == 0:
            out.append(torch.zeros(0, device=device))
            continue
        a = torch.div(act.to(device), stride, rounding_mode="floor")
        H = int(max(int(a[:, 1].max()) if a.shape[0] else 0, int(cand[:, 1].max()))) + 1
        a_keys = a[:, 0].long() * H + a[:, 1].long()
        c_keys = cand[:, 0].long().to(device) * H + cand[:, 1].long().to(device)
        out.append(torch.isin(c_keys, a_keys).float())
    return out


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
        backbone_name: str = "attn_mae",
        use_proj_head: bool = False,
        proj_head_hidden_dim: int = 256,
        proj_head_output_dim: int = 256,
        proj_head_n_layers: int = 4,
        encoding_range: float = 125.0,
        encoding_dim: int = 32,
        objective: str = "hybrid",
    ):
        """
        Args:
            backbone_name:        Key into BACKBONE_REGISTRY (e.g., "attn_mae", "base")
            use_proj_head:        Attach a DINO MLP projection head after the backbone
            proj_head_hidden_dim: Inner MLP width (DINO paper uses 2048)
            proj_head_output_dim: Output dimension of the final FC layer
            proj_head_n_layers:   Number of MLP layers before the final FC
            encoding_range:       Sinusoidal positional encoding range passed to the backbone
            encoding_dim:         Number of sinusoidal encoding channels per spatial axis
            objective:            "dino" | "hybrid" | "mae". Only mae changes the structure
                                  here: it builds the reconstruction heads and no teacher.
        """
        super().__init__()

        self.objective = objective
        self.use_recon_heads = objective == "mae"

        # Instantiate both backbones (sparse: Voxels -> Voxels)
        backbone_cls = BACKBONE_REGISTRY[backbone_name]
        arch_flags = dict(encoding_range=encoding_range, encoding_dim=encoding_dim)
        if self.use_recon_heads:
            # only requested when needed: backbone_kwargs raises on a flag the chosen
            # backbone cannot honour, and the non-MAE backbones do not accept this one
            arch_flags["use_recon_heads"] = True
        kwargs = backbone_kwargs(backbone_cls, **arch_flags)

        # Detect whether this backbone supports masked_coords injection (MAE backbones).
        self._student_accepts_masked_coords = (
            "masked_coords" in inspect.signature(backbone_cls.forward).parameters
        )

        print("Initializing STUDENT backbone:")
        self.student = backbone_cls(**kwargs)

        # mae reconstructs its own input, so there is nothing for a teacher to teach.
        # It is left as None rather than built and frozen: a checkpoint carrying a
        # never-trained teacher would let `--source=teacher` extraction quietly return
        # features from initialisation weights.
        if self.use_recon_heads:
            self.teacher = None
            print("Objective is mae: no teacher backbone")
        else:
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
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad = False
        if self.teacher_head is not None:
            for p in self.teacher_head.parameters():
                p.requires_grad = False

        # Teacher always in eval mode (batchnorm, dropout, etc.)
        if self.teacher is not None:
            self.teacher.eval()
        if self.teacher_head is not None:
            self.teacher_head.eval()

    def train(self, mode: bool = True):
        """Override train() to keep teacher (and teacher head) in eval mode."""
        super().train(mode)
        if self.teacher is not None:
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
        # No teacher under mae. The condition is a property of the objective, identical on
        # every rank, so this never desynchronises a distributed step.
        if self.teacher is None:
            return
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

    def _mae_step(self, xs, masker, timer, lambda_charge, lambda_occ, occ_grow_iters):
        """One reconstruction step: no teacher, no views, one backward.

        The objective is supervised entirely by the input. Charge asks for the value at
        each pixel masking removed; occupancy asks, over a candidate set that includes
        genuine empties, which of them held charge at all. Where those candidates come
        from is the masker's call: one that empties whole cells of a known grid hands
        them over already labelled, and one that scatters its removals leaves them to be
        grown from the surviving structure in the backbone. Both terms are normalised
        per image before their weights are applied, so the weights mean the same thing
        on a busy image as on an empty one.

        Cropping is rejected for this objective upstream, so there is exactly one view and
        therefore exactly one backward -- which is also what DistributedDataParallel needs
        to reduce correctly.
        """
        with timer.stage("mask"):
            view, masked_coords, masked_feats, cand_coords, cand_occ = masker(xs)
        if masked_feats is None:
            raise RuntimeError(
                "objective='mae' needs the masker's removed features as its charge "
                "target; construct the masker with return_masked_feats=True"
            )

        with timer.stage("stud"):
            backbone_out, charge_vox, occ_vox, occ_coords = self.student(
                view, masked_coords=masked_coords,
                return_recon=True, occ_grow_iters=occ_grow_iters,
                occ_coords=cand_coords,
            )

        device = backbone_out.feature_tensor.device
        B = len(xs.offsets) - 1

        with timer.stage("loss"):
            chg_pred, chg_tgt, chg_counts = gather_at_coords(
                charge_vox, masked_coords, masked_feats)
            l_charge = charge_loss(chg_pred, chg_tgt, chg_counts)

            # The request coordinates and their labels must be the SAME list, row for
            # row. The backbone's occ_coords is not that list: injection drops any
            # candidate the skip already carries, so it comes back shorter than what the
            # masker labelled. Ask at the labelled coordinates instead and let
            # gather_at_coords drop the pairs it cannot find a prediction for -- it
            # discards the target alongside the request, so the two stay aligned.
            if cand_occ is not None:
                # The masker enumerated the candidates, so it also knows their labels
                # exactly; recomputing them here could only introduce a disagreement.
                occ_req, occ_tgt_list = cand_coords, cand_occ
            else:
                # Grown candidates carry no labels, so they are scored against the
                # pre-mask active set: what was here before anything was removed.
                active = [xs.coordinate_tensor[int(xs.offsets[b]):int(xs.offsets[b + 1])]
                          for b in range(B)]
                occ_req = occ_coords
                occ_tgt_list = occupancy_target(occ_coords, active, stride=2, device=device)
            occ_pred, occ_tgt, occ_counts = gather_at_coords(
                occ_vox, occ_req, occ_tgt_list)
            l_occ = occupancy_loss(occ_pred, occ_tgt, occ_counts)

            total = lambda_charge * l_charge + lambda_occ * l_occ

        with timer.stage("bwd"):
            total.backward()

        return ForwardStats(
            loss=total.item(),
            n_pairs=1,
            loss_charge=l_charge.item(),
            loss_occ=l_occ.item(),
            s_backbone=backbone_out,
            s_out=backbone_out,
        )

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
        objective: str = None,
        lambda_charge: float = 0.1,
        lambda_occ: float = 1.0,
        occ_grow_iters: int = 1,
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
            objective:    "dino" | "hybrid" | "mae" — decides whether the coordinates the
                          masker removed are reinjected as placeholders and supervised

        Returns:
            ForwardStats — `loss` is the mean over all (student, teacher) pairs; the
            diagnostics are averaged the same way; the Voxels fields are the last
            student view and the first teacher global view.
        """
        objective = objective if objective is not None else self.objective
        if objective == "mae":
            return self._mae_step(xs, masker, timer or NULL_TIMER,
                                  lambda_charge, lambda_occ, occ_grow_iters)

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
        # Injection is what makes a masked same-index pair a prediction task rather than a
        # self-comparison, so both the pair plan and the placeholder handoff key on it.
        # Under "dino" the masker still runs -- masking stays available as an augmentation --
        # but the removed coordinates are never restored, so they are simply absent from the
        # student's output and match_and_gather's intersection drops them from the loss.
        inject_on = (use_masking and objective_injects(objective)
                     and self._student_accepts_masked_coords)
        keep_same_index = inject_on

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
                    view_k_masked, masked_coords_k = masker(all_views[k])[:2]
            else:
                view_k_masked, masked_coords_k = all_views[k], None

            # None means "do not restore these"; the backbone then runs its base decoder and
            # the masked/unmasked loss split is reported as absent rather than as a split of
            # positions that are not in the output.
            inject_coords_k = masked_coords_k if inject_on else None

            # execute the model, returning backbone and head outputs
            with timer.stage("stud"):
                student_backbone_k, student_out_k = self.encode_student(
                    view_k_masked, masked_coords=inject_coords_k,
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
                        masked_coords_per_batch=inject_coords_k,
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
        return ForwardStats(
            loss=total_loss.item(),
            n_pairs=n_pairs,
            t_ent=avg_t_ent, s_ent=avg_s_ent, kl=avg_kl,
            cov=avg_cov, var=avg_var,
            loss_masked=avg_loss_masked, loss_unmasked=avg_loss_unmasked,
            s_backbone=student_backbone_k, t_backbone=teacher_backbone_log,
            s_out=student_out_k, t_out=teacher_out_log,
        )
