"""Per-pixel DINO-style loss for self-supervised training."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from warpconvnet.geometry.types.voxels import Voxels


class PixelDINOLoss(nn.Module):
    """
    Per-pixel DINO-style knowledge distillation loss.

    Compares feature vectors pixel-by-pixel between student and teacher at unmasked
    active positions. Teacher (frozen, EMA updated) provides soft targets; student
    (trainable) learns to match these targets.

    Implemented on dense backbone outputs [B, D, H, W]. Loss computed only at positions
    where the student actually computed features (unmasked active pixels), not at
    structural zeros from sparse→dense conversion.

    The loss is the cross-entropy between softmax(teacher/tau_t) and
    log_softmax(student/tau_s): the D-dim feature vector is treated as logits, which
    creates per-dimension competition and so prevents dimensional collapse.
    """

    def __init__(
        self,
        center_momentum: float = 0.9,
        use_centering: bool = True,
        teacher_temp: float = 1.0,
        student_temp: float = 1.0,
        normalize_features: bool = True,
        use_cov_penalty: bool = False,
        cov_penalty_weight: float = 1e-3,
        use_var_penalty: bool = False,
        var_penalty_weight: float = 1.0,
        var_gamma: float = 1.0,
    ):
        """
        Args:
            center_momentum:     EMA decay for the teacher center buffer (default 0.9)
            use_centering:       if True, subtract running center from teacher features before
                                 computing the loss; the center buffer is always updated regardless
            teacher_temp:        softmax temperature for teacher logits
            student_temp:        softmax temperature for student logits
            normalize_features:  if True, L2-normalise student and teacher features before the
                                 loss; set to False when a projection head is used (the head's
                                 internal L2 norm already normalises the features)
            use_cov_penalty:     if True, add a VICReg-style covariance decorrelation penalty on
                                 student features to prevent dimensional collapse
            cov_penalty_weight:  scalar weight for the covariance penalty term (default 1e-3)
            use_var_penalty:     if True, add a VICReg-style variance penalty on student features
                                 to prevent dimensional collapse by keeping per-dim std >= var_gamma
            var_penalty_weight:  scalar weight for the variance penalty term (default 1.0)
            var_gamma:           target minimum std per feature dimension (default 1.0)
        """
        super().__init__()
        self.center_momentum = center_momentum
        self.use_centering = use_centering
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.normalize_features = normalize_features
        self.use_cov_penalty = use_cov_penalty
        self.cov_penalty_weight = cov_penalty_weight
        self.use_var_penalty = use_var_penalty
        self.var_penalty_weight = var_penalty_weight
        self.var_gamma = var_gamma
        # Lazily initialized on first forward call once feature dim D is known.
        # register_buffer ensures it moves with .to(device) and is saved in checkpoints.
        self.register_buffer("center", None)

    @staticmethod
    def _img_mean(loss_px: Tensor, batch_idx: Tensor, B: int, device) -> Tensor | None:
        """Scatter per-pixel loss into per-image means, then average over non-empty images."""
        if loss_px.numel() == 0:
            return None
        per_img = torch.zeros(B, device=device, dtype=loss_px.dtype)
        per_img.scatter_add_(0, batch_idx, loss_px)
        cnt = torch.zeros(B, device=device, dtype=loss_px.dtype)
        cnt.scatter_add_(0, batch_idx, torch.ones_like(loss_px))
        per_img = per_img / cnt.clamp(min=1.0)
        valid = cnt > 0
        return per_img[valid].mean() if valid.any() else None

    def forward(
        self,
        s: Tensor,           # [N_matched, D]    student head features
        s_backbone: Tensor,  # [N_matched, D_bb] student backbone features
        t: Tensor,           # [N_matched, D]    teacher head features
        counts: Tensor,      # [B] per-image matched voxel counts
        is_masked: Tensor | None = None,  # [N_matched] bool — True for mask-token positions
    ) -> Tensor:
        """
        Compute per-voxel DINO loss on pre-aligned student/teacher features.

        The caller is responsible for spatial matching (by coordinates) and
        gathering aligned feature pairs.  This method receives flat tensors
        concatenated over the batch, plus per-image counts for the two-stage
        (per-image mean, then batch mean) reduction.

        Args:
            s:          [N_matched, D] student features (head output or backbone)
            s_backbone: [N_matched, D_bb] student backbone features (for cov/var penalty)
            t:          [N_matched, D] teacher features (detached)
            counts:     [B] int64 tensor — number of matched voxels per image
            is_masked:  [N_matched] bool — which positions came from masked tokens;
                        when provided, masked/unmasked losses are returned as extra
                        diagnostics (no effect on scalar_loss used for training)

        Returns:
            Scalar loss value (unchanged), plus diagnostic scalars including
            loss_masked and loss_unmasked when is_masked is provided
        """
        B      = counts.shape[0]
        device = s.device

        # Lazy-initialize center on first forward call
        if self.center is None:
            self.center = torch.zeros(
                t.shape[-1],
                device=t.device,
                dtype=t.dtype,
            )

        # Centering: subtract running mean from teacher 
        # to remove the dominant direction
        if self.use_centering:
            t = t - self.center

        # L2-normalize to the unit sphere so the loss is scale-invariant.
        # normalize_features=False skips this when a projection head is used
        # (the head's internal L2 norm already puts features on the sphere).
        if self.normalize_features:
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)

        # Per-pixel cross-entropy H(P_t, P_s) = H(P_t) + KL(P_t || P_s).
        # Both student and teacher are treated as raw logits over D dimensions.
        t_prob = F.softmax(t / self.teacher_temp, dim=-1)      # [N_valid, D]
        s_logp = F.log_softmax(s / self.student_temp, dim=-1)  # [N_valid, D]
        t_logp = F.log_softmax(t / self.teacher_temp, dim=-1)  # [N_valid, D]
        loss = -(t_prob * s_logp).sum(dim=-1)                  # H(P_t, P_s) [N_valid]

        # decompose loss into teacher entropy and KL divergence for diagnostics:
        s_prob = F.softmax(s / self.student_temp, dim=-1)      # [N_valid, D]
        teacher_entropy_px = -(t_prob * t_logp).sum(dim=-1)    # H(P_t)      [N_valid]
        student_entropy_px = -(s_prob * s_logp).sum(dim=-1)    # H(P_s)      [N_valid]
        kl_px = loss - teacher_entropy_px                      # KL(P_t|P_s)[N_valid]

        # Optional covariance decorrelation penalty on raw backbone features (64-dim).
        # Computed before the projection head and before L2-normalization so the
        # penalty sees the actual feature scale and covariance structure.
        cov_penalty = None
        if self.use_cov_penalty:
            cov_penalty = self._cov_penalty(s_backbone)

        var_penalty = None
        if self.use_var_penalty:
            var_penalty = self._var_penalty(s_backbone)

        # Two-stage reduction: sum per image via scatter, divide by count, then mean.
        # Mirrors DINOv2: sum(loss * mask) / mask.sum() per image, then .mean()
        counts_dev = counts.to(device)
        batch_idx = torch.repeat_interleave(torch.arange(B, device=device), counts_dev)
        scalar_loss = self._img_mean(loss, batch_idx, B, device)

        if self.use_cov_penalty:
            scalar_loss = scalar_loss + self.cov_penalty_weight * cov_penalty
        if self.use_var_penalty:
            scalar_loss = scalar_loss + self.var_penalty_weight * var_penalty

        cov_penalty_item = cov_penalty.item() if cov_penalty is not None else None
        var_penalty_item = var_penalty.item() if var_penalty is not None else None

        t_ent = self._img_mean(teacher_entropy_px, batch_idx, B, device).item()
        s_ent = self._img_mean(student_entropy_px, batch_idx, B, device).item()
        kl    = self._img_mean(kl_px,              batch_idx, B, device).item()

        # Split loss into masked / unmasked positions — diagnostics only, no effect on training.
        loss_masked_item = loss_unmasked_item = None
        if is_masked is not None:
            with torch.no_grad():
                unmasked = ~is_masked
                m  = self._img_mean(loss[is_masked], batch_idx[is_masked], B, device)
                um = self._img_mean(loss[unmasked],  batch_idx[unmasked],  B, device)
                loss_masked_item   = m.item()  if m  is not None else None
                loss_unmasked_item = um.item() if um is not None else None

        return scalar_loss, t_ent, s_ent, kl, cov_penalty_item, var_penalty_item, loss_masked_item, loss_unmasked_item

    def _cov_penalty(self, s: Tensor) -> Tensor:
        """
        VICReg-style covariance decorrelation penalty.

        Penalizes off-diagonal entries of the feature covariance matrix, pushing
        each pair of dimensions to be uncorrelated and thus spread information
        across the full feature space.

        Args:
            s: Raw backbone features [N, D] (valid pixels only, before head and L2-norm)

        Returns:
            Scalar penalty: sum of squared off-diagonal covariance entries, divided by D
        """
        N, D = s.shape
        if N < 2:
            return s.new_tensor(0.0)
        z = s - s.mean(dim=0)                   # center features
        C = (z.T @ z) / (N - 1)                 # [D, D] covariance matrix
        # penalize only off-diagonal entries
        off_diag_sq = C.pow(2).sum() - C.diagonal().pow(2).sum()
        return off_diag_sq / D

    def _var_penalty(self, s: Tensor) -> Tensor:
        """
        VICReg-style variance penalty.

        Hinge loss that pushes the per-dimension std (over the batch of voxels) to
        stay above var_gamma, preventing any single feature dimension from collapsing
        to a constant.

        Args:
            s: Raw backbone features [N, D] (valid pixels only, before head and L2-norm)

        Returns:
            Scalar penalty: mean over dimensions of max(0, gamma - std_j)
        """
        N, D = s.shape
        if N < 2:
            return s.new_tensor(0.0)
        std = torch.sqrt(s.var(dim=0) + 1e-4)  # [D], std per feature dimension
        return torch.mean(torch.clamp(self.var_gamma - std, min=0.0))

    @torch.no_grad()
    def update_center(self, teacher_out: Voxels) -> None:
        """
        Update the running center with the EMA of teacher features at active positions.

        Should be called once per training batch, AFTER the loss backward and optimizer
        step. Must NOT be called during validation to avoid shifting the baseline with
        eval-mode teacher outputs.

        Args:
            teacher_out: Teacher backbone output (Voxels); all voxels are active by definition
        """
        teacher_flat = teacher_out.feature_tensor  # [N_active, D]

        # Sum and count rather than mean, so under DDP the centre is the mean over the
        # whole step and not one per rank -- per-rank centres would drift apart and the
        # ranks would stop optimising the same objective.
        #
        # The empty-batch check happens AFTER the collectives on purpose: returning
        # early on a locally empty batch would leave the other ranks waiting in an
        # all-reduce that never comes. Sum and count of an empty [0, D] are well
        # defined, and `count` is global, so every rank returns together or not at all.
        distributed = dist.is_available() and dist.is_initialized()

        feat_sum = teacher_flat.sum(dim=0)                       # [D]; zeros if empty
        count = torch.tensor([teacher_flat.shape[0]],
                             device=feat_sum.device, dtype=feat_sum.dtype)
        if distributed:
            dist.all_reduce(feat_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        if count.item() == 0:
            return

        batch_mean = feat_sum / count  # [D]

        if self.center is None:
            self.center = batch_mean.clone()
        else:
            self.center = self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_mean


# ---------------------------------------------------------------------------
# Reconstruction losses (objective="mae")
#
# The mae objective has no teacher: both terms are supervised directly by the input.
# Charge asks "what value was here?" at the pixels masking removed; occupancy asks
# "was anything here?" over a candidate set that deliberately contains empties too,
# because a question whose answer is always yes teaches nothing.
# ---------------------------------------------------------------------------

def two_stage_mean(per_voxel: Tensor, counts: Tensor) -> Tensor:
    """Mean within each image, then mean over images -- the reduction PixelDINOLoss uses.

    Per-image normalisation first is what makes the loss weights comparable across
    batches whose images differ wildly in how much charge they hold; a flat mean would
    let the busiest image set the scale.

    `per_voxel` must be grouped by image, matching `counts`. Images contributing nothing
    are dropped from the outer mean rather than averaged in as zeros. Returns a scalar
    that is still attached to the graph even when everything is empty, so the term can
    always be summed into the total without breaking backward.
    """
    if per_voxel.numel() == 0:
        return per_voxel.sum()
    B = counts.shape[0]
    device = per_voxel.device
    batch_idx = torch.repeat_interleave(
        torch.arange(B, device=device), counts.to(device))
    per_img = torch.zeros(B, device=device, dtype=per_voxel.dtype)
    per_img.scatter_add_(0, batch_idx, per_voxel)
    counts_f = counts.to(device=device, dtype=per_voxel.dtype)
    per_img = per_img / counts_f.clamp(min=1.0)
    valid = counts_f > 0
    if not bool(valid.any()):
        return per_voxel.sum() * 0.0
    return per_img[valid].mean()


def charge_loss(pred: Tensor, target: Tensor, counts: Tensor) -> Tensor:
    """L1 on the charge that masking removed, normalised per image.

    Both sides are already in the normalizer's log space -- FeatureLogTransform runs
    before masking -- so a plain L1 here is an L1 on log-charge, and no further
    transform belongs in this function.
    """
    if pred.numel() == 0:
        return pred.sum()
    return two_stage_mean((pred - target).abs(), counts)


def occupancy_loss(logits: Tensor, target: Tensor, counts: Tensor,
                   alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    """Focal binary cross-entropy over occupancy candidates, normalised per image.

    The candidate set is mostly empty, so a plain BCE converges to "predict empty
    everywhere" and stays there. Focal loss handles that inside the term: `gamma`
    discounts examples the model already gets right, `alpha` reweights the positive
    class. Defaults are the standard ones and are deliberately not exposed yet -- the
    grown candidate set measures a few percent positive, which is close to what these
    values were designed for.
    """
    if logits.numel() == 0:
        return logits.sum()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)          # probability of the true class
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return two_stage_mean(alpha_t * (1.0 - p_t).pow(gamma) * bce, counts)
