"""Sparse-aware masking strategies for DINO student augmentation.

Every masker here answers the same call and returns the same five things, so a caller
never branches on which one it was handed:

    student_voxels, masked_coords, masked_feats, cand_coords, occ_targets = masker(voxels)

The last three are None when that masker was not asked for them. `masked_feats` is the
charge-reconstruction target; `cand_coords` / `occ_targets` are the occupancy candidate
set and its labels, which only a masker whose removals have known geometry can build --
see SparseRegionMasker.
"""

import math
from typing import List, Tuple

import torch
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures


def batch_index(voxels: Voxels) -> torch.Tensor:
    """Per-voxel batch id [Ntot], on the voxel device, derived from offsets."""
    counts = (voxels.offsets[1:] - voxels.offsets[:-1]).to(
        device=voxels.coordinate_tensor.device, dtype=torch.int64)
    return torch.repeat_interleave(torch.arange(counts.shape[0], device=counts.device), counts)


def segment_rank(batch_idx_sorted: torch.Tensor, counts_per_b: torch.Tensor) -> torch.Tensor:
    """0-based rank within each batch segment, for an array already grouped by batch.

    Turns a "first k of each image" selection into one vectorised comparison, which is
    what lets the whole batch be masked without a per-image loop.
    """
    seg_start = torch.cat([counts_per_b.new_zeros(1), counts_per_b.cumsum(0)[:-1]])
    M = batch_idx_sorted.shape[0]
    return torch.arange(M, device=batch_idx_sorted.device) - seg_start[batch_idx_sorted]


def assemble_masked_voxels(voxels: Voxels, batch_idx: torch.Tensor, masked: torch.Tensor):
    """Turn a per-voxel boolean mask into (student_voxels, masked_coords, masked_feats).

    The input is batch-ordered, so a boolean index preserves that grouping and the
    student's offsets are just the cumulative kept counts. Splitting the dropped rows
    per image costs the single host sync in this function.
    """
    B = len(voxels.offsets) - 1
    coords, feats = voxels.coordinate_tensor, voxels.feature_tensor
    coord_dim, feat_dim = coords.shape[1], feats.shape[1]
    keep = ~masked

    keep_counts = torch.bincount(batch_idx[keep], minlength=B)
    new_offsets = torch.cat([keep_counts.new_zeros(1), keep_counts.cumsum(0)])
    student_voxels = Voxels(
        batched_coordinates=IntCoords(coords[keep], offsets=new_offsets),
        batched_features=CatFeatures(feats[keep], offsets=new_offsets),
        offsets=new_offsets,
    )

    split_sizes = torch.bincount(batch_idx[masked], minlength=B).tolist()
    masked_coords = list(torch.split(coords[masked], split_sizes))
    masked_feats = list(torch.split(feats[masked], split_sizes))
    if len(masked_coords) != B:          # splitting a 0-row tensor yields one piece, not B
        masked_coords = [coords.new_zeros(0, coord_dim) for _ in range(B)]
        masked_feats = [feats.new_zeros(0, feat_dim) for _ in range(B)]
    return student_voxels, masked_coords, masked_feats


def cap_negatives(cand_b: torch.Tensor, occ: torch.Tensor, B: int,
                  neg_per_pos: float = None, max_neg: int = None) -> torch.Tensor:
    """Sub-sample the empty candidates to bound memory, keeping every positive.

    Densifying whole cells produces far more empties than actives -- that imbalance is
    the point, since an occupancy question whose answer is always yes teaches nothing --
    but it is also what drives memory, and it is the empties that are cheap to give up.

    The per-image negative budget is the smallest of whichever caps are set:
      neg_per_pos K -- keep K empties per positive, holding the class balance fixed
        whatever the cell size. Memory then scales with each image's positive count, so
        the busiest image in the batch sets the peak.
      max_neg N -- keep at most N empties per image, whatever the positives. Memory
        becomes positives + N, additive rather than multiplicative, so a dense image can
        no longer blow the budget. The realised ratio then floats: sparse images get more
        empties per positive (each is worth more), busy ones fewer (they have positives
        to spare).

    Returns a boolean keep-mask aligned to `cand_b`. With no cap set nothing is dropped.
    """
    is_pos = occ > 0.5
    keep = torch.ones(cand_b.shape[0], dtype=torch.bool, device=cand_b.device)
    neg_idx = (~is_pos).nonzero(as_tuple=False).squeeze(1)
    if neg_idx.numel() == 0:
        return keep

    budget = None
    if neg_per_pos is not None:
        n_pos = torch.bincount(cand_b[is_pos], minlength=B)
        budget = (float(neg_per_pos) * n_pos.double()).round().long()
    if max_neg is not None:
        abs_cap = torch.full((B,), int(max_neg), dtype=torch.int64, device=cand_b.device)
        budget = abs_cap if budget is None else torch.minimum(budget, abs_cap)
    if budget is None:
        return keep

    nb = cand_b[neg_idx]
    neg_counts = torch.bincount(nb, minlength=B)
    # rank the empties of each image in a random order, keep the first `budget` of them
    order = (nb.double() + torch.rand(neg_idx.shape[0], device=cand_b.device).double()).argsort()
    rank = segment_rank(nb[order], neg_counts)
    keep[neg_idx[order][~(rank < budget[nb[order]])]] = False
    return keep


def label_candidates(cand_coords: torch.Tensor, cand_b: torch.Tensor,
                     masked_coords: torch.Tensor, masked_b: torch.Tensor,
                     B: int, image_w: int, image_h: int, stride: int = 1,
                     neg_per_pos: float = None, max_neg: int = None):
    """Label occupancy candidates and split them per image.

    A candidate is positive exactly when it coincides with a voxel masking removed, so
    the label is membership in the removed set and nothing else is consulted -- a
    candidate cannot be positive because of structure the student can still see.

    `stride` > 1 maps candidates and removed voxels onto a coarser grid by floor division
    and deduplicates, which turns the label into a block-OR ("did this cell hold anything
    active?") over ~1/stride^2 as many candidates. The returned coordinates are in those
    coarse units, matching the decoder level the candidates are injected at.

    Membership is tested with batch-unique flat keys so one call covers the whole batch.

    Returns (candidate_coords_per_batch, occ_target_per_batch), both lists of B tensors.
    """
    device = cand_coords.device
    s = int(stride) if stride else 1

    if s > 1:
        cand_coords = torch.div(cand_coords, s, rounding_mode="floor")
        if masked_coords.shape[0] > 0:
            masked_coords = torch.div(masked_coords, s, rounding_mode="floor")
        W = (int(image_w) + s - 1) // s
        H = (int(image_h) + s - 1) // s
        # dedupe: several full-resolution candidates collapse onto one coarse cell
        k = cand_b * (W * H + W) + cand_coords[:, 1].long() * W + cand_coords[:, 0].long()
        srt = torch.argsort(k, stable=True)
        first = torch.ones(k.shape[0], dtype=torch.bool, device=device)
        first[1:] = k[srt][1:] != k[srt][:-1]
        uniq = torch.zeros(k.shape[0], dtype=torch.bool, device=device)
        uniq[srt[first]] = True
        cand_coords, cand_b = cand_coords[uniq], cand_b[uniq]
    else:
        W, H = int(image_w), int(image_h)
    STRIDE = W * H + W                      # exceeds any in-image key y*W + x

    # group by image so torch.split yields the per-image tensors
    order = torch.argsort(cand_b, stable=True)
    cand_b, cand_coords = cand_b[order], cand_coords[order]

    cand_key = cand_b * STRIDE + cand_coords[:, 1].long() * W + cand_coords[:, 0].long()
    if masked_coords.shape[0] > 0:
        m_key = masked_b * STRIDE + masked_coords[:, 1].long() * W + masked_coords[:, 0].long()
        occ = torch.isin(cand_key, m_key).float()
    else:
        occ = torch.zeros(cand_coords.shape[0], device=device)

    if neg_per_pos is not None or max_neg is not None:
        keep = cap_negatives(cand_b, occ, B, neg_per_pos=neg_per_pos, max_neg=max_neg)
        cand_b, cand_coords, occ = cand_b[keep], cand_coords[keep], occ[keep]

    counts = torch.bincount(cand_b, minlength=B).tolist()
    cand_list = list(torch.split(cand_coords, counts))
    occ_list = list(torch.split(occ, counts))
    if len(cand_list) != B:
        cand_list = [cand_coords.new_zeros(0, 2) for _ in range(B)]
        occ_list = [occ.new_zeros(0) for _ in range(B)]
    return cand_list, occ_list


class SparseVoxelMasker:
    """
    Masks active voxels for the DINO student by removing entries from a Voxels object.

    For each image in a batch:
    1. Randomly select a fraction of active voxels to keep (1 - mask_ratio)
    2. Returns: 
        - reduced student Voxels
        - (x, y) coordinates of the masked voxels per batch item

    The masked_coords can be used to inject learnable mask tokens at the dropped positions,
    so the student can predict teacher features there.
    
    NOTE: Student/teacher alignment in the loss continues to use
    coordinate intersection via match_and_gather -- no index bookkeeping needed.
    """

    def __init__(self, mask_ratio: float = 0.5, seed: int = None,
                 return_masked_feats: bool = False):
        """
        Args:
            mask_ratio: Fraction of active voxels to mask (0.0 to 1.0)
            seed: Optional random seed for reproducibility
            return_masked_feats: also collect the dropped voxels' features. Needed by the
                mae objective, whose charge term regresses exactly those values; off
                otherwise so the work is not done for nothing.
        """
        self.mask_ratio = mask_ratio
        self.return_masked_feats = return_masked_feats
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, voxels: Voxels) -> Tuple[Voxels, List[torch.Tensor]]:
        """
        Apply masking to a batched Voxels object.

        Args:
            voxels: Batched Voxels with batch_size B

        Returns:
            student_voxels:          Voxels with ~(1 - mask_ratio) of the original voxels
            masked_coords_per_batch: List of B tensors, each [N_masked_b, 2] holding
                                     the (x, y) integer coords of voxels dropped from
                                     that image. Same dtype/device as input coords.
            masked_feats_per_batch:  List of B tensors [N_masked_b, F] with those voxels'
                                     features, or None when return_masked_feats is off.
                                     Already in the normalizer's log space, since
                                     FeatureLogTransform runs before masking.
            cand_coords, occ_targets: always None here -- see the module docstring. The
                                     arity of the return never changes, only which
                                     elements are None, so no caller branches on mode.
        """
        
        #number of images in the batch
        B = len(voxels.offsets) - 1 

        device = voxels.coordinate_tensor.device

        # number of coordinate dimensions (2D or 3D)
        coord_dim = voxels.coordinate_tensor.shape[1]

        masked_coords_per_batch = []
        masked_feats_per_batch = [] if self.return_masked_feats else None
        coords_list = []
        feats_list  = []

        # for each image in the batch
        for b in range(B):

            # find start/end indices of the voxels for image b
            # and count them 
            start = int(voxels.offsets[b])
            end   = int(voxels.offsets[b + 1])
            N     = end - start

            if N == 0:
                # empty image: append empty entries to ALL lists so that
                # student_voxels.offsets stays length B+1 and aligns with
                # masked_coords_per_batch (which is always length B).
                masked_coords_per_batch.append(
                    voxels.coordinate_tensor.new_zeros(0, coord_dim)
                )
                if masked_feats_per_batch is not None:
                    masked_feats_per_batch.append(
                        voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1])
                    )
                coords_list.append(voxels.coordinate_tensor.new_zeros(0, coord_dim))
                feats_list.append(voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1]))
                continue

            # max(1, ...) guarantees at least one voxel is always kept
            n_keep   = max(1, N - int(N * self.mask_ratio))

            # generate a random permutation of the voxel indices, then 
            # keep the first n_keep and drop the the rest.
            # sort the indices to preserve spatial order.
            perm     = torch.randperm(N, device=device)
            keep     = perm[:n_keep].sort().values    
            masked   = perm[n_keep:].sort().values

            masked_coords_per_batch.append(voxels.coordinate_tensor[start:end][masked])
            if masked_feats_per_batch is not None:
                masked_feats_per_batch.append(voxels.feature_tensor[start:end][masked])
            coords_list.append(voxels.coordinate_tensor[start:end][keep])
            feats_list.append(voxels.feature_tensor[start:end][keep])

        # number of surviving voxels per batch item
        n_kept      = [c.shape[0] for c in coords_list] # 
        counts      = torch.tensor(n_kept, dtype=torch.int64, device=device)
        
        # offsets for the new batched Voxels object
        new_offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=device),
            counts.cumsum(0),
        ])

        new_coords = (torch.cat(coords_list, dim=0) if coords_list
                      else voxels.coordinate_tensor.new_zeros(0, coord_dim))
        new_feats  = (torch.cat(feats_list,  dim=0) if feats_list
                      else voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1]))

        # package new Voxels object
        student_voxels = Voxels(
            batched_coordinates=IntCoords(new_coords, offsets=new_offsets),
            batched_features=CatFeatures(new_feats, offsets=new_offsets),
            offsets=new_offsets,
        )

        # None, None: neither masker knows the geometry of what it removed well enough to
        # enumerate occupancy candidates around it; that set is grown in the backbone instead.
        return student_voxels, masked_coords_per_batch, masked_feats_per_batch, None, None


class SparseBlockMasker:
    """
    Block-masks active voxels for the DINO student by removing entire spatial regions.

    For each image in the batch:
    1. Estimate K block centers needed to cover mask_ratio of voxels using the
       geometric coverage formula: E[fraction] = 1 - (1 - p)^K, where
       p = min(block_area, N) / N is the expected per-block coverage rate.
    2. Sample K random centers from active voxels (no replacement).
    3. Mask all voxels within [±win_ch, ±win_tick] of any center in one vectorised op.
    4. Remove masked voxels from the student input; return their coordinates.

    Drop-in replacement for SparseVoxelMasker — same (Voxels) → (Voxels, List[coords])
    interface, so match_and_gather, encode_student, and the loss need no changes.

    Because blocks may overlap, the actual masked fraction varies around mask_ratio.
    """

    def __init__(self, mask_ratio: float = 0.5, win_ch: int = 5, win_tick: int = 5,
                 return_masked_feats: bool = False):
        """
        Args:
            mask_ratio: Target fraction of active voxels to mask (0.0 to 1.0).
            win_ch:     Half-window radius in the channel direction (voxels).
            win_tick:   Half-window radius in the tick direction (voxels).
            return_masked_feats: also collect the dropped voxels' features (mae's charge
                target); see SparseVoxelMasker for the contract.
        """
        self.return_masked_feats = return_masked_feats
        self.mask_ratio  = mask_ratio
        self.win_ch      = win_ch
        self.win_tick    = win_tick
        self._block_area = (2 * win_ch + 1) * (2 * win_tick + 1)
        # Adaptive estimate of the effective per-block coverage rate p.
        # Initialised to None (falls back to the analytic formula on the first call)
        # then updated via EMA from observed coverage — no extra GPU syncs needed.
        self._p_eff: float = None

    def __call__(self, voxels: Voxels) -> Tuple[Voxels, List[torch.Tensor]]:
        """
        Apply block masking to a batched Voxels object.

        Args:
            voxels: Batched Voxels with batch_size B.

        Returns:
            student_voxels:          Voxels with block-masked voxels removed.
            masked_coords_per_batch: List of B tensors, each [N_masked_b, 2] holding
                                     the (channel, tick) coords of removed voxels.
            masked_feats_per_batch:  Those voxels' features, or None when
                                     return_masked_feats is off.
            cand_coords, occ_targets: always None here. Same contract as
                                     SparseVoxelMasker -- the arity never changes.
        """
        # Reset per-call so the EMA from one crop type (e.g. sparse global crop)
        # does not contaminate K estimates for a different crop type (dense local
        # crop) in the next call.  The EMA still converges within a call across
        # batch elements, which is when it is actually useful.
        self._p_eff = None

        B         = len(voxels.offsets) - 1
        device    = voxels.coordinate_tensor.device
        coord_dim = voxels.coordinate_tensor.shape[1]

        masked_coords_per_batch = []
        masked_feats_per_batch = [] if self.return_masked_feats else None
        coords_list = []
        feats_list  = []

        for b in range(B):
            start = int(voxels.offsets[b])
            end   = int(voxels.offsets[b + 1])
            N     = end - start

            if N == 0:
                # empty image: append empty entries to ALL lists so that
                # student_voxels.offsets stays length B+1 and aligns with
                # masked_coords_per_batch (which is always length B).
                masked_coords_per_batch.append(
                    voxels.coordinate_tensor.new_zeros(0, coord_dim)
                )
                if masked_feats_per_batch is not None:
                    masked_feats_per_batch.append(
                        voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1])
                    )
                coords_list.append(voxels.coordinate_tensor.new_zeros(0, coord_dim))
                feats_list.append(voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1]))
                continue

            coords_i = voxels.coordinate_tensor[start:end]  # (N, 2)
            feats_i  = voxels.feature_tensor[start:end]     # (N, C)

            # Estimate how many block centers K are needed to cover mask_ratio of voxels.
            # Geometric coverage model: E[covered] = 1 - (1-p)^K.
            # p is the effective per-block coverage rate: ideally block_area/N, but for
            # sparse data the active voxels per window are far fewer than block_area, so
            # we learn p from observed coverage via EMA (all CPU math, no extra GPU sync).
            p_formula = min(self._block_area, N) / N
            p = self._p_eff if self._p_eff is not None else p_formula
            p = max(1e-6, min(p, 1.0 - 1e-6))
            K = min(
                math.ceil(math.log(1.0 - self.mask_ratio) / math.log(1.0 - p)),
                N - 1,
            )

            center_idx = torch.randperm(N, device=device)[:K]
            centers    = coords_i[center_idx]                           # (K, 2)
            diff       = coords_i.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, 2)
            mask_bool  = (
                (diff[..., 0].abs() <= self.win_ch) &
                (diff[..., 1].abs() <= self.win_tick)
            ).any(dim=1)                                                # (N,) — no sync

            # Guard: guarantee at least one voxel survives (rare with K ≤ N-1).
            if mask_bool.all():
                mask_bool[torch.randint(0, N, (1,), device=device)] = False

            keep   = (~mask_bool).nonzero(as_tuple=False).squeeze(1)
            masked = mask_bool.nonzero(as_tuple=False).squeeze(1)

            # Update EMA of effective p from observed coverage (keep.shape[0] is a
            # Python int after nonzero — no extra GPU sync).
            actual = (N - keep.shape[0]) / N
            if K > 0 and 0.0 < actual < 1.0:
                p_measured = 1.0 - (1.0 - actual) ** (1.0 / K)
                self._p_eff = (p_measured if self._p_eff is None
                               else 0.9 * self._p_eff + 0.1 * p_measured)

            masked_coords_per_batch.append(coords_i[masked])
            if masked_feats_per_batch is not None:
                masked_feats_per_batch.append(feats_i[masked])
            coords_list.append(coords_i[keep])
            feats_list.append(feats_i[keep])

        n_kept      = [c.shape[0] for c in coords_list]
        counts      = torch.tensor(n_kept, dtype=torch.int64, device=device)
        new_offsets = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=device),
            counts.cumsum(0),
        ])

        new_coords = (torch.cat(coords_list, dim=0) if coords_list
                      else voxels.coordinate_tensor.new_zeros(0, coord_dim))
        new_feats  = (torch.cat(feats_list,  dim=0) if feats_list
                      else voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1]))

        student_voxels = Voxels(
            batched_coordinates=IntCoords(new_coords, offsets=new_offsets),
            batched_features=CatFeatures(new_feats, offsets=new_offsets),
            offsets=new_offsets,
        )

        # None, None: neither masker knows the geometry of what it removed well enough to
        # enumerate occupancy candidates around it; that set is grown in the backbone instead.
        return student_voxels, masked_coords_per_batch, masked_feats_per_batch, None, None


class SparseRegionMasker:
    """
    Masks whole cells of a fixed grid laid over the image canvas.

    A cell_w x cell_h grid covers the (image_w, image_h) canvas. Only cells holding at
    least one active voxel are eligible -- the detector image is ~99.8% empty, so
    selecting empty cells would spend the mask budget on nothing. Two flavours:

      "wipe":      remove every voxel of the chosen cells. A hard, contiguous occlusion,
                   so the student must infer structure it cannot see any part of.
      "randomize": choose the same cells but drop voxels inside them at rate r2 -- a
                   softer, localised thinning that leaves partial structure behind.

    Unlike the other maskers this one knows exactly which region it emptied, which is
    what lets it enumerate occupancy candidates: every pixel of a wiped cell is a
    candidate, and the ones that held charge are the positives. Under "wipe" that
    labelling is exact, because only cells left completely empty are enumerated -- a cell
    that kept anything could answer the question early. Under "randomize" the surviving
    voxels inside a chosen cell are active pixels that the label calls empty, so the
    target leaks by construction; the flavour is kept for masking studies and rejected
    upstream as a reconstruction target.
    """

    def __init__(self, image_w: int, image_h: int,
                 cell_w: int = 70, cell_h: int = 100,
                 r1: float = 0.5, r2: float = 0.75,
                 flavor: str = "wipe", wipe_max: float = 0.75,
                 return_masked_feats: bool = False,
                 build_candidates: bool = False, cand_stride: int = 2,
                 neg_per_pos: float = None, max_neg: int = None):
        """
        Args:
            image_w, image_h: canvas size in (wire, tick) units. The grid is defined on
                the canvas rather than on the data, so these are required.
            cell_w, cell_h:   cell size in the same units. Both must divide the canvas
                evenly and be even numbers -- see the checks below for why.
            r1:               fraction of active cells to select ("randomize" only).
            r2:               within-cell voxel dropout rate ("randomize" only).
            flavor:           "wipe" or "randomize".
            wipe_max:         "wipe" only -- ceiling on the fraction of voxels removed.
                Whole cells are wiped in random order until the next one would cross the
                ceiling, so the student always keeps at least (1 - wipe_max) of its
                voxels and a single huge cell cannot empty the image.
            return_masked_feats: also collect the removed voxels' features (the charge
                target); same contract as the other maskers.
            build_candidates: also enumerate the occupancy candidate set and its labels.
            cand_stride:      candidates are reported on this coarser grid (coordinates
                floor-divided), matching the decoder level they are injected at.
            neg_per_pos, max_neg: per-image caps on the empty candidates; see
                cap_negatives. Densifying whole cells is what makes these necessary.
        """
        assert flavor in ("wipe", "randomize"), f"unknown flavor {flavor!r}"
        # An indivisible canvas leaves partial edge cells, which densify to a different
        # candidate count than interior ones and silently skew the positive rate.
        assert image_w % cell_w == 0 and image_h % cell_h == 0, (
            f"cell {cell_w}x{cell_h} does not tile a {image_w}x{image_h} canvas evenly"
        )
        # An odd cell dimension puts the coarse footprint of a wiped cell half a cell out
        # of step with the grid, so a candidate can land on a coarse cell that still holds
        # surviving charge -- a false negative the model is then trained on.
        assert cell_w % cand_stride == 0 and cell_h % cand_stride == 0, (
            f"cell {cell_w}x{cell_h} must divide by the candidate stride {cand_stride}"
        )
        self.image_w = image_w
        self.image_h = image_h
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.r1 = r1
        self.r2 = r2
        self.flavor = flavor
        self.wipe_max = wipe_max
        self.return_masked_feats = return_masked_feats
        self.build_candidates = build_candidates
        self.cand_stride = cand_stride
        self.neg_per_pos = neg_per_pos
        self.max_neg = max_neg
        self.n_cols = image_w // cell_w
        self.n_rows = image_h // cell_h
        self.n_cells = self.n_cols * self.n_rows

    def _densify(self, take_cell, ucells, device, coord_dtype):
        """Enumerate every pixel of each selected cell: the occupancy candidate set.

        One offset template is built for a cell and broadcast over all of them, so the
        cost is a single expand rather than a loop over cells. The canvas divides evenly
        by construction, so every cell contributes exactly cell_w * cell_h candidates and
        no bounds filtering is needed.

        Returns (cand_coords [N, 2], cand_b [N]) at full resolution.
        """
        taken = ucells[take_cell]
        if taken.shape[0] == 0:
            return (torch.zeros(0, 2, dtype=coord_dtype, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device))
        cb = torch.div(taken, self.n_cells, rounding_mode="floor")
        local = taken - cb * self.n_cells
        base_x = (local % self.n_cols) * self.cell_w
        base_y = torch.div(local, self.n_cols, rounding_mode="floor") * self.cell_h

        ox, oy = torch.meshgrid(torch.arange(self.cell_w, device=device),
                                torch.arange(self.cell_h, device=device), indexing="ij")
        ox, oy = ox.reshape(-1), oy.reshape(-1)
        cand_x = (base_x[:, None] + ox[None, :]).reshape(-1)
        cand_y = (base_y[:, None] + oy[None, :]).reshape(-1)
        cand_b = cb[:, None].expand(-1, ox.shape[0]).reshape(-1)
        return torch.stack([cand_x, cand_y], dim=1).to(coord_dtype), cand_b

    def __call__(self, voxels: Voxels):
        """
        Args:
            voxels: batched full-image Voxels, B items.

        Returns:
            student_voxels:          Voxels with the masked voxels removed.
            masked_coords_per_batch: B tensors [N_masked_b, 2] of removed (x, y) coords.
            masked_feats_per_batch:  their features, or None when not requested.
            cand_coords_per_batch:   B tensors [N_cand_b, 2] of occupancy candidates in
                                     cand_stride units, or None when not requested.
            occ_target_per_batch:    B tensors [N_cand_b] of 1.0/0.0 labels, or None.
        """
        B = len(voxels.offsets) - 1
        coords = voxels.coordinate_tensor
        device = coords.device
        Ntot = coords.shape[0]

        if Ntot == 0:
            empty_c = [coords.new_zeros(0, coords.shape[1]) for _ in range(B)]
            empty_f = [voxels.feature_tensor.new_zeros(0, voxels.feature_tensor.shape[1])
                       for _ in range(B)]
            return (voxels, empty_c,
                    empty_f if self.return_masked_feats else None,
                    empty_c if self.build_candidates else None,
                    [coords.new_zeros(0).float() for _ in range(B)] if self.build_candidates else None)

        batch_idx = batch_index(voxels)
        N = torch.bincount(batch_idx, minlength=B)          # voxels per image

        # Global cell id per voxel, offset by image so cells never collide across the batch.
        cx = torch.div(coords[:, 0], self.cell_w, rounding_mode="floor")
        cy = torch.div(coords[:, 1], self.cell_h, rounding_mode="floor")
        cell_global = batch_idx * self.n_cells + (cy * self.n_cols + cx)

        ucells, inv, ucounts = torch.unique(cell_global, return_inverse=True, return_counts=True)
        cell_batch = torch.div(ucells, self.n_cells, rounding_mode="floor")
        n_u = ucells.shape[0]
        n_active = torch.bincount(cell_batch, minlength=B)  # active cells per image

        # Random cell order within each image: sort by (image, random key) in one pass.
        ckey = torch.rand(n_u, device=device)
        corder = (cell_batch.double() + ckey.double()).argsort()
        cb_sorted = cell_batch[corder]
        rank_sorted = segment_rank(cb_sorted, n_active)

        if self.flavor == "wipe":
            # Take whole cells in that random order while the running voxel count stays
            # under the ceiling. The first cell is always taken, so an image whose single
            # densest cell already exceeds the ceiling still gets masked at all.
            csum = ucounts[corder].cumsum(0)
            seg_start = torch.cat([n_active.new_zeros(1), n_active.cumsum(0)[:-1]])
            before = torch.cat([csum.new_zeros(1), csum])[seg_start[cb_sorted]]
            cum_within = csum - before
            cap = self.wipe_max * N.double()
            take_sorted = (cum_within <= cap[cb_sorted]) | (rank_sorted == 0)
            take_cell = torch.zeros(n_u, dtype=torch.bool, device=device)
            take_cell[corder] = take_sorted
            dense_cell = take_cell
            masked = take_cell[inv]

            # That forced first cell can overshoot the ceiling on its own. Randomly
            # release the surplus so the student is never left with nothing to encode.
            n_cap = cap.long()
            mc = torch.bincount(batch_idx[masked], minlength=B)
            surplus = (mc - n_cap).clamp(min=0)
            if bool((surplus > 0).any()):
                m_idx = masked.nonzero(as_tuple=False).squeeze(1)
                mb = batch_idx[m_idx]
                morder = (mb.double() + torch.rand(m_idx.shape[0], device=device).double()).argsort()
                mrank = segment_rank(mb[morder], mc)
                masked[m_idx[morder][mrank < surplus[mb[morder]]]] = False

            # Densify only the cells that ended up completely empty. Releasing the
            # surplus puts charge back inside a cell that was taken, and that charge is
            # visible to the student -- densifying such a cell would label a pixel it can
            # still see as empty. Recomputed from the FINAL mask rather than from the
            # selection, so the exactness the occupancy label claims actually holds.
            has_survivor = torch.zeros(n_u, dtype=torch.bool, device=device)
            has_survivor[inv[~masked]] = True
            dense_cell = take_cell & ~has_survivor
        else:
            # randomize: take r1 of the active cells, then drop r2 of the voxels inside them
            n_sel = torch.minimum(n_active,
                                  (self.r1 * n_active.double()).round().long().clamp(min=1))
            sel_sorted = rank_sorted < n_sel[cb_sorted]
            sel_cell = torch.zeros(n_u, dtype=torch.bool, device=device)
            sel_cell[corder] = sel_sorted
            dense_cell = sel_cell
            masked = sel_cell[inv] & (torch.rand(Ntot, device=device) < self.r2)

        student, masked_coords, masked_feats = assemble_masked_voxels(voxels, batch_idx, masked)
        if not self.return_masked_feats:
            masked_feats = None
        if not self.build_candidates:
            return student, masked_coords, masked_feats, None, None

        cand_coords, cand_b = self._densify(dense_cell, ucells, device, coords.dtype)
        m_idx = masked.nonzero(as_tuple=False).squeeze(1)
        cand_list, occ_list = label_candidates(
            cand_coords, cand_b, coords[m_idx], batch_idx[m_idx], B,
            self.image_w, self.image_h, stride=self.cand_stride,
            neg_per_pos=self.neg_per_pos, max_neg=self.max_neg,
        )
        return student, masked_coords, masked_feats, cand_list, occ_list
