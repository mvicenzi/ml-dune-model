"""
test_mae_components.py
──────────────────────
The pieces the mae objective is built from, checked without a GPU.

mae reconstructs what masking removed. That needs three things this file exercises:

  1. **Candidate growth.** Occupancy asks "did this cell hold charge?", which only teaches
     something if the candidate set contains genuine empties. They are grown from the
     structure that survived masking -- coordinates doubled from the bottleneck scale,
     then a 3x3 halo -- so the empties sit right against real structure and nothing about
     the answer leaks into the choice of question.
  2. **Reading a prediction at a coordinate.** Predictions are matched to requests by
     coordinate, never by row position: no sparse-convolution API promises rows come back
     in the order they went in, and a target misaligned by one row trains perfectly
     happily while meaning nothing.
  3. **The occupancy label**, which is membership of a candidate cell in the pre-mask
     active set at the candidate's own scale.

Plus the structural consequence of the objective: mae builds reconstruction heads and no
teacher, and `model.train()` must survive that -- the override used to call
`self.teacher.eval()` unconditionally, which would fail on the very first training step.

Everything here is plain torch or module construction, so no GPU is needed. What is NOT
covered: an actual forward pass, which needs sparse convolutions and therefore a GPU.

Run:  python -u tests/test_mae_components.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

from models.minkunet_attention import MinkUNetSparseAttentionMAE as MAE
from dino.model import DINODuneModel, gather_at_coords, occupancy_target


def make_vox(coords: torch.Tensor, values: torch.Tensor | None = None) -> Voxels:
    offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)
    feats = values if values is not None else torch.zeros(coords.shape[0], 1)
    return Voxels(
        batched_coordinates=IntCoords(coords.to(torch.int32), offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


def pairs(coords: torch.Tensor) -> set:
    return {(int(r[0]), int(r[1])) for r in coords}


# ── 1. candidate growth ──────────────────────────────────────────────────────

def check_growth_doubles_then_haloes() -> None:
    """One surviving cell becomes a 3x3 ring around its doubled coordinate."""
    grown = MAE._grow_from_bottleneck(make_vox(torch.tensor([[5, 5]], dtype=torch.int64)))[0]
    assert pairs(grown) == {(x, y) for x in (9, 10, 11) for y in (9, 10, 11)}, pairs(grown)


def check_growth_reach_scales_with_iters() -> None:
    """Each extra pass reaches one more cell out; one pass only ever covers the rim."""
    b = make_vox(torch.tensor([[5, 5]], dtype=torch.int64))
    sizes = [MAE._grow_from_bottleneck(b, iters=k)[0].shape[0] for k in (1, 2, 3)]
    assert sizes == [9, 25, 49], sizes


def check_growth_is_deduplicated() -> None:
    """Repeated dilation must not return 9^iters copies of the same cells."""
    b = make_vox(torch.tensor([[5, 5], [6, 6], [20, 20]], dtype=torch.int64))
    for k in (1, 2, 3):
        g = MAE._grow_from_bottleneck(b, iters=k)[0]
        assert g.shape[0] == torch.unique(g, dim=0).shape[0], f"iters={k} returned duplicates"


def check_growth_never_leaves_the_canvas() -> None:
    """A cell on the edge must not grow to a negative coordinate."""
    g = MAE._grow_from_bottleneck(make_vox(torch.tensor([[0, 0]], dtype=torch.int64)))[0]
    assert int(g.min()) >= 0, g


def check_growth_handles_empty_image() -> None:
    empty = make_vox(torch.zeros(0, 2, dtype=torch.int64))
    assert MAE._grow_from_bottleneck(empty)[0].shape[0] == 0


# ── 2. reading predictions at coordinates ────────────────────────────────────

def check_gather_matches_by_coordinate_not_row() -> None:
    """Requests in a different order than the output must still read the right values."""
    vox = make_vox(torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.int64),
                   torch.tensor([[10.], [20.], [30.]]))
    pred, target, counts = gather_at_coords(
        vox,
        [torch.tensor([[3, 3], [1, 1]], dtype=torch.int64)],   # reversed on purpose
        [torch.tensor([300., 100.])],
    )
    assert pred.tolist() == [30., 10.], pred.tolist()
    assert target.tolist() == [300., 100.], target.tolist()
    assert counts.tolist() == [2]


def check_gather_drops_target_with_missing_prediction() -> None:
    """An unmatched request drops its target too, or the two lists shift apart."""
    vox = make_vox(torch.tensor([[1, 1]], dtype=torch.int64), torch.tensor([[10.]]))
    pred, target, counts = gather_at_coords(
        vox,
        [torch.tensor([[9, 9], [1, 1]], dtype=torch.int64)],
        [torch.tensor([999., 100.])],
    )
    assert pred.tolist() == [10.], pred.tolist()
    assert target.tolist() == [100.], target.tolist()
    assert counts.tolist() == [1], "the loss denominator must count matches, not requests"


def check_gather_keeps_labels_on_their_own_coords() -> None:
    """The pairing bug that trains happily on wrong answers.

    Injection drops any candidate the skip already carries, so the coordinate list the
    backbone reports back is a SHORTER, filtered version of the one the masker labelled.
    Asking at the reported list while holding the masker's labels would slide every label
    onto a different coordinate. The loss would fall, and it would mean nothing.

    Here the prediction Voxels is missing the middle candidate, standing in for one that
    injection removed. Requesting at the labelled list must return the surviving pairs
    still matched to their own labels.
    """
    cands = torch.tensor([[1, 1], [2, 2], [3, 3]])
    labels = torch.tensor([1.0, 0.0, 1.0])          # note: the dropped one is the 0
    # prediction carries only (1,1) and (3,3); the value encodes the coordinate
    vox = make_vox(torch.tensor([[3, 3], [1, 1]]), torch.tensor([[33.0], [11.0]]))

    pred, target, counts = gather_at_coords(vox, [cands], [labels])
    assert pred.numel() == 2, f"expected 2 matched candidates, got {pred.numel()}"
    assert int(counts[0]) == 2, f"count should reflect what matched, got {int(counts[0])}"
    got = {(round(float(p)), float(t)) for p, t in zip(pred, target)}
    assert got == {(11, 1.0), (33, 1.0)}, (
        f"labels slid off their coordinates: {sorted(got)}; (11, 1.0) and (33, 1.0) "
        f"are the pairs that belong together"
    )


def check_gather_empty_request() -> None:
    vox = make_vox(torch.tensor([[1, 1]], dtype=torch.int64), torch.tensor([[10.]]))
    pred, _, counts = gather_at_coords(vox, [torch.zeros(0, 2, dtype=torch.int64)], None)
    assert pred.numel() == 0 and counts.tolist() == [0]


# ── 3. the occupancy label ───────────────────────────────────────────────────

def check_occupancy_label_is_cell_membership() -> None:
    """A half-resolution cell counts as occupied if any pixel in it was active."""
    active = [torch.tensor([[4, 4], [5, 5]], dtype=torch.int64)]   # both fall in cell (2, 2)
    cand = [torch.tensor([[2, 2], [7, 7]], dtype=torch.int64)]
    got = occupancy_target(cand, active, stride=2, device=torch.device("cpu"))[0]
    assert got.tolist() == [1.0, 0.0], got.tolist()


def check_occupancy_label_empty_candidates() -> None:
    got = occupancy_target([torch.zeros(0, 2, dtype=torch.int64)],
                           [torch.tensor([[1, 1]], dtype=torch.int64)],
                           stride=2, device=torch.device("cpu"))[0]
    assert got.numel() == 0


# ── 4. what the objective builds ─────────────────────────────────────────────

def build(objective: str, use_proj_head: bool) -> DINODuneModel:
    torch.manual_seed(42)
    with contextlib.redirect_stdout(io.StringIO()):     # the backbones announce themselves
        return DINODuneModel(backbone_name="attn_mae", use_proj_head=use_proj_head,
                             objective=objective)


def check_mae_builds_heads_and_no_teacher() -> None:
    m = build("mae", use_proj_head=False)
    assert m.teacher is None, "a saved dead teacher would make --source=teacher return noise"
    assert m.student_head is None
    keys = set(m.student.state_dict())
    for name in ("charge_head", "occ_coarse_block", "occupancy_head_coarse"):
        assert any(k.startswith(name) for k in keys), f"{name} missing from the student"


def check_hybrid_builds_no_recon_heads() -> None:
    """Gating construction, not just the calls: an unused head still gets decayed."""
    m = build("hybrid", use_proj_head=True)
    assert m.teacher is not None
    keys = set(m.student.state_dict())
    for name in ("charge_head", "occ_coarse_block", "occupancy_head_coarse"):
        assert not any(k.startswith(name) for k in keys), f"{name} built for hybrid"


def check_train_mode_survives_a_missing_teacher() -> None:
    """The override used to call self.teacher.eval() unconditionally."""
    build("mae", use_proj_head=False).train()


CHECKS = (
    check_growth_doubles_then_haloes,
    check_growth_reach_scales_with_iters,
    check_growth_is_deduplicated,
    check_growth_never_leaves_the_canvas,
    check_growth_handles_empty_image,
    check_gather_matches_by_coordinate_not_row,
    check_gather_drops_target_with_missing_prediction,
    check_gather_keeps_labels_on_their_own_coords,
    check_gather_empty_request,
    check_occupancy_label_is_cell_membership,
    check_occupancy_label_empty_candidates,
    check_mae_builds_heads_and_no_teacher,
    check_hybrid_builds_no_recon_heads,
    check_train_mode_survives_a_missing_teacher,
)


def main() -> int:
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL  {check.__name__}")
            traceback.print_exc()
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
