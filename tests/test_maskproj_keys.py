"""
test_maskproj_keys.py
─────────────────────
Does `_project_masked_to_skip` drop a masked coordinate that is not actually in the skip?

The method decides "is this coordinate already in the skip?" by packing each (x, y) into
a single integer `y*W + x` and comparing integers. That is only a faithful encoding while
W exceeds every x it is applied to. W used to be sized from the skip's own largest x, but
it is also applied to the projected masked coordinates -- and a masked coordinate can sit
further right than anything left in the skip, because the masker removes exactly the
pixels the skip no longer has. Such a coordinate's key wraps into the next row's range,
where it can land on a real skip key, and the coordinate is discarded as a duplicate that
it is not. The mask token is then missing at that position.

Worked example, which is what the first check builds:

    skip's largest x = 100  ->  W = 102
    masked (x=105, y=3)     ->  3*102 + 105 = 411
    skip   (x=3,   y=4)     ->  4*102 + 3   = 411      <-- same key, coord wrongly dropped

Sizing W over both sets removes the wrap. This is the form the reference fork uses.

The second check guards the other direction: a coordinate genuinely already present in the
skip must still be dropped, or the fix would have bought correctness by disabling the
deduplication the method exists to do.

Pure torch on the coordinate tensors, so this runs on CPU -- no GPU, no sparse-conv ops.

Run:  python -u tests/test_maskproj_keys.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

from models.minkunet_attention import MinkUNetSparseAttentionMAE as MAE


def make_skip(coords: torch.Tensor) -> Voxels:
    """Single-image Voxels holding `coords`; features are irrelevant here."""
    offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)
    feats = torch.zeros(coords.shape[0], 4)
    return Voxels(
        batched_coordinates=IntCoords(coords.to(torch.int32), offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


def as_pairs(coords: torch.Tensor) -> set:
    return {(int(r[0]), int(r[1])) for r in coords}


def check_far_right_coord_survives() -> None:
    """A masked coord to the right of every skip coord must not be swallowed."""
    # (100, 0) sets the skip's largest x; (3, 4) is the collision partner.
    skip = make_skip(torch.tensor([[100, 0], [3, 4], [7, 9]], dtype=torch.int64))
    masked = [torch.tensor([[105, 3]], dtype=torch.int64)]

    out = MAE._project_masked_to_skip(masked, skip, stride=1)[0]

    assert (105, 3) in as_pairs(out), (
        "masked coord (105, 3) was dropped: its key collided with skip coord (3, 4) "
        "because W was sized from the skip's largest x (100) alone"
    )


def check_real_duplicate_still_dropped() -> None:
    """The deduplication the method exists for must still happen."""
    skip = make_skip(torch.tensor([[100, 0], [3, 4], [7, 9]], dtype=torch.int64))
    masked = [torch.tensor([[7, 9], [105, 3]], dtype=torch.int64)]

    out = MAE._project_masked_to_skip(masked, skip, stride=1)[0]
    pairs = as_pairs(out)

    assert (7, 9) not in pairs, "(7, 9) is already in the skip and must be dropped"
    assert (105, 3) in pairs, "(105, 3) is not in the skip and must survive"


def check_downsampling_still_applied() -> None:
    """stride > 1 floor-divides before the comparison, and W follows the projection."""
    skip = make_skip(torch.tensor([[50, 0], [1, 2]], dtype=torch.int64))
    # (211, 7) -> (105, 3) at stride 2, which is the same far-right case as above.
    masked = [torch.tensor([[211, 7]], dtype=torch.int64)]

    out = MAE._project_masked_to_skip(masked, skip, stride=2)[0]

    assert (105, 3) in as_pairs(out), f"expected projected (105, 3), got {as_pairs(out)}"


def check_empty_inputs() -> None:
    """Empty masked list and empty skip must not raise (W has no operands to size from)."""
    skip = make_skip(torch.tensor([[5, 5]], dtype=torch.int64))
    assert MAE._project_masked_to_skip([torch.zeros(0, 2, dtype=torch.int64)], skip,
                                       stride=1)[0].shape[0] == 0

    empty_skip = make_skip(torch.zeros(0, 2, dtype=torch.int64))
    out = MAE._project_masked_to_skip([torch.tensor([[3, 4]], dtype=torch.int64)],
                                      empty_skip, stride=1)
    # an empty skip has no batch items, so there is nothing to project onto
    assert isinstance(out, list)


CHECKS = (
    check_far_right_coord_survives,
    check_real_duplicate_still_dropped,
    check_downsampling_still_applied,
    check_empty_inputs,
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
