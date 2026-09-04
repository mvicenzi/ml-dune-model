"""
test_masker_contract.py
───────────────────────
Do both maskers hand back what they removed, and does the split stay lossless?

The mae objective regresses the charge that masking removed, but until now both maskers
kept only `feats[keep]` and discarded `feats[masked]` -- the target did not exist for any
mask type. They now collect it under `return_masked_feats`, and always return a three-item
tuple with `None` in that slot when the flag is off, so no caller has to branch on mode.

What is checked:
  - kept and masked partition the input exactly: no pixel lost, none duplicated;
  - the returned features are the ones belonging to the returned coordinates, row for row
    (a target misaligned by one row would train perfectly happily and mean nothing);
  - the flag is genuinely off by default, so runs that do not need this do not pay for it;
  - an empty image still produces aligned, empty entries in every list, since offsets must
    stay length B+1.

Both maskers get the same checks: they are drop-in replacements for each other, so a
contract that holds for one and not the other is a trap.

Pure torch on Voxels containers -- runs on CPU.

Run:  python -u tests/test_masker_contract.py
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

from dino.masking import SparseVoxelMasker, SparseBlockMasker, SparseRegionMasker


CANVAS_W, CANVAS_H = 60, 60      # tiled exactly by the 10x10 cells used below


def make_batch(counts=(40, 0, 25), seed: int = 0) -> Voxels:
    """A batch whose middle image is empty -- the alignment case that bites."""
    g = torch.Generator().manual_seed(seed)
    coords, feats = [], []
    for n in counts:
        c = torch.stack([torch.randint(0, 60, (n,), generator=g),
                         torch.randint(0, 60, (n,), generator=g)], dim=1)
        c = torch.unique(c, dim=0)
        coords.append(c)
        # feature value encodes the coordinate, so a misalignment is detectable
        feats.append((c[:, 0] * 1000 + c[:, 1]).float().unsqueeze(1))
    offsets = torch.tensor([0] + list(torch.tensor([c.shape[0] for c in coords]).cumsum(0)),
                           dtype=torch.int64)
    all_c = torch.cat(coords).to(torch.int32)
    all_f = torch.cat(feats)
    return Voxels(
        batched_coordinates=IntCoords(all_c, offsets=offsets),
        batched_features=CatFeatures(all_f, offsets=offsets),
        offsets=offsets,
    )


def feature_for(coord: torch.Tensor) -> float:
    return float(coord[0]) * 1000 + float(coord[1])


def pairs(coords: torch.Tensor) -> set:
    return {(int(r[0]), int(r[1])) for r in coords}


def maskers(**kw):
    """Every masker, under the same checks: they are drop-in replacements for each other."""
    torch.manual_seed(0)
    yield "pixel", SparseVoxelMasker(mask_ratio=0.5, **kw)
    torch.manual_seed(0)
    yield "block", SparseBlockMasker(mask_ratio=0.5, win_ch=3, win_tick=3, **kw)
    torch.manual_seed(0)
    yield "region", SparseRegionMasker(image_w=CANVAS_W, image_h=CANVAS_H,
                                       cell_w=10, cell_h=10, **kw)


def check_arity_is_constant() -> None:
    """Five elements from every masker; only which of them are None depends on the flags."""
    vox = make_batch()
    for name, m in maskers():
        out = m(vox)
        assert len(out) == 5, f"{name}: expected 5 elements, got {len(out)}"
        assert out[2] is None, f"{name}: masked feats should be None when the flag is off"
        assert out[3] is None and out[4] is None, f"{name}: candidates not requested"
    for name, m in maskers(return_masked_feats=True):
        out = m(vox)
        assert len(out) == 5, f"{name}: expected 5 elements, got {len(out)}"
        assert out[2] is not None, f"{name}: masked feats requested but None"


def check_partition_is_lossless() -> None:
    """kept and masked must together be exactly the input, per image."""
    vox = make_batch()
    for name, m in maskers(return_masked_feats=True):
        student, masked_coords = m(vox)[:2]
        for b in range(len(vox.offsets) - 1):
            orig = pairs(vox.coordinate_tensor[vox.offsets[b]:vox.offsets[b + 1]])
            kept = pairs(student.coordinate_tensor[student.offsets[b]:student.offsets[b + 1]])
            gone = pairs(masked_coords[b])
            assert kept | gone == orig, f"{name} image {b}: kept+masked != original"
            assert not (kept & gone), f"{name} image {b}: {len(kept & gone)} pixels in both"


def check_features_match_their_coords() -> None:
    """Row i of masked_feats must belong to row i of masked_coords."""
    vox = make_batch()
    for name, m in maskers(return_masked_feats=True):
        _, masked_coords, masked_feats = m(vox)[:3]
        assert len(masked_feats) == len(masked_coords), f"{name}: list lengths differ"
        for b, (c, f) in enumerate(zip(masked_coords, masked_feats)):
            assert c.shape[0] == f.shape[0], f"{name} image {b}: row counts differ"
            for row in range(c.shape[0]):
                assert abs(float(f[row, 0]) - feature_for(c[row])) < 1e-6, (
                    f"{name} image {b} row {row}: feature {float(f[row, 0])} does not "
                    f"belong to coord {c[row].tolist()}"
                )


def check_empty_image_stays_aligned() -> None:
    """The empty middle image must still contribute an entry to every list."""
    vox = make_batch(counts=(30, 0, 20))
    B = len(vox.offsets) - 1
    for name, m in maskers(return_masked_feats=True):
        student, masked_coords, masked_feats = m(vox)[:3]
        assert len(student.offsets) - 1 == B, f"{name}: offsets lost an image"
        assert len(masked_coords) == B, f"{name}: masked_coords has {len(masked_coords)}"
        assert len(masked_feats) == B, f"{name}: masked_feats has {len(masked_feats)}"
        assert masked_coords[1].shape[0] == 0 and masked_feats[1].shape[0] == 0


def check_flag_defaults_off() -> None:
    """A run that does not need the targets should not be collecting them."""
    for _, m in maskers():
        assert m.return_masked_feats is False


CHECKS = (
    check_arity_is_constant,
    check_partition_is_lossless,
    check_features_match_their_coords,
    check_empty_image_stays_aligned,
    check_flag_defaults_off,
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
