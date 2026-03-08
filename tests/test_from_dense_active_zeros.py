"""
test_from_dense_active_zeros.py
────────────────────────────────
Demonstrates Voxels.from_dense with target_spatial_sparse_tensor:
  → keeps specified pixels active even when their value is exactly 0.

Two use-cases shown:
  1. Force a hand-picked set of coordinates to be active.
  2. Reuse the active-coordinate mask from one channel (truth / pid frame)
     when reading a second channel (reco frame) at the same locations —
     even where reco is zero.

IMPORTANT CONSTRAINT
────────────────────
from_dense(dense, target_spatial_sparse_tensor=template) internally shifts
all spatial coordinates by the global minimum of the template:

    shifted = template.coordinate_tensor - template.coordinate_tensor.min(dim=0).values

These shifted indices are then used to look up values in the dense tensor.
As a result, the feature lookup is only correct when:

    template.coordinate_tensor.min(dim=0).values == [0, 0]

i.e. at least one active voxel must sit at row=0 globally AND at least one
must sit at col=0 globally (across all batch items combined).

If this condition is not met, the column (or row) lookup is silently offset
and incorrect feature values are read.  Both test cases below are designed to
satisfy this constraint.
"""

import torch

try:
    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.geometry.coords.integer import IntCoords
    from warpconvnet.geometry.features.cat import CatFeatures
except ModuleNotFoundError as exc:
    raise SystemExit(
        "warpconvnet is not installed. Source setup.sh then rerun."
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a Voxels from explicit (batch, row, col) coordinate triples
# ─────────────────────────────────────────────────────────────────────────────

def voxels_from_coords(batch_row_col: list[tuple[int, int, int]], n_channels: int = 1) -> Voxels:
    """
    Construct a Voxels whose active set is exactly the given (b, r, c) triples.
    Features are all zero — this object is used purely as a coordinate template.
    """
    coords  = torch.tensor([[r, c] for b, r, c in batch_row_col], dtype=torch.int32)
    feats   = torch.zeros(len(batch_row_col), n_channels)

    # Build per-batch offsets from the batch indices
    batch_idx = torch.tensor([b for b, r, c in batch_row_col], dtype=torch.int64)
    B = int(batch_idx.max().item()) + 1
    counts  = torch.bincount(batch_idx, minlength=B)
    offsets = torch.cat([torch.tensor([0]), counts.cumsum(0)]).long()

    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Use-case 1: force specific pixels active, even when value is 0
# ─────────────────────────────────────────────────────────────────────────────

def test_force_zero_pixels_active():
    """
    Default from_dense drops exactly-zero pixels.
    With target_spatial_sparse_tensor the caller controls the active set,
    and zero-valued features are preserved.
    """
    # 1×1×4×4 dense image: only pixel (1,2) is non-zero
    dense = torch.zeros(1, 1, 4, 4)
    dense[0, 0, 1, 2] = 5.0

    # ── Default behaviour: only non-zero pixel is active ──────────────────
    vox_default = Voxels.from_dense(dense)
    default_coords = set(map(tuple, vox_default.coordinate_tensor.tolist()))
    assert default_coords == {(1, 2)}, f"Expected {{(1,2)}}, got {default_coords}"

    # ── Force pixels (0,0), (1,2), (3,3) active ───────────────────────────
    # (0,0) and (3,3) are zero in the dense tensor
    forced_pixels = [(0, 0, 0), (0, 1, 2), (0, 3, 3)]   # (batch, row, col)
    template = voxels_from_coords(forced_pixels)

    vox_forced = Voxels.from_dense(dense, target_spatial_sparse_tensor=template)

    forced_coords = set(map(tuple, vox_forced.coordinate_tensor.tolist()))
    assert forced_coords == {(0, 0), (1, 2), (3, 3)}, \
        f"Forced coords mismatch: {forced_coords}"

    # Feature at (1,2) should be 5.0; features at (0,0) and (3,3) should be 0.0
    feats = vox_forced.feature_tensor   # [3, 1]
    coord_to_feat = {
        tuple(vox_forced.coordinate_tensor[i].tolist()): feats[i, 0].item()
        for i in range(feats.shape[0])
    }
    assert coord_to_feat[(1, 2)] == 5.0, f"Expected 5.0, got {coord_to_feat[(1, 2)]}"
    assert coord_to_feat[(0, 0)] == 0.0, f"Expected 0.0, got {coord_to_feat[(0, 0)]}"
    assert coord_to_feat[(3, 3)] == 0.0, f"Expected 0.0, got {coord_to_feat[(3, 3)]}"

    print("test_force_zero_pixels_active  PASSED")
    print(f"  default active coords : {sorted(default_coords)}")
    print(f"  forced  active coords : {sorted(forced_coords)}")
    print(f"  features at forced coords: {coord_to_feat}")


# ─────────────────────────────────────────────────────────────────────────────
# Use-case 2: share active-set between two frames (truth mask → reco values)
# ─────────────────────────────────────────────────────────────────────────────

def test_share_active_mask_across_frames():
    """
    Mirrors the DUNE dataset pattern where a 'truth' frame (frame_pid_1st)
    defines which wire-plane channels fired, and we want to read the reco
    values (frame_rebinned_reco) at exactly those same locations — even where
    the reco amplitude happens to be 0.

    Batch size 2, spatial grid 3×5.

    Constraint satisfied: the truth hits include (row=0, col=0) so that
    template.coordinate_tensor.min(dim=0) == [0, 0] and no coordinate shift
    occurs inside from_dense.
    """
    # Truth frame: marks which pixels are 'interesting' (non-zero = true hit).
    # Hit at (0,0,0) ensures global min_coords = [0,0], satisfying the
    # target_spatial_sparse_tensor lookup constraint (see module docstring).
    truth = torch.zeros(2, 1, 3, 5)
    truth[0, 0, 0, 0] = 1.0   # batch 0: hit at (0,0)  ← anchors min_col=0
    truth[0, 0, 2, 3] = 1.0   #           hit at (2,3)
    truth[1, 0, 1, 4] = 1.0   # batch 1: hit at (1,4)

    # Reco frame: measured values at those locations.
    # Crucially, batch-0 pixel (2,3) has zero reco amplitude.
    reco = torch.zeros(2, 1, 3, 5)
    reco[0, 0, 0, 0] = 3.7    # batch 0: reco present at (0,0)
    reco[0, 0, 2, 3] = 0.0    # batch 0: reco = 0 at (2,3)  ← would be dropped by default
    reco[1, 0, 1, 4] = 2.1    # batch 1: reco present at (1,4)

    # Build the active-set template from the truth frame
    template = Voxels.from_dense(truth)   # active = true-hit locations

    # Read reco values at the truth locations (preserves the zero at (2,3))
    vox_reco = Voxels.from_dense(reco, target_spatial_sparse_tensor=template)

    # ── Verify ────────────────────────────────────────────────────────────
    # template and vox_reco must share the same coordinate structure
    assert torch.equal(template.coordinate_tensor, vox_reco.coordinate_tensor), \
        "Coordinate tensors should be identical"
    assert torch.equal(template.offsets, vox_reco.offsets), \
        "Offsets should be identical"

    # Check feature values: 3.7, 0.0, 2.1 in the order of the template coords
    feats = vox_reco.feature_tensor.squeeze(1).tolist()   # [3]

    # Batch offsets split: batch 0 → indices 0,1 ; batch 1 → index 2
    b0_feats = feats[vox_reco.offsets[0] : vox_reco.offsets[1]]
    b1_feats = feats[vox_reco.offsets[1] : vox_reco.offsets[2]]

    assert 3.7 in [round(f, 5) for f in b0_feats], f"Expected 3.7 in batch-0 feats: {b0_feats}"
    assert 0.0 in [round(f, 5) for f in b0_feats], f"Expected 0.0 in batch-0 feats: {b0_feats}"
    assert round(b1_feats[0], 5) == 2.1, f"Expected 2.1 for batch-1: {b1_feats}"

    print("test_share_active_mask_across_frames  PASSED")
    print(f"  template coords  : {template.coordinate_tensor.tolist()}")
    print(f"  template offsets : {template.offsets.tolist()}")
    print(f"  reco features    : {[round(f, 4) for f in feats]}")
    print(f"    batch 0 → {[round(f,4) for f in b0_feats]}   (3.7 at (0,0), 0.0 at (2,3))")
    print(f"    batch 1 → {[round(f,4) for f in b1_feats]}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_force_zero_pixels_active()
    print()
    test_share_active_mask_across_frames()
