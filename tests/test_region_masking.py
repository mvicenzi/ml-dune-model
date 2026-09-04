"""
test_region_masking.py
──────────────────────
Does region masking remove whole cells, and are the occupancy candidates it hands back
labelled correctly?

Region masking is the only masker that knows the geometry of what it removed: it empties
whole cells of a fixed grid, so it can enumerate every pixel of those cells as an
occupancy candidate and say exactly which of them held charge. Everything worth checking
follows from that claim, and each part of it can fail quietly:

  - a cell is emptied COMPLETELY, or the label "empty" is wrong for the survivors;
  - the wipe ceiling holds, or an image can be masked out of existence and the student
    encodes nothing;
  - the candidate set is exactly the wiped cells at the candidate stride, no more (which
    would score untouched regions) and no less (which would skip the hole's interior);
  - a positive is a pixel that was removed, and every other candidate is a negative --
    checked against the pre-mask image, not against the masker's own bookkeeping;
  - the coarse footprint of a wiped cell never reaches a cell that still holds charge,
    which is what the cell-size divisibility rules exist to guarantee;
  - capping the negatives keeps every positive, because those are the answer.

The "randomize" flavour is checked for the opposite reason: it is EXPECTED to leak, since
voxels surviving inside a selected cell are active pixels the label calls empty. The test
pins that leak down, which is what justifies rejecting the flavour as a reconstruction
target upstream rather than quietly training on a wrong answer.

Pure torch on Voxels containers -- runs on CPU, no sparse-conv ops.

Run:  python -u tests/test_region_masking.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.types.voxels import Voxels

from dino.masking import SparseRegionMasker, cap_negatives, label_candidates

# A small canvas tiled exactly by the cells, mirroring how 70 x 100 tiles 1050 x 1500.
CANVAS_W, CANVAS_H = 60, 40
CELL_W, CELL_H = 10, 10
STRIDE = 2


def make_batch(counts=(120, 0, 80), seed: int = 0) -> Voxels:
    """A batch of scattered active pixels whose middle image is empty."""
    g = torch.Generator().manual_seed(seed)
    coords, feats = [], []
    for n in counts:
        c = torch.stack([torch.randint(0, CANVAS_W, (n,), generator=g),
                         torch.randint(0, CANVAS_H, (n,), generator=g)], dim=1)
        c = torch.unique(c, dim=0)
        coords.append(c)
        feats.append((c[:, 0] * 1000 + c[:, 1]).float().unsqueeze(1))
    offsets = torch.tensor([0] + list(torch.tensor([c.shape[0] for c in coords]).cumsum(0)),
                           dtype=torch.int64)
    return Voxels(
        batched_coordinates=IntCoords(torch.cat(coords).to(torch.int32), offsets=offsets),
        batched_features=CatFeatures(torch.cat(feats), offsets=offsets),
        offsets=offsets,
    )


def pairs(coords: torch.Tensor) -> set:
    return {(int(x), int(y)) for x, y in coords}


def cell_of(x: int, y: int) -> tuple:
    return (x // CELL_W, y // CELL_H)


def per_image(vox: Voxels, b: int) -> torch.Tensor:
    return vox.coordinate_tensor[int(vox.offsets[b]):int(vox.offsets[b + 1])]


def masker(**kw) -> SparseRegionMasker:
    torch.manual_seed(0)
    return SparseRegionMasker(image_w=CANVAS_W, image_h=CANVAS_H,
                              cell_w=CELL_W, cell_h=CELL_H, **kw)


# ---------------------------------------------------------------- wipe geometry


def check_wipe_empties_whole_cells() -> None:
    """No cell may be left half-masked: that is what makes the occupancy label exact."""
    vox = make_batch()
    student, masked_coords = masker(flavor="wipe", wipe_max=0.75)(vox)[:2]
    for b in range(len(vox.offsets) - 1):
        kept_cells = {cell_of(*c) for c in pairs(per_image(student, b))}
        gone_cells = {cell_of(*c) for c in pairs(masked_coords[b])}
        both = kept_cells & gone_cells
        assert not both, f"image {b}: cells {sorted(both)} are partly masked, partly kept"


def check_wipe_ceiling_holds() -> None:
    """The student must always be left with something to encode."""
    vox = make_batch()
    for wipe_max in (0.25, 0.5, 0.75):
        student, masked_coords = masker(flavor="wipe", wipe_max=wipe_max)(vox)[:2]
        for b in range(len(vox.offsets) - 1):
            n = per_image(vox, b).shape[0]
            if n == 0:
                continue
            n_masked = masked_coords[b].shape[0]
            assert n_masked <= int(wipe_max * n), (
                f"wipe_max={wipe_max} image {b}: masked {n_masked} of {n}"
            )
            assert per_image(student, b).shape[0] > 0, (
                f"wipe_max={wipe_max} image {b}: student left with no voxels"
            )


def check_wipe_masks_something() -> None:
    """A ceiling that removed nothing would make the objective vacuous.

    An image that comes back with nothing masked contributes no charge target either,
    so this is checked over several seeds and occupancies rather than once: the release
    of surplus that keeps the ceiling honest is the step that could, in principle, hand
    back everything it took.
    """
    for seed in range(5):
        for counts in ((120, 0, 80), (400, 0, 30), (12, 0, 9)):
            vox = make_batch(counts=counts, seed=seed)
            torch.manual_seed(seed)
            m = SparseRegionMasker(image_w=CANVAS_W, image_h=CANVAS_H,
                                   cell_w=CELL_W, cell_h=CELL_H,
                                   flavor="wipe", wipe_max=0.75)
            _, masked_coords = m(vox)[:2]
            for b in range(len(vox.offsets) - 1):
                if per_image(vox, b).shape[0] == 0:
                    continue
                assert masked_coords[b].shape[0] > 0, (
                    f"seed {seed} counts {counts} image {b}: a non-empty image was left "
                    f"entirely unmasked, so it has no charge target"
                )


# ------------------------------------------------------------ candidate set


def check_surplus_release_does_not_leak() -> None:
    """The ceiling can hand charge back into a cell that was taken.

    A cell dense enough to bust the ceiling on its own is still taken (an image must be
    masked at all), and the surplus is then released at random -- which puts visible
    charge back inside a taken cell. Densifying that cell would label pixels the student
    can still see as empty. Only fully emptied cells may be enumerated.
    """
    # all the charge in one 10x10 cell, plus a few pixels elsewhere, and a ceiling below
    # that cell's size: the release must fire
    dense = torch.stack(torch.meshgrid(torch.arange(0, 10), torch.arange(0, 10),
                                       indexing="ij"), -1).reshape(-1, 2)
    coords = torch.cat([dense, torch.tensor([[25, 25], [26, 26], [27, 27]])])
    offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)
    vox = Voxels(
        batched_coordinates=IntCoords(coords.to(torch.int32), offsets=offsets),
        batched_features=CatFeatures(torch.ones(coords.shape[0], 1), offsets=offsets),
        offsets=offsets,
    )
    fired = False
    for seed in range(8):
        torch.manual_seed(seed)
        m = SparseRegionMasker(image_w=CANVAS_W, image_h=CANVAS_H,
                               cell_w=CELL_W, cell_h=CELL_H, flavor="wipe", wipe_max=0.5,
                               return_masked_feats=True, build_candidates=True,
                               cand_stride=STRIDE)
        student, masked_coords, _, cand, occ = m(vox)
        n_masked = masked_coords[0].shape[0]
        if 0 < n_masked < dense.shape[0]:
            fired = True                       # the release actually happened
        survivors = {(int(x) // STRIDE, int(y) // STRIDE) for x, y in student.coordinate_tensor}
        for row, c in enumerate(pairs_in_order(cand[0])):
            assert c not in survivors, (
                f"seed {seed}: candidate {c} labelled {float(occ[0][row])} but charge is "
                f"still visible there — the surplus release put it back"
            )
    assert fired, "the surplus-release branch never ran; this test proved nothing"


def check_candidates_are_exactly_the_wiped_cells() -> None:
    """Every pixel of a wiped cell, at the candidate stride, and nothing outside one."""
    vox = make_batch()
    m = masker(flavor="wipe", wipe_max=0.75, return_masked_feats=True,
               build_candidates=True, cand_stride=STRIDE)
    _, masked_coords, _, cand, _ = m(vox)
    for b in range(len(vox.offsets) - 1):
        wiped_cells = {cell_of(*c) for c in pairs(masked_coords[b])}
        # every candidate lies in a wiped cell (coarse coords scale back up by STRIDE)
        for cx, cy in pairs(cand[b]):
            cell = cell_of(cx * STRIDE, cy * STRIDE)
            assert cell in wiped_cells, f"image {b}: candidate {(cx, cy)} outside a wiped cell"
        # and every coarse position of every wiped cell is present
        expected = {(x // STRIDE, y // STRIDE)
                    for gx, gy in wiped_cells
                    for x in range(gx * CELL_W, (gx + 1) * CELL_W)
                    for y in range(gy * CELL_H, (gy + 1) * CELL_H)}
        assert pairs(cand[b]) == expected, (
            f"image {b}: candidate set differs from the wiped cells by "
            f"{len(pairs(cand[b]) ^ expected)} positions"
        )


def check_candidates_are_deduplicated() -> None:
    """Several full-resolution pixels share a coarse cell; each must appear once."""
    vox = make_batch()
    m = masker(flavor="wipe", wipe_max=0.75, return_masked_feats=True,
               build_candidates=True, cand_stride=STRIDE)
    cand = m(vox)[3]
    for b, c in enumerate(cand):
        assert c.shape[0] == len(pairs(c)), f"image {b}: {c.shape[0] - len(pairs(c))} duplicates"


# ------------------------------------------------------------ occupancy labels


def check_labels_against_the_pre_mask_image() -> None:
    """A positive is a coarse cell that held charge before masking -- checked independently."""
    vox = make_batch()
    m = masker(flavor="wipe", wipe_max=0.75, return_masked_feats=True,
               build_candidates=True, cand_stride=STRIDE)
    _, masked_coords, _, cand, occ = m(vox)
    for b in range(len(vox.offsets) - 1):
        # recompute the truth from the ORIGINAL image, not from the masker's bookkeeping
        active_coarse = {(int(x) // STRIDE, int(y) // STRIDE) for x, y in per_image(vox, b)}
        for row, (cx, cy) in enumerate(pairs_in_order(cand[b])):
            want = 1.0 if (cx, cy) in active_coarse else 0.0
            got = float(occ[b][row])
            assert got == want, (
                f"image {b} candidate {(cx, cy)}: label {got}, but the pre-mask image "
                f"says {want}"
            )


def check_no_survivor_is_labelled_empty() -> None:
    """The leakage check: under wipe, no candidate may cover a pixel still visible."""
    vox = make_batch()
    m = masker(flavor="wipe", wipe_max=0.75, return_masked_feats=True,
               build_candidates=True, cand_stride=STRIDE)
    student, _, _, cand, occ = m(vox)
    for b in range(len(vox.offsets) - 1):
        survivor_coarse = {(int(x) // STRIDE, int(y) // STRIDE) for x, y in per_image(student, b)}
        for row, c in enumerate(pairs_in_order(cand[b])):
            if c in survivor_coarse:
                raise AssertionError(
                    f"image {b}: candidate {c} labelled {float(occ[b][row])} but the "
                    f"student can still see charge there"
                )


def check_both_classes_are_present() -> None:
    """Empties are the whole point; a candidate set of all positives teaches nothing."""
    vox = make_batch()
    m = masker(flavor="wipe", wipe_max=0.75, return_masked_feats=True,
               build_candidates=True, cand_stride=STRIDE)
    occ = m(vox)[4]
    for b in (0, 2):
        n_pos = int(occ[b].sum())
        n_neg = int(occ[b].numel() - n_pos)
        assert n_pos > 0, f"image {b}: no positives"
        assert n_neg > 0, f"image {b}: no negatives"


def check_randomize_leaks_as_documented() -> None:
    """randomize must be shown to leak -- that is why mae rejects it."""
    vox = make_batch()
    m = masker(flavor="randomize", r1=0.9, r2=0.5, return_masked_feats=True,
               build_candidates=True, cand_stride=STRIDE)
    student, _, _, cand, occ = m(vox)
    leaks = 0
    for b in range(len(vox.offsets) - 1):
        survivor_coarse = {(int(x) // STRIDE, int(y) // STRIDE) for x, y in per_image(student, b)}
        for row, c in enumerate(pairs_in_order(cand[b])):
            if c in survivor_coarse and float(occ[b][row]) == 0.0:
                leaks += 1
    assert leaks > 0, (
        "randomize was expected to label visible charge as empty; if it no longer does, "
        "the rejection in validate_config is over-strict and should be revisited"
    )


# ------------------------------------------------------------ negative capping


def check_cap_keeps_every_positive() -> None:
    """Positives are the answer; only empties may be given up to save memory."""
    cand_b = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    occ = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    keep = cap_negatives(cand_b, occ, B=2, max_neg=1)
    assert bool(keep[occ > 0.5].all()), "a positive was dropped"
    for b in (0, 1):
        n_neg = int(((cand_b == b) & keep & (occ < 0.5)).sum())
        assert n_neg <= 1, f"image {b}: {n_neg} negatives kept, cap was 1"


def check_cap_by_ratio() -> None:
    """neg_per_pos scales the budget with each image's own positive count."""
    cand_b = torch.zeros(21, dtype=torch.int64)
    occ = torch.cat([torch.ones(3), torch.zeros(18)])
    keep = cap_negatives(cand_b, occ, B=1, neg_per_pos=2.0)
    assert int((keep & (occ < 0.5)).sum()) == 6, "expected 2 negatives per positive"


def check_cap_is_off_by_default() -> None:
    """No cap set means nothing is dropped."""
    cand_b = torch.zeros(10, dtype=torch.int64)
    occ = torch.cat([torch.ones(2), torch.zeros(8)])
    assert bool(cap_negatives(cand_b, occ, B=1).all())


def check_capped_run_still_labels_correctly() -> None:
    """Capping must sub-sample, never relabel."""
    vox = make_batch()
    m = masker(flavor="wipe", wipe_max=0.75, return_masked_feats=True,
               build_candidates=True, cand_stride=STRIDE, max_neg=5)
    _, _, _, cand, occ = m(vox)
    for b in range(len(vox.offsets) - 1):
        active_coarse = {(int(x) // STRIDE, int(y) // STRIDE) for x, y in per_image(vox, b)}
        n_neg = 0
        for row, c in enumerate(pairs_in_order(cand[b])):
            want = 1.0 if c in active_coarse else 0.0
            assert float(occ[b][row]) == want, f"image {b}: {c} relabelled by capping"
            n_neg += want == 0.0
        assert n_neg <= 5, f"image {b}: {n_neg} negatives survived a cap of 5"


# ------------------------------------------------------------ construction rules


def check_indivisible_canvas_is_rejected() -> None:
    """A partial edge cell densifies to a different count and skews the positive rate."""
    try:
        SparseRegionMasker(image_w=1050, image_h=1500, cell_w=64, cell_h=100)
    except AssertionError as exc:
        assert "evenly" in str(exc), f"message does not name the problem: {exc}"
        return
    raise AssertionError("a cell size that does not tile the canvas was accepted")


def check_cell_must_divide_the_candidate_stride() -> None:
    """An odd cell puts a wiped cell's coarse footprint out of step with the grid."""
    try:
        SparseRegionMasker(image_w=105, image_h=100, cell_w=7, cell_h=10, cand_stride=2)
    except AssertionError as exc:
        assert "stride" in str(exc), f"message does not name the problem: {exc}"
        return
    raise AssertionError("an odd cell size was accepted at stride 2")


def check_production_grid_is_exact() -> None:
    """The shipped defaults must tile the real canvas: 70 x 100 over 1050 x 1500."""
    m = SparseRegionMasker(image_w=1050, image_h=1500, cell_w=70, cell_h=100)
    assert (m.n_cols, m.n_rows) == (15, 15), f"grid is {m.n_cols} x {m.n_rows}, expected 15 x 15"


def pairs_in_order(coords: torch.Tensor) -> list:
    return [(int(x), int(y)) for x, y in coords]


CHECKS = (
    check_wipe_empties_whole_cells,
    check_wipe_ceiling_holds,
    check_wipe_masks_something,
    check_surplus_release_does_not_leak,
    check_candidates_are_exactly_the_wiped_cells,
    check_candidates_are_deduplicated,
    check_labels_against_the_pre_mask_image,
    check_no_survivor_is_labelled_empty,
    check_both_classes_are_present,
    check_randomize_leaks_as_documented,
    check_cap_keeps_every_positive,
    check_cap_by_ratio,
    check_cap_is_off_by_default,
    check_capped_run_still_labels_correctly,
    check_indivisible_canvas_is_rejected,
    check_cell_must_divide_the_candidate_stride,
    check_production_grid_is_exact,
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
    raise SystemExit(main())
