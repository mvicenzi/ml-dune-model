"""
test_generative_2d_approaches.py
────────────────────────────────
How do we grow occupancy candidate coordinates in 2D, and are the approaches
equivalent?

Occupancy asks "of these cells, which were actually active?". Answering it needs a
*candidate set*: coordinates injected into the sparse tensor that the decoder then
scores. Injection itself is already solved -- `_augment_skip_with_masked` concatenates
arbitrary coordinates with a mask-token feature and recomputes offsets, which is exactly
what a generative convolution does to the coordinate set. The open question is not how to
inject but **which coordinates to inject**, and whether the library can pick them for us.

Five approaches, and the reason each is here:

  A  native 2D generative conv   -- SparseConv2d(generative=True, transposed=True).
                                    Expected to fail; this pins down *where*.
  B  z-padded 3D generative conv -- pad a dummy z=0 axis and use SparseConv3d with
                                    stride/kernel 1 in z. If this runs it is the only
                                    way to get the library's own answer in 2D.
  C  torch coordinate growth     -- coords x2 then a 3x3 dilation, done by hand. This is
                                    the mechanism we plan to port, on the claim that it
                                    emits "the SAME coordinate set a generative
                                    transposed stride-2 k3 conv would emit".
  D  densify the wiped region    -- enumerate every cell in the wiped rectangle. The
                                    plainest "just add the pixels back" reading.
  E  inject at masked coords     -- inject only where active pixels were removed.

**B vs C is the headline.** C is what we would port, and its correctness rests entirely
on that "SAME coordinate set" claim, which no test in either repo checks. B is the only
available oracle.

**E is the trap.** It is the most natural thing to reach for -- we already inject at
masked coordinates for the charge head -- but every injected cell was active by
construction, so the occupancy target is all-ones and the term carries no signal. E
asserts that, so the reasoning is documented rather than rediscovered. The charge head is
unaffected: being told where to predict a value and predicting the value is a real task.

A/B/C also differ in whether they dedup. C with `up_stages=1, iters=1` returns its raw
9x-expanded set and relies on `_project_masked_to_skip` to drop duplicates downstream,
while B dedups internally through the hash table -- so the comparison is on unique sets,
and the dedup conventions are checked against each other separately (they use different
flat-key strides, which is a silent-divergence risk when C lands next to our helpers).

Cases C, D and E are pure torch and run anywhere. A and B need a GPU: warpconvnet's
`expand_coords` raises on non-CUDA tensors regardless of dimensionality.

This is an investigation, not a gate. It reports what each approach produces -- set sizes
and positive fractions, the numbers that decide which one to port -- and hard-asserts only
what must hold: C's determinism, B == C if B runs at all, E's all-ones target, and the
dedup agreement.

Run:  python -u tests/test_generative_2d_approaches.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Canvas and region-cell geometry from dino/config.py and the docs/20 region-masker
# defaults, so the set sizes reported here are the ones we would actually see.
IMAGE_W, IMAGE_H = 1050, 1500
CELL_W, CELL_H = 64, 100
BOTTLENECK_STRIDE = 4       # num_stages=2 -> bottleneck at /4
HALF_STRIDE = 2             # grown candidates land at /2


# ─────────────────────────────────────────────────────────────────────────────
# Toy data
# ─────────────────────────────────────────────────────────────────────────────

def make_toy_actives(n_tracks: int = 3, seed: int = 0) -> torch.Tensor:
    """A handful of diagonal tracks on the full canvas -> [N, 2] int64 (x, y) coords.

    Diagonal rather than random: growth and densification differ most where structure is
    locally sparse but globally connected, which is what a track is and what noise is not.
    """
    g = torch.Generator().manual_seed(seed)
    pts = []
    for _ in range(n_tracks):
        x0 = torch.randint(0, IMAGE_W // 2, (1,), generator=g).item()
        y0 = torch.randint(0, IMAGE_H // 2, (1,), generator=g).item()
        n = torch.randint(200, 600, (1,), generator=g).item()
        t = torch.arange(n)
        x = (x0 + t).clamp(max=IMAGE_W - 1)
        y = (y0 + 2 * t).clamp(max=IMAGE_H - 1)
        pts.append(torch.stack([x, y], dim=1))
    return torch.unique(torch.cat(pts, dim=0), dim=0)


def wipe_one_cell(actives: torch.Tensor, cell_ix: int, cell_iy: int):
    """Whole-cell wipe. Returns (kept, masked) coords -- masked = actives inside the cell.

    Whole-cell (not per-pixel) is what makes an occupancy target leakage-free: inside the
    cell nothing survives to hint at what was there.
    """
    x0, x1 = cell_ix * CELL_W, (cell_ix + 1) * CELL_W
    y0, y1 = cell_iy * CELL_H, (cell_iy + 1) * CELL_H
    inside = ((actives[:, 0] >= x0) & (actives[:, 0] < x1)
              & (actives[:, 1] >= y0) & (actives[:, 1] < y1))
    return actives[~inside], actives[inside], (x0, x1, y0, y1)


def downsample_unique(coords: torch.Tensor, stride: int) -> torch.Tensor:
    """Floor-divide to a coarser grid and dedup -- warpconvnet's striding convention."""
    return torch.unique(torch.div(coords, stride, rounding_mode="floor"), dim=0)


def unique_rows(coords: torch.Tensor) -> torch.Tensor:
    return torch.unique(coords, dim=0)


def as_key_set(coords: torch.Tensor) -> set:
    """Hashable set of (x, y) pairs, for order-independent set comparison."""
    return {(int(r[0]), int(r[1])) for r in coords}


# ─────────────────────────────────────────────────────────────────────────────
# Occupancy target -- shared by every approach
# ─────────────────────────────────────────────────────────────────────────────

def occ_target(candidates: torch.Tensor, pre_mask_actives_low: torch.Tensor) -> torch.Tensor:
    """Membership of each candidate in the pre-mask active set, at the candidate's scale.

    This is the whole occupancy label: "was this cell active before masking?". The target
    is built on the emitted set, so pred and target shrink together under any cap.
    """
    if candidates.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    H = max(int(pre_mask_actives_low[:, 1].max()) if pre_mask_actives_low.numel() else 0,
            int(candidates[:, 1].max())) + 1
    cand_keys = candidates[:, 0].long() * H + candidates[:, 1].long()
    act_keys = pre_mask_actives_low[:, 0].long() * H + pre_mask_actives_low[:, 1].long()
    return torch.isin(cand_keys, act_keys)


# ─────────────────────────────────────────────────────────────────────────────
# Approach C -- torch coordinate growth (the mechanism we would port)
# ─────────────────────────────────────────────────────────────────────────────

def grow_coords_torch(bottleneck_coords: torch.Tensor, iters: int = 1,
                      up_stages: int = 1, dedup: bool = False) -> torch.Tensor:
    """Deterministic x2 upsample + 3x3 dilation, repeated -- single-image port.

    Mirrors the reference `_grow_coords_from_bottleneck` for one image: scale coords by
    the stride, kernel-expand by 3x3, optionally repeat. With iters=1 and up_stages=1 the
    raw result keeps its 9x duplicates (the reference leaves dedup to
    `_project_masked_to_skip`); `dedup=True` makes the unique set explicit for comparison.
    """
    off = torch.tensor([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
                       dtype=bottleneck_coords.dtype, device=bottleneck_coords.device)
    grown = ((bottleneck_coords * 2)[:, None, :] + off[None, :, :]).reshape(-1, 2).clamp_(min=0)
    for _ in range(int(up_stages) - 1):
        grown = ((unique_rows(grown) * 2)[:, None, :] + off[None, :, :]).reshape(-1, 2).clamp_(min=0)
    for _ in range(int(iters) - 1):
        grown = (unique_rows(grown)[:, None, :] + off[None, :, :]).reshape(-1, 2).clamp_(min=0)
    return unique_rows(grown) if dedup else grown


# ─────────────────────────────────────────────────────────────────────────────
# Approach D -- densify the wiped region
# ─────────────────────────────────────────────────────────────────────────────

def densify_cell(bbox, stride: int) -> torch.Tensor:
    """Every cell of the wiped rectangle at the given scale -- "just add the pixels back"."""
    x0, x1, y0, y1 = bbox
    xs = torch.arange(x0 // stride, (x1 + stride - 1) // stride)
    ys = torch.arange(y0 // stride, (y1 + stride - 1) // stride)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def subsample_negatives(candidates: torch.Tensor, target: torch.Tensor,
                        neg_per_pos: float = 8.0, seed: int = 0) -> torch.Tensor:
    """Keep every positive and K uniformly-drawn negatives per positive.

    The ratio form, not an absolute cap: it fixes class balance instead of truncating
    coverage. This is the "partial inclusion" lever -- how many negatives to pay for.
    """
    g = torch.Generator().manual_seed(seed)
    pos, neg = candidates[target], candidates[~target]
    keep = min(neg.shape[0], int(pos.shape[0] * neg_per_pos))
    if keep < neg.shape[0]:
        neg = neg[torch.randperm(neg.shape[0], generator=g)[:keep]]
    return torch.cat([pos, neg], dim=0)


def hard_negatives(candidates: torch.Tensor, target: torch.Tensor,
                   kept_low: torch.Tensor, neg_per_pos: float = 8.0,
                   frac: float = 0.5, window: int = 1, seed: int = 0) -> torch.Tensor:
    """Fill the SAME negative budget with `frac` boundary-biased negatives, rest uniform.

    The other half of "partial": *which* negatives, not how many. In a ~99.8%-empty image
    most empties are far from any track and are classified correctly from epoch one; the
    informative ones sit on the boundary of surviving structure.

    Budget-preserving on purpose -- this is a signal-quality lever, not a memory one, so
    it must come out the same size as plain subsampling at the same `neg_per_pos`. A
    version that filtered down to boundary negatives only would look like a memory win
    that the mechanism does not actually deliver.
    """
    g = torch.Generator().manual_seed(seed)
    pos, neg = candidates[target], candidates[~target]
    budget = min(neg.shape[0], int(pos.shape[0] * neg_per_pos))
    if neg.shape[0] == 0 or kept_low.shape[0] == 0 or budget == 0:
        return torch.cat([pos, neg[:budget]], dim=0)

    off = torch.tensor([[dx, dy] for dx in range(-window, window + 1)
                        for dy in range(-window, window + 1)], dtype=kept_low.dtype)
    ring = unique_rows((kept_low[:, None, :] + off[None, :, :]).reshape(-1, 2).clamp_(min=0))
    H = int(max(int(ring[:, 1].max()), int(neg[:, 1].max()))) + 1
    is_hard = torch.isin(neg[:, 0].long() * H + neg[:, 1].long(),
                         ring[:, 0].long() * H + ring[:, 1].long())

    hard, easy = neg[is_hard], neg[~is_hard]
    n_hard = min(hard.shape[0], int(budget * frac))
    n_easy = min(easy.shape[0], budget - n_hard)
    # Boundary negatives are scarce; spend any shortfall on uniform ones to hold budget.
    if n_hard + n_easy < budget:
        n_hard = min(hard.shape[0], budget - n_easy)
    pick_h = hard[torch.randperm(hard.shape[0], generator=g)[:n_hard]]
    pick_e = easy[torch.randperm(easy.shape[0], generator=g)[:n_easy]]
    return torch.cat([pos, pick_h, pick_e], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# warpconvnet plumbing (A and B only)
# ─────────────────────────────────────────────────────────────────────────────

def make_voxels(coords: torch.Tensor, channels: int, device, tensor_stride):
    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.geometry.coords.integer import IntCoords
    from warpconvnet.geometry.features.cat import CatFeatures

    coords = coords.to(device=device, dtype=torch.int32)
    offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)
    feats = torch.randn(coords.shape[0], channels, device=device)
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets, tensor_stride=tensor_stride),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


def final_frame(exc: BaseException) -> str:
    """"file.py:line in func" for the deepest frame -- names *which* assert fired."""
    tb = traceback.extract_tb(exc.__traceback__)
    if not tb:
        return "<no traceback>"
    f = tb[-1]
    return f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}()"


# ─────────────────────────────────────────────────────────────────────────────
# Cases
# ─────────────────────────────────────────────────────────────────────────────

def case_a_native_2d(bott_coords, device) -> dict:
    """A: does SparseConv2d(generative=True, transposed=True) work on 2D coords?"""
    from warpconvnet.nn.modules.sparse_conv import SparseConv2d

    vox = make_voxels(bott_coords, channels=8, device=device,
                      tensor_stride=(BOTTLENECK_STRIDE, BOTTLENECK_STRIDE))
    conv = SparseConv2d(8, 8, kernel_size=3, stride=2, transposed=True,
                        generative=True, bias=False).to(device)
    try:
        out = conv(vox)
        coords = out.coordinate_tensor[:, :2].long().cpu()
        return {"ran": True, "n_unique": len(as_key_set(coords)), "coords": coords,
                "note": "2D generative ran -- the limitation is gone"}
    except Exception as exc:                                  # noqa: BLE001 -- reporting
        return {"ran": False, "exc": type(exc).__name__, "where": final_frame(exc),
                "msg": (str(exc) or "<empty assert>")[:120]}


def case_b_padded_3d(bott_coords, device) -> dict:
    """B: pad a dummy z=0 axis and let SparseConv3d's generative path pick the coords.

    stride and kernel are 1 in z so the dummy axis cannot grow: any z != 0 in the output
    means the padding leaked into the expansion and the approach is not sound.
    """
    from warpconvnet.nn.modules.sparse_conv import SparseConv3d

    c3 = torch.cat([bott_coords, torch.zeros(bott_coords.shape[0], 1,
                                             dtype=bott_coords.dtype)], dim=1)
    vox = make_voxels(c3, channels=8, device=device,
                      tensor_stride=(BOTTLENECK_STRIDE, BOTTLENECK_STRIDE, 1))
    conv = SparseConv3d(8, 8, kernel_size=(3, 3, 1), stride=(2, 2, 1), transposed=True,
                        generative=True, bias=False).to(device)
    try:
        out = conv(vox)
        oc = out.coordinate_tensor.long().cpu()
        z_vals = sorted({int(v) for v in oc[:, 2]})
        coords = oc[:, :2]
        return {"ran": True, "z_vals": z_vals, "z_clean": z_vals == [0],
                "n_unique": len(as_key_set(coords)), "coords": coords}
    except Exception as exc:                                  # noqa: BLE001 -- reporting
        return {"ran": False, "exc": type(exc).__name__, "where": final_frame(exc),
                "msg": (str(exc) or "<empty assert>")[:120]}


def case_c_torch_growth(bott_coords) -> dict:
    """C: the torch growth we would port. Determinism is asserted, not reported."""
    raw = grow_coords_torch(bott_coords, iters=1, up_stages=1, dedup=False)
    uniq = grow_coords_torch(bott_coords, iters=1, up_stages=1, dedup=True)
    again = grow_coords_torch(bott_coords, iters=1, up_stages=1, dedup=True)
    assert torch.equal(uniq, again), "C is not deterministic"
    assert raw.shape[0] == bott_coords.shape[0] * 9, \
        f"raw growth should be exactly 9x the input, got {raw.shape[0]}"
    iters2 = grow_coords_torch(bott_coords, iters=2, up_stages=1, dedup=True)
    assert len(as_key_set(uniq)) < len(as_key_set(iters2)), \
        "iters=2 must reach strictly further than iters=1"
    return {"n_raw": raw.shape[0], "n_unique": len(as_key_set(uniq)),
            "n_unique_iters2": len(as_key_set(iters2)), "coords": uniq}


def case_d_densify(bbox, stride) -> dict:
    """D: enumerate the wiped rectangle."""
    cand = densify_cell(bbox, stride)
    return {"n_unique": len(as_key_set(cand)), "coords": cand}


def case_e_leakage(masked_coords, pre_mask_low, stride) -> dict:
    """E: inject only at masked coords -> the occupancy target is all-ones. The trap."""
    cand = downsample_unique(masked_coords, stride)
    tgt = occ_target(cand, pre_mask_low)
    assert cand.numel() > 0, "toy wiped no active pixels -- pick a busier cell"
    assert bool(tgt.all()), \
        "injecting only at masked coords must give an all-positive target"
    return {"n_unique": cand.shape[0], "pos_frac": float(tgt.float().mean()),
            "coords": cand}


def case_f_dedup_convention(grown_coords, kept_low) -> dict:
    """Do C's growth and our `_project_masked_to_skip` agree on what a duplicate is?

    C keys on `x*H + y` with H from its own max; `_project_masked_to_skip` keys on
    `y*W + x` with W from the skip's max + 2. Different strides, different axis order --
    if they disagree, injection silently keeps or drops cells. Checked by result, not by
    reading: the survivors must be exactly C's coords not already in the skip.
    """
    from models.minkunet_attention import MinkUNetSparseAttentionMAE as MAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skip = make_voxels(kept_low, channels=4, device=device,
                       tensor_stride=(HALF_STRIDE, HALF_STRIDE))
    # _project_masked_to_skip downsamples by `stride`, so pass stride=1 to compare the
    # already-half-res grown set against the half-res skip directly.
    out = MAE._project_masked_to_skip([grown_coords.to(device)], skip, stride=1)[0]
    survivors = as_key_set(out.long().cpu())
    expected = as_key_set(grown_coords) - as_key_set(kept_low)
    assert survivors == expected, (
        f"dedup conventions disagree: {len(survivors)} survivors vs {len(expected)} "
        f"expected; {len(survivors - expected)} unexpected, "
        f"{len(expected - survivors)} missing"
    )
    return {"n_in": grown_coords.shape[0], "n_survivors": len(survivors),
            "n_already_in_skip": len(as_key_set(grown_coords) & as_key_set(kept_low))}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    torch.manual_seed(0)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")

    print("=" * 78)
    print("2D generative-candidate approaches")
    print("=" * 78)
    print(f"canvas {IMAGE_W}x{IMAGE_H}  cell {CELL_W}x{CELL_H}  "
          f"bottleneck /{BOTTLENECK_STRIDE}  candidates /{HALF_STRIDE}")
    print(f"CUDA: {'yes -- ' + torch.cuda.get_device_name(0) if has_cuda else 'NO (A/B will skip)'}")
    print()

    # --- geometry -----------------------------------------------------------
    actives = make_toy_actives()
    # Pick the cell containing the most active pixels, so the wipe is a real hole.
    best, best_n = (0, 0), -1
    for ix in range(IMAGE_W // CELL_W):
        for iy in range(IMAGE_H // CELL_H):
            _, m, _ = wipe_one_cell(actives, ix, iy)
            if m.shape[0] > best_n:
                best, best_n = (ix, iy), m.shape[0]
    kept, masked, bbox = wipe_one_cell(actives, *best)

    pre_mask_low = downsample_unique(actives, HALF_STRIDE)
    kept_low = downsample_unique(kept, HALF_STRIDE)
    bott_coords = downsample_unique(kept, BOTTLENECK_STRIDE)

    print(f"actives {actives.shape[0]}  wiped cell {best} holds {masked.shape[0]} of them")
    print(f"kept {kept.shape[0]} -> /2 {kept_low.shape[0]} -> /4 bottleneck "
          f"{bott_coords.shape[0]}")
    print()

    results, failures = {}, []

    def run(name, fn, *a):
        try:
            results[name] = fn(*a)
            return results[name]
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
            results[name] = {"assert_failed": str(exc)}
            return results[name]

    # --- C, D, E, F: no GPU needed for the maths -----------------------------
    c = run("C", case_c_torch_growth, bott_coords)
    d = run("D", case_d_densify, bbox, HALF_STRIDE)
    e = run("E", case_e_leakage, masked, pre_mask_low, HALF_STRIDE)

    # --- A, B: warpconvnet, GPU only ----------------------------------------
    if has_cuda:
        a = run("A", case_a_native_2d, bott_coords, device)
        b = run("B", case_b_padded_3d, bott_coords, device)
    else:
        a = b = {"skipped": "no CUDA"}
        results["A"], results["B"] = a, b

    print("-" * 78)
    print("A  native 2D generative conv")
    if "skipped" in a:
        print("   SKIPPED (no CUDA)")
    elif a.get("ran"):
        print(f"   RAN -- {a['n_unique']} unique coords. {a['note']}")
    else:
        print(f"   FAILED as expected: {a['exc']} at {a['where']}")
        print(f"   message: {a['msg']}")

    print("B  z-padded 3D generative conv")
    if "skipped" in b:
        print("   SKIPPED (no CUDA)")
    elif b.get("ran"):
        print(f"   RAN -- {b['n_unique']} unique coords, z values {b['z_vals']}")
        if not b["z_clean"]:
            failures.append("B: dummy z axis grew -- padding leaked into the expansion")
    else:
        print(f"   FAILED: {b['exc']} at {b['where']}")
        print(f"   message: {b['msg']}")

    print("C  torch coordinate growth")
    print(f"   {c.get('n_raw')} raw (9x, duplicates kept) -> {c.get('n_unique')} unique;"
          f" iters=2 reaches {c.get('n_unique_iters2')}")

    print("D  densify the wiped cell")
    print(f"   {d.get('n_unique')} unique coords")

    print("E  inject at masked coords only")
    print(f"   {e.get('n_unique')} unique coords, positive fraction "
          f"{e.get('pos_frac')} <-- no negatives, no signal")
    print()

    # --- the headline: does the library agree with our hand-rolled growth? ---
    print("-" * 78)
    print("B vs C -- is the ported growth the conv's own coordinate set?")
    if b.get("ran") and "coords" in c:
        sb, sc = as_key_set(b["coords"]), as_key_set(c["coords"])
        if sb == sc:
            print(f"   MATCH on {len(sb)} coords -- the port reproduces the conv exactly")
        else:
            print(f"   MISMATCH: B {len(sb)}, C {len(sc)}, shared {len(sb & sc)}")
            print(f"   B-only {len(sb - sc)} e.g. {sorted(sb - sc)[:5]}")
            print(f"   C-only {len(sc - sb)} e.g. {sorted(sc - sb)[:5]}")
            # C clamps negatives to 0; the conv may not. Re-check without the clamp.
            cb = {p for p in sb if p[0] >= 0 and p[1] >= 0}
            print(f"   after dropping B's negative coords: "
                  f"{'MATCH' if cb == sc else 'still differs'}")
            failures.append("B != C: the ported growth is not the conv's coordinate set")
    else:
        print("   NOT TESTABLE -- B did not run, so nothing validates C's claim")
    print()

    # --- candidate-set economics --------------------------------------------
    print("-" * 78)
    print("Which candidate set would we train on?")
    print(f"{'approach':<38}{'coords':>9}{'positives':>11}{'pos frac':>10}")

    # Each approach is scored against the pre-mask actives at ITS OWN scale -- mixing
    # a /4 candidate set against a /2 reference would miscount every positive.
    pre_mask_at = {HALF_STRIDE: pre_mask_low,
                   BOTTLENECK_STRIDE: downsample_unique(actives, BOTTLENECK_STRIDE)}

    def row(label, coords, stride=HALF_STRIDE):
        t = occ_target(unique_rows(coords), pre_mask_at[stride])
        print(f"{label:<38}{t.numel():>9}{int(t.sum()):>11}{float(t.float().mean()):>10.3f}")
        return t

    dense = unique_rows(results["D"]["coords"])
    dense_t = occ_target(dense, pre_mask_low)
    # The fork's full option space, all of them policies over the same injection step.
    row("D  densified, all candidates", dense)
    row("D+K  densified, 8 neg/positive",
        subsample_negatives(dense, dense_t, neg_per_pos=8.0))
    row("D+hard  same budget, 50% boundary",
        hard_negatives(dense, dense_t, kept_low, neg_per_pos=8.0, frac=0.5))
    row("D/4  densified at coarse stride 4",
        densify_cell(bbox, BOTTLENECK_STRIDE), stride=BOTTLENECK_STRIDE)
    if "coords" in c:
        row("C  grown from bottleneck", c["coords"])
    row("E  masked coords only  <-- leaks", results["E"]["coords"])
    print()
    print("These sets are not the same KIND of set, and the counts must not be read as a")
    print("like-for-like swap. D covers the wiped cell only. C grows off the whole")
    print("surviving bottleneck, so it supervises a ring around every structure in the")
    print("image and emits nothing inside a fully-wiped hole (no support to grow from) --")
    print("that is what iters>1 buys. Choosing between them is a choice of where")
    print("occupancy is supervised, not just how many cells it costs.")
    print()
    print("Positive fraction is the number that matters for tuning: focal-loss defaults")
    print("assume a large imbalance, so a near-balanced grown set needs them retuned or")
    print("lambda_occ cut.")
    print()

    # --- dedup convention ----------------------------------------------------
    print("-" * 78)
    print("F  do C's dedup and _project_masked_to_skip agree?")
    if "coords" in c:
        try:
            f = run("F", case_f_dedup_convention, c["coords"], kept_low)
            if "assert_failed" not in f:
                print(f"   AGREE -- {f['n_in']} in, {f['n_survivors']} survive, "
                      f"{f['n_already_in_skip']} already in the skip")
        except Exception as exc:                              # noqa: BLE001
            print(f"   ERROR: {type(exc).__name__}: {exc}")
            failures.append(f"F: {type(exc).__name__}: {exc}")
    print()

    print("=" * 78)
    if failures:
        print(f"FAIL -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS -- every assertion held; see the tables above for the porting decision")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
