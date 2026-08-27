"""
test_match_and_gather.py
────────────────────────
Exact-equality gate for the vectorised `match_and_gather`.

The function intersects student and teacher voxels by coordinate. It used to do that one
image at a time; it now encodes the batch index into the sort key and does the whole
batch in one sort plus one searchsorted. That is a pure speed change, so the bar is not
"close enough" but element-for-element identity with the straightforward per-image
version, which is kept here as the reference to compare against.

Identity is expected to hold exactly, not approximately: both versions *select* rows of
the same feature tensors, they do not compute new values. So the features are compared
with torch.equal, not allclose.

The cases cover what a real batch does: ragged images, images with no overlap, empty
images on either side, a batch empty on one side entirely, B=1, and both with and
without masked coordinates.

Needs a GPU. Plain script, no pytest — the cluster venv has none.

Run:  python -u tests/test_match_and_gather.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

from dino.model import match_and_gather

IMAGE_H, IMAGE_W = 1500, 1050


def _reference(s_out, s_backbone, t_out, masked_coords_per_batch=None):
    """Per-image intersection: the straightforward version the fast one must reproduce."""
    B = len(s_out.offsets) - 1
    device = s_out.feature_tensor.device

    W = 1
    if s_out.coordinate_tensor.shape[0] > 0:
        W = max(W, int(s_out.coordinate_tensor[:, 0].max().item()) + 1)
    if t_out.coordinate_tensor.shape[0] > 0:
        W = max(W, int(t_out.coordinate_tensor[:, 0].max().item()) + 1)

    s_global_idx, t_global_idx, counts_list = [], [], []
    is_masked_list = [] if masked_coords_per_batch is not None else None

    for b in range(B):
        s_start, s_end = int(s_out.offsets[b]), int(s_out.offsets[b + 1])
        t_start, t_end = int(t_out.offsets[b]), int(t_out.offsets[b + 1])
        s_coords = s_out.coordinate_tensor[s_start:s_end]
        t_coords = t_out.coordinate_tensor[t_start:t_end]

        if s_coords.shape[0] == 0 or t_coords.shape[0] == 0:
            counts_list.append(0)
            continue

        s_keys = s_coords[:, 1].long() * W + s_coords[:, 0].long()
        t_keys = t_coords[:, 1].long() * W + t_coords[:, 0].long()
        t_sorted, t_order = t_keys.sort()
        pos = torch.searchsorted(t_sorted, s_keys).clamp(max=t_sorted.shape[0] - 1)
        valid = t_sorted[pos] == s_keys

        s_local = valid.nonzero(as_tuple=False).squeeze(1)
        if s_local.numel() == 0:
            counts_list.append(0)
            continue
        t_local = t_order[pos[valid]]

        s_global_idx.append(s_local + s_start)
        t_global_idx.append(t_local + t_start)
        counts_list.append(s_local.shape[0])

        if is_masked_list is not None:
            m_coords = masked_coords_per_batch[b]
            if m_coords.shape[0] > 0:
                m_keys = m_coords[:, 1].long() * W + m_coords[:, 0].long()
                is_masked_list.append(torch.isin(s_keys[s_local], m_keys))
            else:
                is_masked_list.append(
                    torch.zeros(s_local.shape[0], dtype=torch.bool, device=device))

    counts = torch.tensor(counts_list, dtype=torch.int64, device=device)

    if s_global_idx:
        s_idx = torch.cat(s_global_idx)
        t_idx = torch.cat(t_global_idx)
        s_feats    = s_out.feature_tensor[s_idx]
        s_bb_feats = s_backbone.feature_tensor[s_idx]
        t_feats    = t_out.feature_tensor[t_idx]
    else:
        s_feats    = s_out.feature_tensor.new_zeros(0, s_out.feature_tensor.shape[1])
        s_bb_feats = s_backbone.feature_tensor.new_zeros(0, s_backbone.feature_tensor.shape[1])
        t_feats    = t_out.feature_tensor.new_zeros(0, t_out.feature_tensor.shape[1])

    is_masked = torch.cat(is_masked_list) if is_masked_list else None
    return s_feats, s_bb_feats, t_feats, counts, is_masked


# ─────────────────────────────────────────────────────────────────────────────
# Batch construction

def _voxels(coords_per_image, feats_per_image, device):
    counts = torch.tensor([c.shape[0] for c in coords_per_image], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    coords = (torch.cat(coords_per_image) if coords_per_image
              else torch.zeros(0, 2, dtype=torch.int32))
    feats = (torch.cat(feats_per_image) if feats_per_image
             else torch.zeros(0, feats_per_image[0].shape[1]))
    return Voxels(
        batched_coordinates=IntCoords(coords.to(device), offsets=offsets),
        batched_features=CatFeatures(feats.to(device), offsets=offsets),
        offsets=offsets,
    )


def _case(device, sizes, overlap=0.6, d_head=128, d_bb=64, seed=0, masked_frac=0.5):
    """A student/teacher pair per image: `overlap` of the student's coords also in the teacher.

    sizes is a list of (n_student, n_teacher) per image; either may be 0.
    """
    g = torch.Generator().manual_seed(seed)
    s_c, t_c, s_f, s_bb, t_f, masked = [], [], [], [], [], []

    for i, (n_s, n_t) in enumerate(sizes):
        # Draw coordinates without repeats: a coordinate appears at most once per image.
        pool = torch.randperm(IMAGE_H * IMAGE_W, generator=g)[: n_s + n_t]
        s_flat = pool[:n_s]
        n_shared = int(overlap * min(n_s, n_t))
        t_flat = torch.cat([s_flat[:n_shared], pool[n_s: n_s + n_t - n_shared]])

        def to_xy(flat):
            return torch.stack([flat % IMAGE_W, flat // IMAGE_W], dim=1).int()

        s_c.append(to_xy(s_flat))
        t_c.append(to_xy(t_flat))
        s_f.append(torch.randn(n_s, d_head, generator=g))
        s_bb.append(torch.randn(n_s, d_bb, generator=g))
        t_f.append(torch.randn(t_flat.shape[0], d_head, generator=g))

        n_m = int(masked_frac * n_s)
        masked.append(to_xy(s_flat[:n_m]).to(device))

    return (_voxels(s_c, s_f, device), _voxels(s_c, s_bb, device),
            _voxels(t_c, t_f, device), masked)


def _assert_same(fast, ref, label):
    names = ("s_feats", "s_bb_feats", "t_feats", "counts", "is_masked")
    for name, a, b in zip(names, fast, ref):
        if a is None or b is None:
            assert a is None and b is None, f"{label}: {name} is None in only one version"
            continue
        assert a.shape == b.shape, f"{label}: {name} shape {tuple(a.shape)} vs {tuple(b.shape)}"
        assert a.dtype == b.dtype, f"{label}: {name} dtype {a.dtype} vs {b.dtype}"
        assert torch.equal(a, b), f"{label}: {name} differs"
    print(f"  {label}: {int(fast[3].sum())} matched over {fast[3].numel()} images — identical")


# ─────────────────────────────────────────────────────────────────────────────
# Checks

def check_typical_batch(device):
    """A realistic ragged batch, with and without masked coordinates."""
    sizes = [(900, 1100), (1500, 1400), (700, 700), (2000, 1800)]
    s, s_bb, t, m = _case(device, sizes)
    _assert_same(match_and_gather(s, s_bb, t, m), _reference(s, s_bb, t, m), "with mask")
    _assert_same(match_and_gather(s, s_bb, t), _reference(s, s_bb, t), "no mask")


def check_empty_images(device):
    """Images empty on the student side, the teacher side, and both."""
    sizes = [(0, 800), (900, 0), (0, 0), (1200, 1000)]
    s, s_bb, t, m = _case(device, sizes)
    _assert_same(match_and_gather(s, s_bb, t, m), _reference(s, s_bb, t, m), "empty images")


def check_no_overlap(device):
    """Images whose student and teacher coordinates are disjoint contribute nothing."""
    sizes = [(600, 600), (800, 800)]
    s, s_bb, t, m = _case(device, sizes, overlap=0.0)
    _assert_same(match_and_gather(s, s_bb, t, m), _reference(s, s_bb, t, m), "no overlap")


def check_one_side_entirely_empty(device):
    """A batch with no teacher voxels at all — the clamp has nothing to clamp against."""
    sizes = [(500, 0), (700, 0)]
    s, s_bb, t, m = _case(device, sizes)
    _assert_same(match_and_gather(s, s_bb, t, m), _reference(s, s_bb, t, m), "teacher empty")


def check_single_image(device):
    """B = 1: the batch prefix in the key must not disturb the degenerate case."""
    s, s_bb, t, m = _case(device, [(1000, 1000)])
    _assert_same(match_and_gather(s, s_bb, t, m), _reference(s, s_bb, t, m), "B=1")


def check_no_masked_coords(device):
    """Masked coords supplied but empty everywhere: all matches tagged unmasked."""
    sizes = [(800, 800), (900, 900)]
    s, s_bb, t, m = _case(device, sizes, masked_frac=0.0)
    fast = match_and_gather(s, s_bb, t, m)
    _assert_same(fast, _reference(s, s_bb, t, m), "no masked coords")
    assert fast[4] is not None and not fast[4].any(), "expected all-unmasked tags"


def check_speedup(device):
    """Not a correctness gate: report what the rewrite bought at a realistic batch size."""
    sizes = [(900 + 7 * i, 1000 + 5 * i) for i in range(100)]
    s, s_bb, t, m = _case(device, sizes)

    def timeit(fn, n=10):
        for _ in range(3):
            fn(s, s_bb, t, m)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn(s, s_bb, t, m)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    fast_ms, ref_ms = timeit(match_and_gather), timeit(_reference)
    print(f"  B=100: {ref_ms:.1f} ms per-image → {fast_ms:.1f} ms vectorised "
          f"({ref_ms / max(fast_ms, 1e-9):.1f}x)")
    _assert_same(match_and_gather(s, s_bb, t, m), _reference(s, s_bb, t, m), "B=100")


CHECKS = [
    ("typical_batch", check_typical_batch),
    ("empty_images", check_empty_images),
    ("no_overlap", check_no_overlap),
    ("one_side_entirely_empty", check_one_side_entirely_empty),
    ("single_image", check_single_image),
    ("no_masked_coords", check_no_masked_coords),
    ("speedup", check_speedup),
]


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs a GPU")
        return 0

    device = torch.device("cuda")
    failures = []
    for name, fn in CHECKS:
        print(f"\n[{name}]")
        try:
            fn(device)
            print("  PASS")
        except Exception:
            traceback.print_exc()
            failures.append(name)
            print("  FAIL")

    print("\n" + "=" * 60)
    print(f"{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
