"""
test_ddp_gradients.py
─────────────────────
Does the DDP wrapper produce the RIGHT gradients for our multi-forward structure?

`test_ddp_wrapper.py` showed the wrapper does not raise on three student forwards
followed by one backward -- but it ran at world_size=1, where the all-reduce is a no-op,
so it could not see a wrong answer. The concern is that DDP's reducer finalises on the
last forward and averages a gradient that is missing the earlier views' contributions.

This runs two real ranks and compares three quantities per parameter:

  local    each rank's own gradient, no synchronisation at all
  manual   the mean of the two local gradients, all-reduced by hand -- the reference,
           since averaging per-rank gradients is what data-parallel training means
  ddp      what DistributedDataParallel produces for the same inputs

If ddp == manual, the wrapper handles our structure and the hand-rolled path is dead
code. If it does not, the wrapper is silently wrong here and that is worth knowing
before a 40-hour run.

Needs 2 GPUs. Spawns its own ranks, so it runs under the plain test wrapper.

Run:  python -u tests/test_ddp_gradients.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_H, IMAGE_W = 400, 300
BATCH, N_PER_IMAGE = 2, 900
WORLD = 2


def _make_voxels(device, seed, scale=1.0):
    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.geometry.coords.integer import IntCoords
    from warpconvnet.geometry.features.cat import CatFeatures

    g = torch.Generator().manual_seed(seed)
    coords, feats = [], []
    for _ in range(BATCH):
        centre = torch.tensor([IMAGE_W // 2, IMAGE_H // 2])
        xy = (centre + torch.randn(N_PER_IMAGE, 2, generator=g) * 60).round().long()
        xy[:, 0].clamp_(0, IMAGE_W - 1)
        xy[:, 1].clamp_(0, IMAGE_H - 1)
        coords.append(xy.int())
        feats.append(torch.rand(N_PER_IMAGE, 1, generator=g) * scale)
    counts = torch.tensor([c.shape[0] for c in coords], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    return Voxels(
        batched_coordinates=IntCoords(torch.cat(coords).to(device), offsets=offsets),
        batched_features=CatFeatures(torch.cat(feats).to(device), offsets=offsets),
        offsets=offsets,
    )


class _Loss:
    """Scaled stand-in for the DINO loss.

    The scale is how the ranks are made to differ. Rescaling the *input* does not work:
    the normalisation layers absorb it and the ranks' gradients stay within the stack's
    own 2.6e-2 run-to-run noise, which is what made three earlier attempts undecidable.
    A loss scale passes straight through to every gradient and nothing can absorb it.
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, s_feats, s_bb_feats, t_feats, counts, is_masked=None):
        return s_feats.mean() * self.scale, None, None, None, None, None, None, None


def _build(device):
    from dino.model import DINODuneModel
    from dino.cropping import CropConfig, SparseCropper
    from dino.masking import SparseBlockMasker

    torch.manual_seed(0)                      # identical weights on every rank
    model = DINODuneModel(backbone_name="attn_mae", use_proj_head=False).to(device)
    model.train()
    cropper = SparseCropper(CropConfig(n_global=1, n_local=2, min_active_pixels=10,
                                       image_h=IMAGE_H, image_w=IMAGE_W))
    masker = SparseBlockMasker(mask_ratio=0.5, win_ch=5, win_tick=5)
    return model, cropper, masker


def _grads(model):
    return [p.grad.detach().clone() if p.grad is not None else None
            for p in model.student.parameters()]


def _flat(grads, ref):
    """One vector from a parameter-wise gradient list; missing grads count as zeros."""
    parts = []
    for g, r in zip(grads, ref):
        parts.append(g.flatten() if g is not None else torch.zeros_like(r).flatten())
    return torch.cat(parts)


def _rel(a, b):
    return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def _case(rank, device, label, use_cropping, use_masking, find_unused=False):
    """Compare local / manual-average / DDP gradients for one forward structure.

    Reported as relative L2 over the whole flattened gradient. A max-of-per-tensor
    ratios is not usable here: parameters whose gradient is near zero make the ratio
    explode, so it swings run to run and says nothing about whether the reduction is
    right.
    """
    from dino.model import DINODuneModel
    from dino.cropping import CropConfig, SparseCropper
    from dino.masking import SparseBlockMasker

    def fresh():
        torch.manual_seed(0)                  # identical weights on every rank
        model = DINODuneModel(backbone_name="attn_mae", use_proj_head=False).to(device)
        model.train()
        cropper = SparseCropper(CropConfig(n_global=1, n_local=2, min_active_pixels=10,
                                           image_h=IMAGE_H, image_w=IMAGE_W))
        masker = SparseBlockMasker(mask_ratio=0.5, win_ch=5, win_tick=5)
        torch.manual_seed(1234)               # identical augmentation draws
        return model, cropper, masker

    # --- local: this rank alone, no synchronisation ---
    model, cropper, masker = fresh()
    # Same data on every rank, and the ranks differ only in the loss scale below. With
    # rank 1 at 5x, the correct average is 3x rank 0's own gradient -- a 67% gap, far
    # clear of the 2.6e-2 noise, so "averaged" and "not averaged" cannot be confused.
    loss_scale = 1.0 + 4.0 * rank
    xs = _make_voxels(device, seed=0)
    model.forward_backward(xs, cropper, masker, _Loss(loss_scale),
                           use_cropping=use_cropping, use_masking=use_masking)
    params = list(model.student.parameters())
    local = [p.grad.detach().clone() if p.grad is not None else None for p in params]

    # --- manual: the mean of the ranks' local gradients (the reference) ---
    flat_local = _flat(local, params)
    flat_manual = flat_local.clone()
    dist.all_reduce(flat_manual, op=dist.ReduceOp.AVG)

    # --- ddp: identical inputs, through the wrapper ---
    model2, cropper2, masker2 = fresh()
    model2.student = DistributedDataParallel(model2.student, device_ids=[rank],
                                            find_unused_parameters=find_unused)
    xs2 = _make_voxels(device, seed=0)
    model2.forward_backward(xs2, cropper2, masker2, _Loss(loss_scale),
                            use_cropping=use_cropping, use_masking=use_masking)
    params2 = list(model2.student.module.parameters())
    ddp = [p.grad.detach().clone() if p.grad is not None else None for p in params2]
    flat_ddp = _flat(ddp, params2)

    if rank == 0:
        spread = _rel(flat_local, flat_manual)
        print(f"\n  [{label}]")
        print(f"    ||local  - manual|| / ||manual|| = {spread:.3e}   "
              f"(rank spread; the test is vacuous if ~0)")
        print(f"    ||ddp    - manual|| / ||manual|| = {_rel(flat_ddp, flat_manual):.3e}")
        print(f"    ||ddp    - local ||  / ||manual|| = {_rel(flat_ddp, flat_local):.3e}   "
              f"(~0.67 expected when DDP reduces; ~0 means it never did)")
        return _rel(flat_ddp, flat_manual), spread
    return None, None


def _run(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29519"
    dist.init_process_group(backend="nccl", rank=rank, world_size=WORLD)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Masking decides whether the MAE mask-token parameters get a gradient at all. With
    # DDP's default find_unused_parameters=False, a parameter that never receives one
    # leaves its bucket unfinished and the reduction silently does not happen -- which
    # is why the simplest case can be the one that fails. These four separate that
    # question from the number of views.
    cases = [
        ("single view, no mask",
         _case(rank, device, "single view, no mask", False, False)),
        ("single view, masked",
         _case(rank, device, "single view, masked", False, True)),
        ("three views, masked",
         _case(rank, device, "three views, masked", True, True)),
        ("no mask + find_unused",
         _case(rank, device, "single view, no mask, find_unused=True", False, False,
               find_unused=True)),
        # The production combination. Every case above either runs one view, where
        # forward_backward's no_sync() path never engages because the only view is the
        # last one, or runs find_unused_parameters=False, which training never uses.
        # Multi-view plus find_unused=True is the one that exercises the per-view
        # backward under no_sync with the reducer's unused-parameter scan, and it is
        # the combination train_dino.py actually configures.
        ("three views + find_unused",
         _case(rank, device, "three views, masked, find_unused=True", True, True,
               find_unused=True)),
    ]

    if rank == 0:
        print("\n" + "=" * 60)
        # Judged against the rank spread, not an absolute epsilon: the floor is the
        # stack's own nondeterminism, so "equal" can only mean "far closer to the
        # average than to this rank's own gradient".
        for label, (err, spread) in cases:
            verdict = "MATCH" if err < 0.25 * spread else "MISMATCH"
            print(f"  {label:24}: ddp vs manual {verdict}  "
                  f"(err {err:.2e} vs rank spread {spread:.2e}, "
                  f"ratio {err / max(spread, 1e-12):.3f})")

    dist.destroy_process_group()


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        print(f"SKIP: needs {WORLD} GPUs, found {torch.cuda.device_count()}")
        return 0
    try:
        mp.spawn(_run, nprocs=WORLD, join=True)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
