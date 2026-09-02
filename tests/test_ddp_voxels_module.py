"""
test_ddp_voxels_module.py
─────────────────────────
Does DistributedDataParallel reduce gradients for a module whose forward returns a
Voxels object rather than a tensor?

DDP walks a module's forward output to arm its backward pass. Our backbones return
`Voxels`, which is not a tensor, list, tuple or dict -- if DDP cannot see into it, the
reducer may never fire and the gradients are silently per-rank.

Measuring that through the real backbone failed: two identical fp32 runs of this stack
differ by ~2.6e-2 because sparse scatter uses atomics, and no arrangement of inputs made
the ranks differ enough to clear that floor (normalisation layers absorb an input
rescale). So this uses the smallest module that reproduces the *interface* -- one
nn.Linear over voxel features, wrapped back into a Voxels -- which is exact to float
precision and answers the DDP question without the noise.

The two ranks get deliberately different data, so a correctly reduced gradient is far
from either rank's own. Both structures are covered: one forward per backward, and the
three-forwards shape `forward_backward` uses.

Needs 2 GPUs. Spawns its own ranks.

Run:  python -u tests/test_ddp_voxels_module.py
"""

from __future__ import annotations

import contextlib
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

WORLD = 2
DIM_IN, DIM_OUT = 8, 16
N_VOX = 512


class VoxelLinear(nn.Module):
    """Minimal stand-in for a backbone: Voxels in, Voxels out, one weight to reduce."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DIM_IN, DIM_OUT)

    def forward(self, x: Voxels) -> Voxels:
        feats = self.fc(x.feature_tensor)
        return Voxels(
            batched_coordinates=IntCoords(x.coordinate_tensor, offsets=x.offsets),
            batched_features=CatFeatures(feats, offsets=x.offsets),
            offsets=x.offsets,
        )


class PlainLinear(nn.Module):
    """Control: identical maths, tensor in and tensor out."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DIM_IN, DIM_OUT)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)


def _inputs(device, rank, voxels: bool):
    g = torch.Generator().manual_seed(100 + rank)          # different data per rank
    feats = torch.randn(N_VOX, DIM_IN, generator=g).to(device) * (1.0 + rank)
    if not voxels:
        return feats
    coords = torch.arange(N_VOX, dtype=torch.int32).unsqueeze(1).repeat(1, 2).to(device)
    offsets = torch.tensor([0, N_VOX], dtype=torch.int64)
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


def _out_feats(y):
    return y.feature_tensor if isinstance(y, Voxels) else y


def _grads(module):
    core = module.module if isinstance(module, DistributedDataParallel) else module
    return torch.cat([p.grad.flatten() for p in core.parameters()])


def _run_structure(device, rank, cls, voxels, wrap, n_views):
    torch.manual_seed(0)                                   # identical weights per rank
    module = cls().to(device)
    if wrap:
        module = DistributedDataParallel(module, device_ids=[rank])
    module.zero_grad(set_to_none=False)

    xs = _inputs(device, rank, voxels)
    for view in range(n_views):
        is_last = view == n_views - 1
        with contextlib.ExitStack() as stack:
            if wrap and not is_last:
                stack.enter_context(module.no_sync())
            loss = _out_feats(module(xs)).pow(2).mean() / n_views
            loss.backward()
    return _grads(module)


def _case(device, rank, label, cls, voxels, n_views):
    local = _run_structure(device, rank, cls, voxels, wrap=False, n_views=n_views)
    manual = local.clone()
    dist.all_reduce(manual, op=dist.ReduceOp.AVG)
    ddp = _run_structure(device, rank, cls, voxels, wrap=True, n_views=n_views)

    def rel(a, b):
        return (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()

    if rank == 0:
        spread = rel(local, manual)
        to_manual = rel(ddp, manual)
        to_local = rel(ddp, local)
        verdict = "reduced" if to_manual < 1e-6 else (
            "NOT reduced" if to_local < 1e-6 else "neither")
        print(f"  {label:34} spread={spread:.3f}  "
              f"ddp→manual={to_manual:.2e}  ddp→local={to_local:.2e}  → {verdict}")
        return verdict
    return None


def _run(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29521"
    dist.init_process_group(backend="nccl", rank=rank, world_size=WORLD)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("\n  'reduced' = DDP produced the cross-rank average (what we need)\n")
    results = {}
    results["tensor, 1 forward"] = _case(device, rank, "tensor out, 1 forward",
                                         PlainLinear, False, 1)
    results["tensor, 3 forwards"] = _case(device, rank, "tensor out, 3 forwards+no_sync",
                                          PlainLinear, False, 3)
    results["voxels, 1 forward"] = _case(device, rank, "Voxels out, 1 forward",
                                         VoxelLinear, True, 1)
    results["voxels, 3 forwards"] = _case(device, rank, "Voxels out, 3 forwards+no_sync",
                                          VoxelLinear, True, 3)

    if rank == 0:
        print()
        ok = all(v == "reduced" for v in results.values())
        print("  ALL REDUCED — the wrapper works with our interface"
              if ok else "  see above: some structure is not reduced")
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
