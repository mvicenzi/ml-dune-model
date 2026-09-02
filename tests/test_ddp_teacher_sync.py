"""
test_ddp_teacher_sync.py
────────────────────────
Do all ranks start from the same model under DDP -- teacher included?

DDP's wrapper broadcast synchronises only the modules it wraps (student, student
head). The teacher is a construction-time copy of the rank-local student and is
never wrapped, so it is only rank-identical if (a) every rank builds the model
from the same RNG state and (b) the explicit teacher broadcast runs. If either
is missing, each rank distils against a different teacher: the objective itself
differs per rank, with no error raised anywhere.

Two checks:
  1. Negative control -- build under a per-rank seed with no teacher broadcast,
     the way the bug looked: the teachers MUST differ, or this test cannot
     detect the failure it exists for.
  2. The training-loop sequence -- shared seed, DDP wrap, teacher broadcast
     from rank 0: student, head, teacher and teacher head must be bit-identical
     across ranks.

Needs 2 GPUs. Plain script, no pytest -- the cluster venv has none.

Run:  python -u tests/test_ddp_teacher_sync.py
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

from dino.model import DINODuneModel

WORLD = 2
SEED = 42


def _build(device):
    return DINODuneModel(backbone_name="attn_mae", use_proj_head=True).to(device)


def _tensors(module):
    return list(module.parameters()) + list(module.buffers())


def _identical_across_ranks(module, device) -> bool:
    """True when every parameter and buffer is bit-identical on all ranks."""
    ok = True
    for t in _tensors(module):
        ref = t.data.clone()
        dist.broadcast(ref, src=0)
        if not torch.equal(ref, t.data):
            ok = False
    flag = torch.tensor([1.0 if ok else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item() == 1.0)


def _run(rank, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29521"
    dist.init_process_group(backend="nccl", rank=rank, world_size=WORLD)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # ── 1. Negative control: per-rank seed, no teacher broadcast ──
    torch.manual_seed(SEED + rank)
    model = _build(device)
    model.student = DistributedDataParallel(model.student, device_ids=[rank])
    model.student_head = DistributedDataParallel(model.student_head, device_ids=[rank])
    teacher_diverged = not _identical_across_ranks(model.teacher, device)
    # the wrapper broadcast must have fixed the student regardless
    student_synced_anyway = _identical_across_ranks(model.student.module, device)
    del model

    # ── 2. The training-loop sequence ──
    torch.manual_seed(SEED)
    model = _build(device)
    model.student = DistributedDataParallel(model.student, device_ids=[rank])
    model.student_head = DistributedDataParallel(model.student_head, device_ids=[rank])
    for module in (model.teacher, model.teacher_head):
        if module is not None:
            for t in _tensors(module):
                dist.broadcast(t.data, src=0)

    checks = {
        "negative control: per-rank seed diverges the teacher": teacher_diverged,
        "negative control: DDP wrap still syncs the student":   student_synced_anyway,
        "fixed sequence: student identical across ranks":
            _identical_across_ranks(model.student.module, device),
        "fixed sequence: student head identical across ranks":
            _identical_across_ranks(model.student_head.module, device),
        "fixed sequence: teacher identical across ranks":
            _identical_across_ranks(model.teacher, device),
        "fixed sequence: teacher head identical across ranks":
            _identical_across_ranks(model.teacher_head, device),
    }

    if rank == 0:
        print("=" * 60)
        for label, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        results["n_fail"] = sum(1 for ok in checks.values() if not ok)

    dist.destroy_process_group()


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        print(f"SKIP: needs {WORLD} GPUs, found {torch.cuda.device_count()}")
        return 0
    results = mp.Manager().dict()
    try:
        mp.spawn(_run, args=(results,), nprocs=WORLD, join=True)
    except Exception:
        traceback.print_exc()
        return 1
    n_fail = results.get("n_fail", -1)
    if n_fail != 0:
        print(f"\n{n_fail if n_fail >= 0 else 'unknown'} check(s) FAILED")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
