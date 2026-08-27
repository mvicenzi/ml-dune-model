"""
test_ddp_wrapper.py
───────────────────
Does torch's DistributedDataParallel wrapper work with our forward_backward?

The wrapper is the standard tool and the default choice; using a hand-rolled all-reduce
instead needs evidence, not preference. The concern is structural: `forward_backward`
encodes each view with the student (three calls at crop_n_global=1, crop_n_local=2) and
then runs a single `.backward()` on their summed loss. DDP's reducer marks a parameter
ready when its gradient hook fires, and expects that once per backward.

Run with world_size=1 in a single process: the reducer still runs, so the structural
question is answered without needing multiple GPUs.

Needs a GPU. Plain script, no pytest -- the cluster venv has none.

Run:  python -u tests/test_ddp_wrapper.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

from dino.model import DINODuneModel
from dino.cropping import CropConfig, SparseCropper
from dino.masking import SparseBlockMasker

IMAGE_H, IMAGE_W = 400, 300
BATCH, N_PER_IMAGE = 2, 900


def _make_voxels(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords, feats = [], []
    for _ in range(BATCH):
        centre = torch.tensor([IMAGE_W // 2, IMAGE_H // 2])
        xy = (centre + torch.randn(N_PER_IMAGE, 2, generator=g) * 60).round().long()
        xy[:, 0].clamp_(0, IMAGE_W - 1)
        xy[:, 1].clamp_(0, IMAGE_H - 1)
        coords.append(xy.int())
        feats.append(torch.rand(N_PER_IMAGE, 1, generator=g))
    counts = torch.tensor([c.shape[0] for c in coords], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    return Voxels(
        batched_coordinates=IntCoords(torch.cat(coords).to(device), offsets=offsets),
        batched_features=CatFeatures(torch.cat(feats).to(device), offsets=offsets),
        offsets=offsets,
    )


class _Loss:
    """Minimal stand-in for PixelDINOLoss: forward_backward only needs 8 returns."""
    def __call__(self, s_feats, s_bb_feats, t_feats, counts, is_masked=None):
        return s_feats.mean(), None, None, None, None, None, None, None


def _build(device, n_global=1, n_local=2):
    torch.manual_seed(0)
    model = DINODuneModel(backbone_name="attn_mae", use_proj_head=False).to(device)
    model.train()
    cropper = SparseCropper(CropConfig(n_global=n_global, n_local=n_local,
                                       min_active_pixels=10,
                                       image_h=IMAGE_H, image_w=IMAGE_W))
    masker = SparseBlockMasker(mask_ratio=0.5, win_ch=5, win_tick=5)
    return model, cropper, masker


def check_wrapper_with_multi_forward(device):
    """Wrap the student and run forward_backward as the training loop does."""
    model, cropper, masker = _build(device)
    model.student = DistributedDataParallel(model.student, device_ids=[0])

    try:
        model.forward_backward(_make_voxels(device), cropper, masker, _Loss(),
                               use_cropping=True, use_masking=True)
    except RuntimeError as e:
        print(f"  DDP wrapper REJECTS our structure:\n    {str(e).splitlines()[0]}")
        return "rejected"
    print("  DDP wrapper accepted three student forwards and one backward")
    return "accepted"


def check_wrapper_with_single_view(device):
    """Control: one view means one forward, which the wrapper is built for."""
    model, cropper, masker = _build(device)
    model.student = DistributedDataParallel(model.student, device_ids=[0])
    try:
        model.forward_backward(_make_voxels(device), cropper, masker, _Loss(),
                               use_cropping=False, use_masking=True)
    except RuntimeError as e:
        print(f"  unexpectedly rejected the single-view case: {str(e).splitlines()[0]}")
        return "rejected"
    print("  single forward + single backward: accepted")
    return "accepted"


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs a GPU")
        return 0

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    results = {}
    for name, fn in (("multi_forward", check_wrapper_with_multi_forward),
                     ("single_view_control", check_wrapper_with_single_view)):
        print(f"\n[{name}]")
        try:
            results[name] = fn(device)
        except Exception:
            traceback.print_exc()
            results[name] = "error"

    print("\n" + "=" * 60)
    print(f"multi-forward: {results.get('multi_forward')}   "
          f"single-view control: {results.get('single_view_control')}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
