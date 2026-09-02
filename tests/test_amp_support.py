"""
test_amp_support.py
───────────────────
What mixed precision actually buys on our backbone, and whether it is safe to use.

warpconvnet takes its compute dtype from the autocast context
(`nn/functional/sparse_conv/helper.py:251`), so the sparse-convolution stages should
follow an `autocast` block rather than staying fp32. Should is not does: this measures
a real forward+backward of `attn_mae` at production settings in fp32, bf16 and fp16 and
reports time, peak memory, output dtype and how far the features drift.

It asserts only what must hold for AMP to be usable at all -- outputs finite, gradients
finite, the autocast dtype actually reaching the output. The speed and drift numbers are
reported, not asserted: they are the input to the decision, not a pass/fail.

Needs a GPU. Plain script, no pytest -- the cluster venv has none.

Run:  python -u tests/test_amp_support.py
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

from models import BACKBONE_REGISTRY, backbone_kwargs

# Production settings (gridutils/train/config_mae_pairfix_mixed.json).
BACKBONE = "attn_mae"
ENC_RANGE = 320.0
ENC_DIM = 32
IMAGE_H, IMAGE_W = 1500, 1050

# A batch shaped like a global crop: 100 images, a couple of thousand active voxels each.
BATCH = 100
N_PER_IMAGE = 2000


def _make_voxels(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords, feats = [], []
    for _ in range(BATCH):
        centre = torch.tensor([IMAGE_W // 2, IMAGE_H // 2])
        xy = (centre + torch.randn(N_PER_IMAGE, 2, generator=g) * 150).round().long()
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


def _build(device):
    torch.manual_seed(0)
    cls = BACKBONE_REGISTRY[BACKBONE]
    kwargs = backbone_kwargs(cls, encoding_range=ENC_RANGE, encoding_dim=ENC_DIM)
    return cls(**kwargs).to(device).train()


def _run_once(model, xs, amp_dtype):
    """One forward+backward, optionally under autocast. Returns the output features."""
    model.zero_grad(set_to_none=True)
    ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None
           else torch.autocast("cuda", enabled=False))
    with ctx:
        out = model(xs)
        feats = out.feature_tensor
        # Stand-in for the DINO loss: reduced in fp32, as the real one should be.
        loss = feats.float().pow(2).mean()
    loss.backward()
    return feats, loss


def _measure(model, xs, amp_dtype, iters=8):
    for _ in range(3):                                  # warmup + warpconvnet autotune
        _run_once(model, xs, amp_dtype)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        feats, loss = _run_once(model, xs, amp_dtype)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_finite = all(torch.isfinite(g).all().item() for g in grads)
    return {
        "ms": ms,
        "peak_gib": peak,
        "dtype": feats.dtype,
        "loss": float(loss),
        "finite": bool(torch.isfinite(feats).all().item()) and grad_finite,
        "feats": feats.detach().float(),
    }


def check_amp(device):
    model = _build(device)
    xs = _make_voxels(device)

    # fp32 twice: sparse gather/scatter uses atomics, so two identical runs need not
    # agree bit for bit. Without this control a "drift" number has no floor to be read
    # against and any value looks alarming or reassuring at random.
    modes = [("fp32", None), ("fp32'", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)]
    results = {}
    for name, dt in modes:
        try:
            results[name] = _measure(model, xs, dt)
        except Exception as e:
            print(f"  {name}: FAILED to run — {type(e).__name__}: {e}")
            results[name] = None

    base = results["fp32"]
    print(f"\n  {'mode':6} {'ms/step':>9} {'speedup':>8} {'peak GiB':>9} "
          f"{'out dtype':>12} {'finite':>7}  {'rel.drift':>10}  {'cosine':>8}")
    for name, _ in modes:
        r = results[name]
        if r is None:
            continue
        speed = base["ms"] / max(r["ms"], 1e-9)
        if name == "fp32":
            drift, cos = "-", "-"
        else:
            num = (r["feats"] - base["feats"]).abs().mean().item()
            den = base["feats"].abs().mean().item()
            drift = f"{num / max(den, 1e-12):.2e}"
            # Cosine says whether the features point the same way, which is what a
            # cosine-based loss and a kNN probe actually read; a mean-relative error
            # over near-zero features can look catastrophic while the direction holds.
            cos = f"{torch.nn.functional.cosine_similarity(r['feats'], base['feats'], dim=-1).mean().item():.4f}"
        print(f"  {name:6} {r['ms']:9.1f} {speed:7.2f}x {r['peak_gib']:9.2f} "
              f"{str(r['dtype']):>12} {str(r['finite']):>7}  {drift:>10}  {cos:>8}")

    b = base["feats"]
    print(f"\n  fp32 features: mean={b.mean().item():.3e} absmean={b.abs().mean().item():.3e} "
          f"std={b.std().item():.3e}")

    # What must hold for AMP to be usable at all.
    for name in ("bf16", "fp16"):
        r = results[name]
        assert r is not None, f"{name} autocast did not run"
        assert r["finite"], f"{name} produced non-finite features or gradients"
    assert results["bf16"]["dtype"] == torch.bfloat16, (
        f"bf16 autocast did not reach the backbone output "
        f"(got {results['bf16']['dtype']}) — the sparse convs are ignoring the context"
    )


CHECKS = [("amp_support", check_amp)]


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs a GPU")
        return 0
    device = torch.device("cuda")
    failures = []
    for name, fn in CHECKS:
        print(f"[{name}]")
        try:
            fn(device)
            print("\n  PASS")
        except Exception:
            traceback.print_exc()
            failures.append(name)
            print("  FAIL")
    print("=" * 60)
    print(f"{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
