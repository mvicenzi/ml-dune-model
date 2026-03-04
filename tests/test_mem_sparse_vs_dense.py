"""
test_mem_sparse_vs_dense.py
───────────────────────────
Compares peak GPU memory between:
  • Dense 2D CNN   (standard torch.nn.Conv2d)
  • Sparse 2D CNN  (WarpConvNet SparseConv2d + Voxels)
for MNIST-style classification  (1×28×28 input, 10 classes).

Memory is measured for:
  • Inference step  (torch.no_grad forward pass)
  • Training step   (forward + backward + optimizer.step)

Batch sizes tested: 16, 64, 256.
Input sparsity: ~80 % zero pixels (mimics real MNIST digit images).

Run individually:
    python tests/test_mem_sparse_vs_dense.py

Run via pytest:
    pytest tests/test_mem_sparse_vs_dense.py -v -s
"""

from __future__ import annotations

import gc
import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ─────────────────────────────────────────────────────────────────────────────
# Optional WarpConvNet import
# ─────────────────────────────────────────────────────────────────────────────
try:
    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.nn.modules.sparse_conv import SparseConv2d
    from warpconvnet.nn.modules.activations import ReLU as SparseReLU
    from warpconvnet.nn.modules.sequential import Sequential as SparseSequential
    WARPCONVNET_AVAILABLE = True
except ImportError:
    WARPCONVNET_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Global pytest markers
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
needs_sparse = pytest.mark.skipif(
    not WARPCONVNET_AVAILABLE,
    reason="warpconvnet not installed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class DenseCNN(nn.Module):
    """Standard dense 2D CNN for MNIST-style classification.

    Architecture (28×28 input):
        Conv2d(1→32,   k=3, s=1, pad=1) + BN2d + ReLU  →  28×28
        Conv2d(32→64,  k=2, s=2)        + BN2d + ReLU  →  14×14
        Conv2d(64→128, k=2, s=2)        + BN2d + ReLU  →   7×7
        AdaptiveAvgPool2d(1) → Flatten → Linear(128, n_classes)
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  32,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64,  kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class SparseCNN(nn.Module):
    """WarpConvNet sparse 2D CNN for MNIST-style classification.

    Architecture (28×28 input, converted to sparse Voxels):
        SparseConv2d(1→32,   k=3, s=1) + BN1d + ReLU  →  28×28 sparse
        SparseConv2d(32→64,  k=2, s=2) + BN1d + ReLU  →  14×14 sparse
        SparseConv2d(64→128, k=2, s=2) + BN1d + ReLU  →   7×7  sparse
        to_dense(7×7) → AdaptiveAvgPool2d(1) → Flatten → Linear(128, n_classes)

    Spatial math: two k=2, s=2 downsamples on a 28×28 grid give 14×14 then 7×7.
    BN1d operates on feature_tensor shape (N_active, C), matching WarpConvNet's
    SparseSequential convention (same as used in models/blocks.py).
    """

    _DENSE_SPATIAL = (7, 7)  # expected spatial size after two stride-2 downsamples

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.enc0 = SparseSequential(
            SparseConv2d(1,  32,  kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            SparseReLU(inplace=True),
        )
        self.enc1 = SparseSequential(
            SparseConv2d(32, 64,  kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            SparseReLU(inplace=True),
        )
        self.enc2 = SparseSequential(
            SparseConv2d(64, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            SparseReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        xs = Voxels.from_dense(x)       # dense [B,1,28,28] → sparse Voxels
        xs = self.enc0(xs)
        xs = self.enc1(xs)
        xs = self.enc2(xs)
        dense = xs.to_dense(channel_dim=1, spatial_shape=self._DENSE_SPATIAL)
        return self.head(dense)         # [B, n_classes]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────

def make_batch(
    batch_size: int,
    sparsity: float = 0.80,
    device: torch.device = DEVICE,
) -> tuple[Tensor, Tensor]:
    """
    Synthetic MNIST-like batch.

    Creates 1×28×28 images where approximately `sparsity` fraction of pixels
    are exactly 0 (background) and the remainder are drawn from Uniform(0, 1].
    This matches the structure of real MNIST digits (~20 % active pixels).
    """
    x = torch.rand(batch_size, 1, 28, 28, device=device)
    x *= (torch.rand_like(x) > sparsity).float()   # zero out ~sparsity fraction
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Memory measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_reset(device: torch.device) -> None:
    """Free caches and reset peak-memory tracker."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> float:
    """Return peak allocated GPU memory (MB) since last reset."""
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def measure_inference(model: nn.Module, x: Tensor, device: torch.device) -> float:
    """
    Peak GPU memory (MB) for a single forward pass under torch.no_grad.

    Measurement window: from just before the forward call to synchronization
    after it.  Model weights already on GPU are included in the baseline.
    """
    model.eval()
    _gpu_reset(device)
    with torch.no_grad():
        model(x)
    return _peak_mb(device)


def measure_training(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    device: torch.device,
) -> float:
    """
    Peak GPU memory (MB) for one forward + backward + optimizer.step.

    Uses plain SGD (no momentum) to keep optimizer state minimal and keep
    the comparison focused on activation and gradient memory.
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    _gpu_reset(device)
    opt.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()
    return _peak_mb(device)


# ─────────────────────────────────────────────────────────────────────────────
# Individual model tests
# ─────────────────────────────────────────────────────────────────────────────

@needs_gpu
@pytest.mark.parametrize("batch_size", [16, 64, 256])
def test_dense_inference_memory(batch_size: int) -> None:
    """Dense CNN: inference peak memory is positive and reported."""
    x, _ = make_batch(batch_size)
    model = DenseCNN().to(DEVICE)
    mem = measure_inference(model, x, DEVICE)
    print(f"\n[Dense  inference | batch={batch_size:>3}]  peak = {mem:7.1f} MB")
    assert mem > 0


@needs_gpu
@pytest.mark.parametrize("batch_size", [16, 64, 256])
def test_dense_training_memory(batch_size: int) -> None:
    """Dense CNN: training peak memory is positive and reported."""
    x, y = make_batch(batch_size)
    model = DenseCNN().to(DEVICE)
    mem = measure_training(model, x, y, DEVICE)
    print(f"\n[Dense  training  | batch={batch_size:>3}]  peak = {mem:7.1f} MB")
    assert mem > 0
    # Training must allocate more than inference (gradients + activations)
    inf_mem = measure_inference(model, x, DEVICE)
    assert mem >= inf_mem, "Training peak should be >= inference peak"


@needs_gpu
@needs_sparse
@pytest.mark.parametrize("batch_size", [16, 64, 256])
def test_sparse_inference_memory(batch_size: int) -> None:
    """Sparse CNN: inference peak memory is positive and reported."""
    x, _ = make_batch(batch_size)
    model = SparseCNN().to(DEVICE)
    mem = measure_inference(model, x, DEVICE)
    print(f"\n[Sparse inference | batch={batch_size:>3}]  peak = {mem:7.1f} MB")
    assert mem > 0


@needs_gpu
@needs_sparse
@pytest.mark.parametrize("batch_size", [16, 64, 256])
def test_sparse_training_memory(batch_size: int) -> None:
    """Sparse CNN: training peak memory is positive and reported."""
    x, y = make_batch(batch_size)
    model = SparseCNN().to(DEVICE)
    mem = measure_training(model, x, y, DEVICE)
    print(f"\n[Sparse training  | batch={batch_size:>3}]  peak = {mem:7.1f} MB")
    assert mem > 0


# ─────────────────────────────────────────────────────────────────────────────
# Head-to-head comparison
# ─────────────────────────────────────────────────────────────────────────────

@needs_gpu
@needs_sparse
@pytest.mark.parametrize("batch_size", [16, 64, 256])
def test_memory_comparison(batch_size: int) -> None:
    """
    Head-to-head peak GPU memory comparison: dense vs. sparse CNN.

    Both models have identical channel widths (32→64→128) and the same
    two stride-2 downsampling stages, making the comparison fair.

    Assertion: sparse model must not use more than 5× the dense model's peak
    memory for 80 %-sparse MNIST-like inputs.  In practice, at high sparsity
    and large batch sizes the sparse model should use *less* memory than dense.
    """
    SPARSITY = 0.80
    x, y = make_batch(batch_size, sparsity=SPARSITY)

    # ── Dense model ────────────────────────────────────────────────────────
    dense = DenseCNN().to(DEVICE)
    d_inf = measure_inference(dense, x, DEVICE)
    d_tr  = measure_training (dense, x, y, DEVICE)
    del dense
    _gpu_reset(DEVICE)

    # ── Sparse model ───────────────────────────────────────────────────────
    sparse = SparseCNN().to(DEVICE)
    s_inf = measure_inference(sparse, x, DEVICE)
    s_tr  = measure_training (sparse, x, y, DEVICE)
    del sparse
    _gpu_reset(DEVICE)

    # ── Report ─────────────────────────────────────────────────────────────
    W = 58
    print(f"\n{'='*W}")
    print(f" Memory comparison  batch={batch_size}  sparsity={SPARSITY:.0%}")
    print(f"{'─'*W}")
    print(f"  {'Mode':<12} {'Dense':>9} MB   {'Sparse':>9} MB   {'Ratio':>7}")
    print(f"{'─'*W}")
    print(f"  {'Inference':<12} {d_inf:>9.1f}      {s_inf:>9.1f}      {s_inf/d_inf:>6.2f}×")
    print(f"  {'Training':<12} {d_tr:>9.1f}      {s_tr:>9.1f}      {s_tr/d_tr:>6.2f}×")
    print(f"{'='*W}")

    # ── Assertions ─────────────────────────────────────────────────────────
    assert s_inf / d_inf < 5.0, (
        f"Sparse inference memory ({s_inf:.1f} MB) is >5× dense ({d_inf:.1f} MB) "
        f"for batch_size={batch_size}"
    )
    assert s_tr / d_tr < 5.0, (
        f"Sparse training memory ({s_tr:.1f} MB) is >5× dense ({d_tr:.1f} MB) "
        f"for batch_size={batch_size}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone smoke-run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SPARSITY = 0.80
    BATCH_SIZES = [16, 64, 256]

    print(f"Device        : {DEVICE}")
    print(f"WarpConvNet   : {'available' if WARPCONVNET_AVAILABLE else 'NOT available'}")
    print(f"Input sparsity: {SPARSITY:.0%}")

    if not torch.cuda.is_available():
        print("\nNo CUDA device — skipping memory measurements.")
    else:
        W = 72
        hdr = f"  {'batch':>5}  {'D-inf':>8}  {'D-tr':>8}"
        if WARPCONVNET_AVAILABLE:
            hdr += f"  {'S-inf':>8}  {'S-tr':>8}  {'inf-ratio':>10}  {'tr-ratio':>9}"
        print(f"\n{'─'*W}")
        print(hdr)
        print(f"{'─'*W}")

        for bs in BATCH_SIZES:
            x, y = make_batch(bs, sparsity=SPARSITY)

            dense = DenseCNN().to(DEVICE)
            d_inf = measure_inference(dense, x, DEVICE)
            d_tr  = measure_training (dense, x, y, DEVICE)
            del dense; _gpu_reset(DEVICE)

            row = f"  {bs:>5}  {d_inf:>7.1f}M  {d_tr:>7.1f}M"

            if WARPCONVNET_AVAILABLE:
                sparse = SparseCNN().to(DEVICE)
                s_inf = measure_inference(sparse, x, DEVICE)
                s_tr  = measure_training (sparse, x, y, DEVICE)
                del sparse; _gpu_reset(DEVICE)
                row += (
                    f"  {s_inf:>7.1f}M  {s_tr:>7.1f}M"
                    f"  {s_inf/d_inf:>9.2f}×  {s_tr/d_tr:>8.2f}×"
                )
            else:
                row += "         —         —           —         —"

            print(row)

        print(f"{'─'*W}")
