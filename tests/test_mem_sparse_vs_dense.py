"""
test_mem_sparse_vs_dense.py
───────────────────────────
Peak GPU memory comparison for 500×500 image classification across three models:

  Model 1 – DenseCNN        : standard Conv2d.  Tensor → Tensor.
  Model 2 – SparseCNN_Dense : sparse encoder + sparse attention bottleneck.
                               Tensor → Tensor  (Voxels.from_dense called inside forward).
  Model 3 – SparseCNN_Voxels: same sparse architecture.
                               Voxels → Tensor  (caller pre-converts; from_dense cost excluded).

Encoder channels match the real project model (1→32→64, bottleneck attn_channels=128).
Dense bottleneck uses N_ATTN residual conv3×3 layers  (dense self-attention at 125×125
would require O(125^4) memory — impractical).

Memory measured for:
  • Inference  – torch.no_grad forward pass
  • Training   – forward + backward + SGD step

Run standalone:   python tests/test_mem_sparse_vs_dense.py
Run via pytest:   source setup.sh && python -m pytest tests/test_mem_sparse_vs_dense.py -v -s
"""

from __future__ import annotations

import gc
import sys
import os
import pytest
import torch
import torch.nn as nn
from torch import Tensor

# Make project root importable so "from models.blocks import ..." works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────────
# Optional WarpConvNet + project-model imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    from warpconvnet.geometry.types.voxels import Voxels
    from warpconvnet.nn.modules.sparse_conv import SparseConv2d
    from warpconvnet.nn.modules.activations import ReLU as SparseReLU
    from warpconvnet.nn.modules.sequential import Sequential as SparseSequential
    from models.blocks import BottleneckSparseAttention2D
    WARPCONVNET_AVAILABLE = True
except ImportError:
    WARPCONVNET_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Pytest markers
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
needs_sparse = pytest.mark.skipif(
    not WARPCONVNET_AVAILABLE, reason="warpconvnet not installed"
)

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants  (match real MinkUNetSparseAttention)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE       = 500    # H = W of input image
N_CLASSES      = 4      # numu / nue / nutau / NC
N_ATTN         = 2      # attention (or residual-conv) layers in bottleneck
BOTTLENECK_CH  = 64     # channels entering the bottleneck
ATTN_CH        = 128    # projected attention dimension inside BottleneckSparseAttention2D
ATTN_HEADS     = 4
# Spatial coords at 125×125 resolution span [0, 124];  encoding_range=125 normalises them.
ENCODING_RANGE = 125.0

# ─────────────────────────────────────────────────────────────────────────────
# Model 1 – Dense CNN  (with comparable attention bottleneck)
# ─────────────────────────────────────────────────────────────────────────────

class DenseBottleneckAttention(nn.Module):
    """
    Dense spatial multi-head self-attention, structurally identical to
    BottleneckSparseAttention2D:

        pre_proj (Conv2d 1×1)
        → LayerNorm → MHA → residual
        → LayerNorm → MLP → residual
        → post_proj (Conv2d 1×1)

    Treats every spatial location as a token: [B, C, H, W] → [B, H·W, C].
    Uses nn.MultiheadAttention with need_weights=False to activate PyTorch's
    flash-attention fast path (avoids O(N²) attention-matrix materialisation).

    At 125×125 = 15,625 tokens this processes 5× more tokens than the sparse
    counterpart (~3,125 active voxels at 80% sparsity), making it the key
    variable in the memory comparison.
    """

    def __init__(
        self,
        channels:     int,
        attn_channels: int,
        heads:        int   = 4,
        mlp_ratio:    float = 2.0,
    ) -> None:
        super().__init__()
        self.pre_proj  = nn.Conv2d(channels, attn_channels, kernel_size=1)
        self.norm1     = nn.LayerNorm(attn_channels)
        self.norm2     = nn.LayerNorm(attn_channels)
        self.attn      = nn.MultiheadAttention(attn_channels, heads,
                                               batch_first=True, bias=True)
        hidden = int(attn_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(attn_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, attn_channels),
        )
        self.post_proj = nn.Conv2d(attn_channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:           # x: [B, C, H, W]
        B, _, H, W = x.shape
        x2 = self.pre_proj(x)                         # [B, attn_ch, H, W]
        x2 = x2.flatten(2).permute(0, 2, 1)          # [B, H·W, attn_ch]

        # Attention sub-block  (mirrors sparse: norm → attn → residual)
        x2n = self.norm1(x2)
        h, _ = self.attn(x2n, x2n, x2n, need_weights=False)
        x2 = x2n + h

        # MLP sub-block  (mirrors sparse: norm → mlp → residual)
        x2n = self.norm2(x2)
        x2 = x2n + self.mlp(x2n)

        x2 = x2.permute(0, 2, 1).reshape(B, -1, H, W)   # [B, attn_ch, H, W]
        return self.post_proj(x2)                         # [B, C, H, W]


class DenseCNN(nn.Module):
    """
    Dense 2D CNN with spatial attention bottleneck for 500×500 image classification.

    Architecture:
        Conv2d(1→32,  k=3, s=1, pad=1) + BN + ReLU  →  500×500
        Conv2d(32→64, k=2, s=2)        + BN + ReLU  →  250×250
        Conv2d(64→64, k=2, s=2)        + BN + ReLU  →  125×125
        DenseBottleneckAttention(64, attn_ch=128, heads=4) × N_ATTN
        AdaptiveAvgPool2d(1) → Flatten → Linear(64, n_classes)
    """

    def __init__(self, n_classes: int = N_CLASSES, n_bottleneck: int = N_ATTN) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(*[
            DenseBottleneckAttention(
                channels=BOTTLENECK_CH, attn_channels=ATTN_CH, heads=ATTN_HEADS,
            )
            for _ in range(n_bottleneck)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.bottleneck(self.enc(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Shared sparse backbone  (used by both Model 2 and Model 3)
# ─────────────────────────────────────────────────────────────────────────────

class _SparseBackbone(nn.Module):
    """
    Sparse encoder + sparse self-attention bottleneck.  Operates on Voxels throughout.

    Architecture (500×500 sparse input):
        SparseConv2d(1→32,  k=3, s=1) + BN1d + ReLU  →  500×500 sparse
        SparseConv2d(32→64, k=2, s=2) + BN1d + ReLU  →  250×250 sparse
        SparseConv2d(64→64, k=2, s=2) + BN1d + ReLU  →  125×125 sparse
        BottleneckSparseAttention2D(channels=64, attn_channels=128) × N_ATTN

    Channel widths and attention params match MinkUNetSparseAttention.
    """

    def __init__(self, n_attn: int = N_ATTN) -> None:
        super().__init__()
        self.enc0 = SparseSequential(
            SparseConv2d(1,  32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            SparseReLU(inplace=True),
        )
        self.enc1 = SparseSequential(
            SparseConv2d(32, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            SparseReLU(inplace=True),
        )
        self.enc2 = SparseSequential(
            SparseConv2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            SparseReLU(inplace=True),
        )
        # BottleneckSparseAttention2D prints one line per layer at init time.
        self.attn_layers = nn.ModuleList([
            BottleneckSparseAttention2D(
                channels       = BOTTLENECK_CH,
                attn_channels  = ATTN_CH,
                heads          = ATTN_HEADS,
                encoding       = True,
                encoding_range = ENCODING_RANGE,
                flash          = True,
            )
            for _ in range(n_attn)
        ])

    def forward(self, xs: Voxels) -> Voxels:
        xs = self.enc0(xs)
        xs = self.enc1(xs)
        xs = self.enc2(xs)
        for attn in self.attn_layers:
            xs = attn(xs)
        return xs


# ─────────────────────────────────────────────────────────────────────────────
# Global average pooling over sparse features  (no to_dense needed)
# ─────────────────────────────────────────────────────────────────────────────

def sparse_global_avg_pool(vox: Voxels) -> Tensor:
    """
    Batch-wise mean pool over active voxels without materialising a dense tensor.

    Input:  Voxels  with feature_tensor [N_total, C]  and offsets [B+1]  (CPU).
    Output: dense   [B, C]  on the same device as feature_tensor.
    """
    feats   = vox.feature_tensor                              # [N_total, C]  GPU
    offsets = vox.offsets                                     # [B+1]         CPU
    B       = len(offsets) - 1
    dev     = feats.device
    counts  = (offsets[1:] - offsets[:-1]).long()             # [B]  CPU

    batch_idx = torch.repeat_interleave(
        torch.arange(B, device=dev),
        counts.to(dev),
    )                                                         # [N_total]  GPU

    pooled = torch.zeros(B, feats.shape[1], device=dev, dtype=feats.dtype)
    pooled.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(feats), feats)
    pooled = pooled / counts.to(dev).float().unsqueeze(1).clamp(min=1)
    return pooled                                             # [B, C]


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 – Sparse CNN, Tensor I/O
# ─────────────────────────────────────────────────────────────────────────────

class SparseCNN_TensorIO(nn.Module):
    """
    Sparse encoder + attention bottleneck.  Interface: Tensor → Tensor.
    Calls Voxels.from_dense at the start of forward — that cost IS included
    in every memory measurement for this model.
    """

    def __init__(self, n_classes: int = N_CLASSES, n_attn: int = N_ATTN) -> None:
        super().__init__()
        self.backbone = _SparseBackbone(n_attn)
        self.head = nn.Linear(BOTTLENECK_CH, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        xs = Voxels.from_dense(x)           # dense [B,1,500,500] → Voxels
        xs = self.backbone(xs)
        return self.head(sparse_global_avg_pool(xs))   # [B, n_classes]


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 – Sparse CNN, Voxels I/O
# ─────────────────────────────────────────────────────────────────────────────

class SparseCNN_VoxelIO(nn.Module):
    """
    Sparse encoder + attention bottleneck.  Interface: Voxels → Tensor.
    The caller pre-converts dense input to Voxels (e.g. APASparseDataset).
    Voxels.from_dense cost is NOT inside forward and NOT counted in measurements.
    """

    def __init__(self, n_classes: int = N_CLASSES, n_attn: int = N_ATTN) -> None:
        super().__init__()
        self.backbone = _SparseBackbone(n_attn)
        self.head = nn.Linear(BOTTLENECK_CH, n_classes)

    def forward(self, xs: Voxels) -> Tensor:
        xs = self.backbone(xs)
        return self.head(sparse_global_avg_pool(xs))   # [B, n_classes]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────

def make_batch(
    batch_size: int,
    sparsity: float = 0.80,
    device: torch.device = DEVICE,
) -> tuple[Tensor, Tensor]:
    """
    Synthetic 500×500 batch: `sparsity` fraction of pixels are exactly 0,
    the rest are drawn from Uniform(0, 1].  Mimics DUNE detector images.
    """
    x = torch.rand(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    x *= (torch.rand_like(x) > sparsity).float()
    y = torch.randint(0, N_CLASSES, (batch_size,), device=device)
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_reset(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def measure_inference(model: nn.Module, inp, device: torch.device) -> float:
    """Peak GPU memory (MB) for one no_grad forward pass.  inp: Tensor or Voxels."""
    model.eval()
    _gpu_reset(device)
    with torch.no_grad():
        model(inp)
    return _peak_mb(device)


def measure_training(
    model: nn.Module,
    inp,
    y: Tensor,
    device: torch.device,
) -> float:
    """Peak GPU memory (MB) for one forward + backward + SGD step."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    _gpu_reset(device)
    opt.zero_grad(set_to_none=True)
    loss = crit(model(inp), y)
    loss.backward()
    opt.step()
    return _peak_mb(device)


# ─────────────────────────────────────────────────────────────────────────────
# Individual model tests
# ─────────────────────────────────────────────────────────────────────────────

@needs_gpu
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_dense_memory(batch_size: int) -> None:
    """Dense CNN: memory is positive; training >= inference."""
    x, y = make_batch(batch_size)
    m = DenseCNN().to(DEVICE)
    inf = measure_inference(m, x, DEVICE)
    tr  = measure_training (m, x, y, DEVICE)
    print(f"\n[Dense         | bs={batch_size}]  inf={inf:7.1f} MB   tr={tr:7.1f} MB")
    assert inf > 0 and tr >= inf


@needs_gpu
@needs_sparse
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_sparse_tensor_io_memory(batch_size: int) -> None:
    """SparseCNN Tensor I/O: memory is positive; training >= inference."""
    x, y = make_batch(batch_size)
    m = SparseCNN_TensorIO().to(DEVICE)
    inf = measure_inference(m, x, DEVICE)
    tr  = measure_training (m, x, y, DEVICE)
    print(f"\n[Sparse Tensor | bs={batch_size}]  inf={inf:7.1f} MB   tr={tr:7.1f} MB")
    assert inf > 0 and tr >= inf


@needs_gpu
@needs_sparse
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_sparse_voxel_io_memory(batch_size: int) -> None:
    """SparseCNN Voxels I/O: memory is positive; training >= inference."""
    x, y = make_batch(batch_size)
    # Pre-convert outside measurement window – mimics APASparseDataset.
    with torch.no_grad():
        xs = Voxels.from_dense(x)
    m = SparseCNN_VoxelIO().to(DEVICE)
    inf = measure_inference(m, xs, DEVICE)
    tr  = measure_training (m, xs, y, DEVICE)
    print(f"\n[Sparse Voxels | bs={batch_size}]  inf={inf:7.1f} MB   tr={tr:7.1f} MB")
    assert inf > 0 and tr >= inf


# ─────────────────────────────────────────────────────────────────────────────
# Head-to-head comparison
# ─────────────────────────────────────────────────────────────────────────────

@needs_gpu
@needs_sparse
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_memory_comparison(batch_size: int) -> None:
    """
    Head-to-head peak memory: Dense vs SparseCNN_TensorIO vs SparseCNN_VoxelIO.

    Each model is created fresh and deleted before the next to avoid
    cross-contamination of GPU allocations between measurements.

    Assertion: both sparse variants must use less than 5× the dense model's
    peak memory for 80%-sparse inputs.  At high sparsity sparse should be
    substantially cheaper than dense for the encoder stages.

    Note: SparseCNN_VoxelIO excludes Voxels.from_dense from its measurement
    window, representing a pipeline where sparse data arrives from a loader
    (APASparseDataset) rather than being converted from dense on-the-fly.
    """
    SPARSITY = 0.80
    x, y = make_batch(batch_size, sparsity=SPARSITY)

    # ── Dense ────────────────────────────────────────────────────────────────
    m = DenseCNN().to(DEVICE)
    d_inf = measure_inference(m, x, DEVICE)
    d_tr  = measure_training (m, x, y, DEVICE)
    del m; _gpu_reset(DEVICE)

    # ── Sparse – Tensor I/O  (from_dense IS inside forward) ─────────────────
    m = SparseCNN_TensorIO().to(DEVICE)
    st_inf = measure_inference(m, x, DEVICE)
    st_tr  = measure_training (m, x, y, DEVICE)
    del m; _gpu_reset(DEVICE)

    # ── Sparse – Voxels I/O  (from_dense is OUTSIDE measurement) ────────────
    with torch.no_grad():
        xs = Voxels.from_dense(x)           # pre-convert; cost excluded
    m = SparseCNN_VoxelIO().to(DEVICE)
    sv_inf = measure_inference(m, xs, DEVICE)
    sv_tr  = measure_training (m, xs, y, DEVICE)
    del m; _gpu_reset(DEVICE)

    W = 78
    print(f"\n{'='*W}")
    print(f"  Memory comparison  batch={batch_size}  sparsity={SPARSITY:.0%}"
          f"  attn_layers={N_ATTN}")
    print(f"{'─'*W}")
    print(f"  {'Model':<26} {'Inference':>10} MB   {'Training':>10} MB   {'vs Dense':>9}")
    print(f"{'─'*W}")
    print(f"  {'Dense (conv bottleneck)':<26} {d_inf:>10.1f}      {d_tr:>10.1f}")
    print(f"  {'Sparse (Tensor I/O)':<26} {st_inf:>10.1f}      {st_tr:>10.1f}"
          f"      {st_inf/d_inf:.2f}× / {st_tr/d_tr:.2f}×")
    print(f"  {'Sparse (Voxels I/O)*':<26} {sv_inf:>10.1f}      {sv_tr:>10.1f}"
          f"      {sv_inf/d_inf:.2f}× / {sv_tr/d_tr:.2f}×")
    print(f"{'─'*W}")
    print(f"  * Voxels I/O excludes Voxels.from_dense conversion cost")
    print(f"{'='*W}")

    assert st_inf / d_inf < 5.0, f"Sparse TensorIO inference >5× dense: {st_inf/d_inf:.2f}×"
    assert st_tr  / d_tr  < 5.0, f"Sparse TensorIO training  >5× dense: {st_tr/d_tr:.2f}×"
    assert sv_inf / d_inf < 5.0, f"Sparse VoxelIO  inference >5× dense: {sv_inf/d_inf:.2f}×"
    assert sv_tr  / d_tr  < 5.0, f"Sparse VoxelIO  training  >5× dense: {sv_tr/d_tr:.2f}×"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone smoke-run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SPARSITY    = 0.80
    BATCH_SIZES = [2, 4, 8]

    print(f"Device        : {DEVICE}")
    print(f"WarpConvNet   : {'available' if WARPCONVNET_AVAILABLE else 'NOT available'}")
    print(f"Image size    : {IMG_SIZE}×{IMG_SIZE}")
    print(f"Input sparsity: {SPARSITY:.0%}")
    print(f"Attn layers   : {N_ATTN}  (channels={BOTTLENECK_CH}, attn_ch={ATTN_CH})")

    if not torch.cuda.is_available():
        print("\nNo CUDA device — skipping memory measurements.")
    else:
        W = 82
        print(f"\n{'─'*W}")
        print(f"  {'bs':>3}  {'D-inf':>8}  {'D-tr':>8}"
              f"  {'ST-inf':>8}  {'ST-tr':>8}"
              f"  {'SV-inf':>8}  {'SV-tr':>8}")
        print(f"{'─'*W}")

        for bs in BATCH_SIZES:
            x, y = make_batch(bs, sparsity=SPARSITY)

            m     = DenseCNN().to(DEVICE)
            d_inf = measure_inference(m, x, DEVICE)
            d_tr  = measure_training (m, x, y, DEVICE)
            del m; _gpu_reset(DEVICE)

            if WARPCONVNET_AVAILABLE:
                m      = SparseCNN_TensorIO().to(DEVICE)
                st_inf = measure_inference(m, x, DEVICE)
                st_tr  = measure_training (m, x, y, DEVICE)
                del m; _gpu_reset(DEVICE)

                with torch.no_grad():
                    xs = Voxels.from_dense(x)
                m      = SparseCNN_VoxelIO().to(DEVICE)
                sv_inf = measure_inference(m, xs, DEVICE)
                sv_tr  = measure_training (m, xs, y, DEVICE)
                del m; _gpu_reset(DEVICE)

                print(f"  {bs:>3}  {d_inf:>7.1f}M  {d_tr:>7.1f}M"
                      f"  {st_inf:>7.1f}M  {st_tr:>7.1f}M"
                      f"  {sv_inf:>7.1f}M  {sv_tr:>7.1f}M")
            else:
                print(f"  {bs:>3}  {d_inf:>7.1f}M  {d_tr:>7.1f}M"
                      f"         —         —         —         —")

        print(f"{'─'*W}")
        print("D=Dense  ST=Sparse(TensorIO)  SV=Sparse(VoxelIO,from_dense excluded)")
