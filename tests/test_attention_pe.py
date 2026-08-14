"""
test_attention_pe.py
────────────────────
Positional-encoding contract for the 2-D sparse attention bottleneck.

The flash and dense paths run different kernels, but they must compute the same
operator: the encoding is head-dim wide and enters q and k after the qkv projection on
both. These checks pin that down, plus the two properties that follow from it —

  • the flash path never builds the padded tensors or the quadratic [B, 1, N, N] mask
    that only the dense path reads,
  • the encoding actually reaches the attention logits, i.e. moving a point changes
    its output.

They also check that `encoding_range` survives the trip from a config to the
SinusoidalEncoding, which a leaf-only signature check does not manage for backbones
that forward through `**kw`.

Needs a GPU and the flash_attn package. Plain script, no pytest — the cluster venv has
none.

Run:  python -u tests/test_attention_pe.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback

import torch

# Make project root importable so "from models... import ..." works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

from models import BACKBONE_REGISTRY, backbone_kwargs
from models import attention2D
from models.attention2D import SpatialFeatureAttention2D

# Bottleneck geometry: attn_channels=128, heads=4 (models/minkunet_attention.py).
DIM = 128
HEADS = 4
ENC_CHANNELS = 32
ENC_RANGE = 320.0

# Bottleneck coordinate span at ÷4 is x∈[0,239], y∈[0,356]; 240 keeps the test inside it.
COORD_SPAN = 240

# flash runs in fp16 while the dense path stays fp32, so the two agree only to the
# half-precision floor. Same tolerance warpconvnet uses for its own flash comparisons.
ATOL = 2e-3
RTOL = 1e-3


def _make_voxels(device, seed=0, n_per_event=(97, 53, 131)):
    """A ragged 2-D batch shaped like what reaches the bottleneck."""
    g = torch.Generator().manual_seed(seed)
    all_coords, all_feats = [], []
    for n in n_per_event:
        all_coords.append(torch.randint(0, COORD_SPAN, (n, 2), generator=g))
        all_feats.append(torch.randn(n, DIM, generator=g))

    counts = torch.tensor([c.shape[0] for c in all_coords], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    return _assemble(torch.cat(all_coords), torch.cat(all_feats), offsets, device)


def _assemble(coords, feats, offsets, device):
    return Voxels(
        batched_coordinates=IntCoords(coords.to(device), offsets=offsets),
        batched_features=CatFeatures(feats.to(device), offsets=offsets),
        offsets=offsets,
    )


def _reposition(x, seed=1):
    """The same features at shuffled coordinates, batch structure untouched.

    A permutation rather than a translation: it destroys the spatial arrangement instead
    of sliding it, so the output has to move for any encoding that reaches the logits.
    """
    coords = x.coordinate_tensor
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(coords.shape[0], generator=g).to(coords.device)
    return _assemble(coords[perm], x.feature_tensor, x.offsets, coords.device)


def _make_attn(device, enable_flash, use_encoding=True, seed=0):
    torch.manual_seed(seed)
    return (
        SpatialFeatureAttention2D(
            dim=DIM,
            num_heads=HEADS,
            num_encoding_channels=ENC_CHANNELS,
            encoding_range=ENC_RANGE,
            use_encoding=use_encoding,
            enable_flash=enable_flash,
        )
        .to(device)
        .eval()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_encoding_width_is_head_dim(device):
    """One width serves both paths, because both inject after the qkv projection.

    A full-C encoding here means the flash path is back to adding position to the input
    features, where it also lands on v and reaches the logits only through weights
    trained for content.
    """
    attn = _make_attn(device, enable_flash=True)
    width = attn.to_attn.encoding[1].out_features
    print(f"  encoding width = {width} (head_dim = {DIM // HEADS})")
    assert width == DIM // HEADS, f"encoding is {width} wide, expected {DIM // HEADS}"


def _rescale_encoding(attn, x, target_rms):
    """Bring the encoding down to a magnitude comparable to the content signal.

    At random init the encoding Linear sees raw coordinates (concat_input appends them
    unnormalised, 0..239), so it emits an encoding ~140x larger than q. Feeding that to
    flash puts q and k where an fp16 ulp is 0.03 against a content rms of 0.12, and the
    kernel quantises the content away — a precision effect that would mask whatever the
    two paths actually compute. Scaling to a trained-like magnitude isolates the operator
    from the arithmetic.
    """
    with torch.no_grad():
        pe = attn.to_attn.forward_flash(x)
        factor = target_rms / pe.pow(2).mean().sqrt().item()
        attn.to_attn.encoding[1].weight.mul_(factor)
        attn.to_attn.encoding[1].bias.mul_(factor)


def check_flash_matches_dense(device, use_encoding):
    """The two paths are the same operator up to fp16 and reduction order.

    This is the case warpconvnet's own consistency test cannot cover: it asserts
    flash/dense equivalence only with the positional encoding disabled.
    """
    x = _make_voxels(device)

    flash = _make_attn(device, enable_flash=True, use_encoding=use_encoding)
    if use_encoding:
        _rescale_encoding(flash, x, target_rms=0.1)
    dense = _make_attn(device, enable_flash=False, use_encoding=use_encoding)
    dense.load_state_dict(flash.state_dict())
    with torch.no_grad():
        out_flash = flash(x).feature_tensor
        out_dense = dense(x).feature_tensor

    assert out_flash.shape == out_dense.shape
    delta = (out_flash - out_dense).abs().max().item()
    print(f"  use_encoding={use_encoding}: max |flash - dense| = {delta:.3e}")
    assert torch.allclose(out_flash, out_dense, atol=ATOL, rtol=RTOL), (
        f"flash and dense disagree by {delta:.3e}, above the fp16 floor"
    )


def check_encoding_reaches_the_logits(device, enable_flash):
    """Moving the points must move the output — on both paths, by a similar amount.

    This is the collapse detector for the encoding: an injection site that leaves the
    encoding effectively inert still produces plausible features and a plausible loss,
    so only position-sensitivity distinguishes it.
    """
    attn = _make_attn(device, enable_flash=enable_flash, use_encoding=True)

    x = _make_voxels(device)
    y = _reposition(x)
    with torch.no_grad():
        a = attn(x).feature_tensor.float()
        b = attn(y).feature_tensor.float()

    rel = ((a - b).abs().mean() / a.abs().mean()).item()
    print(f"  enable_flash={enable_flash}: relative change from repositioning = {rel:.4f}")
    assert rel > 1e-2, (
        f"output barely moved ({rel:.2e}) when the coordinates did — "
        f"the positional encoding is not reaching the attention logits"
    )


def check_flash_skips_the_padded_path(device):
    """The varlen kernel consumes the concatenated features directly.

    Building the padded copies and the [B, 1, N, N] mask for it would reintroduce
    exactly the quadratic memory the flash path exists to avoid.
    """
    attn = _make_attn(device, enable_flash=True)
    x = _make_voxels(device)

    calls = []
    original = attention2D.offset_to_mask
    attention2D.offset_to_mask = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        with torch.no_grad():
            attn(x)
    finally:
        attention2D.offset_to_mask = original

    print(f"  offset_to_mask calls on the flash path: {len(calls)}")
    assert not calls, "flash path built the dense mask"


def check_encoding_range_reaches_the_encoder(device):
    """A configured encoding_range must survive to SinusoidalEncoding's freqs table.

    Backbones that forward through `def __init__(self, **kw)` hide their base class's
    parameters from a leaf-only signature check, which silently drops the flag and
    leaves the hardcoded default in place.
    """
    for name in ("attn_default", "attn_mae", "attn_noflash"):
        cls = BACKBONE_REGISTRY[name]
        model = cls(**backbone_kwargs(cls, encoding_range=ENC_RANGE, encoding_dim=ENC_CHANNELS))
        # freqs[0] == 2*pi / data_range, so the range is recoverable from the buffer.
        freqs = model.bottleneck.attn.to_attn.encoding[0].freqs
        data_range = 2 * math.pi / float(freqs[0])
        print(f"  {name}: data_range = {data_range:.1f}")
        assert abs(data_range - ENC_RANGE) < 1e-3, (
            f"{name} built its encoding at data_range={data_range:.1f}, not {ENC_RANGE}"
        )


def check_unsupported_flag_raises(device):
    """A backbone that cannot honour a flag says so instead of ignoring it."""
    try:
        backbone_kwargs(BACKBONE_REGISTRY["base"], encoding_range=ENC_RANGE)
    except ValueError as e:
        print(f"  raised as expected: {e}")
        return
    raise AssertionError("an unsupported flag was accepted silently")


CHECKS = [
    ("encoding_width_is_head_dim", check_encoding_width_is_head_dim, {}),
    ("flash_matches_dense[enc=on]", check_flash_matches_dense, {"use_encoding": True}),
    ("flash_matches_dense[enc=off]", check_flash_matches_dense, {"use_encoding": False}),
    ("encoding_reaches_logits[flash]", check_encoding_reaches_the_logits, {"enable_flash": True}),
    ("encoding_reaches_logits[dense]", check_encoding_reaches_the_logits, {"enable_flash": False}),
    ("flash_skips_padded_path", check_flash_skips_the_padded_path, {}),
    ("encoding_range_reaches_encoder", check_encoding_range_reaches_the_encoder, {}),
    ("unsupported_flag_raises", check_unsupported_flag_raises, {}),
]


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs a GPU")
        return 0
    if attention2D.flash_attn is None:
        print("SKIP: needs the flash_attn package")
        return 0

    device = torch.device("cuda")
    failures = []
    for name, fn, kw in CHECKS:
        print(f"\n[{name}]")
        try:
            fn(device, **kw)
            print(f"  PASS")
        except Exception:
            traceback.print_exc()
            failures.append(name)
            print(f"  FAIL")

    print("\n" + "=" * 60)
    print(f"{len(CHECKS) - len(failures)}/{len(CHECKS)} checks passed")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
