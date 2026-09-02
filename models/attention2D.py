import torch
import torch.nn as nn
from typing import Literal, Optional
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.encodings import SinusoidalEncoding
from warpconvnet.geometry.features.ops.convert import cat_to_pad_tensor

from warpconvnet.nn.modules.attention import offset_to_mask, zero_out_points
from warpconvnet.nn.modules.attention import Attention, ToSpatialFeatures

try:
    import flash_attn
except ImportError:
    flash_attn = None

### This is based on what offered by WarpConvNet but adjusting
### the final tensor shape of the spatial encoding
### to support both standard and flash attention mechanism.
class ToAttentionSmart(BaseSpatialModule):
    def __init__(
        self,
        out_channels: int,
        use_encoding: bool = False,
        num_encoding_channels: Optional[int] = None,
        encoding_range: Optional[float] = None,
        num_heads: int = 1,
        concat_input: bool = True,
        num_spatial_features: int = 3,
        out_type: Literal["nested", "cat"] = "cat",
    ):
        super().__init__()
        self.out_type = out_type
        self.use_encoding = use_encoding

        if use_encoding:
            assert num_encoding_channels is not None, "num_encoding_channels must be provided"
            assert encoding_range is not None, "encoding_range must be provided"
            assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

            # The encoding is injected into q and k after the qkv projection, where a
            # head-dim-wide tensor broadcasts across heads. Both attention paths do this,
            # so a single width serves them.
            pos_out = out_channels // num_heads

            in_feats = num_encoding_channels * num_spatial_features + (
                num_spatial_features if concat_input else 0
            )

            print("SinusouidalEncoding settings:")
            print(f"  num_channels={num_encoding_channels}")
            print(f"  data_range={encoding_range}")
            print(f"  concat_input={concat_input}")

            self.encoding = nn.Sequential(
                SinusoidalEncoding(
                    num_channels=num_encoding_channels,
                    data_range=encoding_range,
                    concat_input=concat_input,
                ),
                nn.Linear(in_feats, pos_out),
            )

    def forward(self, x: Geometry):
        if self.out_type == "nested":
            features = x.nested_features
            coordinates = x.nested_coordinates
            # NOTE: if nested path is used, you'll need offsets for padding/mask;
            # leaving as-is since your current usage appears out_type="cat".
        else:
            features_cat, offsets = x.features, x.offsets
            features = cat_to_pad_tensor(features_cat, offsets)          # [B, N, C]
            coordinates = x.coordinate_tensor                            # [M, D]
            num_points = offsets.diff()                                  # [B]

        if self.use_encoding:
            pos_enc_cat = self.encoding(coordinates)                     # [M, pos_out]
            pos_enc = cat_to_pad_tensor(pos_enc_cat, offsets)            # [B, N, pos_out]
        else:
            pos_enc = None

        mask = offset_to_mask(features, offsets, features.shape[1])      # [B, 1, N, N] (bool)
        return features, pos_enc, mask, num_points

    def forward_flash(self, x: Geometry):
        """Positional encoding alone, on the concatenated [M, C] layout.

        The varlen flash kernel consumes the concatenated features directly, so the
        padded features, the padded encoding and the quadratic [B, 1, N, N] mask that
        forward() builds for the dense path are never read on that branch.
        """
        if not self.use_encoding:
            return None
        return self.encoding(x.coordinate_tensor)                        # [M, pos_out]


### This is based on what offered by WarpConvNet but adjusting
### the expected spatial dimensions to 2D. It also uses the
### ToAttentionSmart() block to support both standard and flash attention mechanism
### in case spatial encoding is enabled.
class SpatialFeatureAttention2D(Attention):
    """
    SpatialFeatureAttention for 2D coordinates (x, y).
    Supports:
      - flash ON/OFF
      - encoding ON/OFF
    Both paths inject the positional encoding the same way: head-dim wide, into q and
    k after the qkv projection. They differ only in the attention kernel and in whether
    the features are padded, so they compute the same operator.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_encoding_channels: int = 32,
        encoding_range: float = 1.0,
        use_encoding: bool = True,
        enable_flash: bool = True,
        use_batched_qkv: bool = True,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_flash=enable_flash,
            use_batched_qkv=use_batched_qkv,
        )

        self.to_attn = ToAttentionSmart(
            out_channels=dim,
            use_encoding=use_encoding,
            num_encoding_channels=num_encoding_channels,
            encoding_range=encoding_range,
            num_heads=num_heads,
            concat_input=True,
            num_spatial_features=2,     # <-- the whole point: 2D
            out_type="cat",
        )
        self.from_attn = ToSpatialFeatures()

    def forward(self, x: Geometry) -> Geometry:

        if not self.enable_flash:
            # extract padded tensors from the original concatenated tensor
            # this path needs them, plus an appropriate mask
            features, pos_enc, mask, num_points = self.to_attn(x)

            B, N, C = features.shape
            qkv = self.qkv(features).reshape(B, N, 3, C)

            # Reshape to [B, N, 3, num_heads, head_dim]
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)

            # Apply positional encoding to the query and key (non-flash path)
            if pos_enc is not None:
                q = q + pos_enc.unsqueeze(1)
                k = k + pos_enc.unsqueeze(1)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                assert mask.device == attn.device
                attn_bias = torch.zeros(mask.shape, dtype = attn.dtype, device = attn.device)
                # dtype-aware sentinel: -1e9 is outside fp16 range, so a literal
                # cannot be written under half precision
                attn_bias.masked_fill_(mask.logical_not(), torch.finfo(attn.dtype).min)
                attn = attn + attn_bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            y = attn @ v
            y = y.transpose(1, 2).reshape(B, N, C)

            y = self.proj(y)
            y = self.proj_drop(y)

            if num_points is not None:
                y = zero_out_points(y, num_points)

            return self.from_attn(y, x)
        # use flash_attn on the concatenated tensor directly, no need to use padded and convert it back
        else:
            pos_enc_cat = self.to_attn.forward_flash(x)                   # [M, head_dim]
            feats, offsets = x.features, x.offsets
            M, C = feats.shape[:2]

            qkv = self.qkv(feats).reshape(M, 3, self.num_heads, C // self.num_heads)
            # Inject the encoding per-head into q and k only, never v, so it acts as a
            # pure attention bias instead of leaking position into aggregated content.
            # Done before the fp16 cast below so the encoding keeps full precision.
            if pos_enc_cat is not None:
                pe = pos_enc_cat.unsqueeze(1)                             # [M, 1, head_dim]
                qkv = torch.stack([qkv[:, 0] + pe, qkv[:, 1] + pe, qkv[:, 2]], dim=1)

            if qkv.dtype not in [torch.float16, torch.bfloat16]:
                qkv = qkv.to(torch.float16)
            # Warning: When the loss is NaN, this module will fail during backward with
            # index out of bounds error.
            # e.g. /pytorch/aten/src/ATen/native/cuda/ScatterGatherKernel.cu:144: operator(): block: [192,0,0], thread: [32,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "
            # https://discuss.pytorch.org/t/scattergatherkernel-cu-assertion-idx-dim-0-idx-dim-index-size-index-out-of-bounds/195356
            max_seqlen = int(offsets.diff().max())
            attn_offsets = offsets.to(device=qkv.device, dtype=torch.int32)
            out_feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv,
                attn_offsets,
                max_seqlen=max_seqlen,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                softmax_scale=self.scale,
            )
            out_feat = out_feat.reshape(M, C).to(feats.dtype)

            out_feat = self.proj(out_feat)
            out_feat = self.proj_drop(out_feat)

            return x.replace(batched_features=out_feat.to(feats.dtype))
