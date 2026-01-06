
import torch.nn as nn
from typing import Literal, Optional
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.encodings import SinusoidalEncoding
from warpconvnet.geometry.features.ops.convert import cat_to_pad_tensor

from warpconvnet.nn.modules.attention import offset_to_mask
from warpconvnet.nn.modules.attention import Attention, ToSpatialFeatures

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
        # NEW: how wide should pos_enc be?
        # - "per_head" -> out_channels // num_heads  (for non-flash: add to Q/K)
        # - "full"     -> out_channels              (for flash: add to x)
        pos_enc_mode: Literal["per_head", "full"] = "per_head",
    ):
        super().__init__()
        self.out_type = out_type
        self.use_encoding = use_encoding

        if use_encoding:
            assert num_encoding_channels is not None, "num_encoding_channels must be provided"
            assert encoding_range is not None, "encoding_range must be provided"
            assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

            if pos_enc_mode == "per_head":
                pos_out = out_channels // num_heads
            elif pos_enc_mode == "full":
                pos_out = out_channels
            else:
                raise ValueError(f"Unknown pos_enc_mode={pos_enc_mode}")

            in_feats = num_encoding_channels * num_spatial_features + (
                num_spatial_features if concat_input else 0
            )

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
    and chooses positional encoding width to be compatible with the injection site.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_encoding_channels: int = 32,
        encoding_range: float = 1.0,
        use_encoding: bool = False,
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

        # Decide how wide pos_enc should be:
        # - flash path adds pos_enc to x -> needs full C
        # - non-flash adds pos_enc to q/k per head -> head_dim
        pos_enc_mode = "full" if (enable_flash and use_encoding) else "per_head"

        self.to_attn = ToAttentionSmart(
            out_channels=dim,
            use_encoding=use_encoding,
            num_encoding_channels=num_encoding_channels,
            encoding_range=encoding_range,
            num_heads=num_heads,
            concat_input=True,
            num_spatial_features=2,     # <-- the whole point: 2D
            out_type="cat",
            pos_enc_mode=pos_enc_mode,  # <-- new: resolves flash/encoding mismatch
        )
        self.from_attn = ToSpatialFeatures()

    def forward(self, x: Geometry) -> Geometry:
        features, pos_enc, mask, num_points = self.to_attn(x)

        # Note: with flash + encoding, pos_enc is [B, N, C] so x + pos_enc works.
        # With non-flash + encoding, pos_enc is [B, N, head_dim] so q/k addition works.
        y = super().forward(features, pos_enc, mask, num_points)

        return self.from_attn(y, x)
