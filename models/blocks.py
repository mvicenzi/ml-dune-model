from torch import Tensor
import torch.nn as nn              # Neural network base classes

from warpconvnet.geometry.base.geometry import Geometry                 # Voxels derives from this
from warpconvnet.geometry.types.voxels import Voxels                    # Sparse voxel data structure
from warpconvnet.nn.modules.sparse_conv import SparseConv2d             # 2D sparse convolution
from warpconvnet.nn.modules.sequential import Sequential                # Ordered list of sparse modules
from warpconvnet.nn.modules.activations import ReLU                     # Sparse-aware ReLU activation
from warpconvnet.nn.modules.normalizations import LayerNorm             # layer normalization 
from warpconvnet.nn.modules.activations import GELU                     # Sparse-aware ReLU activation

#from warpconvnet.nn.modules.attention import PatchAttention, SpatialFeatureAttention  # Sparse attention
from .attention2D import SpatialFeatureAttention2D

# ---------------------------------------------------------------------------
# Building blocks: small modular components used to construct the main model
# ---------------------------------------------------------------------------

class ConvBlock2D(Sequential):
    """
    Sparse 2D convolutional block based on WarpConvNet functions.
    Composition: 
        SparseConv2d -> BatchNorm1d -> ReLU
    - this is the main conv layer in base resnet block  
    - note: relu activation needs to be disabled in some cases!
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=False, relu=True):
        super().__init__(
            SparseConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=bias),
            nn.BatchNorm1d(out_ch),
            ReLU(inplace=True) if relu is True else nn.Identity(),
        )  


# ---------------------------------------------------------------------------

class ConvTrBlock2D(nn.Module):
    """
    Sparse transposed convolution (upsampling block).
    Used in the decoder to increase spatial resolution.
    - 'transposed=True' performs the reverse of a convolution (learned upsampling).
    - out_spatial_sparsity defines which coordinates the upsampled result should align to.
    - relu activation always active
    """
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, bias=False):
        super().__init__()
        self.deconv = SparseConv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride,
            transposed=True, bias=bias
        )
        self.norm_act = Sequential(
            nn.BatchNorm1d(out_ch),
            ReLU(inplace=True),
        )

    def forward(self, x_sparse: Voxels, out_spatial_sparsity: Voxels) -> Voxels:
        # Perform sparse transposed convolution guided by the skip tensor geometry
        y = self.deconv(x_sparse, out_spatial_sparsity)
        return self.norm_act(y)

# ---------------------------------------------------------------------------

class ResidualSparseBlock2D(nn.Module):
    """
    Sparse residual block (the core computation unit of the encoder/decoder).
    This is the ResNet "BasicBlock" from mink_unet.py:
        Conv → BN → ReLU
        Conv → BN
        Add residual
        ReLU 
    - Sparse convolution layers based on ConvBlock2D 
    - "stride" parameter always at 1: size downsampling is external!
    - 'relu=False' makes the second layer without activation
    - Skip connection: adds input ('identity') to output ('out')
    - Preserves sparse coordinate structure (no densification)
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        # if you downsample or add feature dimensions to pixels
        # input must be adapted to properly be added to output in residual skip connection
        # we do this with a simple convolution block
        self.downsample = None
        if stride!= 1 or in_ch != out_ch:
            self.downsample = ConvBlock2D(in_ch, out_ch, kernel_size=1, stride=stride, relu=False)

        # First convolution: SparseConv2d + BatchNorm1d + ReLU
        self.conv1 = ConvBlock2D(in_ch, out_ch, kernel_size=3, stride=stride)

        # Second convolution: SparseConv2d + BatchNorm1d
        self.conv2 = ConvBlock2D(out_ch, out_ch, kernel_size=3, stride=1, relu=False)

        # Final activation (after skip addition)
        self.act = ReLU(inplace=True)


    def forward(self, x_sparse: Voxels) -> Voxels:

        identity = x_sparse 
        if self.downsample is not None:
            identity = self.downsample(x_sparse)

        # Forward through two sparse conv layers
        out = self.conv1(x_sparse)
        out = self.conv2(out)

        out += identity  # Residual skip addition (still sparse)
        out = self.act(out)

        return out
    
# ---------------------------------------------------------------------------
# Bottleneck Attention block (dense)
# ---------------------------------------------------------------------------

class BottleneckDenseAttention2D(nn.Module):
    """
    Dense multi-head self-attention at the bottleneck.
    Operates on small 7×7 dense maps → global receptive field.
    Flow:
        x -> LayerNorm -> MHSA -> +skip
        -> LayerNorm -> MLP -> +skip
    """
    def __init__(self, channels: int, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()

        # Pre-attention layernorm
        self.norm1 = nn.LayerNorm(channels)

        # Dense Multi-Head Self-Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True,
            dropout=dropout,
        )

        # Pre-MLP layernorm
        self.norm2 = nn.LayerNorm(channels)

        # 1×1 Conv MLP
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        # ---- PRE-ATTENTION NORMALIZATION ----
        tokens = x.flatten(2).transpose(1, 2)        # [B, HW, C]
        tokens = self.norm1(tokens)

        # ---- MHSA ----
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = tokens + attn_out                   # Skip 1

        # ---- RESTRUCTURE BACK TO MAP ----
        x = tokens.transpose(1, 2).reshape(b, c, h, w)

        # ---- PRE-MLP NORMALIZATION ----
        x_norm = x.flatten(2).transpose(1, 2)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.transpose(1, 2).reshape(b, c, h, w)

        # ---- MLP + SKIP ----
        x = x + self.mlp(x_norm)

        return x



# ---------------------------------------------------------------------------
# Bottleneck Attention block (sparse)
# ---------------------------------------------------------------------------

class BottleneckSparseAttention2D(nn.Module):
    """
    Sparse transformer-style bottleneck using WarpConvNet's PatchAttention.
    Operates directly on Geometry (e.g. Voxels), so we never densify.
    Flow: norm -> PathAttention -> residual -> MLP -> residual.
    """
    def __init__(self, channels: int, attn_channels: int, heads: int = 4, patch_size: int = 64, 
                 attn_drop: float = 0.0, proj_drop: float = 0.0, mlp_ratio: float = 2.0,
                 encoding: bool = True, flash: bool = True ):
        super().__init__()

        print(f"[BottleneckSparseAttention2D] encoding={encoding}, flash={flash}")

        # project before attention (sparse 1x1 conv)
        self.pre_proj = SparseConv2d(channels, attn_channels, kernel_size=1)
        
        # with not batchNorm?
        self.norm1 = LayerNorm(attn_channels)
        self.norm2 = LayerNorm(attn_channels)

        # Sparse attention over features (positional encoding is 2D-safe)
        # this uses custom modules in attention2D.py
        self.attn = SpatialFeatureAttention2D(dim=attn_channels, 
                                              num_heads=heads, 
                                              qkv_bias=True, 
                                              qk_scale=None,
                                              attn_drop=attn_drop, 
                                              proj_drop=proj_drop,
                                              num_encoding_channels=32, 
                                              encoding_range=1.0,
                                              use_encoding=encoding,
                                              enable_flash=flash, 
                                              use_batched_qkv=True)

        hidden = int(attn_channels * mlp_ratio)
        # A small MLP (2-layer 1x1 conv) adds non-linear mixing after attention
        self.mlp = Sequential(
            SparseConv2d(attn_channels, hidden, kernel_size=1),
            GELU(),
            SparseConv2d(hidden, attn_channels, kernel_size=1)
        )

        # project back after attention (sparse 1x1 conv)
        self.post_proj = SparseConv2d(attn_channels, channels, kernel_size=1)

    def forward(self, x: Geometry) -> Geometry:

        x2 = self.pre_proj(x)

        # Geometry in/out 
        x2_norm1 = self.norm1(x2)
        h = self.attn(x2_norm1)
        x2 = x2_norm1 + h 

        # MLP sub-block
        x2_norm2 = self.norm2(x2)
        h2 = self.mlp(x2_norm2) 
        x2 = x2_norm2 + h2 

        x_out = self.post_proj(x2)
        return x_out
