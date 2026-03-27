"""DINO-style MLP projection head for per-voxel feature transformation."""

import torch.nn as nn
import torch.nn.functional as F

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.nn.modules.mlp import Linear
from warpconvnet.nn.modules.activations import GELU
from warpconvnet.nn.modules.sequential import Sequential


class L2Normalize(nn.Module):
    """L2-normalises Voxels features along the channel dimension."""
    def forward(self, x: Voxels) -> Voxels:
        feats = F.normalize(x.feature_tensor, dim=-1)
        return Voxels(
            batched_coordinates=x.batched_coordinates,
            batched_features=CatFeatures(feats, x.offsets),
            offsets=x.offsets,
        )


class DINOProjectionHead(nn.Module):
    """
    Per-voxel DINO projection head.

    Architecture (following DINO paper):
        Linear → GELU            ⎫
        Linear → GELU            ⎬  n_layers MLP (GELU on all but the last layer)
        ...                      ⎟
        Linear (no activation)   ⎭
        L2 normalise
        Linear (no bias)         ←  final FC layer

    Operates directly on the Voxels feature tensor via WarpConvNet's native
    Linear module, so no dense materialisation is needed.

    Args:
        in_dim:     backbone output channels (e.g. 64)
        hidden_dim: inner MLP width (e.g. 256; DINO paper uses 2048)
        out_dim:    output projection dimension (e.g. 256)
        n_layers:   total MLP layers before the final FC (default 4, as in DINO)
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 4):
        super().__init__()

        mlp_layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden_dim
            mlp_layers.append(Linear(d_in, hidden_dim, bias=True))
            if i < n_layers - 1:
                mlp_layers.append(GELU())
        self.mlp       = Sequential(*mlp_layers)
        self.normalize = L2Normalize()
        self.last      = Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: Voxels) -> Voxels:
        return self.last(self.normalize(self.mlp(x)))
