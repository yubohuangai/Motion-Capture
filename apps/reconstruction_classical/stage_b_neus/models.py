"""NeuS-style SDF + color networks (trained from scratch, per scene).

We deliberately keep the implementation small and self-contained: no data
priors, no pretrained weights, no learned hash-grids. The MLP is randomly
initialised and optimised only on the observed multi-view images.

Reference
---------
Wang et al. (NeurIPS '21), "NeuS: Learning Neural Implicit Surfaces by
Volume Rendering for Multi-view Reconstruction" (https://arxiv.org/abs/2106.10689).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Standard NeRF/NeuS positional encoding with log-spaced bands."""

    def __init__(self, n_bands: int, include_input: bool = True) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(n_bands, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)
        self.out_dim_multiplier = (1 if include_input else 0) + 2 * n_bands

    def out_dim(self, in_dim: int) -> int:
        return in_dim * self.out_dim_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_bands == 0:
            return x if self.include_input else x.new_zeros((*x.shape[:-1], 0))
        xb = x[..., None, :] * self.freqs[:, None]        # (..., L, D)
        sin = torch.sin(xb); cos = torch.cos(xb)
        enc = torch.stack([sin, cos], dim=-1).flatten(-3)  # (..., L*2*D)
        if self.include_input:
            return torch.cat([x, enc], dim=-1)
        return enc


# ---------------------------------------------------------------------------
# SDF network
# ---------------------------------------------------------------------------


class SDFNetwork(nn.Module):
    """MLP that maps 3D position -> (sdf, feature vector).

    Uses the geometric initialisation from Atzmon & Lipman (SIGGRAPH '20) so
    that the initial SDF is approximately a sphere. This is a strong inductive
    bias that dramatically stabilises optimisation.
    """

    def __init__(self,
                 d_in: int = 3,
                 d_hidden: int = 256,
                 n_layers: int = 8,
                 skip_in: Tuple[int, ...] = (4,),
                 n_pos_bands: int = 6,
                 feature_dim: int = 256,
                 geometric_init_bias: float = 0.5,
                 scale: float = 1.0) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(n_pos_bands, include_input=True)
        self.scale = scale
        dims = [self.pos_enc.out_dim(d_in)] + [d_hidden] * (n_layers - 1) + [1 + feature_dim]
        self.num_layers = len(dims)
        self.skip_in = set(skip_in)

        layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if (i + 1) in self.skip_in:
                out_dim -= dims[0]
            lin = nn.Linear(in_dim, out_dim)
            self._geometric_init(lin, i, out_dim, bias=geometric_init_bias,
                                 is_last=(i == self.num_layers - 2),
                                 uses_skip=(i in self.skip_in))
            layers.append(lin)
        self.layers = layers
        self.activation = nn.Softplus(beta=100)

    @staticmethod
    def _geometric_init(lin: nn.Linear, idx: int, out_dim: int,
                        bias: float, is_last: bool, uses_skip: bool) -> None:
        """Initialise so that the initial SDF is a sphere of radius ``bias``."""
        with torch.no_grad():
            if is_last:
                nn.init.normal_(lin.weight, mean=math.sqrt(math.pi) / math.sqrt(lin.in_features), std=1e-4)
                nn.init.constant_(lin.bias, -bias)
                # keep the feature-vector outputs at small init
                if lin.weight.shape[0] > 1:
                    lin.weight[1:].data.normal_(0.0, 1e-4)
                    lin.bias[1:].data.zero_()
            elif idx == 0:
                nn.init.normal_(lin.weight, 0.0, math.sqrt(2.0 / out_dim))
                nn.init.constant_(lin.bias, 0.0)
                # zero-out all frequency-encoded channels except the raw xyz
                if lin.weight.shape[1] > 3:
                    lin.weight[:, 3:].data.zero_()
            elif uses_skip:
                nn.init.normal_(lin.weight, 0.0, math.sqrt(2.0 / out_dim))
                nn.init.constant_(lin.bias, 0.0)
                if lin.weight.shape[1] > 3:
                    lin.weight[:, -(lin.in_features - 3):].data.zero_()
            else:
                nn.init.normal_(lin.weight, 0.0, math.sqrt(2.0 / out_dim))
                nn.init.constant_(lin.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (sdf, feature) concatenated as a single tensor.

        Shapes: x is (N, 3) in normalised object space; output is (N, 1+F).
        """
        x = x * self.scale
        h0 = self.pos_enc(x)
        h = h0
        for i, lin in enumerate(self.layers):
            if i in self.skip_in:
                h = torch.cat([h, h0], dim=-1) / math.sqrt(2.0)
            h = lin(h)
            if i < self.num_layers - 2:
                h = self.activation(h)
        return h

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[..., :1] / self.scale

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Analytic gradient of the SDF w.r.t. the input (used for normals)."""
        x = x.detach().clone().requires_grad_(True)
        sdf = self.sdf(x).sum()
        grad = torch.autograd.grad(sdf, x, create_graph=True)[0]
        return grad


# ---------------------------------------------------------------------------
# Color network
# ---------------------------------------------------------------------------


class ColorNetwork(nn.Module):
    """View-dependent radiance conditioned on position, normal, view dir, feature."""

    def __init__(self,
                 d_feature: int = 256,
                 d_hidden: int = 256,
                 n_layers: int = 4,
                 n_dir_bands: int = 4,
                 weight_norm: bool = True) -> None:
        super().__init__()
        self.dir_enc = PositionalEncoding(n_dir_bands, include_input=True)
        d_pos = 3
        d_dir = self.dir_enc.out_dim(3)
        d_normal = 3
        d_in = d_pos + d_dir + d_normal + d_feature
        dims = [d_in] + [d_hidden] * (n_layers - 1) + [3]
        layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            layers.append(lin)
        self.layers = layers

    def forward(self,
                x: torch.Tensor,
                normal: torch.Tensor,
                view_dir: torch.Tensor,
                feature: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, self.dir_enc(view_dir), normal, feature], dim=-1)
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if i < len(self.layers) - 1:
                h = F.relu(h, inplace=True)
        return torch.sigmoid(h)


# ---------------------------------------------------------------------------
# Single-scalar learnable "density scale" parameter
# ---------------------------------------------------------------------------


class SingleVariance(nn.Module):
    """Learnable log-s for NeuS's s-density sigmoid."""

    def __init__(self, init_val: float = 0.3) -> None:
        super().__init__()
        self.variance = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.variance * 10.0)
