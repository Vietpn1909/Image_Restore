"""
FiLM (Feature-wise Linear Modulation).
Reference: Perez et al., AAAI 2018.

Vai trò: inject degradation vector vào feature map của generator để điều khiển
chiến lược phục chế theo mức suy giảm.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, feat_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.feat_channels = feat_channels
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_channels * 2),
        )
        # Init: gamma=1, beta=0 (identity)
        with torch.no_grad():
            last = self.mlp[-1]
            last.weight.zero_()
            last.bias.zero_()
            last.bias[:feat_channels].fill_(1.0)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        params = self.mlp(cond)
        gamma, beta = params.chunk(2, dim=1)
        return gamma.view(B, C, 1, 1) * feat + beta.view(B, C, 1, 1)
