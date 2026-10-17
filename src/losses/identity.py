"""ArcFace identity loss — preserve danh tính khuôn mặt."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceFeatureExtractor(nn.Module):
    """Differentiable ArcFace feature extractor (frozen)."""

    def __init__(self):
        super().__init__()
        try:
            from facexlib.recognition import init_recognition_model
        except ImportError:
            raise ImportError("pip install facexlib")
        self.net = init_recognition_model("arcface", device="cuda")
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0
        feat = self.net(x)
        return F.normalize(feat, dim=1)


def identity_loss(restored, target, extractor):
    fr = extractor(restored)
    ft = extractor(target)
    return (1.0 - (fr * ft).sum(dim=1)).mean()
