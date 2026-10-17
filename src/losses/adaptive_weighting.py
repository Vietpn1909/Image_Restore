"""
Adaptive Loss Weighting.

Trọng số loss thay đổi theo mức suy giảm (severity = mean của degradation vector):
- Ảnh nhẹ → tăng pixel loss (preserve chi tiết).
- Ảnh nặng → tăng adversarial/perceptual (generate plausible content).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossWeights:
    pixel: float
    perceptual: float
    adversarial: float
    identity: float
    cycle: float = 0.0


class AdaptiveLossWeighter:
    def __init__(self, base: LossWeights):
        self.base = base

    def __call__(self, degradation_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        severity = degradation_vec.mean(dim=1)  # (B,)
        return {
            "pixel": self.base.pixel * (1.0 - 0.5 * severity),
            "perceptual": self.base.perceptual * (1.0 + 0.5 * severity),
            "adversarial": self.base.adversarial * (1.0 + severity),
            "identity": torch.full_like(severity, self.base.identity),
            "cycle": torch.full_like(severity, self.base.cycle),
        }
