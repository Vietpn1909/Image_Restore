"""GAN adversarial loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, gan_type: str = "lsgan"):
        super().__init__()
        self.gan_type = gan_type

    def forward(self, pred, target_is_real):
        if self.gan_type == "lsgan":
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return F.mse_loss(pred, target)
        if self.gan_type == "vanilla":
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return F.binary_cross_entropy_with_logits(pred, target)
        if self.gan_type == "hinge":
            if target_is_real:
                return F.relu(1.0 - pred).mean()
            return F.relu(1.0 + pred).mean()
        raise ValueError(f"Unknown gan_type: {self.gan_type}")
