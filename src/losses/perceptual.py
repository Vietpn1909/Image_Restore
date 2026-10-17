"""VGG19 perceptual loss (Johnson et al., ECCV 2016)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    LAYER_MAP = {"conv1_2": 3, "conv2_2": 8, "conv3_4": 17, "conv4_4": 26, "conv5_4": 35}

    def __init__(self, layer_weights=None, norm_input=True, use_l1=True):
        super().__init__()
        if layer_weights is None:
            layer_weights = {"conv1_2": 0.1, "conv2_2": 0.1,
                            "conv3_4": 1.0, "conv4_4": 1.0, "conv5_4": 1.0}
        self.layer_weights = layer_weights
        self.norm_input = norm_input
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        for p in vgg.parameters():
            p.requires_grad = False
        vgg.eval()
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x):
        return (x - self.mean) / self.std if self.norm_input else x

    def _extract(self, x):
        x = self._norm(x)
        feats = {}
        target_idx = set(self.LAYER_MAP.values())
        max_idx = max(target_idx)
        idx_to_name = {v: k for k, v in self.LAYER_MAP.items()}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in target_idx:
                feats[idx_to_name[i]] = x
            if i >= max_idx:
                break
        return feats

    def forward(self, pred, target):
        fp = self._extract(pred)
        ft = self._extract(target)
        loss = 0.0
        for name, w in self.layer_weights.items():
            loss = loss + w * self.criterion(fp[name], ft[name].detach())
        return loss
