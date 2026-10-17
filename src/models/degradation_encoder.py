"""
Degradation-Aware Encoder.

ĐÓNG GÓP NGHIÊN CỨU SỐ 3:
Encoder nhẹ dự đoán vector suy giảm 8-d ∈ [0, 1], inject vào generator qua FiLM.

Vector 8-d: [blur, noise, jpeg, downsample, scratch, stain, fade, yellowing]

Tham khảo: DASR (Wang et al., CVPR 2021).
Khác biệt:
    1. Supervised từ degradation vector GT (sinh bởi degradation pipeline).
    2. 8 chiều cụ thể theo miền VN.
    3. Inject qua FiLM thay vì concat.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DegradationEncoder(nn.Module):
    def __init__(self, out_dim: int = 8, input_size: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.input_size = input_size

        self.encoder = nn.Sequential(
            ConvBNAct(3, 32, 1),
            ConvBNAct(32, 32, 2),    # 128
            ConvBNAct(32, 64, 1),
            ConvBNAct(64, 64, 2),    # 64
            ConvBNAct(64, 128, 1),
            ConvBNAct(128, 128, 2),  # 32
            ConvBNAct(128, 256, 1),
            ConvBNAct(256, 256, 2),  # 16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                            mode="bilinear", align_corners=False)
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)
