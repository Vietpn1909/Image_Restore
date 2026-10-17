"""
Transformer Cross-Attention Fusion.

ĐÓNG GÓP NGHIÊN CỨU SỐ 4:
Module fusion sử dụng Transformer cross-attention giữa face feature (GFPGAN)
và background feature (Real-ESRGAN). Mục đích: làm hài hòa mặt với nền,
giảm artifact rìa mặt, đồng bộ tone màu/ánh sáng.

Kiến trúc: MDTA (Multi-Dconv head Transposed Attention) từ Restormer
(Zamir et al., CVPR 2022) — linear complexity theo pixel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class MDTACrossAttention(nn.Module):
    """Multi-DConv head Transposed Cross-Attention.
    Q từ face, K/V từ background.
    """

    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=bias)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, feat_face, feat_bg):
        B, C, H, W = feat_face.shape
        q = self.q_dwconv(self.q_proj(feat_face))
        kv = self.kv_dwconv(self.kv_proj(feat_bg))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, "b (h c) x y -> b h c (x y)", h=self.num_heads)
        k = rearrange(k, "b (h c) x y -> b h c (x y)", h=self.num_heads)
        v = rearrange(v, "b (h c) x y -> b h c (x y)", h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b h c (x y) -> b (h c) x y", x=H, y=W)
        return self.out_proj(out)


class GatedDConvFFN(nn.Module):
    def __init__(self, dim, expansion=2.66, bias=False):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion=2.66):
        super().__init__()
        self.norm_q = LayerNorm2d(dim)
        self.norm_kv = LayerNorm2d(dim)
        self.cross_attn = MDTACrossAttention(dim, num_heads)
        self.norm_ffn = LayerNorm2d(dim)
        self.ffn = GatedDConvFFN(dim, ffn_expansion)

    def forward(self, feat_face, feat_bg):
        feat = feat_face + self.cross_attn(self.norm_q(feat_face), self.norm_kv(feat_bg))
        feat = feat + self.ffn(self.norm_ffn(feat))
        return feat


class TransformerFusion(nn.Module):
    def __init__(self, dim=64, num_blocks=2, num_heads=8, ffn_expansion=2.66,
                 in_channels_face=3, in_channels_bg=3, out_channels=3):
        super().__init__()
        self.embed_face = nn.Conv2d(in_channels_face, dim, 3, padding=1)
        self.embed_bg = nn.Conv2d(in_channels_bg, dim, 3, padding=1)
        self.blocks = nn.ModuleList([
            CrossAttentionFusionBlock(dim, num_heads, ffn_expansion)
            for _ in range(num_blocks)
        ])
        self.decode = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, out_channels, 3, padding=1),
        )

    def forward(self, face_img, bg_img):
        feat_face = self.embed_face(face_img)
        feat_bg = self.embed_bg(bg_img)
        for block in self.blocks:
            feat_face = block(feat_face, feat_bg)
        return face_img + self.decode(feat_face)
