"""
Discriminator cho GAN training ở Stage 2-3.

ĐÓNG GÓP NGHIÊN CỨU PHỤ TRỢ:
    Adversarial loss giúp output trông "thật" hơn — quan trọng cho phục chế
    ảnh chân dung nơi pixel loss không đủ để tạo texture sắc nét.

KIẾN TRÚC — PatchGAN Style

PatchGAN (Isola et al., CVPR 2017):
    Thay vì discriminate cả ảnh là real/fake (1 scalar),
    PatchGAN chia ảnh thành các PATCH và discriminate mỗi patch.
    
    Output: (B, 1, H', W') — mỗi pixel ở output là prediction cho 1 patch.

Ưu điểm so với full-image discriminator:
1. Ít params hơn, train nhanh hơn.
2. Focus vào texture local (quan trọng cho ảnh face).
3. Robust với resolution khác nhau.

Cấu trúc (cho input 256x256):

    Input: (B, 3, 256, 256)
         │
    ┌────▼─────────────────┐
    │ Conv 4x4 s=2 (3→64)  │  + LeakyReLU(0.2)
    └────┬─────────────────┘  → (B, 64, 128, 128)
         │
    ┌────▼─────────────────┐
    │ Conv 4x4 s=2 (64→128)│  + BN + LeakyReLU
    └────┬─────────────────┘  → (B, 128, 64, 64)
         │
    ┌────▼─────────────────┐
    │ Conv 4x4 s=2 (128→256)│  + BN + LeakyReLU
    └────┬─────────────────┘  → (B, 256, 32, 32)
         │
    ┌────▼─────────────────┐
    │ Conv 4x4 s=1 (256→512)│  + BN + LeakyReLU
    └────┬─────────────────┘  → (B, 512, 31, 31)
         │
    ┌────▼─────────────────┐
    │ Conv 4x4 s=1 (512→1) │  → final logits
    └────┬─────────────────┘  → (B, 1, 30, 30)
         │
    Output: 30×30 patch predictions


SPECTRAL NORM (optional)

Real-ESRGAN dùng U-Net Discriminator + Spectral Normalization để ổn định
GAN training. Tôi implement option dùng spectral norm.

Spectral norm: normalize weight matrix bằng spectral norm (max singular value)
→ Lipschitz constraint → ổn định training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ========================================================================
# PATCHGAN DISCRIMINATOR
# ========================================================================
class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator cho ảnh 256x256.
    
    Output: (B, 1, 30, 30) — patch predictions.
    
    Args:
        in_channels: số channel input (3 cho RGB).
        base_channels: base channels (64).
        n_layers: số down-sampling layers (3 — cho 256→30).
        use_spectral_norm: dùng spectral norm cho stable GAN training.
        use_bn: dùng batch norm (False nếu dùng SN).
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
        use_bn: bool = False,
    ):
        super().__init__()

        def conv_layer(in_ch: int, out_ch: int, stride: int):
            """Conv layer với optional spectral norm + BN."""
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride,
                            padding=1, bias=not use_bn)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            layers = [conv]
            if use_bn and not use_spectral_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # First layer: no BN
        layers = [
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Down-sampling layers
        in_ch = base_channels
        for i in range(1, n_layers):
            out_ch = min(in_ch * 2, base_channels * 8)
            layers.append(conv_layer(in_ch, out_ch, stride=2))
            in_ch = out_ch

        # 1 layer without downsample (stride=1)
        out_ch = min(in_ch * 2, base_channels * 8)
        layers.append(conv_layer(in_ch, out_ch, stride=1))
        in_ch = out_ch

        # Final output layer (1 channel logits)
        final_conv = nn.Conv2d(in_ch, 1, 4, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        layers.append(final_conv)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image (B, 3, H, W) ∈ [0, 1] hoặc [-1, 1]
        Returns:
            Patch predictions (B, 1, H', W')
        """
        return self.model(x)


# UNET DISCRIMINATOR (Real-ESRGAN style, optional)
class UNetDiscriminator(nn.Module):
    """
    U-Net discriminator như Real-ESRGAN.
    
    Output cùng spatial size với input (per-pixel prediction).
    Hiệu quả hơn PatchGAN cho ảnh face HR.
    
    Args:
        in_channels: 3 cho RGB.
        base_channels: 64.
        use_spectral_norm: dùng SN (luôn nên True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        def sn_conv(in_ch, out_ch, k, s, p, bias=True):
            conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            return conv

        # Encoder
        self.conv0 = sn_conv(in_channels, base_channels, 3, 1, 1)
        self.conv1 = sn_conv(base_channels, base_channels*2, 4, 2, 1)
        self.conv2 = sn_conv(base_channels*2, base_channels*4, 4, 2, 1)
        self.conv3 = sn_conv(base_channels*4, base_channels*8, 4, 2, 1)

        # Decoder (upsample + conv)
        self.conv4 = sn_conv(base_channels*8, base_channels*4, 3, 1, 1)
        self.conv5 = sn_conv(base_channels*4, base_channels*2, 3, 1, 1)
        self.conv6 = sn_conv(base_channels*2, base_channels, 3, 1, 1)

        # Final
        self.conv7 = sn_conv(base_channels, base_channels, 3, 1, 1)
        self.conv8 = sn_conv(base_channels, base_channels, 3, 1, 1)
        self.conv9 = nn.Conv2d(base_channels, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)

        # Decoder with skip
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        x4 = x4 + x2   # skip

        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)
        x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)
        x6 = x6 + x0

        x7 = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        x8 = F.leaky_relu(self.conv8(x7), 0.2, inplace=True)
        out = self.conv9(x8)

        return out


# SMOKE TEST
if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST — Discriminators")
    print("=" * 70)

    # PatchGAN
    d_patch = PatchGANDiscriminator(use_spectral_norm=True)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out_patch = d_patch(x)
    n_params = sum(p.numel() for p in d_patch.parameters())
    print(f"\nPatchGAN:")
    print(f"  Params: {n_params / 1e6:.2f}M")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out_patch.shape}")

    # U-Net
    d_unet = UNetDiscriminator(use_spectral_norm=True)
    with torch.no_grad():
        out_unet = d_unet(x)
    n_params = sum(p.numel() for p in d_unet.parameters())
    print(f"\nU-Net Discriminator:")
    print(f"  Params: {n_params / 1e6:.2f}M")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out_unet.shape}")

    print("\n" + "=" * 70)
    print("PASSED")
    print("=" * 70)
