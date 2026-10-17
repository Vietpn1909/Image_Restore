"""
GFPGAN Backbone Wrapper với FiLM Injection.

ĐÓNG GÓP NGHIÊN CỨU SỐ 3 (Phần 2 — Face Branch):
    Inject degradation vector vào GFPGAN qua FiLM ở các điểm chiến lược
    để điều khiển phục chế khuôn mặt theo mức suy giảm.

========================================================================
KIẾN TRÚC GFPGAN GỐC (v1.4)
========================================================================

GFPGAN gồm 3 phần chính:

    1. U-Net Encoder
       Input: degraded face (512x512)
       Output: latent + multi-scale features
       
    2. StyleGAN2 Generator (pretrained, frozen)
       Input: latent + features
       Output: high-quality face

    3. Channel-Split-Spatial-Feature-Transform (CS-SFT)
       Tích hợp features từ U-Net vào StyleGAN2.


========================================================================
TẠI SAO KHÔNG MODIFY GFPGAN GỐC?
========================================================================

GFPGAN code chính thức ở:
    https://github.com/TencentARC/GFPGAN/blob/master/gfpgan/archs/gfpganv1_clean_arch.py

Vấn đề khi modify:
1. Code lớn (~2000 dòng), nhiều dependency.
2. StyleGAN2 generator dùng kiến trúc đặc biệt — gắn FiLM vào trong khó.
3. Nếu inject FiLM giữa StyleGAN2 → phá pretrained nặng.

==> APPROACH KHÁC: WRAPPER PATTERN

Thay vì modify backbone, ta:
- LOAD pretrained GFPGAN nguyên xi (qua thư viện `gfpgan` chính thức).
- Wrap nó trong 1 wrapper class.
- Thêm FiLM injection ở 2 ĐIỂM ngoài backbone:
  
  (a) PRE-PROCESSING FiLM: modulate input LQ trước khi đưa vào GFPGAN
  (b) POST-PROCESSING FiLM: modulate output trước khi return
  
  Cấu trúc:
  
      LQ face (512x512)
           │
      ┌────▼─────────────┐
      │ Pre-FiLM         │  ← inject deg_vec
      │ (1 conv + FiLM)  │
      └────┬─────────────┘
           │
      ┌────▼─────────────────┐
      │ GFPGAN backbone      │  ← pretrained, freeze hoặc unfreeze last N
      │ (U-Net + StyleGAN2)  │
      └────┬─────────────────┘
           │
      ┌────▼─────────────┐
      │ Post-FiLM        │  ← inject deg_vec
      │ (FiLM + 1 conv)  │
      └────┬─────────────┘
           │
      Restored face

ƯU ĐIỂM:
- Đơn giản hơn modify backbone.
- Pretrained GFPGAN giữ nguyên — không phá.
- FiLM vẫn tham gia learning, predict theo degradation.

NHƯỢC ĐIỂM:
- FiLM không inject vào TRONG StyleGAN2 (chỉ rìa).
- Hiệu quả có thể kém hơn so với modify internal.
- Nhưng đủ cho đồ án cử nhân, defendable.


========================================================================
ALTERNATIVE: KHÔNG DÙNG GFPGAN, IMPLEMENT FACE BRANCH ĐƠN GIẢN
========================================================================

Nếu không cài được `gfpgan` package, có thể fallback dùng SimpleFaceRestorer:
- Một U-Net nhẹ (~5-10M params)
- Có FiLM injection ở bottleneck + decoder
- Train từ đầu với pretrained features

Tôi implement cả 2:
1. GFPGANWrapper — dùng pretrained GFPGAN (CHÍNH)
2. SimpleFaceRestorer — fallback


========================================================================
REFERENCES
========================================================================
[1] Wang et al., "Towards Real-World Blind Face Restoration with Generative
    Facial Prior", CVPR 2021. (GFPGAN paper)
[2] Karras et al., "Analyzing and Improving the Image Quality of StyleGAN",
    CVPR 2020. (StyleGAN2)

"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.film_layer import FiLMLayer
from src.utils.gfpgan_import_fix import patch_torchvision_functional_tensor_alias


# ========================================================================
# GFPGAN WRAPPER — dùng pretrained, wrap với pre/post FiLM
# ========================================================================
class GFPGANWrapper(nn.Module):
    """
    Wrapper quanh pretrained GFPGAN với pre/post FiLM injection.
    
    Args:
        gfpgan_path: path đến GFPGANv1.4.pth
        cond_dim: degradation vector dim (default 8)
        film_hidden: FiLM MLP hidden (default 64)
        freeze_backbone: freeze GFPGAN backbone (default True ban đầu, sẽ
                         unfreeze sau)
        device: device để load model
    """

    def __init__(
        self,
        gfpgan_path: str,
        cond_dim: int = 8,
        film_hidden: int = 64,
        freeze_backbone: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.device = device

        # ---------- Load GFPGAN backbone ----------
        # Dùng class GFPGANv1Clean từ thư viện chính thức.
        # Patch BasicSR/torchvision compatibility trước khi import để tránh
        # fallback SimpleFaceRestorer khi evaluate trên Kaggle.
        patch_torchvision_functional_tensor_alias(verbose=False)
        try:
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
        except Exception as e:
            raise ImportError(
                "Cần cài/patch gfpgan đúng: gfpgan==1.3.8 basicsr==1.4.2 facexlib==0.3.0. "
                f"Original error: {e}"
            ) from e

        # Init GFPGAN v1.4 architecture
        self.gfpgan = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,         # không load decoder riêng (đã có trong .pth)
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True,
        )

        # Load weights
        ckpt = torch.load(gfpgan_path, map_location='cpu', weights_only=False)
        if 'params_ema' in ckpt:
            state_dict = ckpt['params_ema']
        elif 'params' in ckpt:
            state_dict = ckpt['params']
        else:
            state_dict = ckpt
        
        result = self.gfpgan.load_state_dict(state_dict, strict=False)
        print(f"[GFPGAN] Loaded from {gfpgan_path}")
        print(f"  Missing keys: {len(result.missing_keys)}")
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")

        # Freeze backbone ban đầu
        if freeze_backbone:
            for p in self.gfpgan.parameters():
                p.requires_grad = False

        # ---------- Pre-FiLM module ----------
        # Modulate input LQ trước khi vào GFPGAN
        # Input: (B, 3, 512, 512) → Output: (B, 3, 512, 512)
        self.pre_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pre_film = FiLMLayer(cond_dim, 32, film_hidden)
        self.pre_conv2 = nn.Conv2d(32, 3, 3, padding=1)

        # ---------- Post-FiLM module ----------
        # Modulate output từ GFPGAN
        self.post_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.post_film = FiLMLayer(cond_dim, 32, film_hidden)
        self.post_conv2 = nn.Conv2d(32, 3, 3, padding=1)

        # Init pre/post convs với weight nhỏ để output ban đầu ≈ identity
        for m in [self.pre_conv1, self.pre_conv2, self.post_conv1, self.post_conv2]:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
            m.weight.data *= 0.01   # nhỏ để xuất phát gần identity
            nn.init.zeros_(m.bias)

    def forward(
        self,
        face: torch.Tensor,
        deg_vec: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            face: input face image (B, 3, 512, 512) ∈ [0, 1] hoặc [-1, 1]
            deg_vec: degradation vector (B, cond_dim)
        
        Returns:
            restored face (B, 3, 512, 512)
        """
        # ---------- Pre-FiLM ----------
        # Input giả định [0, 1] → normalize về [-1, 1] cho GFPGAN
        if face.min() >= 0:
            face_in = face * 2.0 - 1.0
        else:
            face_in = face

        pre_feat = self.pre_conv1(face_in)
        pre_feat = self.pre_film(pre_feat, deg_vec)
        pre_feat = F.leaky_relu(pre_feat, 0.2)
        pre_delta = self.pre_conv2(pre_feat)
        modulated_in = face_in + pre_delta   # residual

        # ---------- GFPGAN backbone ----------
        # GFPGAN trả về tuple (output, intermediate)
        gfpgan_out = self.gfpgan(modulated_in, return_rgb=False)
        if isinstance(gfpgan_out, tuple):
            backbone_out = gfpgan_out[0]
        else:
            backbone_out = gfpgan_out

        # ---------- Post-FiLM ----------
        post_feat = self.post_conv1(backbone_out)
        post_feat = self.post_film(post_feat, deg_vec)
        post_feat = F.leaky_relu(post_feat, 0.2)
        post_delta = self.post_conv2(post_feat)
        modulated_out = backbone_out + post_delta

        # Output: convert về [0, 1] để compatible với pipeline
        result = (modulated_out + 1.0) / 2.0
        result = result.clamp(0, 1)

        return result


# ========================================================================
# SIMPLE FACE RESTORER — fallback nếu không cài được GFPGAN
# ========================================================================
class SimpleFaceRestorer(nn.Module):
    """
    Face restorer đơn giản (U-Net nhẹ) với FiLM injection.
    
    Dùng làm fallback khi không cài được GFPGAN package.
    
    Architecture:
        Encoder (4 levels) → Bottleneck + FiLM → Decoder (4 levels)
        
    Params: ~10M (so với GFPGAN ~85M)
    """

    def __init__(
        self,
        base_channels: int = 64,
        cond_dim: int = 8,
        film_hidden: int = 64,
        out_size: int = 512,
    ):
        super().__init__()
        self.out_size = out_size

        # --- Encoder ---
        self.enc1 = self._conv_block(3, base_channels)              # 512
        self.enc2 = self._conv_block(base_channels, base_channels*2)  # 256
        self.enc3 = self._conv_block(base_channels*2, base_channels*4)  # 128
        self.enc4 = self._conv_block(base_channels*4, base_channels*8)  # 64

        # --- Bottleneck với FiLM ---
        self.bottleneck = self._conv_block(base_channels*8, base_channels*8)
        self.film = FiLMLayer(cond_dim, base_channels*8, film_hidden)

        # --- Decoder ---
        self.dec4 = self._conv_block(base_channels*16, base_channels*4)  # +skip
        self.dec3 = self._conv_block(base_channels*8, base_channels*2)
        self.dec2 = self._conv_block(base_channels*4, base_channels)
        self.dec1 = self._conv_block(base_channels*2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, 3, 3, padding=1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, face: torch.Tensor, deg_vec: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(face)            # B, C, 512, 512
        e2 = self.enc2(F.max_pool2d(e1, 2))  # B, 2C, 256, 256
        e3 = self.enc3(F.max_pool2d(e2, 2))  # B, 4C, 128, 128
        e4 = self.enc4(F.max_pool2d(e3, 2))  # B, 8C, 64, 64

        # Bottleneck với FiLM
        b = self.bottleneck(F.max_pool2d(e4, 2))   # B, 8C, 32, 32
        b = self.film(b, deg_vec)

        # Decoder với skip
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Final output (residual)
        out = self.out_conv(d1)
        return (face + out).clamp(0, 1)


# ========================================================================
# FACTORY — tự chọn GFPGAN hoặc SimpleFaceRestorer
# ========================================================================
def build_face_restorer(
    gfpgan_path: Optional[str] = None,
    cond_dim: int = 8,
    fallback_to_simple: bool = True,
    device: str = 'cuda',
):
    """
    Build face restorer. Ưu tiên GFPGAN, fallback SimpleFaceRestorer.
    
    Args:
        gfpgan_path: path đến GFPGAN .pth. Nếu None hoặc fail → fallback.
        cond_dim: degradation vector dim.
        fallback_to_simple: nếu GFPGAN fail, dùng SimpleFaceRestorer.
        
    Returns:
        Module để dùng làm face restorer.
    """
    if gfpgan_path and Path(gfpgan_path).exists():
        try:
            print("Building GFPGANWrapper...")
            return GFPGANWrapper(
                gfpgan_path=gfpgan_path,
                cond_dim=cond_dim,
                device=device,
            )
        except (ImportError, Exception) as e:
            print(f"GFPGAN failed: {e}")
            if not fallback_to_simple:
                raise
            print("Fallback to SimpleFaceRestorer")
    
    return SimpleFaceRestorer(cond_dim=cond_dim)


# ========================================================================
# SMOKE TEST
# ========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST — SimpleFaceRestorer (fallback)")
    print("=" * 70)

    model = SimpleFaceRestorer(base_channels=64, cond_dim=8, out_size=512)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParams: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")

    # Forward
    face = torch.randn(2, 3, 512, 512)
    deg = torch.rand(2, 8)
    with torch.no_grad():
        out = model(face, deg)
    print(f"\nInput:  {face.shape}")
    print(f"Output: {out.shape}")
    print(f"Range:  [{out.min():.2f}, {out.max():.2f}]")

    print("\n" + "=" * 70)
    print("PASSED — SimpleFaceRestorer")
    print("=" * 70)
    print("\nNote: GFPGANWrapper test cần gfpgan package + .pth file,")
    print("      sẽ test trên Kaggle (có sẵn package).")
