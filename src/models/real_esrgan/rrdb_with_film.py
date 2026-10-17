"""
Real-ESRGAN RRDB Backbone với FiLM Injection.

ĐÓNG GÓP NGHIÊN CỨU SỐ 3 (Phần 1 — Background Branch):
    Inject degradation vector vào backbone Real-ESRGAN qua FiLM
    ở sau N RRDB block CUỐI để điều khiển chiến lược phục chế background
    theo mức suy giảm.

========================================================================
KIẾN TRÚC TỔNG QUAN
========================================================================

Real-ESRGAN gốc dùng RRDBNet (Residual-in-Residual Dense Block Network):

    Input LQ (3 channels)
         │
    ┌────▼──────┐
    │ Conv 3x3  │  → first conv (3 → 64 channels)
    └────┬──────┘
         │
         ├────────────────────────────┐ (residual skip)
         │                            │
    ┌────▼─────────────────────┐      │
    │ 23 × RRDB Block          │      │
    │ (mỗi block: 3 RDB lồng)  │      │
    └────┬─────────────────────┘      │
         │                            │
    ┌────▼─────────────────┐          │
    │ Conv 3x3 (trunk)     │          │
    └────┬─────────────────┘          │
         │                            │
         ├───────────────────────────◄┘  (cộng skip)
         │
    ┌────▼─────────────────┐
    │ Upsample x4          │  (PixelShuffle hoặc interpolate)
    └────┬─────────────────┘
         │
    ┌────▼─────────────────┐
    │ Final Conv (out)     │  (64 → 3 channels)
    └────┬─────────────────┘
         │
    Output HR (3 channels)


========================================================================
MODIFICATION: GẮN FiLM VÀO ĐÂU?
========================================================================

Phân tích:
    - 23 RRDB block có vai trò KHÁC NHAU:
        * Block 1-15 đầu: học LOW-LEVEL features (edges, textures, local patterns)
          → Tương đối "an toàn", ít liên quan đến degradation cụ thể
        * Block 16-23 cuối: học HIGH-LEVEL features (combination, refinement)
          → Quyết định output cuối, NHẠY CẢM với degradation
    
    → INJECT FiLM vào N block CUỐI (default N=4) để model điều chỉnh
      chiến lược refinement theo degradation vector.

Cụ thể:
    
    Input LQ
         │
    ┌────▼──────┐
    │ Conv 3x3  │
    └────┬──────┘
         │
         ├─────────────────────────────────────────────────┐
         │                                                 │
    ┌────▼──────┐                                          │
    │ RRDB 1    │   ← không có FiLM                        │
    └────┬──────┘                                          │
         │                                                 │
    ┌────▼──────┐                                          │
    │ RRDB 2    │   ← không có FiLM                        │
    └────┬──────┘                                          │
         │                                                 │
       ... (đến RRDB 19)                                   │
         │                                                 │
    ┌────▼──────┐                                          │
    │ RRDB 20   │                                          │
    │ + FiLM    │   ← INJECT FILM (4 block cuối)           │
    └────┬──────┘                                          │
         │                                                 │
    ┌────▼──────┐                                          │
    │ RRDB 21   │                                          │
    │ + FiLM    │                                          │
    └────┬──────┘                                          │
         │                                                 │
    ┌────▼──────┐                                          │
    │ RRDB 22   │                                          │
    │ + FiLM    │                                          │
    └────┬──────┘                                          │
         │                                                 │
    ┌────▼──────┐                                          │
    │ RRDB 23   │                                          │
    │ + FiLM    │                                          │
    └────┬──────┘                                          │
         │                                                 │
    ┌────▼─────────────────┐                               │
    │ Conv 3x3 (trunk)     │                               │
    └────┬─────────────────┘                               │
         │                                                 │
         ├────────────────────────────────────────────────◄┘
         ▼
    Upsample x4 → Final Conv → Output


========================================================================
TẠI SAO CHỈ 4 BLOCK CUỐI?
========================================================================

1. Giữ low-level features của block đầu (đã pretrained tốt).
2. FiLM nhẹ (~5k params/block) — không ảnh hưởng performance đáng kể.
3. Phù hợp với philosophy "fine-tuning từ block cuối" của Stage 2.
4. Nếu inject hết 23 block → 23 FiLM → quá nhiều, dễ phá pretrained.


========================================================================
LOAD PRETRAINED WEIGHTS
========================================================================

Real-ESRGAN x4plus pretrained có format:
    {
        'params_ema': {...}  (hoặc 'params')
    }

state_dict gồm:
    - conv_first.weight, conv_first.bias
    - body.0.rdb1.conv1.weight, ... (RRDB blocks)
    - conv_body.weight, conv_body.bias
    - conv_up1.weight, conv_up2.weight (upsampling)
    - conv_hr.weight, conv_last.weight (final convs)

Khi load:
    - load với strict=False
    - FiLM layers (mới thêm) không match → giữ random init (sẽ train).
    - Backbone weights match → load OK.


========================================================================
REFERENCES
========================================================================
[1] Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
    with Pure Synthetic Data", ICCVW 2021.
[2] Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial
    Networks", ECCVW 2018.
[3] Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
    AAAI 2018.

Author: Đặng Thành Tựu
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.film_layer import FiLMLayer


# ========================================================================
# Helper: pixel unshuffle (for x4 input downsampling option)
# ========================================================================
def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Pixel unshuffle — ngược của PixelShuffle.
    Dùng khi Real-ESRGAN nhận input đã downsampled.
    
    Input:  (B, C, H, W)
    Output: (B, C*scale*scale, H/scale, W/scale)
    """
    b, c, hh, hw = x.shape
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# ========================================================================
# RESIDUAL DENSE BLOCK (RDB) — building block của RRDB
# ========================================================================
class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block.
    
    Architecture (5 conv layers với dense connection):
    
        x ──┬─► Conv1 ──► LRelu ──► x1
            │                        │
            ├──concat([x, x1])──► Conv2 ──► LRelu ──► x2
            │                                          │
            ├──concat([x, x1, x2])──► Conv3 ──► LRelu ──► x3
            │                                              │
            ├──concat([x, x1, x2, x3])──► Conv4 ──► LRelu ──► x4
            │                                                  │
            └──concat([x, x1, x2, x3, x4])──► Conv5 ──► x5
                                                          │
                              return x + 0.2 * x5 ◄───────┘
    
    Đặc điểm:
    - Dense connection: mỗi conv concat với tất cả output trước đó
    - Residual với scale 0.2 (giúp ổn định training)
    - Không có batch norm (theo ESRGAN — BN gây artifact ở SR)
    
    Args:
        num_feat: số channel chính (default 64)
        num_grow_ch: số channel mỗi conv layer thêm vào (default 32)
    """
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        # 5 conv layers, mỗi conv lấy input là concat của tất cả output trước
        # Conv1: num_feat → num_grow_ch
        # Conv2: num_feat + num_grow_ch → num_grow_ch
        # Conv3: num_feat + 2*num_grow_ch → num_grow_ch
        # Conv4: num_feat + 3*num_grow_ch → num_grow_ch
        # Conv5: num_feat + 4*num_grow_ch → num_feat (về channel gốc)
        
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Init weights nhẹ hơn để ổn định (theo ESRGAN paper)
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            m.weight.data *= 0.1   # scale nhỏ
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual với scale 0.2
        return x5 * 0.2 + x


# ========================================================================
# RRDB BLOCK — Residual-in-Residual Dense Block
# ========================================================================
class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block.
    
    Cấu trúc: 3 RDB lồng nhau + residual outer.
    
        x ──► RDB1 ──► RDB2 ──► RDB3 ──┐
              (out1)   (out2)   (out3)  │
        x ──────────────────────────────┘ (skip)
                                        │
                            return x + 0.2 * out3
    
    Đây là "block of blocks" — residual ở 2 cấp:
    1. Inner: trong mỗi RDB.
    2. Outer: skip qua cả 3 RDB.
    """
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Outer residual với scale 0.2
        return out * 0.2 + x


# ========================================================================
# RRDB BLOCK + FiLM — modification của chúng tôi
# ========================================================================
class RRDB_FiLM(nn.Module):
    """
    RRDB block với FiLM injection ở CUỐI.
    
    Cấu trúc:
        x ──► RDB1 ──► RDB2 ──► RDB3 ──► FiLM(deg_vec) ──┐
                                            ▲             │
                                      deg_vec input       │
        x ─────────────────────────────────────────────────┘ (skip)
                                                          │
                                              return x + 0.2 * out
    
    Lý do FiLM ở CUỐI block (sau RDB3, trước residual outer):
    1. FiLM modulate OUTPUT của block — ảnh hưởng nhiều nhất.
    2. Residual outer giữ skip pristine → ổn định training.
    3. Nếu FiLM ở giữa (sau RDB1 hay RDB2) → ảnh hưởng đến RDB sau → khó train.
    
    Args:
        num_feat: channel chính.
        num_grow_ch: channel của RDB.
        cond_dim: số chiều của degradation vector (default 8).
        film_hidden: hidden dim của FiLM MLP (default 64).
    """
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32,
                 cond_dim: int = 8, film_hidden: int = 64):
        super().__init__()
        # 3 RDB blocks như RRDB gốc
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        
        # FiLM layer mới — modulate output của RDB3
        self.film = FiLMLayer(
            cond_dim=cond_dim,
            feat_channels=num_feat,
            hidden_dim=film_hidden,
        )

    def forward(self, x: torch.Tensor, deg_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: feature (B, C, H, W)
            deg_vec: degradation vector (B, cond_dim)
        """
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Apply FiLM modulation
        out = self.film(out, deg_vec)
        # Outer residual
        return out * 0.2 + x


# ========================================================================
# RRDBNet + FiLM — backbone chính
# ========================================================================
class RRDBNet_FiLM(nn.Module):
    """
    Real-ESRGAN RRDBNet với FiLM injection ở N block CUỐI.
    
    Args:
        num_in_ch: số channel input (default 3 cho RGB).
        num_out_ch: số channel output (default 3).
        num_feat: số channel chính trong network (default 64).
        num_block: tổng số RRDB block (default 23 — chuẩn Real-ESRGAN).
        num_grow_ch: channel grow trong RDB (default 32).
        scale: upscale factor (4 cho RealESRGAN_x4plus).
        film_last_n: số RRDB cuối có FiLM (default 4).
        cond_dim: degradation vector dim (default 8).
        film_hidden: hidden dim của FiLM MLP.
    
    Pretrained loading note:
        Khi load RealESRGAN_x4plus.pth:
        - 19 RRDB block đầu là RRDB thường → load weights bình thường.
        - 4 RRDB block cuối là RRDB_FiLM → backbone RDB load OK, FiLM giữ random init.
        - Phần upsampling load OK.
        Phải dùng strict=False.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
        film_last_n: int = 4,
        cond_dim: int = 8,
        film_hidden: int = 64,
    ):
        super().__init__()
        self.scale = scale
        self.film_last_n = film_last_n
        self.cond_dim = cond_dim

        # Validate
        assert film_last_n <= num_block, \
            f"film_last_n ({film_last_n}) > num_block ({num_block})"

        # ---------- Backbone ----------
        # First conv (3 → 64)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # 23 RRDB blocks
        # - First (num_block - film_last_n) blocks: RRDB thường
        # - Last film_last_n blocks: RRDB_FiLM (có FiLM injection)
        self.body = nn.ModuleList()
        for i in range(num_block):
            if i < num_block - film_last_n:
                # RRDB thường
                self.body.append(RRDB(num_feat, num_grow_ch))
            else:
                # RRDB với FiLM
                self.body.append(RRDB_FiLM(
                    num_feat=num_feat,
                    num_grow_ch=num_grow_ch,
                    cond_dim=cond_dim,
                    film_hidden=film_hidden,
                ))

        # Body conv (trunk)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # ---------- Upsampling ----------
        # x4 upsampling: 2 stages of nearest interpolation + conv
        # (Real-ESRGAN dùng cách này, KHÔNG dùng PixelShuffle)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        deg_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input LQ image (B, 3, H, W) ∈ [0, 1]
            deg_vec: degradation vector (B, cond_dim) ∈ [0, 1]
                     Bắt buộc nếu film_last_n > 0.
        
        Returns:
            HR output (B, 3, H*scale, W*scale)
        """
        if self.film_last_n > 0 and deg_vec is None:
            raise ValueError(f"deg_vec is required when film_last_n={self.film_last_n}")

        # 1. First conv
        feat = self.conv_first(x)
        feat_skip = feat  # giữ skip cho residual outer

        # 2. Pass through 23 RRDB blocks
        for i, block in enumerate(self.body):
            if isinstance(block, RRDB_FiLM):
                # Block có FiLM → cần deg_vec
                feat = block(feat, deg_vec)
            else:
                # RRDB thường
                feat = block(feat)

        # 3. Body conv (trunk) + skip residual
        feat = self.conv_body(feat)
        feat = feat + feat_skip

        # 4. Upsampling x4 (2 lần x2 với nearest interpolate)
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        # 5. Final convs
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)

        return out


# ========================================================================
# LOAD PRETRAINED HELPER
# ========================================================================
def load_realesrgan_pretrained(
    model: RRDBNet_FiLM,
    pretrained_path: str,
    verbose: bool = True,
) -> tuple[list, list]:
    """
    Load pretrained Real-ESRGAN weights vào RRDBNet_FiLM.
    
    Pretrained file format (RealESRGAN_x4plus.pth):
        {
            'params_ema': {...}  # EMA weights (tốt hơn, ưu tiên)
            'params': {...}      # raw weights (fallback)
        }
    
    Khi load:
    - Backbone weights match → load.
    - FiLM weights không có trong pretrained → giữ random init.
    - Báo cáo missing/unexpected keys.
    
    Returns:
        (missing_keys, unexpected_keys)
    """
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    # Get state_dict (ưu tiên EMA)
    if 'params_ema' in ckpt:
        state_dict = ckpt['params_ema']
        if verbose:
            print(f"[Real-ESRGAN] Loading params_ema (EMA weights)")
    elif 'params' in ckpt:
        state_dict = ckpt['params']
        if verbose:
            print(f"[Real-ESRGAN] Loading params (raw weights)")
    else:
        # Maybe direct state_dict
        state_dict = ckpt
        if verbose:
            print(f"[Real-ESRGAN] Loading direct state_dict")

    # Load với strict=False (cho phép missing FiLM keys)
    result = model.load_state_dict(state_dict, strict=False)

    # missing: keys trong model nhưng không có trong pretrained → FiLM layers (expected)
    # unexpected: keys trong pretrained nhưng không có trong model → có thể là bug
    if verbose:
        print(f"  Missing keys: {len(result.missing_keys)}")
        # In ra vài missing keys (chủ yếu là FiLM)
        film_missing = [k for k in result.missing_keys if 'film' in k.lower()]
        non_film_missing = [k for k in result.missing_keys if 'film' not in k.lower()]
        if film_missing:
            print(f"    FiLM keys (expected): {len(film_missing)}")
        if non_film_missing:
            print(f"    Non-FiLM missing (CẦN CHECK): {non_film_missing[:5]}")
            if len(non_film_missing) > 5:
                print(f"    ... và {len(non_film_missing) - 5} keys khác")

        print(f"  Unexpected keys: {len(result.unexpected_keys)}")
        if result.unexpected_keys:
            print(f"    Examples: {result.unexpected_keys[:3]}")

    return result.missing_keys, result.unexpected_keys


# ========================================================================
# FREEZE/UNFREEZE HELPERS
# ========================================================================
def freeze_rrdbnet_body(model: RRDBNet_FiLM, unfreeze_last_n: int = 0):
    """
    Freeze RRDBNet body, optional unfreeze last N RRDB blocks.
    
    Stage 2 strategy:
    - Freeze: conv_first, body[0:19] (19 RRDB đầu)
    - Unfreeze: body[19:23] (4 RRDB cuối — có FiLM), conv_body, upsampling
    
    Args:
        model: RRDBNet_FiLM
        unfreeze_last_n: số RRDB cuối unfreeze (default 0 = freeze all body).
    """
    # Freeze tất cả tham số trước
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze conv_body và upsampling (luôn train)
    for module in [model.conv_body, model.conv_up1, model.conv_up2,
                   model.conv_hr, model.conv_last]:
        for p in module.parameters():
            p.requires_grad = True

    # Unfreeze N RRDB block cuối
    if unfreeze_last_n > 0:
        for block in model.body[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

    # FiLM layers luôn trainable (mới thêm)
    for block in model.body:
        if isinstance(block, RRDB_FiLM):
            for p in block.film.parameters():
                p.requires_grad = True


def count_params(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable, total) params."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ========================================================================
# SMOKE TEST
# ========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SMOKE TEST — RRDBNet_FiLM")
    print("=" * 70)

    # Tạo model
    model = RRDBNet_FiLM(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
        film_last_n=4,
        cond_dim=8,
    )

    trainable, total = count_params(model)
    print(f"\nModel size:")
    print(f"  Total params:     {total / 1e6:.2f}M")
    print(f"  Trainable params: {trainable / 1e6:.2f}M")

    # Test forward
    print(f"\nForward test:")
    dummy_lq = torch.randn(2, 3, 64, 64)
    dummy_deg = torch.rand(2, 8)
    with torch.no_grad():
        out = model(dummy_lq, dummy_deg)
    print(f"  Input shape:  {dummy_lq.shape}")
    print(f"  Deg shape:    {dummy_deg.shape}")
    print(f"  Output shape: {out.shape}  (expected: [2, 3, 256, 256] for x4)")

    # Test freeze
    print(f"\nFreeze test (unfreeze_last_n=4):")
    freeze_rrdbnet_body(model, unfreeze_last_n=4)
    trainable, total = count_params(model)
    print(f"  After freeze: trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*trainable/total:.1f}%)")
    
    print("\n" + "=" * 70)
    print("PASSED")
    print("=" * 70)
