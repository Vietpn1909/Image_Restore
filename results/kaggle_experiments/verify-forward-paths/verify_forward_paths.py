"""
Test Stage 2 với 2 forward path khác nhau để tìm bug:
  1. SKIP UPSAMPLE — path em pipeline đang dùng
  2. FULL UPSAMPLE — path forward_generator() GỐC trong trainer_stage2.py

Lý thuyết: Stage 2 ckpt được train với 1 trong 2 path. Pipeline phải match.
Nếu Stage 2 train với FULL path nhưng em dùng SKIP → output sai.

Stage 3 train với SKIP path đã verified (PSNR 28 dB).
Stage 2 có thể train với FULL path → pipeline em sai cho Stage 2.

Cách chạy:
    from scripts.verify_forward_paths import test_both_paths
    test_both_paths(STAGE2_CKPT, GFPGAN_PRE, CELEBA, n_samples=30)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.degradation_encoder import DegradationEncoder
from src.models.real_esrgan.rrdb_with_film import RRDBNet_FiLM
from src.models.gfpgan.gfpgan_wrapper import build_face_restorer
from src.models.transformer_fusion import TransformerFusion
from src.degradation.vietnam_pipeline import VietnamDegradationPipeline


@torch.no_grad()
def forward_skip_upsample(deg_encoder, bg_net, face_net, fusion, lq):
    """Path em pipeline đang dùng — skip x4 upsample của Real-ESRGAN."""
    deg_pred = deg_encoder(lq)
    
    # BG branch — KHÔNG đi qua conv_up1, conv_up2 (PixelShuffle)
    feat = bg_net.conv_first(lq)
    feat_skip = feat
    for block in bg_net.body:
        if hasattr(block, 'film'):
            feat = block(feat, deg_pred)
        else:
            feat = block(feat)
    feat = bg_net.conv_body(feat) + feat_skip
    bg_out = bg_net.conv_last(bg_net.lrelu(bg_net.conv_hr(feat))).clamp(0, 1)
    
    # Face branch
    face_input = F.interpolate(lq, size=(512, 512), mode='bilinear', align_corners=False)
    face_512 = face_net(face_input, deg_pred)
    face_out = F.interpolate(face_512, size=lq.shape[-2:], mode='bilinear', align_corners=False)
    
    final = fusion(face_out, bg_out).clamp(0, 1)
    return final


@torch.no_grad()
def forward_full_upsample(deg_encoder, bg_net, face_net, fusion, lq):
    """Path forward_generator() GỐC trong trainer_stage2.py:
       bg_net(lq) → output 4x size → downsample về size LQ.
    """
    deg_pred = deg_encoder(lq)
    
    # BG branch — FULL forward including conv_up1, conv_up2 → output 4x
    bg_out_4x = bg_net(lq, deg_pred)  # (B, 3, 4H, 4W)
    # Downsample về size LQ cho fusion
    bg_out = F.interpolate(bg_out_4x, size=lq.shape[-2:],
                            mode='bilinear', align_corners=False).clamp(0, 1)
    
    # Face branch (same)
    face_input = F.interpolate(lq, size=(512, 512), mode='bilinear', align_corners=False)
    face_512 = face_net(face_input, deg_pred)
    face_out = F.interpolate(face_512, size=lq.shape[-2:], mode='bilinear', align_corners=False)
    
    final = fusion(face_out, bg_out).clamp(0, 1)
    return final


def build_and_load(stage2_ckpt: str, gfpgan_pretrained: str, device='cuda'):
    """Build models giống trainer_stage2 init và load Stage 2 ckpt."""
    ckpt = torch.load(stage2_ckpt, map_location=device, weights_only=False)
    
    mcfg = ckpt.get('config', {}).get('model', {})
    deg_dim = mcfg.get('degradation_encoder', {}).get('out_dim', 8)
    fusion_blocks = mcfg.get('transformer_fusion', {}).get('num_blocks', 2)
    
    deg_encoder = DegradationEncoder(out_dim=deg_dim).to(device)
    bg_net = RRDBNet_FiLM(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                          num_grow_ch=32, scale=4, film_last_n=4,
                          cond_dim=deg_dim).to(device)
    face_net = build_face_restorer(
        gfpgan_path=gfpgan_pretrained, cond_dim=deg_dim,
        fallback_to_simple=False, device=device).to(device)
    fusion = TransformerFusion(
        dim=64, num_blocks=fusion_blocks, num_heads=8,
        in_channels_face=3, in_channels_bg=3, out_channels=3).to(device)
    
    deg_encoder.load_state_dict(ckpt['deg_encoder'])
    bg_net.load_state_dict(ckpt['bg_net'])
    face_net.load_state_dict(ckpt['face_net'])
    fusion.load_state_dict(ckpt['fusion'])
    
    for m in [deg_encoder, bg_net, face_net, fusion]:
        m.eval()
    
    return deg_encoder, bg_net, face_net, fusion


def test_both_paths(stage2_ckpt: str, gfpgan_pretrained: str, celeba_dir: str,
                    n_samples: int = 30, device: str = 'cuda'):
    """Chạy cả 2 path trên cùng val set, so PSNR."""
    print("=" * 70)
    print("TEST 2 FORWARD PATHS — Stage 2")
    print("=" * 70)
    
    deg_encoder, bg_net, face_net, fusion = build_and_load(
        stage2_ckpt, gfpgan_pretrained, device)
    print(f"Models loaded strict\n")
    
    # Val set 150 ảnh đầu CelebA-HQ (replicate ValDataset)
    extensions = {'.jpg', '.jpeg', '.png'}
    val_paths = sorted([p for p in Path(celeba_dir).rglob('*')
                         if p.is_file() and p.suffix.lower() in extensions])[:n_samples]
    print(f"Test trên {len(val_paths)} ảnh val set 150 đầu CelebA-HQ\n")
    
    deg_pipe = VietnamDegradationPipeline(level='medium')
    
    skip_pixel_sum, skip_psnr_sum = 0.0, 0.0
    full_pixel_sum, full_psnr_sum = 0.0, 0.0
    n = 0
    
    for idx, p in enumerate(val_paths):
        deg_pipe.rng.seed(12345 + idx)
        np.random.seed(12345 + idx)
        
        img = Image.open(p).convert('RGB')
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BILINEAR)
        hq_np = np.array(img)
        lq_np, _ = deg_pipe(hq_np)
        
        hq_t = torch.from_numpy(hq_np.astype(np.float32)/255.0)\
            .permute(2,0,1).unsqueeze(0).to(device)
        lq_t = torch.from_numpy(lq_np.astype(np.float32)/255.0)\
            .permute(2,0,1).unsqueeze(0).to(device)
        
        # Path 1: SKIP upsample
        out_skip = forward_skip_upsample(deg_encoder, bg_net, face_net, fusion, lq_t)
        skip_pixel = (out_skip - hq_t).abs().mean().item()
        skip_mse = ((out_skip - hq_t)**2).mean().item()
        skip_psnr = 20 * np.log10(1.0 / np.sqrt(skip_mse + 1e-10))
        
        # Path 2: FULL upsample (trainer_stage2 forward_generator gốc)
        try:
            out_full = forward_full_upsample(deg_encoder, bg_net, face_net, fusion, lq_t)
            full_pixel = (out_full - hq_t).abs().mean().item()
            full_mse = ((out_full - hq_t)**2).mean().item()
            full_psnr = 20 * np.log10(1.0 / np.sqrt(full_mse + 1e-10))
        except RuntimeError as e:
            print(f"  [iter {idx}] OOM at FULL path: {e}")
            return None
        
        skip_pixel_sum += skip_pixel
        skip_psnr_sum += skip_psnr
        full_pixel_sum += full_pixel
        full_psnr_sum += full_psnr
        n += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(val_paths)}] SKIP: PSNR={skip_psnr:.2f}, pixel={skip_pixel:.4f}")
            print(f"                     FULL: PSNR={full_psnr:.2f}, pixel={full_pixel:.4f}")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n[Path 1: SKIP upsample] (em pipeline)")
    print(f"  Pixel L1: {skip_pixel_sum/n:.4f}")
    print(f"  PSNR:     {skip_psnr_sum/n:.2f} dB")
    
    print(f"\n[Path 2: FULL upsample → downsample] (trainer_stage2 forward_generator gốc)")
    print(f"  Pixel L1: {full_pixel_sum/n:.4f}")
    print(f"  PSNR:     {full_psnr_sum/n:.2f} dB")
    
    print(f"\n[Target] Training pixel L1 = 0.023, PSNR ≈ 32 dB")
    
    print("\n" + "=" * 70)
    print("PHÂN TÍCH")
    print("=" * 70)
    
    skip_pixel = skip_pixel_sum / n
    full_pixel = full_pixel_sum / n
    
    if full_pixel < 0.05 and skip_pixel > 0.10:
        print("✅ ROOT CAUSE FOUND!")
        print("   Stage 2 train với FULL UPSAMPLE path (forward_generator GỐC trainer_stage2.py)")
        print("   Pipeline em đang dùng SKIP UPSAMPLE → SAI cho Stage 2.")
        print("")
        print("   FIX: Pipeline phải detect stage:")
        print("   - Stage 2 ckpt → dùng forward_full_upsample")
        print("   - Stage 3 ckpt → dùng forward_skip_upsample")
        print("   Hoặc đơn giản: dùng forward_full_upsample cho mọi stage")
        print("   (Stage 3 cũng có thể tương thích).")
    elif skip_pixel < 0.05 and full_pixel > 0.10:
        print("⚠️ Stage 2 train với SKIP path, FULL path lại sai.")
        print("   Bug nằm ở chỗ khác, không phải forward path.")
    elif skip_pixel > 0.10 and full_pixel > 0.10:
        print("⚠️ CẢ 2 PATH đều sai — vấn đề không phải forward path.")
        print("   Có thể Stage 2 ckpt bị save state hỏng, hoặc bug ở model build.")
        print("   Cần verify trực tiếp trainer.validate() trên cùng ckpt.")
    else:
        print(f"⚠️ Cả 2 path có pixel ~ nhau. Cần debug khác.")
        print(f"   SKIP pixel = {skip_pixel:.4f}, FULL pixel = {full_pixel:.4f}")
    
    return {
        'skip': {'pixel': skip_pixel_sum/n, 'psnr': skip_psnr_sum/n},
        'full': {'pixel': full_pixel_sum/n, 'psnr': full_psnr_sum/n},
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2', required=True)
    parser.add_argument('--gfpgan', required=True)
    parser.add_argument('--celeba', required=True)
    parser.add_argument('--n', type=int, default=30)
    args = parser.parse_args()
    test_both_paths(args.stage2, args.gfpgan, args.celeba, args.n)
