"""
Verify Stage 2 ckpt bằng cách replicate EXACT validation process trong training.

Nếu pipeline em viết bị bug, script này sẽ ra PSNR khác training.
Nếu pipeline OK, script sẽ ra pixel loss ≈ 0.023 và PSNR ≈ 30-32 dB
(matching Stage 2 best val ở iter 3000).

Chạy trong notebook Kaggle:
    !python scripts/verify_stage2_on_val.py \
        --stage2 $STAGE2_CKPT \
        --gfpgan $GFPGAN_PRE \
        --celeba $CELEBA \
        --level medium

Hoặc import vào notebook:
    from scripts.verify_stage2_on_val import verify
    verify(STAGE2_CKPT, GFPGAN_PRE, CELEBA, level='medium')
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.pipeline.restore_pipeline import OldPortraitRestorationPipeline
from src.degradation.vietnam_pipeline import VietnamDegradationPipeline


def list_val_images(celeba_dir: str, max_samples: int = 150):
    """Replicate ValDataset path collection: 150 ảnh đầu (sorted) của CelebA-HQ."""
    celeba_dir = Path(celeba_dir)
    extensions = {'.jpg', '.jpeg', '.png'}
    all_imgs = sorted([p for p in celeba_dir.rglob('*')
                        if p.is_file() and p.suffix.lower() in extensions])
    return all_imgs[:max_samples]


def verify(stage2_ckpt: str, gfpgan_pretrained: str, celeba_dir: str,
           level: str = 'medium', n_samples: int = 150, device: str = 'cuda',
           verbose: bool = True):
    """Verify Stage 2 ckpt trên CHÍNH val set 150 ảnh đầu."""
    
    print("=" * 70)
    print(f"VERIFY STAGE 2 trên val set {n_samples} ảnh đầu CelebA-HQ")
    print(f"Level: {level}")
    print("=" * 70)
    
    # 1. Build pipeline với strict load
    pipe = OldPortraitRestorationPipeline(
        checkpoint_path=stage2_ckpt,
        gfpgan_pretrained=gfpgan_pretrained,
        device=device,
        allow_simple_fallback=False,
        strict_load=True,
        verbose=verbose,
    )
    
    # 2. Replicate EXACT val process từ ValDataset.__getitem__
    val_paths = list_val_images(celeba_dir, max_samples=n_samples)
    print(f"\n[Val] {len(val_paths)} images")
    
    # Single deg pipeline instance — same as ValDataset (re-seeded per item)
    deg_pipe = VietnamDegradationPipeline(level=level)
    
    pixel_sum = 0.0
    psnr_sum = 0.0
    mse_sum = 0.0
    n = 0
    
    with torch.no_grad():
        for idx, p in enumerate(val_paths):
            # Match ValDataset seed strategy (data_loader.py line 263-264)
            deg_pipe.rng.seed(12345 + idx)
            np.random.seed(12345 + idx)
            
            # Load + center_crop_resize equivalent (CelebA-HQ 256x256 sẵn)
            img = Image.open(p).convert('RGB')
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.BILINEAR)
            hq_np = np.array(img)  # uint8 (256, 256, 3)
            
            # Generate LQ — uint8 in → uint8 out (EXACT như ValDataset)
            lq_np, _ = deg_pipe(hq_np)
            
            # to_tensor: uint8 → /255 → float32 [0,1]
            hq_t = torch.from_numpy(hq_np.astype(np.float32)/255.0)\
                .permute(2,0,1).unsqueeze(0).to(device)
            lq_t = torch.from_numpy(lq_np.astype(np.float32)/255.0)\
                .permute(2,0,1).unsqueeze(0).to(device)
            
            # Forward — same pipeline as evaluation
            out = pipe.restore_256(lq_t)
            
            # Metrics
            pixel_l1 = (out - hq_t).abs().mean().item()
            mse = ((out - hq_t)**2).mean().item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))
            
            pixel_sum += pixel_l1
            mse_sum += mse
            psnr_sum += psnr
            n += 1
            
            if (idx + 1) % 30 == 0:
                print(f"  [{idx+1}/{len(val_paths)}] pixel={pixel_l1:.4f}, PSNR={psnr:.2f}")
    
    avg_pixel = pixel_sum / n
    avg_mse = mse_sum / n
    avg_psnr = psnr_sum / n
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Stage 2 trên val set {n} ảnh đầu (seed 12345+idx):")
    print(f"  Pixel L1: {avg_pixel:.4f}")
    print(f"  MSE:      {avg_mse:.6f}")
    print(f"  PSNR:     {avg_psnr:.2f} dB")
    print()
    print(f"So sánh với training Stage 2 best (iter 3000):")
    print(f"  Pixel L1 training = 0.023 (val 150 ảnh, same set)")
    print(f"  PSNR tương đương  ≈ 32 dB")
    print()
    
    if avg_pixel < 0.05 and avg_psnr > 25:
        print("✅ KẾT LUẬN: Pipeline em viết ĐÚNG — Stage 2 ckpt OK.")
        print("   Vấn đề evaluation PSNR 15.93 là do test set 200 ảnh KHÁC val set 150 ảnh →")
        print("   Stage 2 overfit val set, không generalize. Đây không phải bug pipeline.")
    elif avg_pixel > 0.15 or avg_psnr < 18:
        print("⚠️ KẾT LUẬN: Có vấn đề THỰC SỰ — pixel L1 trên val set cũng cao.")
        print("   Hoặc Stage 2 ckpt bị save hỏng, hoặc có bug pipeline em chưa thấy.")
        print("   Cần debug sâu hơn:")
        print("   - Verify ckpt size, hash")
        print("   - Test Stage 2 trainer.validate() trực tiếp (không qua pipeline)")
    else:
        print("⚠️ KẾT LUẬN: Số ở giữa — Stage 2 đã train nhưng kém hơn kỳ vọng.")
    
    return {
        'pixel_l1': avg_pixel,
        'mse': avg_mse,
        'psnr': avg_psnr,
        'n': n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2', required=True, help='Stage 2 best.pth path')
    parser.add_argument('--gfpgan', required=True, help='GFPGANv1.4.pth path')
    parser.add_argument('--celeba', required=True, help='CelebA-HQ dir')
    parser.add_argument('--level', default='medium', choices=['light', 'medium', 'hard'])
    parser.add_argument('--n', type=int, default=150)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    verify(
        stage2_ckpt=args.stage2,
        gfpgan_pretrained=args.gfpgan,
        celeba_dir=args.celeba,
        level=args.level,
        n_samples=args.n,
        device=args.device,
    )


if __name__ == '__main__':
    main()
