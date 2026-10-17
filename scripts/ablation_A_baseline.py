"""
Ablation Variant A — BASELINE.

Cấu hình:
- Real-ESRGAN gốc (KHÔNG có FiLM injection)
- GFPGAN gốc (KHÔNG có pre/post FiLM wrapper)
- KHÔNG có Degradation Encoder
- KHÔNG có Transformer Fusion
- Blend bằng Gaussian face mask đơn giản

Mục đích: đo mức "sàn" khi chỉ ghép 2 pretrained model lại, không có
đóng góp nào của đồ án. So sánh với B (+FiLM), C (Stage 2), D (Stage 3)
để chứng minh giá trị từng đóng góp.

QUAN TRỌNG — fair comparison:
    Script dùng CÙNG hàm collect_test_images(seed=42) như evaluate_pipeline.py
    và CÙNG cách degrade (float input, RNG nội bộ tự increment) để fair
    compare với Stage 2/3 metric đã có.

Cách dùng:
    # Trong notebook Kaggle (sau khi sys.path đã có project):
    from scripts.ablation_A_baseline import run_ablation_A
    
    run_ablation_A(
        esrgan_path=REALESRGAN_PATH,
        gfpgan_path=GFPGAN_PATH,
        test_dir=CELEBA,
        output_dir='/kaggle/working/ablation_A_results',
        n_samples=200,
        levels=['medium', 'hard'],
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd

from src.models.real_esrgan.rrdb_with_film import RRDBNet_FiLM
from src.degradation.vietnam_pipeline import VietnamDegradationPipeline
from src.utils.gfpgan_import_fix import assert_gfpgan_available

# Reuse các hàm metric + collect_test_images từ evaluate_pipeline.py
from scripts.evaluate_pipeline import (
    psnr, ssim, LPIPSMetric, IDSimilarityMetric, collect_test_images,
)


# ────────────────────────────── Ablation A model ──────────────────────────────
class AblationABaseline(torch.nn.Module):
    """Real-ESRGAN gốc + GFPGAN gốc + Gaussian mask blend.
    
    KHÔNG có FiLM, KHÔNG có Fusion, KHÔNG có Degradation Encoder.
    Tương đương phương pháp "ghép 2 model SOTA bằng face mask đơn giản".
    """
    
    def __init__(self, esrgan_path: str, gfpgan_path: str,
                 device: str = 'cuda', verbose: bool = True):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # ───── BG: Real-ESRGAN GỐC (film_last_n=0 → KHÔNG có FiLM) ─────
        if verbose:
            print("[A] Building Real-ESRGAN gốc (no FiLM)...")
        self.bg_net = RRDBNet_FiLM(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
            num_grow_ch=32, scale=4,
            film_last_n=0,  # KHÔNG có FiLM block nào
            cond_dim=8,
        ).to(self.device)
        
        # Load pretrained
        ckpt = torch.load(esrgan_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('params_ema', ckpt.get('params', ckpt))
        result = self.bg_net.load_state_dict(sd, strict=False)
        if verbose:
            print(f"  Missing: {len(result.missing_keys)}, "
                  f"Unexpected: {len(result.unexpected_keys)}")
        
        # ───── Face: GFPGAN GỐC (no wrapper FiLM) ─────
        if verbose:
            print("[A] Building GFPGAN gốc (no wrapper)...")
        assert_gfpgan_available(verbose=verbose)
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
        
        self.face_net = GFPGANv1Clean(
            out_size=512, num_style_feat=512, channel_multiplier=2,
            decoder_load_path=None, fix_decoder=False,
            num_mlp=8, input_is_latent=True, different_w=True,
            narrow=1, sft_half=True,
        ).to(self.device)
        
        ckpt = torch.load(gfpgan_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('params_ema', ckpt.get('params', ckpt))
        result = self.face_net.load_state_dict(sd, strict=False)
        if verbose:
            print(f"  Missing: {len(result.missing_keys)}, "
                  f"Unexpected: {len(result.unexpected_keys)}")
        
        # ───── Face mask Gaussian (precompute) ─────
        self.register_buffer('face_mask',
                              self._make_gaussian_face_mask(256))
        
        # Eval mode + freeze
        self.bg_net.eval()
        self.face_net.eval()
        for p in self.parameters():
            p.requires_grad = False
        
        n_bg = sum(p.numel() for p in self.bg_net.parameters()) / 1e6
        n_face = sum(p.numel() for p in self.face_net.parameters()) / 1e6
        if verbose:
            print(f"[A] Ready: BG {n_bg:.2f}M + Face {n_face:.2f}M = "
                  f"{n_bg + n_face:.2f}M params (no extra)")
    
    @staticmethod
    def _make_gaussian_face_mask(size: int = 256,
                                  ratio_x: float = 0.55,
                                  ratio_y: float = 0.65,
                                  sharpness: float = 2.5) -> torch.Tensor:
        """Gaussian ellipse mask centered, hơi cao hơn rộng (giống mặt người).
        
        - ratio_x=0.55: bán kính ngang (so với nửa ảnh)
        - ratio_y=0.65: bán kính dọc (mặt thường dài hơn)
        - sharpness=2.5: độ sắc của biên (cao = biên rõ)
        
        Returns: (1, 1, size, size) tensor, values ∈ [0, 1]
        """
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size),
            torch.linspace(-1, 1, size),
            indexing='ij',
        )
        dist_sq = (x / ratio_x) ** 2 + (y / ratio_y) ** 2
        mask = torch.exp(-dist_sq * sharpness).clamp(0, 1)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    @torch.no_grad()
    def restore_256(self, lq_256: torch.Tensor) -> torch.Tensor:
        """lq_256: (B, 3, 256, 256) [0,1]
        
        Returns: (B, 3, 256, 256) [0,1]
        """
        # ── BG branch: Real-ESRGAN full upsample x4 → downsample về 256 ──
        bg_4x = self.bg_net(lq_256)  # (B, 3, 1024, 1024), no deg_vec
        bg_out = F.interpolate(bg_4x, size=(256, 256),
                                mode='bilinear', align_corners=False).clamp(0, 1)
        
        # ── Face branch: GFPGAN expect 512×512 normalized [-1, 1] ──
        face_512 = F.interpolate(lq_256, size=(512, 512),
                                  mode='bilinear', align_corners=False)
        face_512_norm = face_512 * 2 - 1  # [0,1] → [-1, 1]
        
        # GFPGAN forward (no FiLM, no deg_vec)
        gfpgan_out = self.face_net(face_512_norm, return_rgb=False)
        if isinstance(gfpgan_out, tuple):
            face_out_512 = gfpgan_out[0]
        else:
            face_out_512 = gfpgan_out
        
        face_out_512 = (face_out_512 + 1) / 2  # [-1, 1] → [0, 1]
        face_out = F.interpolate(face_out_512, size=(256, 256),
                                  mode='bilinear', align_corners=False).clamp(0, 1)
        
        # ── Blend: Gaussian mask đơn giản ──
        # mask = 1 ở vùng mặt → lấy face_out
        # mask = 0 ở vùng nền → lấy bg_out
        final = self.face_mask * face_out + (1 - self.face_mask) * bg_out
        return final.clamp(0, 1)
    
    def forward(self, lq_256):
        return self.restore_256(lq_256)


# ────────────────────────────── Runner ──────────────────────────────
def run_ablation_A(
    esrgan_path: str,
    gfpgan_path: str,
    test_dir: str,
    output_dir: str,
    n_samples: int = 200,
    levels: List[str] = ['medium', 'hard'],
    image_size: int = 256,
    save_samples: int = 10,
    device: str = 'cuda',
):
    """Chạy biến thể A trên cùng test set với evaluate_pipeline (fair compare).
    
    Args:
        esrgan_path: path RealESRGAN_x4plus.pth
        gfpgan_path: path GFPGANv1.4.pth
        test_dir: thư mục chứa ảnh CelebA-HQ test
        output_dir: thư mục output (sẽ tạo nếu chưa có)
        n_samples: số ảnh test (mặc định 200, giống Stage 2/3 eval)
        levels: list mức suy giảm, mặc định ['medium', 'hard']
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / 'samples_A'
    samples_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("ABLATION A — BASELINE (Real-ESRGAN + GFPGAN gốc + Gaussian mask)")
    print("=" * 70)
    
    # Build model
    model = AblationABaseline(esrgan_path, gfpgan_path, device=device)
    
    # Build metrics
    lpips_m = LPIPSMetric(device=device)
    id_m = IDSimilarityMetric(device=device)
    print(f"\n  LPIPS: {'ON' if lpips_m.ok else 'OFF'}")
    print(f"  ID-sim: {'ON' if id_m.ok else 'OFF'}")
    
    # ─────────── Collect test images (CÙNG seed=42 với evaluate_pipeline) ───────────
    test_imgs = collect_test_images(test_dir, n_samples=n_samples)
    print(f"\n  Test set: {len(test_imgs)} ảnh (cùng test set với Stage 2/3 eval)")
    
    # ─────────────────────── Evaluation loop ───────────────────────
    results = {lvl: {'psnr': [], 'ssim': [], 'lpips': [], 'id': []}
                for lvl in levels}
    per_image = []
    
    for level in levels:
        print(f"\n─── Level: {level} ───")
        # Tạo deg_pipeline mới mỗi level (giống evaluate_pipeline.py)
        deg_pipeline = VietnamDegradationPipeline(level=level)
        
        for i, img_path in enumerate(test_imgs):
            try:
                # Load HQ (giống evaluate_pipeline.py line 239-241)
                hq_img = Image.open(img_path).convert('RGB').resize(
                    (image_size, image_size), Image.BILINEAR)
                hq_np = np.array(hq_img).astype(np.float32) / 255.0
                
                # Degrade
                lq_np, _ = deg_pipeline(hq_np.copy())
                
                # To tensor
                hq_t = torch.from_numpy(hq_np).permute(2, 0, 1).unsqueeze(0).to(device)
                lq_t = torch.from_numpy(lq_np).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Restore với A baseline
                pred = model.restore_256(lq_t)
                
                # Metrics
                m_psnr = psnr(pred, hq_t)
                m_ssim = ssim(pred, hq_t)
                m_lpips = lpips_m(pred, hq_t) if lpips_m.ok else float('nan')
                m_id = id_m(pred, hq_t) if id_m.ok else float('nan')
                
                results[level]['psnr'].append(m_psnr)
                results[level]['ssim'].append(m_ssim)
                results[level]['lpips'].append(m_lpips)
                results[level]['id'].append(m_id)
                
                per_image.append({
                    'level': level, 'idx': i, 'image': img_path.name,
                    'method': 'ablation_A',
                    'psnr': m_psnr, 'ssim': m_ssim,
                    'lpips': m_lpips, 'id': m_id,
                })
                
                # Save sample visualization (first N): LQ | A | HQ
                if i < save_samples:
                    grid = torch.cat([lq_t[0], pred[0], hq_t[0]], dim=2)
                    grid_np = (grid.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    Image.fromarray(grid_np).save(
                        samples_dir / f"{level}_{i:03d}_LQ_A_HQ.png")
                
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{n_samples}] PSNR={m_psnr:.2f}, "
                          f"LPIPS={m_lpips:.4f}, ID={m_id:.4f}")
            
            except Exception as e:
                print(f"  ⚠️ [{i}] {img_path.name} failed: {e}")
                continue
    
    # ─────────────────────── Summary ───────────────────────
    summary = []
    for level in levels:
        r = results[level]
        if not r['psnr']:
            continue
        summary.append({
            'level': level, 'method': 'ablation_A',
            'PSNR': float(np.mean(r['psnr'])),
            'SSIM': float(np.mean(r['ssim'])),
            'LPIPS': float(np.nanmean(r['lpips'])),
            'ID-sim': float(np.nanmean(r['id'])),
            'N': len(r['psnr']),
        })
    
    # Save CSV
    df_summary = pd.DataFrame(summary)
    df_per = pd.DataFrame(per_image)
    df_summary.to_csv(output_dir / 'summary_A.csv', index=False)
    df_per.to_csv(output_dir / 'per_image_A.csv', index=False)
    
    print("\n" + "=" * 70)
    print("KẾT QUẢ ABLATION A — BASELINE")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    
    # In so sánh với Stage 2/3 đã có
    print("\n" + "─" * 70)
    print("So sánh với Stage 2/3 hiện tại (200 ảnh CelebA-HQ test):")
    print("─" * 70)
    print(f"{'Method':<20}{'PSNR (medium)':<18}{'LPIPS (medium)':<18}{'ID (medium)':<15}")
    print(f"{'─' * 70}")
    print(f"{'LQ baseline':<20}{'26.82':<18}{'0.320':<18}{'0.893':<15}")
    A_med = next((s for s in summary if s['level'] == 'medium'), None)
    if A_med:
        print(f"{'A baseline (this)':<20}{A_med['PSNR']:<18.2f}{A_med['LPIPS']:<18.4f}{A_med['ID-sim']:<15.4f}")
    print(f"{'Stage 2 (=C)':<20}{'27.24':<18}{'0.114':<18}{'0.942':<15}")
    print(f"{'Stage 3 (=D)':<20}{'28.74':<18}{'0.099':<18}{'0.952':<15}")
    
    print(f"\n✓ Saved: {output_dir}/summary_A.csv")
    print(f"✓ Saved: {output_dir}/per_image_A.csv")
    print(f"✓ Samples: {samples_dir}")
    
    return df_summary, df_per


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--esrgan', required=True, help='RealESRGAN_x4plus.pth')
    parser.add_argument('--gfpgan', required=True, help='GFPGANv1.4.pth')
    parser.add_argument('--test_dir', required=True, help='CelebA-HQ dir')
    parser.add_argument('--out', required=True, help='Output dir')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--levels', nargs='+', default=['medium', 'hard'])
    args = parser.parse_args()
    
    run_ablation_A(
        esrgan_path=args.esrgan,
        gfpgan_path=args.gfpgan,
        test_dir=args.test_dir,
        output_dir=args.out,
        n_samples=args.n,
        levels=args.levels,
    )
