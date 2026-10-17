"""
Ablation Variant B — +FiLM (Degradation Encoder + FiLM Injection).

Cấu hình B:
- Real-ESRGAN với FiLM ở 4 RRDB block cuối (Stage 2 weights)
- GFPGAN với pre/post FiLM wrapper (Stage 2 weights)
- Degradation Encoder dự đoán vector 8D (Stage 2 weights)
- Blend Gaussian face mask (KHÔNG có Transformer Fusion)

Mục đích: đo riêng đóng góp 3 (Degradation-aware Encoder + FiLM).
So với A (mức sàn không FiLM), B cho biết FiLM giúp ích bao nhiêu.
So với C = Stage 2 đầy đủ, gap B↔C đo đóng góp của Transformer Fusion.

Design rationale:
    B chính là Stage 2 model NHƯNG thay Transformer Fusion bằng Gaussian
    mask blend. Tất cả weights khác (deg_encoder, bg_net với FiLM, face_net
    với FiLM wrapper) lấy từ Stage 2 checkpoint.

Cách dùng:
    from scripts.ablation_B_film import run_ablation_B
    
    run_ablation_B(
        stage2_ckpt=STAGE2_CKPT,
        gfpgan_path=GFPGAN_PATH,
        test_dir=CELEBA,
        output_dir='/kaggle/working/ablation_B_results',
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

from src.models.degradation_encoder import DegradationEncoder
from src.models.real_esrgan.rrdb_with_film import RRDBNet_FiLM
from src.models.gfpgan.gfpgan_wrapper import (
    build_face_restorer, GFPGANWrapper, SimpleFaceRestorer,
)
from src.degradation.vietnam_pipeline import VietnamDegradationPipeline
from src.utils.gfpgan_import_fix import assert_gfpgan_available

from scripts.evaluate_pipeline import (
    psnr, ssim, LPIPSMetric, IDSimMetric, collect_test_images,
)


# ────────────────────────────── Ablation B model ──────────────────────────────
class AblationBFiLM(torch.nn.Module):
    """Real-ESRGAN + GFPGAN với FiLM (Stage 2 weights) + Gaussian mask blend.
    
    Khác Stage 2 ở 1 điểm: thay Transformer Fusion bằng face mask blend.
    """
    
    def __init__(self, stage2_ckpt: str, gfpgan_path: str,
                 device: str = 'cuda', verbose: bool = True):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # ───── Load Stage 2 checkpoint ─────
        if verbose:
            print(f"[B] Loading Stage 2 checkpoint: {stage2_ckpt}")
        ckpt = torch.load(stage2_ckpt, map_location=self.device, weights_only=False)
        config = ckpt.get('config', {})
        mcfg = config.get('model', {})
        deg_dim = mcfg.get('degradation_encoder', {}).get('out_dim', 8)
        if verbose:
            print(f"  Stage: {ckpt.get('stage', 'unknown')}, "
                  f"Iter: {ckpt.get('iter', 'unknown')}, "
                  f"deg_dim: {deg_dim}")
        
        # ───── GFPGAN import check ─────
        assert_gfpgan_available(verbose=verbose)
        
        # ───── Build models (giống Stage 2 setup) ─────
        if verbose:
            print(f"[B] Building models...")
        
        self.deg_encoder = DegradationEncoder(out_dim=deg_dim).to(self.device)
        
        # bg_net: Real-ESRGAN với FiLM 4 block cuối
        self.bg_net = RRDBNet_FiLM(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
            num_grow_ch=32, scale=4, film_last_n=4, cond_dim=deg_dim,
        ).to(self.device)
        
        # face_net: GFPGAN với pre/post FiLM wrapper
        self.face_net = build_face_restorer(
            gfpgan_path=gfpgan_path, cond_dim=deg_dim,
            fallback_to_simple=False, device=self.device,
        ).to(self.device)
        
        if isinstance(self.face_net, SimpleFaceRestorer):
            raise RuntimeError(
                "face_net fell back to SimpleFaceRestorer; B requires real GFPGAN wrapper.")
        
        # ───── Load weights (strict) ─────
        # Skip 'fusion' — B doesn't have it
        for name, model in [
            ('deg_encoder', self.deg_encoder),
            ('bg_net', self.bg_net),
            ('face_net', self.face_net),
        ]:
            if name not in ckpt:
                raise KeyError(f"Stage 2 checkpoint missing: {name}")
            model.load_state_dict(ckpt[name], strict=True)
            if verbose:
                print(f"  ✓ {name} (strict)")
        
        if verbose:
            n_total = sum(p.numel() for p in self.parameters()) / 1e6
            print(f"  ✓ face_net type: GFPGANWrapper "
                  f"({sum(p.numel() for p in self.face_net.parameters())/1e6:.2f}M params)")
            print(f"[B] Ready: {n_total:.2f}M total params (no Fusion)")
        
        # ───── Face mask Gaussian (cùng tham số với A để fair) ─────
        self.register_buffer('face_mask',
                              self._make_gaussian_face_mask(256))
        
        # Eval mode + freeze
        for m in [self.deg_encoder, self.bg_net, self.face_net]:
            m.eval()
        for p in self.parameters():
            p.requires_grad = False
    
    @staticmethod
    def _make_gaussian_face_mask(size: int = 256,
                                  ratio_x: float = 0.55,
                                  ratio_y: float = 0.65,
                                  sharpness: float = 2.5) -> torch.Tensor:
        """Gaussian ellipse mask giống A baseline (fair comparison)."""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size),
            torch.linspace(-1, 1, size),
            indexing='ij',
        )
        dist_sq = (x / ratio_x) ** 2 + (y / ratio_y) ** 2
        mask = torch.exp(-dist_sq * sharpness).clamp(0, 1)
        return mask.unsqueeze(0).unsqueeze(0)
    
    @torch.no_grad()
    def restore_256(self, lq_256: torch.Tensor) -> torch.Tensor:
        """lq_256: (B, 3, 256, 256) [0,1]
        
        Forward giống Stage 2 (full upsample path) nhưng thay Fusion bằng mask blend.
        """
        # 1. Predict degradation vector
        deg_pred = self.deg_encoder(lq_256)
        
        # 2. BG branch: Real-ESRGAN với FiLM (Stage 2 path: full upsample x4)
        bg_4x = self.bg_net(lq_256, deg_pred)  # (B, 3, 1024, 1024)
        bg_out = F.interpolate(bg_4x, size=(256, 256),
                                mode='bilinear', align_corners=False).clamp(0, 1)
        
        # 3. Face branch: GFPGAN wrapper với pre/post FiLM
        face_input = F.interpolate(lq_256, size=(512, 512),
                                    mode='bilinear', align_corners=False)
        face_512 = self.face_net(face_input, deg_pred)  # (B, 3, 512, 512)
        face_out = F.interpolate(face_512, size=(256, 256),
                                  mode='bilinear', align_corners=False).clamp(0, 1)
        
        # 4. Blend Gaussian mask (THAY VÌ Transformer Fusion)
        mask = self.face_mask.to(device=face_out.device, dtype=face_out.dtype)
        final = mask * face_out + (1 - mask) * bg_out
        return final.clamp(0, 1)
    
    def forward(self, lq_256):
        return self.restore_256(lq_256)


# ────────────────────────────── Runner ──────────────────────────────
def run_ablation_B(
    stage2_ckpt: str,
    gfpgan_path: str,
    test_dir: str,
    output_dir: str,
    n_samples: int = 200,
    levels: List[str] = ['medium', 'hard'],
    image_size: int = 256,
    save_samples: int = 10,
    device: str = 'cuda',
):
    """Chạy biến thể B trên cùng test set với evaluate_pipeline (fair compare)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / 'samples_B'
    samples_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("ABLATION B — +FiLM (Stage 2 weights + Gaussian mask blend)")
    print("=" * 70)
    
    # Build model
    model = AblationBFiLM(stage2_ckpt, gfpgan_path, device=device)
    
    # Build metrics
    lpips_m = LPIPSMetric(device=device)
    id_m = IDSimMetric(device=device)
    print(f"\n  LPIPS: {'ON' if lpips_m.ok else 'OFF'}")
    print(f"  ID-sim: {'ON' if id_m.ok else 'OFF'}")
    
    # Collect test images (cùng seed=42 với evaluate_pipeline)
    test_imgs = collect_test_images(test_dir, n_samples=n_samples)
    print(f"\n  Test set: {len(test_imgs)} ảnh (cùng test set với Stage 2/3)")
    
    # Evaluation loop
    results = {lvl: {'psnr': [], 'ssim': [], 'lpips': [], 'id': []}
                for lvl in levels}
    per_image = []
    
    for level in levels:
        print(f"\n─── Level: {level} ───")
        deg_pipeline = VietnamDegradationPipeline(level=level)
        
        for i, img_path in enumerate(test_imgs):
            try:
                hq_img = Image.open(img_path).convert('RGB').resize(
                    (image_size, image_size), Image.BILINEAR)
                hq_np = np.array(hq_img).astype(np.float32) / 255.0
                lq_np, _ = deg_pipeline(hq_np.copy())
                
                hq_t = torch.from_numpy(hq_np).permute(2, 0, 1).unsqueeze(0).to(device)
                lq_t = torch.from_numpy(lq_np).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Restore với B
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
                    'method': 'ablation_B',
                    'psnr': m_psnr, 'ssim': m_ssim,
                    'lpips': m_lpips, 'id': m_id,
                })
                
                if i < save_samples:
                    grid = torch.cat([lq_t[0], pred[0], hq_t[0]], dim=2)
                    grid_np = (grid.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    Image.fromarray(grid_np).save(
                        samples_dir / f"{level}_{i:03d}_LQ_B_HQ.png")
                
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{n_samples}] PSNR={m_psnr:.2f}, "
                          f"LPIPS={m_lpips:.4f}, ID={m_id:.4f}")
            
            except Exception as e:
                print(f"  ⚠️ [{i}] {img_path.name} failed: {e}")
                continue
    
    # Summary
    summary = []
    for level in levels:
        r = results[level]
        if not r['psnr']:
            continue
        summary.append({
            'level': level, 'method': 'ablation_B',
            'PSNR': float(np.mean(r['psnr'])),
            'SSIM': float(np.mean(r['ssim'])),
            'LPIPS': float(np.nanmean(r['lpips'])),
            'ID-sim': float(np.nanmean(r['id'])),
            'N': len(r['psnr']),
        })
    
    df_summary = pd.DataFrame(summary)
    df_per = pd.DataFrame(per_image)
    df_summary.to_csv(output_dir / 'summary_B.csv', index=False)
    df_per.to_csv(output_dir / 'per_image_B.csv', index=False)
    
    print("\n" + "=" * 70)
    print("KẾT QUẢ ABLATION B — +FiLM")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    
    # In bảng so sánh đầy đủ
    print("\n" + "─" * 84)
    print("BẢNG SO SÁNH ABLATION (200 ảnh CelebA-HQ test, medium):")
    print("─" * 84)
    print(f"{'Method':<28}{'PSNR':<10}{'SSIM':<10}{'LPIPS':<10}{'ID-sim':<10}")
    print("─" * 84)
    print(f"{'LQ baseline':<28}{'26.82':<10}{'0.783':<10}{'0.320':<10}{'0.893':<10}")
    print(f"{'A: ESRGAN + GFPGAN gốc':<28}{'26.50':<10}{'0.809':<10}{'0.201':<10}{'0.895':<10}")
    B_med = next((s for s in summary if s['level'] == 'medium'), None)
    if B_med:
        print(f"{'B: A + FiLM (this)':<28}"
              f"{B_med['PSNR']:<10.2f}"
              f"{B_med['SSIM']:<10.3f}"
              f"{B_med['LPIPS']:<10.3f}"
              f"{B_med['ID-sim']:<10.3f}")
    print(f"{'C: B + Fusion (Stage 2)':<28}{'27.24':<10}{'0.814':<10}{'0.114':<10}{'0.942':<10}")
    print(f"{'D: Stage 3 (full)':<28}{'28.74':<10}{'0.838':<10}{'0.099':<10}{'0.952':<10}")
    
    print(f"\n✓ Saved: {output_dir}/summary_B.csv")
    print(f"✓ Saved: {output_dir}/per_image_B.csv")
    print(f"✓ Samples: {samples_dir}")
    
    return df_summary, df_per


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2', required=True, help='Stage 2 best.pth')
    parser.add_argument('--gfpgan', required=True, help='GFPGANv1.4.pth')
    parser.add_argument('--test_dir', required=True, help='CelebA-HQ dir')
    parser.add_argument('--out', required=True, help='Output dir')
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--levels', nargs='+', default=['medium', 'hard'])
    args = parser.parse_args()
    
    run_ablation_B(
        stage2_ckpt=args.stage2,
        gfpgan_path=args.gfpgan,
        test_dir=args.test_dir,
        output_dir=args.out,
        n_samples=args.n,
        levels=args.levels,
    )
