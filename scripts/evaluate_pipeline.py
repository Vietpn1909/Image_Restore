"""
Script đánh giá định lượng pipeline restoration.

Tính 4 metric trên test set:
    - PSNR  (cao = tốt, đơn vị dB)
    - SSIM  (cao = tốt, [0, 1])
    - LPIPS (thấp = tốt, [0, 1])
    - ID-similarity (cosine giữa 2 ArcFace embedding, cao = tốt, [-1, 1])

So sánh Stage 2 vs Stage 3 trên 2 mức degradation (medium + hard).

Output:
    - CSV với từng ảnh × stage × level
    - Bảng tóm tắt console + LaTeX

Cách dùng (trong notebook Kaggle):
    from scripts.evaluate_pipeline import run_evaluation
    
    results = run_evaluation(
        stage2_ckpt='/kaggle/input/.../stage2/best.pth',
        stage3_ckpt='/kaggle/input/.../stage3/best.pth',
        gfpgan_pretrained='/kaggle/input/.../GFPGANv1.4.pth',
        test_dir='/kaggle/input/celebahq...',
        n_samples=200,
        output_dir='/kaggle/working/eval_results',
    )
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.pipeline.restore_pipeline import OldPortraitRestorationPipeline, to_tensor
from src.utils.gfpgan_import_fix import assert_gfpgan_available
from src.degradation.vietnam_pipeline import VietnamDegradationPipeline


def psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB).
    Input: tensor (B, 3, H, W) [0, 1].
    """
    mse = ((pred - gt) ** 2).mean(dim=[1, 2, 3])
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-10))
    return psnr_val.mean().item()


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Structural Similarity Index.
    Implementation đơn giản từ window 11×11.
    """
    from skimage.metrics import structural_similarity as sk_ssim
    
    pred_np = pred.clamp(0, 1).cpu().numpy()
    gt_np = gt.clamp(0, 1).cpu().numpy()
    scores = []
    for i in range(pred_np.shape[0]):
        # (3, H, W) → (H, W, 3)
        p = pred_np[i].transpose(1, 2, 0)
        g = gt_np[i].transpose(1, 2, 0)
        s = sk_ssim(p, g, channel_axis=2, data_range=1.0)
        scores.append(s)
    return float(np.mean(scores))


class LPIPSMetric:
    """Wrapper for LPIPS (Learned Perceptual Image Patch Similarity)."""
    
    def __init__(self, device='cuda', net='alex'):
        try:
            import lpips
            self.model = lpips.LPIPS(net=net, verbose=False).to(device)
            self.model.eval()
            self.device = device
            self.ok = True
        except Exception as e:
            print(f"⚠️ LPIPS không init được: {e}")
            self.ok = False
    
    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        if not self.ok:
            return float('nan')
        # LPIPS expect [-1, 1]
        pred_norm = pred.clamp(0, 1) * 2 - 1
        gt_norm = gt.clamp(0, 1) * 2 - 1
        d = self.model(pred_norm.to(self.device), gt_norm.to(self.device))
        return d.mean().item()


class IDSimMetric:
    """Identity similarity dùng ArcFace."""
    
    def __init__(self, device='cuda'):
        try:
            from facexlib.recognition import init_recognition_model
            self.model = init_recognition_model('arcface', device=device)
            self.model.eval()
            self.device = device
            self.ok = True
        except Exception as e:
            print(f"⚠️ ArcFace không init được: {e}")
            self.ok = False
    
    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        if not self.ok:
            return float('nan')
        # ArcFace expects 112×112
        pred_112 = F.interpolate(pred, size=(112, 112),
                                  mode='bilinear', align_corners=False)
        gt_112 = F.interpolate(gt, size=(112, 112),
                                mode='bilinear', align_corners=False)
        # Normalize to [-1, 1]
        pred_norm = (pred_112.clamp(0, 1) - 0.5) / 0.5
        gt_norm = (gt_112.clamp(0, 1) - 0.5) / 0.5
        
        feat_pred = self.model(pred_norm.to(self.device))
        feat_gt = self.model(gt_norm.to(self.device))
        # Cosine similarity
        cos = F.cosine_similarity(feat_pred, feat_gt, dim=1)
        return cos.mean().item()


def collect_test_images(test_dir: str, n_samples: int = 200,
                        seed: int = 42) -> List[Path]:
    """Pick n_samples ảnh ngẫu nhiên từ test_dir.
    
    Skip những ảnh đầu (thường dùng cho val) để tránh overlap.
    """
    test_dir = Path(test_dir)
    extensions = {'.jpg', '.jpeg', '.png'}
    
    all_imgs = sorted([p for p in test_dir.rglob('*')
                        if p.is_file() and p.suffix.lower() in extensions])
    
    if len(all_imgs) < n_samples + 200:
        print(f"⚠️ Test dir chỉ có {len(all_imgs)} ảnh, có thể overlap với val.")
    
    # Lấy từ phần CUỐI để tránh overlap với val set (val thường lấy 150 đầu)
    candidates = all_imgs[200:] if len(all_imgs) > 200 + n_samples else all_imgs
    
    random.seed(seed)
    selected = random.sample(candidates, min(n_samples, len(candidates)))
    print(f"[Test] Selected {len(selected)} images from {test_dir}")
    return selected


def run_evaluation(
    stage2_ckpt: str,
    stage3_ckpt: str,
    gfpgan_pretrained: str,
    test_dir: str,
    output_dir: str = '/kaggle/working/eval_results',
    n_samples: int = 200,
    levels: List[str] = ['medium', 'hard'],
    image_size: int = 256,
    save_samples: int = 10,
    device: str = 'cuda',
) -> Dict:
    """Chạy evaluation đầy đủ.
    
    Args:
        stage2_ckpt: path Stage 2 best.pth
        stage3_ckpt: path Stage 3 best.pth
        gfpgan_pretrained: GFPGAN pretrained
        test_dir: folder ảnh HQ test (vd CelebA-HQ)
        output_dir: nơi lưu kết quả + qualitative samples
        n_samples: số ảnh test
        levels: list mức degradation
        image_size: 256
        save_samples: số ảnh save qualitative (LQ + S2 + S3 + HQ)
    
    Returns:
        dict với keys 'summary' (mean metrics) và 'per_image' (chi tiết)
    """
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = Path(output_dir) / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("EVALUATION PIPELINE — STRICT GFPGAN MODE")
    print("=" * 70)
    assert_gfpgan_available(verbose=True)
    
    # ─────────── Load pipelines ───────────
    print("\n[1/5] Loading Stage 2 pipeline...")
    pipe2 = OldPortraitRestorationPipeline(
        checkpoint_path=stage2_ckpt,
        gfpgan_pretrained=gfpgan_pretrained,
        device=device,
        image_size=image_size,
        allow_simple_fallback=False,
        strict_load=True,
    )
    
    print("\n[2/5] Loading Stage 3 pipeline...")
    pipe3 = OldPortraitRestorationPipeline(
        checkpoint_path=stage3_ckpt,
        gfpgan_pretrained=gfpgan_pretrained,
        device=device,
        image_size=image_size,
        allow_simple_fallback=False,
        strict_load=True,
    )
    
    # ─────────── Init metrics ───────────
    print("\n[3/5] Initializing metrics...")
    lpips_metric = LPIPSMetric(device=device)
    id_metric = IDSimMetric(device=device)
    print(f"  LPIPS: {'ON' if lpips_metric.ok else 'OFF (skip)'}")
    print(f"  ID-sim: {'ON' if id_metric.ok else 'OFF (skip)'}")
    
    # ─────────── Collect test images ───────────
    print("\n[4/5] Collecting test images...")
    test_imgs = collect_test_images(test_dir, n_samples=n_samples)
    
    # ─────────── Evaluation loop ───────────
    print("\n[5/5] Running evaluation...")
    
    results = {level: {'stage2': {'psnr': [], 'ssim': [], 'lpips': [], 'id': []},
                       'stage3': {'psnr': [], 'ssim': [], 'lpips': [], 'id': []},
                       'lq_baseline': {'psnr': [], 'ssim': [], 'lpips': [], 'id': []}}
               for level in levels}
    
    for level in levels:
        print(f"\n─── Level: {level} ───")
        deg_pipeline = VietnamDegradationPipeline(level=level)
        
        for i, img_path in enumerate(test_imgs):
            try:
                # Load HQ
                hq_img = Image.open(img_path).convert('RGB').resize(
                    (image_size, image_size), Image.BILINEAR)
                hq_np = np.array(hq_img).astype(np.float32) / 255.0  # (H, W, 3)
                
                # Apply degradation
                lq_np, _ = deg_pipeline(hq_np.copy())
                
                # Convert to tensor
                hq_t = torch.from_numpy(hq_np).permute(2, 0, 1).unsqueeze(0).to(device)
                lq_t = torch.from_numpy(lq_np).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Restore with Stage 2 and Stage 3
                s2_out = pipe2.restore_256(lq_t)
                s3_out = pipe3.restore_256(lq_t)
                
                # Compute metrics
                for name, pred in [('stage2', s2_out), ('stage3', s3_out),
                                   ('lq_baseline', lq_t)]:
                    results[level][name]['psnr'].append(psnr(pred, hq_t))
                    results[level][name]['ssim'].append(ssim(pred, hq_t))
                    if lpips_metric.ok:
                        results[level][name]['lpips'].append(lpips_metric(pred, hq_t))
                    if id_metric.ok:
                        results[level][name]['id'].append(id_metric(pred, hq_t))
                
                # Save qualitative sample
                if i < save_samples:
                    save_sample(
                        samples_dir / f'{level}_{i:03d}.png',
                        lq=lq_t, s2=s2_out, s3=s3_out, hq=hq_t,
                    )
                
                if (i + 1) % 20 == 0:
                    print(f"  [{level}] Processed {i+1}/{len(test_imgs)}")
            
            except Exception as e:
                print(f"  ✗ Error on {img_path.name}: {e}")
                continue
    
    # ─────────── Summary ───────────
    summary = {}
    for level in levels:
        summary[level] = {}
        for name in ['lq_baseline', 'stage2', 'stage3']:
            r = results[level][name]
            summary[level][name] = {
                'psnr': float(np.mean(r['psnr'])) if r['psnr'] else float('nan'),
                'ssim': float(np.mean(r['ssim'])) if r['ssim'] else float('nan'),
                'lpips': float(np.mean(r['lpips'])) if r['lpips'] else float('nan'),
                'id': float(np.mean(r['id'])) if r['id'] else float('nan'),
                'n': len(r['psnr']),
            }
    
    # Print bảng
    print_summary_table(summary)
    
    # Save CSV
    save_csv(summary, results, output_dir)
    save_latex_table(summary, output_dir)
    
    return {'summary': summary, 'per_image': results}


def save_sample(path: Path, lq, s2, s3, hq):
    """Concatenate 4 ảnh thành 1 grid (LQ | S2 | S3 | HQ) và save."""
    import torchvision.utils as vutils
    grid = torch.cat([lq, s2, s3, hq], dim=0)
    vutils.save_image(grid, str(path), nrow=4, padding=4, pad_value=1.0)


def print_summary_table(summary: Dict):
    """Print bảng tổng kết đẹp."""
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    for level, stages in summary.items():
        print(f"\n── Level: {level.upper()} ──")
        print(f"{'Method':<15} {'PSNR (↑)':<12} {'SSIM (↑)':<12} "
              f"{'LPIPS (↓)':<12} {'ID-sim (↑)':<12} {'N':<5}")
        print("-" * 70)
        for name in ['lq_baseline', 'stage2', 'stage3']:
            m = stages[name]
            n_disp = m['n']
            print(f"{name:<15} "
                  f"{m['psnr']:<12.4f} "
                  f"{m['ssim']:<12.4f} "
                  f"{m['lpips']:<12.4f} "
                  f"{m['id']:<12.4f} "
                  f"{n_disp:<5}")
        
        # Cải thiện Stage 3 vs Stage 2
        s2 = stages['stage2']
        s3 = stages['stage3']
        print(f"\nStage 3 vs Stage 2 improvement:")
        print(f"  ΔPSNR  = {s3['psnr']-s2['psnr']:+.4f} dB")
        print(f"  ΔSSIM  = {s3['ssim']-s2['ssim']:+.4f}")
        print(f"  ΔLPIPS = {s3['lpips']-s2['lpips']:+.4f}  (negative = better)")
        print(f"  ΔID    = {s3['id']-s2['id']:+.4f}")


def save_csv(summary: Dict, per_image: Dict, output_dir: str):
    """Save 2 CSV: summary + per-image."""
    import csv
    
    # Summary CSV
    summary_path = Path(output_dir) / 'summary.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['level', 'method', 'PSNR', 'SSIM', 'LPIPS', 'ID-sim', 'N'])
        for level, stages in summary.items():
            for name in ['lq_baseline', 'stage2', 'stage3']:
                m = stages[name]
                w.writerow([level, name, f"{m['psnr']:.4f}",
                            f"{m['ssim']:.4f}", f"{m['lpips']:.4f}",
                            f"{m['id']:.4f}", m['n']])
    print(f"\n✓ Summary CSV: {summary_path}")
    
    # Per-image CSV (cho phân tích sâu)
    per_path = Path(output_dir) / 'per_image.csv'
    with open(per_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['level', 'method', 'idx', 'PSNR', 'SSIM', 'LPIPS', 'ID-sim'])
        for level in per_image:
            for name in ['lq_baseline', 'stage2', 'stage3']:
                r = per_image[level][name]
                n = len(r['psnr'])
                for i in range(n):
                    w.writerow([
                        level, name, i,
                        f"{r['psnr'][i]:.4f}",
                        f"{r['ssim'][i]:.4f}",
                        f"{r['lpips'][i]:.4f}" if i < len(r['lpips']) else '',
                        f"{r['id'][i]:.4f}" if i < len(r['id']) else '',
                    ])
    print(f"✓ Per-image CSV: {per_path}")


def save_latex_table(summary: Dict, output_dir: str):
    """Tạo LaTeX table sẵn sàng paste vào Ch4."""
    path = Path(output_dir) / 'table_for_ch4.tex'
    
    lines = []
    lines.append(r"% Bảng đánh giá định lượng Stage 2 vs Stage 3")
    lines.append(r"% Auto-generated từ evaluate_pipeline.py")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{So sánh định lượng Stage 2 và Stage 3 trên CelebA-HQ test set, "
                 r"đánh giá tại hai mức suy giảm. PSNR/SSIM/ID càng cao càng tốt, LPIPS càng thấp càng tốt.}")
    lines.append(r"\label{tab:eval_stage2_vs_stage3}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Mức suy giảm & Phương pháp & PSNR $\uparrow$ & SSIM $\uparrow$ & "
                 r"LPIPS $\downarrow$ & ID-sim $\uparrow$ \\")
    lines.append(r"\midrule")
    
    for level, stages in summary.items():
        for j, name in enumerate(['lq_baseline', 'stage2', 'stage3']):
            m = stages[name]
            label = {'lq_baseline': 'LQ (input)', 'stage2': 'Stage 2',
                     'stage3': 'Stage 3'}[name]
            level_str = level.capitalize() if j == 0 else ''
            lines.append(f"{level_str} & {label} & "
                         f"{m['psnr']:.2f} & {m['ssim']:.4f} & "
                         f"{m['lpips']:.4f} & {m['id']:.4f} \\\\")
        lines.append(r"\midrule")
    
    lines[-1] = r"\bottomrule"  # Replace last midrule
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✓ LaTeX table: {path}")


if __name__ == '__main__':
    # Standalone test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2', required=True)
    parser.add_argument('--stage3', required=True)
    parser.add_argument('--gfpgan', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--output_dir', default='./eval_results')
    parser.add_argument('--n_samples', type=int, default=200)
    args = parser.parse_args()
    
    run_evaluation(
        stage2_ckpt=args.stage2, stage3_ckpt=args.stage3,
        gfpgan_pretrained=args.gfpgan, test_dir=args.test_dir,
        output_dir=args.output_dir, n_samples=args.n_samples,
    )
