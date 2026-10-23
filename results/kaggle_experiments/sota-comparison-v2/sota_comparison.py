"""
SOTA Comparison: Stage 3 (ours) vs GFPGAN v1.4 vs CodeFormer.

Đánh giá fair trên cùng 200 ảnh CelebA-HQ test, cùng Vietnam Degradation Pipeline
(2 mức medium + hard). Cả 3 method đều output 256x256 để so sánh đồng nhất.

Các đường ống so sánh:
    - LQ baseline   : ảnh LQ sau degradation (không restore)
    - GFPGAN v1.4   : load arch GFPGANv1Clean + checkpoint pretrained,
                      input 256 -> upsample 512 -> network -> downsample 256
    - CodeFormer    : load arch CodeFormer + checkpoint pretrained,
                      has_aligned mode (CelebA-HQ đã aligned), w=0.5,
                      input 256 -> upsample 512 -> network -> downsample 256
    - Stage 3 ours  : pipeline đầy đủ (RealESRGAN+FiLM + GFPGAN+FiLM + Fusion)

Output:
    - summary.csv               : bảng tổng kết
    - per_image.csv             : metric từng ảnh
    - sota_table.tex            : LaTeX table sẵn paste vào Ch5
    - samples_sota/*.png        : qualitative grid (LQ|GFPGAN|CodeFormer|Stage3|HQ)

Cách dùng (trong Kaggle notebook):
    from scripts.sota_comparison import run_sota_comparison

    results = run_sota_comparison(
        stage3_ckpt=STAGE3_CKPT,
        gfpgan_pretrained=GFPGAN_PRE,
        codeformer_ckpt=CODEFORMER_CKPT,
        codeformer_repo=CODEFORMER_REPO,
        test_dir=CELEBA,
        output_dir='/kaggle/working/sota_results',
        n_samples=200,
        levels=['medium', 'hard'],
    )
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Reuse các thành phần đã có trong evaluate_pipeline.py để đảm bảo metric đồng nhất
from scripts.evaluate_pipeline import (
    psnr, ssim, LPIPSMetric, IDSimMetric, collect_test_images,
)
from src.pipeline.restore_pipeline import OldPortraitRestorationPipeline
from src.utils.gfpgan_import_fix import assert_gfpgan_available
from src.degradation.vietnam_pipeline import VietnamDegradationPipeline


# ──────────────────────────────────────────────────────────────────────────────
#                        GFPGAN v1.4 baseline (pretrained gốc)
# ──────────────────────────────────────────────────────────────────────────────
class GFPGANBaseline:
    """GFPGAN v1.4 chạy network trực tiếp trên ảnh đã aligned.

    Bypass FaceRestoreHelper vì CelebA-HQ đã face-aligned sẵn — không cần
    detect/crop/paste lại (tránh sai số do face detector miss + face fusion
    boundary artifact đề cập trong note tác giả CodeFormer).

    Input/output đều ở range [0, 1], normalize bên trong sang [-1, 1].
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

        # Kiến trúc theo gfpgan/utils.py default cho GFPGANv1.4
        self.net = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True,
        )

        loadnet = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'params_ema' in loadnet:
            sd = loadnet['params_ema']
        elif 'params' in loadnet:
            sd = loadnet['params']
        else:
            sd = loadnet
        self.net.load_state_dict(sd, strict=True)
        self.net.eval().to(device)
        self.device = device

    @torch.no_grad()
    def restore_256(self, lq_t: torch.Tensor) -> torch.Tensor:
        """LQ (1,3,256,256) [0,1] -> restored (1,3,256,256) [0,1]."""
        # Upsample sang 512 cho network input
        lq_512 = F.interpolate(lq_t, size=(512, 512),
                                mode='bilinear', align_corners=False)
        # Normalize [0,1] -> [-1,1]
        lq_512 = lq_512 * 2.0 - 1.0
        out, _ = self.net(lq_512, return_rgb=False, weight=0.5)
        # out (1, 3, 512, 512) range [-1, 1]
        out = (out + 1.0) / 2.0
        # Downsample về 256 để so sánh đồng nhất với HQ 256
        out_256 = F.interpolate(out, size=(256, 256),
                                 mode='bilinear', align_corners=False)
        return out_256.clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
#                        CodeFormer baseline (pretrained gốc)
# ──────────────────────────────────────────────────────────────────────────────
class CodeFormerBaseline:
    """CodeFormer chạy network trực tiếp với has_aligned=True.

    Theo note tác giả: "If you want to compare CodeFormer in your paper,
    please run the following command indicating --has_aligned (for cropped
    and aligned face), as the command for the whole image will involve a
    process of face-background fusion that may damage hair texture on the
    boundary, which leads to unfair comparison."

    => Dùng has_aligned mode, bypass FaceRestoreHelper.
    """

    def __init__(self, model_path: str, codeformer_repo_path: str,
                 device: str = 'cuda', fidelity_weight: float = 0.5):
        # System basicsr 1.4.2 (cài qua pip) KHÔNG có CodeFormer arch.
        # CodeFormer arch chỉ có trong basicsr fork ship cùng CodeFormer repo.
        # → Phải force pick up basicsr từ CodeFormer/basicsr/ thay vì system.
        #
        # Stage 3 + GFPGAN đã load XONG model lên GPU trước cell này,
        # nên clear sys.modules['basicsr'] không ảnh hưởng inference Stage 3.

        # 0. Tạo stub basicsr/version.py.
        #    CodeFormer's basicsr/__init__.py import `.version` (gitsha + version)
        #    File này được tạo bởi `python basicsr/setup.py develop` — ta skip
        #    step đó vì setup develop có thể override system packages.
        #    → Tạo stub thủ công.
        version_file = os.path.join(codeformer_repo_path, 'basicsr', 'version.py')
        if not os.path.exists(version_file):
            with open(version_file, 'w') as f:
                f.write(
                    "# Stub generated by sota_comparison.py — bypass setup.py develop\n"
                    "__version__ = '1.3.5'\n"
                    "__gitsha__ = 'unknown'\n"
                    "version_info = (1, 3, 5)\n"
                )
            print(f'  [CodeFormer] Created stub: {version_file}')

        # 1. Xóa cached basicsr modules
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('basicsr'):
                del sys.modules[mod_name]

        # 2. Insert CodeFormer repo lên ĐẦU sys.path để import basicsr pick up fork
        if codeformer_repo_path in sys.path:
            sys.path.remove(codeformer_repo_path)
        sys.path.insert(0, codeformer_repo_path)

        # 3. Import basicsr.archs để trigger auto-registration tất cả *_arch.py
        #    (gồm codeformer_arch.py + vqgan_arch.py)
        from basicsr.utils.registry import ARCH_REGISTRY
        import basicsr.archs  # noqa: F401 — chính import này trigger register

        # 4. Fallback nếu vẫn không tìm thấy
        try:
            CodeFormerCls = ARCH_REGISTRY.get('CodeFormer')
        except KeyError:
            from basicsr.archs.codeformer_arch import CodeFormer as CodeFormerCls

        self.net = CodeFormerCls(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=['32', '64', '128', '256'],
        ).to(device)

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'params_ema' in ckpt:
            sd = ckpt['params_ema']
        elif 'params' in ckpt:
            sd = ckpt['params']
        else:
            sd = ckpt
        self.net.load_state_dict(sd, strict=True)
        self.net.eval()
        self.w = fidelity_weight
        self.device = device

    @torch.no_grad()
    def restore_256(self, lq_t: torch.Tensor) -> torch.Tensor:
        """LQ (1,3,256,256) [0,1] -> restored (1,3,256,256) [0,1]."""
        # Upsample sang 512 cho network input
        lq_512 = F.interpolate(lq_t, size=(512, 512),
                                mode='bilinear', align_corners=False)
        # Normalize [0,1] -> [-1,1]
        lq_n = lq_512 * 2.0 - 1.0
        # CodeFormer forward: (output, _, _) hoặc tuple đầu là output
        out = self.net(lq_n, w=self.w, adain=True)[0]
        # out range [-1, 1]
        out = (out + 1.0) / 2.0
        out_256 = F.interpolate(out, size=(256, 256),
                                 mode='bilinear', align_corners=False)
        return out_256.clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
#                              Main runner
# ──────────────────────────────────────────────────────────────────────────────
def run_sota_comparison(
    stage3_ckpt: str,
    gfpgan_pretrained: str,
    codeformer_ckpt: str,
    codeformer_repo: str,
    test_dir: str,
    output_dir: str = '/kaggle/working/sota_results',
    n_samples: int = 200,
    levels: List[str] = ['medium', 'hard'],
    image_size: int = 256,
    save_samples: int = 10,
    codeformer_w: float = 0.5,
    device: str = 'cuda',
    seed: int = 42,
) -> Dict:
    """Chạy SOTA comparison đầy đủ.

    Args:
        stage3_ckpt        : checkpoint Stage 3 best.pth (mô hình của ta)
        gfpgan_pretrained  : GFPGANv1.4.pth (cùng file dùng trong Stage 3)
        codeformer_ckpt    : codeformer.pth từ GitHub release v0.1.0
        codeformer_repo    : path đến CodeFormer repo (đã clone)
        test_dir           : folder ảnh HQ test (CelebA-HQ)
        output_dir         : nơi lưu kết quả
        n_samples          : số ảnh test (đồng bộ với evaluate_pipeline = 200)
        levels             : list mức degradation
        image_size         : 256 (đồng bộ với evaluate_pipeline)
        save_samples       : số ảnh save qualitative grid 5-column
        codeformer_w       : fidelity weight CodeFormer (0.5 cân bằng chất lượng/danh tính)
        device             : cuda

    Returns:
        dict {'summary': ..., 'per_image': ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = Path(output_dir) / 'samples_sota'
    samples_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print("SOTA COMPARISON — Stage 3 (ours) vs GFPGAN v1.4 vs CodeFormer")
    print("=" * 78)
    assert_gfpgan_available(verbose=True)

    # ─── Load 3 method ───
    print("\n[1/5] Loading Stage 3 pipeline (ours)...")
    pipe_s3 = OldPortraitRestorationPipeline(
        checkpoint_path=stage3_ckpt,
        gfpgan_pretrained=gfpgan_pretrained,
        device=device,
        image_size=image_size,
        allow_simple_fallback=False,
        strict_load=True,
    )

    print("\n[2/5] Loading GFPGAN v1.4 baseline (pretrained gốc)...")
    gfpgan_base = GFPGANBaseline(model_path=gfpgan_pretrained, device=device)

    print("\n[3/5] Loading CodeFormer baseline (pretrained gốc, w={})...".format(codeformer_w))
    codeformer_base = CodeFormerBaseline(
        model_path=codeformer_ckpt,
        codeformer_repo_path=codeformer_repo,
        device=device,
        fidelity_weight=codeformer_w,
    )

    # ─── Init metrics ───
    print("\n[4/5] Initializing metrics...")
    lpips_metric = LPIPSMetric(device=device)
    id_metric = IDSimMetric(device=device)
    print(f"  LPIPS: {'ON' if lpips_metric.ok else 'OFF'}")
    print(f"  ID-sim: {'ON' if id_metric.ok else 'OFF'}")

    # ─── Collect test images ───
    print(f"\n[5/5] Collecting {n_samples} test images...")
    test_imgs = collect_test_images(test_dir, n_samples=n_samples, seed=seed)

    # ─── Eval loop ───
    methods = ['lq', 'gfpgan_v14', 'codeformer', 'stage3_ours']
    results: Dict = {
        level: {m: {'psnr': [], 'ssim': [], 'lpips': [], 'id': []}
                for m in methods}
        for level in levels
    }

    print("\n" + "─" * 78)
    print("Running evaluation...")
    print("─" * 78)

    for level in levels:
        print(f"\n── Level: {level} ──")
        deg = VietnamDegradationPipeline(level=level)

        for i, img_path in enumerate(test_imgs):
            try:
                # Load HQ
                hq_img = Image.open(img_path).convert('RGB').resize(
                    (image_size, image_size), Image.BILINEAR)
                hq_np = np.array(hq_img).astype(np.float32) / 255.0

                # Apply degradation (seed deterministic theo ảnh để tái lập)
                lq_np, _ = deg(hq_np.copy())

                # To tensor
                hq_t = torch.from_numpy(hq_np).permute(2, 0, 1).unsqueeze(0).to(device)
                lq_t = torch.from_numpy(lq_np).permute(2, 0, 1).unsqueeze(0).to(device)

                # Run 3 method
                s3_out = pipe_s3.restore_256(lq_t)
                gf_out = gfpgan_base.restore_256(lq_t)
                cf_out = codeformer_base.restore_256(lq_t)

                # Compute metric cho cả 4 (gồm LQ baseline)
                for name, pred in [
                    ('lq', lq_t),
                    ('gfpgan_v14', gf_out),
                    ('codeformer', cf_out),
                    ('stage3_ours', s3_out),
                ]:
                    results[level][name]['psnr'].append(psnr(pred, hq_t))
                    results[level][name]['ssim'].append(ssim(pred, hq_t))
                    if lpips_metric.ok:
                        results[level][name]['lpips'].append(lpips_metric(pred, hq_t))
                    if id_metric.ok:
                        results[level][name]['id'].append(id_metric(pred, hq_t))

                # Save qualitative grid 5-column
                if i < save_samples:
                    save_sample_5col(
                        samples_dir / f'{level}_{i:03d}.png',
                        lq=lq_t, gf=gf_out, cf=cf_out, s3=s3_out, hq=hq_t,
                    )

                if (i + 1) % 20 == 0:
                    print(f"  [{level}] {i+1}/{len(test_imgs)}")

            except Exception as e:
                print(f"  ✗ {img_path.name}: {e}")
                continue

    # ─── Summary ───
    summary = {}
    for level in levels:
        summary[level] = {}
        for m in methods:
            r = results[level][m]
            summary[level][m] = {
                'psnr': float(np.mean(r['psnr'])) if r['psnr'] else float('nan'),
                'ssim': float(np.mean(r['ssim'])) if r['ssim'] else float('nan'),
                'lpips': float(np.mean(r['lpips'])) if r['lpips'] else float('nan'),
                'id': float(np.mean(r['id'])) if r['id'] else float('nan'),
                'n': len(r['psnr']),
            }

    # Print + save
    print_summary_table(summary)
    save_csv(summary, results, output_dir)
    save_latex_table(summary, output_dir, codeformer_w=codeformer_w)

    return {'summary': summary, 'per_image': results}


# ──────────────────────────────────────────────────────────────────────────────
#                              IO helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_sample_5col(path: Path, lq, gf, cf, s3, hq):
    """Lưu grid 5-column: LQ | GFPGAN | CodeFormer | Stage3 | HQ."""
    import torchvision.utils as vutils
    grid = torch.cat([lq, gf, cf, s3, hq], dim=0)
    vutils.save_image(grid, str(path), nrow=5, padding=4, pad_value=1.0)


def print_summary_table(summary: Dict):
    """In bảng tổng kết SOTA."""
    print("\n" + "=" * 110)
    print("SOTA COMPARISON SUMMARY")
    print("=" * 110)

    label_map = {
        'lq': 'LQ (input)',
        'gfpgan_v14': 'GFPGAN v1.4',
        'codeformer': 'CodeFormer',
        'stage3_ours': 'Stage 3 (ours)',
    }

    for level, stages in summary.items():
        print(f"\n── Level: {level.upper()} ──")
        print(f"{'Method':<18} {'PSNR (↑)':<12} {'SSIM (↑)':<12} "
              f"{'LPIPS (↓)':<12} {'ID-sim (↑)':<12} {'N':<5}")
        print("-" * 75)
        for m in ['lq', 'gfpgan_v14', 'codeformer', 'stage3_ours']:
            d = stages[m]
            print(f"{label_map[m]:<18} "
                  f"{d['psnr']:<12.4f} "
                  f"{d['ssim']:<12.4f} "
                  f"{d['lpips']:<12.4f} "
                  f"{d['id']:<12.4f} "
                  f"{d['n']:<5}")

        # Stage 3 vs GFPGAN / CodeFormer
        s3 = stages['stage3_ours']
        gf = stages['gfpgan_v14']
        cf = stages['codeformer']
        print(f"\nStage 3 (ours) vs GFPGAN v1.4 (pretrained):")
        print(f"  ΔPSNR  = {s3['psnr']-gf['psnr']:+.4f} dB")
        print(f"  ΔSSIM  = {s3['ssim']-gf['ssim']:+.4f}")
        print(f"  ΔLPIPS = {s3['lpips']-gf['lpips']:+.4f} (negative = better)")
        print(f"  ΔID    = {s3['id']-gf['id']:+.4f}")

        print(f"\nStage 3 (ours) vs CodeFormer (pretrained):")
        print(f"  ΔPSNR  = {s3['psnr']-cf['psnr']:+.4f} dB")
        print(f"  ΔSSIM  = {s3['ssim']-cf['ssim']:+.4f}")
        print(f"  ΔLPIPS = {s3['lpips']-cf['lpips']:+.4f} (negative = better)")
        print(f"  ΔID    = {s3['id']-cf['id']:+.4f}")


def save_csv(summary: Dict, per_image: Dict, output_dir: str):
    """Save summary CSV + per-image CSV."""
    import csv

    summary_path = Path(output_dir) / 'summary_sota.csv'
    methods = ['lq', 'gfpgan_v14', 'codeformer', 'stage3_ours']
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['level', 'method', 'PSNR', 'SSIM', 'LPIPS', 'ID-sim', 'N'])
        for level, stages in summary.items():
            for m in methods:
                d = stages[m]
                w.writerow([level, m, f"{d['psnr']:.4f}",
                            f"{d['ssim']:.4f}", f"{d['lpips']:.4f}",
                            f"{d['id']:.4f}", d['n']])
    print(f"\n✓ Summary CSV: {summary_path}")

    per_path = Path(output_dir) / 'per_image_sota.csv'
    with open(per_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['level', 'method', 'idx', 'PSNR', 'SSIM', 'LPIPS', 'ID-sim'])
        for level in per_image:
            for m in methods:
                r = per_image[level][m]
                n = len(r['psnr'])
                for i in range(n):
                    w.writerow([
                        level, m, i,
                        f"{r['psnr'][i]:.4f}",
                        f"{r['ssim'][i]:.4f}",
                        f"{r['lpips'][i]:.4f}" if i < len(r['lpips']) else '',
                        f"{r['id'][i]:.4f}" if i < len(r['id']) else '',
                    ])
    print(f"✓ Per-image CSV: {per_path}")


def save_latex_table(summary: Dict, output_dir: str, codeformer_w: float = 0.5):
    """Tạo LaTeX table SOTA so sánh sẵn paste vào Ch5."""
    path = Path(output_dir) / 'sota_table.tex'

    label_map = {
        'lq': 'LQ (đầu vào)',
        'gfpgan_v14': 'GFPGAN v1.4',
        'codeformer': f'CodeFormer ($w={codeformer_w}$)',
        'stage3_ours': '\\textbf{Stage 3 (đề xuất)}',
    }

    lines = []
    lines.append(r"% So sánh SOTA — auto-generated từ sota_comparison.py")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{So sánh phương pháp đề xuất (Stage 3) với hai phương pháp SOTA "
                 r"GFPGAN v1.4 và CodeFormer trên 200 ảnh CelebA-HQ test, "
                 r"đánh giá tại hai mức suy giảm theo Vietnam Degradation Pipeline. "
                 r"Tất cả output đã đưa về kích thước $256{\times}256$ để so sánh đồng nhất. "
                 r"PSNR/SSIM/ID-sim càng cao càng tốt, LPIPS càng thấp càng tốt.}")
    lines.append(r"\label{tab:sota_comparison}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Mức suy giảm & Phương pháp & PSNR $\uparrow$ & SSIM $\uparrow$ & "
                 r"LPIPS $\downarrow$ & ID-sim $\uparrow$ \\")
    lines.append(r"\midrule")

    methods = ['lq', 'gfpgan_v14', 'codeformer', 'stage3_ours']
    for level, stages in summary.items():
        for j, m in enumerate(methods):
            d = stages[m]
            level_str = level.capitalize() if j == 0 else ''
            lines.append(f"{level_str} & {label_map[m]} & "
                         f"{d['psnr']:.2f} & {d['ssim']:.4f} & "
                         f"{d['lpips']:.4f} & {d['id']:.4f} \\\\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✓ LaTeX table: {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage3', required=True)
    parser.add_argument('--gfpgan', required=True)
    parser.add_argument('--codeformer', required=True)
    parser.add_argument('--codeformer_repo', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--output_dir', default='./sota_results')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--codeformer_w', type=float, default=0.5)
    args = parser.parse_args()

    run_sota_comparison(
        stage3_ckpt=args.stage3,
        gfpgan_pretrained=args.gfpgan,
        codeformer_ckpt=args.codeformer,
        codeformer_repo=args.codeformer_repo,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        codeformer_w=args.codeformer_w,
    )
