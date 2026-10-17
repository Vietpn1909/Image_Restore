"""Quick diagnostic for the common 'Stage 2 worse than LQ' problem.

Run this before full evaluation. It verifies:
1. GFPGAN imports successfully.
2. Stage 2 checkpoint loads strictly into GFPGANWrapper (no Simple fallback).
3. A few qualitative samples are saved with intermediates for inspection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

from src.degradation.vietnam_pipeline import VietnamDegradationPipeline
from src.pipeline.restore_pipeline import OldPortraitRestorationPipeline
from src.utils.gfpgan_import_fix import assert_gfpgan_available
from scripts.evaluate_pipeline import psnr, ssim


def collect_images(test_dir: str, n: int):
    exts = {'.jpg', '.jpeg', '.png'}
    imgs = sorted([p for p in Path(test_dir).rglob('*') if p.suffix.lower() in exts])
    if not imgs:
        raise FileNotFoundError(f'No images found in {test_dir}')
    return imgs[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2', required=True)
    parser.add_argument('--gfpgan', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--level', default='medium')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--output_dir', default='/kaggle/working/stage2_diagnose')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert_gfpgan_available(verbose=True)
    pipe = OldPortraitRestorationPipeline(
        checkpoint_path=args.stage2,
        gfpgan_pretrained=args.gfpgan,
        device=args.device,
        allow_simple_fallback=False,
        strict_load=True,
        verbose=True,
    )

    deg = VietnamDegradationPipeline(level=args.level, seed=123)
    rows = []
    for idx, path in enumerate(collect_images(args.test_dir, args.n)):
        hq_img = Image.open(path).convert('RGB').resize((256, 256), Image.BILINEAR)
        hq_np = np.asarray(hq_img).astype(np.float32) / 255.0
        lq_np, _ = deg(hq_np.copy())

        hq = torch.from_numpy(hq_np).permute(2, 0, 1).unsqueeze(0).to(pipe.device)
        lq = torch.from_numpy(lq_np).permute(2, 0, 1).unsqueeze(0).to(pipe.device)
        with torch.no_grad():
            outs = pipe.forward_generator_lowvram(lq)
            final = outs['final']

        rows.append({
            'idx': idx,
            'file': path.name,
            'lq_psnr': psnr(lq, hq),
            's2_psnr': psnr(final, hq),
            'lq_ssim': ssim(lq, hq),
            's2_ssim': ssim(final, hq),
        })
        grid = torch.cat([lq, outs['bg_out'], outs['face_out'], final, hq], dim=0)
        vutils.save_image(grid, str(out_dir / f'{idx:03d}_lq_bg_face_s2_hq.png'), nrow=5, padding=4, pad_value=1.0)

    print('\nidx | file | LQ PSNR | S2 PSNR | LQ SSIM | S2 SSIM')
    print('-' * 78)
    for r in rows:
        print(f"{r['idx']:>3} | {r['file'][:24]:<24} | {r['lq_psnr']:>7.2f} | {r['s2_psnr']:>7.2f} | {r['lq_ssim']:>7.4f} | {r['s2_ssim']:>7.4f}")

    lq_mean = sum(r['lq_psnr'] for r in rows) / len(rows)
    s2_mean = sum(r['s2_psnr'] for r in rows) / len(rows)
    print(f'\nMean PSNR: LQ={lq_mean:.2f}, Stage2={s2_mean:.2f}')
    if s2_mean < lq_mean:
        print('⚠️ Stage 2 is still below LQ. Inspect saved grids and check checkpoint/path.')
    else:
        print('✓ Stage 2 beats LQ on this quick diagnostic.')
    print(f'Qualitative grids saved to: {out_dir}')


if __name__ == '__main__':
    main()
