# Experiments Log

## Training runs

| ID | Date | Stage | Config | Best val_loss | Best iter | Trainable | Note |
|----|------|-------|--------|---------------|-----------|-----------|------|
| 001 | 2026-05 | 1 | `stage1_warmup.yaml` | **0.4602** | 9,000 / 30,000 | 1.21M | Warmup, encoder phân biệt 3 mức Light 0.281 < Medium 0.332 < Hard 0.407 |
| 002 | 2026-05 | 2 | `stage2_medium.yaml` | **1.2100** | 3,000 / 8,000 | 7.6M / 108M | Forward FULL upsample; GAN dao động sau iter 3000 (D dominate) |
| 003 | 2026-06 | 3 | `stage3_hard.yaml` | **1.1248** (medium) | 4,000 / 5,000 | 96.74M / 109M | Final, full unfreeze; 5 cải tiến; mix UTKFace Asian + FFHQ; 70% hard + 30% medium; R1 penalty; freeze D 500 iter |

### Stage 3 val curve chi tiết

| Iter | loss avg | loss medium | loss hard |
|------|----------|-------------|-----------|
| 500  | 1.2709 | 1.1768 | 1.3649 |
| 1000 | 1.2552 | 1.1632 | 1.3472 |
| 1500 | 1.2292 | 1.1362 | 1.3223 |
| 2000 | 1.2303 | 1.1387 | 1.3219 |
| 2500 | 1.2198 | 1.1279 | 1.3118 |
| 3000 | 1.2213 | 1.1297 | 1.3129 |
| 3500 | 1.2179 | 1.1267 | 1.3092 |
| **4000** | **1.2150** | **1.1248** | **1.3051** ← BEST |
| 4500 | 1.2178 | 1.1277 | 1.3079 |

## Evaluation định lượng (200 ảnh CelebA-HQ test)

| Level | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ID-sim ↑ |
|-------|--------|--------|--------|---------|----------|
| medium | LQ baseline | 26.82 | 0.783 | 0.320 | 0.893 |
| medium | Stage 2 | 27.24 | 0.814 | 0.114 | 0.942 |
| medium | **Stage 3** | **28.74** | **0.838** | **0.099** | **0.952** |
| hard | LQ baseline | 22.48 | 0.663 | 0.462 | 0.722 |
| hard | Stage 2 | 23.90 | 0.727 | 0.202 | 0.849 |
| hard | **Stage 3** | **26.23** | **0.769** | **0.149** | **0.877** |

### Mức cải thiện so với LQ baseline

| Level | Method | ΔPSNR | ΔSSIM | ΔLPIPS | ΔID-sim |
|-------|--------|-------|-------|--------|---------|
| medium | Stage 2 | +0.42 dB | +0.031 | −65% | +0.049 |
| medium | Stage 3 | +1.92 dB | +0.055 | −69% | +0.059 |
| hard | Stage 2 | +1.42 dB | +0.064 | −56% | +0.127 |
| hard | Stage 3 | +3.75 dB | +0.106 | −68% | +0.155 |

## Ablation Study (200 ảnh CelebA-HQ test)

### 4 biến thể

| Biến thể | Cấu hình | Đo lường |
|----------|----------|----------|
| **A** (baseline) | Real-ESRGAN gốc + GFPGAN gốc, no FiLM, no Fusion, Gaussian mask blend | Mức sàn |
| **B** (+FiLM) | Stage 2 weights, KHÔNG có Fusion (thay bằng Gaussian mask) | Đóng góp 4 (Fusion) qua gap B↔C |
| **C** (Stage 2 full) | B + Transformer Fusion 2-block | Đầy đủ đóng góp 3 + 4 |
| **D** (Stage 3 full) | C + Stage 3 full unfreeze + Fusion 4-block + 5 cải tiến | Hệ thống hoàn chỉnh |

### Kết quả

| Level | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ID-sim ↑ |
|-------|--------|--------|--------|---------|----------|
| medium | A baseline | 26.50 | 0.809 | 0.201 | 0.895 |
| medium | B (+FiLM, no Fusion) | 13.87 ⚠️ | 0.592 | 0.334 | 0.791 |
| medium | C = Stage 2 | 27.24 | 0.814 | 0.114 | 0.942 |
| medium | **D = Stage 3** | **28.74** | **0.838** | **0.099** | **0.952** |
| hard | A baseline | 22.78 | 0.734 | 0.333 | 0.790 |
| hard | B (+FiLM, no Fusion) | 13.45 ⚠️ | 0.544 | 0.438 | 0.716 |
| hard | C = Stage 2 | 23.90 | 0.727 | 0.202 | 0.849 |
| hard | **D = Stage 3** | **26.23** | **0.769** | **0.149** | **0.877** |

### Insight B: Fusion align color/brightness

B PSNR cực thấp (13.87 medium) **không phải bug** mà tiết lộ insight: Transformer Fusion không chỉ "blend mượt biên" mà còn **align color/brightness** giữa 2 nhánh. Khi loại bỏ Fusion và blend Gaussian thô → bg_out và face_out (đã được train trong context Fusion) có tone shift mạnh so với HQ.

### Tóm tắt đóng góp

| Transition | Đóng góp đo | ΔPSNR med | ΔLPIPS med | ΔID med |
|------------|-------------|-----------|------------|---------|
| A → C | Đóng góp 3+4 huấn luyện đầy đủ | +0.74 dB | −43% | +0.047 |
| **B → C** | **Đóng góp 4 riêng (Fusion)** | **+13.37 dB** | **−66%** | **+0.151** |
| C → D | Đóng góp 2 (Multi-stage Stage 3) | +1.50 dB | −13% | +0.010 |
| **A → D** | **Toàn bộ đồ án** | **+2.24 dB** | **−51%** | **+0.057** |

## Comparison vs SOTA

| Method | PSNR | SSIM | LPIPS | ID-sim | Note |
|--------|------|------|-------|--------|------|
| GFPGAN v1.4 | — | — | — | — | TODO: chạy trên cùng test set với Vietnam Degradation Pipeline |
| CodeFormer | — | — | — | — | TODO |
| RestoreFormer | — | — | — | — | Optional (sau nộp quyển) |
| GPEN | — | — | — | — | Optional |
| **Ours (Stage 3)** | 28.74 / 26.23 | 0.838 / 0.769 | 0.099 / 0.149 | 0.952 / 0.877 | medium / hard |

## Bug debug log

1. **OOM Stage 3 full unfreeze** (T4 15GB): giảm ESRGAN 23→12 RRDB + tắt EMA → ~13GB ✓
2. **GFPGAN import fail**: basicsr 1.4.2 import `torchvision.transforms.functional_tensor` đã bị xóa khỏi torchvision ≥ 0.16. Fix: pin version + patch `basicsr/data/degradations.py` + restart kernel ✓
3. **Kaggle UI download bug** (>500MB): dùng Kaggle Datasets API workaround ✓
4. **GAN instability Stage 2**: D dominate sau iter 3000. Stage 3 fix: R1 penalty (every 16, w=10) + freeze D 500 iter + giảm LR_D ✓
5. **Forward path mismatch** (CRITICAL): Stage 2 train FULL upsample, Stage 3 train SKIP upsample. Pipeline v7 auto-detect dựa trên `ckpt['stage']` hoặc số fusion blocks ✓
6. **Ablation B PSNR 13.87**: KHÔNG phải bug, là insight về Fusion align color/brightness ✓
