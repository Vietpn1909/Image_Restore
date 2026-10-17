# VN Old Portrait Restoration


**Tên đề tài**: Nghiên cứu và phát triển hệ thống phục chế ảnh chân dung cũ dựa trên học sâu, kết hợp Real-ESRGAN, GFPGAN và mô-đun tăng cường khuôn mặt dựa trên Transformer với chiến lược fine-tuning đa giai đoạn và học thích nghi suy giảm.



---

## Tổng quan hệ thống

Pipeline **hai nhánh + Transformer fusion**:

```
        Old Portrait (input)
              │
        ┌─────┴─────┐
        ▼           ▼
  Real-ESRGAN   Face Detect
   (BG branch)   & GFPGAN
        │           │
        └─────┬─────┘
              ▼
     Transformer Fusion
     (Cross-Attention)
              │
              ▼
      Restored Portrait
```

Toàn pipeline điều khiển bởi **Degradation-Aware Encoder** dự đoán mức suy giảm
và inject vào generator qua FiLM.

---

## Bốn đóng góp nghiên cứu

| # | Đóng góp | Code |
|---|----------|------|
| 1 | Mô hình hóa suy giảm theo miền VN (blur, scratch, stain, mold, fade, yellowing) | `src/degradation/vietnam_pipeline.py` |
| 2 | Fine-tuning đa giai đoạn (3 stage: warmup → medium → hard+real) | `src/training/trainer_stage{1,2,3}.py` |
| 3 | Degradation-aware encoder + FiLM modulation | `src/models/degradation_encoder.py`, `src/models/film_layer.py` |
| 4 | Transformer cross-attention fusion (face × background) | `src/models/transformer_fusion.py` |

---

## Chiến lược dữ liệu

| Loại | Nguồn | Số lượng | Dùng cho |
|------|-------|----------|----------|
| HQ training | Kaggle dataset (FFHQ + Asian face HQ) — attach 1 click | 10-15k | Stage 1, 2, 3 |
| LQ training | Online sinh từ HQ qua `vietnam_pipeline.py` | (đồng bộ HQ) | Stage 1, 2, 3 |
| Real VN old | gallery-dl manhhai + scan gia đình | 100-300 | Stage 3 (30%) + test |
| Test synthetic | CelebA-HQ test set + degradation | 500-2000 | Eval PSNR/SSIM/LPIPS |

**Không crawl Unsplash/Pexels/Pixabay/Flickr API** — chỉ dùng dataset có sẵn trên Kaggle + gallery-dl 1 lần cho ảnh thật VN.

---

## Cấu trúc project

```
vn-old-portrait-restoration/
├── configs/                 # YAML config 3 stage + inference
├── data/
│   ├── hq_train/            # FFHQ + Asian (attach trên Kaggle)
│   ├── real_vn_old/         # Ảnh cũ VN thật
│   ├── test_synthetic/      # CelebA-HQ test
│   └── degradation_assets/  # Scratch/stain overlay (optional)
├── src/
│   ├── degradation/         # Đóng góp 1
│   ├── models/              # Đóng góp 3, 4 + Real-ESRGAN, GFPGAN
│   ├── losses/              # L1, VGG, ArcFace, Adaptive, Adversarial
│   ├── training/            # Đóng góp 2 (3 stage)
│   ├── pipeline/            # Inference end-to-end
│   ├── evaluation/          # PSNR/SSIM/LPIPS/FID/NIQE/MUSIQ
│   └── utils/
├── scripts/
│   ├── train.py                  # CLI training cho 3 stage
│   ├── inference.py              # CLI inference end-to-end
│   ├── evaluate_pipeline.py      # 4 metric: PSNR/SSIM/LPIPS/ID-sim
│   ├── ablation_A_baseline.py    # Ablation A (mức sàn)
│   ├── ablation_B_film.py        # Ablation B (no Fusion)
│   ├── verify_stage2_on_val.py   # Debug Stage 2
│   ├── verify_forward_paths.py   # Debug forward path
│   ├── diagnose_stage2_pipeline.py
│   ├── filter_vn_real.py         # Lọc ảnh VN real (future work)
│   ├── download_pretrained.sh    # Tải Real-ESRGAN + GFPGAN
│   └── download_vn_real.sh       # Tải VN real data (future work)
├── notebooks/               # Kaggle notebooks (train + eval + SOTA)
├── checkpoints/             # Pretrained + 3 stage checkpoints
├── results/                 # Output, log, comparisons
└── docs/                    # Notes, design decisions, experiments log
```

---

## Quick start

```bash
# 1. Cài deps
pip install -r requirements.txt

# 2. Download pretrained weights (Real-ESRGAN + GFPGAN)
bash scripts/download_pretrained.sh

# 3. Train (3 stage)
python scripts/train.py --config configs/stage1_warmup.yaml
python scripts/train.py --config configs/stage2_medium.yaml
python scripts/train.py --config configs/stage3_hard.yaml

# 4. Inference end-to-end (pipeline v7 auto-detect forward path)
python scripts/inference.py --input old_photo.jpg --output restored.jpg
```

### Notebook trên Kaggle

- `notebooks/02_train_stage1.ipynb` — Train Stage 1
- `notebooks/doantrainstage3_setup.ipynb` — Train Stage 3 (full unfreeze + 5 cải tiến)
- `notebooks/evaluation_setup.ipynb` — Đánh giá định lượng 4 metric
- `notebooks/sota_comparison.ipynb` — So sánh với SOTA

---

## Đánh giá

### Định lượng (đã chạy 200 ảnh CelebA-HQ test)

| Level | LQ | Stage 2 | **Stage 3 (best)** |
|-------|----|---------|---------------------|
| PSNR medium | 26.82 | 27.24 | **28.74** |
| LPIPS medium | 0.320 | 0.114 | **0.099** |
| ID-sim medium | 0.893 | 0.942 | **0.952** |
| PSNR hard | 22.48 | 23.90 | **26.23** |
| LPIPS hard | 0.462 | 0.202 | **0.149** |
| ID-sim hard | 0.722 | 0.849 | **0.877** |

Chi tiết: xem `docs/experiments_log.md`.

```bash
# Đánh giá định lượng (PSNR, SSIM, LPIPS, ID-sim)
python scripts/evaluate_pipeline.py \
    --stage2 path/to/stage2_best.pth \
    --stage3 path/to/stage3_best.pth \
    --gfpgan path/to/GFPGANv1.4.pth \
    --test_dir path/to/celeba_hq \
    --out results/eval/

# Ablation A baseline (Real-ESRGAN + GFPGAN gốc + Gaussian mask)
python scripts/ablation_A_baseline.py \
    --esrgan path/to/RealESRGAN_x4plus.pth \
    --gfpgan path/to/GFPGANv1.4.pth \
    --test_dir path/to/celeba_hq \
    --out results/ablation_A/

# Ablation B (Stage 2 weights, no Fusion → đo riêng đóng góp Fusion)
python scripts/ablation_B_film.py \
    --stage2 path/to/stage2_best.pth \
    --gfpgan path/to/GFPGANv1.4.pth \
    --test_dir path/to/celeba_hq \
    --out results/ablation_B/
```

### Debug & verify

```bash
# Verify Stage 2 ckpt trên val set gốc (debug forward path mismatch)
python scripts/verify_stage2_on_val.py --stage2 ... --gfpgan ... --celeba ...

# Test cả 2 forward path (skip vs full upsample)
python scripts/verify_forward_paths.py --stage2 ... --gfpgan ... --celeba ...
```

### Roadmap đánh giá

- [x] **Synthetic** (có GT): PSNR, SSIM, LPIPS, ArcFace ID-sim
- [x] **Ablation 4 biến thể**: A baseline → B (+FiLM) → C (Stage 2) → D (Stage 3)
- [ ] **So sánh SOTA**: GFPGAN v1.4, CodeFormer trên cùng test set
- [ ] **Real VN** (không GT): NIQE, MUSIQ — sau khi crawl Flickr/manhhai
- [ ] **User study**: blind A/B test 15–20 người

