# Stage 2 Package — Files mới cần thêm vào project

## File trong zip này

1. `src/models/real_esrgan/rrdb_with_film.py` — Real-ESRGAN backbone với FiLM injection
2. `src/models/gfpgan/gfpgan_wrapper.py` — GFPGAN wrapper với pre/post FiLM
3. `src/models/discriminator.py` — PatchGAN + UNet Discriminator
4. `src/training/trainer_stage2.py` — Stage 2 Trainer đầy đủ
5. `src/models/real_esrgan/__init__.py` — empty (để Python nhận folder là package)
6. `src/models/gfpgan/__init__.py` — empty

## Cách cài đặt

Giải nén zip này lên trên project `vn-old-portrait-restoration/` cũ:
```
unzip stage2_package.zip
# Sẽ tạo các file vào đúng folder src/models/, src/training/
```

Sau đó upload lại project (đã có thêm Stage 2 files) thành Kaggle Dataset mới:
`vn-old-portrait-code-v5`

## Sau khi upload code-v5

1. Tạo notebook Stage 2 mới (hoặc dùng `doanTrainStage2` hiện tại).
2. Detach dataset `vn-old-portrait-code-v4` cũ.
3. Attach dataset `vn-old-portrait-code-v5` mới.
4. Chạy các cell theo notebook setup.

## Kiến trúc Stage 2

```
LQ (256x256)
   │
   ├─→ Degradation Encoder (load từ Stage 1)
   │       │
   │     deg_vec (8 dim)
   │       │
   ├──┬────┴──────────────────┐
   │  │                       │
   │  ▼                       ▼
   │  Real-ESRGAN+FiLM       GFPGAN+FiLM
   │  (16.73M)                (85M, freeze)
   │  unfreeze last 4 RRDB    pre/post FiLM
   │  output x4               output 512x512
   │  ↓ downsample            ↓ resize → 256
   │  BG (256x256)            Face (256x256)
   │           │              │
   │           ▼              ▼
   │      Transformer Fusion (cross-attention)
   │           │
   │           ▼
   │      Final restored (256x256)
   │           │
   ▼           ▼
   Discriminator → GAN Loss
```

## Tham số

- BG branch: 16.73M total, ~3M trainable (unfreeze 4 RRDB cuối + FiLM)
- Face branch: ~85M total (GFPGAN), ~50k trainable (pre/post FiLM only)
- Discriminator: ~4.4M (UNet style)
- Fusion: ~0.15M

Tổng trainable: ~7.5M params, hợp lý cho T4 15GB.
