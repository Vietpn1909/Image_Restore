# Stage 3 v2 Implementation Notes — Phạm vi thực thi vs Thiết kế

## Đối chiếu thiết kế (Ch3) vs thực thi v2

| Thành phần | Thiết kế Ch3 | Thực thi v2 | Trạng thái |
|------------|-------------|-------------|------------|
| Backbone state | Full unfreeze | Full unfreeze | ✅ |
| Transformer fusion | 4 blocks | 4 blocks (expansion 2→4, copy init) | ✅ |
| GAN loss | ON | LSGAN + **R1 penalty** + freeze D 500 đầu | ✅ + cải tiến |
| Dataset | Asian (UTKFace+AAF) + Real VN | **UTKFace Đông Á (~3.4k) + FFHQ subset (10k)** | ✅ Match domain shift |
| Degradation level | Hard + Real | **Hard 70% + Medium 30%** (anti-shock) | ✅ Match + cải tiến |
| Cycle loss | ON | OFF | ⚠️ Drop (cần Real data) |
| LR backbone | 5e-5 | 5e-5 | ✅ |
| Iter | 15,000 | 5,000 | ⚠️ Giảm (compute) |
| EMA | (không nói) | **ON (decay 0.999)** | + cải tiến |
| Save freq | (không nói) | **500 iter (10 candidates)** | + cải tiến |

## 5 cải tiến v2 (so với v1)

### 1. Mix source dataset (50% UTKFace Asian + 50% FFHQ subset)
**Vấn đề**: Stage 2 train trên FFHQ 70k, full unfreeze chuyển sang 3.4k UTKFace
Asian → tỷ lệ 1 sample / 30k trainable params → nguy cơ catastrophic forgetting.

**Giải pháp**: Mix 50/50 hai nguồn. FFHQ subset (~10k) làm anchor giữ prior,
UTKFace Asian (~3.4k) tạo domain shift.

**Match thiết kế**: vẫn là "Western → shift toward Asian" — tỷ lệ Asian
trong training tăng từ ~10% (FFHQ) lên ~50% (mixed). Đây là *gradual domain
shift* thay vì *abrupt switch*, vẫn đúng tinh thần "tránh shock" của Ch3.

### 2. Mix degradation curriculum (70% hard + 30% medium)
**Vấn đề**: Stage 2 train medium, Stage 3 100% hard → shock degradation.
Ch3 viết "tránh shock training khi vừa chuyển backbone vừa chuyển domain"
nhưng v1 vi phạm chính nguyên tắc này.

**Giải pháp**: 70% hard + 30% medium thay vì 100% hard.

**Match thiết kế**: Stage 3 vẫn dominate ở mức hard, chỉ thêm anchor medium
30% để tránh forgetting.

### 3. EMA (Exponential Moving Average) weights
**Vấn đề**: GAN training noisy, single-iter weights không tối ưu.

**Giải pháp**: Shadow copy của Generator weights với decay 0.999.
- Mỗi iter: `shadow = 0.999 * shadow + 0.001 * current`
- Validation và checkpoint save dùng EMA weights (swap in, validate, swap out)
- Inference cũng dùng EMA → ảnh smooth hơn

**Đây là industry standard**: Real-ESRGAN, GFPGAN, StyleGAN2, BSRGAN đều dùng.

### 4. R1 gradient penalty cho Discriminator
**Vấn đề**: Stage 2 GAN dao động sau iter 3000, D quá mạnh.

**Giải pháp**: R1 penalty = E[||∇_x D(x)||²] on real samples.
- Áp dụng lazy regularization (every 16 step) theo StyleGAN2 paper
- Weight 10.0, scale by every_n_steps để equivalent với non-lazy
- Ổn định GAN hơn LSGAN thuần, đặc biệt khi full unfreeze

### 5. Save frequency dày hơn (500 thay vì 1000)
**Vấn đề**: 5 candidates trong 5000 iter quá ít, dễ bỏ lỡ sweet spot.

**Giải pháp**: 500 iter → 10 candidates.

**Cost**: Storage 2x (10 file × 510 MB ≈ 5 GB) — vẫn fit trong working dir Kaggle.

## Tác động dự kiến

| Metric | Stage 3 v1 | Stage 3 v2 |
|--------|-----------|-----------|
| Xác suất best < 1.21 (Stage 2) | ~50% | **~80%** |
| Best val_loss dự kiến | 1.05-1.25 | **0.95-1.10** |
| Setup time | 0 | +30 phút code |
| Train time | 3.6h | ~3.8h (EMA copy +5%) |
| VRAM | base | base + 200MB (EMA + R1) |
| Save storage | ~3 GB | ~5 GB |

## Cấu hình kỹ thuật v2 (bảng cho Ch4)

| Tham số | Giá trị |
|---------|---------|
| Train set Asian | UTKFace race=2 (Đông Á) ~3.4k |
| Train set anchor | FFHQ subset 10k |
| Mix ratio sources | 50% Asian + 50% FFHQ |
| Mix ratio degradation | 70% hard + 30% medium |
| Val sets | CelebA-HQ × 2 levels (medium + hard) |
| Image size | 256 × 256 |
| Batch size | 2 |
| Iter | 5,000 |
| LR backbone | 5e-5 |
| LR deg_encoder | 2e-5 |
| LR fusion | 1e-4 |
| LR discriminator | 5e-5 |
| Warmup | 200 iter |
| Freeze D | 500 iter đầu |
| **EMA decay** | 0.999 |
| **R1 weight / freq** | 10.0 / every 16 step |
| Save every | 500 iter (10 candidates) |
| pixel L1 | 1.0 |
| perceptual VGG | 1.0 |
| identity ArcFace | 0.7 |
| adversarial LSGAN | 0.05 |
| deg supervised | 0.3 |
| Mixed precision | FP16 |

## Mẫu đoạn viết Ch4

> **4.X.1 Phạm vi thực thi Stage 3**
>
> Stage 3 bám sát thiết kế Chương 3: full unfreeze backbone, mở rộng
> Transformer fusion từ 2 lên 4 block, learning rate 5×10⁻⁵, degradation
> dominate ở mức hard. Dataset shift sang Đông Á thông qua hỗn hợp 50%
> tập con Đông Á UTKFace (lọc theo race=2 trong filename, ~3,400 ảnh)
> và 50% FFHQ subset (~10,000 ảnh) — thực hiện gradual domain shift
> để tránh catastrophic forgetting prior Stage 2.
>
> Bổ sung 3 cải tiến để ổn định GAN training và tăng chất lượng:
> (i) **EMA** weights cho Generator (decay 0.999) — industry standard
> trong Real-ESRGAN, GFPGAN, StyleGAN2;
> (ii) **R1 gradient penalty** cho Discriminator áp dụng lazy regularization
> mỗi 16 iteration với hệ số 10.0;
> (iii) Curriculum degradation 70% hard + 30% medium thay vì 100% hard,
> đúng theo nguyên tắc "tránh shock" được nêu trong thiết kế.
>
> Hai điều chỉnh so với thiết kế: (i) chưa tích hợp real VN data và
> cycle loss do giới hạn thu thập dữ liệu, đề xuất trong hướng phát
> triển; (ii) giảm iteration từ 15,000 xuống 5,000 do giới hạn compute
> T4 15GB, bù lại bằng save frequency dày hơn (500 iter, 10 candidates)
> và EMA để tận dụng tốt 5,000 iter có.
