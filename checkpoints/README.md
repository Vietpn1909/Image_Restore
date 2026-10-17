# Checkpoints — Hướng dẫn lấy trọng số mô hình

Thư mục này KHÔNG chứa file `.pth` trực tiếp .
Người dùng tải checkpoint từ Kaggle theo hướng dẫn dưới đây và đặt vào đúng vị trí.

## Cấu trúc thư mục đầy đủ sau khi tải

```
checkpoints/
├── pretrained/
│   ├── RealESRGAN_x4plus.pth      # ~64 MB
│   └── GFPGANv1.4.pth             # ~340 MB
├── stage1/
│   └── best.pth                   # ~5 MB (chỉ Degradation Encoder)
├── stage2/
│   └── best.pth                   # ~509 MB (full pipeline, full upsample)
└── stage3/
    ├── best.pth                   # ~437 MB (full pipeline, skip upsample)
    ├── iter_4000.pth              # ~437 MB (checkpoint trung gian)
    └── final.pth                  # ~437 MB (iteration cuối)
```

**Tổng dung lượng**: ~2.6 GB.

## Cách tải

### Tải qua giao diện web Kaggle (đơn giản)

Truy cập từng dataset bên dưới, bấm nút **Download** ở góc trên phải, giải nén
và copy file vào đúng thư mục con tương ứng:

| Dataset Kaggle | Đặt vào | Nội dung |
|----------------|---------|----------|
| `mrtaintou/vn-pretrained-model-gan-v1` | `checkpoints/pretrained/` | RealESRGAN_x4plus.pth + GFPGANv1.4.pth |
| `mrtaintou/stage1-degradation-encoder` | `checkpoints/stage1/` | best.pth |
| `mrtaintou/stage2-checkpoint-v1` | `checkpoints/stage2/` | best.pth |
| `mrtaintou/vn-stage3-final-completed` | `checkpoints/stage3/` | best.pth + iter_4000.pth + final.pth |

### Tải qua Kaggle API (khuyến nghị cho người dùng kỹ thuật)

```bash
# Cài kaggle CLI và đặt API token (~/.kaggle/kaggle.json)
pip install kaggle

# Trong thư mục gốc project, chạy:
bash scripts/download_pretrained.sh
```

Script sẽ tải tất cả 4 dataset trên và sắp xếp vào đúng thư mục.

## Khi nào dùng checkpoint nào

| Mục đích | Checkpoint cần |
|----------|----------------|
| **Inference** trên ảnh đầu vào (demo, đánh giá cuối) | `stage3/best.pth` |
| **Tiếp tục huấn luyện** từ Stage 3 | `stage3/iter_4000.pth` hoặc `stage3/final.pth` |
| **So sánh ablation** với pipeline trung gian | `stage2/best.pth` |
| **Chỉ chạy Degradation Encoder** | `stage1/best.pth` |
| **Khởi tạo huấn luyện từ pretrained gốc** | `pretrained/*.pth` |

## Lưu ý về forward path khi load checkpoint

Stage 2 và Stage 3 được huấn luyện với hai forward path khác nhau, pipeline
inference đã được thiết kế để tự động phát hiện và lựa chọn đúng forward path
tương ứng dựa trên metadata của checkpoint.

- **Stage 2 checkpoint** → tự động dùng `forward_full_upsample` (BG branch
  Real-ESRGAN x4 lên $1024{\times}1024$ rồi downsample về $256$; Face branch
  StyleGAN2 ở $512$ rồi downsample về $256$).
- **Stage 3 checkpoint** → tự động dùng `forward_skip_upsample` (BG branch
  skip conv\_up1/conv\_up2, trực tiếp $256{\times}256$; Face branch vẫn $512$).

Người dùng không cần làm gì thêm — pipeline tự xử lý.

## Tham khảo cross-check

- Mã MD5 của checkpoint, log val_loss và thông tin cấu hình huấn luyện đầy đủ
  được ghi trong `docs/experiments_log.md`.
- File cấu hình tương ứng từng stage nằm trong `configs/`.
