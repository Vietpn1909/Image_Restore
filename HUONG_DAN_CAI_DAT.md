# Hướng dẫn cài đặt và chạy demo

Tài liệu này hướng dẫn từng bước cài đặt môi trường, tải mô hình và chạy demo
phục chế ảnh chân dung cũ trên máy cá nhân.

## 1. Yêu cầu hệ thống

### Phần cứng tối thiểu

- **CPU**: Intel Core i5 (gen 8+) hoặc tương đương.
- **RAM**: 8 GB.
- **Ổ cứng**: 5 GB trống cho mô hình và môi trường.
- **GPU (tùy chọn nhưng khuyến nghị)**: NVIDIA với $\geq 4$ GB VRAM, hỗ trợ CUDA 11.8.

### Phần cứng khuyến nghị

- **GPU**: NVIDIA T4 / RTX 2060 / RTX 3060 trở lên, $\geq 8$ GB VRAM.

### Phần mềm

- **Hệ điều hành**: Windows 10/11, Ubuntu 20.04+, macOS 12+.
- **Python**: 3.10 hoặc 3.11.
- **CUDA Toolkit** (nếu có GPU NVIDIA): 11.8 hoặc 12.1.

## 2. Cài đặt môi trường


### Bước 2 — Tạo môi trường ảo Python

**Với conda** (khuyến nghị):

```bash
conda create -n vnrestore python=3.10 -y
conda activate vnrestore
```

**Với venv**:

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### Bước 3 — Cài PyTorch

**Có GPU NVIDIA** (CUDA 11.8):

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

**Chỉ CPU** (chạy chậm hơn nhiều, không khuyến nghị cho demo):

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
```

### Bước 4 — Cài các package phụ thuộc còn lại

```bash
pip install -r requirements.txt
```

Đợi khoảng 5 phút cho các package cài hoàn tất.

### Bước 5 — Vá lỗi import của basicsr (chỉ thực hiện một lần)

Phiên bản `basicsr 1.4.2` import một module đã bị xóa khỏi `torchvision >= 0.16`,
chạy đoạn lệnh sau để vá:

```bash
python -c "
import glob, os
for f in glob.glob('**/basicsr/data/degradations.py', recursive=True):
    with open(f) as fp:
        c = fp.read()
    if 'torchvision.transforms.functional_tensor' in c:
        with open(f, 'w') as fp:
            fp.write(c.replace('torchvision.transforms.functional_tensor', 'torchvision.transforms.functional'))
        print(f'Patched {f}')
"
```

## 3. Tải mô hình đã huấn luyện

Đọc chi tiết hướng dẫn tải trong `checkpoints/README.md`. Tóm tắt:

```bash
# Cài Kaggle CLI và đặt API token vào ~/.kaggle/kaggle.json
pip install kaggle

# Tải pretrained backbones + Stage 3 final (cần thiết cho demo)
kaggle datasets download mrtaintou/vn-pretrained-model-gan-v1 -p checkpoints/pretrained/
kaggle datasets download mrtaintou/vn-stage3-final-completed   -p checkpoints/stage3/

cd checkpoints/pretrained && unzip -o vn-pretrained-model-gan-v1.zip && cd ../..
cd checkpoints/stage3     && unzip -o vn-stage3-final-completed.zip && cd ../..
```

Sau khi tải, thư mục `checkpoints/` phải có:

```
checkpoints/
├── pretrained/RealESRGAN_x4plus.pth
├── pretrained/GFPGANv1.4.pth
└── stage3/best.pth
```

## 4. Chạy demo

### Cách 1 — Demo qua giao diện web Gradio (khuyến nghị)

```bash
python scripts/demo_gradio.py
```

Sau khi khởi động, mở trình duyệt và truy cập `http://localhost:7860`. Giao diện
cho phép:

1. Tải ảnh chân dung cũ cần phục chế (kéo thả hoặc bấm chọn).
2. Tùy chọn mức suy giảm dự đoán (None / Light / Medium / Hard --- để mô hình
   tự xác định hoặc người dùng gợi ý cho mô hình).
3. Bấm "Phục chế" và đợi kết quả (1--5 giây tùy phần cứng).
4. So sánh trực quan ảnh gốc và ảnh sau phục chế bên cạnh nhau.
5. Tải ảnh đã phục chế về máy.

### Cách 2 — Demo qua dòng lệnh (cho người dùng kỹ thuật)

```bash
python scripts/inference.py \
    --input  duong_dan/anh_cua_ban.jpg \
    --output ket_qua/anh_phuc_che.jpg \
    --ckpt   checkpoints/stage3/best.pth
```

### Cách 3 — Đánh giá định lượng trên tập test

```bash
python scripts/evaluate_pipeline.py \
    --ckpt checkpoints/stage3/best.pth \
    --test_dir data/test_synthetic/celeba_hq_val \
    --output_dir results/eval_stage3
```

Kết quả gồm bảng metric (PSNR, SSIM, LPIPS, ID-similarity) và bảng LaTeX được
ghi vào `results/eval_stage3/`.

## 5. Huấn luyện lại từ đầu (tùy chọn)

Cần GPU $\geq 15$ GB VRAM (Kaggle T4 là phù hợp nhất).

```bash
# Stage 1 (warmup Degradation Encoder, ~30 phút trên T4)
python scripts/train.py --config configs/stage1_warmup.yaml

# Stage 2 (medium fine-tuning, ~3 giờ trên T4)
python scripts/train.py --config configs/stage2_medium.yaml

# Stage 3 (hard fine-tuning + domain shift, ~2 giờ trên T4)
python scripts/train.py --config configs/stage3_hard.yaml
```

Chi tiết về cấu hình huấn luyện đầy đủ và các bước tiền xử lý dữ liệu xem trong
`data/README.md` và `docs/experiments_log.md`.

## 6. Khắc phục sự cố thường gặp

**Lỗi `CUDA out of memory`**: giảm batch size trong file config (`stage*_*.yaml`),
hoặc dùng forward path `skip_upsample`, hoặc chuyển sang chế độ CPU.

**Lỗi import `basicsr.data.degradations`**: chưa thực hiện bước vá ở Mục 2.5,
quay lại thực hiện rồi restart Python kernel.

**Lỗi không tải được Kaggle dataset**: kiểm tra file `~/.kaggle/kaggle.json` đã
đặt đúng vị trí và có quyền 600 (`chmod 600 ~/.kaggle/kaggle.json` trên Linux).

**Kết quả phục chế bị artifact lạ**: thử dùng `forward_full_upsample` thay vì
`skip_upsample` (chỉnh trong `configs/inference.yaml`), hoặc kiểm tra ảnh đầu vào
đã được căn khuôn mặt vào giữa chưa.

**Demo Gradio không mở được**: kiểm tra port 7860 không bị chiếm bởi tiến trình
khác (`netstat -ano | findstr :7860` trên Windows). Có thể đổi port bằng tham số
`--port 7870` khi chạy `demo_gradio.py`.

