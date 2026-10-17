# Data — Hướng dẫn lấy dữ liệu huấn luyện và đánh giá

Thư mục này KHÔNG chứa file ảnh trực tiếp (do giới hạn dung lượng zip nộp).
Người dùng tự tải các dataset công khai và đặt vào đúng vị trí theo cấu trúc dưới.

## Cấu trúc thư mục đầy đủ sau khi tải

```
data/
├── hq_train/                       # Ảnh HQ huấn luyện (~50k+ ảnh)
│   ├── ffhq/                       # FFHQ 70k, resize 256×256
│   ├── celeba_hq/                  # CelebA-HQ 30k, resize 256×256
│   └── utkface_asian/              # UTKFace race=2 (Đông Á), ~3.4k ảnh
│
├── test_synthetic/                 # Ảnh test với suy giảm tổng hợp
│   └── celeba_hq_val/              # 200 ảnh val CelebA-HQ
│
├── real_vn_old/                    # Ảnh chân dung VN thật (tùy chọn)
│   └── (crawl từ Flickr manhhai)
│
└── degradation_assets/             # Assets cho Vietnam Degradation Pipeline
    ├── scratches/                  # ~20 ảnh PNG vết xước (alpha mask)
    ├── molds/                      # ~15 ảnh PNG vết mốc
    └── stains/                     # ~15 ảnh PNG vết ố
```

## Dataset chính dùng trong đồ án

### 1. FFHQ — Flickr-Faces-HQ

- **Vai trò**: Dataset HQ chính cho huấn luyện Stage 2 và Stage 3 (50% mix ở Stage 3).
- **Kích thước**: 70,000 ảnh chân dung chất lượng cao đa dạng độ tuổi và sắc tộc.
- **Resolution**: $1024{\times}1024$ gốc, đồ án resize về $256{\times}256$.
- **Nguồn tải**:
  - Kaggle: `xhlulu/flickrfaceshq-dataset-nvidia-resized-256px` (đã resize 256, ~1.5 GB)
  - Hoặc trang gốc: https://github.com/NVlabs/ffhq-dataset
- **Đặt vào**: `data/hq_train/ffhq/`

### 2. CelebA-HQ

- **Vai trò**: Dataset val (200 ảnh) và test (200 ảnh) cho đánh giá định lượng.
  Cũng dùng làm tập huấn luyện phụ ở Stage 2.
- **Kích thước**: 30,000 ảnh khuôn mặt celeb.
- **Resolution**: $256{\times}256$.
- **Nguồn tải**: Kaggle `badasstechie/celebahq-resized-256x256` (~700 MB).
- **Đặt vào**:
  - 200 ảnh đầu tiên (chia theo seed cố định trong code): `data/test_synthetic/celeba_hq_val/`
  - Phần còn lại: `data/hq_train/celeba_hq/`

### 3. UTKFace — sub-set Đông Á

- **Vai trò**: Dịch chuyển miền sang khuôn mặt Đông Á ở Stage 3 (50% mix).
- **Kích thước**: ~20,000 ảnh tổng. Sau khi lọc nhãn `race=2` (Asian) còn ~3,434 ảnh.
- **Resolution**: $200{\times}200$ gốc, resize lên $256{\times}256$.
- **Nguồn tải**: Kaggle `jangedoo/utkface-new` (~330 MB).
- **Xử lý**:
  ```bash
  python scripts/filter_vn_real.py \
      --src_dir /path/to/utkface-new \
      --dst_dir data/hq_train/utkface_asian \
      --race 2
  ```

### 4. Ảnh chân dung Việt Nam thật (tùy chọn, hướng phát triển tương lai)

- **Nguồn tham khảo**:
  - Flickr collection `manhhai`: https://www.flickr.com/photos/13476480@N07/
  - Tạp chí cũ được scan công khai trên thư viện số Việt Nam.
  - Album gia đình (cần xin phép, không công khai).
- **Đặt vào**: `data/real_vn_old/`
- **Lưu ý**: Đồ án chính KHÔNG huấn luyện trên dữ liệu này (do chưa có ground truth
  đủ và quy mô lớn). Chỉ dùng cho qualitative demonstration ở phần phụ lục.

## Vietnam Degradation Pipeline assets

Pipeline tổng hợp suy giảm cần 3 nhóm overlay PNG để mô phỏng vết xước, mốc và ố.

- **scratches/** — 20 ảnh PNG với kênh alpha biểu diễn vết xước/vết gấp giấy.
- **molds/** — 15 ảnh PNG biểu diễn vết mốc loang.
- **stains/** — 15 ảnh PNG biểu diễn vết ố loang.

Các assets này được em tự thu thập và xử lý từ một số nguồn ảnh stock miễn phí
(Unsplash, Pexels) và scan trực tiếp từ một số ảnh giấy cũ trong gia đình em.
Toàn bộ nằm trong dataset Kaggle `mrtaintou/vn-degradation-assets` để tiện chia sẻ.

```bash
kaggle datasets download mrtaintou/vn-degradation-assets -p data/degradation_assets/
cd data/degradation_assets && unzip vn-degradation-assets.zip
```

## Quy trình chuẩn để chuẩn bị dữ liệu (tóm tắt)

```bash
# 1. Tạo cấu trúc thư mục
mkdir -p data/{hq_train/{ffhq,celeba_hq,utkface_asian},test_synthetic,real_vn_old,degradation_assets}

# 2. Cài kaggle CLI và đặt API token
pip install kaggle  # ~/.kaggle/kaggle.json

# 3. Tải các dataset
kaggle datasets download xhlulu/flickrfaceshq-dataset-nvidia-resized-256px -p data/hq_train/ffhq/
kaggle datasets download badasstechie/celebahq-resized-256x256 -p data/hq_train/celeba_hq/
kaggle datasets download jangedoo/utkface-new -p data/hq_train/utkface_asian/
kaggle datasets download mrtaintou/vn-degradation-assets -p data/degradation_assets/

# 4. Giải nén
find data -name "*.zip" -exec unzip -q {} \;

# 5. Lọc UTKFace lấy nhánh Đông Á (race=2)
python scripts/filter_vn_real.py \
    --src_dir data/hq_train/utkface_asian/utkface_aligned_cropped \
    --dst_dir data/hq_train/utkface_asian/asian_only \
    --race 2
```

Sau bước này, hệ thống đã sẵn sàng để chạy huấn luyện từ đầu. Tổng dung lượng
dữ liệu khoảng **3.5 GB** sau giải nén.

## Ghi chú về bản quyền

Tất cả các dataset trên đều có giấy phép cho phép sử dụng cho mục đích nghiên cứu
và giáo dục phi thương mại. Trích dẫn đầy đủ các nguồn này được liệt kê trong
`Danh_sach_tai_lieu_tham_khao.bib` của quyển báo cáo.

Riêng các ảnh chân dung Việt Nam thật trong `real_vn_old/`, nếu là tư liệu gia đình
cá nhân thì cần được giữ riêng tư và không công khai trong quá trình demonstration.
