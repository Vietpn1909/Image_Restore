# Design Decisions

> Mỗi quyết định thiết kế quan trọng được ghi vào đây.
> Format: **Quyết định** → **Lựa chọn cân nhắc** → **Lý do** → **Trade-off**.

---

## DD-01: Pipeline hai nhánh (face + background)

**Quyết định**: Pipeline gồm 2 nhánh — Real-ESRGAN (background), GFPGAN (face) — kết hợp qua Transformer fusion.

**Lựa chọn**: Single network end-to-end vs. **pipeline hai nhánh**.

**Lý do**: Face có cấu trúc semantic cố định (mắt-mũi-miệng), cần prior chuyên biệt (StyleGAN2 trong GFPGAN). Background đa dạng, cần generalist (Real-ESRGAN). Tách 2 nhánh cho phép fine-tune độc lập.

**Trade-off**: Pipeline phức tạp hơn, nhiều moving parts. → giải quyết bằng đóng góp 4 (Transformer fusion).

---

## DD-02: Restormer block thay vì Swin / ViT

**Quyết định**: MDTA (Multi-Dconv head Transposed Attention) từ Restormer (CVPR 2022).

**Lựa chọn**: ViT / Swin / **Restormer MDTA** / Uformer.

**Lý do**: ViT có O((HW)²) → không scale với 512×512. Swin window-based, khó cross-attention. Restormer MDTA linear theo pixel (O(C²·HW)), đã chứng minh hiệu quả image restoration.

**Trade-off**: Channel-wise attention thay vì spatial. Bù lại bằng depth-wise conv trong block.

---

## DD-03: Degradation vector 8 chiều

**Quyết định**: 8 chiều `[blur, noise, jpeg, downsample, scratch, stain, fade, yellowing]`.

**Lựa chọn**: 4 / **8** / 16+.

**Lý do**: 4 bỏ sót VN-specific. 16+ khó học, dễ overfit. 8 cân bằng.

---

## DD-04: FiLM modulation

**Quyết định**: Inject degradation vector qua FiLM (Feature-wise Linear Modulation).

**Lựa chọn**: Concat / **FiLM** / SPADE / Cross-attention.

**Lý do**: FiLM nhẹ (2 linear), hiệu quả đã chứng minh cho conditional generation. SPADE phụ thuộc spatial — không cần với degradation đồng nhất toàn ảnh.

---

## DD-05: 3 stage fine-tune

**Quyết định**: warmup (light) → medium → hard + real data.

**Lựa chọn**: 2 / **3** / 4 stage.

**Lý do**: 2 stage nhảy độ khó quá lớn. 4 stage marginal benefit không rõ. 3 stage có vai trò rõ ràng: align thành phần mới / unfreeze backbone / ground vào dữ liệu thật.

---

## DD-06: ArcFace identity loss

**Quyết định**: Pretrained ArcFace (qua facexlib) cho identity loss.

**Lựa chọn**: FaceNet / **ArcFace** / CosFace.

**Lý do**: SOTA face recognition, available, GFPGAN paper gốc cũng dùng → consistent baseline.

---

## DD-07: Data strategy không crawl Unsplash/Pexels/Pixabay/Flickr API

**Quyết định**: Chỉ dùng Kaggle dataset có sẵn (FFHQ, Asian face) + gallery-dl 1 lần cho ảnh VN thật. Không crawl các nguồn cần API key.

**Lựa chọn**:
1. Crawl 4 nguồn (Unsplash, Pexels, Pixabay, Flickr) → mất ~2 ngày, phải đăng ký 4 keys.
2. **Chỉ Kaggle + gallery-dl** → 0 phút đăng ký, < 1 đêm chạy.

**Lý do**:
- Kaggle dataset có sẵn FFHQ 70k + nhiều Asian face dataset, đủ dùng cho train.
- Flickr blocked account VN, các nguồn còn lại marginal benefit.
- Tập trung thời gian vào code & experiment thay vì data engineering.

**Trade-off**: Ảnh VN modern HQ giảm. → bù bằng tăng tỉ lệ Asian dataset trong stage 2-3.
