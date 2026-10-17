"""
Demo Gradio cho hệ thống phục chế ảnh chân dung cũ Việt Nam.

Hai tab chính:
    1. Phục chế ảnh: chọn giữa Stage 2, Stage 3, GFPGAN v1.4, Real-ESRGAN
    2. Tạo ảnh LQ từ HQ (Vietnam Degradation): mô phỏng suy giảm và tải về

Cách chạy:
    python scripts/demo_gradio.py                    # local
    python scripts/demo_gradio.py --share            # public URL 24h
    python scripts/demo_gradio.py --port 7870        # đổi port

Yêu cầu:
    - checkpoints/stage3/best.pth (khuyến nghị)
    - checkpoints/stage2/best.pth (tùy chọn, cho tab so sánh giai đoạn)
    - checkpoints/pretrained/RealESRGAN_x4plus.pth
    - checkpoints/pretrained/GFPGANv1.4.pth
    - pip install gradio
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

# Đảm bảo có thể import src/ từ thư mục gốc project
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.restore_pipeline import OldPortraitRestorationPipeline
from src.utils.gfpgan_import_fix import assert_gfpgan_available


# ══════════════════════════════════════════════════════════════════════════════
#                           CONFIG PATHS + DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_STAGE3_CKPT = ROOT / "checkpoints" / "stage3" / "best.pth"
DEFAULT_STAGE2_CKPT = ROOT / "checkpoints" / "stage2" / "best.pth"
DEFAULT_GFPGAN_PATH = ROOT / "checkpoints" / "pretrained" / "GFPGANv1.4.pth"
DEFAULT_REALESR_PATH = ROOT / "checkpoints" / "pretrained" / "RealESRGAN_x4plus.pth"

METHOD_CHOICES = [
    ("Stage 3 — Đề xuất (khuyến nghị)", "stage3"),
    ("Stage 2 — Trung gian", "stage2"),
    ("GFPGAN v1.4 — Pretrained gốc", "gfpgan"),
    ("Real-ESRGAN x4 — Pretrained gốc", "realesrgan"),
]

LEVEL_CHOICES = [
    ("Nhẹ (light) — 2-3 loại suy giảm nhẹ", "light"),
    ("Trung bình (medium) — 4-5 loại suy giảm vừa phải", "medium"),
    ("Nặng (hard) — 8 loại suy giảm mạnh (ảnh cũ tồi tệ)", "hard"),
]


# ══════════════════════════════════════════════════════════════════════════════
#                       CLASS 1: MULTI-METHOD RESTORER
# ══════════════════════════════════════════════════════════════════════════════

class MultiMethodRestorer:
    """Wrapper cho 4 phương pháp phục chế, lazy-load để tiết kiệm VRAM.

    Model chỉ được load khi user chọn phương pháp lần đầu. Sau đó cache lại.
    """

    def __init__(self, device: str = "cuda", stage3_ckpt: Optional[Path] = None,
                 stage2_ckpt: Optional[Path] = None,
                 gfpgan_path: Optional[Path] = None,
                 realesr_path: Optional[Path] = None):
        self.device = device
        self._stage3_ckpt = stage3_ckpt or DEFAULT_STAGE3_CKPT
        self._stage2_ckpt = stage2_ckpt or DEFAULT_STAGE2_CKPT
        self._gfpgan_path = gfpgan_path or DEFAULT_GFPGAN_PATH
        self._realesr_path = realesr_path or DEFAULT_REALESR_PATH
        self._models = {}  # cache

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_stage(self, stage: str):
        """Load Stage 2 hoặc Stage 3 pipeline (dùng chung class)."""
        if stage in self._models:
            return self._models[stage]

        ckpt = self._stage3_ckpt if stage == "stage3" else self._stage2_ckpt
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint {stage} tại {ckpt}. "
                f"Tải theo hướng dẫn trong checkpoints/README.md."
            )

        print(f"[Restorer] Loading {stage} từ {ckpt}...")
        assert_gfpgan_available(verbose=False)
        pipe = OldPortraitRestorationPipeline(
            checkpoint_path=str(ckpt),
            gfpgan_pretrained=str(self._gfpgan_path),
            device=self.device,
            image_size=256,
            allow_simple_fallback=False,
            strict_load=True,
        )
        self._models[stage] = pipe
        print(f"[Restorer] {stage} loaded xong.")
        return pipe

    def _load_gfpgan(self):
        """Load GFPGAN v1.4 gốc (không có FiLM, không có degradation encoder)."""
        if "gfpgan" in self._models:
            return self._models["gfpgan"]

        if not self._gfpgan_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy GFPGAN pretrained tại {self._gfpgan_path}."
            )

        print(f"[Restorer] Loading GFPGAN v1.4 từ {self._gfpgan_path}...")
        from gfpgan import GFPGANer
        gfpgan = GFPGANer(
            model_path=str(self._gfpgan_path),
            upscale=1,          # giữ nguyên độ phân giải
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,  # không dùng bg upsampler
            device=self.device,
        )
        self._models["gfpgan"] = gfpgan
        print(f"[Restorer] GFPGAN loaded xong.")
        return gfpgan

    def _load_realesrgan(self):
        """Load Real-ESRGAN x4 gốc — chỉ upscale, không phục chế artefacts."""
        if "realesrgan" in self._models:
            return self._models["realesrgan"]

        if not self._realesr_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy Real-ESRGAN pretrained tại {self._realesr_path}."
            )

        print(f"[Restorer] Loading Real-ESRGAN x4 từ {self._realesr_path}...")
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(self._realesr_path),
            model=model,
            tile=0,       # không tile, ảnh nhỏ
            tile_pad=10,
            pre_pad=0,
            half=False,   # False để tránh precision issues
            device=self.device,
        )
        self._models["realesrgan"] = upsampler
        print(f"[Restorer] Real-ESRGAN loaded xong.")
        return upsampler

    # ------------------------------------------------------------------
    # Restore dispatcher
    # ------------------------------------------------------------------

    def restore(self, img_np: np.ndarray, method: str) -> tuple[np.ndarray, str]:
        """Phục chế 1 ảnh với method đã chọn.

        Args:
            img_np: ảnh RGB numpy (H, W, 3) uint8.
            method: 'stage3', 'stage2', 'gfpgan', hoặc 'realesrgan'.

        Returns:
            (ảnh phục chế uint8, thông tin string)
        """
        H, W = img_np.shape[:2]

        # Stage 2 và Stage 3 — pipeline chung
        if method in ("stage3", "stage2"):
            pipe = self._load_stage(method)

            # Resize về 256, forward, resize về gốc
            img_256 = np.array(
                Image.fromarray(img_np).convert("RGB").resize((256, 256), Image.BILINEAR)
            ).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_256).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = pipe.restore_256(tensor)

            out_np = out.squeeze(0).cpu().permute(1, 2, 0).numpy()
            out_np = (out_np * 255).clip(0, 255).astype(np.uint8)
            out_pil = Image.fromarray(out_np).resize((W, H), Image.BICUBIC)

            info = (
                f"✓ Phục chế xong bằng {method.upper()}. "
                f"Kích thước gốc: {W}×{H}. Xử lý ở 256×256 rồi upscale về gốc."
            )
            return np.array(out_pil), info

        # GFPGAN v1.4 gốc — dùng FaceRestoreHelper của GFPGAN
        elif method == "gfpgan":
            gfpgan = self._load_gfpgan()

            # GFPGAN nhận BGR uint8, cần convert từ RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            _, _, restored = gfpgan.enhance(
                img_bgr, has_aligned=False, only_center_face=True, paste_back=True
            )
            if restored is None:
                info = "⚠ GFPGAN không phát hiện được khuôn mặt trong ảnh."
                return img_np, info

            out_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
            info = (
                f"✓ Phục chế xong bằng GFPGAN v1.4. "
                f"GFPGAN chỉ tập trung vào khuôn mặt, phần nền giữ nguyên."
            )
            return out_rgb, info

        # Real-ESRGAN x4 — upscale toàn ảnh
        elif method == "realesrgan":
            upsampler = self._load_realesrgan()

            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            output, _ = upsampler.enhance(img_bgr, outscale=4)
            # Downsample về kích thước gốc để so sánh công bằng
            output = cv2.resize(output, (W, H), interpolation=cv2.INTER_LANCZOS4)
            out_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            info = (
                f"✓ Xử lý xong bằng Real-ESRGAN x4. "
                f"Lưu ý: Real-ESRGAN chỉ super-resolution tổng quát, không chuyên "
                f"về khuôn mặt cũng không xử lý xước/mốc/ố vàng."
            )
            return out_rgb, info

        else:
            raise ValueError(f"Method không hợp lệ: {method}")


# ══════════════════════════════════════════════════════════════════════════════
#                    CLASS 2: DEGRADATION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class DegradationSimulator:
    """Tạo ảnh LQ từ ảnh HQ bằng Vietnam Degradation Pipeline.

    Ưu tiên dùng pipeline chuẩn trong src/degradation. Nếu import lỗi, dùng
    fallback lightweight bằng OpenCV + numpy trực tiếp.
    """

    def __init__(self):
        self.pipeline = None
        self.PipelineClass = None
        self.LevelEnum = None
        try:
            from src.degradation.vietnam_pipeline import (
                VietnamDegradationPipeline, DegradationLevel
            )
            self.PipelineClass = VietnamDegradationPipeline
            self.LevelEnum = DegradationLevel
            print("[Degradation] Dùng VietnamDegradationPipeline chính thức.")
        except Exception as e:
            print(f"[Degradation] Không import được VietnamDegradationPipeline ({e}), "
                  f"dùng fallback lightweight.")

    def degrade(self, img_np: np.ndarray, level: str,
                seed: Optional[int] = None) -> tuple[np.ndarray, str]:
        """Tạo ảnh LQ từ ảnh HQ.

        Args:
            img_np: ảnh RGB numpy (H, W, 3) uint8.
            level: 'light' / 'medium' / 'hard'.
            seed: hạt ngẫu nhiên (tùy chọn) để reproduce kết quả.

        Returns:
            (ảnh LQ uint8, thông tin string)
        """
        if seed is not None:
            np.random.seed(seed)

        # Nếu có pipeline chính thức thì gọi
        if self.PipelineClass is not None:
            try:
                overlay_dir = ROOT / "data" / "degradation_assets"
                pipeline = self.PipelineClass(
                    level=self.LevelEnum(level),
                    overlay_dir=str(overlay_dir) if overlay_dir.exists() else None,
                    seed=seed,
                )
                lq, deg_vec = pipeline(img_np)
                # Format vector 8-D để hiển thị
                labels = ["blur", "noise", "JPEG", "downsample",
                          "scratch", "stain", "fade", "yellow"]
                vec_str = ", ".join([f"{l}={v:.2f}" for l, v in zip(labels, deg_vec)])
                info = (
                    f"✓ Tạo LQ mức '{level}' bằng VietnamDegradationPipeline.\n"
                    f"Vector 8-D dự đoán được bởi degradation encoder: {vec_str}"
                )
                return lq, info
            except Exception as e:
                print(f"[Degradation] Pipeline lỗi ({e}), fallback lightweight.")

        # Fallback: implement thủ công
        return self._fallback_degrade(img_np, level)

    # ------------------------------------------------------------------
    # Fallback: 8 loại suy giảm implement thủ công
    # ------------------------------------------------------------------

    def _fallback_degrade(self, img_np: np.ndarray, level: str) -> tuple[np.ndarray, str]:
        """Implement 8 loại suy giảm bằng OpenCV + numpy."""
        arr = img_np.astype(np.float32) / 255.0
        H, W = arr.shape[:2]

        applied = []

        # Config theo mức
        cfg = {
            "light":  {"blur_k": 3, "noise": 0.02, "jpeg_q": 85, "downsample": 1.0,
                       "n_scratch": 0, "n_stain": 0, "fade": 0.0, "yellow": 0.05,
                       "n_types": 3},
            "medium": {"blur_k": 5, "noise": 0.05, "jpeg_q": 60, "downsample": 0.7,
                       "n_scratch": 3, "n_stain": 2, "fade": 0.15, "yellow": 0.15,
                       "n_types": 5},
            "hard":   {"blur_k": 9, "noise": 0.10, "jpeg_q": 25, "downsample": 0.5,
                       "n_scratch": 8, "n_stain": 4, "fade": 0.30, "yellow": 0.30,
                       "n_types": 8},
        }[level]

        # 1. Blur Gauss
        if cfg["blur_k"] > 1:
            k = cfg["blur_k"] | 1  # đảm bảo lẻ
            arr = cv2.GaussianBlur(arr, (k, k), 0)
            applied.append("blur")

        # 2. Downsample + upsample (mô phỏng mất chi tiết)
        if cfg["downsample"] < 1.0:
            new_w, new_h = int(W * cfg["downsample"]), int(H * cfg["downsample"])
            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
            applied.append("downsample")

        # 3. Noise Gauss
        if cfg["noise"] > 0:
            arr = arr + np.random.randn(*arr.shape).astype(np.float32) * cfg["noise"]
            arr = np.clip(arr, 0, 1)
            applied.append("noise")

        # 4. JPEG compression
        if cfg["jpeg_q"] < 95:
            arr_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
            arr_bgr = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode(".jpg", arr_bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg_q"]])
            arr_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            arr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            applied.append("JPEG")

        # 5. Scratch (vẽ line thẳng ngẫu nhiên)
        if cfg["n_scratch"] > 0:
            arr_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
            for _ in range(cfg["n_scratch"]):
                x1, y1 = np.random.randint(0, W), np.random.randint(0, H)
                x2, y2 = np.random.randint(0, W), np.random.randint(0, H)
                thickness = np.random.randint(1, 3)
                color_val = np.random.randint(200, 255)
                cv2.line(arr_u8, (x1, y1), (x2, y2),
                         (color_val, color_val, color_val), thickness)
            arr = arr_u8.astype(np.float32) / 255.0
            applied.append("scratch")

        # 6. Stain (vẽ hình tròn mờ ngẫu nhiên)
        if cfg["n_stain"] > 0:
            arr_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
            overlay = arr_u8.copy()
            for _ in range(cfg["n_stain"]):
                cx, cy = np.random.randint(0, W), np.random.randint(0, H)
                r = np.random.randint(10, 40)
                # Màu ố vàng-nâu
                color = (np.random.randint(100, 180),
                         np.random.randint(80, 140),
                         np.random.randint(40, 90))
                cv2.circle(overlay, (cx, cy), r, color, -1)
            arr_u8 = cv2.addWeighted(arr_u8, 0.75, overlay, 0.25, 0)
            # Blur nhẹ để trông tự nhiên
            arr_u8 = cv2.GaussianBlur(arr_u8, (5, 5), 0)
            arr = arr_u8.astype(np.float32) / 255.0
            applied.append("stain")

        # 7. Fade (giảm contrast)
        if cfg["fade"] > 0:
            mean = arr.mean()
            arr = arr * (1 - cfg["fade"]) + mean * cfg["fade"]
            applied.append("fade")

        # 8. Yellowing (dịch màu về vàng)
        if cfg["yellow"] > 0:
            # Tăng R+G, giảm B
            arr[..., 0] = np.clip(arr[..., 0] + cfg["yellow"] * 0.3, 0, 1)
            arr[..., 1] = np.clip(arr[..., 1] + cfg["yellow"] * 0.2, 0, 1)
            arr[..., 2] = np.clip(arr[..., 2] - cfg["yellow"] * 0.5, 0, 1)
            applied.append("yellowing")

        out = (arr * 255).clip(0, 255).astype(np.uint8)
        info = (
            f"✓ Tạo ảnh LQ mức '{level}' (fallback lightweight). "
            f"Áp dụng {len(applied)} loại suy giảm: {', '.join(applied)}."
        )
        return out, info


# ══════════════════════════════════════════════════════════════════════════════
#                          GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def build_interface(restorer: MultiMethodRestorer, degrader: DegradationSimulator,
                    device: str) -> gr.Blocks:
    """Xây dựng giao diện 2 tab."""

    intro_md = f"""
    # Hệ thống phục chế ảnh chân dung cũ Việt Nam

    Demo tương tác cho đồ án tốt nghiệp:
    *"Nghiên cứu và phát triển hệ thống phục chế ảnh chân dung cũ ứng dụng học sâu"*

    **SV**: Đặng Thành Tựu (MSSV 20225951) | **GVHD**: TS. Trịnh Văn Chiến
    **Thiết bị đang chạy**: `{device.upper()}`
    """

    # ==== Tab 1: Phục chế ảnh ====
    def restore_handler(img: np.ndarray, method_label: str):
        if img is None:
            return None, "⚠ Vui lòng tải ảnh trước khi phục chế."
        method_key = dict(METHOD_CHOICES)[method_label]
        try:
            out, info = restorer.restore(img, method_key)
            return out, info
        except FileNotFoundError as e:
            return None, f" {str(e)}"
        except Exception as e:
            return None, f" Lỗi khi phục chế: {str(e)}"

    # ==== Tab 2: Tạo ảnh LQ ====
    def degrade_handler(img: np.ndarray, level_label: str, seed: int):
        if img is None:
            return None, "⚠ Vui lòng tải ảnh HQ trước khi tạo LQ."
        level_key = dict(LEVEL_CHOICES)[level_label]
        seed_val = int(seed) if seed and seed > 0 else None
        try:
            out, info = degrader.degrade(img, level_key, seed=seed_val)
            return out, info
        except Exception as e:
            return None, f" Lỗi khi tạo LQ: {str(e)}"

    with gr.Blocks(title="Phục chế ảnh chân dung cũ VN — Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(intro_md)

        with gr.Tabs():

            # ─── Tab 1: Phục chế ───────────────────────────────────
            with gr.Tab(" Phục chế ảnh"):
                gr.Markdown("""
                ### Chức năng
                Tải ảnh chân dung cũ bị hư hại và chọn một trong bốn phương pháp
                phục chế để so sánh trực quan.

                **Gợi ý**: Stage 3 là phương pháp cuối của đồ án (khuyến nghị dùng
                trước). Các phương pháp khác giúp so sánh trực quan với các baseline
                phổ biến.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        input_img1 = gr.Image(
                            label="Ảnh đầu vào (LQ)",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            height=400,
                        )
                        method_dropdown = gr.Dropdown(
                            choices=[label for label, _ in METHOD_CHOICES],
                            value=METHOD_CHOICES[0][0],
                            label="Phương pháp phục chế",
                            info="Model sẽ được load lần đầu chọn (mất vài giây).",
                        )
                        restore_btn = gr.Button(" Phục chế", variant="primary",
                                                size="lg")

                    with gr.Column(scale=1):
                        output_img1 = gr.Image(
                            label="Ảnh sau phục chế",
                            type="numpy",
                            height=400,
                            interactive=False,
                        )
                        info_text1 = gr.Textbox(
                            label="Thông tin",
                            interactive=False,
                            lines=3,
                        )

                # Auto-load ảnh mẫu (nếu có)
                examples_dir = ROOT / "data" / "real_vn_old"
                if examples_dir.exists():
                    sample_imgs = sorted(
                        list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.png"))
                    )[:6]
                    if sample_imgs:
                        gr.Examples(
                            examples=[[str(p), METHOD_CHOICES[0][0]] for p in sample_imgs],
                            inputs=[input_img1, method_dropdown],
                            label="Ảnh mẫu để thử nhanh",
                        )

                restore_btn.click(
                    fn=restore_handler,
                    inputs=[input_img1, method_dropdown],
                    outputs=[output_img1, info_text1],
                )

            # ─── Tab 2: Tạo ảnh LQ từ HQ ──────────────────────────────
            with gr.Tab("🎭 Tạo ảnh LQ (Vietnam Degradation)"):
                gr.Markdown("""
                ### Chức năng
                Tải ảnh chất lượng cao (HQ) và chọn mức suy giảm để mô phỏng ảnh
                chân dung cũ Việt Nam. Ảnh LQ tạo ra có thể tải về máy.

                **Ứng dụng**: minh họa Vietnam Degradation Pipeline (Đóng góp #1
                của đồ án), sinh dữ liệu test tùy chỉnh, hoặc chạy chuỗi *tạo LQ
                → phục chế* để so sánh với ảnh gốc.

                **8 loại suy giảm**: mờ, nhiễu, JPEG, giảm mẫu, xước, ố/mốc, bạc
                màu, ố vàng.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        input_img2 = gr.Image(
                            label="Ảnh chất lượng cao (HQ)",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            height=400,
                        )
                        level_radio = gr.Radio(
                            choices=[label for label, _ in LEVEL_CHOICES],
                            value=LEVEL_CHOICES[1][0],  # medium default
                            label="Mức suy giảm",
                        )
                        seed_input = gr.Number(
                            label="Hạt ngẫu nhiên (tùy chọn)",
                            value=0,
                            precision=0,
                            info="Để 0 nếu muốn kết quả ngẫu nhiên mỗi lần chạy. "
                                 "Đặt số cố định để tái lập cùng ảnh LQ.",
                        )
                        degrade_btn = gr.Button(" Tạo LQ", variant="primary",
                                                size="lg")

                    with gr.Column(scale=1):
                        output_img2 = gr.Image(
                            label="Ảnh LQ mô phỏng (bấm chuột phải → 'Save image as' để tải)",
                            type="numpy",
                            height=400,
                            interactive=False,
                            show_download_button=True,   # nút download
                        )
                        info_text2 = gr.Textbox(
                            label="Thông tin",
                            interactive=False,
                            lines=3,
                        )

                degrade_btn.click(
                    fn=degrade_handler,
                    inputs=[input_img2, level_radio, seed_input],
                    outputs=[output_img2, info_text2],
                )

        gr.Markdown("""
        ---
        **Ghi chú**: Model được huấn luyện chủ yếu trên ảnh khuôn mặt đã căn giữa.
        Ảnh chứa khuôn mặt lệch hoặc nhiều khuôn mặt có thể cho kết quả kém — trong
        trường hợp đó nên cắt lại chỉ giữ khuôn mặt chính ở giữa trước khi upload.
        """)

    return demo


# ══════════════════════════════════════════════════════════════════════════════
#                                MAIN ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3_ckpt", type=str, default=str(DEFAULT_STAGE3_CKPT))
    parser.add_argument("--stage2_ckpt", type=str, default=str(DEFAULT_STAGE2_CKPT))
    parser.add_argument("--gfpgan", type=str, default=str(DEFAULT_GFPGAN_PATH))
    parser.add_argument("--realesrgan", type=str, default=str(DEFAULT_REALESR_PATH))
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true",
                        help="Tạo public URL Gradio (24h)")
    args = parser.parse_args()

    # Xác định thiết bị
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Demo] Sử dụng thiết bị: {device.upper()}")

    # Kiểm tra checkpoint tối thiểu (Stage 3 + GFPGAN pretrained)
    stage3_ckpt = Path(args.stage3_ckpt)
    gfpgan_path = Path(args.gfpgan)
    if not stage3_ckpt.exists():
        print(f" CẢNH BÁO: Không tìm thấy Stage 3 checkpoint tại {stage3_ckpt}")
        print("  Chọn phương pháp Stage 3 trong demo sẽ báo lỗi. Các phương pháp "
              "khác vẫn dùng được nếu có checkpoint tương ứng.")

    # Khởi tạo restorer và degrader
    restorer = MultiMethodRestorer(
        device=device,
        stage3_ckpt=stage3_ckpt,
        stage2_ckpt=Path(args.stage2_ckpt),
        gfpgan_path=gfpgan_path,
        realesr_path=Path(args.realesrgan),
    )
    degrader = DegradationSimulator()

    # Build và chạy demo
    demo = build_interface(restorer, degrader, device)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
