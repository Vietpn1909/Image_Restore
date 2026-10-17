"""
ĐÓNG GÓP NGHIÊN CỨU SỐ 1:
Mô hình hóa suy giảm theo miền ảnh chân dung cũ .

Tham khảo / mở rộng từ:
    - Real-ESRGAN (Wang et al., ICCVW 2021) — second-order degradation
    - GFPGAN (Wang et al., CVPR 2021) — degradation cho face
    - BSRGAN (Zhang et al., ICCV 2021) — random shuffled degradation

Khác biệt:
    - Bổ sung suy giảm đặc trưng VN: scratch, stain, mold, fade, yellowing.
    - Phân tầng 3 mức độ light/medium/hard tương ứng 3 stage fine-tune.
    - Sinh kèm degradation vector 8-d làm label cho degradation encoder.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class DegradationLevel(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class DegradationParams:
    blur_kernel_size_range: tuple[int, int] = (7, 21)
    blur_sigma_range: tuple[float, float] = (0.5, 3.0)
    blur_prob: float = 0.8

    gaussian_noise_sigma_range: tuple[float, float] = (1, 25)
    poisson_noise_scale_range: tuple[float, float] = (0.05, 2.5)
    noise_prob: float = 0.7

    jpeg_quality_range: tuple[int, int] = (40, 90)
    jpeg_prob: float = 0.7

    downsample_scale_range: tuple[float, float] = (1.0, 4.0)
    downsample_prob: float = 1.0

    scratch_prob: float = 0.3
    scratch_count_range: tuple[int, int] = (1, 5)

    stain_prob: float = 0.2
    stain_count_range: tuple[int, int] = (1, 3)

    mold_prob: float = 0.15

    color_fade_prob: float = 0.4
    color_fade_strength_range: tuple[float, float] = (0.1, 0.5)

    yellowing_prob: float = 0.5
    yellowing_strength_range: tuple[float, float] = (0.05, 0.3)


LEVEL_PRESETS: dict[DegradationLevel, DegradationParams] = {
    DegradationLevel.LIGHT: DegradationParams(
        blur_sigma_range=(0.3, 1.5),
        gaussian_noise_sigma_range=(1, 10),
        jpeg_quality_range=(70, 95),
        downsample_scale_range=(1.0, 2.0),
        scratch_prob=0.1,
        stain_prob=0.05,
        mold_prob=0.0,
        color_fade_prob=0.2,
        yellowing_prob=0.3,
    ),
    DegradationLevel.MEDIUM: DegradationParams(),
    DegradationLevel.HARD: DegradationParams(
        blur_sigma_range=(1.0, 4.0),
        gaussian_noise_sigma_range=(5, 35),
        jpeg_quality_range=(20, 70),
        downsample_scale_range=(2.0, 4.0),
        scratch_prob=0.6,
        scratch_count_range=(2, 8),
        stain_prob=0.5,
        stain_count_range=(2, 5),
        mold_prob=0.4,
        color_fade_prob=0.7,
        color_fade_strength_range=(0.3, 0.7),
        yellowing_prob=0.8,
        yellowing_strength_range=(0.2, 0.5),
    ),
}


class VietnamDegradationPipeline:
    """
    Sinh ảnh LQ từ ảnh HQ, kèm degradation vector 8-d.

    Vector 8-d: [blur, noise, jpeg, downsample, scratch, stain, fade, yellowing]
    Mỗi chiều ∈ [0, 1], biểu thị mức suy giảm tương ứng.
    """

    def __init__(
        self,
        level: DegradationLevel = DegradationLevel.MEDIUM,
        overlay_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.level = level
        self.params = LEVEL_PRESETS[level]
        self.overlay_dir = Path(overlay_dir) if overlay_dir else None
        self.rng = random.Random(seed)
        self._scratch_overlays: list[np.ndarray] = []
        self._stain_overlays: list[np.ndarray] = []
        self._mold_overlays: list[np.ndarray] = []
        if self.overlay_dir is not None and self.overlay_dir.exists():
            self._load_overlays()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
            return_uint8 = True
        else:
            img = image.copy()
            return_uint8 = False

        deg_vec = np.zeros(8, dtype=np.float32)

        if self.rng.random() < self.params.blur_prob:
            img, deg_vec[0] = self._apply_blur(img)
        if self.rng.random() < self.params.noise_prob:
            img, deg_vec[1] = self._apply_noise(img)
        if self.rng.random() < self.params.jpeg_prob:
            img, deg_vec[2] = self._apply_jpeg(img)
        img, deg_vec[3] = self._apply_downsample(img)
        if self.rng.random() < self.params.scratch_prob:
            img, deg_vec[4] = self._apply_scratch(img)
        if self.rng.random() < self.params.stain_prob:
            img, deg_vec[5] = self._apply_stain(img)
        if self.rng.random() < self.params.color_fade_prob:
            img, deg_vec[6] = self._apply_color_fade(img)
        if self.rng.random() < self.params.yellowing_prob:
            img, deg_vec[7] = self._apply_yellowing(img)

        img = np.clip(img, 0.0, 1.0)
        if return_uint8:
            img = (img * 255.0).astype(np.uint8)
        return img, deg_vec

    def _apply_blur(self, img):
        s_lo, s_hi = self.params.blur_sigma_range
        sigma = self.rng.uniform(s_lo, s_hi)
        k_lo, k_hi = self.params.blur_kernel_size_range
        ksize = self.rng.choice(list(range(k_lo, k_hi + 1, 2)))
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        return img, float((sigma - s_lo) / (s_hi - s_lo + 1e-8))

    def _apply_noise(self, img):
        s_lo, s_hi = self.params.gaussian_noise_sigma_range
        sigma = self.rng.uniform(s_lo, s_hi) / 255.0
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        img = img + noise
        return img, float((sigma * 255 - s_lo) / (s_hi - s_lo + 1e-8))

    def _apply_jpeg(self, img):
        q_lo, q_hi = self.params.jpeg_quality_range
        q = self.rng.randint(q_lo, q_hi)
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
                              [int(cv2.IMWRITE_JPEG_QUALITY), q])
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img, float(1.0 - (q - q_lo) / (q_hi - q_lo + 1e-8))

    def _apply_downsample(self, img):
        s_lo, s_hi = self.params.downsample_scale_range
        scale = self.rng.uniform(s_lo, s_hi)
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img, float((scale - s_lo) / (s_hi - s_lo + 1e-8))

    def _apply_scratch(self, img):
        h, w = img.shape[:2]
        n_lo, n_hi = self.params.scratch_count_range
        n = self.rng.randint(n_lo, n_hi)
        if self._scratch_overlays:
            for _ in range(n):
                overlay = self.rng.choice(self._scratch_overlays)
                overlay = cv2.resize(overlay, (w, h))
                mask = (overlay if overlay.ndim == 2 else overlay[..., 0]).astype(np.float32) / 255.0
                strength = self.rng.uniform(0.3, 0.8)
                img = img * (1 - mask[..., None] * strength) + mask[..., None] * strength
        else:
            for _ in range(n):
                img = self._draw_random_scratch(img)
        return img, float(min(1.0, n / n_hi))

    def _draw_random_scratch(self, img):
        h, w = img.shape[:2]
        x1, y1 = self.rng.randint(0, w - 1), self.rng.randint(0, h - 1)
        length = self.rng.randint(int(min(h, w) * 0.2), int(min(h, w) * 0.8))
        angle = self.rng.uniform(0, 2 * np.pi)
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.line(mask, (x1, y1), (x2, y2), self.rng.uniform(0.7, 1.0), self.rng.randint(1, 3))
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
        return img * (1 - mask[..., None] * 0.6) + mask[..., None] * 0.6

    def _apply_stain(self, img):
        h, w = img.shape[:2]
        n_lo, n_hi = self.params.stain_count_range
        n = self.rng.randint(n_lo, n_hi)
        for _ in range(n):
            cx, cy = self.rng.randint(0, w - 1), self.rng.randint(0, h - 1)
            radius = self.rng.randint(int(min(h, w) * 0.05), int(min(h, w) * 0.2))
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), radius / 3)
            color = np.array(
                [self.rng.uniform(0.4, 0.7), self.rng.uniform(0.3, 0.5), self.rng.uniform(0.2, 0.4)],
                dtype=np.float32)
            alpha = self.rng.uniform(0.2, 0.5)
            img = img * (1 - mask[..., None] * alpha) + color[None, None, :] * mask[..., None] * alpha
        return img, float(min(1.0, n / n_hi))

    def _apply_color_fade(self, img):
        s_lo, s_hi = self.params.color_fade_strength_range
        strength = self.rng.uniform(s_lo, s_hi)
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        img = img * (1 - strength) + gray_rgb * strength
        return img, float((strength - s_lo) / (s_hi - s_lo + 1e-8))

    def _apply_yellowing(self, img):
        s_lo, s_hi = self.params.yellowing_strength_range
        strength = self.rng.uniform(s_lo, s_hi)
        yellow = np.array([1.0, 0.95, 0.7], dtype=np.float32)
        img = img * (1 - strength) + img * yellow[None, None, :] * strength
        return img, float((strength - s_lo) / (s_hi - s_lo + 1e-8))

    def _load_overlays(self):
        for sub, target in [("scratches", self._scratch_overlays),
                            ("stains", self._stain_overlays),
                            ("molds", self._mold_overlays)]:
            d = self.overlay_dir / sub
            if not d.exists():
                continue
            for p in sorted(d.glob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if arr is not None:
                        target.append(arr)


if __name__ == "__main__":
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    for lvl in DegradationLevel:
        p = VietnamDegradationPipeline(level=lvl, seed=42)
        out, vec = p(dummy)
        print(f"[{lvl.value}] out.shape={out.shape}, vec={vec.round(2).tolist()}")
