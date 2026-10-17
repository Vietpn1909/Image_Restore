"""
Filter ảnh đầu vào theo checklist:
1. Resolution >= 400x400 (loose hơn cho ảnh cũ vốn nhỏ)
2. Có ít nhất 1 mặt rõ (MediaPipe detection)
3. Vùng mặt >= 150x150 pixel
4. Không quá blurry (Laplacian variance)

Dùng cho:
    - Ảnh VN scan từ gia đình
    - Ảnh tải về từ gallery-dl

Usage:
    from src.utils.filter_helper import ImageFilter
    f = ImageFilter()
    result = f.check("path/to/image.jpg")
    if result.accepted: ...
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FilterResult:
    accepted: bool
    reason: str
    face_count: int = 0
    max_face_size: int = 0
    resolution: tuple[int, int] = (0, 0)
    blur_score: float = 0.0


class ImageFilter:
    def __init__(
        self,
        min_resolution: int = 400,
        min_face_size: int = 150,
        min_blur_score: float = 15.0,
    ):
        self.min_resolution = min_resolution
        self.min_face_size = min_face_size
        self.min_blur_score = min_blur_score
        self._detector = None

    def _init_detector(self):
        if self._detector is None:
            import mediapipe as mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)

    def check(self, img_path) -> FilterResult:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            return FilterResult(False, "cannot_read")

        h, w = img.shape[:2]
        if min(h, w) < self.min_resolution:
            return FilterResult(False, f"low_resolution_{w}x{h}", resolution=(w, h))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.min_blur_score:
            return FilterResult(False, f"too_blurry_{blur_score:.1f}",
                              resolution=(w, h), blur_score=blur_score)

        self._init_detector()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        if not results.detections:
            return FilterResult(False, "no_face", resolution=(w, h), blur_score=blur_score)

        face_sizes = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            face_sizes.append(min(int(bbox.width * w), int(bbox.height * h)))

        max_face = max(face_sizes)
        if max_face < self.min_face_size:
            return FilterResult(False, f"face_too_small_{max_face}",
                              face_count=len(face_sizes), max_face_size=max_face,
                              resolution=(w, h), blur_score=blur_score)

        return FilterResult(True, "ok",
                          face_count=len(face_sizes), max_face_size=max_face,
                          resolution=(w, h), blur_score=blur_score)


def hash_image(img_path) -> str:
    with open(img_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
