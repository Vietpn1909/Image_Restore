"""
Dataset classes cho training pipeline phục chế ảnh chân dung cũ.

Hỗ trợ:
    - Load HQ images từ NHIỀU folder (FFHQ + Asian dataset mix).
    - Online sinh LQ + degradation vector qua VietnamDegradationPipeline.
    - Stage 3: mix với ảnh thật VN (không có GT, dùng cho cycle loss).

Format trả về:
    Stage 1, 2: (lq, hq, deg_vec) — paired
    Stage 3:    (lq, hq, deg_vec, is_real) — is_real flag để phân biệt
                ảnh tổng hợp (paired) vs ảnh thật (cycle loss)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.degradation.vietnam_pipeline import (
    DegradationLevel,
    VietnamDegradationPipeline,
)



# Utils

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folders: list[str | Path]) -> list[Path]:
    """Liệt kê tất cả ảnh trong nhiều folder, đệ quy."""
    paths = []
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            print(f"[WARNING] Folder không tồn tại: {folder}")
            continue
        for ext in IMG_EXTS:
            paths.extend(folder.rglob(f"*{ext}"))
            paths.extend(folder.rglob(f"*{ext.upper()}"))
    paths = sorted(set(paths))
    return paths


def read_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Đọc ảnh, return RGB uint8. None nếu lỗi."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def center_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Center crop về vuông rồi resize về size×size."""
    h, w = img.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    img = img[top:top + s, left:left + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) → tensor float32 (3, H, W) ∈ [0, 1]."""
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()



# Paired dataset — HQ → LQ qua degradation pipeline

class PairedDegradationDataset(Dataset):
    """
    Dùng cho stage 1, 2, 3 (phần synthetic).

    Quy trình mỗi sample:
        1. Load HQ ảnh từ folder.
        2. Center crop + resize về image_size×image_size.
        3. Sinh LQ + degradation vector qua VietnamDegradationPipeline.
        4. Augmentation cơ bản (flip horizontal).
        5. Convert tensor.

    Return:
        lq:      (3, H, W) float32 ∈ [0, 1]
        hq:      (3, H, W) float32 ∈ [0, 1]
        deg_vec: (8,) float32 ∈ [0, 1]
    """

    def __init__(
        self,
        hq_dirs: list[str],
        image_size: int = 512,
        degradation_level: str = "medium",
        overlay_dir: Optional[str] = None,
        flip: bool = True,
        seed: Optional[int] = None,
    ):
        self.image_size = image_size
        self.flip = flip
        self.paths = list_images(hq_dirs)
        if len(self.paths) == 0:
            raise ValueError(f"Không tìm thấy ảnh nào trong: {hq_dirs}")
        print(f"[PairedDataset] {len(self.paths)} HQ images từ {len(hq_dirs)} folder(s)")

        # Degradation pipeline — seed=None để mỗi worker random độc lập
        self.deg_pipeline = VietnamDegradationPipeline(
            level=DegradationLevel(degradation_level),
            overlay_dir=overlay_dir,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # Retry nếu ảnh lỗi
        for _ in range(5):
            path = self.paths[idx % len(self.paths)]
            img = read_image_rgb(path)
            if img is not None and min(img.shape[:2]) >= 64:
                break
            idx = (idx + 1) % len(self.paths)
        else:
            # Fallback: ảnh đen
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Center crop + resize
        hq = center_crop_resize(img, self.image_size)

        # Augmentation
        if self.flip and random.random() < 0.5:
            hq = hq[:, ::-1, :].copy()

        # Sinh LQ + degradation vector
        lq, deg_vec = self.deg_pipeline(hq)

        return {
            "lq": to_tensor(lq),
            "hq": to_tensor(hq),
            "deg_vec": torch.from_numpy(deg_vec),
            "is_real": torch.tensor(0, dtype=torch.long),
        }



# Real VN dataset — ảnh cũ thật, không có GT

class RealVNOldDataset(Dataset):
    """
    Dùng cho stage 3, mix với PairedDegradationDataset.
    Ảnh chân dung VN thật bị suy giảm → không có HQ ground truth.

    Return:
        lq:      (3, H, W) — ảnh thật (coi như LQ input)
        hq:      (3, H, W) — copy của lq (placeholder, sẽ không dùng trong pixel loss)
        deg_vec: (8,) — zeros (chưa biết mức degradation thật)
        is_real: 1 (flag để loss biết bỏ qua pixel/perceptual, dùng cycle loss)
    """

    def __init__(self, real_dir: str, image_size: int = 512, flip: bool = True):
        self.image_size = image_size
        self.flip = flip
        self.paths = list_images([real_dir])
        if len(self.paths) == 0:
            print(f"[WARNING] RealVNOldDataset: không tìm thấy ảnh trong {real_dir}")
        print(f"[RealVN] {len(self.paths)} real images từ {real_dir}")

    def __len__(self) -> int:
        return max(1, len(self.paths))

    def __getitem__(self, idx: int):
        if len(self.paths) == 0:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            for _ in range(5):
                path = self.paths[idx % len(self.paths)]
                img = read_image_rgb(path)
                if img is not None and min(img.shape[:2]) >= 64:
                    break
                idx = (idx + 1) % len(self.paths)
            else:
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        lq_img = center_crop_resize(img, self.image_size)
        if self.flip and random.random() < 0.5:
            lq_img = lq_img[:, ::-1, :].copy()

        lq_tensor = to_tensor(lq_img)
        return {
            "lq": lq_tensor,
            "hq": lq_tensor.clone(),    # placeholder, không dùng cho pixel loss
            "deg_vec": torch.zeros(8, dtype=torch.float32),
            "is_real": torch.tensor(1, dtype=torch.long),
        }



# Mixed dataset — synthetic + real (stage 3)

class MixedDataset(Dataset):
    """
    Mix PairedDegradationDataset (synthetic) + RealVNOldDataset (real)
    theo tỉ lệ real_ratio. Length = length của synthetic dataset.
    """

    def __init__(
        self,
        synthetic: PairedDegradationDataset,
        real: RealVNOldDataset,
        real_ratio: float = 0.3,
    ):
        self.synthetic = synthetic
        self.real = real
        self.real_ratio = real_ratio

    def __len__(self) -> int:
        return len(self.synthetic)

    def __getitem__(self, idx: int):
        if random.random() < self.real_ratio and len(self.real.paths) > 0:
            return self.real[idx % len(self.real)]
        return self.synthetic[idx]



# Validation dataset — cố định seed để eval consistent

class ValDataset(Dataset):
    """
    Val dataset: load HQ, sinh LQ với seed cố định để eval consistent giữa các epoch.
    """

    def __init__(
        self,
        hq_dir: str,
        image_size: int = 512,
        degradation_level: str = "medium",
        max_samples: int = 200,
    ):
        self.image_size = image_size
        self.paths = list_images([hq_dir])[:max_samples]
        print(f"[ValDataset] {len(self.paths)} val images từ {hq_dir}")
        self.deg_pipeline = VietnamDegradationPipeline(
            level=DegradationLevel(degradation_level),
            seed=12345,    # cố định
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # Re-seed pipeline theo idx → cùng idx luôn ra cùng LQ
        self.deg_pipeline.rng.seed(12345 + idx)
        np.random.seed(12345 + idx)

        path = self.paths[idx]
        img = read_image_rgb(path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        hq = center_crop_resize(img, self.image_size)
        lq, deg_vec = self.deg_pipeline(hq)
        return {
            "lq": to_tensor(lq),
            "hq": to_tensor(hq),
            "deg_vec": torch.from_numpy(deg_vec),
            "is_real": torch.tensor(0, dtype=torch.long),
            "filename": path.name,
        }



# Factory function — build dataset từ config

def build_train_dataset(config: dict) -> Dataset:
    """Build training dataset từ config YAML."""
    data_cfg = config["data"]
    hq_dirs = data_cfg["hq_dirs"]
    image_size = data_cfg["image_size"]
    deg_level = data_cfg["degradation_level"]

    syn = PairedDegradationDataset(
        hq_dirs=hq_dirs,
        image_size=image_size,
        degradation_level=deg_level,
        overlay_dir=data_cfg.get("overlay_dir"),
    )

    if data_cfg.get("use_real_vn", False):
        real_dir = data_cfg["real_vn_dir"]
        real = RealVNOldDataset(real_dir=real_dir, image_size=image_size)
        return MixedDataset(syn, real, real_ratio=data_cfg.get("real_vn_ratio", 0.3))

    return syn


def build_val_dataset(config: dict) -> Dataset:
    data_cfg = config["data"]
    return ValDataset(
        hq_dir=data_cfg["val_dir"],
        image_size=data_cfg["image_size"],
        degradation_level=data_cfg["degradation_level"],
        max_samples=200,
    )



# Smoke test

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Tạo dataset dummy với folder ảnh giả lập
    import tempfile
    tmpdir = Path(tempfile.mkdtemp())
    for i in range(5):
        dummy = np.random.randint(50, 200, (1024, 1024, 3), dtype=np.uint8)
        cv2.imwrite(str(tmpdir / f"test_{i}.jpg"), cv2.cvtColor(dummy, cv2.COLOR_RGB2BGR))

    ds = PairedDegradationDataset(
        hq_dirs=[str(tmpdir)],
        image_size=256,
        degradation_level="medium",
    )
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"LQ:  {sample['lq'].shape}, range=[{sample['lq'].min():.2f},{sample['lq'].max():.2f}]")
    print(f"HQ:  {sample['hq'].shape}, range=[{sample['hq'].min():.2f},{sample['hq'].max():.2f}]")
    print(f"vec: {sample['deg_vec'].shape}, {sample['deg_vec'].numpy().round(2).tolist()}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
