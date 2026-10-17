"""
Mixed dataset utilities cho Stage 3.

Cho phép mix:
  - Axis 1: 2 nguồn ảnh (UTKFace Asian + FFHQ subset) theo tỷ lệ
  - Axis 2: 2 mức degradation (hard + medium) theo tỷ lệ

Architecture: nested mixing — outer mix degradation, inner mix sources.
Logic:
    mỗi __getitem__:
        if random < ratio_hard:  # 70%
            chọn từ pool hard
        else:                    # 30%
            chọn từ pool medium
        # bên trong mỗi pool:
        if random < ratio_a:     # 50%
            sample từ UTKFace Asian
        else:                    # 50%
            sample từ FFHQ subset
"""
from __future__ import annotations

import random
from typing import Optional

from torch.utils.data import Dataset

from src.utils.data_loader import PairedDegradationDataset


class MixedTwoSyntheticDataset(Dataset):
    """Mix 2 synthetic datasets (cùng degradation level) theo ratio.
    
    Length = max của 2 dataset (sampling có hoàn lại).
    """

    def __init__(
        self,
        ds_a: PairedDegradationDataset,
        ds_b: PairedDegradationDataset,
        ratio_a: float = 0.5,
        name_a: str = "A",
        name_b: str = "B",
    ):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.ratio_a = ratio_a
        self.name_a = name_a
        self.name_b = name_b
        print(f"[MixedTwo] {name_a}={len(ds_a)} + {name_b}={len(ds_b)}, "
              f"ratio={ratio_a:.2f}/{1-ratio_a:.2f}")

    def __len__(self) -> int:
        return max(len(self.ds_a), len(self.ds_b))

    def __getitem__(self, idx: int):
        if random.random() < self.ratio_a:
            return self.ds_a[idx % len(self.ds_a)]
        return self.ds_b[idx % len(self.ds_b)]


class MixedDegradationLevelDataset(Dataset):
    """Mix 2 mức degradation (vd hard 70% + medium 30%) trên cùng nguồn ảnh.
    
    Mỗi mức là 1 PairedDegradationDataset độc lập (mỗi cái có
    VietnamDegradationPipeline riêng với level riêng).
    """

    def __init__(
        self,
        ds_hard: Dataset,
        ds_medium: Dataset,
        ratio_hard: float = 0.7,
    ):
        self.ds_hard = ds_hard
        self.ds_medium = ds_medium
        self.ratio_hard = ratio_hard
        print(f"[MixedDeg] hard={len(ds_hard)} + medium={len(ds_medium)}, "
              f"ratio_hard={ratio_hard:.2f}")

    def __len__(self) -> int:
        return max(len(self.ds_hard), len(self.ds_medium))

    def __getitem__(self, idx: int):
        if random.random() < self.ratio_hard:
            return self.ds_hard[idx % len(self.ds_hard)]
        return self.ds_medium[idx % len(self.ds_medium)]


def build_stage3_mixed_train_dataset(
    asian_dir: str,
    ffhq_dir: str,
    image_size: int = 256,
    ratio_asian: float = 0.5,
    ratio_hard: float = 0.7,
    ffhq_max_samples: Optional[int] = None,
):
    """Build Stage 3 train dataset với 2 axes mixing.
    
    Args:
        asian_dir: folder UTKFace race=2 (Đông Á)
        ffhq_dir: folder FFHQ (sẽ subsample nếu ffhq_max_samples đặt)
        image_size: 256
        ratio_asian: tỷ lệ UTKFace Asian trong mỗi batch (default 0.5)
        ratio_hard: tỷ lệ hard degradation (default 0.7)
        ffhq_max_samples: nếu đặt, subsample FFHQ (vd 10000)
    
    Returns:
        Dataset object để pass vào DataLoader
    """
    # Build 4 base datasets (2 sources × 2 levels)
    asian_hard = PairedDegradationDataset(
        hq_dirs=[asian_dir],
        image_size=image_size,
        degradation_level="hard",
    )
    asian_medium = PairedDegradationDataset(
        hq_dirs=[asian_dir],
        image_size=image_size,
        degradation_level="medium",
    )

    # FFHQ với optional subsample
    ffhq_hard_full = PairedDegradationDataset(
        hq_dirs=[ffhq_dir],
        image_size=image_size,
        degradation_level="hard",
    )
    ffhq_medium_full = PairedDegradationDataset(
        hq_dirs=[ffhq_dir],
        image_size=image_size,
        degradation_level="medium",
    )

    # Subsample FFHQ paths nếu cần
    if ffhq_max_samples and ffhq_max_samples < len(ffhq_hard_full.paths):
        # Same subset cho cả hard và medium để consistent
        random.seed(42)
        subset_paths = random.sample(ffhq_hard_full.paths, ffhq_max_samples)
        ffhq_hard_full.paths = subset_paths
        ffhq_medium_full.paths = subset_paths
        print(f"[FFHQ] Subsampled: {len(subset_paths)}")

    # Mix sources cho mỗi mức degradation
    hard_mixed = MixedTwoSyntheticDataset(
        asian_hard, ffhq_hard_full,
        ratio_a=ratio_asian,
        name_a="Asian", name_b="FFHQ",
    )
    medium_mixed = MixedTwoSyntheticDataset(
        asian_medium, ffhq_medium_full,
        ratio_a=ratio_asian,
        name_a="Asian", name_b="FFHQ",
    )

    # Mix 2 mức degradation
    final = MixedDegradationLevelDataset(
        ds_hard=hard_mixed,
        ds_medium=medium_mixed,
        ratio_hard=ratio_hard,
    )
    return final
