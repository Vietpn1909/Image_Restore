"""
Filter ảnh VN sau khi crawl bằng gallery-dl.

Usage:
    python scripts/filter_vn_real.py \
        --in_dir data/real_vn_old/raw \
        --out_dir data/real_vn_old/filtered
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.filter_helper import ImageFilter, hash_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--min_resolution", type=int, default=400)
    parser.add_argument("--min_face_size", type=int, default=150)
    parser.add_argument("--min_blur_score", type=float, default=15.0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_filter = ImageFilter(
        min_resolution=args.min_resolution,
        min_face_size=args.min_face_size,
        min_blur_score=args.min_blur_score,
    )

    seen = set()
    metadata = []
    stats = {"total": 0, "accepted": 0, "duplicates": 0, "rejected": {}}

    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        imgs.extend(in_dir.rglob(ext))
    imgs = sorted(set(imgs))

    print(f"Found {len(imgs)} images")

    for img in tqdm(imgs, desc="Filtering"):
        stats["total"] += 1
        try:
            h = hash_image(img)
        except Exception:
            continue
        if h in seen:
            stats["duplicates"] += 1
            continue
        seen.add(h)

        res = img_filter.check(img)
        if not res.accepted:
            stats["rejected"][res.reason] = stats["rejected"].get(res.reason, 0) + 1
            continue

        new_name = f"{len(metadata):06d}_vn_old.jpg"
        (out_dir / new_name).write_bytes(img.read_bytes())
        metadata.append({
            "id": len(metadata),
            "filename": new_name,
            "original": img.name,
            "resolution": f"{res.resolution[0]}x{res.resolution[1]}",
            "face_count": res.face_count,
            "max_face_size": res.max_face_size,
            "blur_score": round(res.blur_score, 1),
        })
        stats["accepted"] += 1

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2))
    print(f"\n→ {stats['accepted']} ảnh accepted vào {out_dir}")


if __name__ == "__main__":
    main()
