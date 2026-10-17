"""Inference script for single image restoration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/inference.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("Implement after training pipeline is ready")


if __name__ == "__main__":
    main()
