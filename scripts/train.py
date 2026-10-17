"""
Main training entry point.

Usage:
    python scripts/train.py --config configs/stage1_warmup.yaml
    python scripts/train.py --config configs/stage2_medium.yaml
    python scripts/train.py --config configs/stage3_hard.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    stage = config["stage"]
    print(f"=== Training Stage {stage}: {config['name']} ===")

    if stage == 1:
        from src.training.trainer_stage1 import Stage1Trainer
        trainer = Stage1Trainer(config, resume=args.resume)
    elif stage == 2:
        from src.training.trainer_stage2 import Stage2Trainer
        trainer = Stage2Trainer(config, resume=args.resume)
    elif stage == 3:
        from src.training.trainer_stage3 import Stage3Trainer
        trainer = Stage3Trainer(config, resume=args.resume)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    trainer.train()


if __name__ == "__main__":
    main()
