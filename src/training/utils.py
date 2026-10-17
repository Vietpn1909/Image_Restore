"""
Multi-Stage Training Utilities (đóng góp 2).
Helper functions cho freeze/unfreeze trong protocol 3 stage.
"""
from __future__ import annotations

import torch.nn as nn


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True
    model.train()


def unfreeze_last_n_blocks(model: nn.Module, n: int, block_attr: str = "body"):
    freeze_all(model)
    blocks = getattr(model, block_attr, None)
    if blocks is None:
        for name, child in model.named_children():
            if isinstance(child, (nn.Sequential, nn.ModuleList)):
                blocks = child
                break
    if blocks is None:
        raise ValueError(f"Không tìm thấy block_attr={block_attr}")
    for blk in list(blocks)[-n:]:
        for p in blk.parameters():
            p.requires_grad = True
        blk.train()


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def report_trainable(model: nn.Module, name: str = "model"):
    total = sum(p.numel() for p in model.parameters())
    train = count_trainable(model)
    print(f"[{name}] trainable: {train/1e6:.2f}M / {total/1e6:.2f}M ({100*train/max(total,1):.1f}%)")
