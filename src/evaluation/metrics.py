"""Evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def ssim(pred, target):
    from skimage.metrics import structural_similarity as sk_ssim
    p = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    t = target.detach().cpu().permute(0, 2, 3, 1).numpy()
    return float(np.mean([sk_ssim(p[i], t[i], channel_axis=2, data_range=1.0)
                          for i in range(p.shape[0])]))


class LPIPSMetric:
    def __init__(self, net="alex", device="cuda"):
        import lpips
        self.model = lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred, target):
        pred = pred.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1
        return float(self.model(pred, target).mean().item())


class IdentitySimilarity:
    def __init__(self, device="cuda"):
        from src.losses.identity import ArcFaceFeatureExtractor
        self.ext = ArcFaceFeatureExtractor()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred, target):
        fp = self.ext(pred.to(self.device))
        ft = self.ext(target.to(self.device))
        return float((fp * ft).sum(dim=1).mean().item())


class NoReferenceMetrics:
    def __init__(self, device="cuda"):
        import pyiqa
        self.niqe = pyiqa.create_metric("niqe", device=device)
        self.musiq = pyiqa.create_metric("musiq", device=device)
        self.maniqa = pyiqa.create_metric("maniqa", device=device)

    @torch.no_grad()
    def __call__(self, img):
        return {
            "niqe": float(self.niqe(img).mean().item()),
            "musiq": float(self.musiq(img).mean().item()),
            "maniqa": float(self.maniqa(img).mean().item()),
        }
