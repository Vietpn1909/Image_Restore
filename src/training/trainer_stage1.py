"""
Stage 1 Trainer — Warmup (FIXED VERSION).

CRITICAL FIXES so với version cũ:
1. FORCE ABSOLUTE PATH cho output_dir (chống reset CWD khi kernel restart)
2. VERIFY mỗi save: check file tồn tại + size > 0
3. AUTO-BACKUP mỗi N iter ra ngoài project folder
4. Print absolute path rõ ràng để user kiểm tra
5. Save với atomic write (tmp file → rename) để chống corrupt
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.models.degradation_encoder import DegradationEncoder
from src.models.film_layer import FiLMLayer
from src.utils.data_loader import build_train_dataset, build_val_dataset


class Stage1Trainer:
    """Stage 1 warmup trainer — FIXED VERSION."""

    def __init__(self, config: dict, resume: Optional[str] = None):
        self.config = config
        self.resume = resume

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_seed()
        self._setup_output_dirs()  # FIXED: force absolute

        self._build_models()
        self._build_optimizers()
        self._build_losses()
        self._build_data()
        self._build_logging()

        self.iter = 0
        self.best_val_loss = float("inf")

        if resume:
            self._load_checkpoint(resume)

    def _setup_seed(self):
        seed = self.config["train"].get("seed", 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _setup_output_dirs(self):
        """FIXED: convert path → absolute, tạo backup dir ngoài project."""
        # Main checkpoint dir
        raw_path = self.config["log"]["output_dir"]
        self.ckpt_dir = Path(raw_path).resolve()  # FORCE ABSOLUTE
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # BACKUP dir — ngoài project, ở /kaggle/working trực tiếp
        self.backup_dir = Path("/kaggle/working/ckpt_backup_stage1").resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.tb_dir = Path(self.config["log"]["tensorboard_dir"]).resolve()
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # PRINT rõ ràng cho user
        print("=" * 60)
        print("CHECKPOINT PATHS (ABSOLUTE):")
        print(f"  Main:    {self.ckpt_dir}")
        print(f"  Backup:  {self.backup_dir}")
        print(f"  TB:      {self.tb_dir}")
        print("=" * 60)

    def _build_models(self):
        mcfg = self.config["model"]

        self.deg_encoder = DegradationEncoder(
            out_dim=mcfg["degradation_encoder"]["out_dim"],
        ).to(self.device)

        self.film_adapters = nn.ModuleList([
            FiLMLayer(cond_dim=mcfg["degradation_encoder"]["out_dim"], feat_channels=64),
            FiLMLayer(cond_dim=mcfg["degradation_encoder"]["out_dim"], feat_channels=128),
            FiLMLayer(cond_dim=mcfg["degradation_encoder"]["out_dim"], feat_channels=256),
        ]).to(self.device)

        self.restore_head = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        ).to(self.device)

        n_trainable = sum(p.numel() for p in self._trainable_params())
        print(f"[Stage1] Trainable params: {n_trainable / 1e6:.2f}M")

    def _trainable_params(self):
        return list(self.deg_encoder.parameters()) \
             + list(self.film_adapters.parameters()) \
             + list(self.restore_head.parameters())

    def _build_optimizers(self):
        ocfg = self.config["optim"]
        self.optimizer = AdamW(
            self._trainable_params(),
            lr=ocfg["lr"],
            betas=tuple(ocfg["betas"]),
            weight_decay=ocfg["weight_decay"],
        )

        sched_cfg = ocfg["scheduler"]
        warmup = LinearLR(self.optimizer, start_factor=0.01, total_iters=sched_cfg["warmup_iters"])
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=sched_cfg["total_iters"] - sched_cfg["warmup_iters"],
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[sched_cfg["warmup_iters"]]
        )

        self.use_amp = self.config["train"].get("mixed_precision", True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _build_losses(self):
        lcfg = self.config["loss"]
        self.loss_weights = {
            "pixel": lcfg["pixel_l1"]["weight"] if lcfg["pixel_l1"]["enable"] else 0.0,
            "perceptual": lcfg["perceptual_vgg"]["weight"] if lcfg["perceptual_vgg"]["enable"] else 0.0,
            "identity": lcfg["identity_arcface"]["weight"] if lcfg["identity_arcface"]["enable"] else 0.0,
            "deg_supervised": lcfg["degradation_supervised"]["weight"]
                              if lcfg["degradation_supervised"]["enable"] else 0.0,
        }

        # FIXED: dùng reduction='mean' đúng
        self.pixel_criterion = nn.L1Loss(reduction='mean')

        self.perceptual_criterion = None
        if self.loss_weights["perceptual"] > 0:
            from src.losses.perceptual import VGGPerceptualLoss
            self.perceptual_criterion = VGGPerceptualLoss().to(self.device).eval()

        self.identity_extractor = None
        if self.loss_weights["identity"] > 0:
            try:
                from src.losses.identity import ArcFaceFeatureExtractor
                self.identity_extractor = ArcFaceFeatureExtractor().eval()
            except Exception as e:
                print(f"[WARNING] Không load được ArcFace: {e}. Tắt identity loss.")
                self.loss_weights["identity"] = 0.0

    def _build_data(self):
        dcfg = self.config["data"]
        train_ds = build_train_dataset(self.config)
        val_ds = build_val_dataset(self.config)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=dcfg["batch_size"],
            shuffle=True,
            num_workers=dcfg["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=dcfg["num_workers"] > 0,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=dcfg["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def _build_logging(self):
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(str(self.tb_dir))

        self.use_wandb = self.config["log"].get("wandb", {}).get("enable", False)
        self.wandb = None
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config["log"]["wandb"]["project"],
                    name=self.config["log"]["wandb"]["run_name"],
                    config=self.config,
                )
                self.wandb = wandb
            except Exception as e:
                print(f"[WARNING] W&B init failed: {e}")
                self.use_wandb = False

    def forward(self, lq: torch.Tensor, deg_gt: torch.Tensor):
        deg_pred = self.deg_encoder(lq)
        restored = self.restore_head(lq) + lq
        restored = restored.clamp(0, 1)
        return restored, deg_pred

    def compute_loss(self, restored, hq, deg_pred, deg_gt, is_real):
        # FIXED: pixel loss đúng, không chia thêm C*H*W
        syn_mask = (is_real == 0)
        losses = {}

        if self.loss_weights["pixel"] > 0:
            if syn_mask.any():
                losses["pixel"] = self.pixel_criterion(restored[syn_mask], hq[syn_mask])
            else:
                losses["pixel"] = torch.tensor(0.0, device=self.device)

        if self.perceptual_criterion is not None and self.loss_weights["perceptual"] > 0:
            if syn_mask.any():
                losses["perceptual"] = self.perceptual_criterion(restored[syn_mask], hq[syn_mask])
            else:
                losses["perceptual"] = torch.tensor(0.0, device=self.device)

        if self.identity_extractor is not None and self.loss_weights["identity"] > 0:
            from src.losses.identity import identity_loss
            try:
                if syn_mask.any():
                    losses["identity"] = identity_loss(
                        restored[syn_mask], hq[syn_mask], self.identity_extractor
                    )
                else:
                    losses["identity"] = torch.tensor(0.0, device=self.device)
            except Exception:
                losses["identity"] = torch.tensor(0.0, device=self.device)

        if self.loss_weights["deg_supervised"] > 0:
            if syn_mask.any():
                losses["deg_supervised"] = nn.functional.mse_loss(
                    deg_pred[syn_mask], deg_gt[syn_mask]
                )
            else:
                losses["deg_supervised"] = torch.tensor(0.0, device=self.device)

        total = sum(self.loss_weights[k] * v for k, v in losses.items() if k in self.loss_weights)
        losses["total"] = total
        return losses

    def train(self):
        total_iters = self.config["train"]["total_iters"]
        val_every = self.config["train"]["val_every"]
        save_every = self.config["train"]["save_every"]
        log_every = self.config["train"]["log_every"]

        print(f" Stage 1 training start, total_iters={total_iters} ")
        print(f" Save every {save_every} iters → backup auto ")
        t_start = time.time()
        train_iter = self._infinite_loader(self.train_loader)

        for self.iter in range(self.iter, total_iters):
            batch = next(train_iter)
            lq = batch["lq"].to(self.device, non_blocking=True)
            hq = batch["hq"].to(self.device, non_blocking=True)
            deg_gt = batch["deg_vec"].to(self.device, non_blocking=True)
            is_real = batch["is_real"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                restored, deg_pred = self.forward(lq, deg_gt)
                losses = self.compute_loss(restored, hq, deg_pred, deg_gt, is_real)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            if self.iter % log_every == 0:
                self._log_train(losses)

            if self.iter > 0 and self.iter % val_every == 0:
                val_metrics = self.validate()
                self._log_val(val_metrics)
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self._save_checkpoint("best.pth", is_best=True)

            if self.iter > 0 and self.iter % save_every == 0:
                self._save_checkpoint(f"iter_{self.iter}.pth")

        elapsed = time.time() - t_start
        print(f" Stage 1 training done in {elapsed/3600:.1f}h ")
        self._save_checkpoint("final.pth")

    @staticmethod
    def _infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    @torch.no_grad()
    def validate(self) -> dict:
        self.deg_encoder.eval()
        self.film_adapters.eval()
        self.restore_head.eval()

        total_loss = 0.0
        total_pixel = 0.0
        total_deg = 0.0
        n = 0

        for batch in self.val_loader:
            lq = batch["lq"].to(self.device)
            hq = batch["hq"].to(self.device)
            deg_gt = batch["deg_vec"].to(self.device)
            is_real = batch["is_real"].to(self.device)

            restored, deg_pred = self.forward(lq, deg_gt)
            losses = self.compute_loss(restored, hq, deg_pred, deg_gt, is_real)

            total_loss += losses["total"].item() * lq.shape[0]
            total_pixel += losses.get("pixel", torch.tensor(0)).item() * lq.shape[0]
            total_deg += losses.get("deg_supervised", torch.tensor(0)).item() * lq.shape[0]
            n += lq.shape[0]

        self.deg_encoder.train()
        self.film_adapters.train()
        self.restore_head.train()

        return {
            "loss": total_loss / max(n, 1),
            "pixel": total_pixel / max(n, 1),
            "deg": total_deg / max(n, 1),
        }

    def _log_train(self, losses):
        lr = self.optimizer.param_groups[0]["lr"]
        log_dict = {f"train/{k}": v.item() if torch.is_tensor(v) else v
                    for k, v in losses.items()}
        log_dict["train/lr"] = lr

        for k, v in log_dict.items():
            self.tb.add_scalar(k, v, self.iter)
        if self.use_wandb:
            self.wandb.log(log_dict, step=self.iter)

        if self.iter % (10 * self.config["train"]["log_every"]) == 0:
            loss_str = ", ".join(f"{k}={v.item():.4f}" if torch.is_tensor(v) else f"{k}={v:.4f}"
                                 for k, v in losses.items())
            print(f"[iter {self.iter}] lr={lr:.2e} | {loss_str}")

    def _log_val(self, metrics):
        for k, v in metrics.items():
            self.tb.add_scalar(f"val/{k}", v, self.iter)
        if self.use_wandb:
            self.wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=self.iter)
        print(f"[VAL iter {self.iter}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def _save_checkpoint(self, name: str, is_best: bool = False):
        """FIXED: atomic write + verify + backup."""
        ckpt = {
            "iter": self.iter,
            "best_val_loss": self.best_val_loss,
            "deg_encoder": self.deg_encoder.state_dict(),
            "film_adapters": self.film_adapters.state_dict(),
            "restore_head": self.restore_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }

        #  Main save 
        main_path = self.ckpt_dir / name
        tmp_path = self.ckpt_dir / (name + ".tmp")

        try:
            # Atomic write: tmp → rename
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, main_path)

            # VERIFY
            if not main_path.exists():
                raise RuntimeError(f"Save failed - file not found: {main_path}")
            size_mb = main_path.stat().st_size / 1e6
            if size_mb < 0.1:
                raise RuntimeError(f"Save failed - file too small ({size_mb:.2f}MB): {main_path}")

            print(f"  [✓] MAIN saved: {main_path} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"  [✗] MAIN save FAILED: {e}")

        #  Backup save (luôn luôn save vào backup dir) 
        try:
            backup_path = self.backup_dir / name
            tmp_backup = self.backup_dir / (name + ".tmp")
            torch.save(ckpt, tmp_backup)
            os.replace(tmp_backup, backup_path)

            if backup_path.exists() and backup_path.stat().st_size > 100_000:
                print(f"  [✓] BACKUP saved: {backup_path}")
            else:
                print(f"  [✗] BACKUP save FAILED")
        except Exception as e:
            print(f"  [✗] BACKUP save FAILED: {e}")

        if is_best:
            print(f"  ★ This is BEST checkpoint so far (val_loss={self.best_val_loss:.4f})")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.iter = ckpt["iter"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.deg_encoder.load_state_dict(ckpt["deg_encoder"])
        self.film_adapters.load_state_dict(ckpt["film_adapters"])
        self.restore_head.load_state_dict(ckpt["restore_head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        print(f"[✓] Resumed from {path} at iter {self.iter}")
