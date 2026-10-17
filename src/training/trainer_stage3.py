"""
Stage 3 Trainer v2 — Hard Fine-tuning với 5 cải tiến.


THIẾT KẾ (theo Chương 3 quyển báo cáo)

Bám sát thiết kế gốc:
    - Full unfreeze Real-ESRGAN + GFPGAN
    - Transformer fusion 4 blocks (Stage 2: 2 → Stage 3: 4)
    - LR 5e-5
    - Hard degradation
    - Load Stage 2 ckpt làm starting point
    - Domain shift to Asian (UTKFace race=2)


5 CẢI TIẾN so với v1 (tăng khả thi từ ~50% lên ~80%)

1. **Mix source dataset** (anti catastrophic forgetting + val consistency):
   Train 50% UTKFace Asian + 50% FFHQ subset (~10k).
   Vẫn match thiết kế "shift toward Asian" nhưng có anchor FFHQ.

2. **Mix degradation curriculum** (tránh shock "medium → hard"):
   Train 70% hard + 30% medium thay vì 100% hard.

3. **EMA (Exponential Moving Average)** weights cho Generator:
   Shadow copy với decay 0.999, dùng cho validation và inference.
   Industry standard (Real-ESRGAN, GFPGAN, StyleGAN2).

4. **R1 gradient penalty** cho Discriminator:
   Lazy regularization (mỗi 16 step), weight 10.0.
   Ổn định GAN hơn LSGAN thuần khi full unfreeze.

5. **Save frequency 500 iter** thay vì 1000:
   10 candidates thay vì 5 → 2x cơ hội bắt sweet spot.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.training.trainer_stage2 import Stage2Trainer
from src.models.degradation_encoder import DegradationEncoder
from src.models.real_esrgan.rrdb_with_film import (
    RRDBNet_FiLM,
    load_realesrgan_pretrained,
    freeze_rrdbnet_body,
    count_params,
)
from src.models.gfpgan.gfpgan_wrapper import build_face_restorer, GFPGANWrapper, SimpleFaceRestorer
from src.models.transformer_fusion import TransformerFusion
from src.models.discriminator import UNetDiscriminator
from src.utils.data_loader import ValDataset
from src.utils.mixed_dataset_utils import build_stage3_mixed_train_dataset
from src.losses.adversarial import GANLoss



# EMA helper — shadow weights cho Generator

class MultiModelEMA:
    """EMA shadow weights cho nhiều model components.

    Usage:
        ema = MultiModelEMA({'bg_net': bg_net, ...}, decay=0.999)
        # mỗi iter train:
        ema.update({'bg_net': bg_net, ...})
        # khi validate/save:
        ema.swap_in({'bg_net': bg_net, ...})
        ... val/save ...
        ema.swap_out({'bg_net': bg_net, ...})
    """

    def __init__(self, models: dict, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, model in models.items():
            self.shadow[name] = {
                n: p.data.clone().detach()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
        self.backup = None
        print(f"[EMA] Initialized for {list(models.keys())}, decay={decay}")

    @torch.no_grad()
    def update(self, models: dict):
        for name, model in models.items():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow[name]:
                    self.shadow[name][n].mul_(self.decay).add_(
                        p.data, alpha=1 - self.decay
                    )

    @torch.no_grad()
    def swap_in(self, models: dict):
        """Swap shadow weights vào model, lưu original để restore."""
        self.backup = {}
        for name, model in models.items():
            self.backup[name] = {}
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow[name]:
                    self.backup[name][n] = p.data.clone()
                    p.data.copy_(self.shadow[name][n])

    @torch.no_grad()
    def swap_out(self, models: dict):
        if self.backup is None:
            return
        for name, model in models.items():
            for n, p in model.named_parameters():
                if n in self.backup.get(name, {}):
                    p.data.copy_(self.backup[name][n])
        self.backup = None

    def state_dict(self) -> dict:
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state: dict):
        self.decay = state['decay']
        self.shadow = state['shadow']


class Stage3Trainer(Stage2Trainer):
    """Stage 3 hard fine-tuning với 5 cải tiến."""

    def __init__(self, config: dict, resume: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_seed()
        self._setup_output_dirs()

        self._build_models()
        self._load_stage2_warmup()
        self._setup_freeze_unfreeze()
        self._build_optimizers()
        self._build_losses()
        self._build_data()
        self._build_logging()
        self._build_ema()

        self.iter = 0
        self.best_val_loss = float("inf")
        self.freeze_d_iters = self.config["train"].get("freeze_d_first_iters", 0)

        # R1 config
        r1_cfg = self.config["loss"].get("r1_penalty", {})
        self.r1_enable = r1_cfg.get("enable", True)
        self.r1_weight = r1_cfg.get("weight", 10.0)
        self.r1_every = r1_cfg.get("every_n_steps", 16)

        if resume:
            self._load_stage3_checkpoint(resume)

    def _setup_output_dirs(self):
        self.ckpt_dir = Path(self.config["log"]["output_dir"]).resolve()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = Path("/kaggle/working/ckpt_backup_stage3").resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = Path(self.config["log"]["tensorboard_dir"]).resolve()
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print("STAGE 3 CHECKPOINT PATHS (ABSOLUTE):")
        print(f"  Main:   {self.ckpt_dir}")
        print(f"  Backup: {self.backup_dir}")
        print(f"  TB:     {self.tb_dir}")
        print("=" * 60)

    def _build_models(self):
        mcfg = self.config["model"]
        self.deg_encoder = DegradationEncoder(
            out_dim=mcfg["degradation_encoder"]["out_dim"],
        ).to(self.device)
        self.bg_net = RRDBNet_FiLM(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
            num_grow_ch=32, scale=4, film_last_n=4,
            cond_dim=mcfg["degradation_encoder"]["out_dim"],
        ).to(self.device)
        gfpgan_path = mcfg["gfpgan"]["pretrained"]
        self.face_net = build_face_restorer(
            gfpgan_path=gfpgan_path,
            cond_dim=mcfg["degradation_encoder"]["out_dim"],
            fallback_to_simple=True,
            device=self.device,
        ).to(self.device)
        if isinstance(self.face_net, GFPGANWrapper):
            print("[Face] Using GFPGAN backbone")
        elif isinstance(self.face_net, SimpleFaceRestorer):
            print("[Face] Using SimpleFaceRestorer (fallback)")
        tf_cfg = mcfg["transformer_fusion"]
        self.fusion = TransformerFusion(
            dim=tf_cfg.get("dim", 64),
            num_blocks=tf_cfg.get("num_blocks", 4),
            num_heads=tf_cfg.get("num_heads", 8),
            in_channels_face=3, in_channels_bg=3, out_channels=3,
        ).to(self.device)
        self.discriminator = UNetDiscriminator(
            in_channels=3, base_channels=64, use_spectral_norm=True,
        ).to(self.device)
        print("\n=== Stage 3 Models ===")
        for name, m in [
            ("deg_encoder", self.deg_encoder), ("bg_net", self.bg_net),
            ("face_net", self.face_net), ("fusion", self.fusion),
            ("discriminator", self.discriminator),
        ]:
            tot = sum(p.numel() for p in m.parameters())
            print(f"  {name:<15} total={tot/1e6:.2f}M params")

    def _load_stage1_warmup(self):
        pass

    def _load_stage2_warmup(self):
        stage2_path = self.config["model"].get("stage2_checkpoint", "")
        if not stage2_path or not Path(stage2_path).exists():
            raise FileNotFoundError(f"Stage 2 ckpt missing: {stage2_path}")
        print(f"\n[Stage 2 Warmup] Loading {stage2_path}")
        ckpt = torch.load(stage2_path, map_location=self.device, weights_only=False)
        print(f"  Stage 2 iter: {ckpt.get('iter')}, best_val_loss: {ckpt.get('best_val_loss')}")

        if "deg_encoder" in ckpt:
            self.deg_encoder.load_state_dict(ckpt["deg_encoder"])
            print("  ✓ deg_encoder")
        if "bg_net" in ckpt:
            self.bg_net.load_state_dict(ckpt["bg_net"])
            print("  ✓ bg_net")
        if "face_net" in ckpt:
            try:
                self.face_net.load_state_dict(ckpt["face_net"])
                print("  ✓ face_net (strict)")
            except RuntimeError:
                missing, unexpected = self.face_net.load_state_dict(
                    ckpt["face_net"], strict=False)
                print(f"  ✓ face_net (non-strict: {len(missing)} missing, {len(unexpected)} unexpected)")
        if "fusion" in ckpt:
            self._load_fusion_with_expansion(
                ckpt["fusion"],
                strategy=self.config["model"].get("fusion_expansion_strategy", "copy"),
            )
        if "discriminator" in ckpt:
            self.discriminator.load_state_dict(ckpt["discriminator"])
            print("  ✓ discriminator")

        self.stage2_best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"\n  → Stage 3 baseline = {self.stage2_best_val_loss:.4f}")

    def _load_fusion_with_expansion(self, fusion_sd_stage2: dict, strategy: str = "copy"):
        new_sd = self.fusion.state_dict()
        n_blocks = self.config["model"]["transformer_fusion"]["num_blocks"]
        loaded, skipped = [], []
        for key, val in fusion_sd_stage2.items():
            if key in new_sd and new_sd[key].shape == val.shape:
                new_sd[key] = val
                loaded.append(key)
            else:
                skipped.append(key)
        if strategy == "copy":
            for new_idx in range(2, n_blocks):
                src_idx = new_idx % 2
                prefix_src = f"blocks.{src_idx}."
                prefix_new = f"blocks.{new_idx}."
                for key in fusion_sd_stage2:
                    if key.startswith(prefix_src):
                        new_key = prefix_new + key[len(prefix_src):]
                        if new_key in new_sd:
                            new_sd[new_key] = fusion_sd_stage2[key].clone()
                print(f"    Block {new_idx} ← cloned from block {src_idx}")
        self.fusion.load_state_dict(new_sd)
        print(f"  ✓ fusion ({strategy}, loaded {len(loaded)}, skipped {len(skipped)})")

    def _setup_freeze_unfreeze(self):
        mcfg = self.config["model"]
        unfreeze_n = mcfg["real_esrgan"].get("unfreeze_n", 23)
        if unfreeze_n >= 23:
            for p in self.bg_net.parameters():
                p.requires_grad = True
            print(f"\n[Freeze] bg_net: FULL UNFREEZE")
        else:
            freeze_rrdbnet_body(self.bg_net, unfreeze_last_n=unfreeze_n)
            print(f"\n[Freeze] bg_net: unfreeze last {unfreeze_n} RRDB")

        gfpgan_strategy = mcfg["gfpgan"].get("unfreeze_strategy", "full")
        gfpgan_n = mcfg["gfpgan"].get("unfreeze_n", 8)
        if isinstance(self.face_net, GFPGANWrapper):
            if gfpgan_strategy == "full":
                for p in self.face_net.parameters():
                    p.requires_grad = True
                print(f"[Freeze] face_net (GFPGAN): FULL UNFREEZE")
            else:
                self._unfreeze_gfpgan_last_n(gfpgan_n)
                print(f"[Freeze] face_net (GFPGAN): last {gfpgan_n}")
                for module in [self.face_net.pre_conv1, self.face_net.pre_film,
                               self.face_net.pre_conv2, self.face_net.post_conv1,
                               self.face_net.post_film, self.face_net.post_conv2]:
                    for p in module.parameters():
                        p.requires_grad = True

        for p in self.fusion.parameters():
            p.requires_grad = True
        for p in self.deg_encoder.parameters():
            p.requires_grad = True

        total_g = sum(p.numel() for m in [
            self.deg_encoder, self.bg_net, self.face_net, self.fusion
        ] for p in m.parameters() if p.requires_grad)
        print(f"\n[Freeze] TOTAL G trainable: {total_g/1e6:.2f}M params")

    def _unfreeze_gfpgan_last_n(self, n: int):
        for p in self.face_net.gfpgan.parameters():
            p.requires_grad = False
        gfpgan = self.face_net.gfpgan
        targets = []
        if hasattr(gfpgan, 'conv_body_up'):
            cbu = gfpgan.conv_body_up
            if hasattr(cbu, '__len__'):
                total = len(cbu)
                for i in range(max(0, total - n), total):
                    targets.append(cbu[i])
        if hasattr(gfpgan, 'toRGB'):
            targets.append(gfpgan.toRGB)
        for m in targets:
            for p in m.parameters():
                p.requires_grad = True

    def _build_data(self):
        """MIX 2 sources × 2 deg levels."""
        dcfg = self.config["data"]
        asian_dir = dcfg["asian_dir"]
        ffhq_dir = dcfg["ffhq_dir"]
        ratio_asian = dcfg.get("ratio_asian", 0.5)
        ratio_hard = dcfg.get("ratio_hard", 0.7)
        ffhq_max = dcfg.get("ffhq_max_samples", 10000)

        print(f"\n[Train data] Mixed sources + mixed degradation")
        train_ds = build_stage3_mixed_train_dataset(
            asian_dir=asian_dir,
            ffhq_dir=ffhq_dir,
            image_size=dcfg["image_size"],
            ratio_asian=ratio_asian,
            ratio_hard=ratio_hard,
            ffhq_max_samples=ffhq_max,
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=dcfg["batch_size"], shuffle=True,
            num_workers=dcfg["num_workers"], pin_memory=True,
            drop_last=True, persistent_workers=dcfg["num_workers"] > 0,
        )

        val_levels = dcfg.get("val_degradation_levels", ["medium", "hard"])
        val_max = dcfg.get("val_max_samples", 150)
        self.val_loaders = {}
        for level in val_levels:
            val_ds = ValDataset(
                hq_dir=dcfg["val_dir"],
                image_size=dcfg["image_size"],
                degradation_level=level,
                max_samples=val_max,
            )
            self.val_loaders[level] = DataLoader(
                val_ds, batch_size=dcfg["batch_size"], shuffle=False,
                num_workers=2, pin_memory=True,
            )
            print(f"[Val/{level}] {len(val_ds)} samples")

    def _build_ema(self):
        ema_cfg = self.config["train"].get("ema", {})
        self.ema_enable = ema_cfg.get("enable", True)
        if not self.ema_enable:
            self.ema = None
            print("[EMA] Disabled")
            return
        decay = ema_cfg.get("decay", 0.999)
        self.ema = MultiModelEMA(
            models=self._ema_models_dict(),
            decay=decay,
        )

    def _ema_models_dict(self):
        return {
            'deg_encoder': self.deg_encoder,
            'bg_net': self.bg_net,
            'face_net': self.face_net,
            'fusion': self.fusion,
        }

    def _compute_r1_penalty(self, real_inputs: torch.Tensor) -> torch.Tensor:
        """R1 gradient penalty: ||∇_x D(x)||^2 evaluated on real samples."""
        real_inputs = real_inputs.detach().clone().requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=False):
            d_real = self.discriminator(real_inputs.float())
        grad = torch.autograd.grad(
            outputs=d_real.sum(),
            inputs=real_inputs,
            create_graph=True,
            only_inputs=True,
        )[0]
        r1 = grad.pow(2).reshape(grad.size(0), -1).sum(dim=1).mean()
        return r1

    @torch.no_grad()
    def validate(self):
        # Swap in EMA for validation
        if self.ema_enable:
            self.ema.swap_in(self._ema_models_dict())

        for m in [self.deg_encoder, self.bg_net, self.face_net, self.fusion]:
            m.eval()

        all_metrics = {}
        total_loss_sum = 0.0
        n_sets = 0

        for level, loader in self.val_loaders.items():
            total, total_pixel, total_deg = 0.0, 0.0, 0.0
            n = 0
            for batch in loader:
                lq = batch["lq"].to(self.device)
                hq = batch["hq"].to(self.device)
                deg_gt = batch["deg_vec"].to(self.device)
                is_real = batch["is_real"].to(self.device)
                output = self.forward_generator(lq)
                losses = self.compute_g_loss(output, hq, deg_gt, is_real)
                total += losses["total"].item() * lq.shape[0]
                total_pixel += losses.get("pixel", torch.tensor(0)).item() * lq.shape[0]
                total_deg += losses.get("deg_supervised", torch.tensor(0)).item() * lq.shape[0]
                n += lq.shape[0]
            avg_loss = total / max(n, 1)
            all_metrics[f"loss_{level}"] = avg_loss
            all_metrics[f"pixel_{level}"] = total_pixel / max(n, 1)
            all_metrics[f"deg_{level}"] = total_deg / max(n, 1)
            total_loss_sum += avg_loss
            n_sets += 1

        for m in [self.deg_encoder, self.bg_net, self.face_net, self.fusion]:
            m.train()

        # Restore non-EMA weights
        if self.ema_enable:
            self.ema.swap_out(self._ema_models_dict())

        all_metrics["loss"] = total_loss_sum / max(n_sets, 1)
        return all_metrics

    def _log_val(self, metrics):
        for k, v in metrics.items():
            self.tb.add_scalar(f"val/{k}", v, self.iter)
        msg = f"[VAL iter {self.iter}] loss_avg={metrics['loss']:.4f}"
        for level in ["medium", "hard"]:
            if f"loss_{level}" in metrics:
                msg += f" | {level}: loss={metrics[f'loss_{level}']:.4f}"
                msg += f" pixel={metrics[f'pixel_{level}']:.4f}"
        print(msg)

    def train(self):
        total_iters = self.config["train"]["total_iters"]
        val_every = self.config["train"]["val_every"]
        save_every = self.config["train"]["save_every"]
        log_every = self.config["train"]["log_every"]

        print(f"\n=== Stage 3 v2 training start ===")
        print(f"  total_iters       = {total_iters}")
        print(f"  freeze_d first    = {self.freeze_d_iters} iters")
        ema_status = f"ON (decay={self.ema.decay})" if self.ema_enable else "OFF"
        r1_status = f"ON (every {self.r1_every}, w={self.r1_weight})" if self.r1_enable else "OFF"
        print(f"  EMA               = {ema_status}")
        print(f"  R1 penalty        = {r1_status}")
        print(f"  save_every        = {save_every}")
        print(f"  val_every         = {val_every}")
        print(f"=== Stage 2 baseline val_loss = {self.stage2_best_val_loss:.4f} ===\n")

        t_start = time.time()
        train_iter = self._infinite_loader(self.train_loader)

        for self.iter in range(self.iter, total_iters):
            batch = next(train_iter)
            lq = batch["lq"].to(self.device, non_blocking=True)
            hq = batch["hq"].to(self.device, non_blocking=True)
            deg_gt = batch["deg_vec"].to(self.device, non_blocking=True)
            is_real = batch["is_real"].to(self.device, non_blocking=True)

            # Generator step
            self.optimizer_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.forward_generator(lq)
                losses_g = self.compute_g_loss(output, hq, deg_gt, is_real)
            self.scaler_g.scale(losses_g["total"]).backward()
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
            self.scheduler_g.step()

            # EMA update
            if self.ema_enable:
                self.ema.update(self._ema_models_dict())

            # Discriminator step + R1
            losses_d = None
            should_train_d = (
                self.gan_criterion is not None and
                self.iter >= self.freeze_d_iters
            )
            if should_train_d:
                self.optimizer_d.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss_d = self.compute_d_loss(output, hq)
                if loss_d is not None:
                    r1_val = None
                    if self.r1_enable and self.iter % self.r1_every == 0 and self.iter > 0:
                        r1_val = self._compute_r1_penalty(hq)
                        # Lazy reg: scale by every_n_steps
                        loss_d_total = loss_d + self.r1_weight * self.r1_every * 0.5 * r1_val
                    else:
                        loss_d_total = loss_d
                    self.scaler_d.scale(loss_d_total).backward()
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                    self.scheduler_d.step()
                    losses_d = {"d_total": loss_d}
                    if r1_val is not None:
                        losses_d["r1"] = r1_val
            elif self.iter == 0 and self.freeze_d_iters > 0:
                print(f"[Init] Freezing D for first {self.freeze_d_iters} iters")
            elif self.iter == self.freeze_d_iters and self.freeze_d_iters > 0:
                print(f"[iter {self.iter}] D unfrozen")

            if self.iter % log_every == 0:
                self._log_train(losses_g, losses_d)

            if self.iter > 0 and self.iter % val_every == 0:
                val_metrics = self.validate()
                self._log_val(val_metrics)
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self._save_checkpoint("best.pth", is_best=True)

            if self.iter > 0 and self.iter % save_every == 0:
                self._save_checkpoint(f"iter_{self.iter}.pth")

        elapsed = time.time() - t_start
        print(f"\n=== Stage 3 done in {elapsed/3600:.1f}h ===")
        self._save_checkpoint("final.pth")

        print(f"\n=== SUMMARY ===")
        print(f"  Stage 2 baseline:    {self.stage2_best_val_loss:.4f}")
        print(f"  Stage 3 best (EMA):  {self.best_val_loss:.4f}")
        improvement = self.stage2_best_val_loss - self.best_val_loss
        print(f"  Improvement:         {improvement:+.4f} "
              f"({'BETTER ✓' if improvement > 0 else 'WORSE — use Stage 2'})")

    def _save_checkpoint(self, name: str, is_best: bool = False):
        ckpt = {
            "iter": self.iter,
            "best_val_loss": self.best_val_loss,
            "stage2_baseline_val_loss": getattr(self, 'stage2_best_val_loss', None),
            "deg_encoder": self.deg_encoder.state_dict(),
            "bg_net": self.bg_net.state_dict(),
            "face_net": self.face_net.state_dict(),
            "fusion": self.fusion.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
            "config": self.config,
            "stage": 3,
        }
        if self.ema_enable and self.ema is not None:
            ckpt["ema"] = self.ema.state_dict()

        for d in [self.ckpt_dir, self.backup_dir]:
            try:
                path = d / name
                tmp = d / (name + ".tmp")
                torch.save(ckpt, tmp)
                os.replace(tmp, path)
                if path.exists() and path.stat().st_size > 1_000_000:
                    size_mb = path.stat().st_size / 1e6
                    label = "MAIN" if d == self.ckpt_dir else "BACKUP"
                    print(f"  [✓] {label} saved: {path} ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"  [✗] Save failed at {d}: {e}")

        if is_best:
            print(f"  ★ BEST so far (val_loss={self.best_val_loss:.4f})")

    def _load_stage3_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.iter = ckpt["iter"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.stage2_best_val_loss = ckpt.get("stage2_baseline_val_loss", float("inf"))
        self.deg_encoder.load_state_dict(ckpt["deg_encoder"])
        self.bg_net.load_state_dict(ckpt["bg_net"])
        self.face_net.load_state_dict(ckpt["face_net"])
        self.fusion.load_state_dict(ckpt["fusion"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        self.scheduler_g.load_state_dict(ckpt["scheduler_g"])
        self.scheduler_d.load_state_dict(ckpt["scheduler_d"])
        if "ema" in ckpt and self.ema is not None:
            self.ema.load_state_dict(ckpt["ema"])
            print("  ✓ EMA shadow weights restored")
        print(f"[✓] Resumed Stage 3 from {path} at iter {self.iter}")
