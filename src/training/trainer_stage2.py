"""
Stage 2 Trainer — Medium Fine-tuning với Real-ESRGAN + GFPGAN backbone.


KHÁC BIỆT SO VỚI STAGE 1


Stage 1 → Stage 2:
    - Lightweight restore head → THẬT SỰ DÙNG Real-ESRGAN + GFPGAN backbone
    - Train Degradation Encoder từ scratch → LOAD từ Stage 1 + fine-tune
    - Train FiLM từ scratch → LOAD từ Stage 1 (channels 64,128,256) + thêm
      FiLM mới gắn trong backbone
    - Không có Transformer fusion → BẬT Transformer fusion (2 blocks)
    - Không có GAN → BẬT GAN với discriminator
    - Không có adaptive weighting → BẬT adaptive weighting
    - Light degradation → MEDIUM degradation


PIPELINE Forward


    LQ input (256x256)
         │
         ├──────────────────────────────────────────┐
         │                                          │
    ┌────▼────────────┐                             │
    │ Degradation     │                             │
    │ Encoder         │  → deg_vec (B, 8)           │
    │ (load from S1)  │                             │
    └────┬────────────┘                             │
         │                                          │
    deg_vec ◄─────────────────────────────┐         │
         │                                │         │
         ▼                                │         │
    ┌────────────────────────────┐  ┌────▼────────────────┐
    │ Real-ESRGAN RRDBNet_FiLM   │  │ GFPGAN Wrapper       │
    │ (load pretrained)          │  │ (load pretrained)    │
    │ Input: LQ (256x256)        │  │ Input: face (256x256)│
    │ Output: BG (1024x1024)     │  │ Output: face (256x256)│
    │ Note: x4 upsample          │  │                      │
    └────┬───────────────────────┘  └─────┬────────────────┘
         │                                 │
         ▼                                 ▼
    BG enhanced (1024x1024)         Face restored (256x256 - skip upsample)
         │                                 │
         │ (downsample x4 về 256x256 hoặc │
         │  resize face về 1024x1024)     │
         │                                 │
         └─────────┐         ┌─────────────┘
                   ▼         ▼
            ┌───────────────────┐
            │ Transformer       │
            │ Fusion            │
            │ (Cross-Attention) │
            └─────────┬─────────┘
                      ▼
            Final restored (256x256)
                      │
                      ▼
              ┌───────────────┐
              │ Discriminator │  → real/fake
              └───────────────┘

NOTE QUAN TRỌNG VỀ RESOLUTION:
    - Stage 2 train ở 256x256 input để fit T4 15GB.
    - Real-ESRGAN x4 → output 1024x1024 → cần downsample về 256 cho fusion.
    - HOẶC ta dùng Real-ESRGAN ở chế độ x1 (no upsample) — đơn giản hơn.

    LỰA CHỌN: dùng Real-ESRGAN ở chế độ "restore-only" (không upsample).
    Cách làm: lấy feature trước upsampling layers.
    
    Đơn giản hơn nữa: input 256, output 256 cho cả 2 branches.
    Real-ESRGAN gốc upsample x4 → ta resize output về 256.



LOSS FUNCTION


Total loss = 
    w_pixel * L1(restored, hq) +
    w_perceptual * VGG_Loss(restored, hq) +
    w_identity * ArcFace_Loss(restored, hq) +
    w_adv * GAN_Loss(restored) +              # mới
    w_deg * MSE(pred_deg, gt_deg)

Mỗi weight có thể adaptive theo severity (AdaptiveLossWeighter).


TRAINING STRATEGY


Optimizer:
    - Generator: AdamW, LR=1e-4 (1/2 của stage 1)
    - Discriminator: AdamW, LR=1e-4
    - Degradation Encoder: LR=5e-5 (thấp hơn vì đã pretrained)

Scheduler:
    - Linear warmup (500 iters) + Cosine decay
    - Total iter: 40000

Mixed precision: FP16 (T4 hỗ trợ)

Author: Đặng Thành Tựu
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

# Local imports
from src.models.degradation_encoder import DegradationEncoder
from src.models.film_layer import FiLMLayer
from src.models.transformer_fusion import TransformerFusion
from src.models.real_esrgan.rrdb_with_film import (
    RRDBNet_FiLM, load_realesrgan_pretrained, freeze_rrdbnet_body, count_params,
)
from src.models.gfpgan.gfpgan_wrapper import build_face_restorer, GFPGANWrapper, SimpleFaceRestorer
from src.models.discriminator import UNetDiscriminator
from src.utils.data_loader import build_train_dataset, build_val_dataset
from src.losses.adversarial import GANLoss


class Stage2Trainer:
    """Stage 2 medium fine-tuning trainer."""

    def __init__(self, config: dict, resume: Optional[str] = None):
        """
        Args:
            config: stage2 YAML config
            resume: path đến checkpoint Stage 2 (nếu resume từ stage 2)
                    KHÔNG dùng path Stage 1 ở đây — Stage 1 load qua config.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_seed()
        self._setup_output_dirs()

        # Build components theo thứ tự
        self._build_models()
        self._load_stage1_warmup()       # Quan trọng: load Degradation Encoder + FiLM từ Stage 1
        self._setup_freeze_unfreeze()
        self._build_optimizers()
        self._build_losses()
        self._build_data()
        self._build_logging()

        # State
        self.iter = 0
        self.best_val_loss = float("inf")

        if resume:
            self._load_stage2_checkpoint(resume)

    
    # SETUP
    
    def _setup_seed(self):
        seed = self.config["train"].get("seed", 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _setup_output_dirs(self):
        """Force absolute path."""
        self.ckpt_dir = Path(self.config["log"]["output_dir"]).resolve()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.backup_dir = Path("/kaggle/working/ckpt_backup_stage2").resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.tb_dir = Path(self.config["log"]["tensorboard_dir"]).resolve()
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("STAGE 2 CHECKPOINT PATHS (ABSOLUTE):")
        print(f"  Main:   {self.ckpt_dir}")
        print(f"  Backup: {self.backup_dir}")
        print(f"  TB:     {self.tb_dir}")
        print("=" * 60)

    
    # MODELS
    
    def _build_models(self):
        """Build tất cả components Stage 2."""
        mcfg = self.config["model"]

        # ---------- 1. Degradation Encoder ----------
        self.deg_encoder = DegradationEncoder(
            out_dim=mcfg["degradation_encoder"]["out_dim"],
        ).to(self.device)

        # ---------- 2. Real-ESRGAN backbone (background) ----------
        self.bg_net = RRDBNet_FiLM(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
            film_last_n=4,
            cond_dim=mcfg["degradation_encoder"]["out_dim"],
        ).to(self.device)

        # Load pretrained Real-ESRGAN
        real_esrgan_path = mcfg["real_esrgan"]["pretrained"]
        if real_esrgan_path and Path(real_esrgan_path).exists():
            load_realesrgan_pretrained(self.bg_net, real_esrgan_path)
        else:
            print(f"[WARNING] Real-ESRGAN pretrained không tồn tại: {real_esrgan_path}")

        # ---------- 3. GFPGAN backbone (face) ----------
        gfpgan_path = mcfg["gfpgan"]["pretrained"]
        self.face_net = build_face_restorer(
            gfpgan_path=gfpgan_path,
            cond_dim=mcfg["degradation_encoder"]["out_dim"],
            fallback_to_simple=True,
            device=self.device,
        ).to(self.device)

        # Báo cáo loại face restorer
        if isinstance(self.face_net, GFPGANWrapper):
            print("[Face] Using GFPGAN backbone (pretrained)")
        elif isinstance(self.face_net, SimpleFaceRestorer):
            print("[Face] Using SimpleFaceRestorer (fallback)")

        # ---------- 4. Transformer Fusion ----------
        tf_cfg = mcfg["transformer_fusion"]
        self.fusion = TransformerFusion(
            dim=tf_cfg.get("dim", 64),
            num_blocks=tf_cfg.get("num_blocks", 2),
            num_heads=tf_cfg.get("num_heads", 8),
            in_channels_face=3,
            in_channels_bg=3,
            out_channels=3,
        ).to(self.device)

        # ---------- 5. Discriminator ----------
        self.discriminator = UNetDiscriminator(
            in_channels=3,
            base_channels=64,
            use_spectral_norm=True,
        ).to(self.device)

        # Tổng kết
        for name, m in [
            ("deg_encoder", self.deg_encoder),
            ("bg_net", self.bg_net),
            ("face_net", self.face_net),
            ("fusion", self.fusion),
            ("discriminator", self.discriminator),
        ]:
            tr, tot = count_params(m) if hasattr(m, '__call__') else (
                sum(p.numel() for p in m.parameters() if p.requires_grad),
                sum(p.numel() for p in m.parameters()),
            )
            print(f"  {name:<15} total={tot/1e6:.2f}M trainable={tr/1e6:.2f}M")

    def _load_stage1_warmup(self):
        """Load Degradation Encoder + FiLM weights từ Stage 1 checkpoint."""
        stage1_ckpt_path = self.config["model"].get("stage1_checkpoint")
        if not stage1_ckpt_path or not Path(stage1_ckpt_path).exists():
            print(f"[WARNING] Stage 1 checkpoint không tồn tại: {stage1_ckpt_path}")
            print("           Degradation Encoder + FiLM sẽ train từ scratch.")
            return

        print(f"\n[Stage 1 Warmup] Loading {stage1_ckpt_path}")
        ckpt = torch.load(stage1_ckpt_path, map_location=self.device, weights_only=False)

        # Load Degradation Encoder (giống kiến trúc → load nguyên xi)
        if "deg_encoder" in ckpt:
            self.deg_encoder.load_state_dict(ckpt["deg_encoder"])
            print("  ✓ Degradation Encoder loaded from Stage 1")
        else:
            print("  ✗ deg_encoder không có trong checkpoint")

        # FiLM adapters của Stage 1 có channels (64, 128, 256)
        # Stage 2 FiLM nằm trong backbone (channels khác) → KHÔNG load trực tiếp.
        # Chỉ load Degradation Encoder. FiLM trong backbone init random.
        print("  ℹ FiLM adapters in backbone init random (different channels from Stage 1)")

        print(f"  Stage 1 best val_loss: {ckpt.get('best_val_loss', 'N/A')}")

    def _setup_freeze_unfreeze(self):
        """Apply freeze strategy theo config."""
        mcfg = self.config["model"]

        # ---------- Real-ESRGAN ----------
        unfreeze_n = mcfg["real_esrgan"].get("unfreeze_n", 4)
        freeze_rrdbnet_body(self.bg_net, unfreeze_last_n=unfreeze_n)
        tr, tot = count_params(self.bg_net)
        print(f"\n[Freeze] bg_net: trainable {tr/1e6:.2f}M / {tot/1e6:.2f}M")

        # ---------- GFPGAN ----------
        # Nếu là GFPGANWrapper, freeze backbone, unfreeze pre/post FiLM
        if isinstance(self.face_net, GFPGANWrapper):
            for p in self.face_net.gfpgan.parameters():
                p.requires_grad = False
            # Pre/post FiLM giữ trainable
            for module in [self.face_net.pre_conv1, self.face_net.pre_film,
                           self.face_net.pre_conv2, self.face_net.post_conv1,
                           self.face_net.post_film, self.face_net.post_conv2]:
                for p in module.parameters():
                    p.requires_grad = True
        # SimpleFaceRestorer → tất cả trainable (default)

        tr = sum(p.numel() for p in self.face_net.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.face_net.parameters())
        print(f"[Freeze] face_net: trainable {tr/1e6:.2f}M / {tot/1e6:.2f}M")

    
    # OPTIMIZERS
    
    def _build_optimizers(self):
        ocfg = self.config["optim"]

        # Generator params (mọi thứ TRỪ discriminator)
        # Với 3 LR khác nhau:
        # - deg_encoder: thấp (đã pretrained)
        # - bg_net + face_net: trung bình
        # - fusion + film: cao (mới)
        g_params = [
            {
                "params": filter(lambda p: p.requires_grad, self.deg_encoder.parameters()),
                "lr": ocfg["lr_deg_encoder"],
            },
            {
                "params": filter(lambda p: p.requires_grad, self.bg_net.parameters()),
                "lr": ocfg["lr"],
            },
            {
                "params": filter(lambda p: p.requires_grad, self.face_net.parameters()),
                "lr": ocfg["lr"],
            },
            {
                "params": self.fusion.parameters(),
                "lr": ocfg["lr_new"],   # cao hơn vì mới
            },
        ]

        self.optimizer_g = AdamW(
            g_params,
            betas=tuple(ocfg["betas"]),
            weight_decay=ocfg["weight_decay"],
        )

        # Discriminator optimizer
        self.optimizer_d = AdamW(
            self.discriminator.parameters(),
            lr=ocfg["lr_d"],
            betas=tuple(ocfg["betas"]),
            weight_decay=ocfg["weight_decay"],
        )

        # Schedulers
        sched_cfg = ocfg["scheduler"]
        total_iters = sched_cfg["total_iters"]
        warmup_iters = sched_cfg["warmup_iters"]

        warmup_g = LinearLR(self.optimizer_g, start_factor=0.01, total_iters=warmup_iters)
        cosine_g = CosineAnnealingLR(self.optimizer_g, T_max=total_iters - warmup_iters)
        self.scheduler_g = SequentialLR(self.optimizer_g, [warmup_g, cosine_g],
                                         milestones=[warmup_iters])

        warmup_d = LinearLR(self.optimizer_d, start_factor=0.01, total_iters=warmup_iters)
        cosine_d = CosineAnnealingLR(self.optimizer_d, T_max=total_iters - warmup_iters)
        self.scheduler_d = SequentialLR(self.optimizer_d, [warmup_d, cosine_d],
                                         milestones=[warmup_iters])

        # Mixed precision
        self.use_amp = self.config["train"].get("mixed_precision", True)
        self.scaler_g = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scaler_d = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    
    # LOSSES
    
    def _build_losses(self):
        lcfg = self.config["loss"]
        self.loss_weights = {
            "pixel": lcfg["pixel_l1"]["weight"] if lcfg["pixel_l1"]["enable"] else 0.0,
            "perceptual": lcfg["perceptual_vgg"]["weight"] if lcfg["perceptual_vgg"]["enable"] else 0.0,
            "identity": lcfg["identity_arcface"]["weight"] if lcfg["identity_arcface"]["enable"] else 0.0,
            "adversarial": lcfg["adversarial"]["weight"] if lcfg["adversarial"]["enable"] else 0.0,
            "deg_supervised": lcfg["degradation_supervised"]["weight"] if lcfg["degradation_supervised"]["enable"] else 0.0,
        }

        # Pixel loss
        self.pixel_criterion = nn.L1Loss()

        # Perceptual VGG
        self.perceptual_criterion = None
        if self.loss_weights["perceptual"] > 0:
            from src.losses.perceptual import VGGPerceptualLoss
            self.perceptual_criterion = VGGPerceptualLoss().to(self.device).eval()
            for p in self.perceptual_criterion.parameters():
                p.requires_grad = False

        # Identity ArcFace
        self.identity_extractor = None
        if self.loss_weights["identity"] > 0:
            try:
                from src.losses.identity import ArcFaceFeatureExtractor
                self.identity_extractor = ArcFaceFeatureExtractor()
                self.identity_extractor.eval()
                print("[Identity] ArcFace loaded")
            except Exception as e:
                print(f"[Identity] Disabled: {e}")
                self.loss_weights["identity"] = 0.0

        # GAN loss
        self.gan_criterion = None
        if self.loss_weights["adversarial"] > 0:
            gan_type = lcfg["adversarial"].get("gan_type", "lsgan")
            self.gan_criterion = GANLoss(gan_type=gan_type)
            print(f"[GAN] {gan_type} loss enabled")

    
    # DATA
    
    def _build_data(self):
        dcfg = self.config["data"]
        train_ds = build_train_dataset(self.config)
        val_ds = build_val_dataset(self.config)

        self.train_loader = DataLoader(
            train_ds, batch_size=dcfg["batch_size"], shuffle=True,
            num_workers=dcfg["num_workers"], pin_memory=True,
            drop_last=True, persistent_workers=dcfg["num_workers"] > 0,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=dcfg["batch_size"], shuffle=False,
            num_workers=2, pin_memory=True,
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
                print(f"[W&B] Disabled: {e}")
                self.use_wandb = False

    
    # FORWARD
    
    def forward_generator(self, lq: torch.Tensor):
        """
        Full forward pass.
        
        Args:
            lq: (B, 3, H, W) ∈ [0, 1]
        
        Returns:
            dict with keys:
                'deg_pred': (B, 8)
                'bg_out': (B, 3, H*4, W*4) — Real-ESRGAN output, upscaled
                'face_out': (B, 3, H, W) — face restored
                'final': (B, 3, H, W) — sau Transformer fusion
        """
        # 1. Predict degradation vector
        deg_pred = self.deg_encoder(lq)

        # 2. Background branch (Real-ESRGAN x4)
        bg_out_4x = self.bg_net(lq, deg_pred)              # (B, 3, 4H, 4W)
        # Downsample về size gốc cho fusion
        bg_out = F.interpolate(bg_out_4x, size=lq.shape[-2:],
                              mode='bilinear', align_corners=False)

        # 3. Face branch (GFPGAN)
        # GFPGAN expect 512x512 → resize input
        face_input = F.interpolate(lq, size=(512, 512),
                                   mode='bilinear', align_corners=False)
        face_512 = self.face_net(face_input, deg_pred)     # (B, 3, 512, 512)
        # Resize về size gốc cho fusion
        face_out = F.interpolate(face_512, size=lq.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # 4. Transformer fusion (face × bg)
        final = self.fusion(face_out, bg_out)
        final = final.clamp(0, 1)

        return {
            'deg_pred': deg_pred,
            'bg_out': bg_out_4x,
            'face_out': face_out,
            'final': final,
        }

    
    # LOSS COMPUTATION
    
    def compute_g_loss(self, output, hq, deg_gt, is_real):
        """Generator loss (mọi loss TRỪ discriminator's own loss)."""
        syn_mask = (is_real == 0)
        restored = output['final']
        deg_pred = output['deg_pred']
        losses = {}

        # Pixel L1
        if self.loss_weights["pixel"] > 0 and syn_mask.any():
            losses["pixel"] = self.pixel_criterion(restored[syn_mask], hq[syn_mask])

        # Perceptual VGG
        if self.perceptual_criterion is not None and self.loss_weights["perceptual"] > 0:
            if syn_mask.any():
                losses["perceptual"] = self.perceptual_criterion(
                    restored[syn_mask], hq[syn_mask]
                )

        # Identity ArcFace
        if self.identity_extractor is not None and self.loss_weights["identity"] > 0:
            from src.losses.identity import identity_loss
            try:
                if syn_mask.any():
                    losses["identity"] = identity_loss(
                        restored[syn_mask], hq[syn_mask], self.identity_extractor
                    )
            except Exception:
                pass

        # Adversarial (Generator side: want D to think fake is real)
        if self.gan_criterion is not None and self.loss_weights["adversarial"] > 0:
            d_fake = self.discriminator(restored)
            losses["adversarial"] = self.gan_criterion(d_fake, target_is_real=True)

        # Degradation supervised
        if self.loss_weights["deg_supervised"] > 0 and syn_mask.any():
            losses["deg_supervised"] = F.mse_loss(
                deg_pred[syn_mask], deg_gt[syn_mask]
            )

        # Weighted sum
        total = sum(self.loss_weights[k] * v for k, v in losses.items()
                   if k in self.loss_weights)
        losses["total"] = total
        return losses

    def compute_d_loss(self, output, hq):
        """Discriminator loss."""
        if self.gan_criterion is None:
            return None
        
        restored = output['final'].detach()
        # Real samples
        d_real = self.discriminator(hq)
        loss_real = self.gan_criterion(d_real, target_is_real=True)
        # Fake samples
        d_fake = self.discriminator(restored)
        loss_fake = self.gan_criterion(d_fake, target_is_real=False)

        return (loss_real + loss_fake) * 0.5

    
    # TRAIN LOOP
    
    def train(self):
        total_iters = self.config["train"]["total_iters"]
        val_every = self.config["train"]["val_every"]
        save_every = self.config["train"]["save_every"]
        log_every = self.config["train"]["log_every"]

        print(f"\n=== Stage 2 training start, total_iters={total_iters} ===")
        print(f"=== Save every {save_every} iters (with backup) ===\n")
        t_start = time.time()
        train_iter = self._infinite_loader(self.train_loader)

        for self.iter in range(self.iter, total_iters):
            batch = next(train_iter)
            lq = batch["lq"].to(self.device, non_blocking=True)
            hq = batch["hq"].to(self.device, non_blocking=True)
            deg_gt = batch["deg_vec"].to(self.device, non_blocking=True)
            is_real = batch["is_real"].to(self.device, non_blocking=True)

            # ---------- Update Generator ----------
            self.optimizer_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.forward_generator(lq)
                losses_g = self.compute_g_loss(output, hq, deg_gt, is_real)

            self.scaler_g.scale(losses_g["total"]).backward()
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
            self.scheduler_g.step()

            # ---------- Update Discriminator ----------
            losses_d = None
            if self.gan_criterion is not None:
                self.optimizer_d.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss_d = self.compute_d_loss(output, hq)
                if loss_d is not None:
                    self.scaler_d.scale(loss_d).backward()
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                    self.scheduler_d.step()
                    losses_d = {"d_total": loss_d}

            # ---------- Logging ----------
            if self.iter % log_every == 0:
                self._log_train(losses_g, losses_d)

            # ---------- Validation ----------
            if self.iter > 0 and self.iter % val_every == 0:
                val_metrics = self.validate()
                self._log_val(val_metrics)
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self._save_checkpoint("best.pth", is_best=True)

            # ---------- Save ----------
            if self.iter > 0 and self.iter % save_every == 0:
                self._save_checkpoint(f"iter_{self.iter}.pth")

        elapsed = time.time() - t_start
        print(f"\n=== Stage 2 done in {elapsed/3600:.1f}h ===")
        self._save_checkpoint("final.pth")

    @staticmethod
    def _infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    
    # VALIDATION
    
    @torch.no_grad()
    def validate(self):
        for m in [self.deg_encoder, self.bg_net, self.face_net, self.fusion]:
            m.eval()

        total, total_pixel, total_deg = 0.0, 0.0, 0.0
        n = 0
        for batch in self.val_loader:
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

        for m in [self.deg_encoder, self.bg_net, self.face_net, self.fusion]:
            m.train()

        return {"loss": total/max(n,1), "pixel": total_pixel/max(n,1), "deg": total_deg/max(n,1)}

    
    # LOGGING & CHECKPOINT
    
    def _log_train(self, losses_g, losses_d):
        lr_g = self.optimizer_g.param_groups[0]["lr"]
        log_dict = {f"train/g_{k}": (v.item() if torch.is_tensor(v) else v)
                    for k, v in losses_g.items()}
        log_dict["train/lr_g"] = lr_g
        if losses_d:
            for k, v in losses_d.items():
                log_dict[f"train/d_{k}"] = v.item() if torch.is_tensor(v) else v

        for k, v in log_dict.items():
            self.tb.add_scalar(k, v, self.iter)
        if self.use_wandb:
            self.wandb.log(log_dict, step=self.iter)

        if self.iter % (10 * self.config["train"]["log_every"]) == 0:
            loss_str = ", ".join(
                f"{k}={v.item():.4f}" if torch.is_tensor(v) else f"{k}={v:.4f}"
                for k, v in losses_g.items()
            )
            print(f"[iter {self.iter}] lr={lr_g:.2e} | G: {loss_str}", end="")
            if losses_d:
                d_str = ", ".join(f"{k}={v.item():.4f}" for k, v in losses_d.items())
                print(f" | D: {d_str}")
            else:
                print()

    def _log_val(self, metrics):
        for k, v in metrics.items():
            self.tb.add_scalar(f"val/{k}", v, self.iter)
        print(f"[VAL iter {self.iter}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def _save_checkpoint(self, name: str, is_best: bool = False):
        """Save với atomic write + backup."""
        ckpt = {
            "iter": self.iter,
            "best_val_loss": self.best_val_loss,
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
        }

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
            print(f"  ★ This is BEST so far (val_loss={self.best_val_loss:.4f})")

    def _load_stage2_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.iter = ckpt["iter"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.deg_encoder.load_state_dict(ckpt["deg_encoder"])
        self.bg_net.load_state_dict(ckpt["bg_net"])
        self.face_net.load_state_dict(ckpt["face_net"])
        self.fusion.load_state_dict(ckpt["fusion"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        self.scheduler_g.load_state_dict(ckpt["scheduler_g"])
        self.scheduler_d.load_state_dict(ckpt["scheduler_d"])
        print(f"[✓] Resumed from {path} at iter {self.iter}")
