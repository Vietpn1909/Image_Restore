"""
End-to-end Restoration Pipeline cho Stage 2 / Stage 3 checkpoint.

CRITICAL FIX (v7):
    Stage 2 train với FULL UPSAMPLE path (forward_generator gốc trainer_stage2.py):
        bg_net(lq, deg_pred) → (B, 3, 512, 512) → downsample về (256, 256)
    Stage 3 train với SKIP UPSAMPLE path (monkey patch lowvram trong notebook):
        conv_first → body → conv_body → conv_hr → conv_last → (B, 3, 256, 256)
    
    Pipeline phải dùng ĐÚNG forward path tương ứng với stage của ckpt.
    Auto-detect dựa trên:
        - ckpt['stage'] nếu có
        - Fallback: số block của fusion (Stage 2: 2 blocks, Stage 3: 4 blocks)
    
    User có thể override bằng tham số `forward_mode`.

Cách dùng:
    pipeline = OldPortraitRestorationPipeline(
        checkpoint_path='/path/to/best.pth',
        gfpgan_pretrained='/path/to/GFPGANv1.4.pth',
        device='cuda',
        forward_mode='auto',  # hoặc 'skip_upsample' / 'full_upsample'
    )
    restored = pipeline.restore(image)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.degradation_encoder import DegradationEncoder
from src.models.real_esrgan.rrdb_with_film import RRDBNet_FiLM
from src.models.gfpgan.gfpgan_wrapper import (
    build_face_restorer, GFPGANWrapper, SimpleFaceRestorer,
)
from src.models.transformer_fusion import TransformerFusion
from src.utils.gfpgan_import_fix import assert_gfpgan_available


#  Helpers 
def to_tensor(image: Union[np.ndarray, Image.Image], device: str = 'cuda') -> torch.Tensor:
    """Convert image bất kỳ format → tensor (1, 3, H, W) [0,1]."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        tensor = image
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported type: {type(image)}")
    return tensor.float().to(device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor (1, 3, H, W) [0,1] → numpy uint8 (H, W, 3)."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.clamp(0, 1).cpu().numpy()
    arr = (arr.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return arr


#  Pipeline 
class OldPortraitRestorationPipeline:
    """End-to-end restoration pipeline với auto-detect forward path."""

    def __init__(
        self,
        checkpoint_path: str,
        gfpgan_pretrained: str,
        device: str = 'cuda',
        image_size: int = 256,
        verbose: bool = True,
        allow_simple_fallback: bool = False,
        strict_load: bool = True,
        forward_mode: str = 'auto',
    ):
        """
        Args:
            forward_mode: 'auto' | 'skip_upsample' | 'full_upsample'
                - 'auto': detect dựa trên ckpt['stage'] (Stage 2 → full, Stage 3 → skip)
                - 'skip_upsample': skip conv_up1, conv_up2 (Stage 3 path)
                - 'full_upsample': bg_net forward đầy đủ rồi downsample (Stage 2 path)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.verbose = verbose
        self.allow_simple_fallback = allow_simple_fallback
        self.strict_load = strict_load

        #  Load checkpoint 
        if self.verbose:
            print(f"[Pipeline] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.config = ckpt.get('config', {})
        self.stage = ckpt.get('stage', None)
        self.iter = ckpt.get('iter', None)
        self.val_loss = ckpt.get('best_val_loss', None)

        if self.verbose:
            print(f"  Stage:    {self.stage}")
            print(f"  Iter:     {self.iter}")
            print(f"  Val loss: {self.val_loss}")

        #  Resolve forward_mode 
        if forward_mode == 'auto':
            self.forward_mode = self._detect_forward_mode(ckpt)
            if self.verbose:
                print(f"  Forward mode (auto-detected): {self.forward_mode}")
        elif forward_mode in ('skip_upsample', 'full_upsample'):
            self.forward_mode = forward_mode
            if self.verbose:
                print(f"  Forward mode (manual): {self.forward_mode}")
        else:
            raise ValueError(
                f"forward_mode phải là 'auto'/'skip_upsample'/'full_upsample', "
                f"không phải '{forward_mode}'")

        #  GFPGAN must be available 
        if not self.allow_simple_fallback:
            assert_gfpgan_available(verbose=self.verbose)

        #  Build models theo config ckpt 
        self._build_models(gfpgan_pretrained, ckpt)

        #  Load weights 
        self._load_weights(ckpt)

        for m in [self.deg_encoder, self.bg_net, self.face_net, self.fusion]:
            m.eval()

        if self.verbose:
            print(f"[Pipeline] Ready on {self.device}")

    def _detect_forward_mode(self, ckpt: dict) -> str:
        """Detect forward mode dựa trên stage hoặc số fusion blocks.
        
        Logic:
        - Stage 2 train với forward_generator GỐC (full upsample x4 → downsample)
        - Stage 3 train với monkey patch lowvram (skip upsample)
        """
        # Ưu tiên 1: ckpt['stage']
        stage = ckpt.get('stage', None)
        if stage == 3:
            return 'skip_upsample'
        elif stage == 2:
            return 'full_upsample'
        
        # Ưu tiên 2: số fusion blocks (Stage 2: 2, Stage 3: 4)
        fusion_keys = ckpt.get('fusion', {}).keys()
        block_indices = set()
        for k in fusion_keys:
            if k.startswith('blocks.'):
                try:
                    idx = int(k.split('blocks.')[1].split('.')[0])
                    block_indices.add(idx)
                except (ValueError, IndexError):
                    pass
        num_blocks = max(block_indices) + 1 if block_indices else 0
        
        if num_blocks >= 3:
            return 'skip_upsample'  # Stage 3 (4 blocks)
        else:
            return 'full_upsample'  # Stage 2 (2 blocks) hoặc cũ

    def _build_models(self, gfpgan_pretrained: str, ckpt: dict):
        mcfg = self.config.get('model', {})
        deg_dim = mcfg.get('degradation_encoder', {}).get('out_dim', 8)
        fusion_blocks = mcfg.get('transformer_fusion', {}).get('num_blocks', 2)
        fusion_dim = mcfg.get('transformer_fusion', {}).get('dim', 64)
        fusion_heads = mcfg.get('transformer_fusion', {}).get('num_heads', 8)

        if self.verbose:
            print(f"[Pipeline] Building models: deg_dim={deg_dim}, "
                  f"fusion_blocks={fusion_blocks}")

        self.deg_encoder = DegradationEncoder(out_dim=deg_dim).to(self.device)
        self.bg_net = RRDBNet_FiLM(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
            num_grow_ch=32, scale=4, film_last_n=4, cond_dim=deg_dim,
        ).to(self.device)
        self.face_net = build_face_restorer(
            gfpgan_path=gfpgan_pretrained, cond_dim=deg_dim,
            fallback_to_simple=self.allow_simple_fallback, device=self.device,
        ).to(self.device)
        if isinstance(self.face_net, SimpleFaceRestorer):
            msg = ("Using SimpleFaceRestorer fallback. This is only for smoke tests; "
                   "Stage 2/3 metric evaluation is invalid with this fallback.")
            if not self.allow_simple_fallback:
                raise RuntimeError(msg)
            if self.verbose:
                print(f"  ⚠️ {msg}")
        self.fusion = TransformerFusion(
            dim=fusion_dim, num_blocks=fusion_blocks, num_heads=fusion_heads,
            in_channels_face=3, in_channels_bg=3, out_channels=3,
        ).to(self.device)

    def _load_weights(self, ckpt: dict):
        for name, model in [
            ('deg_encoder', self.deg_encoder),
            ('bg_net', self.bg_net),
            ('face_net', self.face_net),
            ('fusion', self.fusion),
        ]:
            if name not in ckpt:
                raise KeyError(f"Checkpoint missing required key: {name}")
            if self.strict_load:
                model.load_state_dict(ckpt[name], strict=True)
                if self.verbose:
                    print(f"  ✓ {name} (strict)")
            else:
                missing, unexpected = model.load_state_dict(ckpt[name], strict=False)
                if self.verbose:
                    print(f"  ⚠ {name} (non-strict: {len(missing)} missing, "
                          f"{len(unexpected)} unexpected)")
        if isinstance(self.face_net, GFPGANWrapper):
            n_face = sum(p.numel() for p in self.face_net.parameters()) / 1e6
            if self.verbose:
                print(f"  ✓ face_net type: GFPGANWrapper ({n_face:.2f}M params)")
        elif not self.allow_simple_fallback:
            raise RuntimeError("face_net is not GFPGANWrapper; aborting evaluation.")

    # ─ 2 forward paths ─
    @torch.no_grad()
    def forward_skip_upsample(self, lq: torch.Tensor) -> dict:
        """Stage 3 path — skip conv_up1, conv_up2 của Real-ESRGAN.
        
        Forward: conv_first → body (23 RRDB với FiLM 4 block cuối) → conv_body
                 → (skip upsample) → conv_hr → conv_last → output 256×256
        """
        deg_pred = self.deg_encoder(lq)
        
        # BG branch — skip x4 upsample
        feat = self.bg_net.conv_first(lq)
        feat_skip = feat
        for block in self.bg_net.body:
            if hasattr(block, 'film'):
                feat = block(feat, deg_pred)
            else:
                feat = block(feat)
        feat = self.bg_net.conv_body(feat) + feat_skip
        bg_out = self.bg_net.conv_last(
            self.bg_net.lrelu(self.bg_net.conv_hr(feat))
        ).clamp(0, 1)
        
        # Face branch (GFPGAN 512×512)
        face_input = F.interpolate(lq, size=(512, 512),
                                    mode='bilinear', align_corners=False)
        face_512 = self.face_net(face_input, deg_pred)
        face_out = F.interpolate(face_512, size=lq.shape[-2:],
                                  mode='bilinear', align_corners=False)
        
        final = self.fusion(face_out, bg_out).clamp(0, 1)
        return {'final': final, 'bg_out': bg_out, 'face_out': face_out, 'deg_pred': deg_pred}

    @torch.no_grad()
    def forward_full_upsample(self, lq: torch.Tensor) -> dict:
        """Stage 2 path — bg_net forward đầy đủ (x4 upsample) rồi downsample về size lq.
        
        Forward: bg_net(lq, deg_pred) → output 1024×1024 → downsample về 256×256.
        Đây là forward_generator GỐC của trainer_stage2.py.
        """
        deg_pred = self.deg_encoder(lq)
        
        # BG branch — full forward including conv_up1, conv_up2
        bg_out_4x = self.bg_net(lq, deg_pred)  # (B, 3, 4H, 4W)
        bg_out = F.interpolate(bg_out_4x, size=lq.shape[-2:],
                                mode='bilinear', align_corners=False).clamp(0, 1)
        
        # Face branch (giống skip path)
        face_input = F.interpolate(lq, size=(512, 512),
                                    mode='bilinear', align_corners=False)
        face_512 = self.face_net(face_input, deg_pred)
        face_out = F.interpolate(face_512, size=lq.shape[-2:],
                                  mode='bilinear', align_corners=False)
        
        final = self.fusion(face_out, bg_out).clamp(0, 1)
        return {'final': final, 'bg_out': bg_out, 'face_out': face_out, 'deg_pred': deg_pred}

    @torch.no_grad()
    def forward_generator(self, lq: torch.Tensor) -> dict:
        """Dispatch theo forward_mode."""
        if self.forward_mode == 'skip_upsample':
            return self.forward_skip_upsample(lq)
        elif self.forward_mode == 'full_upsample':
            return self.forward_full_upsample(lq)
        else:
            raise ValueError(f"Invalid forward_mode: {self.forward_mode}")

    # Giữ alias cũ để không break code đang gọi
    def forward_generator_lowvram(self, lq: torch.Tensor) -> dict:
        return self.forward_generator(lq)

    # ─ Public API ─
    @torch.no_grad()
    def restore(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, dict]:
        lq = to_tensor(image, device=self.device)
        original_size = lq.shape[-2:]
        lq_256 = F.interpolate(lq, size=(self.image_size, self.image_size),
                                mode='bilinear', align_corners=False)
        outputs = self.forward_generator(lq_256)
        final = F.interpolate(outputs['final'], size=original_size,
                               mode='bilinear', align_corners=False)
        if return_intermediates:
            return {
                'final': to_numpy(final),
                'final_256': to_numpy(outputs['final']),
                'bg_out': to_numpy(outputs['bg_out']),
                'face_out': to_numpy(outputs['face_out']),
                'deg_vec': outputs['deg_pred'].squeeze().cpu().numpy(),
            }
        return to_numpy(final)

    @torch.no_grad()
    def restore_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 4,
    ) -> List[np.ndarray]:
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            tensors, sizes = [], []
            for img in batch:
                t = to_tensor(img, device=self.device)
                sizes.append(t.shape[-2:])
                t = F.interpolate(t, size=(self.image_size, self.image_size),
                                   mode='bilinear', align_corners=False)
                tensors.append(t)
            batch_tensor = torch.cat(tensors, dim=0)
            outputs = self.forward_generator(batch_tensor)
            for j, original_size in enumerate(sizes):
                final = outputs['final'][j:j+1]
                final = F.interpolate(final, size=original_size,
                                       mode='bilinear', align_corners=False)
                results.append(to_numpy(final))
        return results

    @torch.no_grad()
    def restore_256(self, lq_tensor: torch.Tensor) -> torch.Tensor:
        """Restore tensor 256×256 trực tiếp (cho evaluation script)."""
        outputs = self.forward_generator(lq_tensor)
        return outputs['final']
