"""Utilities to make GFPGAN / BasicSR import reliable on Kaggle.

Kaggle often ships a new torchvision version where
``torchvision.transforms.functional_tensor`` no longer exists. BasicSR 1.4.2
still imports that module, so GFPGAN can silently fail and the project may fall
back to SimpleFaceRestorer. That fallback makes Stage 2/3 metrics invalid.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path


def patch_torchvision_functional_tensor_alias(verbose: bool = False) -> None:
    """Patch the removed torchvision functional_tensor import.

    This function is safe to call multiple times. It applies two fixes:
    1. Runtime alias: sys.modules['torchvision.transforms.functional_tensor'].
    2. On-disk BasicSR patch if degradations.py is already installed.
    """
    try:
        import torchvision.transforms.functional as TF
        sys.modules.setdefault('torchvision.transforms.functional_tensor', TF)
        if verbose:
            print('[GFPGAN fix] Runtime alias functional_tensor -> functional OK')
    except Exception as e:
        if verbose:
            print(f'[GFPGAN fix] Runtime alias skipped: {e}')

    old = 'from torchvision.transforms.functional_tensor import rgb_to_grayscale'
    new = 'from torchvision.transforms.functional import rgb_to_grayscale'
    candidates = []
    candidates += glob.glob('/usr/local/lib/python*/dist-packages/basicsr/data/degradations.py')
    candidates += glob.glob('/opt/conda/lib/python*/site-packages/basicsr/data/degradations.py')
    candidates += glob.glob('/kaggle/working/**/basicsr/data/degradations.py', recursive=True)

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        path = Path(p)
        try:
            text = path.read_text()
            if old in text:
                path.write_text(text.replace(old, new))
                if verbose:
                    print(f'[GFPGAN fix] Patched {path}')
        except Exception as e:
            if verbose:
                print(f'[GFPGAN fix] Could not patch {path}: {e}')


def assert_gfpgan_available(verbose: bool = True) -> None:
    """Raise immediately if GFPGAN cannot be imported."""
    patch_torchvision_functional_tensor_alias(verbose=verbose)
    try:
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            'GFPGAN import failed. Do NOT run evaluation, otherwise the pipeline '
            'may use SimpleFaceRestorer and produce invalid Stage 2/3 metrics. '
            'Install/patch: gfpgan==1.3.8 basicsr==1.4.2 facexlib==0.3.0. '
            f'Original error: {e}'
        ) from e
    if verbose:
        print('✓ GFPGAN import OK')
