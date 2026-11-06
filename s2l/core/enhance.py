"""Classical image enhancement (scikit-image) used by the Image Enhancer tool.

Cellpose 4 dropped its working restoration models (its ``denoise`` module is
broken in 4.1), so these standard methods replace them. All steps work on a
float image scaled to [0, 1] per channel and are applied in a fixed order:
background → denoise → contrast → sharpen → upsample.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EnhanceSettings:
    background: bool = False
    background_radius: int = 50          # px, Gaussian estimate of uneven illumination
    denoise: str = "none"                # none | gaussian | median | nlmeans
    denoise_strength: float = 1.0        # sigma / size / h-scale
    clahe: bool = False
    clahe_clip: float = 0.01
    sharpen: bool = False
    sharpen_amount: float = 1.0
    upsample: int = 1                    # 1 = off, 2 = double the size


def to_float(img: np.ndarray) -> np.ndarray:
    """(H, W) or (H, W, C) → float32 (H, W, C) in [0, 1] (0.1–99.9 percentile stretch)."""
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 4:
        img = img[:, :, :3]
    out = np.empty(img.shape, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, (0.1, 99.9))
        out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1) if hi > lo else 0
    return out


def enhance(img: np.ndarray, s: EnhanceSettings) -> np.ndarray:
    """Apply the selected steps; returns float32 (H, W, C) in [0, 1]."""
    from scipy import ndimage as ndi
    from skimage import exposure, filters, restoration, transform

    f = to_float(img)
    out = np.empty_like(f)
    for c in range(f.shape[2]):
        ch = f[:, :, c]
        if s.background:
            bg = ndi.gaussian_filter(ch, sigma=max(1, s.background_radius))
            ch = np.clip(ch - bg, 0, None)
            hi = np.percentile(ch, 99.9)
            ch = np.clip(ch / hi, 0, 1) if hi > 0 else ch
        if s.denoise == "gaussian":
            ch = ndi.gaussian_filter(ch, sigma=max(0.3, s.denoise_strength))
        elif s.denoise == "median":
            size = max(3, int(round(s.denoise_strength * 2)) | 1)
            ch = ndi.median_filter(ch, size=size)
        elif s.denoise == "nlmeans":
            # robust noise estimate (MAD of the residual after a 3×3 median);
            # avoids skimage.estimate_sigma, which needs the optional PyWavelets
            sigma = float(np.median(np.abs(ch - ndi.median_filter(ch, size=3)))) * 1.4826 or 1e-3
            ch = restoration.denoise_nl_means(ch, h=0.8 * sigma * s.denoise_strength, sigma=sigma,
                                              patch_size=5, patch_distance=6, fast_mode=True)
        if s.clahe:
            ch = exposure.equalize_adapthist(np.clip(ch, 0, 1), clip_limit=s.clahe_clip)
        if s.sharpen:
            ch = filters.unsharp_mask(ch, radius=2, amount=s.sharpen_amount)
        out[:, :, c] = np.clip(ch, 0, 1)
    if s.upsample > 1:
        out = transform.rescale(out, s.upsample, order=1, channel_axis=2, anti_aliasing=False)
    return out.astype(np.float32)


def save_enhanced(arr: np.ndarray, path: str):
    """Save a [0, 1] float result: 16-bit for TIFF, 8-bit for PNG/JPEG."""
    from skimage.io import imsave
    arr = arr[:, :, 0] if arr.shape[2] == 1 else arr
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        imsave(path, (arr * 65535).round().astype(np.uint16), check_contrast=False)
    else:
        imsave(path, (arr * 255).round().astype(np.uint8), check_contrast=False)
