"""AI-assisted "add this cell": segment one missed cell inside a user-drawn box.

Strategy (first success wins):
  1. Cellpose-SAM on the box padded by 100 % of its size (context matters).
  2. Same crop, telling the model the expected cell diameter (box size).
  3. Bigger context (box padded by 200 %).
  4. Fallback without AI: Otsu threshold of the brightness inside the box.

The chosen object is the one covering the most *unsegmented* area inside the
box, so neighbouring cells that already exist are never picked.
"""
from __future__ import annotations

import logging

import numpy as np

from s2l.core.mask_edit import pick_object

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_GPU = None      # device the cached model was loaded on


def get_model():
    """Cellpose-SAM, loaded once per session (again if the GPU/CPU choice changed). Call from the main thread."""
    global _MODEL, _MODEL_GPU
    from s2l.core.device import use_gpu
    gpu = use_gpu()
    if _MODEL is None or _MODEL_GPU != gpu:
        from cellpose import models
        _MODEL = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
        _MODEL_GPU = gpu
    return _MODEL


def model_loaded() -> bool:
    from s2l.core.device import use_gpu
    return _MODEL is not None and _MODEL_GPU == use_gpu()


def _crop(image, box, pad_frac):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    pad = max(32, int(pad_frac * max(x1 - x0, y1 - y0)))
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad + 1), min(h, y1 + pad + 1)
    return cx0, cy0, cx1, cy1


def _brightness_fallback(intensity, free, box):
    """Largest Otsu component inside the box that lies on unsegmented pixels."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    x0, y0, x1, y1 = box
    sub = intensity[y0:y1 + 1, x0:x1 + 1]
    if sub.size < 25 or sub.max() == sub.min():
        return None
    t = threshold_otsu(sub)
    # cells are the minority class in a tight box: pick bright or dark accordingly
    fg = sub > t if (sub > t).mean() <= 0.5 else sub <= t
    fg &= free[y0:y1 + 1, x0:x1 + 1]
    lab, n = ndi.label(fg)
    if not n:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] < 20:
        return None
    obj = ndi.binary_fill_holes(lab == best)
    full = np.zeros(intensity.shape, bool)
    full[y0:y1 + 1, x0:x1 + 1] = obj
    return full


def segment_box(model, image: np.ndarray, free: np.ndarray, box, from_click: bool = False):
    """Find the missed cell in *box* = (x0, y0, x1, y1).

    *image* is float (H, W, C) in [0, 1]; *free* is True where the current mask
    is background. Returns (boolean object mask in full-image coordinates,
    method string) or (None, reason).
    """
    x0, y0, x1, y1 = box
    attempts = [(1.0, None), (2.0, None)]
    if not from_click:
        attempts.insert(1, (1.0, max(x1 - x0, y1 - y0) * 0.8))
    for pad_frac, diameter in attempts:
        cx0, cy0, cx1, cy1 = _crop(image, box, pad_frac)
        crop = (image[cy0:cy1, cx0:cx1] * 255).astype(np.uint8)
        if crop.shape[2] == 1:
            crop = crop[:, :, 0]
        kw = {"diameter": diameter} if diameter else {}
        try:
            labels = np.asarray(model.eval(crop, **kw)[0])
        except Exception as e:
            logger.warning(f"AI attempt failed: {e}")
            continue
        obj = pick_object(labels, (x0 - cx0, y0 - cy0, x1 - cx0 + 1, y1 - cy0 + 1),
                          free[cy0:cy1, cx0:cx1])
        if obj is not None and (obj & free[cy0:cy1, cx0:cx1]).sum() >= 20:
            full = np.zeros(free.shape, bool)
            full[cy0:cy1, cx0:cx1] = obj
            return full, "Cellpose-SAM"
    obj = _brightness_fallback(image.mean(axis=2), free, box)
    if obj is not None:
        return obj, "brightness (AI found nothing)"
    return None, "no cell found"
