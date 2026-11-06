"""File-naming conventions shared by the segmenters and the ROI step."""
from pathlib import Path

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}
MASK_SUFFIX = "_cp_masks"

# Outputs written by S2L / Cellpose that must never be segmented as inputs
_DERIVED_MARKERS = (MASK_SUFFIX, "_masks", "_flows", "_cellprob", "_cp_output", "_ROI")


def is_derived(path) -> bool:
    """Masks / overlays written by S2L or Cellpose (judged by how the name ends, so raw
    images such as ``cell_ROI_3.tif`` are not mistaken for outputs)."""
    stem = Path(path).stem
    return MASK_SUFFIX in stem or stem.endswith(_DERIVED_MARKERS)


def channels_last(img):
    """(C, H, W) multichannel images (common for microscopy TIFFs) → (H, W, C)."""
    import numpy as np
    if img.ndim == 3 and img.shape[0] <= 4 < img.shape[-1]:
        return np.moveaxis(img, 0, -1)
    return img


def label_dtype(labels):
    """uint16 holds up to 65,535 cells; more would wrap around and merge cells."""
    import numpy as np
    return np.uint32 if int(labels.max(initial=0)) > 65535 else np.uint16


def list_images(directory) -> list[str]:
    """Raw input images directly inside *directory* (masks and overlays excluded)."""
    return sorted(
        str(p) for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not is_derived(p)
    )


def mask_path_for(image_path, output_dir) -> Path:
    return Path(output_dir) / f"{Path(image_path).stem}{MASK_SUFFIX}.tif"


def list_masks(directory) -> list[Path]:
    """Mask files (``<stem>_cp_masks.tif``) produced by a segmentation run."""
    return sorted(
        p for p in Path(directory).glob(f"*{MASK_SUFFIX}.tif*")
        if p.stem.count(MASK_SUFFIX) == 1 and p.stem.endswith(MASK_SUFFIX)
    )


def original_stem(mask_path) -> str:
    return Path(mask_path).stem[: -len(MASK_SUFFIX)]
