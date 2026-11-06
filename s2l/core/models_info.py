"""Model names, download status and environment checks.

Nothing here imports torch / tensorflow at module load.
"""
import os
from pathlib import Path

# Cellpose 4 ships exactly one pretrained model (the older cyto / nuclei / ...
# models need Cellpose 3 and cannot be loaded by Cellpose 4).
CELLPOSE_PRETRAINED = {
    "cpsam": "Cellpose-SAM",
}
# Pretrained models shipped with StarDist
STARDIST_PRETRAINED = {
    "2D_versatile_fluo": "Versatile (fluorescent nuclei)",
    "2D_versatile_he":   "Versatile (H&E nuclei)",
    "2D_paper_dsb2018":  "DSB 2018",
    "2D_demo":           "Demo",
}
PRETRAINED = {"cellpose": CELLPOSE_PRETRAINED, "stardist": STARDIST_PRETRAINED}

MODEL_DESCRIPTIONS = {
    "cpsam": "General model for cells and nuclei in any kind of image (phase, brightfield, "
             "fluorescence). Replaces all older Cellpose models.",
    "2D_versatile_fluo": "Nuclei in fluorescence images (DAPI, Hoechst, PI…). Grayscale images.",
    "2D_versatile_he": "Nuclei in H&E stained tissue. Needs colour (RGB) images.",
    "2D_paper_dsb2018": "Nuclei, model from the StarDist paper (Data Science Bowl 2018). "
                        "Older than Versatile; mainly for comparison.",
    "2D_demo": "Small demo model for testing only.",
}
MODEL_SIZES_MB = {"cpsam": 1180, "2D_versatile_fluo": 5, "2D_versatile_he": 5,
                  "2D_paper_dsb2018": 5, "2D_demo": 5}


def _cellpose_model_dir() -> Path:
    env = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
    return Path(env) if env else Path.home() / ".cellpose" / "models"


def stardist_cache_dir(name: str) -> Path:
    """Where S2L keeps a pretrained StarDist model: ~/.keras/models/StarDist2D/<name>/<name>/."""
    return Path.home() / ".keras" / "models" / "StarDist2D" / name


def is_downloaded(engine: str, name: str) -> bool:
    if engine == "cellpose":
        return (_cellpose_model_dir() / name).is_file()
    return (stardist_cache_dir(name) / name / "config.json").is_file()


def download_pretrained(engine: str, name: str):
    """Download one pretrained model if it is missing (slow; call off the UI thread)."""
    if is_downloaded(engine, name):
        return
    if engine == "cellpose":
        from cellpose import models
        models.cache_CPSAM_model_path()
    else:
        from s2l.core.stardist_segmenter import StarDistSegmenter
        StarDistSegmenter._load_pretrained_safe(name)


def all_pretrained():
    """[(engine, name, label)] for every pretrained model."""
    return [(e, k, label) for e, d in PRETRAINED.items() for k, label in d.items()]


def probe_environment() -> dict:
    """Import the heavy libraries and report what is usable. Slow (several seconds)."""
    info = {"gpu": False, "gpu_name": "", "cellpose_ok": False, "cellpose_msg": "",
            "stardist_ok": False, "stardist_msg": ""}
    try:
        import torch  # noqa: F401  (import errors are reported below)
        from s2l.core.device import gpu_available, gpu_name
        info["gpu"] = gpu_available()
        info["gpu_name"] = gpu_name()
    except Exception as e:
        info["cellpose_msg"] = f"PyTorch unavailable: {e}"
    try:
        from s2l.utils.sam_utils import check_sam_compatibility
        info["cellpose_ok"], msg = check_sam_compatibility()
        info["cellpose_msg"] = info["cellpose_msg"] or msg
    except Exception as e:
        info["cellpose_msg"] = str(e)
    try:
        from s2l.core.stardist_segmenter import STARDIST_AVAILABLE
        info["stardist_ok"] = STARDIST_AVAILABLE
        info["stardist_msg"] = "ready" if STARDIST_AVAILABLE else "not installed (pip install stardist tensorflow)"
    except Exception as e:
        info["stardist_msg"] = str(e)
    return info
