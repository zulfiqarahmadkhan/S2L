"""GPU / CPU choice shared by segmentation, training and the AI tool.

The GPU is used by default when one is available; the sidebar switch can force
the CPU (e.g. when the GPU is busy with another program). The choice is read
each time a model is loaded, so it applies from the next run on.
"""
from __future__ import annotations

_prefer_cpu = False


def set_prefer_cpu(value: bool):
    global _prefer_cpu
    _prefer_cpu = bool(value)


def prefer_cpu() -> bool:
    return _prefer_cpu


def gpu_available() -> bool:
    """An NVIDIA GPU (CUDA) or an Apple-silicon GPU (MPS); Cellpose uses either with gpu=True."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_available())
    except Exception:
        return False


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "Apple GPU (Metal)" if gpu_available() else ""
    except Exception:
        return ""


def use_gpu() -> bool:
    """True when models should run on the GPU: one is available and the user has not chosen the CPU."""
    return not _prefer_cpu and gpu_available()
