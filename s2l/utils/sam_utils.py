"""Utilities for Cellpose-SAM functionality."""
import logging
import numpy as np
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def safe_get_version(module, package_name: str) -> str:
    """Safely get version of a module/package."""
    version = getattr(module, '__version__', None)
    if version:
        return version
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        pass
    return "0.0.0"


def simple_version_compare(version1: str, version2: str) -> int:
    """Simple version comparison. Returns -1, 0, or 1."""
    def normalize(v):
        parts = []
        for part in v.split('+')[0].split('.'):
            try:
                parts.append(int(part))
            except ValueError:
                parts.append(0)
        return parts

    v1, v2 = normalize(version1), normalize(version2)
    max_len = max(len(v1), len(v2))
    v1.extend([0] * (max_len - len(v1)))
    v2.extend([0] * (max_len - len(v2)))

    for a, b in zip(v1, v2):
        if a < b:
            return -1
        if a > b:
            return 1
    return 0


class SAMModelManager:
    """Manager for Cellpose-SAM model operations."""

    AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        'cpsam': {
            'name': 'Cellpose-SAM',
            'description': 'Latest SAM-based model with superhuman generalization',
            'diameter_invariant': True,
            'recommended_params': {
                'flow_threshold': 0.4,
                'cellprob_threshold': 0.0,
                'normalize': True,
            },
        },
        'cyto3': {
            'name': 'Cytoplasm 3',
            'description': 'Traditional cytoplasm model v3',
            'diameter_invariant': False,
            'recommended_params': {
                'flow_threshold': 0.3,
                'cellprob_threshold': 0.0,
                'normalize': True,
            },
        },
        'nuclei': {
            'name': 'Nuclei',
            'description': 'Nuclear segmentation model',
            'diameter_invariant': False,
            'recommended_params': {
                'flow_threshold': 0.3,
                'cellprob_threshold': 0.0,
                'normalize': True,
            },
        },
    }

    def __init__(self):
        self.available_models = self.AVAILABLE_MODELS

    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        return self.available_models.get(model_type, {})

    def is_sam_model(self, model_type: str) -> bool:
        return model_type == 'cpsam'

    def get_recommended_params(self, model_type: str) -> Dict[str, Any]:
        info = self.get_model_info(model_type)
        return dict(info.get('recommended_params', {}))

    def validate_model_params(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        if self.is_sam_model(model_type):
            if validated.get('diameter') == 0:
                validated.pop('diameter', None)
                logger.info("Removed diameter for SAM model (diameter-invariant)")
            validated['flow_threshold'] = max(validated.get('flow_threshold', 0.4), 0.4)
        return validated


class SAMPreprocessor:
    """Preprocessing utilities optimized for SAM models."""

    @staticmethod
    def preprocess_for_sam(image: np.ndarray, normalize: bool = True) -> np.ndarray:
        if normalize:
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            p1, p99 = np.percentile(image, [1, 99])
            if p99 > p1:
                image = np.clip((image - p1) / (p99 - p1), 0, 1)
        return image

    @staticmethod
    def preprocess_for_traditional(image: np.ndarray, smooth_radius: int = 1,
                                   sharpen_radius: int = 0) -> np.ndarray:
        try:
            from cellpose import transforms
            return transforms.smooth_sharpen_img(image, smooth_radius, sharpen_radius)
        except ImportError:
            logger.warning("Cellpose transforms not available, using basic preprocessing")
            return image


class SAMPostprocessor:
    """Post-processing utilities for SAM model outputs."""

    @staticmethod
    def filter_small_masks(masks: np.ndarray, min_size: int = 15) -> np.ndarray:
        sizes = np.bincount(masks.ravel())
        small = sizes < min_size
        small[0] = False
        filtered = masks.copy()
        filtered[small[masks]] = 0
        return filtered

    @staticmethod
    def relabel_masks(masks: np.ndarray) -> np.ndarray:
        from skimage.segmentation import relabel_sequential
        return relabel_sequential(masks)[0]


def get_sam_model_download_info() -> Dict[str, str]:
    return {
        'model_name': 'cpsam',
        'download_url': 'https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam',
        'description': 'Cellpose-SAM model will be downloaded automatically on first use',
        'size_info': 'Model size: ~500MB',
        'requirements': 'Requires cellpose>=4.0.0 and torch>=2.0.0',
    }


def check_sam_compatibility() -> Tuple[bool, str]:
    """Check if the system is compatible with SAM models."""
    try:
        import cellpose
        try:
            cellpose_version = safe_get_version(cellpose, 'cellpose')
            if simple_version_compare(cellpose_version, "4.0.0") < 0:
                return False, f"Cellpose version {cellpose_version} is too old. SAM requires >=4.0.0"
        except Exception:
            pass

        try:
            import torch
            torch_version = safe_get_version(torch, 'torch')
            if simple_version_compare(torch_version, "2.0.0") < 0:
                return False, f"PyTorch version {torch_version} is too old. SAM requires >=2.0.0"
        except ImportError:
            return False, "PyTorch is not installed. SAM requires PyTorch >=2.0.0"

        return True, "System is compatible with Cellpose-SAM"
    except ImportError:
        return False, "Cellpose is not installed"
    except Exception as e:
        return False, f"Error checking compatibility: {e}"
