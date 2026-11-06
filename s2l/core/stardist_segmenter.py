"""StarDist segmentation engine."""
import time
import logging
from pathlib import Path

import numpy as np
import tqdm

logger = logging.getLogger(__name__)

try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize as csbdeep_normalize
    STARDIST_AVAILABLE = True
    logger.info("StarDist imported successfully")
except ImportError as e:
    STARDIST_AVAILABLE = False
    logger.warning(f"StarDist not available: {e}")

from s2l.core.image_files import channels_last, label_dtype, list_images, mask_path_for

from s2l.core.models_info import STARDIST_PRETRAINED  # noqa: E402,F401  (re-exported)


class StarDistSegmenter:
    """Wrapper around StarDist2D for cell/nuclei segmentation."""

    def __init__(self, model_name: str = "2D_versatile_fluo",
                 custom_model_path: str | None = None,
                 custom_model_name: str | None = None):
        if not STARDIST_AVAILABLE:
            raise ImportError(
                "StarDist is not installed. Install with: pip install stardist"
            )

        self.model_name = model_name
        self._load_model(model_name, custom_model_path, custom_model_name)

    @classmethod
    def from_model_dir(cls, model_dir: str):
        """Load a trained model from its folder (the one holding config.json)."""
        p = Path(model_dir)
        if not (p / "config.json").exists():
            raise FileNotFoundError(f"No StarDist model (config.json) in {model_dir}")
        return cls(custom_model_path=str(p.parent), custom_model_name=p.name)

    def _load_model(self, model_name, custom_path, custom_name):
        if custom_path and custom_name:
            # No silent fallback: results must come from the model the user picked
            logger.info(f"Loading custom StarDist model: {custom_name} from {custom_path}")
            self.model = StarDist2D(None, name=custom_name, basedir=custom_path)
            self.model_name = custom_name
            logger.info("Custom StarDist model loaded")
            return

        # No fallback to another model: results must come from the model the user picked
        if model_name not in STARDIST_PRETRAINED:
            raise ValueError(f"Unknown StarDist model '{model_name}'")
        logger.info(f"Loading pretrained StarDist model: {model_name}")
        try:
            self.model = self._load_pretrained_safe(model_name)
        except Exception as e:
            raise RuntimeError(f"Could not load StarDist model '{model_name}' "
                               f"(it is downloaded on first use — check the internet connection): {e}") from e
        self.model_name = model_name
        logger.info(f"StarDist model '{model_name}' ready")

    @staticmethod
    def _load_pretrained_safe(name: str):
        """Load a pretrained model, working around Windows symlink errors."""
        import shutil

        # StarDist2D(None, name=X, basedir=B) expects B/X/config.json
        # from_pretrained stores at ~/.keras/models/StarDist2D/<name>/
        # and tries to rename <name>_extracted -> <name> (fails on Windows)
        cache_parent = Path.home() / ".keras" / "models" / "StarDist2D" / name
        model_dir = cache_parent / name
        extracted = cache_parent / f"{name}_extracted"

        # recover from a previous failed symlink attempt
        if extracted.is_dir() and not model_dir.is_dir():
            logger.info(f"Recovering '{name}' from stale _extracted directory")
            inner = extracted / name
            src = str(inner) if inner.is_dir() else str(extracted)
            shutil.copytree(src, str(model_dir))
            shutil.rmtree(str(extracted))

        # if model files already exist locally, load directly
        if model_dir.is_dir() and (model_dir / "config.json").exists():
            logger.info(f"Loading '{name}' from cache")
            return StarDist2D(None, name=name, basedir=str(cache_parent))

        # fresh download
        try:
            return StarDist2D.from_pretrained(name)
        except OSError as e:
            # WinError 1314: symlink privilege not held
            if "1314" not in str(e) and "privilege" not in str(e).lower():
                raise
            logger.warning(f"Windows symlink error for '{name}', fixing manually")
            if extracted.is_dir():
                inner = extracted / name
                src = str(inner) if inner.is_dir() else str(extracted)
                if model_dir.exists():
                    shutil.rmtree(str(model_dir))
                shutil.copytree(src, str(model_dir))
                shutil.rmtree(str(extracted))
                logger.info(f"Fixed model directory for '{name}'")
            return StarDist2D(None, name=name, basedir=str(cache_parent))


    def segment(self, directory: str, **kwargs):
        """Segment every raw image in *directory*, writing masks next to them."""
        return self.segment_files(list_images(directory), directory, **kwargs)

    @staticmethod
    def _read_image(filepath: str):
        """Read an image, return as numpy array or None."""
        try:
            from skimage.io import imread
            img = channels_last(imread(filepath))
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            return img
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return None

    def _prepare_for_model(self, img: np.ndarray):
        """Adapt image channels to match model's n_channel_in and return (img, axes).
        
        Call this BEFORE normalization so the normalize step works on the right shape.
        """
        n_in = self.model.config.n_channel_in
        logger.debug(f"  _prepare_for_model: input shape={img.shape}, dtype={img.dtype}, model expects n_channel_in={n_in}")

        if n_in == 1:
            if img.ndim == 3:
                # multi-channel → convert to grayscale (mean across channels)
                img = np.mean(img, axis=-1)
            return img, "YX"
        else:
            if img.ndim == 2:
                img = np.stack([img] * n_in, axis=-1)
            elif img.ndim == 3 and img.shape[2] != n_in:
                if img.shape[2] > n_in:
                    img = img[..., :n_in]
                else:
                    pad = np.zeros((*img.shape[:2], n_in - img.shape[2]), dtype=img.dtype)
                    img = np.concatenate([img, pad], axis=-1)
            return img, "YXC"

    def segment_files(self, file_list, output_dir, prob_thresh=0.5,
                      nms_thresh=0.4, n_tiles=None, scale=None,
                      progress_callback=None, should_stop=None):
        """Segment an explicit list of files, saving ``<stem>_cp_masks.tif`` to output_dir."""
        from skimage.io import imsave

        start_time = time.time()
        total = len(file_list)
        results = []  # (image path, number of objects) per segmented image
        if total == 0:
            logger.warning("No image files to process.")
            if progress_callback:
                progress_callback(100)
            return results

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"StarDist segmenting {total} files with '{self.model_name}' → {out}")

        for idx, filepath in enumerate(tqdm.tqdm(file_list, desc="StarDist")):
            if should_stop and should_stop():
                logger.info("Segmentation stopped by user")
                return results
            try:
                img = self._read_image(filepath)
                if img is None:
                    continue
                # adapt channels first, then normalize
                img_ready, axes = self._prepare_for_model(img)
                img_norm = csbdeep_normalize(img_ready, 1, 99.8, axis=(0, 1))
                kw = dict(prob_thresh=prob_thresh, nms_thresh=nms_thresh, axes=axes)
                if n_tiles is not None:
                    kw["n_tiles"] = n_tiles
                if scale is not None:
                    kw["scale"] = scale

                labels, _ = self.model.predict_instances(img_norm, **kw)
                mask_path = mask_path_for(filepath, out)
                imsave(str(mask_path), labels.astype(label_dtype(labels)), check_contrast=False)
                results.append((filepath, int(labels.max())))
                logger.info(f"{mask_path.name}: {labels.max()} objects")
            except Exception as e:
                logger.error(f"Error processing {filepath}: {type(e).__name__}: {e}", exc_info=True)
            if progress_callback:
                progress_callback(((idx + 1) / total) * 100)

        logger.info(f"StarDist segmentation completed in {time.time() - start_time:.2f}s")
        return results
