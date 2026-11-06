"""Cellpose segmentation engine (Cellpose >= 4)."""
import os
import time
import logging

import numpy as np
import tqdm

from s2l.core.image_files import label_dtype, list_images, mask_path_for

logger = logging.getLogger(__name__)

try:
    from cellpose import io, models
    CELLPOSE_AVAILABLE = True
    logger.info("Cellpose imported successfully")
except ImportError as e:
    CELLPOSE_AVAILABLE = False
    logger.warning(f"Cellpose not available: {e}")


class CellposeSegmenter:
    """Wrapper around Cellpose models for cell segmentation.

    Cellpose 4 ships a single built-in model (Cellpose-SAM, ``cpsam``); the old
    ``model_type`` names (cyto3, nuclei, …) are ignored by Cellpose itself.
    Other models are loaded through *custom_model_path* (a fine-tuned ``.pth``
    or a model name in ``~/.cellpose/models``).
    """

    def __init__(self, custom_model_path=None):
        if not CELLPOSE_AVAILABLE:
            raise ImportError("Cellpose is required but not installed. Install with: pip install cellpose[gui]")

        from s2l.core.device import use_gpu
        gpu = use_gpu()
        pretrained = custom_model_path or "cpsam"
        logger.info(f"Loading Cellpose model '{pretrained}' on {'GPU' if gpu else 'CPU'}")
        self.model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained)
        self.model_name = os.path.basename(pretrained)

    def segment(self, directory, **kwargs):
        """Segment every raw image in *directory*, writing masks next to them."""
        return self.segment_files(list_images(directory), directory, **kwargs)

    def segment_files(self, file_list, output_dir, diameter=0, flow_threshold=0.4,
                      cellprob_threshold=0.0, max_iter=None, progress_callback=None,
                      should_stop=None):
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

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Segmenting {total} files with {self.model_name} → {output_dir}")

        eval_params = {'flow_threshold': flow_threshold, 'cellprob_threshold': cellprob_threshold}
        if diameter > 0:
            eval_params['diameter'] = diameter
        if max_iter is not None:
            eval_params['niter'] = max_iter

        for idx, filepath in enumerate(tqdm.tqdm(file_list, desc="Cellpose")):
            if should_stop and should_stop():
                logger.info("Segmentation stopped by user")
                return results
            try:
                img = io.imread(filepath)
                # Cellpose v4 returns masks, flows, styles
                masks = self.model.eval(img, **eval_params)[0]
                mask_path = mask_path_for(filepath, output_dir)
                imsave(str(mask_path), masks.astype(label_dtype(masks)), check_contrast=False)
                results.append((filepath, int(masks.max())))
                logger.info(f"{mask_path.name}: {int(masks.max())} objects")
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
            if progress_callback:
                progress_callback(((idx + 1) / total) * 100)

        logger.info(f"Segmentation completed in {time.time() - start_time:.2f}s")
        return results
