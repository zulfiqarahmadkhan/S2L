"""StarDist model training."""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    from stardist.models import StarDist2D, Config2D
    from csbdeep.utils import normalize as csbdeep_normalize
    STARDIST_AVAILABLE = True
except ImportError as e:
    STARDIST_AVAILABLE = False
    logger.warning(f"StarDist not available for training: {e}")


def load_stardist_model(kind: str, value: str | None):
    """Load a model to start training from: ``("pretrained", name)`` or ``("custom", model folder)``."""
    if kind == "pretrained":
        from s2l.core.stardist_segmenter import StarDistSegmenter
        return StarDistSegmenter._load_pretrained_safe(value)
    p = Path(value)
    if not (p / "config.json").is_file():
        raise FileNotFoundError(f"No StarDist model (config.json) in {value}")
    return StarDist2D(None, name=p.name, basedir=str(p.parent))


class StarDistTrainer:
    """Train a StarDist2D model on a folder of paired images and masks.

    Directory layout expected::

        train_dir/
            image1.tif          ← raw image
            image1_cp_masks.tif ← integer label mask  (same stem + mask_suffix)
            …

    Parameters
    ----------
    train_dir : str
        Folder that contains training images and masks.
    mask_suffix : str
        Filename suffix that identifies mask files (default ``"_cp_masks"``).
    val_fraction : float
        Share of the *training* images used for validation during training
        (checkpoints, threshold tuning). The test images are kept separate.
    """

    def __init__(self, train_dir: str, mask_suffix: str = "_cp_masks",
                 val_fraction: float = 0.1):
        if not STARDIST_AVAILABLE:
            raise ImportError(
                "StarDist is not installed. "
                "Install with:  pip install stardist tensorflow"
            )
        from s2l.core.datasplit import find_pairs
        self.train_dir = Path(train_dir)
        self.mask_suffix = mask_suffix
        self.val_fraction = val_fraction
        self.pairs = find_pairs(train_dir, mask_suffix)
        if not self.pairs:
            raise ValueError(
                f"No image+mask pairs found in '{train_dir}'.\n"
                f"Mask files must end with '{mask_suffix}.tif' (or .png etc.)."
            )
        logger.info(f"StarDistTrainer: {len(self.pairs)} image+mask pair(s) found")

    # ── image I/O ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(path) -> np.ndarray:
        from skimage.io import imread
        from s2l.core.image_files import channels_last
        img = channels_last(imread(str(path)))
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        return img

    @staticmethod
    def _load_mask(path) -> np.ndarray:
        from skimage.io import imread
        mask = imread(str(path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask.astype(np.int32)

    @staticmethod
    def prepare(img: np.ndarray, n_channel_in: int) -> np.ndarray:
        """Grayscale for 1-channel models, RGB for 3-channel models (e.g. 2D_versatile_he)."""
        if n_channel_in == 1:
            return (np.mean(img, axis=-1) if img.ndim == 3 else img).astype(np.float32)
        if img.ndim != 3 or img.shape[-1] != n_channel_in:
            raise ValueError(f"This model needs colour images with {n_channel_in} channels "
                             f"(e.g. RGB); got an image of shape {img.shape}.")
        return img.astype(np.float32)

    # ── training ──────────────────────────────────────────────────────────

    def train(self, model_name: str = "my_stardist_model",
              save_path: str | None = None,
              epochs: int = 100,
              steps_per_epoch: int = 100,
              patch_size: tuple[int, int] = (256, 256),
              n_rays: int = 32,
              start_from: tuple[str, str | None] = ("scratch", None),
              test_percent: float = 20,
              progress_callback=None, should_stop=None, status_callback=None) -> tuple[str, dict]:
        """Train a StarDist2D model; return ``(model folder, report)``.

        *start_from* is ``("scratch", None)``, ``("pretrained", name)`` or
        ``("custom", model folder)``. When starting from a model, its network
        layout (rays, grid, channels) is copied and its weights are fine-tuned;
        *n_rays* is then ignored. *test_percent* of the images are kept out of
        training entirely and listed in ``<model>/s2l_split.json``.
        """
        from s2l.core.datasplit import save_split, split_pairs

        basedir = Path(save_path) if save_path else self.train_dir
        basedir.mkdir(parents=True, exist_ok=True)
        kind, value = start_from

        base = None
        if kind in ("pretrained", "custom"):
            if status_callback:
                status_callback(f"Loading the starting model {Path(str(value)).name}…")
            base = load_stardist_model(kind, value)
        n_in = base.config.n_channel_in if base is not None else 1

        train_pairs, test_pairs = split_pairs(self.pairs, test_percent)
        imgs, masks = [], []
        for img_path, mask_path in train_pairs:
            imgs.append(self.prepare(self._load_image(img_path), n_in))
            masks.append(self._load_mask(mask_path))

        # validation comes from the training images only (the test images stay unseen)
        order = np.random.default_rng(42).permutation(len(imgs))
        n_val = max(1, int(round(len(imgs) * self.val_fraction))) if len(imgs) > 1 else 0
        val_idx, tr_idx = order[:n_val], order[n_val:]
        X_train = [imgs[i] for i in tr_idx] or imgs
        Y_train = [masks[i] for i in tr_idx] or masks
        X_val = [imgs[i] for i in val_idx] or imgs
        Y_val = [masks[i] for i in val_idx] or masks
        logger.info(f"StarDist: train {len(X_train)} | validation {len(X_val)} | test {len(test_pairs)}")
        if status_callback:
            status_callback(f"Training on {len(train_pairs)} image(s), {len(test_pairs)} kept for testing…")

        X_train = [csbdeep_normalize(x, 1.0, 99.8, axis=(0, 1)) for x in X_train]
        X_val = [csbdeep_normalize(x, 1.0, 99.8, axis=(0, 1)) for x in X_val]

        train_kw = dict(train_patch_size=tuple(patch_size), train_epochs=epochs,
                        train_steps_per_epoch=steps_per_epoch, train_batch_size=4,
                        train_tensorboard=False)  # TensorBoard isn't a dependency of S2L
        if base is not None:
            cfg = dict(vars(base.config))
            cfg.update(train_kw)
            conf = Config2D(**cfg)
        else:
            conf = Config2D(n_rays=n_rays, grid=(2, 2), n_channel_in=1, **train_kw)

        model = StarDist2D(conf, name=model_name, basedir=str(basedir))
        if base is not None:
            model.keras_model.set_weights(base.keras_model.get_weights())

        import tensorflow as tf
        losses = []

        class _ProgressCb(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                loss = float((logs or {}).get("loss", 0.0))
                losses.append(loss)
                if progress_callback:
                    progress_callback(epoch + 1, epochs, loss, int((epoch + 1) / epochs * 100))

            def on_train_batch_end(self, batch, logs=None):
                if should_stop and should_stop():
                    self.model.stop_training = True

        # StarDist2D.train() has no callbacks argument: prepare the model first
        # (which builds its own checkpoint callbacks), then add ours to the list.
        model.prepare_for_training()
        model.callbacks.append(_ProgressCb())
        model.train(X_train, Y_train, validation_data=(X_val, Y_val), augmenter=None)

        if status_callback:
            status_callback("Optimising detection thresholds…")
        model.optimize_thresholds(X_val, Y_val)

        saved_dir = str(basedir / model_name)
        start = "from scratch" if kind == "scratch" else str(value)
        save_split(saved_dir, "stardist", train_pairs, test_pairs, start,
                   {"epochs": epochs, "steps_per_epoch": steps_per_epoch, "patch_size": list(patch_size),
                    "n_rays": int(conf.n_rays), "grid": list(conf.grid), "n_channel_in": int(n_in),
                    "validation_images": len(X_val), "test_percent": test_percent}, losses)
        logger.info(f"StarDist training complete — model at: {saved_dir}")
        if progress_callback:
            progress_callback(epochs, epochs, None, 100)
        return saved_dir, {"n_train": len(train_pairs), "n_test": len(test_pairs)}
