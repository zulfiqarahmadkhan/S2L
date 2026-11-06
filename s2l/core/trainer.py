"""Cellpose model training (Cellpose >= 4)."""
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from cellpose import models, train
    CELLPOSE_AVAILABLE = True
except ImportError as e:
    CELLPOSE_AVAILABLE = False
    logger.warning(f"Cellpose not available for training: {e}")

# Cellpose logs "<epoch>, train_loss=..., test_loss=..." at epoch 0, 5 and every 10th
_EPOCH_LOG_RE = re.compile(r"^(\d+), train_loss=([\d.eE+-]+)")


class TrainingAborted(Exception):
    """Raised inside the training loop when the user presses Stop."""


class _EpochLogHandler(logging.Handler):
    """Turns Cellpose's per-epoch log lines into progress callbacks."""

    def __init__(self, n_epochs, progress_callback):
        super().__init__()
        self._n_epochs = n_epochs
        self._cb = progress_callback

    def emit(self, record):
        m = _EPOCH_LOG_RE.match(record.getMessage())
        if m and self._cb:
            epoch = int(m.group(1)) + 1
            self._cb(epoch, self._n_epochs, float(m.group(2)), int(epoch / self._n_epochs * 100))


class Trainer:
    """Fine-tune a Cellpose model on a folder of images + masks.

    Parameters
    ----------
    train_dir : str
        Folder containing the images and their mask files.
    mask_filter : str
        Suffix used by mask files, e.g. ``"_cp_masks"`` (``img.tif`` ↔ ``img_cp_masks.tif``).
    """

    def __init__(self, train_dir: str, mask_filter: str = "_cp_masks"):
        if not CELLPOSE_AVAILABLE:
            raise ImportError(
                "Cellpose is required for training. Install with: pip install cellpose[gui]"
            )
        from s2l.core.datasplit import find_pairs
        self.train_dir = train_dir
        self.mask_filter = mask_filter
        self.pairs = find_pairs(train_dir, mask_filter)
        if not self.pairs:
            raise ValueError(
                f"No image+mask pairs found in '{train_dir}'.\n"
                f"Make sure your mask files end with '{mask_filter}' "
                f"and images do not contain that suffix."
            )
        logger.info(f"Found {len(self.pairs)} image+mask pair(s)")

    def train(self, model_name: str, base_model: str = "cpsam",
              epochs: int = 100, learning_rate: float = 1e-5,
              weight_decay: float = 0.1, batch_size: int = 1,
              normalize: bool = True,
              save_path: str | None = None,
              progress_callback=None, should_stop=None,
              test_percent: float = 20, status_callback=None) -> tuple:
        """Fine-tune *base_model*; return ``(model_path, train_losses, report)``.

        *test_percent* of the images (chosen at random, reproducibly) are kept
        out of training entirely. The split is saved next to the model
        (``<model>_s2l_split.json``) so "Test model" evaluates exactly those images.

        Parameters
        ----------
        base_model : str
            ``"cpsam"`` or a path to a previously fine-tuned model to continue from.
        learning_rate, weight_decay : float
            AdamW settings; Cellpose-SAM's recommended defaults are 1e-5 and 0.1.
        save_path : str, optional
            Directory where the model is written (Cellpose stores it in
            ``<save_path>/models/<model_name>``). Defaults to the training directory.
        progress_callback : callable, optional
            Called with ``(epoch, n_epochs, loss, pct)``. Cellpose reports loss
            at epoch 1, 6 and every 10 epochs, so updates arrive in steps.
        should_stop : callable, optional
            Polled before every training batch; returning True aborts training
            and raises :class:`TrainingAborted`.
        """
        import torch
        from s2l.core.datasplit import save_split, split_pairs
        from s2l.core.device import use_gpu

        sp = Path(save_path) if save_path else Path(self.train_dir)
        sp.mkdir(parents=True, exist_ok=True)
        train_pairs, test_pairs = split_pairs(self.pairs, test_percent)
        if status_callback:
            status_callback(f"Training on {len(train_pairs)} image(s), {len(test_pairs)} kept for testing…")

        gpu = use_gpu()
        model = models.CellposeModel(gpu=gpu, pretrained_model=base_model)
        logger.info(
            f"Training from '{base_model}' on {'GPU' if gpu else 'CPU'}: {len(train_pairs)} train / "
            f"{len(test_pairs)} test images, epochs={epochs}, lr={learning_rate}, wd={weight_decay}, save_path={sp}"
        )

        handler = _EpochLogHandler(epochs, progress_callback)
        train.train_logger.addHandler(handler)
        if train.train_logger.getEffectiveLevel() > logging.INFO:
            train.train_logger.setLevel(logging.INFO)
        original_get_batch = train._get_batch

        def _get_batch_or_abort(*args, **kwargs):
            if should_stop and should_stop():
                raise TrainingAborted("Training stopped by user")
            return original_get_batch(*args, **kwargs)

        train._get_batch = _get_batch_or_abort
        try:
            # the test images are never shown to the training loop
            model_path, train_losses, _ = train.train_seg(
                model.net,
                train_files=[i for i, _ in train_pairs],
                train_labels_files=[m for _, m in train_pairs],
                load_files=True,
                normalize=normalize,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                n_epochs=epochs,
                # Cellpose skips images with < 5 cells by default (and crashes if none are left);
                # every labelled image the user provides should count
                min_train_masks=1,
                model_name=model_name,
                save_path=str(sp),
            )
        except torch.OutOfMemoryError as e:
            raise RuntimeError(
                "GPU ran out of memory during training. Close other GPU programs, "
                "switch to CPU in the sidebar, or use smaller / fewer images."
            ) from e
        finally:
            train._get_batch = original_get_batch
            train.train_logger.removeHandler(handler)

        if progress_callback:
            progress_callback(epochs, epochs, float(train_losses[-1]) if len(train_losses) else 0.0, 100)
        logger.info(f"Training complete — model saved at: {model_path}")
        split = save_split(str(model_path), "cellpose", train_pairs, test_pairs, base_model,
                           {"epochs": epochs, "learning_rate": learning_rate, "weight_decay": weight_decay,
                            "batch_size": batch_size, "normalize": normalize, "test_percent": test_percent,
                            "device": "GPU" if gpu else "CPU"},
                           train_losses)
        report = {"n_train": len(train_pairs), "n_test": len(test_pairs), "split_file": str(split)}
        return model_path, train_losses, report
