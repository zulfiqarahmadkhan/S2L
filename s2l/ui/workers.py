"""Background threads: segmentation + ROI pipeline, training, environment probe."""
import logging
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from s2l.core.device import use_gpu
from s2l.core.image_files import (
    MASK_SUFFIX, list_images, list_masks, original_stem,
)

logger = logging.getLogger(__name__)


class EnvProbe(QThread):
    """Imports torch / cellpose / tensorflow off the UI thread and reports status."""
    ready = pyqtSignal(dict)

    def run(self):
        from s2l.core.models_info import probe_environment
        try:
            info = probe_environment()
        except Exception as e:
            info = {"gpu": False, "cellpose_ok": False, "cellpose_msg": str(e),
                    "stardist_ok": False, "stardist_msg": str(e), "gpu_name": ""}
        self.ready.emit(info)


class WorkerThread(QThread):
    """Runs segmentation and/or ROI analysis. Use QThread's own ``finished`` signal.

    Masks are always written to *output_dir* (``<stem>_cp_masks.tif``), so raw
    data folders are never modified.

    Multi-channel experiments: *seg_channel* limits segmentation to one channel
    (e.g. "Ph"), and *measure_channel* measures intensities in the matching image
    of another channel (e.g. "R") of the same well / stage / type.
    """
    cellpose_progress = pyqtSignal(float)
    labels2rois_progress = pyqtSignal(float)
    error_occurred = pyqtSignal(str, str)   # friendly message, technical details

    def __init__(self, base_dir, output_dir, diameter, run_seg, run_l2r,
                 engine="cellpose", custom_model_path=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, max_iter=0,
                 stardist_model="2D_versatile_fluo", stardist_custom_dir=None,
                 prob_thresh=0.5, nms_thresh=0.4,
                 file_list=None, seg_channel=None, measure_channel=None,
                 skip_existing=False, min_size=0, remove_border=False, save_settings=True):
        super().__init__()
        self.base_dir = Path(base_dir) if base_dir else None
        self.output_dir = Path(output_dir)
        self.diameter = diameter
        self.run_seg = run_seg
        self.run_l2r = run_l2r
        self.engine = engine
        self.custom_model_path = custom_model_path
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.max_iter = max_iter if max_iter > 0 else None
        self.stardist_model = stardist_model
        self.stardist_custom_dir = stardist_custom_dir
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        # explicit list of paths (spreadsheet mode); folder mode lists base_dir
        self.file_list = file_list
        self.seg_channel = seg_channel or None
        self.measure_channel = measure_channel or None
        self.skip_existing = skip_existing
        self.min_size = min_size
        self.remove_border = remove_border
        self.save_settings = save_settings
        self.succeeded = False
        self.stats = {"images": 0, "segmented": 0, "skipped": 0, "cells": 0, "roi_files": 0,
                      "unpaired": 0, "seconds": 0.0}
        self._stop = False

    def stop(self):
        self._stop = True

    def is_stopping(self) -> bool:
        return self._stop

    def all_files(self) -> list[str]:
        if self.file_list:
            return list(self.file_list)
        return list_images(self.base_dir) if self.base_dir and self.base_dir.is_dir() else []

    def inputs(self) -> list[str]:
        """Images to segment / measure (only the chosen channel, if one is set)."""
        from s2l.core.channels import filter_channel
        return filter_channel(self.all_files(), self.seg_channel)

    def _make_segmenter(self):
        # lazy imports keep torch / tensorflow out of app startup;
        # loading happens here, on the worker thread, so the UI stays responsive
        if self.engine == "stardist":
            from s2l.core.stardist_segmenter import StarDistSegmenter
            if self.stardist_custom_dir:
                return StarDistSegmenter.from_model_dir(self.stardist_custom_dir)
            return StarDistSegmenter(model_name=self.stardist_model)
        from s2l.core.segmenter import CellposeSegmenter
        return CellposeSegmenter(custom_model_path=self.custom_model_path)

    def settings_dict(self) -> dict:
        from s2l import __version__
        return {
            "s2l_version": __version__,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_folder": str(self.base_dir) if self.base_dir else None,
            "input_files": len(self.all_files()),
            "output_folder": str(self.output_dir),
            "steps": {"segmentation": self.run_seg, "roi_analysis": self.run_l2r},
            "engine": self.engine,
            "device": "GPU" if use_gpu() else "CPU",
            "cellpose": {"model": self.custom_model_path or "cpsam (built-in)", "diameter": self.diameter,
                         "flow_threshold": self.flow_threshold, "cellprob_threshold": self.cellprob_threshold,
                         "max_iter": self.max_iter},
            "stardist": {"model": self.stardist_custom_dir or self.stardist_model,
                         "prob_thresh": self.prob_thresh, "nms_thresh": self.nms_thresh},
            "channels": {"segment": self.seg_channel or "all images",
                         "measure": self.measure_channel or "same image"},
            "filters": {"min_cell_size_px": self.min_size, "remove_border_cells": self.remove_border},
            "skip_existing_masks": self.skip_existing,
        }

    def run(self):
        import json
        t0 = time.time()
        try:
            inputs = self.inputs()
            self.stats["images"] = len(inputs)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.save_settings:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                with open(self.output_dir / f"run_settings_{stamp}.json", "w", encoding="utf-8") as f:
                    json.dump(self.settings_dict(), f, indent=2)
            if self.run_seg and not self._stop:
                todo = inputs
                if self.skip_existing:
                    from s2l.core.image_files import mask_path_for
                    todo = [p for p in inputs if not mask_path_for(p, self.output_dir).exists()]
                    self.stats["skipped"] = len(inputs) - len(todo)
                results = []
                if todo:
                    seg = self._make_segmenter()
                    if self.engine == "stardist":
                        params = dict(prob_thresh=self.prob_thresh, nms_thresh=self.nms_thresh)
                    else:
                        params = dict(diameter=int(self.diameter),
                                      flow_threshold=self.flow_threshold,
                                      cellprob_threshold=self.cellprob_threshold,
                                      max_iter=self.max_iter)
                    params.update(progress_callback=self.cellpose_progress.emit,
                                  should_stop=self.is_stopping)
                    results = seg.segment_files(todo, str(self.output_dir), **params) or []
                else:
                    self.cellpose_progress.emit(100)
                if self.min_size > 0 or self.remove_border:
                    results = self._apply_filters(results)
                self.stats["segmented"] = len(results)
                self.stats["cells"] = sum(n for _, n in results)
            if self.run_l2r and not self._stop:
                self._labels_to_rois(inputs)
            self.succeeded = not self._stop
        except Exception as e:
            from s2l.ui.widgets import friendly_error
            logger.exception(f"Worker error: {e}")
            self.error_occurred.emit(friendly_error(str(e)), traceback.format_exc())
        finally:
            self.stats["seconds"] = time.time() - t0

    def _apply_filters(self, results):
        """Drop small / border cells from the masks this run wrote; returns updated counts."""
        import numpy as np
        from skimage.io import imread, imsave
        from skimage.segmentation import relabel_sequential
        from s2l.core.image_files import mask_path_for
        from s2l.core.mask_edit import remove_border, remove_small
        out = []
        for img_path, _ in results:
            mp = mask_path_for(img_path, self.output_dir)
            m = imread(str(mp)).astype(np.int32)
            if self.min_size > 0:
                remove_small(m, self.min_size)
            if self.remove_border:
                remove_border(m)
            m = relabel_sequential(m)[0]
            imsave(str(mp), m.astype(np.uint16), check_contrast=False)
            out.append((img_path, int(m.max())))
        return out

    def mask_dir(self) -> Path:
        """Where this run's masks are: the output folder, or (older runs) next to the images."""
        if list_masks(self.output_dir) if self.output_dir.is_dir() else []:
            return self.output_dir
        if not self.file_list and self.base_dir and list_masks(self.base_dir):
            return self.base_dir
        return self.output_dir

    def _labels_to_rois(self, inputs):
        from s2l.core.channels import build_partner_index, find_partner
        from s2l.core.roi_converter import ROIVisualizer
        from s2l.core.summary import generate_summary_sheet

        mask_search_dir = self.mask_dir()
        masks = list_masks(mask_search_dir) if mask_search_dir.is_dir() else []
        originals = {Path(p).stem: p for p in inputs}
        masks = [mp for mp in masks if original_stem(mp) in originals]
        total = len(masks)
        if not total:
            logger.warning(f"No *{MASK_SUFFIX}.tif files for these images in {mask_search_dir}")
            self.labels2rois_progress.emit(100)
            return
        self.labels2rois_progress.emit(0)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        partners = build_partner_index(self.all_files(), self.measure_channel) if self.measure_channel else None
        cells = 0
        written = []   # this run's sheets only, so stale files in the folder don't enter the summary
        for i, mp in enumerate(masks):
            if self._stop:
                break
            orig = originals[original_stem(mp)]
            measure_on = orig
            if partners is not None:
                partner = find_partner(orig, partners)
                if partner:
                    measure_on = partner
                else:
                    self.stats["unpaired"] += 1
                    logger.warning(f"No {self.measure_channel} image found for {Path(orig).name}; "
                                   "measuring the segmented image instead")
            try:
                stem = mp.stem
                xlsx = self.output_dir / f"{stem}.xlsx"
                n = ROIVisualizer(
                    str(mp), str(measure_on), str(xlsx),
                    str(self.output_dir / f"{stem}_ROI.png"),
                    show_labels=True,
                ).save_rois_to_excel()
                written.append(xlsx)
                self.stats["roi_files"] += 1
                cells += n or 0
            except Exception as e:
                logger.error(f"ROI error {mp}: {e}")
            self.labels2rois_progress.emit(((i + 1) / total) * 100)
        # measured cells cover every mask of these images, including skipped ones
        self.stats["cells"] = cells
        if not self._stop and written:
            sd = self.output_dir / "SummarySheet"
            sd.mkdir(exist_ok=True)
            generate_summary_sheet(str(self.output_dir), str(sd / "Summary.xlsx"), files=written)


class _JobProcessWorker(QThread):
    """Runs ``python -m <module> <job.json>`` in a separate process and relays its output.

    Training and model testing run this way: loading / training CUDA models inside the GUI
    process could crash the whole app in native code, and a separate process also frees all
    GPU memory when it ends. The job prints ``PROGRESS …``, ``STATUS …``, ``RESULT <json>``
    or ``STOPPED``; creating ``<job>.stop`` asks it to stop.
    """
    status  = pyqtSignal(str)
    stopped = pyqtSignal()
    error   = pyqtSignal(str, str)               # friendly message, details

    module = ""

    def __init__(self, job: dict):
        super().__init__()
        self._job_data = job
        self._job = None
        self._abort = False

    def stop(self):
        self._abort = True
        if self._job:
            try:
                Path(self._job + ".stop").touch()
            except OSError:
                pass

    def is_stopping(self) -> bool:
        return self._abort

    def _on_line(self, line: str):
        """PROGRESS lines (subclasses)."""

    def _on_result(self, result):
        """RESULT line (subclasses)."""

    def run(self):
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from s2l.core.device import prefer_cpu
        fd, self._job = tempfile.mkstemp(prefix="s2l_job_", suffix=".json")
        os.close(fd)
        tail = []
        try:
            Path(self._job).write_text(json.dumps(dict(self._job_data, prefer_cpu=prefer_cpu())), encoding="utf-8")
            if self._abort:
                Path(self._job + ".stop").touch()
            root = str(Path(__file__).resolve().parents[2])
            env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""),
                       PYTHONIOENCODING="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            proc = subprocess.Popen([sys.executable, "-m", self.module, self._job], cwd=root, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", creationflags=flags)
            result = None
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("PROGRESS "):
                    self._on_line(line)
                elif line.startswith("STATUS "):
                    self.status.emit(line[7:])
                elif line.startswith("RESULT "):
                    result = json.loads(line[7:])
                elif line:
                    logger.info(line)
                    tail = (tail + [line])[-60:]
            code = proc.wait()
            if code == 2 or (self._abort and result is None):
                self.stopped.emit()
            elif code == 0 and result is not None:
                self._on_result(result)
            else:
                msg = next((ln for ln in reversed(tail) if "Error" in ln or "error" in ln),
                           f"The background process ended unexpectedly (code {code}).")
                from s2l.ui.widgets import friendly_error
                self.error.emit(friendly_error(msg), "\n".join(tail))
        except Exception as e:
            from s2l.ui.widgets import friendly_error
            self.error.emit(friendly_error(str(e)), traceback.format_exc() + "\n" + "\n".join(tail))
        finally:
            for f in (self._job, self._job + ".stop"):
                try:
                    os.remove(f)
                except OSError:
                    pass


class TrainingWorker(_JobProcessWorker):
    """Cellpose or StarDist training (in a separate process, see :class:`_JobProcessWorker`)."""
    progress = pyqtSignal(int, int, float, int)   # epoch, total, loss, pct
    done     = pyqtSignal(str, object)            # model_path, report dict
    module = "s2l.core.train_job"

    def __init__(self, engine: str, kwargs: dict):
        kw = dict(kwargs)
        for k in ("patch_size", "start_from"):
            if k in kw:
                kw[k] = list(kw[k])
        super().__init__({"engine": engine, "kwargs": kw})

    def _on_line(self, line):
        _, epoch, total, loss, pct = line.split()
        self.progress.emit(int(epoch), int(total), float(loss), int(pct))

    def _on_result(self, result):
        self.done.emit(result["path"], result["report"])


class EvalWorker(_JobProcessWorker):
    """Tests a trained model (and optionally its starting model) on the saved test images
    (in a separate process, see :class:`_JobProcessWorker`)."""
    progress = pyqtSignal(int)
    done     = pyqtSignal(str, object)            # results folder, {model label: metrics}
    module = "s2l.core.evaluation"

    def __init__(self, engine: str, models: list, test_pairs: list, out_dir: str, train_losses=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        super().__init__({"engine": engine, "models": [list(m) for m in models],
                          "pairs": [list(p) for p in test_pairs], "out_dir": out_dir,
                          "train_losses": list(train_losses or [])})

    def _on_line(self, line):
        self.progress.emit(int(line.split()[1]))

    def _on_result(self, result):
        self.done.emit(self._out_dir, result)


class DownloadWorker(QThread):
    """Downloads the given pretrained models ``[(engine, name)]`` one after another."""
    status = pyqtSignal(str)
    done   = pyqtSignal(list)                     # names that failed

    def __init__(self, items: list):
        super().__init__()
        self._items = items

    def run(self):
        from s2l.core.models_info import download_pretrained
        failed = []
        for engine, name in self._items:
            self.status.emit(f"Downloading {name}…")
            try:
                download_pretrained(engine, name)
            except Exception as e:
                logger.warning(f"Download of {name} failed: {e}")
                failed.append(name)
        self.done.emit(failed)
