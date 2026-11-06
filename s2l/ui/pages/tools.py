"""Tools page: Image Enhancer, external tools and the installation check."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar, QProgressDialog, QPushButton,
    QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from s2l.core.enhance import EnhanceSettings, enhance, save_enhanced
from s2l.ui.theme import COLORS
from s2l.ui.widgets import (
    browse_row, card, field_row, page_shell, remember_text, scrollable, section_header, set_status,
    show_error, variant,
)

_PREVIEW = 380


class _EnhanceWorker(QThread):
    """Enhances one image (preview) or a whole folder (batch) off the UI thread."""
    result = pyqtSignal(object)                 # float array, single-image mode
    progress = pyqtSignal(int, int)             # done, total
    finished_batch = pyqtSignal(int, int)       # written, failed
    error = pyqtSignal(str, str)

    def __init__(self, settings: EnhanceSettings, image=None, files=None, out_dir=None, fmt=".tif"):
        super().__init__()
        self.settings, self.image = settings, image
        self.files, self.out_dir, self.fmt = files or [], out_dir, fmt
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import traceback
        from skimage.io import imread
        try:
            if self.image is not None:
                self.result.emit(enhance(self.image, self.settings))
                return
            ok = failed = 0
            for i, f in enumerate(self.files):
                if self._stop:
                    break
                try:
                    out = enhance(imread(f), self.settings)
                    save_enhanced(out, str(Path(self.out_dir) / f"{Path(f).stem}_enhanced{self.fmt}"))
                    ok += 1
                except Exception:
                    failed += 1
                self.progress.emit(i + 1, len(self.files))
            self.finished_batch.emit(ok, failed)
        except Exception as e:
            self.error.emit(str(e), traceback.format_exc())


class _PipInstaller(QThread):
    """Runs ``pip install`` for this Python environment off the UI thread."""
    done = pyqtSignal(bool, str)   # success, pip output

    def __init__(self, packages: list[str]):
        super().__init__()
        self.packages = packages

    def run(self):
        try:
            res = subprocess.run([sys.executable, "-m", "pip", "install", *self.packages],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.done.emit(res.returncode == 0, res.stdout + res.stderr)
        except Exception as e:
            self.done.emit(False, str(e))


class EnhancerDialog(QDialog):
    """Denoise, background-correct, contrast-enhance, sharpen or upsample images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("S2L — Image Enhancer")
        self.resize(980, 760)
        self.image = None
        self.result = None
        self.image_path = ""
        self.worker: _EnhanceWorker | None = None

        page, col = page_shell("Image Enhancer",
                               "Improve images before segmentation. Steps run top to bottom; "
                               "the original files are never changed.")

        opts = QVBoxLayout()
        opts.addLayout(section_header("Steps"))
        g = QGridLayout()
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(10)
        self.bg_chk = QCheckBox("Background subtraction")
        self.bg_radius = self._spin(5, 500, 50, " px")
        g.addWidget(self.bg_chk, 0, 0)
        g.addLayout(field_row("Radius", self.bg_radius), 0, 1)
        self.dn_combo = QComboBox()
        for label, key in [("No denoising", "none"), ("Gaussian blur", "gaussian"),
                           ("Median filter", "median"), ("Non-local means (best, slow)", "nlmeans")]:
            self.dn_combo.addItem(label, key)
        self.dn_strength = self._dspin(0.3, 5.0, 1.0)
        g.addLayout(field_row("Denoise", self.dn_combo), 1, 0)
        g.addLayout(field_row("Strength", self.dn_strength), 1, 1)
        self.clahe_chk = QCheckBox("Contrast enhancement (CLAHE)")
        self.clahe_clip = self._dspin(0.001, 0.1, 0.01, step=0.005, decimals=3)
        g.addWidget(self.clahe_chk, 2, 0)
        g.addLayout(field_row("Clip limit", self.clahe_clip), 2, 1)
        self.sharp_chk = QCheckBox("Sharpen (unsharp mask)")
        self.sharp_amount = self._dspin(0.1, 5.0, 1.0)
        g.addWidget(self.sharp_chk, 3, 0)
        g.addLayout(field_row("Amount", self.sharp_amount), 3, 1)
        self.up_chk = QCheckBox("Upsample 2× (for very small cells)")
        g.addWidget(self.up_chk, 4, 0)
        opts.addLayout(g)
        col.addWidget(card(opts))

        btns = QHBoxLayout()
        load_btn = QPushButton("Load image…")
        load_btn.clicked.connect(self._load)
        self.apply_btn = variant(QPushButton("Preview"), "primary")
        self.apply_btn.clicked.connect(self._apply)
        self.save_btn = QPushButton("Save result…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        batch_btn = QPushButton("Process a folder…")
        batch_btn.clicked.connect(self._batch)
        for b in (load_btn, self.apply_btn, self.save_btn, batch_btn):
            btns.addWidget(b)
        col.addLayout(btns)

        panels = QHBoxLayout()
        panels.setSpacing(16)
        self.orig_label = self._panel("Load an image to start")
        self.proc_label = self._panel("Result appears here")
        panels.addWidget(self.orig_label)
        panels.addWidget(self.proc_label)
        col.addLayout(panels)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        col.addWidget(self.progress)
        self.status = QLabel("Ready.")
        self.status.setObjectName("StatusPending")
        self.status.setWordWrap(True)
        col.addWidget(self.status)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(page))

    # ── small builders ────────────────────────────────────────────────────

    @staticmethod
    def _spin(lo, hi, val, suffix=""):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    @staticmethod
    def _dspin(lo, hi, val, step=0.1, decimals=2):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setValue(val)
        return s

    @staticmethod
    def _panel(text):
        lbl = QLabel(text)
        lbl.setObjectName("ImagePanel")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumSize(_PREVIEW, _PREVIEW)
        return lbl

    def settings(self) -> EnhanceSettings:
        return EnhanceSettings(
            background=self.bg_chk.isChecked(), background_radius=self.bg_radius.value(),
            denoise=self.dn_combo.currentData(), denoise_strength=self.dn_strength.value(),
            clahe=self.clahe_chk.isChecked(), clahe_clip=self.clahe_clip.value(),
            sharpen=self.sharp_chk.isChecked(), sharpen_amount=self.sharp_amount.value(),
            upsample=2 if self.up_chk.isChecked() else 1,
        )

    @staticmethod
    def _show(label: QLabel, arr: np.ndarray):
        from s2l.ui.manual_correction import ndarray_to_qimage, render_view
        pix = QPixmap.fromImage(ndarray_to_qimage(render_view(arr)))
        label.setPixmap(pix.scaled(_PREVIEW, _PREVIEW, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation))

    # ── actions ───────────────────────────────────────────────────────────

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "",
                                              "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All (*)")
        if not path:
            return
        from skimage.io import imread
        from s2l.core.enhance import to_float
        try:
            self.image = imread(path)
            self.image_path = path
            self._show(self.orig_label, to_float(self.image))
            self.proc_label.setPixmap(QPixmap())
            self.proc_label.setText("Press Preview")
            self.save_btn.setEnabled(False)
            set_status(self.status, f"Loaded {Path(path).name} {self.image.shape}", "Success")
        except Exception as e:
            show_error(self, "Cannot load", "This image could not be read.", str(e))

    def _apply(self):
        if self.image is None:
            QMessageBox.information(self, "No image", "Load an image first.")
            return
        self.apply_btn.setEnabled(False)
        set_status(self.status, "Processing…", "Pending")
        self.worker = _EnhanceWorker(self.settings(), image=self.image)
        self.worker.result.connect(self._on_result)
        self.worker.error.connect(lambda m, d: (show_error(self, "Enhancement failed", m, d),
                                                self.apply_btn.setEnabled(True)))
        self.worker.start()

    def _on_result(self, arr):
        self.result = arr
        self._show(self.proc_label, arr)
        self.apply_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        set_status(self.status, f"Done — result {arr.shape[1]}×{arr.shape[0]} px.", "Success")

    def _save(self):
        if self.result is None:
            return
        suggested = str(Path(self.image_path).with_name(Path(self.image_path).stem + "_enhanced.tif"))
        path, _ = QFileDialog.getSaveFileName(self, "Save enhanced image", suggested,
                                              "TIFF 16-bit (*.tif);;PNG 8-bit (*.png)")
        if path:
            try:
                save_enhanced(self.result, path)
                set_status(self.status, f"Saved {path}", "Success")
            except Exception as e:
                show_error(self, "Save failed", "The image could not be saved.", str(e))

    def _batch(self):
        from s2l.core.image_files import list_images
        src = QFileDialog.getExistingDirectory(self, "Folder with images to enhance")
        if not src:
            return
        files = list_images(src)
        if not files:
            QMessageBox.information(self, "No images", "No images found in that folder.")
            return
        dst = QFileDialog.getExistingDirectory(self, "Where to save the enhanced images")
        if not dst:
            return
        if os.path.abspath(dst) == os.path.abspath(src):
            QMessageBox.warning(self, "Same folder", "Choose a different output folder than the input folder.")
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        set_status(self.status, f"Enhancing {len(files)} images…", "Pending")
        self.worker = _EnhanceWorker(self.settings(), files=files, out_dir=dst)
        self.worker.progress.connect(lambda d, t: self.progress.setValue(d))
        self.worker.finished_batch.connect(
            lambda ok, bad: set_status(self.status, f"✓ {ok} images saved to {dst}"
                                       + (f", {bad} failed" if bad else ""), "Success" if not bad else "Warning"))
        self.worker.start()

    def closeEvent(self, ev):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(ev)


class ToolsPage(QWidget):
    def __init__(self):
        super().__init__()
        page, col = page_shell("Tools", "Image enhancement, external utilities and system checks.")

        enh = QVBoxLayout()
        enh.addLayout(section_header(
            "Image Enhancer",
            "Denoise, subtract uneven background, boost contrast, sharpen or upsample images — "
            "one image with a before/after preview, or a whole folder at once."))
        enh_btn = variant(QPushButton("Open Image Enhancer"), "primary")
        enh_btn.clicked.connect(lambda: EnhancerDialog(self).exec())
        enh.addWidget(enh_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        col.addWidget(card(enh))

        # ── channel completeness check ────────────────────────────────────
        cc = QVBoxLayout()
        cc.addLayout(section_header(
            "Channel check",
            "Before a run, see for every well / stage which channel images (Ph, Bf, R, …) exist and "
            "which are missing."))
        self.cc_edit = QLineEdit()
        self.cc_edit.setPlaceholderText("Folder with the images (e.g. …/Images)")
        cc_row = browse_row(self.cc_edit, "Browse", self._pick_cc_folder)
        cc_go = variant(QPushButton("Check"), "primary")
        cc_go.clicked.connect(self._run_channel_check)
        cc_row.addWidget(cc_go)
        cc.addLayout(field_row("Images folder", cc_row))
        self.cc_sub = QCheckBox("Include subfolders")
        cc.addWidget(self.cc_sub)
        self.cc_summary = QLabel("")
        self.cc_summary.setObjectName("Hint")
        self.cc_summary.setWordWrap(True)
        cc.addWidget(self.cc_summary)
        self.cc_table = QTableWidget(0, 0)
        self.cc_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cc_table.verticalHeader().setVisible(False)
        self.cc_table.setMinimumHeight(220)
        self.cc_table.setVisible(False)
        cc.addWidget(self.cc_table)
        self.cc_export = QPushButton("Export table (CSV)…")
        self.cc_export.setVisible(False)
        self.cc_export.clicked.connect(self._export_channel_check)
        cc.addWidget(self.cc_export, alignment=Qt.AlignmentFlag.AlignLeft)
        remember_text(self.cc_edit, "tools/cc_folder")
        col.addWidget(card(cc))

        # ── pretrained models ─────────────────────────────────────────────
        mdl = QVBoxLayout()
        mdl.addLayout(section_header(
            "Pretrained models",
            "Models that come with Cellpose and StarDist. They are downloaded once and kept on this PC."))
        self.models_table = QTableWidget(0, 4)
        self.models_table.setHorizontalHeaderLabels(["Engine", "Model", "Use for", "Status"])
        self.models_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setWordWrap(True)
        mh = self.models_table.horizontalHeader()
        mh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        mh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        mdl.addWidget(self.models_table)
        mrow = QHBoxLayout()
        self.dl_btn = variant(QPushButton("Download missing models"), "primary")
        self.dl_btn.clicked.connect(self._download_models)
        mrow.addWidget(self.dl_btn)
        self.dl_status = QLabel("")
        self.dl_status.setObjectName("Hint")
        mrow.addWidget(self.dl_status, 1)
        mdl.addLayout(mrow)
        col.addWidget(card(mdl))
        self._refresh_models()

        ext = QVBoxLayout()
        ext.addLayout(section_header("External tools", "Open other programs in the same environment."))
        row = QHBoxLayout()
        cp_btn = QPushButton("Cellpose GUI")
        cp_btn.clicked.connect(lambda: self._launch([sys.executable, "-m", "cellpose"]))
        self._acdc_btn = QPushButton("Cell-ACDC")
        self._acdc_btn.setToolTip("Installed automatically on first use if it is missing")
        self._acdc_btn.clicked.connect(self._launch_cellacdc)
        row.addWidget(cp_btn)
        row.addWidget(self._acdc_btn)
        ext.addLayout(row)
        col.addWidget(card(ext))

        chk = QVBoxLayout()
        chk.addLayout(section_header("System check", "Lists installed packages and whether the GPU works."))
        verify_btn = QPushButton("Verify installation")
        verify_btn.clicked.connect(self._run_verify)
        chk.addWidget(verify_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        col.addWidget(card(chk))
        col.addStretch()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(page))

    # ── pretrained models ─────────────────────────────────────────────────

    def _refresh_models(self):
        from s2l.core.models_info import MODEL_DESCRIPTIONS, MODEL_SIZES_MB, all_pretrained, is_downloaded
        rows = all_pretrained()
        self.models_table.setRowCount(len(rows))
        missing = 0
        for r, (engine, name, _) in enumerate(rows):
            ok = is_downloaded(engine, name)
            missing += not ok
            status = "✓ downloaded" if ok else f"missing (~{MODEL_SIZES_MB.get(name, 5)} MB)"
            for c, v in enumerate(["Cellpose" if engine == "cellpose" else "StarDist", name,
                                   MODEL_DESCRIPTIONS.get(name, ""), status]):
                self.models_table.setItem(r, c, QTableWidgetItem(v))
        self.models_table.resizeRowsToContents()
        h = self.models_table.horizontalHeader().height() + sum(
            self.models_table.rowHeight(r) for r in range(len(rows))) + 4
        self.models_table.setFixedHeight(h)
        self.dl_btn.setEnabled(missing > 0)
        self.dl_btn.setText("Download missing models" if missing else "All models downloaded")

    def _download_models(self):
        from s2l.core.models_info import all_pretrained, is_downloaded
        from s2l.ui.workers import DownloadWorker
        items = [(e, n) for e, n, _ in all_pretrained() if not is_downloaded(e, n)]
        if not items:
            return
        self.dl_btn.setEnabled(False)
        self._dl = DownloadWorker(items)
        self._dl.status.connect(self.dl_status.setText)
        self._dl.done.connect(self._on_downloaded)
        self._dl.start()

    def _on_downloaded(self, failed: list):
        self.dl_status.setText("Some downloads failed (check the internet connection): " + ", ".join(failed)
                               if failed else "✓ Done.")
        self._refresh_models()

    # ── channel check ─────────────────────────────────────────────────────

    def _pick_cc_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Images folder", self.cc_edit.text())
        if d:
            self.cc_edit.setText(d)
            self._run_channel_check()

    def _run_channel_check(self):
        from s2l.core.channels import pairing_table
        from s2l.core.image_files import IMAGE_EXTENSIONS, is_derived
        folder = Path(self.cc_edit.text().strip())
        if not folder.is_dir():
            QMessageBox.warning(self, "No folder", "Choose an existing images folder.")
            return
        walker = folder.rglob("*") if self.cc_sub.isChecked() else folder.glob("*")
        files = [str(p) for p in walker if p.suffix.lower() in IMAGE_EXTENSIONS and not is_derived(p)]
        chans, table = pairing_table(files)
        if not table:
            self.cc_summary.setText(f"{len(files)} images, but no well / stage / channel names were recognised "
                                    "in the file names.")
            self.cc_table.setVisible(False)
            self.cc_export.setVisible(False)
            return
        keys = sorted(table, key=lambda k: (k[0][:1], int(k[0][1:]) if k[0][1:].isdigit() else 0, k[1:]))
        headers = ["Well", "Timepoint", "Stage", "Type"] + chans
        self.cc_table.setColumnCount(len(headers))
        self.cc_table.setRowCount(len(keys))
        self.cc_table.setHorizontalHeaderLabels(headers)
        # a channel is only "missing" if other fields of the same type have it
        # (e.g. Raw images never come as BfRGB / PhRGB, so those are not expected)
        expected: dict[str, set] = {}
        for k, have in table.items():
            expected.setdefault(k[3], set()).update(have)
        missing_total = 0
        incomplete = 0
        self._cc_rows = []
        for r, k in enumerate(keys):
            have = table[k]
            exp = expected.get(k[3], set())
            row = list(k) + ["✓" if c in have else ("✗ missing" if c in exp else "") for c in chans]
            n_miss = sum(1 for c in exp if c not in have)
            missing_total += n_miss
            incomplete += bool(n_miss)
            self._cc_rows.append(row)
            for c, v in enumerate(row):
                item = QTableWidgetItem(v or "—")
                if v == "✗ missing":
                    item.setForeground(QColor(COLORS["error"]))
                elif v == "✓":
                    item.setForeground(QColor(COLORS["success"]))
                if c >= 4 and chans[c - 4] in have:
                    item.setToolTip(Path(have[chans[c - 4]]).name)
                self.cc_table.setItem(r, c, item)
        self.cc_table.resizeColumnsToContents()
        self._cc_headers = headers
        self.cc_summary.setText(
            f"{len(files)} images → {len(keys)} fields (well / stage / type), channels: {', '.join(chans)}. "
            + ("✓ Every field has every channel." if not missing_total else
               f"✗ {incomplete} field(s) are missing {missing_total} image(s) in total."))
        self.cc_table.setVisible(True)
        self.cc_export.setVisible(True)

    def _export_channel_check(self):
        import csv
        path, _ = QFileDialog.getSaveFileName(self, "Export channel check",
                                              str(Path(self.cc_edit.text()) / "channel_check.csv"), "CSV (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(self._cc_headers)
                wr.writerows(self._cc_rows)

    def _launch(self, cmd):
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            show_error(self, "Could not launch", "The program could not be started.", str(e))

    def _launch_cellacdc(self):
        """Start Cell-ACDC, installing it into this environment first if it is missing."""
        if importlib.util.find_spec("cellacdc") is not None:
            self._start_cellacdc()
            return
        # older environments: install it now (new installs get it from requirements.txt)
        self._acdc_btn.setEnabled(False)
        self._acdc_btn.setText("Installing Cell-ACDC…")
        self._progress = QProgressDialog(
            "Cell-ACDC is not installed yet — installing it now (pip install cellacdc[gui]).\n"
            "This takes a minute or two; it starts automatically when done.", None, 0, 0, self)
        self._progress.setWindowTitle("Installing Cell-ACDC")
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.show()
        self._installer = _PipInstaller(["cellacdc[gui]>=1.8"])
        self._installer.done.connect(self._on_acdc_installed)
        self._installer.start()

    def _on_acdc_installed(self, ok: bool, output: str):
        self._progress.close()
        self._acdc_btn.setEnabled(True)
        self._acdc_btn.setText("Cell-ACDC")
        importlib.invalidate_caches()
        if ok and importlib.util.find_spec("cellacdc") is not None:
            self._start_cellacdc()
        else:
            show_error(self, "Installation failed",
                       "Cell-ACDC could not be installed automatically. Check the internet connection, "
                       "or run “pip install cellacdc[gui]” in the S2L environment.", output[-4000:])

    def _start_cellacdc(self):
        # cellacdc's __main__ only defines run(), so "python -m cellacdc" would do nothing
        self._launch([sys.executable, "-c", "from cellacdc.__main__ import run; run()"])

    def _run_verify(self):
        script_path = Path(__file__).resolve().parents[3] / "verify_installation.py"
        try:
            res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
            output = res.stdout or res.stderr
            box = QMessageBox(self)
            box.setWindowTitle("Installation check")
            box.setText("System check completed. Click “Show Details…” for the full report.")
            box.setInformativeText(output.split("-" * 60)[-1].strip() if "-" * 60 in output else output[:300])
            box.setDetailedText(output)
            box.exec()
        except Exception as e:
            show_error(self, "Check failed", "Could not run the installation check.", str(e))
