"""Training page: fine-tune Cellpose or train StarDist, then test the model on held-out images."""
import os
import sys
import time
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from s2l.ui.widgets import (
    ActionBar, FitStack, LossChart, ModelPicker, SegmentedSwitch, ToggleSwitch, browse_row, card,
    field_row, open_in_file_manager, page_shell, page_with_action_bar, register_model, remember_text,
    section_header, set_status, show_error, trained_models, unregister_model, variant,
)
from s2l.ui.workers import EvalWorker, TrainingWorker


class TrainingPage(QWidget):
    open_in_viewer = pyqtSignal(str)
    use_model = pyqtSignal(str, str)     # engine, model path

    def __init__(self):
        super().__init__()
        self._worker: TrainingWorker | None = None
        self._eval: EvalWorker | None = None
        self._last_entry: dict | None = None
        self._last_results = ""
        self._build()

    def _build(self):
        page, col = page_shell("Training", "Fine-tune a Cellpose or StarDist model on your own labelled data, "
                                           "then test it on images it has never seen.")

        self.engine = SegmentedSwitch([("Cellpose", "cellpose"), ("StarDist", "stardist")])
        self.engine.changed.connect(self._on_engine_changed)
        col.addWidget(self.engine)

        # ── card: data ────────────────────────────────────────────────────
        data_lay = QVBoxLayout()
        data_lay.addLayout(section_header(
            "Training data", "A folder of images, each with a mask named <image><suffix>.tif."))
        self.train_dir_edit = QLineEdit()
        self.train_dir_edit.setPlaceholderText("Folder with images + masks…")
        self.train_dir_edit.editingFinished.connect(self._update_split_hint)
        data_lay.addLayout(field_row(
            "Data folder",
            browse_row(self.train_dir_edit, "Browse", self._pick_train_dir, view_callback=self._view_train_dir),
        ))
        row = QHBoxLayout()
        row.setSpacing(12)
        self.mask_filter = QLineEdit("_cp_masks")
        self.mask_filter.editingFinished.connect(self._update_split_hint)
        row.addLayout(field_row("Mask file suffix", self.mask_filter), 1)
        self.model_name_edit = QLineEdit("my_model")
        row.addLayout(field_row("Save model as", self.model_name_edit), 1)
        data_lay.addLayout(row)

        split = QHBoxLayout()
        split.setSpacing(12)
        self.train_pct = QSpinBox()
        self.test_pct = QSpinBox()
        for s in (self.train_pct, self.test_pct):
            s.setRange(0, 100)
            s.setSuffix(" %")
        self.train_pct.setRange(50, 100)
        self.test_pct.setRange(0, 50)
        self.test_pct.setValue(20)
        self.train_pct.setValue(80)
        self.train_pct.valueChanged.connect(lambda v: self._sync_split(self.test_pct, 100 - v))
        self.test_pct.valueChanged.connect(lambda v: self._sync_split(self.train_pct, 100 - v))
        self.train_pct.valueChanged.connect(lambda _: self._update_split_hint())
        self.test_pct.valueChanged.connect(lambda _: self._update_split_hint())
        split.addLayout(field_row("Training", self.train_pct), 1)
        split.addLayout(field_row("Testing", self.test_pct), 1)
        split.addStretch(1)
        data_lay.addLayout(split)
        self.split_hint = QLabel("Testing images are chosen at random, never used for training, and saved "
                                 "with the model so “Test model” uses exactly them.")
        self.split_hint.setObjectName("Hint")
        self.split_hint.setWordWrap(True)
        data_lay.addWidget(self.split_hint)

        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("Empty = save in the data folder")
        data_lay.addLayout(field_row(
            "Model save folder (optional)", browse_row(self.save_dir_edit, "Browse", self._pick_save_dir)))
        self.results_dir_edit = QLineEdit()
        self.results_dir_edit.setPlaceholderText("Empty = next to the model (<model>_test_<date>)")
        data_lay.addLayout(field_row(
            "Test results folder (optional)", browse_row(self.results_dir_edit, "Browse", self._pick_results_dir)))
        col.addWidget(card(data_lay))

        # ── card: settings (engine specific) ──────────────────────────────
        set_lay = QVBoxLayout()
        set_lay.addLayout(section_header("Model and settings"))

        cp_w = QWidget()
        cp_lay = QVBoxLayout(cp_w)
        cp_lay.setContentsMargins(0, 0, 0, 0)
        cp_lay.setSpacing(12)
        self.cp_start = ModelPicker("cellpose", label="Start from", settings_key="train_cellpose")
        cp_lay.addWidget(self.cp_start)
        cp_row = QHBoxLayout()
        cp_row.setSpacing(12)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(100)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0000001, 0.01)
        self.lr_spin.setSingleStep(0.000005)
        self.lr_spin.setDecimals(7)
        self.lr_spin.setValue(0.00001)
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0.0, 1.0)
        self.wd_spin.setDecimals(3)
        self.wd_spin.setSingleStep(0.01)
        self.wd_spin.setValue(0.1)
        cp_row.addLayout(field_row("Epochs", self.epochs_spin, "Passes over the data; 100 is typical"), 1)
        cp_row.addLayout(field_row("Learning rate", self.lr_spin, "Default 0.00001"), 1)
        cp_row.addLayout(field_row("Weight decay", self.wd_spin, "Default 0.1"), 1)
        cp_lay.addLayout(cp_row)
        self.norm_chk = ToggleSwitch("Normalize images", checked=True)
        cp_lay.addWidget(self.norm_chk)

        sd_w = QWidget()
        sd_lay = QVBoxLayout(sd_w)
        sd_lay.setContentsMargins(0, 0, 0, 0)
        sd_lay.setSpacing(12)
        self.sd_start = ModelPicker(
            "stardist", label="Start from", settings_key="train_stardist",
            extra=[("scratch", "From scratch (new network)",
                    "Train a new StarDist network from random weights. Needs more images and epochs "
                    "than fine-tuning a pretrained model.")])
        self.sd_start.changed.connect(self._on_sd_start_changed)
        sd_lay.addWidget(self.sd_start)
        self.sd_epochs_spin = QSpinBox()
        self.sd_epochs_spin.setRange(1, 2000)
        self.sd_epochs_spin.setValue(100)
        self.sd_steps_spin = QSpinBox()
        self.sd_steps_spin.setRange(10, 10000)
        self.sd_steps_spin.setValue(100)
        self.sd_rays_spin = QSpinBox()
        self.sd_rays_spin.setRange(8, 128)
        self.sd_rays_spin.setValue(32)
        r1 = QHBoxLayout()
        r1.setSpacing(12)
        r1.addLayout(field_row("Epochs", self.sd_epochs_spin, "Default 100"), 1)
        r1.addLayout(field_row("Steps per epoch", self.sd_steps_spin, "Default 100"), 1)
        self._rays_row = field_row("Rays", self.sd_rays_spin, "Default 32")
        r1.addLayout(self._rays_row, 1)
        sd_lay.addLayout(r1)
        self.sd_patch_h = QSpinBox()
        self.sd_patch_w = QSpinBox()
        for s in (self.sd_patch_h, self.sd_patch_w):
            s.setRange(64, 1024)
            s.setValue(256)
            s.setSingleStep(32)
            s.setSuffix(" px")
        r2 = QHBoxLayout()
        r2.setSpacing(12)
        r2.addLayout(field_row("Patch height", self.sd_patch_h), 1)
        r2.addLayout(field_row("Patch width", self.sd_patch_w), 1)
        sd_lay.addLayout(r2)
        self._sd_start_note = QLabel("")
        self._sd_start_note.setObjectName("Hint")
        self._sd_start_note.setWordWrap(True)
        sd_lay.addWidget(self._sd_start_note)
        self._sd_warn = QLabel("")
        self._sd_warn.setObjectName("StatusWarning")
        self._sd_warn.setWordWrap(True)
        self._sd_warn.setVisible(False)
        sd_lay.addWidget(self._sd_warn)
        if sys.platform == "win32":
            w = QLabel("TensorFlow has no GPU support on Windows, so StarDist training runs on the CPU "
                       "and can be slow. Cellpose training uses the GPU.")
            w.setObjectName("Hint")
            w.setWordWrap(True)
            sd_lay.addWidget(w)

        self._param_stack = FitStack()
        self._param_stack.addWidget(cp_w)
        self._param_stack.addWidget(sd_w)
        set_lay.addWidget(self._param_stack)
        col.addWidget(card(set_lay))
        self._on_sd_start_changed()

        # ── card: progress (chart + log + result) ─────────────────────────
        log_lay = QVBoxLayout()
        log_lay.addLayout(section_header("Progress", "Loss should go down; Cellpose reports it every 10 epochs."))
        self._chart = LossChart()
        log_lay.addWidget(self._chart)
        self._result_lbl = QLabel("")
        self._result_lbl.setObjectName("RunSummary")
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._result_lbl.setVisible(False)
        log_lay.addWidget(self._result_lbl)
        self._log_box = QTextEdit()
        self._log_box.setObjectName("LogBox")
        self._log_box.setReadOnly(True)
        self._log_box.setFixedHeight(90)
        self._log_box.setPlaceholderText("Loss values appear here while training runs.")
        log_lay.addWidget(self._log_box)
        col.addWidget(card(log_lay))

        # ── card: your trained models ─────────────────────────────────────
        mdl = QVBoxLayout()
        mdl.addLayout(section_header("Your trained models",
                                     "Every model trained in S2L is listed here (newest first)."))
        self._models_table = QTableWidget(0, 7)
        self._models_table.setHorizontalHeaderLabels(
            ["Name", "Engine", "Date", "Train / test", "Started from", "Test AP@0.5", "Location"])
        self._models_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._models_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._models_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._models_table.verticalHeader().setVisible(False)
        hh = self._models_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self._models_table.setMinimumHeight(150)
        self._models_table.itemDoubleClicked.connect(lambda _: self._use_selected())
        mdl.addWidget(self._models_table)
        mb = QHBoxLayout()
        use_btn = variant(QPushButton("Use in Segmentation"), "primary")
        use_btn.clicked.connect(self._use_selected)
        test_btn = QPushButton("🧪 Test selected model")
        test_btn.setToolTip("Test this model on the images kept aside when it was trained")
        test_btn.clicked.connect(lambda: self._test(self._selected_model()))
        open_btn = QPushButton("📂 Open folder")
        open_btn.clicked.connect(self._open_selected_folder)
        rm_btn = QPushButton("Remove from list")
        rm_btn.setToolTip("Only removes it from this list; the model file is kept")
        rm_btn.clicked.connect(self._remove_selected)
        for b in (use_btn, test_btn, open_btn, rm_btn):
            mb.addWidget(b)
        mb.addStretch()
        mdl.addLayout(mb)
        col.addWidget(card(mdl))
        col.addStretch()

        # ── pinned action bar ─────────────────────────────────────────────
        self.bar = ActionBar("Start training")
        self._train_bar = QProgressBar()
        self.bar.content.addWidget(self._train_bar)
        self._epoch_lbl = QLabel("Not started.")
        self._epoch_lbl.setObjectName("StatusPending")
        self.bar.content.addWidget(self._epoch_lbl)
        self.compare_chk = QCheckBox("Also test the starting model")
        self.compare_chk.setChecked(True)
        self.compare_chk.setToolTip("Test the model you started from on the same images, for a before / after "
                                    "comparison (not possible for StarDist “From scratch”).")
        self.compare_chk.setVisible(False)
        self.bar.extras.addWidget(self.compare_chk)
        self.test_btn = variant(QPushButton("🧪 Test model"), "success")
        self.test_btn.setToolTip("Test the new model on the testing images and save all metrics and figures")
        self.test_btn.setVisible(False)
        self.test_btn.clicked.connect(lambda: self._test(self._last_entry))
        self.bar.extras.addWidget(self.test_btn)
        self.open_results_btn = QPushButton("📂 Open test results")
        self.open_results_btn.setVisible(False)
        self.open_results_btn.clicked.connect(lambda: open_in_file_manager(self._last_results))
        self.bar.extras.addWidget(self.open_results_btn)
        self.open_model_btn = QPushButton("📂 Open model folder")
        self.open_model_btn.setVisible(False)
        self.open_model_btn.clicked.connect(
            lambda: open_in_file_manager(str(Path(self._last_entry["path"]).parent)))
        self.bar.extras.addWidget(self.open_model_btn)
        self._train_btn, self._stop_btn = self.bar.start_btn, self.bar.stop_btn
        self._train_btn.clicked.connect(self._train)
        self._stop_btn.clicked.connect(self._stop)
        self.setLayout(page_with_action_bar(page, self.bar))

        for edit, key in [(self.train_dir_edit, "train/dir"), (self.save_dir_edit, "train/save"),
                          (self.results_dir_edit, "train/results")]:
            remember_text(edit, key)
        self._refresh_table()
        self._update_split_hint()

    # ── model list ────────────────────────────────────────────────────────

    def refresh_models(self):
        self._refresh_table()
        self.cp_start.refresh()
        self.sd_start.refresh()

    def _refresh_table(self):
        models = trained_models()
        self._models_table.setRowCount(len(models))
        for r, m in enumerate(models):
            acc = ""
            if m.get("ap50_after") is not None:
                acc = f"{m['ap50_after']:.3f}"
                if m.get("ap50_before") is not None:
                    acc = f"{m['ap50_before']:.3f} → {acc}"
            tt = f"{m['n_train']} / {m['n_test']}" if m.get("n_train") is not None else str(m.get("n_images", ""))
            start = Path(str(m.get("start_from", ""))).name
            vals = [m.get("name", ""), m.get("engine", ""), m.get("date", ""), tt, start, acc, m.get("path", "")]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setData(Qt.ItemDataRole.UserRole, m)
                if c == 5:
                    item.setToolTip("Average precision at IoU 0.5 on the test images (starting model → trained)")
                self._models_table.setItem(r, c, item)

    def _selected_model(self):
        row = self._models_table.currentRow()
        item = self._models_table.item(row, 0) if row >= 0 else None
        if item is None:
            QMessageBox.information(self, "No model", "Select a model in the list first.")
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _use_selected(self):
        m = self._selected_model()
        if m:
            self.use_model.emit(m["engine"], m["path"])

    def _open_selected_folder(self):
        m = self._selected_model()
        if m:
            open_in_file_manager(str(Path(m["path"]).parent))

    def _remove_selected(self):
        m = self._selected_model()
        if m:
            unregister_model(m["path"])
            self.refresh_models()

    # ── helpers ───────────────────────────────────────────────────────────

    def set_env(self, info: dict):
        if not info.get("stardist_ok"):
            self._sd_warn.setText(f"StarDist: {info.get('stardist_msg', 'not available')}")
            self._sd_warn.setVisible(True)

    def _on_engine_changed(self, key: str):
        self._param_stack.setCurrentIndex(0 if key == "cellpose" else 1)

    def _on_sd_start_changed(self):
        kind, value = self.sd_start.value()
        scratch = kind == "scratch"
        self.sd_rays_spin.setEnabled(scratch)
        if scratch:
            self._sd_start_note.setText("")
        else:
            note = ("Fine-tuning: the number of rays, grid and image channels are taken from the starting "
                    "model, so “Rays” is not used.")
            if value == "2D_versatile_he":
                note += " 2D_versatile_he needs colour (RGB) training images."
            self._sd_start_note.setText(note)

    @staticmethod
    def _sync_split(other: QSpinBox, value: int):
        if other.value() != value:
            other.blockSignals(True)
            other.setValue(value)
            other.blockSignals(False)

    def _update_split_hint(self):
        from s2l.core.datasplit import find_pairs, split_pairs
        td = self.train_dir_edit.text().strip()
        base = ("Testing images are chosen at random, never used for training, and saved with the model "
                "so “Test model” uses exactly them.")
        if os.path.isdir(td):
            try:
                pairs = find_pairs(td, self.mask_filter.text().strip() or "_cp_masks")
                tr, te = split_pairs(pairs, self.test_pct.value())
                base = f"{len(pairs)} image+mask pair(s) found → {len(tr)} for training, {len(te)} for testing. " + base
            except OSError:
                pass
        self.split_hint.setText(base)

    def _pick_train_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select data folder", self.train_dir_edit.text())
        if d:
            self.train_dir_edit.setText(d)
            self._update_split_hint()

    def _pick_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select model save folder")
        if d:
            self.save_dir_edit.setText(d)

    def _pick_results_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select test results folder")
        if d:
            self.results_dir_edit.setText(d)

    def _view_train_dir(self):
        path = self.train_dir_edit.text().strip()
        if path and os.path.isdir(path):
            self.open_in_viewer.emit(path)
        else:
            QMessageBox.information(self, "No folder", "Enter or browse a valid folder first.")

    def _busy(self, busy: bool):
        self._train_btn.setEnabled(not busy)
        self._stop_btn.setEnabled(busy)
        self.test_btn.setEnabled(not busy)

    # ── training ──────────────────────────────────────────────────────────

    def _train(self):
        td = self.train_dir_edit.text().strip()
        if not os.path.isdir(td):
            QMessageBox.warning(self, "Missing input", "Select an existing data folder.")
            return

        engine = self.engine.value()
        picker = self.cp_start if engine == "cellpose" else self.sd_start
        kind, value = picker.value()
        if kind == "custom" and not value:
            QMessageBox.warning(self, "No model chosen", "Choose one of your trained models to start from "
                                                         "(or Browse for one), or pick a pretrained model.")
            return
        kw = {
            "train_dir":   td,
            "mask_filter": self.mask_filter.text().strip() or "_cp_masks",
            "model_name":  self.model_name_edit.text().strip() or "my_model",
            "save_path":   self.save_dir_edit.text().strip() or td,
            "test_percent": self.test_pct.value(),
        }
        if engine == "cellpose":
            kw.update(base_model=value or "cpsam", epochs=self.epochs_spin.value(), lr=self.lr_spin.value(),
                      weight_decay=self.wd_spin.value(), normalize=self.norm_chk.isChecked())
            start = value or "cpsam"
        else:
            kw.update(epochs=self.sd_epochs_spin.value(), steps=self.sd_steps_spin.value(),
                      patch_size=(self.sd_patch_h.value(), self.sd_patch_w.value()),
                      n_rays=self.sd_rays_spin.value(), start_from=(kind, value))
            start = "from scratch" if kind == "scratch" else value

        # where the new model will be written (Cellpose: <save>/models/<name>; StarDist: <save>/<name>)
        target = (Path(kw["save_path"]) / "models" / kw["model_name"] if engine == "cellpose"
                  else Path(kw["save_path"]) / kw["model_name"])
        if kind == "custom" and value and Path(value).resolve() == target.resolve():
            QMessageBox.warning(self, "Choose another name",
                                f"“{kw['model_name']}” is the model you start from. Save the new model under "
                                f"another name so the original is kept (and can be compared in the test).")
            return
        if target.exists():
            r = QMessageBox.question(
                self, "Model already exists",
                f"A model named “{kw['model_name']}” already exists in\n{target.parent}\n\n"
                f"Replace it? Its test split and results will no longer match. Choose No to pick another name.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return

        self._kw, self._engine_used, self._start_used = kw, engine, start
        self._train_bar.setValue(0)
        set_status(self._epoch_lbl, "Loading data and model…", "Pending")
        self._log_box.clear()
        self._chart.clear(kw["epochs"])
        self._result_lbl.setVisible(False)
        for b in (self.open_model_btn, self.test_btn, self.compare_chk, self.open_results_btn):
            b.setVisible(False)
        self._busy(True)

        self._worker = TrainingWorker(engine, kw)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_finished)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.status.connect(lambda t: set_status(self._epoch_lbl, t, "Pending"))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop(self):
        # Cooperative stop: the worker aborts at its next batch / image and emits `stopped`.
        for w in (self._worker, self._eval):
            if w is not None and w.isRunning():
                w.stop()
                self._stop_btn.setEnabled(False)
                set_status(self._epoch_lbl, "Stopping…", "Warning")

    def _on_stopped(self):
        self._busy(False)
        set_status(self._epoch_lbl, "Stopped by user.", "Warning")

    def _on_progress(self, epoch: int, total: int, loss: float, pct: int):
        self._train_bar.setValue(pct)
        set_status(self._epoch_lbl, f"Epoch {epoch} / {total}" + (f" — loss {loss:.4f}" if loss > 0 else ""),
                   "Pending")
        if loss > 0:
            self._log_box.append(f"Epoch {epoch:4d}/{total}   loss = {loss:.5f}")
            self._chart.add_point(epoch, loss)

    def _on_finished(self, model_path: str, report: dict):
        entry = {"name": Path(model_path).name, "engine": self._engine_used, "path": model_path,
                 "date": time.strftime("%Y-%m-%d %H:%M"), "epochs": self._kw.get("epochs"),
                 "n_images": report.get("n_images"), "n_train": report.get("n_train"),
                 "n_test": report.get("n_test"), "train_dir": self._kw.get("train_dir"),
                 "start_from": self._start_used}
        register_model(entry)
        self._last_entry = entry
        self.refresh_models()
        self._train_bar.setValue(100)
        self._busy(False)
        set_status(self._epoch_lbl, f"✓ Training complete — model saved: {model_path}", "Success")
        self._log_box.append(f"\n✓ Model saved: {model_path}")
        self.open_model_btn.setVisible(True)
        if report.get("n_test"):
            self._result_lbl.setText(
                f"Trained on {report['n_train']} image(s); {report['n_test']} image(s) were kept aside. "
                f"Press “🧪 Test model” to measure accuracy on them and save the metrics (CSV) and figures.")
            self.test_btn.setVisible(True)
            self.compare_chk.setVisible(entry["start_from"] != "from scratch")
        else:
            self._result_lbl.setText("Trained on all images (Testing = 0 %), so the model cannot be tested. "
                                     "Set Testing to e.g. 20 % to keep images aside for testing.")
        self._result_lbl.setVisible(True)

    def _on_error(self, msg: str, details: str):
        self._busy(False)
        set_status(self._epoch_lbl, "Failed.", "Error")
        self._log_box.append(f"\n✗ Error: {msg}")
        show_error(self, "Training / testing failed", msg, details)

    # ── testing ───────────────────────────────────────────────────────────

    def _test(self, entry: dict | None):
        from s2l.core.datasplit import load_split
        from s2l.core.models_info import PRETRAINED
        if not entry:
            return
        if (self._worker and self._worker.isRunning()) or (self._eval and self._eval.isRunning()):
            QMessageBox.information(self, "Busy", "Wait until the current training or test has finished.")
            return
        split = load_split(entry["path"])
        if not split or not split.get("test"):
            QMessageBox.information(
                self, "No test images",
                "This model has no saved test images (it was trained on all images, or trained before "
                "S2L saved the training / testing split). Train it again with Testing > 0 %.")
            return
        engine = split["engine"]
        pairs = [(t["image"], t["mask"]) for t in split["test"]]
        missing = [p for pair in pairs for p in pair if not os.path.isfile(p)]
        if missing:
            QMessageBox.warning(self, "Test images missing",
                                f"{len(missing)} test file(s) were moved or deleted, e.g.:\n{missing[0]}")
            return
        name = Path(entry["path"]).name
        models = [(name, "custom", entry["path"])]
        start = split.get("start_from", "")
        if self.compare_chk.isChecked() and start and start != "from scratch":
            kind = "pretrained" if start in PRETRAINED[engine] else "custom"
            if kind == "pretrained" or os.path.exists(start):
                models.append((f"{Path(start).name} (start)", kind, start))
        out = self.results_dir_edit.text().strip()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = (str(Path(out) / f"{name}_test_{stamp}") if out
               else str(Path(entry["path"]).parent / f"{name}_test_{stamp}"))

        self._test_entry = entry
        self.compare_chk.setVisible(start not in ("", "from scratch"))
        self._train_bar.setValue(0)
        self._busy(True)
        self.open_results_btn.setVisible(False)
        set_status(self._epoch_lbl, f"Testing on {len(pairs)} image(s)…", "Pending")
        self._log_box.append(f"\nTesting {', '.join(m[0] for m in models)} on {len(pairs)} test image(s)…")
        self._eval = EvalWorker(engine, models, pairs, out, split.get("train_losses"))
        self._eval.progress.connect(self._train_bar.setValue)
        self._eval.status.connect(lambda t: set_status(self._epoch_lbl, t, "Pending"))
        self._eval.done.connect(self._on_tested)
        self._eval.stopped.connect(self._on_stopped)
        self._eval.error.connect(self._on_error)
        self._eval.start()

    def _on_tested(self, out_dir: str, results: dict):
        self._busy(False)
        self._last_results = out_dir
        self._train_bar.setValue(100)
        labels = list(results)
        lines = [f"Test results ({len(labels)} model(s)) — saved to {out_dir}"]
        for lab in labels:
            s = results[lab]
            lines.append(f"• {lab}: AP@0.5 {s['ap@0.5']:.3f} · F1@0.5 {s['f1@0.5']:.3f} · "
                         f"precision {s['precision@0.5']:.3f} · recall {s['recall@0.5']:.3f} · "
                         f"mAP@0.5:0.95 {s['mAP@0.5:0.95']:.3f} · Dice {s['dice']:.3f} · AUROC {s['auroc']:.3f}")
        lines.append("All metrics with 95 % confidence intervals: summary_metrics.csv; figures: figures/ "
                     "(PNG 300 dpi + SVG); see README.txt.")
        self._result_lbl.setText("\n".join(lines))
        self._result_lbl.setVisible(True)
        set_status(self._epoch_lbl, "✓ Testing complete.", "Success")
        self.open_results_btn.setVisible(True)
        # remember the headline accuracy in the model list
        entry = dict(self._test_entry)
        entry["ap50_after"] = results[labels[0]]["ap@0.5"]
        entry["ap50_before"] = results[labels[1]]["ap@0.5"] if len(labels) > 1 else None
        entry["last_test"] = out_dir
        register_model(entry)
        self.refresh_models()
