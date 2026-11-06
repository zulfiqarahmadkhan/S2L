"""Segmentation page: run Cellpose / StarDist and the ROI analysis."""
import logging
import os
import webbrowser
from pathlib import Path

import tempfile

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from s2l.core.channels import available_channels, build_partner_index, filter_channel, find_partner
from s2l.core.image_files import list_images, mask_path_for
from s2l.core.spreadsheet_parser import parse_fim_sheet
from s2l.ui.widgets import (
    ActionBar, Collapsible, FitStack, ModelPicker, SegmentedSwitch, ToggleSwitch, add_recent_folder,
    browse_row, card, field_row, open_in_file_manager, page_shell, page_with_action_bar,
    remember_text, section_header, set_status, show_error, variant,
)
from s2l.ui.workers import WorkerThread

logger = logging.getLogger(__name__)


def _fmt_seconds(s: float) -> str:
    m, s = divmod(int(round(s)), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


class SegmentationPage(QWidget):
    open_in_viewer = pyqtSignal(str)          # emits a folder path
    open_correction = pyqtSignal(str, list)   # mask folder, original image paths

    def __init__(self):
        super().__init__()
        self.worker_thread: WorkerThread | None = None
        self._last_mask_dir = ""
        self._last_output_dir = ""
        self._last_originals: list[str] = []
        self._spreadsheet_records = []   # parsed from xlsx
        self._filter_checks = {}
        self._build()

    # ── layout ────────────────────────────────────────────────────────────

    def _build(self):
        page, col = page_shell("Segmentation", "Run AI-powered cell segmentation and ROI analysis on your images.")

        # ── card: input / output ──────────────────────────────────────────
        dir_lay = QVBoxLayout()
        dir_lay.addLayout(section_header("Input"))
        self.input_mode = SegmentedSwitch([("Folder", "folder"), ("Spreadsheet (xlsx)", "spreadsheet")])
        self.input_mode.changed.connect(self._on_input_mode_changed)
        dir_lay.addLayout(field_row("Input mode", self.input_mode))

        # -- folder mode widgets --
        self._folder_widget = QWidget()
        fw_lay = QVBoxLayout(self._folder_widget)
        fw_lay.setContentsMargins(0, 0, 0, 0)
        self.base_dir_edit = QLineEdit()
        self.base_dir_edit.setPlaceholderText("Path to folder containing your images…")
        fw_lay.addLayout(field_row(
            "Input folder",
            browse_row(self.base_dir_edit, "Browse", self._pick_base_dir,
                       view_callback=lambda: self._view_dir(self.base_dir_edit)),
        ))

        # -- spreadsheet mode widgets --
        self._sheet_widget = QWidget()
        sw_lay = QVBoxLayout(self._sheet_widget)
        sw_lay.setContentsMargins(0, 0, 0, 0)
        sw_lay.setSpacing(10)
        self.xlsx_edit = QLineEdit()
        self.xlsx_edit.setPlaceholderText("Path to experiment .xlsx file…")
        xlsx_row = browse_row(self.xlsx_edit, "Browse", self._pick_xlsx)
        import_btn = variant(QPushButton("Parse"), "primary")
        import_btn.clicked.connect(self._parse_spreadsheet)
        xlsx_row.addWidget(import_btn)
        sw_lay.addLayout(field_row("Spreadsheet", xlsx_row))
        self._tag_count_label = QLabel("")
        self._tag_count_label.setObjectName("FieldLabel")
        sw_lay.addWidget(self._tag_count_label)
        self._build_filter_panel(sw_lay)

        self._input_stack = FitStack()
        self._input_stack.addWidget(self._folder_widget)
        self._input_stack.addWidget(self._sheet_widget)
        dir_lay.addWidget(self._input_stack)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Path to save masks and results…")
        dir_lay.addLayout(field_row(
            "Output folder",
            browse_row(self.output_dir_edit, "Browse", self._pick_output_dir,
                       view_callback=lambda: self._view_dir(self.output_dir_edit)),
            hint="Masks (*_cp_masks.tif), per-image Excel sheets, overlays and the summary are saved here. "
                 "Your input folder is never modified.",
        ))
        col.addWidget(card(dir_lay))

        # ── card: pipeline ────────────────────────────────────────────────
        pipe_lay = QVBoxLayout()
        pipe_lay.addLayout(section_header("Pipeline", "Choose which steps to run."))
        self.chk_seg = ToggleSwitch("Run segmentation", checked=True)
        self.chk_l2r = ToggleSwitch("Run label → ROI analysis (Excel + summary)", checked=True)
        pipe_lay.addWidget(self.chk_seg)
        pipe_lay.addWidget(self.chk_l2r)

        ch_row = QHBoxLayout()
        ch_row.setSpacing(12)
        self.seg_channel_combo = QComboBox()
        self.seg_channel_combo.addItem("All images", "")
        self.measure_channel_combo = QComboBox()
        self.measure_channel_combo.addItem("The segmented image", "")
        ch_row.addLayout(field_row("Segment on", self.seg_channel_combo), 1)
        ch_row.addLayout(field_row("Measure intensity in", self.measure_channel_combo), 1)
        pipe_lay.addLayout(ch_row)
        self.channel_hint = QLabel("")
        self.channel_hint.setObjectName("Hint")
        self.channel_hint.setWordWrap(True)
        pipe_lay.addWidget(self.channel_hint)
        self.seg_channel_combo.currentIndexChanged.connect(lambda _: self._update_channel_hint())
        self.measure_channel_combo.currentIndexChanged.connect(lambda _: self._update_channel_hint())

        opt_row = QHBoxLayout()
        opt_row.setSpacing(12)
        self.skip_chk = QCheckBox("Skip images that already have a mask in the output folder")
        self.skip_chk.setToolTip("Resume an interrupted run without redoing finished images.")
        opt_row.addWidget(self.skip_chk)
        opt_row.addStretch()
        pipe_lay.addLayout(opt_row)
        flt_row = QHBoxLayout()
        flt_row.setSpacing(10)
        self.min_size_chk = QCheckBox("Remove cells smaller than")
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 100000)
        self.min_size_spin.setValue(30)
        self.min_size_spin.setSuffix(" px")
        self.min_size_spin.setFixedWidth(120)
        self.border_chk = QCheckBox("Remove cells touching the image border")
        flt_row.addWidget(self.min_size_chk)
        flt_row.addWidget(self.min_size_spin)
        flt_row.addSpacing(16)
        flt_row.addWidget(self.border_chk)
        flt_row.addStretch()
        pipe_lay.addLayout(flt_row)
        settings_hint = QLabel("The settings of every run are saved as run_settings_<date>.json "
                               "in the output folder, so any result can be reproduced.")
        settings_hint.setObjectName("Hint")
        settings_hint.setWordWrap(True)
        pipe_lay.addWidget(settings_hint)
        col.addWidget(card(pipe_lay))

        # re-detect channels when the input changes (debounced)
        self._chan_timer = QTimer(self)
        self._chan_timer.setSingleShot(True)
        self._chan_timer.setInterval(400)
        self._chan_timer.timeout.connect(self._detect_channels)
        self.base_dir_edit.textChanged.connect(lambda _: self._chan_timer.start())

        # ── card: model ───────────────────────────────────────────────────
        model_lay = QVBoxLayout()
        model_lay.addLayout(section_header("Model", "Select the segmentation engine and model."))
        self.engine = SegmentedSwitch([("Cellpose", "cellpose"), ("StarDist", "stardist")])
        self.engine.changed.connect(lambda k: self._engine_stack.setCurrentIndex(0 if k == "cellpose" else 1))
        model_lay.addLayout(field_row("Engine", self.engine))

        self._cp_widget = QWidget()
        cp_lay = QVBoxLayout(self._cp_widget)
        cp_lay.setContentsMargins(0, 0, 0, 0)
        cp_lay.setSpacing(12)
        self.cp_picker = ModelPicker("cellpose", settings_key="seg_cellpose")
        cp_lay.addWidget(self.cp_picker)
        self.diameter_spin = QSpinBox()
        self.diameter_spin.setRange(0, 500)
        self.diameter_spin.setSpecialValueText("Auto")
        self.diameter_spin.setSuffix(" px")
        cp_lay.addLayout(field_row(
            "Cell diameter", self.diameter_spin,
            hint="Auto works for most images. Set it only for cells much smaller or larger than ~30 px.",
        ))

        self._sd_widget = QWidget()
        sd_lay = QVBoxLayout(self._sd_widget)
        sd_lay.setContentsMargins(0, 0, 0, 0)
        sd_lay.setSpacing(12)
        self.sd_picker = ModelPicker("stardist", settings_key="seg_stardist")
        sd_lay.addWidget(self.sd_picker)
        sd_row = QHBoxLayout()
        self.sd_prob_spin = QDoubleSpinBox()
        self.sd_prob_spin.setRange(0.0, 1.0)
        self.sd_prob_spin.setSingleStep(0.05)
        self.sd_prob_spin.setValue(0.5)
        self.sd_nms_spin = QDoubleSpinBox()
        self.sd_nms_spin.setRange(0.0, 1.0)
        self.sd_nms_spin.setSingleStep(0.05)
        self.sd_nms_spin.setValue(0.4)
        sd_row.addLayout(field_row("Probability threshold", self.sd_prob_spin,
                                   "Objects below this confidence are discarded."), 1)
        sd_row.addLayout(field_row("Overlap (NMS) threshold", self.sd_nms_spin,
                                   "Overlapping objects above this are merged."), 1)
        sd_lay.addLayout(sd_row)
        self._sd_warn = QLabel("")
        self._sd_warn.setObjectName("StatusWarning")
        self._sd_warn.setWordWrap(True)
        self._sd_warn.setVisible(False)
        sd_lay.addWidget(self._sd_warn)

        self._engine_stack = FitStack()
        self._engine_stack.addWidget(self._cp_widget)
        self._engine_stack.addWidget(self._sd_widget)
        model_lay.addWidget(self._engine_stack)

        # advanced Cellpose parameters, collapsed by default
        adv = QVBoxLayout()
        adv.setSpacing(12)
        adv_row = QHBoxLayout()
        adv_row.setSpacing(12)
        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.1, 3.0)
        self.flow_spin.setSingleStep(0.1)
        self.flow_spin.setValue(0.4)
        self.flow_spin.setDecimals(1)
        self.cellprob_spin = QDoubleSpinBox()
        self.cellprob_spin.setRange(-6.0, 6.0)
        self.cellprob_spin.setSingleStep(0.1)
        self.cellprob_spin.setValue(0.0)
        self.cellprob_spin.setDecimals(1)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(0, 10000)
        self.iter_spin.setSingleStep(100)
        self.iter_spin.setSpecialValueText("Auto")
        adv_row.addLayout(field_row("Flow threshold", self.flow_spin, "Default 0.4 — raise if cells are missed."), 1)
        adv_row.addLayout(field_row("Cell-probability threshold", self.cellprob_spin, "Default 0.0"), 1)
        adv_row.addLayout(field_row("Max iterations", self.iter_spin, "Auto = based on diameter."), 1)
        adv.addLayout(adv_row)
        adv_btns = QHBoxLayout()
        reset_btn = variant(QPushButton("Reset to defaults"), "small")
        reset_btn.clicked.connect(self._reset_advanced)
        docs_btn = variant(QPushButton("Open Cellpose docs"), "small")
        docs_btn.clicked.connect(lambda: webbrowser.open("https://cellpose.readthedocs.io/en/latest/settings.html"))
        adv_btns.addWidget(reset_btn)
        adv_btns.addWidget(docs_btn)
        adv_btns.addStretch()
        adv.addLayout(adv_btns)
        self._advanced = Collapsible("Advanced Cellpose parameters", adv, expanded=False)
        self.engine.changed.connect(lambda k: self._advanced.setVisible(k == "cellpose"))
        model_lay.addWidget(self._advanced)
        col.addWidget(card(model_lay))
        col.addStretch()

        # ── pinned action bar: progress, results, start / stop ───────────
        self.bar = ActionBar("Start processing")
        prog = QGridLayout()
        prog.setHorizontalSpacing(10)
        prog.setVerticalSpacing(4)
        self.seg_bar = QProgressBar()
        self.roi_bar = QProgressBar()
        for row, (name, bar) in enumerate([("Segmentation", self.seg_bar), ("ROI analysis", self.roi_bar)]):
            lbl = QLabel(name)
            lbl.setObjectName("FieldLabel")
            prog.addWidget(lbl, row, 0)
            prog.addWidget(bar, row, 1)
        self.bar.content.addLayout(prog)
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setObjectName("StatusPending")
        self.status_lbl.setWordWrap(True)
        self.bar.content.addWidget(self.status_lbl)
        self.open_out_btn = QPushButton("📂 Open output folder")
        self.open_out_btn.setVisible(False)
        self.open_out_btn.clicked.connect(lambda: open_in_file_manager(self._last_output_dir))
        self._correction_btn = variant(QPushButton("🖊 Correct masks"), "success")
        self._correction_btn.setVisible(False)
        self._correction_btn.clicked.connect(self._open_correction)
        self.preview_btn = QPushButton("👁 Preview one image")
        self.preview_btn.setToolTip("Try the current settings on a single image before the full run")
        self.preview_btn.clicked.connect(self._preview)
        self.bar.extras.addWidget(self.preview_btn)
        self.bar.extras.addWidget(self.open_out_btn)
        self.bar.extras.addWidget(self._correction_btn)
        self.run_btn, self.stop_btn = self.bar.start_btn, self.bar.stop_btn
        self.run_btn.clicked.connect(self._run)
        self.stop_btn.clicked.connect(self._stop)

        self.setLayout(page_with_action_bar(page, self.bar))

        for edit, key in [(self.base_dir_edit, "seg/input"), (self.output_dir_edit, "seg/output"),
                          (self.xlsx_edit, "seg/xlsx")]:
            remember_text(edit, key)

    def _build_filter_panel(self, parent_layout):
        self._filter_frame = QFrame()
        self._filter_frame.setObjectName("Card")
        fl = QVBoxLayout(self._filter_frame)
        fl.setContentsMargins(16, 14, 16, 14)
        fl.setSpacing(10)
        fl_title = QLabel("Filters")
        fl_title.setObjectName("SectionTitle")
        fl.addWidget(fl_title)

        def _filter_section(label_text):
            head = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setObjectName("FieldLabel")
            head.addWidget(lbl)
            head.addStretch()
            sel_btn = variant(QPushButton("Select all"), "small")
            unsel_btn = variant(QPushButton("Unselect all"), "small")
            head.addWidget(sel_btn)
            head.addWidget(unsel_btn)
            fl.addLayout(head)
            grid = QGridLayout()
            grid.setSpacing(6)
            fl.addLayout(grid)
            return grid, sel_btn, unsel_btn

        self._well_checks_layout, self._well_sel_btn, self._well_unsel_btn = _filter_section("Wells")
        self._stage_checks_layout, self._stage_sel_btn, self._stage_unsel_btn = _filter_section("Stage")
        self._chan_checks_layout, self._chan_sel_btn, self._chan_unsel_btn = _filter_section("Channel")
        self._type_checks_layout, self._type_sel_btn, self._type_unsel_btn = _filter_section("Type")
        self._filter_frame.setVisible(False)
        parent_layout.addWidget(self._filter_frame)

    # ── environment ───────────────────────────────────────────────────────

    def set_env(self, info: dict):
        """Called once the background environment probe finishes."""
        if not info.get("stardist_ok"):
            self._sd_warn.setText(f"StarDist: {info.get('stardist_msg', 'not available')}")
            self._sd_warn.setVisible(True)

    def set_input_folder(self, path: str):
        self.input_mode.set_value("folder")
        self.base_dir_edit.setText(path)

    # ── helpers ───────────────────────────────────────────────────────────

    def _reset_advanced(self):
        self.flow_spin.setValue(0.4)
        self.cellprob_spin.setValue(0.0)
        self.iter_spin.setValue(0)

    def _on_input_mode_changed(self, key):
        self._input_stack.setCurrentIndex(0 if key == "folder" else 1)
        self._spreadsheet_records.clear()

    def _pick_xlsx(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select spreadsheet", "", "Excel (*.xlsx *.xls);;All (*)")
        if p:
            self.xlsx_edit.setText(p)

    def _parse_spreadsheet(self):
        xlsx_path = self.xlsx_edit.text().strip()
        if not xlsx_path or not os.path.isfile(xlsx_path):
            QMessageBox.warning(self, "No file", "Select a valid .xlsx file first.")
            return
        try:
            self._spreadsheet_records = parse_fim_sheet(xlsx_path)
            self._rebuild_filters()
            self._update_filter_count()
            self._filter_frame.setVisible(bool(self._spreadsheet_records))
            self._detect_channels()
        except Exception as e:
            show_error(self, "Parse error", "The spreadsheet could not be read.", str(e))

    def _rebuild_filters(self):
        """Build checkbox groups from the parsed records."""
        records = self._spreadsheet_records
        self._filter_checks = {}  # (category, value) -> QCheckBox
        cols = 6

        def _populate(grid, category, sel_btn, unsel_btn):
            while grid.count():
                item = grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            values = sorted({r[category] for r in records if r.get(category)})
            cbs = []
            for i, val in enumerate(values):
                cb = QCheckBox(val)
                cb.setChecked(True)
                cb.toggled.connect(lambda _: self._update_filter_count())
                self._filter_checks[(category, val)] = cb
                grid.addWidget(cb, i // cols, i % cols)
                cbs.append(cb)
            for btn in (sel_btn, unsel_btn):
                try:
                    btn.clicked.disconnect()
                except TypeError:
                    pass
            sel_btn.clicked.connect(lambda _c=False, c=cbs: self._set_all_checks(c, True))
            unsel_btn.clicked.connect(lambda _c=False, c=cbs: self._set_all_checks(c, False))

        _populate(self._well_checks_layout, "well", self._well_sel_btn, self._well_unsel_btn)
        _populate(self._stage_checks_layout, "stage", self._stage_sel_btn, self._stage_unsel_btn)
        _populate(self._chan_checks_layout, "channel", self._chan_sel_btn, self._chan_unsel_btn)
        _populate(self._type_checks_layout, "type", self._type_sel_btn, self._type_unsel_btn)

    def _set_all_checks(self, checkboxes, checked: bool):
        for cb in checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._update_filter_count()

    def _get_filtered_records(self) -> list[dict]:
        """Return records matching all checked filters (AND across categories)."""
        checked_by_cat: dict[str, set[str]] = {}
        all_by_cat: dict[str, set[str]] = {}
        for (cat, val), cb in self._filter_checks.items():
            all_by_cat.setdefault(cat, set()).add(val)
            if cb.isChecked():
                checked_by_cat.setdefault(cat, set()).add(val)
        result = []
        for r in self._spreadsheet_records:
            match = True
            for cat in all_by_cat:
                allowed = checked_by_cat.get(cat, set())
                rec_val = r.get(cat, "")
                if (rec_val and rec_val not in allowed) or (not rec_val and not allowed):
                    match = False
                    break
            if match:
                result.append(r)
        return result

    def _update_filter_count(self):
        filtered = self._get_filtered_records()
        self._tag_count_label.setText(f"{len(filtered)} / {len(self._spreadsheet_records)} images selected")

    def _pick_base_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder", self.base_dir_edit.text())
        if d:
            self.base_dir_edit.setText(d)

    def _pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_dir_edit.text())
        if d:
            self.output_dir_edit.setText(d)

    def set_model(self, engine: str, path: str):
        """Select a trained model (from the Training page)."""
        self.engine.set_value(engine)
        (self.cp_picker if engine == "cellpose" else self.sd_picker).set_custom(path)

    def refresh_models(self):
        """Reload the "my trained models" lists (after training a new one)."""
        self.cp_picker.refresh()
        self.sd_picker.refresh()

    def _chosen_models(self):
        """(custom Cellpose path or None, StarDist pretrained name, custom StarDist dir or None)."""
        cp_kind, cp_val = self.cp_picker.value()
        sd_kind, sd_val = self.sd_picker.value()
        return (cp_val if cp_kind == "custom" else None,
                sd_val if sd_kind == "pretrained" else "2D_versatile_fluo",
                sd_val if sd_kind == "custom" else None)

    def _view_dir(self, line_edit: QLineEdit):
        path = line_edit.text().strip()
        if path and os.path.isdir(path):
            self.open_in_viewer.emit(path)
        else:
            QMessageBox.information(self, "No folder", "Enter or browse a valid folder first.")

    # ── run / stop ────────────────────────────────────────────────────────

    def _run(self):
        is_sheet = self.input_mode.value() == "spreadsheet"
        base_dir = self.base_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not is_sheet and not os.path.isdir(base_dir):
            QMessageBox.warning(self, "Missing input", "Select an existing input folder.")
            return
        if is_sheet and not self._spreadsheet_records:
            QMessageBox.warning(self, "No data", "Parse a spreadsheet first.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Missing output", "Select an output folder.")
            return
        if not self.chk_seg.isChecked() and not self.chk_l2r.isChecked():
            QMessageBox.warning(self, "Nothing selected", "Enable at least one pipeline step.")
            return

        engine = self.engine.value()
        custom_cp, _, custom_sd = self._chosen_models()
        if self.chk_seg.isChecked():
            kind, val = (self.cp_picker if engine == "cellpose" else self.sd_picker).value()
            if kind == "custom" and not val:
                QMessageBox.warning(self, "No model chosen",
                                    "Choose one of your trained models (or Browse for one), "
                                    "or pick a pretrained model.")
                return
            if engine == "cellpose" and custom_cp and not os.path.exists(custom_cp):
                QMessageBox.warning(self, "Model not found", f"Custom model not found:\n{custom_cp}")
                return
            if engine == "stardist" and custom_sd and not os.path.isfile(os.path.join(custom_sd, "config.json")):
                QMessageBox.warning(self, "Model not found", f"No StarDist model (config.json) in:\n{custom_sd}")
                return

        file_list = None
        if is_sheet:
            file_list = [r["path"] for r in self._get_filtered_records()]
            if not file_list:
                QMessageBox.warning(self, "No images", "The current filters select no images.")
                return
        elif not list_images(base_dir):
            QMessageBox.warning(self, "No images", f"No image files found directly in:\n{base_dir}")
            return
        else:
            add_recent_folder(base_dir)
        seg_ch = self.seg_channel_combo.currentData() or None
        if not filter_channel(file_list or list_images(base_dir), seg_ch):
            QMessageBox.warning(self, "No images", f"No images of channel {seg_ch} in the input.")
            return

        self._last_output_dir = output_dir
        self._last_originals = filter_channel(file_list or list_images(base_dir), seg_ch)
        self._correction_btn.setVisible(False)
        self.open_out_btn.setVisible(False)
        self.seg_bar.setValue(0)
        self.roi_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("Stop")
        set_status(self.status_lbl, "Loading model and processing… (the first run downloads the model)", "Pending")

        self.worker_thread = WorkerThread(
            None if is_sheet else base_dir, output_dir, self.diameter_spin.value(),
            self.chk_seg.isChecked(), self.chk_l2r.isChecked(),
            file_list=file_list,
            seg_channel=seg_ch,
            measure_channel=self.measure_channel_combo.currentData() or None,
            skip_existing=self.skip_chk.isChecked(),
            **self._model_kwargs(),
        )
        self.preview_btn.setEnabled(False)   # a second model on the GPU at the same time risks running out of memory
        self.worker_thread.cellpose_progress.connect(lambda v: self.seg_bar.setValue(int(v)))
        self.worker_thread.labels2rois_progress.connect(lambda v: self.roi_bar.setValue(int(v)))
        self.worker_thread.finished.connect(self._done)
        self.worker_thread.error_occurred.connect(
            lambda msg, details: show_error(self, "Processing failed", msg, details))
        self.worker_thread.start()

    def _model_kwargs(self) -> dict:
        """Engine / model / filter settings shared by the full run and the preview."""
        custom_cp, sd_name, custom_sd = self._chosen_models()
        return dict(
            engine=self.engine.value(),
            custom_model_path=custom_cp,
            flow_threshold=self.flow_spin.value(),
            cellprob_threshold=self.cellprob_spin.value(),
            max_iter=self.iter_spin.value(),
            stardist_model=sd_name,
            stardist_custom_dir=custom_sd,
            prob_thresh=self.sd_prob_spin.value(),
            nms_thresh=self.sd_nms_spin.value(),
            min_size=self.min_size_spin.value() if self.min_size_chk.isChecked() else 0,
            remove_border=self.border_chk.isChecked(),
        )

    # ── channels ──────────────────────────────────────────────────────────

    def _current_files(self) -> list[str]:
        if self.input_mode.value() == "spreadsheet":
            return [r["path"] for r in self._spreadsheet_records]
        d = self.base_dir_edit.text().strip()
        return list_images(d) if d and os.path.isdir(d) else []

    def _detect_channels(self):
        """Fill the channel choices from the tags in the input file names."""
        files = self._current_files()
        chans = available_channels(files)
        for combo, first in ((self.seg_channel_combo, ("All images", "")),
                             (self.measure_channel_combo, ("The segmented image", ""))):
            keep = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(*first)
            for c in chans:
                combo.addItem(c, c)
            idx = combo.findData(keep)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.setEnabled(bool(chans))
            combo.blockSignals(False)
        self._update_channel_hint()

    def _update_channel_hint(self):
        files = self._current_files()
        if not files:
            self.channel_hint.setText("")
            return
        chans = available_channels(files)
        if not chans:
            self.channel_hint.setText("No channel names (Ph, Bf, R, …) found in the file names — "
                                      "every image is segmented and measured on itself.")
            return
        seg = self.seg_channel_combo.currentData()
        meas = self.measure_channel_combo.currentData()
        targets = filter_channel(files, seg or None)
        text = f"{len(targets)} image(s) will be segmented"
        if meas:
            idx = build_partner_index(files, meas)
            found = sum(1 for t in targets if find_partner(t, idx))
            text += f"; intensities measured in the matching {meas} image ({found}/{len(targets)} found)"
            if found < len(targets):
                text += " — images without a match are measured on themselves"
        self.channel_hint.setText(text + f".   Channels in this input: {', '.join(chans)}.")

    # ── preview ───────────────────────────────────────────────────────────

    def _preview(self):
        files = filter_channel(self._current_files(), self.seg_channel_combo.currentData() or None)
        if not files:
            QMessageBox.information(self, "No images", "Choose an input folder (or parse a spreadsheet) first.")
            return
        PreviewDialog(files, self._model_kwargs(), self.diameter_spin.value(), self).exec()

    def _stop(self):
        # Cooperative stop: the worker finishes the current image, then exits
        # and _done() runs via QThread.finished. Never terminate() a thread
        # that may be in the middle of GPU work.
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping…")
            set_status(self.status_lbl, "Stopping after the current image…", "Warning")

    def _done(self):
        w = self.worker_thread
        self.run_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        if w is None:
            return
        self._last_mask_dir = str(w.mask_dir())
        st = w.stats
        if w.succeeded:
            parts = []
            if w.run_seg:
                parts.append(f"{st['segmented']}/{st['images']} images segmented")
                if st["skipped"]:
                    parts.append(f"{st['skipped']} skipped (already done)")
            if w.run_l2r:
                parts.append(f"{st['roi_files']} Excel sheets")
            parts.append(f"{st['cells']:,} cells")
            if st["unpaired"]:
                parts.append(f"{st['unpaired']} without a {w.measure_channel} partner")
            set_status(self.status_lbl, "✓ Done: " + ", ".join(parts) + f" in {_fmt_seconds(st['seconds'])}.",
                       "Success")
            self._correction_btn.setVisible(True)
        elif w.is_stopping():
            set_status(self.status_lbl, f"Stopped by user after {_fmt_seconds(st['seconds'])}.", "Warning")
        else:
            set_status(self.status_lbl, "Failed — see the error message.", "Error")
        self.open_out_btn.setVisible(os.path.isdir(self._last_output_dir))

    def _open_correction(self):
        """Open the Correction page on the masks from the last run."""
        self.open_correction.emit(self._last_mask_dir, list(self._last_originals))


class PreviewDialog(QDialog):
    """Segment a single image with the current settings and show the result."""

    def __init__(self, files: list[str], model_kwargs: dict, diameter: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview segmentation")
        self.resize(980, 860)
        self._files, self._kw, self._diameter = files, model_kwargs, diameter
        self._tmp = tempfile.mkdtemp(prefix="s2l_preview_")
        self._worker: WorkerThread | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)
        row = QHBoxLayout()
        self.combo = QComboBox()
        for f in files[:500]:
            self.combo.addItem(Path(f).name, f)
        self.run_btn = variant(QPushButton("Run preview"), "primary")
        self.run_btn.clicked.connect(self._run)
        row.addWidget(self.combo, 1)
        row.addWidget(self.run_btn)
        lay.addLayout(row)
        self.status = QLabel("Choose an image and press “Run preview”. Uses the current model and settings.")
        self.status.setObjectName("StatusPending")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.view = QLabel("")
        self.view.setObjectName("ImagePanel")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(880, 680)
        lay.addWidget(self.view, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        lay.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)

    def _run(self):
        img = self.combo.currentData()
        self.run_btn.setEnabled(False)
        set_status(self.status, "Segmenting… (the first run loads the model)", "Pending")
        self._worker = WorkerThread(None, self._tmp, self._diameter, True, False, file_list=[img],
                                    save_settings=False, **self._kw)
        self._worker.finished.connect(lambda: self._done(img))
        self._worker.error_occurred.connect(lambda m, d: show_error(self, "Preview failed", m, d))
        self._worker.start()

    def _done(self, img):
        from s2l.ui.manual_correction import ndarray_to_qimage, render_pair_thumbnail
        self.run_btn.setEnabled(True)
        w = self._worker
        mask = mask_path_for(img, self._tmp)
        if not (w and w.succeeded and mask.exists()):
            set_status(self.status, "Preview failed.", "Error")
            return
        rgb = render_pair_thumbnail(img, mask, 1100)
        pix = QPixmap.fromImage(ndarray_to_qimage(rgb))
        self.view.setPixmap(pix.scaled(self.view.width() - 10, self.view.height() - 10,
                                       Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation))
        set_status(self.status, f"✓ {w.stats['cells']:,} cells found in {_fmt_seconds(w.stats['seconds'])}. "
                                "Change settings on the Segmentation page and preview again, or start the full run.",
                   "Success")

    def reject(self):
        import shutil
        w, tmp = self._worker, self._tmp
        if w is not None and w.isRunning():
            # don't freeze the window waiting (the first run may be loading / downloading the
            # model): let the worker finish in the background, then clean up
            w.stop()
            try:
                w.finished.disconnect()
            except TypeError:
                pass
            _BACKGROUND.append(w)
            w.finished.connect(lambda: (_BACKGROUND.remove(w) if w in _BACKGROUND else None,
                                        shutil.rmtree(tmp, ignore_errors=True)))
        else:
            shutil.rmtree(tmp, ignore_errors=True)
        super().reject()


_BACKGROUND: list = []   # preview workers still finishing after their dialog was closed
