"""Plotting page: distributions from a per-image sheet, comparisons from the summary sheet."""
import io
import os
import re
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from s2l.core.results import NUMERIC_METRICS
from s2l.ui.theme import COLORS, PALETTES
from s2l.ui.widgets import (
    browse_row, card, field_row, page_shell, remember_text, scrollable, section_header,
    set_status, show_error, variant,
)

# plot key -> (label for a per-cell sheet, label for the summary sheet)
PLOTS = {
    "area":      ("Cell size (Area) distribution",                      "Cells per image"),
    "intensity": ("Intensity (Mean Gray Value) distribution",           "Mean cell area per image"),
    "scatter":   ("Size vs. intensity (Area vs. Integrated Density)",   "Mean intensity per image"),
    "shape":     ("Shape (Circularity vs. Solidity)",                   "Mean area vs. mean intensity"),
}
_SERIES = ["#7c6aef", "#e5609a", "#3b9ee8", "#f0a030"]

# Saved figures always use this white "paper" style, whatever the app theme.
PRINT_COLORS = dict(PALETTES["light"], bg_card="#ffffff", bg_input="#ffffff", text="#1a1a1a",
                    text_secondary="#333333", plot_grid="#d0d0d0")


def _short(name: str, n: int = 28) -> str:
    return name if len(name) <= n else "…" + name[-(n - 1):]


def sheet_kind(df) -> str:
    return "summary" if "Number of Objects" in df.columns else "cells"


def _parse_kv(series, *keys):
    out = {k: [] for k in keys}
    for val in series.dropna().astype(str):
        try:
            d = {k.strip(): float(v) for k, v in (p.split("=") for p in val.split(","))}
            for k in keys:
                out[k].append(d[k])
        except (ValueError, KeyError):
            continue
    return [out[k] for k in keys]


def build_figure(df, plots: list[str], C=None):
    """Return a matplotlib Figure (Agg, thread-safe — no pyplot). *C* is the palette (default: app theme)."""
    from matplotlib.figure import Figure

    C = C or COLORS
    kind = sheet_kind(df)
    n = len(plots)
    rows, cols = {1: (1, 1), 2: (1, 2), 3: (1, 3)}.get(n, (2, 2))
    size = {1: (7, 5), 2: (12, 5), 3: (16, 4.8)}.get(n, (11, 8.5))
    fg, sub, grid = C["bg_card"], C["text_secondary"], C["plot_grid"]
    # constrained layout keeps rotated tick labels inside the figure
    fig = Figure(figsize=size, facecolor=fg, dpi=110, layout="constrained")
    axes = fig.subplots(rows, cols, squeeze=False).ravel()

    names = None
    if kind == "summary":
        names = [_short(re.sub(r"_cp_masks$", "", Path(str(p)).stem)) for p in df["File Location"]]

    def missing(ax, what):
        ax.text(0.5, 0.5, f"Column {what} not found", color=C["error"], ha="center", va="center",
                transform=ax.transAxes)

    def bars(ax, col, title, ylabel, colour):
        if col not in df.columns:
            return missing(ax, f"'{col}'")
        x = np.arange(len(df))
        ax.bar(x, df[col].astype(float), color=colour, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(title, color=C["text"], fontsize=12, pad=10)
        ax.set_ylabel(ylabel, color=sub)

    for i, (ax, key) in enumerate(zip(axes, plots)):
        ax.set_facecolor(C["bg_input"])
        colour = _SERIES[i % len(_SERIES)]
        if kind == "summary":
            if key == "area":
                bars(ax, "Number of Objects", "Cells per image", "Cells", colour)
            elif key == "intensity":
                bars(ax, "Mean Area", "Mean cell area per image", "Area (px²)", colour)
            elif key == "scatter":
                bars(ax, "Mean Mean Gray Value", "Mean intensity per image", "Mean gray value", colour)
            elif {"Mean Area", "Mean Mean Gray Value"} <= set(df.columns):
                ax.scatter(df["Mean Area"], df["Mean Mean Gray Value"], color=colour, s=40)
                for nm, x, y in zip(names, df["Mean Area"], df["Mean Mean Gray Value"]):
                    ax.annotate(nm[-12:], (x, y), fontsize=6, color=sub, xytext=(3, 3), textcoords="offset points")
                ax.set_title("Mean area vs. mean intensity", color=C["text"], fontsize=12, pad=10)
                ax.set_xlabel("Mean area (px²)", color=sub)
                ax.set_ylabel("Mean gray value", color=sub)
            else:
                missing(ax, "'Mean Area' / 'Mean Mean Gray Value'")
        else:
            if key == "area":
                if "Area" in df.columns:
                    ax.hist(df["Area"].dropna(), bins=30, color=colour, alpha=0.9)
                    ax.set_title("Cell size distribution", color=C["text"], fontsize=12, pad=10)
                    ax.set_xlabel("Area (px²)", color=sub)
                    ax.set_ylabel("Cells", color=sub)
                else:
                    missing(ax, "'Area'")
            elif key == "intensity":
                if "Mean Gray Value" in df.columns:
                    ax.hist(df["Mean Gray Value"].dropna(), bins=30, color=colour, alpha=0.9)
                    ax.set_title("Intensity distribution", color=C["text"], fontsize=12, pad=10)
                    ax.set_xlabel("Mean gray value", color=sub)
                    ax.set_ylabel("Cells", color=sub)
                else:
                    missing(ax, "'Mean Gray Value'")
            elif key == "scatter":
                if {"Area", "Integrated Density"} <= set(df.columns):
                    d = df[["Area", "Integrated Density"]].dropna()
                    ax.scatter(d["Area"], d["Integrated Density"], color=colour, alpha=0.6, s=14, edgecolors="none")
                    if len(d) > 1:
                        m, b = np.polyfit(d["Area"], d["Integrated Density"], 1)
                        xr = np.linspace(d["Area"].min(), d["Area"].max(), 100)
                        ax.plot(xr, m * xr + b, color=C["success"], linestyle="--", linewidth=1.5,
                                label="Linear fit")
                        ax.legend(facecolor=C["bg_input"], edgecolor="none", labelcolor=sub)
                    ax.set_title("Size vs. intensity", color=C["text"], fontsize=12, pad=10)
                    ax.set_xlabel("Area (px²)", color=sub)
                    ax.set_ylabel("Integrated density", color=sub)
                else:
                    missing(ax, "'Area' / 'Integrated Density'")
            else:
                circ, sol = _parse_kv(df.get("Shape descriptors", []), "circularity", "solidity") \
                    if "Shape descriptors" in df.columns else ([], [])
                if circ:
                    ax.scatter(circ, sol, color=colour, alpha=0.6, s=14, edgecolors="none")
                    ax.set_title("Shape", color=C["text"], fontsize=12, pad=10)
                    ax.set_xlabel("Circularity (1 = circle)", color=sub)
                    ax.set_ylabel("Solidity (1 = smooth outline)", color=sub)
                else:
                    missing(ax, "'Shape descriptors'")
        ax.tick_params(colors=sub, labelsize=8)
        ax.grid(True, color=grid, linestyle=":", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid)

    for ax in axes[len(plots):]:
        fig.delaxes(ax)
    return fig


class _PlotWorker(QThread):
    done = pyqtSignal(object, bytes)       # (df, plots) to rebuild for saving, PNG bytes for the preview
    error = pyqtSignal(str, str)

    def __init__(self, path: str, plots: list[str]):
        super().__init__()
        self.path, self.plots = path, plots

    def run(self):
        import traceback
        import pandas as pd
        try:
            df = pd.read_csv(self.path) if self.path.lower().endswith(".csv") else pd.read_excel(self.path)
            if df.empty:
                raise ValueError("The selected file contains no data.")
            fig = build_figure(df, self.plots)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
            self.done.emit((df, self.plots), buf.getvalue())
        except Exception as e:
            self.error.emit(str(e), traceback.format_exc())


class _LoadWorker(QThread):
    """Reads every per-image sheet of a results folder (can take a few seconds)."""
    done = pyqtSignal(object)
    error = pyqtSignal(str, str)

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder

    def run(self):
        import traceback
        from s2l.core.results import load_results_folder
        try:
            self.done.emit(load_results_folder(self.folder))
        except Exception as e:
            self.error.emit(str(e), traceback.format_exc())


def build_group_figure(df, metric: str, by: str, kind: str = "box", points: bool = True, unit: str = "cells",
                       C=None):
    """Box / violin plot of *metric* per group, styled for the current theme (or palette *C*)."""
    from matplotlib.figure import Figure
    C = C or COLORS
    d = df.dropna(subset=[metric])
    if unit == "images":
        d = d.groupby(list(dict.fromkeys([by, "image"])))[metric].mean().reset_index()
    names = sorted(d[by].astype(str).unique(), key=lambda s: (len(s), s))
    data = [d.loc[d[by].astype(str) == n, metric].to_numpy() for n in names]
    sub, grid = C["text_secondary"], C["plot_grid"]
    fig = Figure(figsize=(max(6.0, 1.1 * len(names) + 2), 5.2), facecolor=C["bg_card"], dpi=110,
                 layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_facecolor(C["bg_input"])
    pos = np.arange(1, len(names) + 1)
    if kind == "violin" and all(len(v) > 1 for v in data):
        parts = ax.violinplot(data, positions=pos, showmedians=True, widths=0.8)
        for i, b in enumerate(parts["bodies"]):
            b.set_facecolor(_SERIES[i % len(_SERIES)])
            b.set_alpha(0.55)
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            if k in parts:
                parts[k].set_color(C["text"])
    else:
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=not points)
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(_SERIES[i % len(_SERIES)])
            b.set_alpha(0.55)
        for k in ("medians", "whiskers", "caps"):
            for line in bp[k]:
                line.set_color(C["text"])
    if points:
        rng = np.random.default_rng(0)
        for i, v in enumerate(data):
            show = v if len(v) <= 3000 else rng.choice(v, 3000, replace=False)
            ax.scatter(pos[i] + rng.uniform(-0.18, 0.18, len(show)), show, s=5 if unit == "cells" else 22,
                       color=C["text"], alpha=0.25 if unit == "cells" else 0.8, edgecolors="none")
    ax.set_xticks(pos)
    labels = [f"{_short(n)}\nn={len(v)}" for n, v in zip(names, data)]
    if max(len(n) for n in names) > 10:     # long names would overlap side by side
        ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(metric + (" (image mean)" if unit == "images" else ""), color=sub)
    ax.set_title(f"{metric} by {by}", color=C["text"], fontsize=12, pad=10)
    ax.tick_params(colors=sub, labelsize=8)
    ax.grid(True, axis="y", color=grid, linestyle=":", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(grid)
    return fig


def _fig_to_pixmap(fig, max_w: int, max_h: int) -> QPixmap:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    pix = QPixmap.fromImage(QImage.fromData(buf.getvalue(), "PNG"))
    return pix.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class PlottingPage(QWidget):
    open_in_viewer = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._fig = None
        self._group_fig = None
        self._worker: _PlotWorker | None = None
        self._loader: _LoadWorker | None = None
        self._df = None            # per-cell table of the loaded results folder
        page, col = page_shell("Plotting", "Visualize, compare and explore the measurements of your cells.")

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_single_tab(), "Distributions (one sheet)")
        self.tabs.addTab(self._build_compare_tab(), "Compare groups")
        self.tabs.addTab(self._build_explore_tab(), "Explore cells")
        col.addWidget(self.tabs)
        col.addStretch()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(page))
        remember_text(self.file_edit, "plot/file")
        self._detect_kind()

    # ══ tab 1: distributions of one sheet ═════════════════════════════════

    def _build_single_tab(self):
        tab = QWidget()
        col = QVBoxLayout(tab)
        col.setContentsMargins(0, 14, 0, 0)
        col.setSpacing(16)
        data_lay = QVBoxLayout()
        data_lay.addLayout(section_header(
            "Data", "A per-image sheet (<image>_cp_masks.xlsx) for distributions, or "
                    "SummarySheet/Summary.xlsx to compare images."))
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Results .xlsx or .csv…")
        self.file_edit.editingFinished.connect(self._detect_kind)
        data_lay.addLayout(field_row("Results file", browse_row(self.file_edit, "Browse", self._pick_file)))
        self.kind_lbl = QLabel("")
        self.kind_lbl.setObjectName("Hint")
        data_lay.addWidget(self.kind_lbl)
        col.addWidget(card(data_lay))

        plots_lay = QVBoxLayout()
        plots_lay.addLayout(section_header("Plots", "Selected plots are drawn side by side in one figure."))
        self.checks = {}
        grid = QGridLayout()
        for i, (key, (label, _)) in enumerate(PLOTS.items()):
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.checks[key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        plots_lay.addLayout(grid)
        btns = QHBoxLayout()
        self.plot_btn = variant(QPushButton("Generate plots"), "primary")
        self.plot_btn.clicked.connect(self._generate)
        self.save_btn = QPushButton("Save plot…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(lambda: self._save_fig(self._fig, self.file_edit.text(), "_plots"))
        btns.addWidget(self.plot_btn)
        btns.addWidget(self.save_btn)
        btns.addStretch()
        plots_lay.addLayout(btns)
        self.status = QLabel("")
        self.status.setObjectName("StatusPending")
        plots_lay.addWidget(self.status)
        col.addWidget(card(plots_lay))

        prev = QVBoxLayout()
        prev.addLayout(section_header("Preview"))
        self.preview_label = QLabel("No plot generated yet.")
        self.preview_label.setObjectName("ImagePanel")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(380)
        prev.addWidget(self.preview_label)
        col.addWidget(card(prev))
        col.addStretch()   # keep cards at their natural height (tabs share the tallest height)
        return tab

    def _pick_file(self):
        start = os.path.dirname(self.file_edit.text()) if self.file_edit.text() else ""
        p, _ = QFileDialog.getOpenFileName(self, "Select results file", start,
                                           "Spreadsheet (*.xlsx *.xls *.csv);;All (*)")
        if p:
            self.file_edit.setText(p)
            self._detect_kind()

    def _detect_kind(self):
        """Relabel the plot choices for a summary sheet vs a per-image sheet."""
        path = self.file_edit.text().strip()
        if not os.path.isfile(path):
            self.kind_lbl.setText("")
            return
        try:
            import pandas as pd
            head = pd.read_csv(path, nrows=1) if path.lower().endswith(".csv") else pd.read_excel(path, nrows=1)
            kind = sheet_kind(head)
        except Exception:
            return
        for key, cb in self.checks.items():
            cb.setText(PLOTS[key][1 if kind == "summary" else 0])
        self.kind_lbl.setText("Summary sheet — plots compare images." if kind == "summary"
                              else "Per-image sheet — plots show the cells of one image.")

    def _generate(self):
        path = self.file_edit.text().strip()
        if not os.path.isfile(path):
            QMessageBox.warning(self, "No file", "Select a valid Excel or CSV results file first.")
            return
        plots = [k for k, cb in self.checks.items() if cb.isChecked()]
        if not plots:
            QMessageBox.warning(self, "No plots selected", "Select at least one plot.")
            return
        self._detect_kind()
        self.plot_btn.setEnabled(False)
        set_status(self.status, "Drawing…", "Pending")
        self._worker = _PlotWorker(path, plots)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, args, png: bytes):
        df, plots = args
        self._fig = lambda: build_figure(df, plots, PRINT_COLORS)
        pix = QPixmap.fromImage(QImage.fromData(png, "PNG"))
        w = max(400, self.preview_label.width() - 20)
        self.preview_label.setPixmap(pix.scaled(w, 560, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        self.plot_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        set_status(self.status, "✓ Plot ready — use “Save plot…” to export it.", "Success")

    def _on_error(self, msg: str, details: str):
        self.plot_btn.setEnabled(True)
        set_status(self.status, "Plotting failed.", "Error")
        show_error(self, "Plotting failed", msg, details)

    def _save_fig(self, make_fig, ref_path: str, suffix: str):
        """*make_fig* builds the figure in the white print style (saved plots never use the dark theme)."""
        if make_fig is None:
            return
        ref = Path(ref_path) if ref_path else Path.home() / "plot"
        suggested = str(ref.with_name(ref.stem + suffix + ".png")) if ref.suffix else str(ref / f"plot{suffix}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", suggested,
                                              "PNG image (*.png);;SVG vector (*.svg);;PDF (*.pdf)")
        if not path:
            return
        try:
            make_fig().savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
            QMessageBox.information(self, "Saved", f"Plot saved to:\n{path}")
        except Exception as e:
            show_error(self, "Save failed", "The plot could not be saved.", str(e))

    # ══ shared: results folder (tabs 2 and 3) ═════════════════════════════

    def _results_card(self, tab_key: str):
        lay = QVBoxLayout()
        lay.addLayout(section_header(
            "Results folder", "An output folder of a segmentation run (all <image>_cp_masks.xlsx sheets)."))
        edit = QLineEdit()
        edit.setPlaceholderText("Output folder with the Excel sheets…")
        row = browse_row(edit, "Browse", lambda: self._pick_results(edit))
        load = variant(QPushButton("Load"), "primary")
        load.clicked.connect(lambda: self._load_results(edit.text().strip()))
        row.addWidget(load)
        lay.addLayout(field_row("Folder", row))
        info = QLabel("")
        info.setObjectName("Hint")
        info.setWordWrap(True)
        lay.addWidget(info)
        setattr(self, f"_res_edit_{tab_key}", edit)
        setattr(self, f"_res_info_{tab_key}", info)
        remember_text(edit, "plot/results")
        return card(lay)

    def _pick_results(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Select results folder", edit.text())
        if d:
            edit.setText(d)
            self._load_results(d)

    def _load_results(self, folder: str):
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Choose an existing results folder.")
            return
        for k in ("compare", "explore"):
            getattr(self, f"_res_edit_{k}").setText(folder)
            getattr(self, f"_res_info_{k}").setText("Loading…")
        old = self._loader
        if old is not None and old.isRunning():
            self._old_loaders = [t for t in getattr(self, "_old_loaders", []) if t.isRunning()] + [old]
        ld = self._loader = _LoadWorker(folder)
        # only the newest load may change the data (an older, slower one would overwrite it)
        ld.done.connect(lambda df: self._on_loaded(folder, df) if ld is self._loader else None)
        ld.error.connect(lambda m, d: self._on_load_error(m, d) if ld is self._loader else None)
        ld.start()

    def _on_load_error(self, msg: str, details: str):
        for k in ("compare", "explore"):
            getattr(self, f"_res_info_{k}").setText("Loading failed.")
        show_error(self, "Cannot load", msg, details)

    def _on_loaded(self, folder: str, df):
        from s2l.core.results import input_folder_from_settings
        self._df = df
        if df is None or df.empty:
            msg = "No *_cp_masks.xlsx sheets in this folder — run the ROI analysis first."
        else:
            wells = sorted({w for w in df["well"] if w})
            msg = (f"{len(df):,} cells from {df['image'].nunique()} images"
                   + (f"; wells: {', '.join(wells)}" if wells else "") + ".")
        for k in ("compare", "explore"):
            getattr(self, f"_res_info_{k}").setText(msg)
        if df is not None and not df.empty:
            metrics = [m for m in NUMERIC_METRICS if m in df.columns]
            for combo in (self.metric_combo, self.x_combo, self.y_combo):
                keep = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                if combo is self.y_combo:
                    combo.addItem("(histogram of X)")
                combo.addItems(metrics)
                idx = combo.findText(keep)
                combo.setCurrentIndex(idx if idx >= 0 else (min(2, combo.count() - 1) if combo is self.y_combo else 0))
                combo.blockSignals(False)
            if self.x_combo.currentText() == "Area" and self.y_combo.currentText().startswith("(histogram"):
                self.y_combo.setCurrentText("Mean Gray Value")
            guess = input_folder_from_settings(folder)
            if guess and not self.images_edit.text().strip():
                self.images_edit.setText(guess)

    def _df_with_conditions(self):
        from s2l.core.results import add_condition, parse_condition_map
        return add_condition(self._df, parse_condition_map(self.cond_edit.text()))

    # ══ tab 2: compare groups ══════════════════════════════════════════════

    def _build_compare_tab(self):
        from s2l.core.results import GROUP_KEYS
        tab = QWidget()
        col = QVBoxLayout(tab)
        col.setContentsMargins(0, 14, 0, 0)
        col.setSpacing(16)
        col.addWidget(self._results_card("compare"))

        opt = QVBoxLayout()
        opt.addLayout(section_header("Comparison"))
        r1 = QHBoxLayout()
        r1.setSpacing(12)
        self.metric_combo = QComboBox()
        self.group_combo = QComboBox()
        for g in GROUP_KEYS:
            self.group_combo.addItem(g)
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["Box plot", "Violin plot"])
        self.unit_combo = QComboBox()
        self.unit_combo.addItem("Cells", "cells")
        self.unit_combo.addItem("Image means", "images")
        r1.addLayout(field_row("Measurement", self.metric_combo), 2)
        r1.addLayout(field_row("Group by", self.group_combo), 1)
        r1.addLayout(field_row("Plot", self.kind_combo), 1)
        r1.addLayout(field_row("Statistical unit", self.unit_combo), 1)
        opt.addLayout(r1)
        self.cond_edit = QLineEdit()
        self.cond_edit.setPlaceholderText("e.g.  A6,A7=WT; A8-A10=mutant")
        opt.addLayout(field_row(
            "Conditions (for “Group by: condition”)", self.cond_edit,
            "Name groups of wells; wells not listed become “(unassigned)”."))
        remember_text(self.cond_edit, "plot/conditions")
        self.points_chk = QCheckBox("Show individual points")
        self.points_chk.setChecked(True)
        opt.addWidget(self.points_chk)
        btns = QHBoxLayout()
        go = variant(QPushButton("Compare"), "primary")
        go.clicked.connect(self._compare)
        self.group_save_btn = QPushButton("Save plot…")
        self.group_save_btn.setEnabled(False)
        self.group_save_btn.clicked.connect(
            lambda: self._save_fig(self._group_fig, self._res_edit_compare.text(), "_compare"))
        self.csv_btn = QPushButton("Export table (CSV)…")
        self.csv_btn.setEnabled(False)
        self.csv_btn.clicked.connect(self._export_csv)
        for b in (go, self.group_save_btn, self.csv_btn):
            btns.addWidget(b)
        btns.addStretch()
        opt.addLayout(btns)
        col.addWidget(card(opt))

        res = QVBoxLayout()
        res.addLayout(section_header("Result"))
        self.stats_lbl = QLabel("Load a results folder and press Compare.")
        self.stats_lbl.setObjectName("RunSummary")
        self.stats_lbl.setWordWrap(True)
        self.stats_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        res.addWidget(self.stats_lbl)
        self.group_table = QTableWidget(0, 0)
        self.group_table.setMinimumHeight(160)
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        res.addWidget(self.group_table)
        self.group_preview = QLabel("")
        self.group_preview.setObjectName("ImagePanel")
        self.group_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.group_preview.setMinimumHeight(420)
        res.addWidget(self.group_preview)
        col.addWidget(card(res))
        col.addStretch()   # keep cards at their natural height (tabs share the tallest height)
        return tab

    def _compare(self):
        from s2l.core.results import compare, group_table
        if self._df is None or self._df.empty:
            QMessageBox.information(self, "No data", "Load a results folder first.")
            return
        df = self._df_with_conditions()
        metric, by = self.metric_combo.currentText(), self.group_combo.currentText()
        unit = self.unit_combo.currentData()
        if df[by].astype(str).nunique() < 2:
            QMessageBox.information(self, "One group only",
                                    f"Every cell has the same “{by}”. Choose another grouping"
                                    + (" or fill in Conditions." if by == "condition" else "."))
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._table = group_table(df, metric, by)
            lines = compare(df, metric, by, unit)
            args = (df, metric, by, "violin" if self.kind_combo.currentIndex() else "box",
                    self.points_chk.isChecked(), unit)
            shown = build_group_figure(*args)
            self._group_fig = lambda: build_group_figure(*args, C=PRINT_COLORS)
        finally:
            QApplication.restoreOverrideCursor()
        self.stats_lbl.setText("\n".join(lines))
        t = self._table
        self.group_table.setColumnCount(len(t.columns))
        self.group_table.setRowCount(len(t))
        self.group_table.setHorizontalHeaderLabels([str(c) for c in t.columns])
        for r in range(len(t)):
            for c, v in enumerate(t.iloc[r]):
                txt = f"{v:.4g}" if isinstance(v, float) else str(v)
                self.group_table.setItem(r, c, QTableWidgetItem(txt))
        self.group_table.resizeColumnsToContents()
        self.group_preview.setPixmap(_fig_to_pixmap(shown,max(500, self.group_preview.width() - 20), 560))
        self.group_save_btn.setEnabled(True)
        self.csv_btn.setEnabled(True)

    def _export_csv(self):
        folder = self._res_edit_compare.text().strip() or str(Path.home())
        name = f"{self.metric_combo.currentText()}_by_{self.group_combo.currentText()}.csv".replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(self, "Export table", str(Path(folder) / name), "CSV (*.csv)")
        if path:
            self._table.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"Table saved to:\n{path}")

    # ══ tab 3: explore cells (interactive) ═════════════════════════════════

    def _build_explore_tab(self):
        from s2l.core.results import GROUP_KEYS
        tab = QWidget()
        col = QVBoxLayout(tab)
        col.setContentsMargins(0, 14, 0, 0)
        col.setSpacing(16)
        col.addWidget(self._results_card("explore"))

        opt = QVBoxLayout()
        opt.addLayout(section_header(
            "Plot", "Hover a point to see which cell it is; click it to open that cell in the correction editor."))
        r1 = QHBoxLayout()
        r1.setSpacing(12)
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.color_combo = QComboBox()
        self.color_combo.addItem("(none)")
        for g in GROUP_KEYS[:-1]:
            self.color_combo.addItem(g)
        r1.addLayout(field_row("X", self.x_combo), 2)
        r1.addLayout(field_row("Y", self.y_combo), 2)
        r1.addLayout(field_row("Colour by", self.color_combo), 1)
        opt.addLayout(r1)
        r2 = QHBoxLayout()
        self.logx_chk = QCheckBox("Log X")
        self.logy_chk = QCheckBox("Log Y")
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 300)
        self.bins_spin.setValue(40)
        self.bins_spin.setPrefix("Bins: ")
        draw = variant(QPushButton("Draw"), "primary")
        draw.clicked.connect(self._draw_explore)
        for wdg in (self.logx_chk, self.logy_chk, self.bins_spin):
            r2.addWidget(wdg)
        r2.addStretch()
        r2.addWidget(draw)
        opt.addLayout(r2)
        self.images_edit = QLineEdit()
        self.images_edit.setPlaceholderText("Folder with the original images (filled in automatically when known)")
        opt.addLayout(field_row(
            "Original images folder (to open cells)",
            browse_row(self.images_edit, "Browse",
                       lambda: self.images_edit.setText(
                           QFileDialog.getExistingDirectory(self, "Original images folder") or self.images_edit.text()))))
        col.addWidget(card(opt))

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        page = self

        class _Toolbar(NavigationToolbar2QT):
            def save_figure(self, *args):      # save in the white print style
                page._save_explore()

        self._xfig = Figure(figsize=(8, 5.5), dpi=100, layout="constrained")
        self._xcanvas = FigureCanvasQTAgg(self._xfig)
        self._xcanvas.setMinimumHeight(520)
        self._xtoolbar = _Toolbar(self._xcanvas, tab)
        plot_lay = QVBoxLayout()
        plot_lay.addWidget(self._xtoolbar)
        plot_lay.addWidget(self._xcanvas)
        self.explore_status = QLabel("")
        self.explore_status.setObjectName("Hint")
        plot_lay.addWidget(self.explore_status)
        col.addWidget(card(plot_lay))
        self._xcanvas.mpl_connect("motion_notify_event", self._on_hover)
        self._xcanvas.mpl_connect("pick_event", self._on_pick)
        self._annot = None
        self._scatter_rows = []
        col.addStretch()   # keep cards at their natural height (tabs share the tallest height)
        return tab

    def _save_explore(self):
        if not self._xfig.axes:
            QMessageBox.information(self, "Nothing to save", "Draw a plot first.")
            return
        self._draw_explore(PRINT_COLORS)            # restyle white, save, then back to the app theme
        try:
            self._save_fig(lambda: self._xfig, self._res_edit_explore.text(), "_explore")
        finally:
            self._draw_explore()

    def _draw_explore(self, C=None):
        if self._df is None or self._df.empty:
            QMessageBox.information(self, "No data", "Load a results folder first.")
            return
        C = C or COLORS
        df = self._df_with_conditions()
        xcol, ycol = self.x_combo.currentText(), self.y_combo.currentText()
        hist = ycol.startswith("(histogram")
        color_by = self.color_combo.currentText()
        fig = self._xfig
        fig.clear()
        fig.set_facecolor(C["bg_card"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(C["bg_input"])
        sub, grid = C["text_secondary"], C["plot_grid"]
        groups = [("all", df)] if color_by == "(none)" else list(df.groupby(color_by))
        self._scatter_rows = []
        for i, (g, d) in enumerate(groups):
            c = _SERIES[i % len(_SERIES)]
            if hist:
                v = d[xcol].dropna()
                if self.logx_chk.isChecked():
                    v = v[v > 0]
                    bins = np.logspace(np.log10(v.min()), np.log10(v.max()), self.bins_spin.value()) if len(v) else 10
                else:
                    bins = self.bins_spin.value()
                ax.hist(v, bins=bins, color=c, alpha=0.6, label=str(g))
            else:
                d = d.dropna(subset=[xcol, ycol])
                sc = ax.scatter(d[xcol], d[ycol], s=10, color=c, alpha=0.6, edgecolors="none",
                                label=str(g), picker=4)
                self._scatter_rows.append((sc, d))
        ax.set_xlabel(xcol, color=sub)
        ax.set_ylabel("Cells" if hist else ycol, color=sub)
        if self.logx_chk.isChecked():
            ax.set_xscale("log")
        if self.logy_chk.isChecked():
            ax.set_yscale("log")
        if color_by != "(none)":
            ax.legend(facecolor=C["bg_input"], edgecolor="none", labelcolor=sub, fontsize=8,
                      markerscale=2, title=color_by, title_fontsize=8)
        ax.tick_params(colors=sub, labelsize=8)
        ax.grid(True, color=grid, linestyle=":", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid)
        self._annot = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                                  bbox=dict(boxstyle="round", fc=C["bg_elevated"], ec=C["accent"]),
                                  color=C["text"], fontsize=8)
        self._annot.set_visible(False)
        self._xcanvas.draw_idle()
        n = sum(len(d) for _, d in self._scatter_rows) if not hist else int(df[xcol].notna().sum())
        self.explore_status.setText(f"{n:,} cells plotted." + ("" if hist else " Click a point to open that cell."))

    def _hit(self, event):
        for sc, d in self._scatter_rows:
            ok, info = sc.contains(event)
            if ok and len(info.get("ind", [])):
                return d.iloc[int(info["ind"][0])]
        return None

    def _on_hover(self, event):
        if self._annot is None or event.inaxes is None or not self._scatter_rows:
            return
        row = self._hit(event)
        if row is None:
            if self._annot.get_visible():
                self._annot.set_visible(False)
                self._xcanvas.draw_idle()
            return
        self._annot.xy = (event.xdata, event.ydata)
        where = " ".join(str(row[k]) for k in ("well", "stage", "channel", "type") if row.get(k)) \
            or str(row["image"])[-40:]
        self._annot.set_text(f"{where}\ncell #{int(row['Label'])}"
                             f"\n{self.x_combo.currentText()}: {row[self.x_combo.currentText()]:.4g}")
        self._annot.set_visible(True)
        self._xcanvas.draw_idle()

    def _on_pick(self, event):
        for sc, d in self._scatter_rows:
            if event.artist is sc and len(event.ind):
                self.open_cell(d.iloc[int(event.ind[0])])
                return

    def open_cell(self, row):
        """Open the correction editor on the image of *row*, centred on that cell."""
        from s2l.core.image_files import list_images
        from s2l.ui.manual_correction import ManualCorrectionDialog
        folder = self.images_edit.text().strip()
        if not os.path.isdir(folder):
            QMessageBox.information(self, "Images folder needed",
                                    "Choose the folder with the original images to open cells.")
            return
        originals = {Path(p).stem: p for p in list_images(folder)}
        img = originals.get(row["image"])
        if not img or not os.path.exists(row["mask_path"]):
            QMessageBox.information(self, "Not found", f"Could not find the image or mask for {row['image']}.")
            return
        try:
            dlg = ManualCorrectionDialog(img, row["mask_path"], self)
        except Exception as e:
            show_error(self, "Cannot open", "Could not open this image.", str(e))
            return
        QTimer.singleShot(150, lambda: dlg.focus_cell(int(row["Label"])))
        dlg.exec()
