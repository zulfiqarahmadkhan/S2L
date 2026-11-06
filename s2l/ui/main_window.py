"""Main window for S2L: sidebar navigation, page stack, theme switching."""
import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QByteArray, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressDialog,
    QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from s2l import __version__
from s2l.ui import theme
from s2l.ui.dataset_viewer import DatasetViewerPage
from s2l.ui.pages.correction import CorrectionPage
from s2l.ui.pages.home import HomePage
from s2l.ui.pages.plotting import PlottingPage
from s2l.ui.pages.segmentation import SegmentationPage
from s2l.ui.pages.tools import ToolsPage
from s2l.ui.pages.training import TrainingPage
from s2l.core.device import prefer_cpu, set_prefer_cpu
from s2l.ui.widgets import set_status, settings, variant
from s2l.ui.workers import EnvProbe, TrainingWorker, WorkerThread  # noqa: F401  (re-exported)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

APP_ICON = Path(__file__).resolve().parents[2] / "app_icon.ico"

# (key, label, icon) in workflow order
NAV_ITEMS = [
    ("home",         "Home",         "🏠"),
    ("segmentation", "Segmentation", "🔬"),
    ("correction",   "Correction",   "🖊"),
    ("training",     "Training",     "🧠"),
    ("viewer",       "Viewer",       "🖼"),
    ("plotting",     "Plotting",     "📊"),
    ("tools",        "Tools",        "🛠"),
]
PAGE_KEYS = [k for k, _, _ in NAV_ITEMS]


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ═══════════════════════════════════════════════════════════════════════════

class _SideBar(QWidget):
    """Vertical navigation rail with a status block and the day/night toggle."""
    page_changed = pyqtSignal(int)
    theme_toggled = pyqtSignal()

    NAV_ITEMS = NAV_ITEMS

    def __init__(self):
        super().__init__()
        self.setObjectName("SideBar")
        self.setFixedWidth(200)
        self._buttons: list[QPushButton] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 20, 12, 16)
        lay.setSpacing(4)

        brand = QLabel("S2L")
        brand.setObjectName("BrandLabel")
        edition = QLabel("Cell Segmentation & Analysis")
        edition.setObjectName("BrandSub")
        lay.addWidget(brand)
        lay.addWidget(edition)
        lay.addSpacing(24)

        for idx, (_, label, icon) in enumerate(NAV_ITEMS):
            btn = QPushButton(f"  {icon}   {label}")
            btn.setObjectName("NavBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(38)
            btn.clicked.connect(lambda _, i=idx: self._select(i))
            lay.addWidget(btn)
            self._buttons.append(btn)

        lay.addStretch()

        # status chips, filled in by the background environment probe
        self._chips = {}
        for key, text in [("gpu", "GPU"), ("cellpose", "Cellpose"), ("stardist", "StarDist")]:
            chip = QLabel(f"○ {text}: checking…")
            chip.setObjectName("StatusPending")
            chip.setWordWrap(True)
            lay.addWidget(chip)
            self._chips[key] = chip
            if key == "gpu":
                lay.addWidget(self._build_device_switch())
        lay.addSpacing(10)

        self.theme_btn = QPushButton()
        self.theme_btn.setObjectName("ThemeToggle")
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.clicked.connect(self.theme_toggled.emit)
        lay.addWidget(self.theme_btn)
        lay.addSpacing(6)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("VersionLabel")
        lay.addWidget(ver)
        self.update_theme_button()

    def _build_device_switch(self) -> QWidget:
        """GPU | CPU buttons: GPU by default, CPU on request (e.g. when the GPU is busy)."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 2, 0, 4)
        row.setSpacing(4)
        self._dev_btns = {}
        for key, text, tip in [("gpu", "GPU", "Run Cellpose on the graphics card (fast)"),
                               ("cpu", "CPU", "Run on the processor — slower, but leaves the GPU free "
                                              "for other programs")]:
            b = variant(QPushButton(text), "small")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, k=key: self.set_device(k))
            row.addWidget(b)
            self._dev_btns[key] = b
        box.setVisible(False)              # shown once a GPU is found
        self._dev_box = box
        self._gpu_name = ""
        self.set_device(settings().value("compute/device", "gpu"), save=False)
        return box

    def set_device(self, key: str, save: bool = True):
        key = "cpu" if key == "cpu" else "gpu"
        set_prefer_cpu(key == "cpu")
        for k, b in self._dev_btns.items():
            b.setChecked(k == key)
        if save:
            settings().setValue("compute/device", key)
        if self._dev_box.isVisible():
            self._show_gpu_chip()

    def _show_gpu_chip(self):
        chip = self._chips["gpu"]
        if prefer_cpu():
            set_status(chip, "● Using CPU (GPU switched off)", "Warning")
        else:
            set_status(chip, "● GPU: on", "Success")
        chip.setToolTip(self._gpu_name)

    def update_theme_button(self):
        dark = theme.current_theme() == "dark"
        self.theme_btn.setText("☀  Day mode" if dark else "🌙  Night mode")
        self.theme_btn.setToolTip("Switch to the light theme" if dark else "Switch to the dark theme")

    def set_env(self, info: dict):
        if info.get("gpu"):
            self._gpu_name = info.get("gpu_name", "")
            self._dev_box.setVisible(True)
            self._show_gpu_chip()
        else:
            set_status(self._chips["gpu"], "● GPU: CPU only", "Warning")
        ok = info.get("cellpose_ok")
        set_status(self._chips["cellpose"], f"● Cellpose: {'ready' if ok else 'unavailable'}",
                   "Success" if ok else "Error")
        self._chips["cellpose"].setToolTip(info.get("cellpose_msg", ""))
        ok = info.get("stardist_ok")
        set_status(self._chips["stardist"], f"● StarDist: {'ready' if ok else 'unavailable'}",
                   "Success" if ok else "Warning")
        self._chips["stardist"].setToolTip(info.get("stardist_msg", ""))

    def _select(self, idx: int):
        for i, b in enumerate(self._buttons):
            b.setProperty("active", "true" if i == idx else "false")
            b.style().unpolish(b)
            b.style().polish(b)
        self.page_changed.emit(idx)


# ═══════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S2L")
        if APP_ICON.exists():
            self.setWindowIcon(QIcon(str(APP_ICON)))
        self.setMinimumSize(1000, 680)
        self.resize(1280, 820)
        self._closing = False
        app = QApplication.instance()
        if app is not None and not app.styleSheet():
            apply_theme(settings().value("ui/theme", "dark", type=str))

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self.sidebar = _SideBar()
        h.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.stack.setObjectName("PageStack")
        self._home_page = HomePage()
        self._seg_page = SegmentationPage()
        self._corr_page = CorrectionPage()
        self._train_page = TrainingPage()
        self._viewer_page = DatasetViewerPage()
        self._plot_page = PlottingPage()
        self._tools_page = ToolsPage()
        self.pages = {
            "home": self._home_page, "segmentation": self._seg_page, "correction": self._corr_page,
            "training": self._train_page, "viewer": self._viewer_page, "plotting": self._plot_page,
            "tools": self._tools_page,
        }
        for key in PAGE_KEYS:
            self.stack.addWidget(self.pages[key])
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.sidebar.page_changed.connect(self._on_page_changed)
        self.sidebar.theme_toggled.connect(self.toggle_theme)
        self._home_page.navigate.connect(self.go)
        self._home_page.open_recent.connect(self._segment_folder)
        for page in (self._seg_page, self._train_page):
            page.open_in_viewer.connect(self._open_in_viewer)
        self._seg_page.open_correction.connect(self._open_correction)
        self._train_page.use_model.connect(self._use_trained_model)

        self._restore_window()

        # heavy libraries load in the background so the window appears quickly
        self._probe = EnvProbe()
        self._probe.ready.connect(self._on_env)
        self._probe.start()

    # ── navigation ────────────────────────────────────────────────────────

    def go(self, key: str):
        self.sidebar._select(PAGE_KEYS.index(key))

    def _on_page_changed(self, idx: int):
        self.stack.setCurrentIndex(idx)
        if PAGE_KEYS[idx] == "home":
            self._home_page.refresh_recent()
        elif PAGE_KEYS[idx] in ("segmentation", "training"):
            # a model may have been trained or removed since the page was last shown
            self._seg_page.refresh_models()
            self._train_page.refresh_models()

    def _open_in_viewer(self, folder_path: str):
        self.go("viewer")
        self._viewer_page.open_folder(folder_path)

    def _open_correction(self, mask_dir: str, originals: list):
        self.go("correction")
        self._corr_page.set_source(mask_dir, originals)

    def _use_trained_model(self, engine: str, path: str):
        self.go("segmentation")
        self._seg_page.set_model(engine, path)

    def _segment_folder(self, folder: str):
        self.go("segmentation")
        self._seg_page.set_input_folder(folder)

    # ── environment / theme ───────────────────────────────────────────────

    def _on_env(self, info: dict):
        logger.info(f"Environment: {info}")
        for target in (self.sidebar, self._home_page, self._seg_page, self._train_page):
            target.set_env(info)

    def toggle_theme(self):
        apply_theme("light" if theme.current_theme() == "dark" else "dark")
        self.sidebar.update_theme_button()

    # ── window state ──────────────────────────────────────────────────────

    def _restore_window(self):
        s = settings()
        geo = s.value("window/geometry")
        if isinstance(geo, QByteArray) and not geo.isEmpty():
            self.restoreGeometry(geo)
        else:
            scr = QApplication.primaryScreen()
            if scr:
                g = self.frameGeometry()
                g.moveCenter(scr.availableGeometry().center())
                self.move(g.topLeft())
        self.go("home")   # always start on the Home page

    def _save_window(self):
        settings().setValue("window/geometry", self.saveGeometry())

    def _running_workers(self):
        return [w for w in (self._seg_page.worker_thread, self._train_page._worker, self._train_page._eval)
                if w is not None and w.isRunning()]

    def closeEvent(self, event):
        running = self._running_workers()
        if not running:
            self._save_window()
            event.accept()
            return
        if self._closing:
            event.ignore()
            return
        reply = QMessageBox.question(
            self, "Confirm exit",
            "Processing, training or model testing is still running. Stop it and exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        event.ignore()
        if reply != QMessageBox.StandardButton.Yes:
            return
        # stop cooperatively and wait without freezing the window
        self._closing = True
        for w in running:
            w.stop()
        dlg = QProgressDialog("Finishing the current image / batch, then closing…", None, 0, 0, self)
        dlg.setWindowTitle("Closing S2L")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.show()

        def _check():
            if self._running_workers():
                QTimer.singleShot(200, _check)
                return
            dlg.close()
            self._closing = False
            self.close()

        QTimer.singleShot(200, _check)


def apply_theme(name: str):
    """Switch palette, restyle every widget, and remember the choice."""
    theme.set_theme(name)
    app = QApplication.instance()
    if app is not None:
        app.setStyleSheet(theme.get_complete_stylesheet())
        for w in app.allWidgets():
            w.update()
    settings().setValue("ui/theme", name)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("S2L")
    app.setOrganizationName("S2L")
    if APP_ICON.exists():
        app.setWindowIcon(QIcon(str(APP_ICON)))
    apply_theme(settings().value("ui/theme", "dark", type=str))
    w = MainApp()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
