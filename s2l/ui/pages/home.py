"""Home page: the first window — what you can do, system status, recent folders."""
import os
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from s2l.ui.widgets import card, page_shell, recent_folders, scrollable, section_header, set_status

# (page key, icon, title, description)
ACTIONS = [
    ("segmentation", "🔬", "Segment images",
     "Find cells in a folder or an experiment spreadsheet and measure them (Excel + summary)."),
    ("correction", "🖊", "Correct masks",
     "Fix segmentation by hand: add missed cells, erase wrong ones, check channels."),
    ("training", "🧠", "Train a model",
     "Fine-tune Cellpose (GPU) or train StarDist on your own labelled images."),
    ("viewer", "🖼", "View a dataset",
     "Browse a folder of images and masks with thumbnails and a zoomable preview."),
    ("plotting", "📊", "Plot results",
     "Distributions and comparisons from per-image sheets or the summary sheet."),
    ("tools", "🛠", "Tools",
     "Image enhancer (denoise, contrast, background), Cellpose GUI, Cell-ACDC, checks."),
]


class _ActionCard(QPushButton):
    """A large clickable card with icon, title and description."""

    def __init__(self, icon: str, title: str, desc: str):
        super().__init__()
        self.setObjectName("HomeCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(14)
        ic = QLabel(icon)
        ic.setObjectName("HomeCardIcon")
        ic.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(4)
        t = QLabel(title)
        t.setObjectName("HomeCardTitle")
        d = QLabel(desc)
        d.setObjectName("HomeCardDesc")
        d.setWordWrap(True)
        for w in (t, d):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            text.addWidget(w)
        text.addStretch()
        lay.addLayout(text, 1)
        self.setMinimumHeight(104)


class HomePage(QWidget):
    navigate = pyqtSignal(str)             # page key
    open_recent = pyqtSignal(str)          # folder to segment

    def __init__(self):
        super().__init__()
        page, col = page_shell("Welcome to S2L", "Cell segmentation and analysis. What would you like to do?")

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        for i, (key, icon, title, desc) in enumerate(ACTIONS):
            c = _ActionCard(icon, title, desc)
            c.clicked.connect(lambda _, k=key: self.navigate.emit(k))
            grid.addWidget(c, i // 2, i % 2)
        col.addLayout(grid)

        bottom = QHBoxLayout()
        bottom.setSpacing(16)

        st = QVBoxLayout()
        st.addLayout(section_header("System"))
        self._status = {}
        for key, label in [("gpu", "GPU"), ("cellpose", "Cellpose"), ("stardist", "StarDist")]:
            row = QHBoxLayout()
            name = QLabel(label)
            name.setObjectName("FieldLabel")
            name.setFixedWidth(80)
            val = QLabel("checking…")
            val.setObjectName("StatusPending")
            val.setWordWrap(True)
            row.addWidget(name)
            row.addWidget(val, 1)
            st.addLayout(row)
            self._status[key] = val
        st.addStretch()
        bottom.addWidget(card(st), 1)

        rec = QVBoxLayout()
        rec.addLayout(section_header("Recent input folders", "Click one to segment it again."))
        self._recent_lay = QVBoxLayout()
        self._recent_lay.setSpacing(2)
        rec.addLayout(self._recent_lay)
        rec.addStretch()
        bottom.addWidget(card(rec), 1)
        col.addLayout(bottom)
        col.addStretch()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scrollable(page))
        self.refresh_recent()

    def refresh_recent(self):
        while self._recent_lay.count():
            item = self._recent_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        folders = recent_folders()
        if not folders:
            empty = QLabel("No folders yet — your last input folders will appear here.")
            empty.setObjectName("Hint")
            empty.setWordWrap(True)
            self._recent_lay.addWidget(empty)
        for f in folders:
            p = Path(f)
            # long paths would widen the page; show the last two folders, full path on hover
            short = f if len(f) <= 70 else f"…{os.sep}{p.parent.name}{os.sep}{p.name}"
            b = QPushButton(short)
            b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            b.setObjectName("RecentFolder")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(f)
            b.clicked.connect(lambda _, p=f: self.open_recent.emit(p))
            self._recent_lay.addWidget(b)

    def set_env(self, info: dict):
        if info.get("gpu"):
            set_status(self._status["gpu"], f"✓ {info.get('gpu_name') or 'CUDA GPU'}", "Success")
        else:
            set_status(self._status["gpu"], "CPU only (no CUDA GPU found) — segmentation will be slower",
                       "Warning")
        if info.get("cellpose_ok"):
            set_status(self._status["cellpose"], "✓ Cellpose-SAM ready", "Success")
        else:
            set_status(self._status["cellpose"], info.get("cellpose_msg") or "not available", "Error")
        if info.get("stardist_ok"):
            set_status(self._status["stardist"], "✓ ready (CPU on Windows)", "Success")
        else:
            set_status(self._status["stardist"], info.get("stardist_msg") or "not available", "Warning")
