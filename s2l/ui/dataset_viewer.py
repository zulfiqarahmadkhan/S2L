"""
Dataset viewer — browse large image folders with lazy-loaded thumbnails
and a full-size preview panel.
"""
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QScrollArea, QFrame, QSizePolicy, QCheckBox,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QRectF,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QBrush, QFont, QWheelEvent,
    QKeyEvent, QMouseEvent,
)

import numpy as np

from s2l.ui.theme import COLORS
from s2l.ui.widgets import remember_text, variant

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}
THUMB_SIZE = 140
THUMB_PADDING = 6


# ═══════════════════════════════════════════════════════════════════════════
# Background thumbnail loader
# ═══════════════════════════════════════════════════════════════════════════

PREVIEW_MAX_SIDE = 2400


def render_file(path: Path, max_side: int) -> tuple[QImage, tuple]:
    """Load any image for display: 16-bit stretched, label masks coloured.

    Returns (QImage, (width, height, channels, dtype)). Safe to call off the UI
    thread because it only creates a QImage (QPixmap must stay on the UI thread).
    """
    import cv2
    from skimage.io import imread
    from s2l.ui.manual_correction import (
        load_image_normalised, ndarray_to_qimage, overlay_mask, render_view,
    )
    from s2l.core.image_files import MASK_SUFFIX

    raw = imread(str(path))
    h, w = raw.shape[:2]
    ch = raw.shape[2] if raw.ndim == 3 else 1
    meta = (w, h, ch, str(raw.dtype))
    if MASK_SUFFIX in path.stem and raw.ndim == 2 and raw.dtype.kind in "iu":
        m = raw.astype(np.int32)
        f = min(1.0, max_side / max(h, w))
        if f < 1:
            m = cv2.resize(m.astype(np.float32), (max(1, int(w * f)), max(1, int(h * f))),
                           interpolation=cv2.INTER_NEAREST).astype(np.int32)
        rgb = overlay_mask(np.zeros((*m.shape, 3), np.uint8), m, 1.0)
    else:
        rgb = render_view(load_image_normalised(path, max_side=max_side))
    return ndarray_to_qimage(rgb), meta


class _ThumbLoader(QThread):
    """Load one thumbnail at a time, emitting (index, QImage) pairs."""
    thumb_ready = pyqtSignal(int, QImage)
    finished_all = pyqtSignal()

    def __init__(self, paths: list[Path], size: int):
        super().__init__()
        self._paths = paths
        self._size = size
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        for idx, p in enumerate(self._paths):
            if self._cancel:
                return
            try:
                img, _ = render_file(p, self._size)
                self.thumb_ready.emit(idx, img)
            except Exception as e:
                logger.debug(f"thumbnail failed for {p}: {e}")
                continue
        self.finished_all.emit()


# ═══════════════════════════════════════════════════════════════════════════
# Single thumbnail tile (custom-painted for performance)
# ═══════════════════════════════════════════════════════════════════════════

class _ThumbTile(QWidget):
    """A single clickable thumbnail with filename label."""
    clicked = pyqtSignal(int)

    def __init__(self, index: int, filename: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.filename = filename
        self.pixmap: Optional[QPixmap] = None
        self.selected = False
        self.hovered = False

        total = THUMB_SIZE + THUMB_PADDING * 2
        self.setFixedSize(total, total + 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def set_pixmap(self, pm: QPixmap):
        self.pixmap = pm
        self.update()

    def set_selected(self, on: bool):
        self.selected = on
        self.update()

    def enterEvent(self, ev):
        self.hovered = True
        self.update()

    def leaveEvent(self, ev):
        self.hovered = False
        self.update()

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # background
        bg = QColor(COLORS["bg_card"])
        border = QColor(COLORS["accent"]) if self.selected else (
            QColor(COLORS["bg_elevated"]) if self.hovered else QColor(COLORS["border"])
        )
        p.setPen(QPen(border, 2 if self.selected else 1))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 8, 8)

        # thumbnail
        img_area = THUMB_SIZE
        ox = (w - img_area) / 2
        oy = THUMB_PADDING

        if self.pixmap:
            pw, ph = self.pixmap.width(), self.pixmap.height()
            tx = ox + (img_area - pw) / 2
            ty = oy + (img_area - ph) / 2
            p.drawPixmap(int(tx), int(ty), self.pixmap)
        else:
            # placeholder
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(COLORS["bg_input"])))
            p.drawRoundedRect(QRectF(ox, oy, img_area, img_area), 6, 6)
            p.setPen(QPen(QColor(COLORS["text_dim"])))
            p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRectF(ox, oy, img_area, img_area),
                       Qt.AlignmentFlag.AlignCenter, "Loading…")

        # filename
        p.setPen(QPen(QColor(COLORS["text_secondary"])))
        p.setFont(QFont("Segoe UI", 9))
        text_rect = QRectF(4, THUMB_SIZE + THUMB_PADDING + 2, w - 8, 20)
        elided = p.fontMetrics().elidedText(
            self.filename, Qt.TextElideMode.ElideMiddle, int(text_rect.width()))
        p.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, elided)

        p.end()


# ═══════════════════════════════════════════════════════════════════════════
# Thumbnail grid (flow layout inside a scroll area)
# ═══════════════════════════════════════════════════════════════════════════

class _FlowContainer(QWidget):
    """Widget that lays out children in a wrapping flow (like CSS flex-wrap)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[QWidget] = []
        self._spacing = 8

    def add_tile(self, tile: QWidget):
        tile.setParent(self)
        self._items.append(tile)
        self._relayout()
        tile.show()

    def add_tiles(self, tiles: list):
        for tile in tiles:
            tile.setParent(self)
            self._items.append(tile)
        self._relayout()
        for tile in tiles:
            tile.show()

    def clear_tiles(self):
        for w in self._items:
            w.setParent(None)
            w.deleteLater()
        self._items.clear()
        self.setMinimumHeight(0)
        self.updateGeometry()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._relayout()

    def _relayout(self):
        if not self._items:
            return
        avail_w = self.width() or 600
        x, y = 0, 0
        row_h = 0
        for w in self._items:
            iw, ih = w.width(), w.height()
            if x + iw > avail_w and x > 0:
                x = 0
                y += row_h + self._spacing
                row_h = 0
            w.move(x, y)
            row_h = max(row_h, ih)
            x += iw + self._spacing
        total_h = y + row_h + self._spacing
        self.setMinimumHeight(total_h)


# ═══════════════════════════════════════════════════════════════════════════
# Preview panel
# ═══════════════════════════════════════════════════════════════════════════

class _PreviewPanel(QWidget):
    """Large image preview with metadata, supports zoom via scroll wheel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._zoom = 1.0
        self._pan_offset = [0.0, 0.0]
        self._drag_start = None
        self._drag_offset_start = None
        self.setMinimumWidth(320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, path: Path):
        """Show *path*; returns (width, height, channels, dtype) of the file, or None."""
        meta = None
        try:
            img, meta = render_file(path, PREVIEW_MAX_SIDE)
            self._pixmap = QPixmap.fromImage(img)
        except Exception as e:
            logger.warning(f"Cannot preview {path}: {e}")
            self._pixmap = None
        self._zoom = 1.0
        self._pan_offset = [0.0, 0.0]
        self.update()
        return meta

    def clear(self):
        self._pixmap = None
        self._zoom = 1.0
        self._pan_offset = [0.0, 0.0]
        self.update()

    def wheelEvent(self, ev: QWheelEvent):
        if self._pixmap:
            delta = ev.angleDelta().y()
            factor = 1.15 if delta > 0 else 1 / 1.15
            self._zoom = max(0.1, min(20.0, self._zoom * factor))
            self.update()

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton and self._pixmap:
            self._drag_start = ev.position()
            self._drag_offset_start = list(self._pan_offset)

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self._drag_start and self._drag_offset_start:
            delta = ev.position() - self._drag_start
            self._pan_offset[0] = self._drag_offset_start[0] + delta.x()
            self._pan_offset[1] = self._drag_offset_start[1] + delta.y()
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._drag_start = None
        self._drag_offset_start = None

    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        self._zoom = 1.0
        self._pan_offset = [0.0, 0.0]
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(COLORS["canvas"]))

        if not self._pixmap:
            p.setPen(QPen(QColor(COLORS["text_dim"])))
            p.setFont(QFont("Segoe UI", 12))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Select an image to preview")
            p.end()
            return

        # fit image into panel, then apply zoom + pan
        pw, ph = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        scale_fit = min(vw / pw, vh / ph, 1.0)
        scale = scale_fit * self._zoom

        draw_w = pw * scale
        draw_h = ph * scale
        cx = (vw - draw_w) / 2 + self._pan_offset[0]
        cy = (vh - draw_h) / 2 + self._pan_offset[1]

        p.drawPixmap(QRectF(cx, cy, draw_w, draw_h),
                     self._pixmap, QRectF(0, 0, pw, ph))
        p.end()


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Viewer page
# ═══════════════════════════════════════════════════════════════════════════

def _card(inner_layout):
    frame = QFrame()
    frame.setObjectName("Card")
    frame.setLayout(inner_layout)
    frame.layout().setContentsMargins(22, 20, 22, 20)
    frame.layout().setSpacing(16)
    return frame


class DatasetViewerPage(QWidget):
    """Full dataset browser with thumbnail grid + preview panel."""

    def __init__(self):
        super().__init__()
        self._paths: list[Path] = []
        self._filtered: list[int] = []  # indices into _paths
        self._tiles: list[_ThumbTile] = []
        self._selected_idx: int = -1
        self._loader: Optional[_ThumbLoader] = None
        self._thumbs: dict[int, QPixmap] = {}   # cached so filtering keeps thumbnails
        self._build()

    # ── build UI ──────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        page = QWidget()
        page.setObjectName("PageContent")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(36, 32, 36, 32)
        outer.setSpacing(16)

        # header
        title = QLabel("Dataset Viewer")
        title.setObjectName("PageTitle")
        sub = QLabel("Browse and inspect your image datasets.")
        sub.setObjectName("PageSubtitle")
        outer.addWidget(title)
        outer.addWidget(sub)

        # toolbar card
        tb_lay = QHBoxLayout()
        tb_lay.setSpacing(10)

        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Select a folder to browse…")
        tb_lay.addWidget(self._dir_edit, 1)

        self._dir_edit.returnPressed.connect(lambda: self.open_folder(self._dir_edit.text().strip()))
        self._sub_chk = QCheckBox("Include subfolders")
        self._sub_chk.toggled.connect(lambda _: self.open_folder(self._dir_edit.text().strip()))
        tb_lay.addWidget(self._sub_chk)
        browse_btn = variant(QPushButton("Open folder"), "primary")
        browse_btn.clicked.connect(self._pick_folder)
        tb_lay.addWidget(browse_btn)

        tb_frame = QFrame()
        tb_frame.setObjectName("Card")
        tb_frame.setLayout(tb_lay)
        tb_frame.layout().setContentsMargins(16, 12, 16, 12)
        outer.addWidget(tb_frame)

        # filter row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter by filename…")
        self._filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_edit, 1)

        self._count_label = QLabel("0 images")
        self._count_label.setObjectName("FieldLabel")
        self._count_label.setFixedWidth(100)
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        filter_row.addWidget(self._count_label)

        outer.addLayout(filter_row)

        # main split: grid | preview
        split = QHBoxLayout()
        split.setSpacing(16)

        # left: thumbnail grid in scroll area
        self._flow = _FlowContainer()
        self._flow.setMinimumHeight(200)
        scroll = QScrollArea()
        scroll.setWidget(self._flow)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(380)
        split.addWidget(scroll, 3)

        # right: preview + info
        right = QVBoxLayout()
        right.setSpacing(12)

        self._preview = _PreviewPanel()
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right.addWidget(self._preview, 1)

        # info card
        info_lay = QVBoxLayout()
        self._info_name = QLabel("—")
        self._info_name.setObjectName("SectionTitle")
        self._info_name.setWordWrap(True)
        self._info_dims = QLabel("")
        self._info_dims.setObjectName("FieldLabel")
        self._info_size = QLabel("")
        self._info_size.setObjectName("FieldLabel")
        info_lay.addWidget(self._info_name)
        info_lay.addWidget(self._info_dims)
        info_lay.addWidget(self._info_size)
        right.addWidget(_card(info_lay))

        # nav buttons
        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self._prev_btn = QPushButton("← Previous")
        self._prev_btn.clicked.connect(lambda: self._navigate(-1))
        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(lambda: self._navigate(1))
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._next_btn)
        right.addLayout(nav_row)

        split.addLayout(right, 2)
        outer.addLayout(split, 1)

        root.addWidget(page)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        remember_text(self._dir_edit, "viewer/dir")

    # ── folder loading ────────────────────────────────────────────────────

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select image folder")
        if d:
            self._dir_edit.setText(d)
            self._load_folder(Path(d))

    def open_folder(self, path: str):
        """Public API — load a folder from an external caller."""
        p = Path(path)
        if path and p.is_dir():          # Path("") is the current directory — never scan that
            self._dir_edit.setText(str(p))
            self._load_folder(p)

    def _load_folder(self, folder: Path):
        """Load a folder (also callable externally via open_folder)."""
        # cancel any running loader; keep a reference until it really ends (a render can
        # take a few seconds) so the thread is never destroyed while running
        if self._loader and self._loader.isRunning():
            old = self._loader
            old.cancel()
            self._old_loaders = [t for t in getattr(self, "_old_loaders", []) if t.isRunning()] + [old]
            old.finished.connect(lambda t=old: self._old_loaders.remove(t) if t in self._old_loaders else None)

        # scan for images (only this folder unless "Include subfolders" is on)
        walker = folder.rglob("*") if self._sub_chk.isChecked() else folder.glob("*")
        self._paths = sorted(
            [p for p in walker if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()],
            key=lambda p: p.name.lower(),
        )
        self._thumbs = {}
        self._filtered = list(range(len(self._paths)))
        self._selected_idx = -1
        self._filter_edit.clear()
        self._rebuild_grid()
        self._update_count()
        self._clear_preview()

        # start background thumb loading
        if self._paths:
            self._loader = _ThumbLoader(self._paths, THUMB_SIZE)
            self._loader.thumb_ready.connect(self._on_thumb_ready)
            self._loader.start()

    # ── grid management ───────────────────────────────────────────────────

    def _rebuild_grid(self):
        self._flow.clear_tiles()
        self._tiles.clear()

        for real_idx in self._filtered:
            tile = _ThumbTile(real_idx, self._paths[real_idx].name)
            tile.clicked.connect(self._on_tile_clicked)
            if real_idx in self._thumbs:
                tile.set_pixmap(self._thumbs[real_idx])
            self._tiles.append(tile)
        self._flow.add_tiles(self._tiles)
        self._tile_by_idx = {t.index: t for t in self._tiles}

    def _on_thumb_ready(self, real_idx: int, img: QImage):
        if self.sender() is not self._loader:
            return          # a late thumbnail from the previous folder
        pm = QPixmap.fromImage(img)
        self._thumbs[real_idx] = pm
        tile = getattr(self, "_tile_by_idx", {}).get(real_idx)
        if tile is not None:
            tile.set_pixmap(pm)

    def _on_tile_clicked(self, real_idx: int):
        self._select(real_idx)

    def _select(self, real_idx: int):
        self._selected_idx = real_idx
        for tile in self._tiles:
            tile.set_selected(tile.index == real_idx)

        p = self._paths[real_idx]
        meta = self._preview.set_image(p)

        # metadata
        self._info_name.setText(p.name)
        if meta:
            w, h, ch, dtype = meta
            self._info_dims.setText(f"{w} × {h} px · {ch} channel{'s' if ch > 1 else ''} · {dtype}")
        else:
            self._info_dims.setText("Cannot read this file")

        try:
            size_bytes = p.stat().st_size
            if size_bytes < 1024:
                self._info_size.setText(f"{size_bytes} B")
            elif size_bytes < 1024 * 1024:
                self._info_size.setText(f"{size_bytes / 1024:.1f} KB")
            else:
                self._info_size.setText(f"{size_bytes / (1024 * 1024):.1f} MB")
        except Exception:
            self._info_size.setText("")

    def _clear_preview(self):
        self._preview.clear()
        self._info_name.setText("—")
        self._info_dims.setText("")
        self._info_size.setText("")

    # ── filter ────────────────────────────────────────────────────────────

    def _apply_filter(self, text: str):
        text = text.strip().lower()
        if not text:
            self._filtered = list(range(len(self._paths)))
        else:
            self._filtered = [
                i for i, p in enumerate(self._paths)
                if text in p.name.lower()
            ]
        self._rebuild_grid()   # tiles pick their thumbnails from the cache
        self._update_count()

    def _update_count(self):
        n = len(self._filtered)
        total = len(self._paths)
        if n == total:
            self._count_label.setText(f"{total} images")
        else:
            self._count_label.setText(f"{n} / {total}")

    # ── keyboard navigation ───────────────────────────────────────────────

    def _navigate(self, delta: int):
        if not self._filtered:
            return
        if self._selected_idx < 0:
            self._select(self._filtered[0])
            return
        # find current position in filtered list
        try:
            pos = self._filtered.index(self._selected_idx)
        except ValueError:
            pos = 0
        new_pos = max(0, min(len(self._filtered) - 1, pos + delta))
        self._select(self._filtered[new_pos])

    def keyPressEvent(self, ev: QKeyEvent):
        key = ev.key()
        if key == Qt.Key.Key_Right or key == Qt.Key.Key_Down:
            self._navigate(1)
        elif key == Qt.Key.Key_Left or key == Qt.Key.Key_Up:
            self._navigate(-1)
        elif key == Qt.Key.Key_Escape:
            self._selected_idx = -1
            for t in self._tiles:
                t.set_selected(False)
            self._clear_preview()
        else:
            super().keyPressEvent(ev)
