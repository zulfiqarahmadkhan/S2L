"""Manual Segmentation Correction editor.

Review a segmentation mask over the original image and fix it: pencil, partial
eraser, outline-and-fill, magic wand, AI add-cell (Cellpose-SAM on a box),
delete (click or rectangle) and fill holes, with undo/redo, clean-up filters,
cell info on hover, cell numbers, a second-channel overlay, and Previous / Next
navigation through a folder with autosave and automatic re-measurement.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox,
    QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from s2l.core.mask_edit import (
    cell_centroids, delete_cell, delete_in_rect, diff_region, fill_holes_all, fill_holes_cell,
    fill_polygon, insert_object, magic_wand, next_label, paint_stroke,
    remove_border, remove_small,
)
from s2l.ui.theme import COLORS

logger = logging.getLogger(__name__)

# ── colour palette for overlay ────────────────────────────────────────────────
_BRUSH_COLOURS = [         # cyclic colours per cell label
    (0, 230, 118), (255, 171, 0), (0, 176, 255), (255, 64, 129),
    (213, 0, 249), (0, 230, 64), (255, 214, 0), (100, 255, 218),
]
_COLOUR_LUT = np.array(_BRUSH_COLOURS, dtype=np.uint8)

_MODES = ("rgb", "r", "g", "b")


# ── image helpers (also used by the Correction page thumbnails) ───────────────

def load_image_normalised(path, max_side: int | None = None) -> np.ndarray:
    """Read an image as float32 (H, W, C) in [0, 1], C = 1 or 3.

    Each channel is stretched between its 0.5 and 99.5 percentiles so 8-bit,
    16-bit and mostly-dark fluorescence images are all visible.
    """
    import cv2
    from skimage.io import imread

    img = imread(str(path))
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.shape[2] == 4:
        img = img[:, :, :3]                  # drop alpha
    elif img.shape[2] == 2:
        img = img[:, :, :1]
    if max_side and max(img.shape[:2]) > max_side:
        f = max_side / max(img.shape[:2])
        size = (max(1, int(img.shape[1] * f)), max(1, int(img.shape[0] * f)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
    out = np.zeros(img.shape, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, (0.5, 99.5))
        if hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def load_mask(path) -> np.ndarray:
    from skimage.io import imread
    mask = imread(str(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int32)


def render_view(raw: np.ndarray, mode: str = "rgb", intensity: float = 1.0,
                brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray:
    """float (H, W, C) image → uint8 RGB for display."""
    h, w, n = raw.shape
    if n == 1:
        rgb = np.repeat(raw, 3, axis=2)
    else:
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, :n] = raw[:, :, :3]
        if mode in ("r", "g", "b"):
            keep = "rgb".index(mode)
            for c in range(3):
                if c != keep:
                    rgb[:, :, c] = 0
    rgb = (rgb * intensity - 0.5) * contrast + 0.5 + brightness
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def cell_outlines(mask: np.ndarray) -> np.ndarray:
    """Boolean map of pixels on the edge of a cell (label differs from a 4-neighbour)."""
    edge = np.zeros(mask.shape, dtype=bool)
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    return edge & (mask > 0)


def overlay_mask(display: np.ndarray, mask: np.ndarray, opacity: float,
                 outline: bool = False, outline_px: int = 1) -> np.ndarray:
    """Blend coloured cell labels onto a uint8 RGB image (only touches cell pixels).

    With *outline* only cell edges are drawn (``outline_px`` thick), so dense
    images stay readable.
    """
    out = display.copy()
    if opacity <= 0:
        return out
    if outline:
        fg = cell_outlines(mask)
        if outline_px > 1:
            import cv2
            fg = cv2.dilate(fg.astype(np.uint8), np.ones((outline_px, outline_px), np.uint8)) > 0
            fg &= mask > 0
        opacity = min(1.0, opacity * 2)
    else:
        fg = mask > 0
    colours = _COLOUR_LUT[(mask[fg] - 1) % len(_COLOUR_LUT)].astype(np.float32)
    out[fg] = (display[fg].astype(np.float32) * (1 - opacity) + colours * opacity).astype(np.uint8)
    return out


def render_pair_thumbnail(image_path, mask_path, max_side: int = 220) -> np.ndarray:
    """Small uint8 RGB preview of an image with its mask overlaid."""
    import cv2
    raw = load_image_normalised(image_path, max_side=max_side)
    mask = load_mask(mask_path)
    mask = cv2.resize(mask.astype(np.float32), (raw.shape[1], raw.shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return overlay_mask(render_view(raw), mask, 0.45)


def ndarray_to_qimage(arr: np.ndarray) -> QImage:
    """uint8 (H, W, 3) → QImage that owns its own copy of the data."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


# ── AI (Cellpose-SAM) helper ──────────────────────────────────────────────────

class _AISegmentWorker(QThread):
    """Runs Cellpose-SAM on the box (inference only; the model is loaded beforehand)."""
    done = pyqtSignal(object, str)   # boolean object mask (full image) or None, method / reason
    failed = pyqtSignal(str)

    def __init__(self, model, image, free, box, from_click):
        super().__init__()
        self.model, self.image, self.free, self.box, self.from_click = model, image, free, box, from_click

    def run(self):
        from s2l.core.ai_assist import segment_box
        try:
            obj, how = segment_box(self.model, self.image, self.free, self.box, self.from_click)
            self.done.emit(obj, how)
        except Exception as e:
            self.failed.emit(str(e))


class _Toast(QLabel):
    """Short message floating over the image (bottom centre), hides after a few seconds."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, kind: str = "Pending", ms: int = 3500):
        colour = {"Success": COLORS["success"], "Warning": COLORS["warning"],
                  "Error": COLORS["error"]}.get(kind, COLORS["accent"])
        self.setStyleSheet(
            f"background: rgba(20, 20, 26, 225); color: #ffffff; border: 2px solid {colour}; "
            f"border-radius: 10px; padding: 10px 16px; font-size: 13px; font-weight: 600;")
        self.setText(text)
        par = self.parentWidget()
        self.setFixedWidth(min(560, max(260, par.width() - 40)))
        self.adjustSize()
        # bottom centre: the top is where an opened / focused cell often sits
        self.move((par.width() - self.width()) // 2, par.height() - self.height() - 18)
        self.raise_()
        self.show()
        if ms > 0:
            self._timer.start(ms)
        else:
            self._timer.stop()


class _MeasureWorker(QThread):
    """Re-runs the ROI measurements (Excel sheet + overlay) for one corrected image."""
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, mask_path: Path, image_path: Path):
        super().__init__()
        self.mask_path, self.image_path = mask_path, image_path

    def run(self):
        try:
            from s2l.core.image_files import list_masks
            from s2l.core.roi_converter import ROIVisualizer
            from s2l.core.summary import generate_summary_sheet
            folder = self.mask_path.parent
            stem = self.mask_path.stem
            xlsx = folder / f"{stem}.xlsx"
            ROIVisualizer(str(self.mask_path), str(self.image_path), str(xlsx),
                          str(folder / f"{stem}_ROI.png"), show_labels=True).save_rois_to_excel()
            summary = folder / "SummarySheet" / "Summary.xlsx"
            if summary.exists():
                sheets = [folder / f"{m.stem}.xlsx" for m in list_masks(folder)]
                generate_summary_sheet(str(folder), str(summary), files=[s for s in sheets if s.exists()])
            self.done.emit(str(xlsx))
        except Exception as e:
            self.failed.emit(str(e))


# ── image canvas widget ───────────────────────────────────────────────────────

TOOLS = [
    # key, label, shortcut, tooltip
    ("pencil",  "✏️ Pencil",      "P", "Paint a cell. Starting inside a cell extends it; on background it adds a "
                                         "new cell. Hold Ctrl to paint over neighbours."),
    ("eraser",  "🩹 Eraser",      "E", "Erase only the pixels you paint over (trim a cell)."),
    ("outline", "➰ Outline",     "L", "Draw around a missed cell; it is filled when you release. Ctrl = over neighbours."),
    ("wand",    "🪄 Wand",        "W", "Click: grow a cell over pixels of similar brightness (tolerance below). "
                                         "Best for evenly bright cells; for textured cells use AI add cell."),
    ("ai",      "🤖 AI add cell", "A", "Drag a box around a missed cell (or click on it): Cellpose-SAM segments it."),
    ("delete",  "🗑 Delete",      "D", "Click a cell to delete it, or drag a rectangle to delete every cell it touches."),
    ("fill",    "🪣 Fill holes",  "F", "Click a cell to fill the holes inside it."),
]
_DRAG_TOOLS = {"outline", "ai", "delete"}


class _Canvas(QLabel):
    """Displays the image + mask overlay and turns mouse input into edits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

        # assigned by the dialog
        self.mask: np.ndarray | None = None        # H×W integer labels
        self.display: np.ndarray | None = None     # H×W×3 uint8 display image
        self.opacity = 0.4
        self.outline = False
        self.show_numbers = False
        self.centroids: dict[int, tuple[float, float]] = {}
        self._scale = 1.0
        self._tool = "pencil"
        self._brush_size = 10
        self._drawing = False
        self._last_pt: tuple[int, int] | None = None
        self._next_label = 1
        self._overwrite = False
        self._drag_start: QPointF | None = None    # widget coords for rect tools
        self._drag_now: QPointF | None = None
        self._path: list[tuple[int, int]] = []     # image coords for the outline tool
        self._pan_from: QPointF | None = None
        self.pending_box: tuple[int, int, int, int] | None = None   # AI box being processed
        self.space_down = False
        self.hover_label = 0
        self._hover_poly: list[QPolygonF] = []

        # callbacks set by the dialog
        self.on_mask_changed = None   # () → after a brush stroke
        self.on_stroke_start = None   # () → before a brush stroke edits the mask
        self.on_zoom_wheel = None     # (steps) → Ctrl + mouse wheel
        self.on_action = None         # (tool, payload, overwrite) → dialog-side edits
        self.on_hover = None          # (label, x, y)
        self.on_pan = None            # (dx, dy) in widget pixels

    # ── rendering ─────────────────────────────────────────────────────────

    def refresh(self):
        if self.display is None or self.mask is None:
            return
        # thicker edges when zoomed out, otherwise 1-px outlines vanish on screen
        px = 1 if self._scale >= 0.6 else min(4, int(round(1 / self._scale)))
        composed = overlay_mask(self.display, self.mask, self.opacity, self.outline, px)
        pix = QPixmap.fromImage(ndarray_to_qimage(composed))
        if self._scale != 1.0:
            # fast scaling while drawing keeps strokes responsive on large images
            mode = (Qt.TransformationMode.FastTransformation if self._drawing
                    else Qt.TransformationMode.SmoothTransformation)
            pix = pix.scaled(max(1, int(pix.width() * self._scale)),
                             max(1, int(pix.height() * self._scale)),
                             Qt.AspectRatioMode.KeepAspectRatio, mode)
        self.setPixmap(pix)
        self.setFixedSize(pix.size())
        self._update_hover_poly()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self._scale
        if self.show_numbers and self.centroids:
            font = QFont()
            font.setPointSizeF(max(6.0, min(11.0, 9 * s ** 0.5)))
            font.setBold(True)
            p.setFont(font)
            visible = self.visibleRegion().boundingRect()
            shown = [(lab, cx * s, cy * s) for lab, (cy, cx) in self.centroids.items()
                     if visible.contains(int(cx * s), int(cy * s))]
            if len(shown) <= 400:   # more than this is unreadable — zoom in to see numbers
                for lab, x, y in shown:
                    p.setPen(QColor(0, 0, 0, 200))
                    p.drawText(QPointF(x - 4, y + 5), str(lab))
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(QPointF(x - 5, y + 4), str(lab))
        if self._hover_poly:
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for poly in self._hover_poly:
                p.drawPolygon(poly)
        if self._drag_start is not None and self._drag_now is not None and self._tool in ("ai", "delete"):
            colour = QColor(COLORS["accent"]) if self._tool == "ai" else QColor(COLORS["error"])
            p.setPen(QPen(colour, 2, Qt.PenStyle.DashLine))
            p.setBrush(QColor(colour.red(), colour.green(), colour.blue(), 40))
            p.drawRect(QRectF(self._drag_start, self._drag_now).normalized())
        if self.pending_box is not None:
            x0, y0, x1, y1 = self.pending_box
            r = QRectF(x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s)
            p.setPen(QPen(QColor(COLORS["accent"]), 2, Qt.PenStyle.DashLine))
            p.setBrush(QColor(124, 106, 239, 50))
            p.drawRect(r)
            p.setPen(QColor(255, 255, 255))
            p.drawText(r.adjusted(4, 2, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "AI…")
        if self._path:
            p.setPen(QPen(QColor(COLORS["accent"]), 2))
            pts = [QPointF(x * s, y * s) for x, y in self._path]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        p.end()

    def _update_hover_poly(self):
        """Outline of the cell under the mouse, in widget coordinates."""
        import cv2
        self._hover_poly = []
        lab = self.hover_label
        if not lab or self.mask is None:
            return
        ys, xs = np.nonzero(self.mask == lab)
        if not len(ys):
            return
        y0, x0 = ys.min(), xs.min()
        sub = (self.mask[y0:ys.max() + 1, x0:xs.max() + 1] == lab).astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        s = self._scale
        for c in contours:
            self._hover_poly.append(QPolygonF([QPointF((pt[0][0] + x0 + 0.5) * s, (pt[0][1] + y0 + 0.5) * s)
                                               for pt in c]))

    # ── coordinate helpers ────────────────────────────────────────────────

    def _canvas_to_img(self, pt) -> tuple[int, int]:
        x = int(pt.x() / self._scale)
        y = int(pt.y() / self._scale)
        if self.mask is not None:
            h, w = self.mask.shape[:2]
            x = min(max(x, 0), w - 1)
            y = min(max(y, 0), h - 1)
        return x, y

    # ── mouse events ──────────────────────────────────────────────────────

    def mousePressEvent(self, ev):
        pos = ev.position()
        if ev.button() == Qt.MouseButton.MiddleButton or (
                ev.button() == Qt.MouseButton.LeftButton and self.space_down):
            self._pan_from = ev.globalPosition()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if ev.button() != Qt.MouseButton.LeftButton or self.mask is None:
            return
        self._overwrite = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        x, y = self._canvas_to_img(pos)
        tool = self._tool
        if tool in ("pencil", "eraser"):
            if self.on_stroke_start:
                self.on_stroke_start()
            self._drawing = True
            clicked = int(self.mask[y, x])
            # starting inside a cell extends it; starting on background makes a new cell
            self._next_label = 0 if tool == "eraser" else (clicked or int(self.mask.max()) + 1)
            self._last_pt = None
            self._apply_stroke(pos)
        elif tool == "outline":
            # 0 = a new cell; its number is chosen on release (the mask may change meanwhile)
            self._next_label = int(self.mask[y, x])
            self._path = [(x, y)]
        elif tool in ("ai", "delete"):
            self._drag_start = self._drag_now = pos
        elif self.on_action:
            self.on_action(tool, (x, y), self._overwrite)

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._pan_from is not None:
            d = ev.globalPosition() - self._pan_from
            self._pan_from = ev.globalPosition()
            if self.on_pan:
                self.on_pan(-d.x(), -d.y())
            return
        if self.mask is None:
            return
        if self._drawing:
            self._apply_stroke(pos)
            return
        if self._path:
            x, y = self._canvas_to_img(pos)
            if (x, y) != self._path[-1]:
                self._path.append((x, y))
                self.update()
            return
        if self._drag_start is not None:
            self._drag_now = pos
            self.update()
            return
        # hover: highlight the cell under the mouse and report its details
        x, y = self._canvas_to_img(pos)
        lab = int(self.mask[y, x])
        if lab != self.hover_label:
            self.hover_label = lab
            self._update_hover_poly()
            self.update()
        if self.on_hover:
            self.on_hover(lab, x, y)

    def mouseReleaseEvent(self, ev):
        if self._pan_from is not None:
            self._pan_from = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.space_down else Qt.CursorShape.CrossCursor)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if self._drawing:
            self._drawing = False
            self._last_pt = None
            self.refresh()   # redraw with smooth scaling
            if self.on_mask_changed:
                self.on_mask_changed()
        elif self._path:
            path, self._path = self._path, []
            self.update()
            if self.on_action:
                self.on_action("outline", (path, self._next_label), self._overwrite)
        elif self._drag_start is not None:
            a = self._canvas_to_img(self._drag_start)
            b = self._canvas_to_img(ev.position())
            self._drag_start = self._drag_now = None
            self.update()
            if self.on_action:
                self.on_action(self._tool, (a, b), self._overwrite)

    def leaveEvent(self, ev):
        if self.hover_label:
            self.hover_label = 0
            self._hover_poly = []
            self.update()
        super().leaveEvent(ev)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier and self.on_zoom_wheel:
            self.on_zoom_wheel(1 if ev.angleDelta().y() > 0 else -1)
            ev.accept()
        else:
            super().wheelEvent(ev)

    def _apply_stroke(self, pt):
        x, y = self._canvas_to_img(pt)
        r = max(1, self._brush_size // 2)
        paint_stroke(self.mask, self._last_pt or (x, y), (x, y), r, self._next_label, self._overwrite)
        self._last_pt = (x, y)
        self.refresh()


# ── main dialog ───────────────────────────────────────────────────────────────

class ManualCorrectionDialog(QDialog):
    """Interactive segmentation correction editor.

    Parameters
    ----------
    image_path, mask_path : str
        The pair to open first.
    pairs : list of (mask_path, image_path), optional
        All pairs in the folder, for Previous / Next navigation.
    index : int
        Position of the first pair in *pairs*.
    """

    _MAX_UNDO = 40
    _ZOOM_STEPS = (0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    def __init__(self, image_path: str, mask_path: str, parent=None,
                 pairs: list[tuple[str, str]] | None = None, index: int = 0):
        super().__init__(parent)
        self.setMinimumSize(1000, 650)
        self.resize(1440, 920)
        self.setModal(True)

        self._pairs = [(str(m), str(i)) for m, i in pairs] if pairs else [(str(mask_path), str(image_path))]
        self._index = index if pairs else 0
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._before: np.ndarray | None = None
        self._dirty = False
        self._saved_path: Path | None = None
        self._ai_worker: _AISegmentWorker | None = None
        self._measure_worker: _MeasureWorker | None = None
        self._overlay: np.ndarray | None = None
        self._mode = "rgb"
        self._intensity = 1.0
        self._brightness = 0.0
        self._contrast = 1.0
        self._last_opacity = 40

        self._load_pair(mask_path, image_path)     # raises on unreadable / mismatched pair
        self._build()
        self._apply_pair_to_ui()
        QTimer.singleShot(0, self._fit_zoom)
        QApplication.instance().installEventFilter(self)

    # ── loading ───────────────────────────────────────────────────────────

    def _load_pair(self, mask_path, image_path):
        from skimage.io import imread
        self._pending_ai = None            # an AI result for the previous image must not land here
        raw = load_image_normalised(image_path)
        mask = load_mask(mask_path)
        if mask.shape != raw.shape[:2]:
            raise ValueError(
                f"Mask size {mask.shape[1]}×{mask.shape[0]} does not match "
                f"image size {raw.shape[1]}×{raw.shape[0]}.\n"
                "Check that you picked the image this mask was made from."
            )
        orig = imread(str(image_path))
        if orig.ndim == 3:
            orig = orig[..., :3].mean(axis=-1)
        self._img_path, self._mask_path = Path(image_path), Path(mask_path)
        self._raw_img = raw                              # float32 (H, W, C) for display
        self._intensity_map = raw.mean(axis=2)           # [0, 1] for the magic wand
        self._raw_gray = orig.astype(np.float32)         # original values for the cell info
        self._mask = mask
        self._mask_backup = mask.copy()                  # for "revert all"
        self._n_channels = raw.shape[2]
        self._is_gray = self._n_channels == 1
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = False
        self._overlay = None

    def _apply_pair_to_ui(self):
        self.setWindowTitle(f"Manual Correction — {self._mask_path.name}")
        self._names_lbl.setText(f"Image: {self._img_path.name}\nMask: {self._mask_path.name}")
        self._canvas.mask = self._mask
        self._mode_btns["rgb"].setText("Gray" if self._is_gray else "RGB")
        for m in ("r", "g", "b"):
            self._mode_btns[m].setEnabled(not self._is_gray)
        if self._is_gray and self._mode != "rgb":
            self._mode = "rgb"
            self._mode_btns["rgb"].setChecked(True)
        self._overlay_chk.setChecked(False)
        self._overlay_chk.setEnabled(False)
        self._overlay_lbl.setText("No channel overlaid.")
        n = len(self._pairs)
        self._nav_lbl.setText(f"{self._index + 1} / {n}")
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < n - 1)
        self._canvas.hover_label = 0
        self._info_lbl.setText("Move the mouse over a cell.")
        self._refresh_display()
        self._mask_changed(dirty=False)

    # ── build UI ──────────────────────────────────────────────────────────

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._canvas = _Canvas()
        self._canvas.on_mask_changed = self._end_edit
        self._canvas.on_stroke_start = self._begin_edit
        self._canvas.on_zoom_wheel = self._zoom_step
        self._canvas.on_action = self._on_canvas_action
        self._canvas.on_hover = self._on_hover
        self._canvas.on_pan = self._pan_by

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._canvas)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setObjectName("ImageCanvas")
        self._toast = _Toast(self._scroll.viewport())

        panel = QWidget()
        panel.setObjectName("SideBar")
        p_lay = QVBoxLayout(panel)
        p_lay.setContentsMargins(16, 16, 16, 16)
        p_lay.setSpacing(12)

        title = QLabel("Manual Correction")
        title.setObjectName("SectionTitle")
        p_lay.addWidget(title)
        self._names_lbl = QLabel("")
        self._names_lbl.setWordWrap(True)
        self._names_lbl.setStyleSheet(_muted_style())
        p_lay.addWidget(self._names_lbl)
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setObjectName("StatusPending")
        p_lay.addWidget(self._status_lbl)

        # ── image navigation ──────────────────────────────────────────────
        nav_grp, nl = _group("Images in this folder", QVBoxLayout)
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Previous")
        self._prev_btn.clicked.connect(lambda: self._go(-1))
        self._nav_lbl = QLabel("1 / 1")
        self._nav_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.clicked.connect(lambda: self._go(1))
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._nav_lbl)
        nav_row.addWidget(self._next_btn)
        nl.addLayout(nav_row)
        self._autosave_chk = QCheckBox("Autosave when moving to another image")
        self._autosave_chk.setChecked(True)
        self._measure_chk = QCheckBox("Update Excel measurements after saving")
        self._measure_chk.setChecked(True)
        nl.addWidget(self._autosave_chk)
        nl.addWidget(self._measure_chk)
        p_lay.addWidget(nav_grp)

        # ── tools ─────────────────────────────────────────────────────────
        tool_grp, tl = _group("Tools  (Ctrl = paint over neighbours)", QVBoxLayout)
        grid = QGridLayout()
        grid.setSpacing(6)
        self._tool_btns = {}
        for i, (key, label, sc, tip) in enumerate(TOOLS):
            b = QPushButton(f"{label}  ({sc})")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setProperty("variant", "small")
            b.clicked.connect(lambda _, k=key: self._set_tool(k))
            grid.addWidget(b, i // 2, i % 2)
            self._tool_btns[key] = b
        tl.addLayout(grid)
        self._tool_hint = QLabel("")
        self._tool_hint.setWordWrap(True)
        self._tool_hint.setStyleSheet(_muted_style())
        tl.addWidget(self._tool_hint)
        self._brush_slider, _ = self._slider_row(tl, "Brush size", 2, 80, 10, lambda v: f"{v} px",
                                                 self._on_brush_changed)
        self._wand_slider, _ = self._slider_row(tl, "Wand tolerance", 1, 50, 15, lambda v: f"{v / 100:.2f}",
                                                lambda v: None)
        p_lay.addWidget(tool_grp)

        # ── cell info ─────────────────────────────────────────────────────
        info_grp, il = _group("Cell under the mouse", QVBoxLayout)
        self._info_lbl = QLabel("Move the mouse over a cell.")
        self._info_lbl.setWordWrap(True)
        self._info_lbl.setStyleSheet(_muted_style())
        il.addWidget(self._info_lbl)
        p_lay.addWidget(info_grp)

        # ── view ──────────────────────────────────────────────────────────
        zoom_grp, zl = _group("Zoom  (+ / −, Ctrl + wheel; pan: Space + drag)", QHBoxLayout)
        for lbl, val in [("Fit", None), ("50%", 0.5), ("100%", 1.0), ("200%", 2.0)]:
            b = QPushButton(lbl)
            b.setProperty("variant", "small")
            b.clicked.connect(lambda _, v=val: self._fit_zoom() if v is None else self._set_zoom(v))
            zl.addWidget(b)
        p_lay.addWidget(zoom_grp)

        col_grp, cl = _group("Colour", QVBoxLayout)
        mode_row = QGridLayout()
        mode_row.setSpacing(6)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_btns = {}
        for i, m in enumerate(_MODES):
            b = QPushButton({"rgb": "RGB", "r": "R", "g": "G", "b": "B"}[m])
            b.setCheckable(True)
            b.setProperty("channel", m)
            b.clicked.connect(lambda _, mm=m: self._set_mode(mm))
            self._mode_group.addButton(b)
            self._mode_btns[m] = b
            mode_row.addWidget(b, 0, i)
        self._mode_btns["rgb"].setChecked(True)
        cl.addLayout(mode_row)
        self._intensity_slider, _ = self._slider_row(cl, "Intensity", 0, 300, 100, lambda v: f"{v}%",
                                                     self._on_intensity)
        self._brightness_slider, _ = self._slider_row(cl, "Brightness", -100, 100, 0, lambda v: f"{v:+d}",
                                                      self._on_brightness)
        self._contrast_slider, _ = self._slider_row(cl, "Contrast", 20, 300, 100, lambda v: f"{v}%",
                                                    self._on_contrast)
        reset_btn = QPushButton("Reset colour")
        reset_btn.setProperty("variant", "small")
        reset_btn.clicked.connect(self._reset_colour)
        cl.addWidget(reset_btn)
        p_lay.addWidget(col_grp)

        # ── second channel overlay ────────────────────────────────────────
        ov_grp, ol = _group("Overlay another channel", QVBoxLayout)
        ov_row = QHBoxLayout()
        ov_btn = QPushButton("Choose image…")
        ov_btn.setProperty("variant", "small")
        ov_btn.clicked.connect(self._pick_overlay)
        self._tint_combo = QComboBox()
        for name, rgb in [("Red", (1, 0, 0)), ("Green", (0, 1, 0)), ("Blue", (0, 0.4, 1)),
                          ("Magenta", (1, 0, 1)), ("Cyan", (0, 1, 1)), ("Yellow", (1, 1, 0))]:
            self._tint_combo.addItem(name, rgb)
        self._tint_combo.currentIndexChanged.connect(lambda _: self._redraw())
        ov_row.addWidget(ov_btn)
        ov_row.addWidget(self._tint_combo)
        ol.addLayout(ov_row)
        self._overlay_chk = QCheckBox("Show overlay")
        self._overlay_chk.toggled.connect(lambda _: self._redraw())
        ol.addWidget(self._overlay_chk)
        self._overlay_slider, _ = self._slider_row(ol, "Overlay strength", 0, 300, 100, lambda v: f"{v}%",
                                                   lambda v: self._redraw())
        self._overlay_lbl = QLabel("No channel overlaid.")
        self._overlay_lbl.setWordWrap(True)
        self._overlay_lbl.setStyleSheet(_muted_style())
        ol.addWidget(self._overlay_lbl)
        p_lay.addWidget(ov_grp)

        # ── mask overlay ──────────────────────────────────────────────────
        mask_grp, ml = _group("Mask  (M = show / hide, O = outline, N = numbers)", QVBoxLayout)
        style_row = QHBoxLayout()
        self._fill_btn = QPushButton("Filled")
        self._outline_btn = QPushButton("Outline")
        for b, on in ((self._fill_btn, False), (self._outline_btn, True)):
            b.setCheckable(True)
            b.setProperty("variant", "small")
            b.clicked.connect(lambda _, o=on: self._set_outline(o))
            style_row.addWidget(b)
        self._fill_btn.setChecked(True)
        ml.addLayout(style_row)
        self._numbers_chk = QCheckBox("Show cell numbers (zoom in on dense images)")
        self._numbers_chk.toggled.connect(self._set_numbers)
        ml.addWidget(self._numbers_chk)
        self._opacity_slider, _ = self._slider_row(
            ml, "Opacity", 0, 100, self._last_opacity, lambda v: f"{v}%", self._on_opacity)
        p_lay.addWidget(mask_grp)

        # ── clean-up ──────────────────────────────────────────────────────
        clean_grp, cg = _group("Clean-up (whole image)", QVBoxLayout)
        small_row = QHBoxLayout()
        self._min_size = QSpinBox()
        self._min_size.setRange(1, 100000)
        self._min_size.setValue(50)
        self._min_size.setSuffix(" px")
        small_btn = QPushButton("Remove smaller than")
        small_btn.setProperty("variant", "small")
        small_btn.clicked.connect(self._remove_small)
        small_row.addWidget(small_btn)
        small_row.addWidget(self._min_size)
        cg.addLayout(small_row)
        clean_row = QHBoxLayout()
        border_btn = QPushButton("Remove border cells")
        border_btn.setProperty("variant", "small")
        border_btn.clicked.connect(self._remove_border)
        holes_btn = QPushButton("Fill all holes")
        holes_btn.setProperty("variant", "small")
        holes_btn.clicked.connect(self._fill_all_holes)
        clean_row.addWidget(border_btn)
        clean_row.addWidget(holes_btn)
        cg.addLayout(clean_row)
        p_lay.addWidget(clean_grp)

        self._stats_lbl = QLabel("")
        self._stats_lbl.setWordWrap(True)
        self._stats_lbl.setStyleSheet(_muted_style())
        p_lay.addWidget(self._stats_lbl)
        p_lay.addStretch()

        # ── action buttons ────────────────────────────────────────────────
        ur = QHBoxLayout()
        undo_btn = QPushButton("↩ Undo (Ctrl+Z)")
        undo_btn.clicked.connect(self._undo)
        redo_btn = QPushButton("↪ Redo (Ctrl+Y)")
        redo_btn.clicked.connect(self._redo)
        ur.addWidget(undo_btn)
        ur.addWidget(redo_btn)
        p_lay.addLayout(ur)
        revert_btn = QPushButton("⟲ Revert all changes to this image")
        revert_btn.clicked.connect(self._revert_all)
        p_lay.addWidget(revert_btn)
        sr = QHBoxLayout()
        save_btn = QPushButton("💾 Save (Ctrl+S)")
        save_btn.setProperty("variant", "solid-success")
        save_btn.setToolTip("Overwrite the mask file")
        save_btn.clicked.connect(lambda: self._save())
        save_as_btn = QPushButton("Save as…")
        save_as_btn.clicked.connect(self._save_as)
        sr.addWidget(save_btn, 2)
        sr.addWidget(save_as_btn, 1)
        p_lay.addLayout(sr)
        cr = QHBoxLayout()
        help_btn = QPushButton("⌨ Shortcuts (F1)")
        help_btn.clicked.connect(self._show_help)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        cr.addWidget(help_btn)
        cr.addWidget(close_btn)
        p_lay.addLayout(cr)

        panel_scroll = QScrollArea()
        panel_scroll.setWidget(panel)
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel_scroll.setFixedWidth(360)
        panel.setMaximumWidth(360)
        panel_scroll.setStyleSheet(
            f"QScrollArea {{ background: {COLORS['bg_base']}; border: none; "
            f"border-left: 1px solid {COLORS['border']}; }}")

        # keyboard shortcuts
        for key, _, sc, _ in TOOLS:
            QShortcut(QKeySequence(sc), self, lambda k=key: self._set_tool(k))
        QShortcut(QKeySequence("M"), self, self._toggle_mask)
        QShortcut(QKeySequence("O"), self, lambda: self._set_outline(not self._canvas.outline))
        QShortcut(QKeySequence("N"), self, lambda: self._numbers_chk.setChecked(not self._numbers_chk.isChecked()))
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._redo)
        QShortcut(QKeySequence("Ctrl+S"), self, lambda: self._save())
        for k in ("+", "="):
            QShortcut(QKeySequence(k), self, lambda: self._zoom_step(1))
        QShortcut(QKeySequence("-"), self, lambda: self._zoom_step(-1))
        QShortcut(QKeySequence("F1"), self, self._show_help)
        QShortcut(QKeySequence("?"), self, self._show_help)
        QShortcut(QKeySequence("PgDown"), self, lambda: self._go(1))
        QShortcut(QKeySequence("PgUp"), self, lambda: self._go(-1))

        root.addWidget(self._scroll, 1)
        root.addWidget(panel_scroll)
        self._set_tool("pencil")

    def _slider_row(self, layout, name, lo, hi, value, fmt, slot):
        """Label + value readout above a horizontal slider; returns (slider, value label)."""
        head = QHBoxLayout()
        lbl = QLabel(name)
        lbl.setStyleSheet(_muted_style())
        val = QLabel(fmt(value))
        val.setStyleSheet(_muted_style())
        val.setAlignment(Qt.AlignmentFlag.AlignRight)
        head.addWidget(lbl)
        head.addWidget(val)
        layout.addLayout(head)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        s.valueChanged.connect(lambda v: (val.setText(fmt(v)), slot(v)))
        layout.addWidget(s)
        return s, val

    # ── keyboard: Space pans while held ───────────────────────────────────

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and self.isActiveWindow():
            if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
                down = ev.type() == QEvent.Type.KeyPress
                self._canvas.space_down = down
                self._canvas.setCursor(Qt.CursorShape.OpenHandCursor if down else Qt.CursorShape.CrossCursor)
                return True
            if ev.key() == Qt.Key.Key_Space:
                return True
        return super().eventFilter(obj, ev)

    def _pan_by(self, dx, dy):
        h, v = self._scroll.horizontalScrollBar(), self._scroll.verticalScrollBar()
        h.setValue(int(h.value() + dx))
        v.setValue(int(v.value() + dy))

    # ── tools ─────────────────────────────────────────────────────────────

    def _set_tool(self, tool_id: str):
        if tool_id == "ai" and hasattr(self, "_canvas") and self._canvas._tool != "ai":
            QTimer.singleShot(50, self._preload_ai)
        self._canvas._tool = tool_id
        for k, b in self._tool_btns.items():
            b.setChecked(k == tool_id)
        self._tool_hint.setText(next(t[3] for t in TOOLS if t[0] == tool_id))
        self._canvas.setCursor(Qt.CursorShape.PointingHandCursor if tool_id in ("delete", "fill", "wand")
                               else Qt.CursorShape.CrossCursor)

    def _preload_ai(self):
        try:
            self._ensure_ai_model()
        except Exception as e:
            self._status(f"Could not load the AI model: {e}", "Error")

    def _on_brush_changed(self, v: int):
        self._canvas._brush_size = v

    def _on_canvas_action(self, tool, payload, overwrite):
        """Edits that need more than the brush: outline, wand, AI, delete, fill."""
        if tool == "ai":
            self._start_ai(payload)
            return
        self._begin_edit()
        msg = ""
        if tool == "outline":
            path, label = payload
            n = fill_polygon(self._mask, path, label or next_label(self._mask), overwrite)
            msg = f"Outline filled ({n:,} px)." if n else "Outline too small — draw around the whole cell."
        elif tool == "wand":
            x, y = payload
            label = next_label(self._mask)
            n = magic_wand(self._mask, self._intensity_map, x, y, self._wand_slider.value() / 100,
                           label, overwrite)
            msg = f"Wand added a cell of {n:,} px." if n else "Nothing grown — click on the cell, or raise the tolerance."
        elif tool == "delete":
            (x0, y0), (x1, y1) = payload
            if abs(x1 - x0) < 3 and abs(y1 - y0) < 3:
                lab = delete_cell(self._mask, x0, y0)
                msg = f"Deleted cell #{lab}." if lab else "Click on a cell to delete it."
            else:
                n = delete_in_rect(self._mask, x0, y0, x1, y1)
                msg = f"Deleted {n} cell{'s' if n != 1 else ''}."
        elif tool == "fill":
            x, y = payload
            lab = int(self._mask[y, x])
            n = fill_holes_cell(self._mask, lab) if lab else 0
            msg = (f"Filled {n:,} px of holes in cell #{lab}." if n else
                   ("Cell #%d has no holes." % lab if lab else "Click on a cell to fill its holes."))
        self._end_edit()
        self._status(msg, "Pending")

    # ── AI add cell ───────────────────────────────────────────────────────

    def _ensure_ai_model(self):
        """Load Cellpose-SAM on the main thread (once), showing that it is happening."""
        from s2l.core.ai_assist import get_model, model_loaded
        if not model_loaded():
            self._status("Loading the AI model (first time only, a few seconds)…", "Pending", toast_ms=0)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            try:
                get_model()
            finally:
                QApplication.restoreOverrideCursor()
            self._status("AI model ready — drag a box around a missed cell, or click on it.", "Success")
        return get_model()

    def _start_ai(self, payload):
        if self._ai_worker is not None and self._ai_worker.isRunning():
            self._status("AI is still working on the previous box…", "Warning")
            return
        (x0, y0), (x1, y1) = payload
        h, w = self._mask.shape
        from_click = abs(x1 - x0) < 6 or abs(y1 - y0) < 6
        if from_click:
            # a click: use a box about twice the typical cell size around it
            sizes = np.bincount(self._mask.ravel())[1:]
            diam = int(2 * np.sqrt(np.median(sizes[sizes > 0]) / np.pi)) if (sizes > 0).any() else 40
            half = max(24, diam)
            x0, x1, y0, y1 = x0 - half, x0 + half, y0 - half, y0 + half
        x0, x1 = sorted((max(0, x0), min(w - 1, x1)))
        y0, y1 = sorted((max(0, y0), min(h - 1, y1)))
        try:
            model = self._ensure_ai_model()
        except Exception as e:
            self._status(f"Could not load the AI model: {e}", "Error")
            return
        self._canvas.pending_box = (x0, y0, x1, y1)
        self._canvas.update()
        self._status("AI is finding the cell in the box…", "Pending", toast_ms=0)
        self._tool_btns["ai"].setEnabled(False)
        self._ai_worker = _AISegmentWorker(model, self._raw_img, self._mask == 0, (x0, y0, x1, y1), from_click)
        self._ai_for = self._mask_path          # results for another image are dropped
        self._ai_worker.done.connect(self._on_ai_done)
        self._ai_worker.failed.connect(self._on_ai_failed)
        self._ai_worker.start()

    def _on_ai_done(self, obj, how):
        self._tool_btns["ai"].setEnabled(True)
        self._canvas.pending_box = None
        self._canvas.update()
        if obj is None:
            self._status("No cell found in that box. Try a slightly larger box around the whole cell, "
                         "or draw it with Outline (L).", "Warning")
            return
        if getattr(self, "_ai_for", None) != self._mask_path or obj.shape != self._mask.shape:
            self._status("AI result discarded: the image was changed while it was working.", "Warning")
            return
        if self._canvas._drawing or self._canvas._path:
            self._pending_ai = (obj, how)        # applied when the current stroke / outline ends
            return
        self._apply_ai(obj, how)

    def _apply_ai(self, obj, how):
        self._begin_edit()
        n = insert_object(self._mask, obj, 0, 0, next_label(self._mask), False)
        self._end_edit()
        if n:
            self._status(f"Added a cell of {n:,} px using {how}. Ctrl+Z to undo.", "Success")
        else:
            self._status("That cell already exists here.", "Warning")

    def _on_ai_failed(self, msg):
        self._tool_btns["ai"].setEnabled(True)
        self._canvas.pending_box = None
        self._canvas.update()
        self._status(f"AI failed: {msg}", "Error")

    # ── undo / redo ───────────────────────────────────────────────────────

    def _begin_edit(self):
        self._before = self._mask.copy()

    def _end_edit(self):
        if self._before is None:
            return
        d = diff_region(self._before, self._mask)
        self._before = None
        pending = getattr(self, "_pending_ai", None)
        if pending is not None:
            # an AI result arrived during this stroke: record the stroke, then add the cell
            self._pending_ai = None
            QTimer.singleShot(0, lambda p=pending: self._apply_ai(*p) if p[0].shape == self._mask.shape else None)
        if d is None:
            self._canvas.refresh()
            return
        self._undo_stack.append(d)
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._mask_changed()

    def _undo(self):
        if not self._undo_stack:
            return
        sl, old, new = self._undo_stack.pop()
        self._mask[sl] = old
        self._redo_stack.append((sl, old, new))
        self._mask_changed()

    def _redo(self):
        if not self._redo_stack:
            return
        sl, old, new = self._redo_stack.pop()
        self._mask[sl] = new
        self._undo_stack.append((sl, old, new))
        self._mask_changed()

    def _revert_all(self):
        self._begin_edit()
        self._mask[:] = self._mask_backup
        self._end_edit()

    def _mask_changed(self, dirty: bool = True):
        if dirty:
            self._dirty = True
        if self._canvas.show_numbers:
            self._canvas.centroids = cell_centroids(self._mask)
        self._canvas.refresh()
        self._update_stats()

    # ── clean-up ──────────────────────────────────────────────────────────

    def _remove_small(self):
        self._begin_edit()
        n = remove_small(self._mask, self._min_size.value())
        self._end_edit()
        self._status(f"Removed {n} cell{'s' if n != 1 else ''} smaller than {self._min_size.value()} px.", "Pending")

    def _remove_border(self):
        self._begin_edit()
        n = remove_border(self._mask)
        self._end_edit()
        self._status(f"Removed {n} cell{'s' if n != 1 else ''} touching the border.", "Pending")

    def _fill_all_holes(self):
        self._begin_edit()
        n = fill_holes_all(self._mask)
        self._end_edit()
        self._status(f"Filled {n:,} px of holes.", "Pending")

    # ── hover info ────────────────────────────────────────────────────────

    def _on_hover(self, lab, x, y):
        if not lab:
            self._info_lbl.setText(f"Background  ·  x {x}, y {y}  ·  value {self._raw_gray[y, x]:.1f}")
            return
        cell = self._mask == lab
        vals = self._raw_gray[cell]
        self._info_lbl.setText(
            f"Cell #{lab}\nArea: {int(cell.sum()):,} px\n"
            f"Mean intensity: {vals.mean():.1f}  (min {vals.min():.0f}, max {vals.max():.0f})\n"
            f"Mouse at x {x}, y {y}")

    # ── zoom ──────────────────────────────────────────────────────────────

    def _show_help(self):
        rows = "".join(f"<tr><td><b>{k}</b></td><td>&nbsp;&nbsp;{v}</td></tr>" for k, v in [
            ("P / E / L / W", "Pencil · Eraser (partial) · Outline + fill · Magic wand"),
            ("A", "AI add cell — drag a box around a missed cell, or click on it"),
            ("D / F", "Delete (click a cell, or drag a rectangle) · Fill holes"),
            ("Ctrl + drag", "Paint / outline / wand over neighbouring cells"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
            ("Ctrl+S", "Save (overwrites the mask; Excel is updated if ticked)"),
            ("PgUp / PgDn", "Previous / next image in the folder (autosaves)"),
            ("+ / − , Ctrl + wheel", "Zoom in / out"),
            ("Space + drag, middle mouse", "Pan"),
            ("M / O / N", "Show-hide mask · Outline view · Cell numbers"),
            ("F1 or ?", "This help"),
        ])
        QMessageBox.information(self, "Keyboard shortcuts", f"<table>{rows}</table>")

    def focus_cell(self, label: int):
        """Zoom in on cell *label*, centre it and outline it (used by Plotting → Explore)."""
        ys, xs = np.nonzero(self._mask == label)
        if not len(ys):
            self._status(f"Cell #{label} is not in this mask (it may have been edited).", "Warning")
            return
        if self._canvas._scale < 1.5:
            self._set_zoom(1.5)
        s = self._canvas._scale
        vp = self._scroll.viewport()
        self._scroll.horizontalScrollBar().setValue(int(xs.mean() * s - vp.width() / 2))
        self._scroll.verticalScrollBar().setValue(int(ys.mean() * s - vp.height() / 2))
        self._canvas.hover_label = label
        self._canvas._update_hover_poly()
        self._canvas.update()
        self._on_hover(label, int(xs.mean()), int(ys.mean()))
        self._status(f"Cell #{label} (outlined in white).", "Success")

    def _set_zoom(self, scale: float):
        self._canvas._scale = max(0.05, min(scale, 8.0))
        self._canvas.refresh()

    def _fit_zoom(self):
        vp = self._scroll.viewport().size()
        h, w = self._mask.shape[:2]
        if vp.width() > 10 and vp.height() > 10:
            self._set_zoom(min(vp.width() / w, vp.height() / h, 1.0) * 0.98)

    def _zoom_step(self, direction: int):
        cur = self._canvas._scale
        if direction > 0:
            nxt = next((z for z in self._ZOOM_STEPS if z > cur + 1e-6), self._ZOOM_STEPS[-1])
        else:
            nxt = next((z for z in reversed(self._ZOOM_STEPS) if z < cur - 1e-6), self._ZOOM_STEPS[0])
        self._set_zoom(nxt)

    # ── colour / overlay ──────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self._mode = mode
        self._mode_btns[mode].setChecked(True)
        self._redraw()

    def _on_intensity(self, v: int):
        self._intensity = v / 100.0
        self._redraw()

    def _on_brightness(self, v: int):
        self._brightness = v / 100.0
        self._redraw()

    def _on_contrast(self, v: int):
        self._contrast = v / 100.0
        self._redraw()

    def _reset_colour(self):
        self._intensity_slider.setValue(100)
        self._brightness_slider.setValue(0)
        self._contrast_slider.setValue(100)
        self._set_mode("rgb")

    def _pick_overlay(self):
        start = str(self._img_path.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Image to overlay (same size, e.g. the R channel)", start,
                                              "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All (*)")
        if path:
            self.set_overlay(path)

    def set_overlay(self, path: str):
        try:
            ov = load_image_normalised(path).mean(axis=2)
        except Exception as e:
            self._status(f"Cannot read overlay: {e}", "Error")
            return
        if ov.shape != self._mask.shape:
            self._status(f"Overlay is {ov.shape[1]}×{ov.shape[0]}, but this image is "
                         f"{self._mask.shape[1]}×{self._mask.shape[0]}.", "Error")
            return
        self._overlay = ov
        self._overlay_chk.setEnabled(True)
        self._overlay_chk.setChecked(True)
        self._overlay_lbl.setText(f"Overlay: {Path(path).name}")
        self._redraw()

    def _redraw(self):
        self._refresh_display()
        self._canvas.refresh()

    def _refresh_display(self):
        """Rebuild the uint8 RGB display image from the raw channels (+ overlay)."""
        disp = render_view(self._raw_img, self._mode, self._intensity, self._brightness, self._contrast)
        if self._overlay is not None and self._overlay_chk.isChecked():
            tint = np.array(self._tint_combo.currentData(), np.float32)
            add = self._overlay[:, :, None] * tint * (self._overlay_slider.value() / 100.0) * 255
            disp = np.clip(disp.astype(np.float32) + add, 0, 255).astype(np.uint8)
        self._canvas.display = disp

    def _on_opacity(self, v: int):
        self._canvas.opacity = v / 100.0
        if v:
            self._last_opacity = v
        self._canvas.refresh()

    def _toggle_mask(self):
        self._opacity_slider.setValue(0 if self._opacity_slider.value() else self._last_opacity)

    def _set_outline(self, on: bool):
        self._canvas.outline = on
        self._fill_btn.setChecked(not on)
        self._outline_btn.setChecked(on)
        self._canvas.refresh()

    def _set_numbers(self, on: bool):
        self._canvas.show_numbers = on
        self._canvas.centroids = cell_centroids(self._mask) if on else {}
        self._canvas.update()

    # ── stats / status ────────────────────────────────────────────────────

    def _update_stats(self):
        n_cells = int(np.count_nonzero(np.bincount(self._mask.ravel())[1:]))
        self._stats_lbl.setText(
            f"Cells: {n_cells:,}{'   • unsaved changes' if self._dirty else ''}\n"
            f"Image: {self._raw_img.shape[1]}×{self._raw_img.shape[0]} px, "
            f"{'1 channel' if self._is_gray else f'{self._n_channels} channels'}")

    def _status(self, text: str, kind: str = "Pending", toast_ms: int = 3500):
        if text and hasattr(self, "_toast"):
            self._toast.show_message(text, kind, toast_ms)
        self._status_lbl.setText(text)
        self._status_lbl.setObjectName(f"Status{kind}")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)

    # ── navigation ────────────────────────────────────────────────────────

    def _go(self, step: int):
        target = self._index + step
        if not 0 <= target < len(self._pairs):
            return
        if self._dirty:
            if self._autosave_chk.isChecked():
                if not self._save():
                    return
            elif not self._confirm_discard():
                return
        mask_path, image_path = self._pairs[target]
        try:
            self._load_pair(mask_path, image_path)
        except Exception as e:
            self._status(f"Cannot open {Path(mask_path).name}: {e}", "Error")
            self._pairs.pop(target)
            if target < self._index:
                self._index -= 1
            self._apply_pair_to_ui()
            return
        self._index = target
        self._apply_pair_to_ui()
        self._fit_zoom()
        self._status(f"Opened {Path(mask_path).name}", "Pending")

    def _confirm_discard(self) -> bool:
        r = QMessageBox.question(
            self, "Unsaved changes", f"Save your changes to {self._mask_path.name}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Save:
            return self._save()
        return r == QMessageBox.StandardButton.Discard

    # ── save ──────────────────────────────────────────────────────────────

    def _save(self, dest: str | None = None) -> bool:
        """Save the mask (default: overwrite the mask file) and refresh its measurements."""
        from skimage.io import imsave
        from skimage.segmentation import relabel_sequential
        dest = Path(dest) if dest else self._mask_path
        try:
            # erased cells leave gaps in the numbering; keep labels 1..N
            out = relabel_sequential(self._mask)[0]
            # uint16 holds up to 65,535 cells; more would wrap around and merge cells
            imsave(str(dest), out.astype(np.uint32 if out.max() > 65535 else np.uint16), check_contrast=False)
        except Exception as e:
            self._status(f"Save failed: {e}", "Error")
            QMessageBox.critical(self, "Save failed", str(e))
            return False
        self._mask[:] = out
        self._mask_backup = self._mask.copy()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._dirty = False
        self._saved_path = dest
        self._mask_changed(dirty=False)
        self._status(f"✓ Saved {dest.name}", "Success")
        if self._measure_chk.isChecked() and dest == self._mask_path:
            self._update_measurements()
        return True

    def _save_as(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Save corrected mask as", str(self._mask_path),
                                              "TIFF (*.tif *.tiff);;PNG (*.png);;All (*)")
        if dest:
            self._save(dest)

    def _update_measurements(self):
        if self._measure_worker is not None and self._measure_worker.isRunning():
            self._measure_worker.wait()
        self._measure_worker = _MeasureWorker(self._mask_path, self._img_path)
        self._measure_worker.done.connect(
            lambda p: self._status(f"✓ Saved and measurements updated ({Path(p).name})", "Success"))
        self._measure_worker.failed.connect(lambda m: self._status(f"Saved, but measuring failed: {m}", "Warning"))
        self._measure_worker.start()
        self._status("✓ Saved — updating the Excel measurements…", "Success")

    def saved_path(self) -> Path | None:
        return self._saved_path

    # ── closing ───────────────────────────────────────────────────────────

    def _can_close(self) -> bool:
        if self._dirty and not self._confirm_discard():
            return False
        for w in (self._ai_worker, self._measure_worker):
            if w is not None and w.isRunning():
                w.wait()
        QApplication.instance().removeEventFilter(self)
        return True

    def reject(self):
        """Close button, Esc and the window X all end here."""
        if self._can_close():
            super().reject()

    def closeEvent(self, ev):
        ev.ignore()
        self.reject()


# ── stylesheet helpers ────────────────────────────────────────────────────────

def _group(title: str, layout_cls):
    grp = QGroupBox(title)
    lay = layout_cls(grp)
    lay.setSpacing(8)
    return grp, lay


def _muted_style() -> str:
    return f"color: {COLORS['text_secondary']}; font-size: 11px; background: transparent;"
