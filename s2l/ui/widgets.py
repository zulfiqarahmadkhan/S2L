"""Shared building blocks for S2L pages: cards, form rows, switches, settings, dialogs."""
import os
import subprocess
import sys

from PyQt6.QtCore import (
    QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSettings, QSize, Qt,
    pyqtProperty, pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)

from s2l.ui.theme import COLORS


# ═══════════════════════════════════════════════════════════════════════════
# Settings (remembered between sessions)
# ═══════════════════════════════════════════════════════════════════════════

def settings() -> QSettings:
    return QSettings("S2L", "S2L")


def remember_text(edit: QLineEdit, key: str):
    """Restore *edit*'s text from settings and save it whenever it changes."""
    saved = settings().value(f"fields/{key}", "", type=str)
    if saved and not edit.text():
        edit.setText(saved)
    edit.textChanged.connect(lambda t: settings().setValue(f"fields/{key}", t))


def add_recent_folder(path: str, limit: int = 6):
    s = settings()
    recent = [p for p in s.value("recent/folders", [], type=list) if p != path]
    s.setValue("recent/folders", [path] + recent[: limit - 1])


def recent_folders() -> list[str]:
    return [p for p in settings().value("recent/folders", [], type=list) if os.path.isdir(p)]


def trained_models(engine: str | None = None) -> list[dict]:
    """Models trained in S2L (newest first) whose files still exist."""
    import json
    try:
        items = json.loads(settings().value("models/registry", "[]", type=str))
    except ValueError:
        items = []
    items = [m for m in items if os.path.exists(m.get("path", ""))]
    return [m for m in items if engine is None or m.get("engine") == engine]


def register_model(entry: dict):
    """Add a model to the list (newest first), or update it in place if it is already listed."""
    import json
    items = trained_models()
    for i, m in enumerate(items):
        if m.get("path") == entry.get("path"):
            items[i] = entry
            break
    else:
        items.insert(0, entry)
    settings().setValue("models/registry", json.dumps(items))


def unregister_model(path: str):
    import json
    items = [m for m in trained_models() if m.get("path") != path]
    settings().setValue("models/registry", json.dumps(items))


# ═══════════════════════════════════════════════════════════════════════════
# Layout helpers
# ═══════════════════════════════════════════════════════════════════════════

def card(inner_layout) -> QFrame:
    """Wrap *inner_layout* in a styled card frame."""
    frame = QFrame()
    frame.setObjectName("Card")
    frame.setLayout(inner_layout)
    frame.layout().setContentsMargins(22, 20, 22, 20)
    frame.layout().setSpacing(16)
    return frame


def section_header(title: str, description: str = "") -> QVBoxLayout:
    """Return a VBox with a bold title and optional muted description."""
    lay = QVBoxLayout()
    lay.setSpacing(2)
    t = QLabel(title)
    t.setObjectName("SectionTitle")
    lay.addWidget(t)
    if description:
        d = QLabel(description)
        d.setObjectName("SectionDesc")
        d.setWordWrap(True)
        lay.addWidget(d)
    return lay


def field_row(label_text: str, widget, hint: str = "") -> QVBoxLayout:
    """Label above widget, optional hint below."""
    col = QVBoxLayout()
    col.setSpacing(5)
    col.setContentsMargins(0, 2, 0, 2)
    lbl = QLabel(label_text)
    lbl.setObjectName("FieldLabel")
    col.addWidget(lbl)
    if isinstance(widget, QHBoxLayout):
        col.addLayout(widget)
    else:
        col.addWidget(widget)
    if hint:
        h = QLabel(hint)
        h.setObjectName("Hint")
        h.setWordWrap(True)
        col.addWidget(h)
    return col


def browse_row(line_edit: QLineEdit, button_text: str, callback,
               view_callback=None) -> QHBoxLayout:
    """LineEdit + Browse button side by side, optional View button."""
    row = QHBoxLayout()
    row.setSpacing(8)
    row.addWidget(line_edit, 1)
    btn = QPushButton(button_text)
    btn.setMinimumWidth(90)
    btn.clicked.connect(callback)
    row.addWidget(btn)
    if view_callback:
        vbtn = QPushButton("View")
        vbtn.setMinimumWidth(56)
        vbtn.setToolTip("Open this folder in the dataset viewer")
        vbtn.clicked.connect(view_callback)
        row.addWidget(vbtn)
    return row


def scrollable(content_widget: QWidget) -> QScrollArea:
    """Wrap a widget in a transparent scroll area."""
    sa = QScrollArea()
    sa.setWidget(content_widget)
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    return sa


def page_shell(title: str, subtitle: str):
    """Standard page: returns (page widget, content column layout)."""
    page = QWidget()
    page.setObjectName("PageContent")
    col = QVBoxLayout(page)
    col.setContentsMargins(36, 32, 36, 32)
    col.setSpacing(20)
    t = QLabel(title)
    t.setObjectName("PageTitle")
    s = QLabel(subtitle)
    s.setObjectName("PageSubtitle")
    s.setWordWrap(True)
    col.addWidget(t)
    col.addWidget(s)
    col.addSpacing(4)
    return page, col


def variant(button: QPushButton, name: str) -> QPushButton:
    """Give a button a themed look: primary, danger, success, small."""
    button.setProperty("variant", name)
    return button


def set_status(label: QLabel, text: str, kind: str):
    """Update a status label; *kind* is Success / Warning / Error / Pending."""
    label.setText(text)
    label.setObjectName(f"Status{kind}")
    label.style().unpolish(label)
    label.style().polish(label)


def open_in_file_manager(path: str):
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 — opening a user folder in Explorer
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


# ═══════════════════════════════════════════════════════════════════════════
# Widgets
# ═══════════════════════════════════════════════════════════════════════════

class FitStack(QStackedWidget):
    """A stacked widget that is only as tall as its *current* page.

    A plain QStackedWidget reserves the height of its tallest page, which left
    large empty gaps on the forms.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(self._update_policies)

    def addWidget(self, w):
        idx = super().addWidget(w)
        self._update_policies()
        return idx

    def _update_policies(self, *_):
        for i in range(self.count()):
            w = self.widget(i)
            pol = QSizePolicy.Policy.Preferred if i == self.currentIndex() else QSizePolicy.Policy.Ignored
            w.setSizePolicy(QSizePolicy.Policy.Preferred, pol)
        self.updateGeometry()

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w else super().minimumSizeHint()


class Collapsible(QWidget):
    """A section with a clickable ▸/▾ header that shows or hides its content."""

    def __init__(self, title: str, content_layout, expanded: bool = False, parent=None):
        super().__init__(parent)
        self._title = title
        self._btn = QToolButton()
        self._btn.setObjectName("CollapseHeader")
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.toggled.connect(self._on_toggled)
        self._body = QWidget()
        self._body.setLayout(content_layout)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(self._btn)
        lay.addWidget(self._body)
        self._on_toggled(expanded)

    def _on_toggled(self, on: bool):
        self._btn.setText(("▾  " if on else "▸  ") + self._title)
        self._body.setVisible(on)


class SegmentedSwitch(QWidget):
    """Two or more mutually exclusive buttons (e.g. Cellpose | StarDist)."""
    changed = pyqtSignal(str)

    def __init__(self, options: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._buttons: dict[str, QPushButton] = {}
        for label, key in options:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setMinimumWidth(110)
            b.clicked.connect(lambda _, k=key: self.set_value(k))
            lay.addWidget(b)
            self._buttons[key] = b
        lay.addStretch()
        self._value = options[0][1]
        self._buttons[self._value].setChecked(True)

    def value(self) -> str:
        return self._value

    def set_value(self, key: str):
        self._value = key
        for k, b in self._buttons.items():
            b.setChecked(k == key)
        self.changed.emit(key)


class ToggleSwitch(QWidget):
    """Animated pill-shaped toggle switch with a label."""
    toggled = pyqtSignal(bool)

    _TRACK_W = 44
    _TRACK_H = 24
    _KNOB_MARGIN = 3
    _KNOB_D = _TRACK_H - _KNOB_MARGIN * 2  # 18

    def __init__(self, text: str = "", checked: bool = False, parent=None):
        super().__init__(parent)
        self._checked = checked
        self._text = text
        self._knob_x = float(self._on_x() if checked else self._off_x())
        self._enabled = True

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self._anim = QPropertyAnimation(self, b"knob_position", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def _get_knob_position(self) -> float:
        return self._knob_x

    def _set_knob_position(self, val: float):
        self._knob_x = val
        self.update()

    knob_position = pyqtProperty(float, _get_knob_position, _set_knob_position)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, on: bool):
        if on == self._checked:
            return
        self._checked = on
        self._animate()
        self.toggled.emit(on)

    def setEnabled(self, enabled: bool):
        self._enabled = enabled
        self.setCursor(Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def isEnabled(self) -> bool:
        return self._enabled

    def _off_x(self) -> float:
        return float(self._KNOB_MARGIN)

    def _on_x(self) -> float:
        return float(self._TRACK_W - self._KNOB_MARGIN - self._KNOB_D)

    def _animate(self):
        self._anim.stop()
        self._anim.setStartValue(self._knob_x)
        self._anim.setEndValue(self._on_x() if self._checked else self._off_x())
        self._anim.start()

    def mousePressEvent(self, ev):
        if self._enabled and ev.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def sizeHint(self):
        text_w = self.fontMetrics().horizontalAdvance(self._text) if self._text else 0
        gap = 10 if self._text else 0
        return QSize(self._TRACK_W + gap + text_w, 30)

    def paintEvent(self, ev):
        # colours are read at paint time so a theme switch applies immediately
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        accent = QColor(COLORS["accent"])
        bg_off = QColor(COLORS["bg_elevated"])
        knob_color = QColor("#ffffff")
        if not self._enabled:
            accent = QColor(COLORS["text_dim"])
            bg_off = QColor(COLORS["bg_input"])
            knob_color = QColor(COLORS["text_dim"])

        track_rect = QRectF(0, (self.height() - self._TRACK_H) / 2, self._TRACK_W, self._TRACK_H)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(accent if self._checked else bg_off))
        p.drawRoundedRect(track_rect, self._TRACK_H / 2, self._TRACK_H / 2)

        knob_y = (self.height() - self._KNOB_D) / 2
        p.setBrush(QBrush(knob_color))
        p.drawEllipse(QPointF(self._knob_x + self._KNOB_D / 2, knob_y + self._KNOB_D / 2),
                      self._KNOB_D / 2, self._KNOB_D / 2)

        if self._text:
            p.setPen(QPen(QColor(COLORS["text"] if self._enabled else COLORS["text_dim"])))
            text_x = self._TRACK_W + 10
            p.drawText(int(text_x), 0, self.width() - int(text_x), self.height(),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._text)
        p.end()


class LossChart(QWidget):
    """Small live line chart of training loss per epoch (theme-aware)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pts: list[tuple[int, float]] = []
        self._total = 1
        self.setMinimumHeight(170)

    def clear(self, total_epochs: int = 1):
        self._pts, self._total = [], max(1, total_epochs)
        self.update()

    def add_point(self, epoch: int, loss: float):
        if loss > 0:
            self._pts.append((epoch, loss))
            self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        ml, mr, mt, mb = 52, 14, 12, 26
        p.fillRect(self.rect(), QColor(COLORS["bg_input"]))
        p.setPen(QPen(QColor(COLORS["plot_grid"]), 1))
        for i in range(5):
            y = mt + (h - mt - mb) * i / 4
            p.drawLine(ml, int(y), w - mr, int(y))
        p.setPen(QColor(COLORS["text_dim"]))
        if not self._pts:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Loss per epoch appears here while training")
            p.end()
            return
        losses = [l for _, l in self._pts]
        lo, hi = min(losses), max(losses)
        if hi - lo < 1e-9:
            lo, hi = lo * 0.9, hi * 1.1 + 1e-9
        X = lambda e: ml + (w - ml - mr) * (e / self._total)
        Y = lambda v: mt + (h - mt - mb) * (1 - (v - lo) / (hi - lo))
        for i in range(5):
            v = hi - (hi - lo) * i / 4
            p.drawText(QRectF(0, Y(v) - 8, ml - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{v:.3g}")
        p.drawText(QRectF(ml, h - mb + 4, w - ml - mr, 18), Qt.AlignmentFlag.AlignCenter,
                   f"epoch (of {self._total})")
        p.setPen(QPen(QColor(COLORS["accent"]), 2))
        pts = [QPointF(X(e), Y(v)) for e, v in self._pts]
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        p.setBrush(QColor(COLORS["accent"]))
        for pt in pts:
            p.drawEllipse(pt, 3, 3)
        p.end()


class ActionBar(QFrame):
    """Bar pinned to the bottom of a page: status text, extras, Start and Stop."""

    def __init__(self, start_text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ActionBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(36, 12, 36, 12)
        lay.setSpacing(12)
        self.content = QVBoxLayout()
        self.content.setSpacing(6)
        lay.addLayout(self.content, 1)
        self.extras = QHBoxLayout()
        self.extras.setSpacing(8)
        lay.addLayout(self.extras)
        self.start_btn = variant(QPushButton(start_text), "primary")
        self.start_btn.setMinimumWidth(170)
        self.stop_btn = variant(QPushButton("Stop"), "danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumWidth(90)
        lay.addWidget(self.start_btn)
        lay.addWidget(self.stop_btn)


def page_with_action_bar(page: QWidget, bar: ActionBar) -> QVBoxLayout:
    """Layout for a page widget: scrollable content above a pinned action bar."""
    root = QVBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    root.addWidget(scrollable(page), 1)
    root.addWidget(bar)
    return root


# ═══════════════════════════════════════════════════════════════════════════
# Error dialog
# ═══════════════════════════════════════════════════════════════════════════

def show_error(parent, title: str, message: str, details: str = ""):
    """Plain-language error, with the technical details one click away."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if details:
        box.setInformativeText("Click “Show Details…” for the technical error, "
                               "or “Copy details” to paste it into a bug report.")
        box.setDetailedText(details)
        copy_btn = box.addButton("Copy details", QMessageBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(f"{message}\n\n{details}"))
    box.addButton(QMessageBox.StandardButton.Ok)
    box.exec()


def friendly_error(exc_text: str) -> str:
    """Turn common technical errors into advice a user can act on."""
    t = exc_text.lower()
    if "out of memory" in t:
        return "The GPU ran out of memory. Close other GPU programs or process fewer / smaller images."
    if "no such file" in t or "not found" in t or "filenotfound" in t:
        return "A file or folder could not be found. Check the paths and try again."
    if "permission" in t:
        return "S2L is not allowed to write there. Choose another output folder or close files open in Excel."
    if "does not match" in t and "shape" in t:
        return "A mask and its image have different sizes. Make sure each mask belongs to its image."
    return exc_text.splitlines()[0] if exc_text else "An unexpected error occurred."


# ═══════════════════════════════════════════════════════════════════════════
# Model picker (Segmentation and Training)
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM = "__custom__"


class ModelPicker(QWidget):
    """Dropdown of pretrained models + "Custom (my trained model)".

    Choosing Custom reveals a second row: a list of the user's trained models
    for this engine (from the Training page) and a Browse button for a model
    trained elsewhere. ``value()`` returns ``(kind, value)`` with kind
    ``"pretrained"`` (value = model name), ``"custom"`` (value = path) or one of
    the *extra* keys (e.g. ``("scratch", None)``).
    """
    changed = pyqtSignal()

    def __init__(self, engine: str, label: str = "Model", extra: list[tuple[str, str, str]] | None = None,
                 settings_key: str = "", parent=None):
        from PyQt6.QtWidgets import QComboBox
        from s2l.core.models_info import MODEL_DESCRIPTIONS, PRETRAINED
        super().__init__(parent)
        self.engine = engine
        self._key = settings_key
        self._desc = dict(MODEL_DESCRIPTIONS)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.combo = QComboBox()
        for key, text, desc in extra or []:
            self.combo.addItem(text, key)
            self._desc[key] = desc
        for key, text in PRETRAINED[engine].items():
            self.combo.addItem(f"{key} — {text}" if key != text else key, key)
        self.combo.addItem("Custom (my trained model)", CUSTOM)
        self._desc[CUSTOM] = "A model you trained, from the list below (or Browse for one trained elsewhere)."
        lay.addLayout(field_row(label, self.combo))
        self.hint = QLabel("")
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        # second row, only shown for Custom: my trained models + Browse
        self._custom_wrap = QWidget()
        cl = QVBoxLayout(self._custom_wrap)
        cl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.mine = QComboBox()
        self.mine.setMinimumWidth(240)
        row.addWidget(self.mine, 1)
        browse = QPushButton("Browse")
        browse.setMinimumWidth(90)
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        cl.addLayout(field_row("My trained model", row))
        lay.addWidget(self._custom_wrap)
        self._custom_wrap.hide()

        self.refresh()
        self.combo.currentIndexChanged.connect(self._on_change)
        self.mine.currentIndexChanged.connect(self._on_change)
        self._restore()
        self._on_change()

    # ── list of the user's models ─────────────────────────────────────────

    def refresh(self):
        """Reload the user's trained models (keeps the current choice)."""
        keep = self.mine.currentData()
        self.mine.blockSignals(True)
        self.mine.clear()
        for m in trained_models(self.engine):
            # (no per-item ToolTipRole data: with this PyQt6 build it corrupted the heap after
            #  CUDA training in the same process; the combo's own tooltip shows the path instead)
            self.mine.addItem(f"{m['name']}  ({m.get('date', '')})", m["path"])
        if keep and self.mine.findData(keep) < 0 and os.path.exists(keep):
            self._add_path(keep)
        missing = bool(keep) and self.mine.findData(keep) < 0
        if missing:
            # never switch silently to another model: show that the chosen one is gone
            self.mine.insertItem(0, f"⚠ Missing: {os.path.basename(os.path.normpath(keep))} — choose another", None)
        if not self.mine.count():
            self.mine.addItem("No trained models yet — train one, or Browse", None)
        idx = 0 if missing else (self.mine.findData(keep) if keep else -1)
        self.mine.setCurrentIndex(max(0, idx))
        self.mine.blockSignals(False)
        if missing and hasattr(self, "_custom_wrap"):
            self._on_change()

    def _add_path(self, path: str):
        if self.mine.count() == 1 and self.mine.itemData(0) is None:
            self.mine.clear()
        name = os.path.basename(os.path.normpath(path))
        self.mine.addItem(f"{name}  (browsed)", path)

    def _browse(self):
        from PyQt6.QtWidgets import QFileDialog
        if self.engine == "stardist":
            p = QFileDialog.getExistingDirectory(self, "Select StarDist model folder (contains config.json)")
            if p and not os.path.isfile(os.path.join(p, "config.json")):
                QMessageBox.warning(self, "Not a StarDist model", f"No config.json in:\n{p}")
                return
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Select trained Cellpose model", "", "All files (*)")
        if p:
            i = self.mine.findData(p)
            if i < 0:
                self._add_path(p)
                i = self.mine.count() - 1
            self.mine.setCurrentIndex(i)

    # ── value ─────────────────────────────────────────────────────────────

    def value(self) -> tuple[str, str | None]:
        key = self.combo.currentData()
        if key == CUSTOM:
            return "custom", self.mine.currentData()
        from s2l.core.models_info import PRETRAINED
        if key in PRETRAINED[self.engine]:
            return "pretrained", key
        return key, None

    def set_custom(self, path: str):
        self.refresh()
        i = self.mine.findData(path)
        if i < 0:
            self._add_path(path)
            i = self.mine.count() - 1
        self.mine.setCurrentIndex(i)
        self.combo.setCurrentIndex(self.combo.findData(CUSTOM))

    def set_choice(self, key: str):
        i = self.combo.findData(key)
        if i >= 0:
            self.combo.setCurrentIndex(i)

    def _on_change(self, *_):
        from s2l.core.models_info import MODEL_SIZES_MB, PRETRAINED, is_downloaded
        key = self.combo.currentData()
        self._custom_wrap.setVisible(key == CUSTOM)
        self.mine.setToolTip(self.mine.currentData() or "")
        text = self._desc.get(key, "")
        if key in PRETRAINED[self.engine] and not is_downloaded(self.engine, key):
            text += f"  Not downloaded yet (~{MODEL_SIZES_MB.get(key, 5)} MB) — it is downloaded on first use."
        self.hint.setText(text)
        if self._key:
            kind, val = self.value()
            settings().setValue(f"models/pick/{self._key}", f"{key}|{val or ''}")
        self.changed.emit()

    def _restore(self):
        if not self._key:
            return
        saved = settings().value(f"models/pick/{self._key}", "", type=str)
        key, _, path = saved.partition("|")
        if key == CUSTOM and path and os.path.exists(path):
            self.set_custom(path)
        elif key:
            self.set_choice(key)
