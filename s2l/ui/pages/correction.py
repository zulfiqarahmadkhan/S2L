"""Correction page: browse masks with thumbnails and open the manual correction editor."""
import logging
import os
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from s2l.core.image_files import MASK_SUFFIX, is_derived, list_images, list_masks, original_stem
from s2l.ui.theme import COLORS
from s2l.ui.widgets import (
    browse_row, card, field_row, page_shell, remember_text, scrollable, section_header,
    show_error, variant,
)

logger = logging.getLogger(__name__)


class _PairThumbLoader(QThread):
    """Renders image + mask thumbnails off the UI thread (QImage only; no QPixmap here)."""
    thumb_ready = pyqtSignal(int, QImage)

    def __init__(self, pairs: list[tuple[str, str | None]], size: int):
        super().__init__()
        self._pairs = pairs
        self._size = size
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from s2l.ui.manual_correction import (
            load_mask, ndarray_to_qimage, overlay_mask, render_pair_thumbnail,
        )
        import cv2
        import numpy as np
        for idx, (mask_path, image_path) in enumerate(self._pairs):
            if self._cancel:
                return
            try:
                if image_path:
                    rgb = render_pair_thumbnail(image_path, mask_path, self._size)
                else:
                    # no image known yet: show the mask alone on black
                    m = load_mask(mask_path)
                    f = self._size / max(m.shape)
                    m = cv2.resize(m.astype(np.float32), (max(1, int(m.shape[1] * f)), max(1, int(m.shape[0] * f))),
                                   interpolation=cv2.INTER_NEAREST).astype(np.int32)
                    rgb = overlay_mask(np.zeros((*m.shape, 3), np.uint8), m, 1.0)
                self.thumb_ready.emit(idx, ndarray_to_qimage(rgb))
            except Exception as e:
                logger.warning(f"Thumbnail failed for {mask_path}: {e}")


class CorrectionPage(QWidget):
    """Pick any mask + original image pair and open the correction editor."""

    _THUMB = 150
    _PREVIEW = 420

    def __init__(self):
        super().__init__()
        self._known_originals: dict[str, str] = {}   # stem -> path, from the last run
        self._chosen: dict[str, str] = {}            # mask path -> image picked by the user
        self._pairs: list[tuple[str, str | None]] = []
        self._loader: _PairThumbLoader | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        page, col = page_shell("Manual Correction",
                               "Fix segmentation masks by hand: add missed cells, erase wrong ones.")

        # ── card: folders ─────────────────────────────────────────────────
        src_lay = QVBoxLayout()
        src_lay.addLayout(section_header(
            "Folders", f"Folder with *{MASK_SUFFIX}.tif files, and where the original images are."))
        src_lay.addSpacing(4)
        self.mask_dir_edit = QLineEdit()
        self.mask_dir_edit.setPlaceholderText("Folder containing mask files…")
        self.mask_dir_edit.editingFinished.connect(self._refresh)
        src_lay.addLayout(field_row(
            "Mask folder", browse_row(self.mask_dir_edit, "Browse", self._pick_mask_dir)))

        self.orig_dir_edit = QLineEdit()
        self.orig_dir_edit.setPlaceholderText("Optional — defaults to the mask folder")
        self.orig_dir_edit.editingFinished.connect(self._refresh)
        src_lay.addLayout(field_row(
            "Original images folder", browse_row(self.orig_dir_edit, "Browse", self._pick_orig_dir),
            hint="Images are matched to masks by file name. If a match is missing or wrong, "
                 "select the mask below and use “Choose image…”.",
        ))
        col.addWidget(card(src_lay))

        # ── card: thumbnails + preview ────────────────────────────────────
        list_lay = QVBoxLayout()
        list_lay.addLayout(section_header(
            "Mask files", "Click a mask to preview it; double-click to open the editor."))

        body = QHBoxLayout()
        body.setSpacing(16)
        self.mask_list = QListWidget()
        self.mask_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.mask_list.setIconSize(QSize(self._THUMB, self._THUMB))
        self.mask_list.setGridSize(QSize(self._THUMB + 24, self._THUMB + 44))
        self.mask_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.mask_list.setMovement(QListWidget.Movement.Static)
        self.mask_list.setWordWrap(True)
        self.mask_list.setMinimumHeight(420)
        self.mask_list.currentRowChanged.connect(self._show_preview)
        self.mask_list.itemDoubleClicked.connect(lambda _: self._open_selected())
        body.addWidget(self.mask_list, 3)

        prev_col = QVBoxLayout()
        prev_col.setSpacing(8)
        self._preview = QLabel("Select a mask")
        self._preview.setObjectName("ImagePanel")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setFixedSize(self._PREVIEW, self._PREVIEW)
        prev_col.addWidget(self._preview)
        self._pair_lbl = QLabel("")
        self._pair_lbl.setObjectName("Hint")
        self._pair_lbl.setWordWrap(True)
        self._pair_lbl.setMaximumWidth(self._PREVIEW)
        prev_col.addWidget(self._pair_lbl)
        btns = QHBoxLayout()
        choose_btn = QPushButton("Choose image…")
        choose_btn.setToolTip("Pick the original image for the selected mask")
        choose_btn.clicked.connect(self._choose_image_for_selected)
        open_btn = variant(QPushButton("Open editor"), "primary")
        open_btn.clicked.connect(self._open_selected)
        btns.addWidget(choose_btn)
        btns.addWidget(open_btn)
        prev_col.addLayout(btns)
        refresh_btn = QPushButton("Refresh list")
        refresh_btn.clicked.connect(self._refresh)
        prev_col.addWidget(refresh_btn)
        prev_col.addStretch()
        body.addLayout(prev_col)
        list_lay.addLayout(body)
        col.addWidget(card(list_lay))

        # ── card: explicit pair ───────────────────────────────────────────
        pair_lay = QVBoxLayout()
        pair_lay.addLayout(section_header(
            "Open a specific pair", "Pick any mask file and the image it was made from."))
        pair_lay.addSpacing(4)
        self.pair_mask_edit = QLineEdit()
        self.pair_mask_edit.setPlaceholderText(f"Mask file (*{MASK_SUFFIX}.tif)…")
        pair_lay.addLayout(field_row(
            "Mask file", browse_row(self.pair_mask_edit, "Browse", self._pick_pair_mask)))
        self.pair_image_edit = QLineEdit()
        self.pair_image_edit.setPlaceholderText("Original image file…")
        pair_lay.addLayout(field_row(
            "Image file", browse_row(self.pair_image_edit, "Browse", self._pick_pair_image)))
        pair_open = variant(QPushButton("Open editor"), "primary")
        pair_open.clicked.connect(self._open_pair)
        pair_lay.addWidget(pair_open)
        col.addWidget(card(pair_lay))

        col.addStretch()
        root.addWidget(scrollable(page))
        remember_text(self.mask_dir_edit, "corr/masks")
        remember_text(self.orig_dir_edit, "corr/images")
        if self.mask_dir_edit.text():
            self._refresh()

    # ── public ────────────────────────────────────────────────────────────

    def set_source(self, mask_dir: str, originals: list[str]):
        """Called after a segmentation run with its mask folder and input images."""
        self._known_originals = {Path(p).stem: p for p in originals}
        self.mask_dir_edit.setText(mask_dir)
        self.orig_dir_edit.clear()
        self._refresh()

    # ── folder / file pickers ─────────────────────────────────────────────

    def _pick_mask_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select mask folder")
        if d:
            self.mask_dir_edit.setText(d)
            self._refresh()

    def _pick_orig_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select original images folder")
        if d:
            self.orig_dir_edit.setText(d)
            self._refresh()

    def _ask_image(self, mask_path: str, start_dir: str = "") -> str | None:
        start = start_dir or self.orig_dir_edit.text().strip() or str(Path(mask_path).parent)
        image_path, _ = QFileDialog.getOpenFileName(
            self, f"Select the original image for {Path(mask_path).name}", start,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All (*)")
        if not image_path:
            return None
        if is_derived(image_path):
            QMessageBox.warning(
                self, "That looks like a mask",
                f"{Path(image_path).name} looks like a mask or overlay file, not an original image.\n"
                "Pick the raw image the mask was made from.")
            return None
        return image_path

    def _pick_pair_mask(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select mask file", self.mask_dir_edit.text().strip(), "TIFF (*.tif *.tiff);;All (*)")
        if p:
            self.pair_mask_edit.setText(p)
            guess = self._originals().get(original_stem(p)) if p.endswith(f"{MASK_SUFFIX}.tif") else None
            if guess and not self.pair_image_edit.text().strip():
                self.pair_image_edit.setText(guess)

    def _pick_pair_image(self):
        mask = self.pair_mask_edit.text().strip()
        p = self._ask_image(mask or "mask", str(Path(mask).parent) if mask else "")
        if p:
            self.pair_image_edit.setText(p)

    # ── list ──────────────────────────────────────────────────────────────

    def _originals(self) -> dict[str, str]:
        found = dict(self._known_originals)
        for folder in (self.orig_dir_edit.text().strip(), self.mask_dir_edit.text().strip()):
            if folder and os.path.isdir(folder):
                for p in list_images(folder):
                    found.setdefault(Path(p).stem, p)
        return found

    def _refresh(self):
        if self._loader is not None:
            self._loader.cancel()
            self._loader.wait()
        self.mask_list.clear()
        self._pairs = []
        self._preview.setPixmap(QPixmap())
        self._preview.setText("Select a mask")
        self._pair_lbl.clear()
        mask_dir = self.mask_dir_edit.text().strip()
        if not mask_dir or not os.path.isdir(mask_dir):
            return
        originals = self._originals()
        placeholder = QPixmap(self._THUMB, self._THUMB)
        placeholder.fill(QColor(COLORS["bg_input"]))
        for mp in list_masks(mask_dir):
            orig = self._chosen.get(str(mp)) or originals.get(original_stem(mp))
            self._pairs.append((str(mp), orig))
            item = QListWidgetItem(QIcon(placeholder), ("" if orig else "? ") + original_stem(mp))
            item.setToolTip(f"{mp.name}\nImage: {orig or 'not found — use Choose image…'}")
            self.mask_list.addItem(item)
        if not self._pairs:
            self._preview.setText(f"No *{MASK_SUFFIX}.tif files in this folder.")
            return
        self._loader = _PairThumbLoader(self._pairs, self._THUMB)
        self._loader.thumb_ready.connect(self._on_thumb)
        self._loader.start()
        self.mask_list.setCurrentRow(0)

    def _on_thumb(self, idx: int, qimg: QImage):
        item = self.mask_list.item(idx)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(qimg)))

    def _show_preview(self, row: int):
        if not 0 <= row < len(self._pairs):
            return
        mask_path, image_path = self._pairs[row]
        self._pair_lbl.setText(
            f"Mask: {Path(mask_path).name}\n"
            f"Image: {Path(image_path).name if image_path else 'not found — use Choose image…'}")
        if not image_path:
            self._preview.setPixmap(QPixmap())
            self._preview.setText("No original image matched.\nUse “Choose image…”.")
            return
        try:
            from s2l.ui.manual_correction import ndarray_to_qimage, render_pair_thumbnail
            rgb = render_pair_thumbnail(image_path, mask_path, self._PREVIEW)
            self._preview.setPixmap(QPixmap.fromImage(ndarray_to_qimage(rgb)))
        except Exception as e:
            self._preview.setPixmap(QPixmap())
            self._preview.setText(f"Cannot preview this pair:\n{e}")

    def _choose_image_for_selected(self):
        row = self.mask_list.currentRow()
        if not 0 <= row < len(self._pairs):
            QMessageBox.information(self, "No mask", "Select a mask first.")
            return
        mask_path, _ = self._pairs[row]
        image_path = self._ask_image(mask_path)
        if image_path:
            self._chosen[mask_path] = image_path
            self._refresh()
            self.mask_list.setCurrentRow(row)

    # ── opening the editor ────────────────────────────────────────────────

    def _open_selected(self):
        row = self.mask_list.currentRow()
        if not 0 <= row < len(self._pairs):
            QMessageBox.information(self, "No mask", "Select a mask file in the list first.")
            return
        mask_path, image_path = self._pairs[row]
        if not image_path:
            image_path = self._ask_image(mask_path)
            if not image_path:
                return
            self._chosen[mask_path] = image_path
        # every pair with a known image, so Previous / Next can walk the folder
        pairs = [(m, i) for m, i in self._pairs if i or m == mask_path]
        pairs = [(m, image_path if m == mask_path else i) for m, i in pairs]
        index = next(k for k, (m, _) in enumerate(pairs) if m == mask_path)
        self._open(mask_path, image_path, pairs, index)
        self._refresh()
        self.mask_list.setCurrentRow(row)

    def _open_pair(self):
        mask_path = self.pair_mask_edit.text().strip()
        image_path = self.pair_image_edit.text().strip()
        if not os.path.isfile(mask_path) or not os.path.isfile(image_path):
            QMessageBox.warning(self, "Missing file", "Choose both an existing mask file and image file.")
            return
        self._open(mask_path, image_path)

    def _open(self, mask_path: str, image_path: str, pairs=None, index: int = 0):
        from s2l.ui.manual_correction import ManualCorrectionDialog
        try:
            dlg = ManualCorrectionDialog(image_path, mask_path, self, pairs=pairs, index=index)
        except Exception as e:
            show_error(self, "Cannot open", "Could not open this mask + image pair.", str(e))
            return
        dlg.exec()
