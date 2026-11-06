"""Convert segmentation masks to ROI data with visualization.

Computes comprehensive per-cell measurements including morphology,
intensity statistics, shape descriptors, and Feret's diameters.
Overlay plot is generated lazily only if a plot output path is provided.
"""
import logging

import cv2
import numpy as np
import xlsxwriter
from scipy import stats as sp_stats
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


class ROIVisualizer:
    """Quantify ROIs from segmentation masks and optionally render overlays."""

    # Column headers matching your requested list exactly
    HEADERS = [
        "Label",
        "Area",
        "Integrated Density",
        "Mean Gray Value",
        "Standard Deviation",
        "Modal gray value",
        "Centroid",
        "Perimeter",
        "Min & max gray value",
        "Center of mass",
        "Bounding rectangle",
        "Fit ellipse",
        "Shape descriptors",
        "Feret's diameter",
        "Skewness",
        "Area fraction",
        "Median",
        "Kurtosis",
        "Stack position"
    ]

    def __init__(self, label_image_path, original_image_path, excel_output_path,
                 plot_output_path=None, show_labels=False, progress_callback=None):
        self.label_image_path = str(label_image_path)
        self.original_image_path = str(original_image_path)
        self.excel_output_path = str(excel_output_path)
        self.plot_output_path = str(plot_output_path) if plot_output_path else None
        self.show_labels = show_labels
        self.progress_callback = progress_callback

    def save_rois_to_excel(self):
        """Compute ROI stats and write to Excel. Generates overlay plot if path was given."""
        from skimage.io import imread

        try:
            label_img = imread(self.label_image_path)
            orig_img = imread(self.original_image_path)
        except Exception as e:
            raise ValueError(f"Unable to load image(s): {e}") from e

        from s2l.core.image_files import channels_last
        orig_img = channels_last(orig_img)
        if label_img.ndim == 3:
            label_img = label_img[..., 0]
        # Ensure label image is integer type for indexing
        if label_img.dtype.kind not in "iu":
            label_img = label_img.astype(np.int32)

        # Keep the native bit depth (cv2.IMREAD_GRAYSCALE would squash 16-bit to 8-bit)
        gray_img = self._to_gray(orig_img)
        if gray_img.shape != label_img.shape:
            raise ValueError(
                f"Mask shape {label_img.shape} does not match image shape {gray_img.shape}"
            )

        roi_data = self._compute_stats(label_img, gray_img)
        self._write_excel(roi_data)

        if self.plot_output_path:
            self._save_overlay(label_img, self._to_display_bgr(orig_img))

        logger.info(f"ROI data saved to {self.excel_output_path}")
        if self.progress_callback:
            self.progress_callback(100)
        return len(roi_data)

    @staticmethod
    def _to_gray(img):
        """Collapse RGB(A) to one channel as ImageJ does: unweighted (R+G+B)/3."""
        if img.ndim == 3:
            img = img[..., :3].mean(axis=-1)
        return img

    @staticmethod
    def _to_display_bgr(img):
        """8-bit BGR rendering of any image for the overlay PNG."""
        if img.ndim == 3:
            img = img[..., :3]
        if img.dtype != np.uint8:
            f = img.astype(np.float32)
            lo, hi = np.percentile(f, (0.5, 99.5))
            f = np.clip((f - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(f)
            img = (f * 255).astype(np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(label_img, gray_img):
        """Compute per-label stats matching the user's requested list."""
        max_label = label_img.max()
        if max_label == 0:
            return []

        props = regionprops(label_img, intensity_image=gray_img)
        roi_data = []

        for p in props:
            # Extract intensity pixels for this region
            try:
                pixels = p.image_intensity[p.image].astype(np.float64)
            except AttributeError:
                pixels = gray_img[label_img == p.label].astype(np.float64)

            area = p.area
            n = len(pixels)

            # Intensity statistics
            mean_val = float(np.mean(pixels)) if n > 0 else 0.0
            std_val = float(np.std(pixels)) if n > 0 else 0.0
            median_val = float(np.median(pixels)) if n > 0 else 0.0
            min_val = float(np.min(pixels)) if n > 0 else 0.0
            max_val = float(np.max(pixels)) if n > 0 else 0.0

            # Modal gray value
            if n > 0:
                vals, counts = np.unique(pixels, return_counts=True)
                modal_val = float(vals[np.argmax(counts)])
            else:
                modal_val = 0.0

            # Integrated Density = Mean * Area
            int_den = mean_val * area

            # Centroid
            cy, cx = p.centroid
            centroid_str = f"({round(float(cx), 2)}, {round(float(cy), 2)})"

            # Center of mass (intensity-weighted centroid)
            try:
                com_y, com_x = p.centroid_weighted
            except AttributeError:
                com_y, com_x = cy, cx
            com_str = f"({round(float(com_x), 2)}, {round(float(com_y), 2)})"

            # Perimeter
            perimeter = p.perimeter

            # Bounding rectangle
            min_row, min_col, max_row, max_col = p.bbox
            bx = min_col
            by = min_row
            bwidth = max_col - min_col
            bheight = max_row - min_row
            bbox_area = bwidth * bheight
            bbox_str = f"x={bx}, y={by}, w={bwidth}, h={bheight}"

            # Fit ellipse
            major = p.axis_major_length
            minor = p.axis_minor_length
            # ImageJ convention: degrees from the x-axis, counter-clockwise, 0-180
            ellipse_angle = float((90.0 + np.degrees(p.orientation)) % 180.0)
            ellipse_str = f"major={round(float(major), 2)}, minor={round(float(minor), 2)}, angle={round(ellipse_angle, 1)}"

            # Shape descriptors
            circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
            solidity = p.solidity
            shape_str = f"circularity={round(float(circularity), 3)}, solidity={round(float(solidity), 3)}"

            # Feret's diameter
            try:
                feret_diam = p.feret_diameter_max
            except AttributeError:
                feret_diam = 0.0

            # Skewness & Kurtosis
            # flat cells (common in dim fluorescence) have no defined skew / kurtosis
            if n > 2 and std_val > 0:
                skewness = float(sp_stats.skew(pixels))
                kurtosis = float(sp_stats.kurtosis(pixels))
            else:
                skewness = 0.0
                kurtosis = 0.0

            # Area fraction
            area_fraction = (area / bbox_area) if bbox_area > 0 else 0.0

            # Stack position
            stack_pos = 1

            # Combined min & max gray value string
            min_max_str = f"{int(min_val)}-{int(max_val)}"

            # Build row matching HEADERS order
            roi_data.append([
                int(p.label),
                int(area),
                round(int_den, 4),
                round(mean_val, 4),
                round(std_val, 4),
                round(modal_val, 4),
                centroid_str,
                round(float(perimeter), 4),
                min_max_str,
                com_str,
                bbox_str,
                ellipse_str,
                shape_str,
                round(float(feret_diam), 4),
                round(float(skewness), 4),
                round(float(area_fraction), 4),
                round(float(median_val), 4),
                round(float(kurtosis), 4),
                int(stack_pos),
            ])

        return roi_data

    # ------------------------------------------------------------------
    # Excel output
    # ------------------------------------------------------------------

    def _write_excel(self, roi_data):
        # nan_inf_to_errors: any undefined value becomes #NUM! instead of aborting the sheet
        with xlsxwriter.Workbook(self.excel_output_path,
                                 {"constant_memory": True, "nan_inf_to_errors": True}) as wb:
            ws = wb.add_worksheet()
            hdr_fmt = wb.add_format({"bold": True, "bg_color": "#D7E4BC"})
            ws.write_row(0, 0, self.HEADERS, hdr_fmt)
            for row, data in enumerate(roi_data, start=1):
                ws.write_row(row, 0, data)

    # ------------------------------------------------------------------
    # Overlay visualisation
    # ------------------------------------------------------------------

    def _save_overlay(self, label_img, color_img):
        """Generate and save an ROI overlay image using OpenCV (no matplotlib)."""
        max_label = int(label_img.max())
        if max_label == 0:
            _imwrite(self.plot_output_path, color_img)
            return

        # Generate random colors for all labels at once
        rng = np.random.RandomState(42)
        colors = rng.randint(0, 256, (max_label + 1, 3), dtype=np.uint8)
        colors[0] = 0  # background = black

        # Vectorized color lookup — no per-label loop
        overlay = colors[label_img]

        # Blend
        alpha = 0.3
        blended = cv2.addWeighted(color_img, 1 - alpha, overlay, alpha, 0)

        # Add label text if requested
        if self.show_labels:
            try:
                from scipy.ndimage import center_of_mass
                centers = center_of_mass(
                    np.ones_like(label_img), label_img, range(1, max_label + 1)
                )
                for lbl, (cy, cx) in enumerate(centers, start=1):
                    if not np.isnan(cy):
                        cv2.putText(blended, str(lbl), (int(cx), int(cy)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1, cv2.LINE_AA)
            except ImportError:
                pass

        _imwrite(self.plot_output_path, blended)


def _imwrite(path: str, img):
    """cv2.imwrite fails silently on non-ASCII paths on Windows (e.g. "Umeå"); encode, then write."""
    from pathlib import Path
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        raise OSError(f"Could not encode the overlay image {path}")
    buf.tofile(path)
