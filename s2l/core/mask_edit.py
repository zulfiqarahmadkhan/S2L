"""Label-mask editing operations used by the manual correction editor.

All functions work in place on an integer label mask (0 = background) and are
plain numpy / OpenCV / SciPy, so they can be tested without the GUI.
``overwrite=False`` means existing cells are never painted over.
"""
from __future__ import annotations

import numpy as np


def next_label(mask: np.ndarray) -> int:
    return int(mask.max()) + 1


def _paintable(region: np.ndarray, label: int, overwrite: bool) -> np.ndarray:
    return np.ones(region.shape, bool) if overwrite else (region == 0) | (region == label)


def paint_stroke(mask, p0, p1, radius, label, overwrite=False):
    """Round brush from p0 to p1 (x, y), joining the points so fast drags leave no gaps."""
    import cv2
    h, w = mask.shape
    x0, x1 = max(0, min(p0[0], p1[0]) - radius - 1), min(w, max(p0[0], p1[0]) + radius + 2)
    y0, y1 = max(0, min(p0[1], p1[1]) - radius - 1), min(h, max(p0[1], p1[1]) + radius + 2)
    if x1 <= x0 or y1 <= y0:
        return
    stroke = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.line(stroke, (p0[0] - x0, p0[1] - y0), (p1[0] - x0, p1[1] - y0), 1, thickness=2 * radius + 1)
    cv2.circle(stroke, (p1[0] - x0, p1[1] - y0), radius, 1, thickness=-1)
    region = mask[y0:y1, x0:x1]
    if label == 0:                      # partial eraser
        region[stroke > 0] = 0
    else:
        region[(stroke > 0) & _paintable(region, label, overwrite)] = label


def fill_polygon(mask, points, label, overwrite=False) -> int:
    """Fill a closed outline [(x, y), ...]; returns the number of pixels set."""
    import cv2
    if len(points) < 3:
        return 0
    pts = np.array(points, np.int32)
    x0, y0 = np.maximum(pts.min(0), 0)
    x1, y1 = np.minimum(pts.max(0) + 1, [mask.shape[1], mask.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return 0
    poly = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(poly, [pts - [x0, y0]], 1)
    region = mask[y0:y1, x0:x1]
    sel = (poly > 0) & _paintable(region, label, overwrite)
    region[sel] = label
    return int(sel.sum())


def delete_cell(mask, x, y) -> int:
    lab = int(mask[y, x])
    if lab:
        mask[mask == lab] = 0
    return lab


def delete_in_rect(mask, x0, y0, x1, y1) -> int:
    """Delete every cell that has pixels inside the rectangle; returns how many."""
    xa, xb = sorted((max(0, x0), max(0, x1)))
    ya, yb = sorted((max(0, y0), max(0, y1)))
    labels = np.unique(mask[ya:yb + 1, xa:xb + 1])
    labels = labels[labels > 0]
    if len(labels):
        mask[np.isin(mask, labels)] = 0
    return len(labels)


def fill_holes_cell(mask, label) -> int:
    from scipy import ndimage as ndi
    sl = ndi.find_objects(mask == label)
    if not sl or sl[0] is None:
        return 0
    sl = sl[0]
    region = mask[sl]
    cell = region == label
    filled = ndi.binary_fill_holes(cell)
    add = filled & ~cell & (region == 0)
    region[add] = label
    return int(add.sum())


def fill_holes_all(mask) -> int:
    from scipy import ndimage as ndi
    total = 0
    for lab, sl in enumerate(ndi.find_objects(mask), start=1):
        if sl is None:
            continue
        region = mask[sl]
        cell = region == lab
        add = ndi.binary_fill_holes(cell) & ~cell & (region == 0)
        region[add] = lab
        total += int(add.sum())
    return total


def magic_wand(mask, intensity, x, y, tolerance, label, overwrite=False, window=300) -> int:
    """Grow a region of similar brightness from (x, y); returns pixels added.

    *intensity* is a float image in [0, 1]. The search is limited to a window
    around the click so a low tolerance on a flat background can't flood the image.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import flood
    h, w = mask.shape
    x0, x1 = max(0, x - window), min(w, x + window + 1)
    y0, y1 = max(0, y - window), min(h, y + window + 1)
    sub = intensity[y0:y1, x0:x1]
    region = mask[y0:y1, x0:x1]
    seed = float(sub[y - y0, x - x0])
    grown = flood(sub, (y - y0, x - x0), tolerance=tolerance)
    # also stop halfway between the clicked brightness and the local background,
    # so the region can't leak through dim halos (bright cells) or bright rims (dark cells)
    bg = float(np.percentile(sub, 10 if seed >= np.median(sub) else 90))
    mid = (seed + bg) / 2
    grown &= (sub >= mid) if seed >= bg else (sub <= mid)
    grown &= _paintable(region, label, overwrite)
    lab_cc, _ = ndi.label(grown)
    keep = lab_cc == lab_cc[y - y0, x - x0] if lab_cc[y - y0, x - x0] else np.zeros_like(grown)
    keep = ndi.binary_fill_holes(keep) & _paintable(region, label, overwrite)
    region[keep] = label
    return int(keep.sum())


def insert_object(mask, obj, x0, y0, label, overwrite=False) -> int:
    """Paste a boolean object (from AI segmentation of a crop) at (x0, y0)."""
    h, w = obj.shape
    region = mask[y0:y0 + h, x0:x0 + w]
    obj = obj[:region.shape[0], :region.shape[1]]
    sel = obj & _paintable(region, label, overwrite)
    region[sel] = label
    return int(sel.sum())


def pick_object(labels, box_in_crop, free=None) -> np.ndarray | None:
    """From a segmented crop choose the object the user meant to add.

    That is the predicted object covering the most *unsegmented* area (*free*,
    True where the current mask is background) inside the box — not simply the
    one under the box centre, which for curved cells is often a neighbour that
    already exists.
    """
    bx0, by0, bx1, by1 = box_in_crop
    inside = labels[by0:by1, bx0:bx1]
    weight = (free[by0:by1, bx0:bx1] if free is not None else np.ones(inside.shape, bool)) & (inside > 0)
    ids, counts = np.unique(inside[weight], return_counts=True)
    if not len(ids):
        return None
    return labels == ids[np.argmax(counts)]


def remove_small(mask, min_size) -> int:
    sizes = np.bincount(mask.ravel())
    small = np.nonzero((sizes < min_size) & (sizes > 0))[0]
    small = small[small > 0]
    if len(small):
        mask[np.isin(mask, small)] = 0
    return len(small)


def remove_border(mask) -> int:
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    labels = np.unique(edge[edge > 0])
    if len(labels):
        mask[np.isin(mask, labels)] = 0
    return len(labels)


def cell_centroids(mask) -> dict[int, tuple[float, float]]:
    """label -> (y, x) centre of each cell."""
    from scipy import ndimage as ndi
    out = {}
    for lab, sl in enumerate(ndi.find_objects(mask), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(mask[sl] == lab)
        if len(ys):
            out[lab] = (sl[0].start + ys.mean(), sl[1].start + xs.mean())
    return out


# ── undo / redo by changed region (keeps memory small on large images) ──────

def diff_region(before, after):
    """Bounding box + old/new pixels of what changed, or None if nothing did."""
    changed = before != after
    if not changed.any():
        return None
    ys, xs = np.nonzero(changed.any(axis=1))[0], np.nonzero(changed.any(axis=0))[0]
    sl = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
    return sl, before[sl].copy(), after[sl].copy()
