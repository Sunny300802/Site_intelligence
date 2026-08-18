"""
core/boxes.py
=============
Pure box geometry - no model, no GPU, no heavy imports.

Kept separate so it can be reasoned about and tested on its own; these
helpers decide whether two detections are "the same person", which is
what stops duplicate counts.
"""
import numpy as np


def iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def centre(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def height(box):
    return max(1, box[3] - box[1])


def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression. Returns the indices to keep.

    Used to merge the duplicate boxes that appear when the same person
    is found by both the full-frame pass and an overlapping tile.
    """
    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = s.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        overlap = np.where(union > 0, inter / union, 0)
        order = rest[overlap <= iou_threshold]
    return keep


def tile_windows(width, height_px, grid, overlap):
    """Overlapping (x1, y1, x2, y2) windows that cover the whole frame."""
    rows, cols = grid
    tw = int(width / cols)
    th = int(height_px / rows)
    ox = int(tw * overlap)
    oy = int(th * overlap)
    windows = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, c * tw - ox)
            y1 = max(0, r * th - oy)
            x2 = min(width, (c + 1) * tw + ox)
            y2 = min(height_px, (r + 1) * th + oy)
            windows.append((x1, y1, x2, y2))
    return windows


def choose_tile_grid(width, height_px, grid="auto"):
    """Pick a tile layout for a frame.

    "auto" reasons about the frame shape. Splitting a wide frame
    left/right produces near-square tiles, which gives the same
    magnification as a full 2x2 grid for fewer passes. Tall frames get
    the opposite treatment.
    """
    if grid != "auto":
        return tuple(grid)
    aspect = width / max(1, height_px)
    if aspect >= 1.8:
        return (1, 2)          # wide: split left/right
    if aspect <= 0.6:
        return (2, 1)          # tall: split top/bottom
    return (2, 2)              # roughly square: quarters
