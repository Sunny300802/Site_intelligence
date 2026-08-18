"""
core/geometry.py
================
Polygon areas.

A rectangle is not enough for a real camera view. The area you actually
care about is usually an irregular shape - the walkway between desks, the
part of the floor inside a marked boundary. This module handles those.

Coordinates are stored as FRACTIONS of the frame (0-1), so the same
polygon keeps working if the camera resolution or the stream profile
changes.
"""
import numpy as np
import cv2


def to_pixels(polygon, width, height):
    """[(0.1, 0.2), ...] -> integer pixel points for this frame size."""
    pts = []
    for x, y in polygon:
        px = int(x * width) if x <= 1 else int(x)
        py = int(y * height) if y <= 1 else int(y)
        pts.append((px, py))
    return np.array(pts, dtype=np.int32)


def point_inside(polygon_px, x, y):
    """Is this point inside the polygon?"""
    return cv2.pointPolygonTest(polygon_px, (float(x), float(y)), False) >= 0


def build_mask(polygon_px, width, height):
    """A white-on-black mask of the polygon."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_px], 255)
    return mask


def bounding_box(polygon_px):
    x, y, w, h = cv2.boundingRect(polygon_px)
    return x, y, x + w, y + h


def apply_mask(frame, mask, bbox=None, darken=0):
    """Black out everything outside the polygon, then optionally crop to
    its bounding box.

    Doing both is what makes this worth it: the detector stops wasting
    attention on the areas you excluded, AND the people who remain take
    up a larger share of the image it sees, so they are easier to detect.
    """
    if darken:
        out = cv2.addWeighted(frame, 1.0, np.zeros_like(frame), 0.0, 0)
        outside = cv2.bitwise_not(mask)
        out[outside > 0] = (out[outside > 0] * (1 - darken)).astype(out.dtype)
    else:
        out = cv2.bitwise_and(frame, frame, mask=mask)
    if bbox:
        x1, y1, x2, y2 = bbox
        out = out[y1:y2, x1:x2]
    return out


def draw_polygon(frame, polygon_px, colour=(60, 90, 220), thickness=2,
                 label=None):
    cv2.polylines(frame, [polygon_px], isClosed=True, color=colour,
                  thickness=thickness)
    if label:
        x, y = polygon_px[0]
        cv2.putText(frame, label, (int(x) + 6, max(16, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


def polygon_area_fraction(polygon_px, width, height):
    """How much of the frame the polygon covers - useful for reporting
    how much work the mask is saving."""
    area = cv2.contourArea(polygon_px)
    return area / float(width * height)
