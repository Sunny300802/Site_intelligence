"""
core/face_align.py
==================
Warp a detected face onto the fixed 112x112 template every ArcFace-family
recognition model - AdaFace included - was trained on.

Why this is not optional
------------------------
A recognition model does not "look at a face". It looks at a 112x112
grid and expects the left eye at (38.3, 51.7), the right eye at
(73.5, 51.5), the nose at (56.0, 71.7) and the mouth corners at
(41.5, 92.4) and (70.7, 92.2). Everything it learned - which pixels
carry identity, which carry pose - is expressed in those coordinates.

Hand it a raw crop instead and the eyes land wherever the person's head
happened to be tilted. The model still returns a confident 512-number
answer; it is simply an answer about a different geometry, and two
photographs of the SAME person taken at different angles end up further
apart than two different people photographed the same way. That is the
mechanism behind "recognition works in the enrollment photos and fails
on the camera", and it is why the old pipeline's accuracy depended so
heavily on people walking straight at the lens.

Aligning first removes head tilt, in-plane rotation and scale from the
comparison entirely, and leaves the model comparing only what it was
trained to compare.

How the transform is found
--------------------------
A SIMILARITY transform (rotation + uniform scale + translation, no
shear) fitted to the five landmarks by the Umeyama least-squares method.
Similarity rather than affine on purpose: an affine fit will happily
stretch a profile face until its landmarks match the frontal template,
which fabricates a frontal face that was never photographed. A
similarity transform cannot do that - it can only rotate and scale - so
a profile stays a profile and the QUALITY filter (core/face_quality.py)
gets an honest picture to reject.
"""
import numpy as np
import cv2

from config.settings import FACE_ALIGN_SIZE

# The ArcFace/AdaFace five-point template, defined at 112x112 and scaled
# from there. These exact numbers are part of the model contract - they
# are not a tuning parameter.
ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],     # left eye
    [73.5318, 51.5014],     # right eye
    [56.0252, 71.7366],     # nose tip
    [41.5493, 92.3655],     # left mouth corner
    [70.7299, 92.2041],     # right mouth corner
], dtype=np.float32)


def template(image_size=None):
    """The five destination points for a given output size."""
    size = int(image_size or FACE_ALIGN_SIZE)
    return ARCFACE_TEMPLATE_112 * (size / 112.0)


def _umeyama(src, dst):
    """Least-squares similarity transform mapping src -> dst.

    Returns a 2x3 matrix suitable for cv2.warpAffine, or None if the
    landmarks are degenerate (all five on top of each other, which does
    happen on a 12-pixel face).

    This is the Umeyama (1991) estimator, written out rather than pulled
    from scikit-image so that the project does not gain a dependency for
    twenty lines of linear algebra.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape or src.shape[0] < 2:
        return None
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        return None

    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    src_var = (src_demean ** 2).sum() / n
    if src_var < 1e-8:
        return None                       # every landmark in one place

    cov = dst_demean.T @ src_demean / n
    u, s, vt = np.linalg.svd(cov)

    # Guard against the reflection the SVD is free to choose. Without
    # this, a noisy landmark set can produce a MIRRORED face, which the
    # recognition model then compares against un-mirrored references -
    # a wrong answer that looks entirely plausible.
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1.0

    rotation = u @ np.diag(d) @ vt
    scale = float((s * d).sum() / src_var)
    translation = dst_mean - scale * (rotation @ src_mean)

    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = (scale * rotation).astype(np.float32)
    matrix[:, 2] = translation.astype(np.float32)
    return matrix


def transform_for(landmarks, image_size=None):
    """The 2x3 warp matrix for one set of five landmarks."""
    return _umeyama(landmarks, template(image_size))


def align(frame, landmarks, image_size=None):
    """Return the aligned face crop, or None if it cannot be produced.

    `landmarks` is the five (x, y) points in FRAME coordinates. The crop
    comes back BGR, exactly the layout AdaFace expects.
    """
    if frame is None or landmarks is None:
        return None
    size = int(image_size or FACE_ALIGN_SIZE)
    matrix = transform_for(landmarks, size)
    if matrix is None:
        return None
    try:
        return cv2.warpAffine(frame, matrix, (size, size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    except cv2.error:
        return None


def align_from_box(frame, box, image_size=None, expand=0.30):
    """Fallback alignment when there are NO landmarks: a padded square
    crop around the box, resized.

    This is deliberately a poor substitute and is only used for
    enrollment photos where a detector found a face but no keypoints.
    Recognition from one of these is noticeably weaker, which is correct
    - it should not look as trustworthy as a properly aligned face, and
    core/face_quality.py penalises it accordingly.
    """
    if frame is None or box is None:
        return None
    size = int(image_size or FACE_ALIGN_SIZE)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + expand)
    if side < 4:
        return None
    sx1 = int(round(max(0, cx - side / 2)))
    sy1 = int(round(max(0, cy - side / 2)))
    sx2 = int(round(min(w, cx + side / 2)))
    sy2 = int(round(min(h, cy + side / 2)))
    if sx2 - sx1 < 4 or sy2 - sy1 < 4:
        return None
    crop = frame[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
