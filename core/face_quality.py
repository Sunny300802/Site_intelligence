"""
core/face_quality.py
====================
Decide whether a face is worth recognising AT ALL, before any embedding
is computed.

The point
---------
A recognition model does not say "I cannot see this face well enough".
Hand it 30 blurred pixels of somebody's cheek and it returns a perfectly
formed 512-number description, and that description will be closest to
SOMEBODY in the gallery. The number that comes back looks exactly like a
real match. This is the single largest source of wrong names in a CCTV
system, and no amount of threshold tuning fixes it, because the bad
scores and the good scores live in the same range.

The fix is to refuse to ask the question. Everything below is a reason
not to run recognition:

  TOO SMALL      under MIN_FACE_SIZE pixels wide there is less detail
                 between the eyes than the model needs to separate two
                 colleagues.
  BLURRY         motion blur, or a heavily compressed stream. Both
                 destroy exactly the high-frequency detail identity
                 lives in.
  OCCLUDED       a hand, a mask, a monitor edge, a lanyard held up. The
                 covered part of the face contributes nothing, but the
                 model still returns a confident answer based on the
                 rest.
  PROFILE        past FACE_MAX_YAW_DEG (60 by default) the far eye has
                 gone and the visible half of the face carries little of
                 what the model was trained on. Measured as an ANGLE, so
                 the bar can be sanity-checked - see the note in
                 config/settings.py about why that matters.
  BADLY LIT      a silhouette against the reception glass, or a face
                 blown out by a window, has no usable contrast.

Each of those is measured, turned into a 0-1 sub-score, and combined
into one number. A face must clear every hard gate AND reach
FACE_QUALITY_THRESHOLD before it is embedded.

The same number does second duty as the BEST-FRAME score: within one
track we keep the highest-quality face seen so far and recognise that,
rather than recognising every frame and hoping.

Everything here is measured on the ALIGNED crop, not the raw box. That
is deliberate - the aligned crop is exactly what the recognition model
will see, so the quality we measure is the quality that will matter.
"""
import math

import numpy as np
import cv2

from config.settings import (MIN_FACE_SIZE, FACE_GOOD_SIZE,
                             FACE_SHARPNESS_MIN, FACE_SHARPNESS_GOOD,
                             FACE_BRIGHTNESS_MIN, FACE_BRIGHTNESS_MAX,
                             FACE_CONTRAST_MIN, FACE_CONTRAST_GOOD,
                             FACE_NOSE_PROJECTION, FACE_GOOD_YAW_DEG,
                             FACE_MAX_YAW_DEG,
                             FACE_MAX_ROLL_DEG, FACE_OCCLUSION_MIN,
                             FACE_OCCLUSION_GOOD, FACE_QUALITY_WEIGHTS,
                             FACE_QUALITY_THRESHOLD,
                             FACE_DETECTION_THRESHOLD)

# Horizontal bands of the aligned 112x112 template, as fractions of its
# height. These follow the template's own geometry: eyes sit at y=51.7,
# the nose tip at 71.7, the mouth at 92.3.
_BANDS = (("eyes", 0.36, 0.55),
          ("nose", 0.52, 0.76),
          ("mouth", 0.72, 0.92))


def _clamp01(value):
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def _ramp(value, low, high):
    """0 at or below `low`, 1 at or above `high`, linear between."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return _clamp01((value - low) / (high - low))


class FaceQuality:
    """One face's quality verdict, with its reasoning kept.

    The reasons are not decoration. When a site reports "it stopped
    recognising anybody", the answer is almost always in the histogram
    of these strings, and guessing at it from a single overall number is
    what turns a five-minute fix into an afternoon.
    """

    __slots__ = ("score", "ok", "reasons", "parts", "width", "height",
                 "det_score", "sharpness", "brightness", "contrast",
                 "yaw", "roll", "occlusion")

    def __init__(self):
        self.score = 0.0
        self.ok = False
        self.reasons = []
        self.parts = {}
        self.width = 0
        self.height = 0
        self.det_score = 0.0
        self.sharpness = 0.0
        self.brightness = 0.0
        self.contrast = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.occlusion = 0.0

    @property
    def reason(self):
        return ", ".join(self.reasons) if self.reasons else "ok"

    def as_row(self):
        """Flat dict, for the CSV trace and the debug line."""
        row = {"quality": round(self.score, 4), "ok": int(self.ok),
               "width": self.width, "det": round(self.det_score, 4),
               "sharpness": round(self.sharpness, 2),
               "brightness": round(self.brightness, 1),
               "contrast": round(self.contrast, 2),
               "yaw": round(self.yaw, 3), "roll": round(self.roll, 1),
               "occlusion": round(self.occlusion, 3),
               "reason": self.reason}
        row.update({f"q_{k}": round(v, 4) for k, v in self.parts.items()})
        return row

    def __repr__(self):
        return f"<FaceQuality {self.score:.2f} {'ok' if self.ok else self.reason}>"


# ------------------------------------------------------------- metrics
def _pose_from_landmarks(landmarks):
    """(yaw_degrees, roll_degrees, pitch_score) from the five points.

    Measured in the eye-line's own frame rather than in image
    coordinates, so head TILT does not masquerade as head TURN. A
    person leaning their head on their hand is perfectly recognisable;
    a person in profile is not, and an image-axis measurement cannot
    tell those two apart.

    yaw_degrees  0 = the nose sits exactly between the eyes (frontal),
                 rising as the head turns. Recovered from the nose's
                 offset along the eye line: under a yaw of theta the
                 nose moves off centre by about
                     FACE_NOSE_PROJECTION * tan(theta)
                 of the interocular distance, so the angle inverts out
                 of that offset.
    pitch_score  1.0 = the nose sits where the template expects it
                 between the eye line and the mouth line
    """
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 5 or not np.isfinite(pts).all():
        return None, None, None

    left_eye, right_eye, nose, left_mouth, right_mouth = pts[:5]

    eye_vector = right_eye - left_eye
    eye_span = float(np.linalg.norm(eye_vector))
    if eye_span < 1e-3:
        return None, None, None

    axis = eye_vector / eye_span                       # along the eye line
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)   # down the face

    # where the nose falls along the eye line: 0 = left eye, 1 = right eye
    along = float(np.dot(nose - left_eye, axis)) / eye_span
    offset = abs(along - 0.5)
    yaw_degrees = math.degrees(
        math.atan(offset / max(1e-3, FACE_NOSE_PROJECTION)))

    roll = math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0])))
    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180

    # how far down the face the nose sits, between eyes and mouth
    mouth_mid = (left_mouth + right_mouth) * 0.5
    eye_mid = (left_eye + right_eye) * 0.5
    depth = float(np.dot(mouth_mid - eye_mid, normal))
    if abs(depth) < 1e-3:
        pitch_score = 0.0
    else:
        down = float(np.dot(nose - eye_mid, normal)) / depth
        # the template puts the nose tip just under half way; anything
        # outside 0.15-0.85 is somebody looking hard up or hard down
        pitch_score = _clamp01(1.0 - abs(down - 0.5) / 0.35)

    return yaw_degrees, roll, pitch_score


def _occlusion_score(gray):
    """How evenly detail is spread over the eye, nose and mouth bands.

    Each band's edge energy is measured RELATIVE to the whole crop's,
    which makes the number independent of lighting and of how sharp the
    picture is overall - both of which are scored separately. What is
    left is the thing we actually want to know: is one part of this face
    missing its detail because something is in front of it?
    """
    if gray is None or gray.size == 0:
        return 0.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.magnitude(grad_x, grad_y)
    overall = float(energy.mean())
    if overall < 1e-6:
        return 0.0

    height = gray.shape[0]
    ratios = []
    for _name, top, bottom in _BANDS:
        y1 = int(round(top * height))
        y2 = max(y1 + 1, int(round(bottom * height)))
        band = energy[y1:y2]
        if band.size:
            ratios.append(float(band.mean()) / overall)
    return min(ratios) if ratios else 0.0


def _landmarks_sane(landmarks, box):
    """Reject geometrically impossible keypoint sets.

    SCRFD occasionally returns a confident box over a patterned surface
    - a chair back, a poster - with its five points collapsed into a
    corner. Aligning on those produces a warp that samples a nearly
    random patch of the frame, and the embedding of a random patch still
    matches somebody.
    """
    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 5 or not np.isfinite(pts).all():
        return False
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    # every point should be within a box-width of the box
    if (pts[:, 0].min() < x1 - width or pts[:, 0].max() > x2 + width or
            pts[:, 1].min() < y1 - height or pts[:, 1].max() > y2 + height):
        return False
    # The eyes must be meaningfully apart relative to the face. This
    # catches a COLLAPSED keypoint set - all five points landing in one
    # spot on a chair back or a poster - and nothing else. It is
    # deliberately far below the value a profile face produces: measured
    # over this site's own enrollment photos, genuine faces sit at 0.30
    # of the box width, three-quarter views around 0.14, and only the
    # bottom 5% (broken sets, and true 90-degree profiles) fall under
    # 0.05. Judging PROFILE here as well was the mistake - it reported
    # perfectly good side-on photographs as "landmarks implausible",
    # which sends the operator looking for a detector fault that does
    # not exist. Pose is measured properly below and reported as pose.
    eye_span = float(np.linalg.norm(pts[1] - pts[0]))
    if eye_span < 0.05 * width:
        return False
    # ...and the mouth must be below the eyes, measured down the face
    eye_mid = (pts[0] + pts[1]) * 0.5
    mouth_mid = (pts[3] + pts[4]) * 0.5
    axis = pts[1] - pts[0]
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    normal /= (np.linalg.norm(normal) + 1e-9)
    if float(np.dot(mouth_mid - eye_mid, normal)) <= 0.05 * height:
        return False
    return True


# ------------------------------------------------------------ assessor
# Reasons that make a crop UNUSABLE rather than merely poor. These stay
# fatal even in non-strict mode, because they mean the pixels we are
# holding are not reliably a face at all.
_FATAL = ("no crop", "landmarks implausible", "landmarks unusable")


def assess(aligned, box, landmarks=None, det_score=1.0,
           threshold=None, min_size=None, strict=True, allow_profile=False):
    """Judge one aligned face. Returns a FaceQuality.

    `aligned`   the 112x112 BGR crop that will be fed to the model
    `box`       the face box in ORIGINAL frame coordinates - size is
                judged there, because that is the real resolution the
                camera gave us; the aligned crop is always 112px
                regardless of how few real pixels went into it
    `landmarks` the five points in original frame coordinates, or None
    `det_score` SCRFD's confidence
    `strict`    True on the live cameras: any hard gate (too small,
                blurry, side-on, badly lit, occluded) fails the face
                outright. False for ENROLLMENT and for evaluation, where
                the input is a deliberately chosen photograph and a
                three-quarter view is a reference worth keeping, not a
                mistake to reject - there, only the overall score and
                the genuinely unusable cases decide.
    `allow_profile`
                Whether a near-profile face may pass in non-strict mode.
                Off by default even there, and for a specific reason:
                the live cameras will never accept a profile (see
                FACE_POSE_YAW_MIN), so a profile REFERENCE can never be
                matched against anything - it cannot help, and it does
                pull the person's centroid away from the frontal views
                that will actually be searched.
    """
    q = FaceQuality()
    q.det_score = float(det_score)

    if box is not None:
        q.width = int(round(box[2] - box[0]))
        q.height = int(round(box[3] - box[1]))

    min_width = MIN_FACE_SIZE if min_size is None else int(min_size)
    bar = FACE_QUALITY_THRESHOLD if threshold is None else float(threshold)

    if aligned is None or aligned.size == 0:
        q.reasons.append("no crop")
        return q

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) \
        if aligned.ndim == 3 else aligned

    # ---- size -------------------------------------------------------
    q.parts["size"] = _ramp(q.width, min_width * 0.5, FACE_GOOD_SIZE)
    if q.width < min_width:
        q.reasons.append(f"too small ({q.width}px < {min_width})")

    # ---- sharpness --------------------------------------------------
    q.sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    q.parts["sharpness"] = _ramp(q.sharpness, FACE_SHARPNESS_MIN * 0.5,
                                 FACE_SHARPNESS_GOOD)
    if q.sharpness < FACE_SHARPNESS_MIN:
        q.reasons.append(f"blurry ({q.sharpness:.0f} < {FACE_SHARPNESS_MIN:.0f})")

    # ---- illumination ----------------------------------------------
    q.brightness = float(gray.mean())
    q.contrast = float(gray.std())
    # brightness is judged as distance from the middle of the usable
    # band, so both a silhouette and a blown-out face are penalised
    mid = (FACE_BRIGHTNESS_MIN + FACE_BRIGHTNESS_MAX) * 0.5
    half = max(1.0, (FACE_BRIGHTNESS_MAX - FACE_BRIGHTNESS_MIN) * 0.5)
    brightness_score = _clamp01(1.0 - abs(q.brightness - mid) / half)
    contrast_score = _ramp(q.contrast, FACE_CONTRAST_MIN * 0.5,
                           FACE_CONTRAST_GOOD)
    q.parts["illumination"] = 0.4 * brightness_score + 0.6 * contrast_score
    if q.brightness < FACE_BRIGHTNESS_MIN:
        q.reasons.append(f"too dark ({q.brightness:.0f})")
    elif q.brightness > FACE_BRIGHTNESS_MAX:
        q.reasons.append(f"blown out ({q.brightness:.0f})")
    if q.contrast < FACE_CONTRAST_MIN:
        q.reasons.append(f"flat ({q.contrast:.0f} contrast)")

    # ---- pose -------------------------------------------------------
    if landmarks is None:
        # No keypoints means the crop was squared off the box rather
        # than aligned. It is usable but demonstrably weaker, and it
        # should never look as trustworthy as a properly aligned face.
        q.parts["pose"] = 0.35
        q.yaw = 0.0
        q.reasons.append("no landmarks")
    elif box is not None and not _landmarks_sane(landmarks, box):
        q.parts["pose"] = 0.0
        q.reasons.append("landmarks implausible")
    else:
        yaw_deg, roll, pitch = _pose_from_landmarks(landmarks)
        if yaw_deg is None:
            q.parts["pose"] = 0.0
            q.reasons.append("landmarks unusable")
        else:
            q.yaw, q.roll = yaw_deg, roll
            # full marks up to FACE_GOOD_YAW_DEG, falling to zero at the
            # hard gate - so a three-quarter view is scored as slightly
            # worse than frontal rather than as worthless
            yaw_score = _clamp01(
                (FACE_MAX_YAW_DEG - yaw_deg) /
                max(1e-6, FACE_MAX_YAW_DEG - FACE_GOOD_YAW_DEG))
            roll_score = _clamp01(1.0 - abs(roll) / max(1.0, FACE_MAX_ROLL_DEG))
            q.parts["pose"] = (0.65 * yaw_score + 0.20 * (pitch or 0.0)
                               + 0.15 * roll_score)
            if yaw_deg > FACE_MAX_YAW_DEG:
                q.reasons.append(f"too side-on ({yaw_deg:.0f} deg turned)")
            if abs(roll) > FACE_MAX_ROLL_DEG:
                q.reasons.append(f"head tilted {abs(roll):.0f} deg")

    # ---- occlusion --------------------------------------------------
    q.occlusion = _occlusion_score(gray)
    q.parts["occlusion"] = _ramp(q.occlusion, FACE_OCCLUSION_MIN * 0.5,
                                 FACE_OCCLUSION_GOOD)
    if q.occlusion < FACE_OCCLUSION_MIN:
        q.reasons.append(f"occluded ({q.occlusion:.2f})")

    # ---- combine ----------------------------------------------------
    total_weight = sum(FACE_QUALITY_WEIGHTS.values()) or 1.0
    base = sum(FACE_QUALITY_WEIGHTS.get(name, 0.0) * value
               for name, value in q.parts.items()) / total_weight

    # The detector's own confidence scales the result rather than being
    # averaged into it. A face SCRFD was unsure about should not be able
    # to reach a high quality score on sharpness and lighting alone -
    # those are properties of the picture, and the question here is
    # whether the picture is of a face.
    span = max(1e-6, 0.90 - FACE_DETECTION_THRESHOLD)
    confidence = 0.60 + 0.40 * _clamp01(
        (q.det_score - FACE_DETECTION_THRESHOLD) / span)

    q.score = _clamp01(base * confidence)

    if strict:
        blocking = list(q.reasons)
    else:
        keep = _FATAL if allow_profile else _FATAL + ("too side-on",)
        blocking = [r for r in q.reasons if r.startswith(keep)]
    q.ok = (not blocking) and q.score >= bar
    if not blocking and not q.ok:
        q.reasons.append(f"quality {q.score:.2f} < {bar:.2f}")
    return q
