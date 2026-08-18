"""
core/body.py
============
Who somebody is by their BODY - how tall they are, how broad they are,
and where they are standing. Not by what they are wearing.

Why the clothes had to go
-------------------------
This file replaces core/appearance.py, which described a person by the
colour and pattern of their clothing. On this site that failed in the
one situation it was written for. Two people overlap; the crop of one
contains a good deal of the other; both descriptors drift toward each
other; and when they separate the name follows the wrong body. It also
cannot survive the thing that happens every single day - a person turns
around, or walks under a different light, and their "appearance" changes
more than the difference between two colleagues.

Worse, clothing was allowed to NAME people. A colour signature tells a
person from a chair very well and one colleague from another badly, so
every busy moment produced a plausible wrong answer, and the name
visibly jumped from body to body.

What is used instead
--------------------
Three measurements, all of them geometric, all of them stable for as
long as somebody is in the building:

  STATURE   How tall this person is, corrected for how far away they
            are. Apparent height alone is useless - the same person is
            300 pixels tall at the door and 90 at the far wall - so we
            learn, per camera, what pixel height a standing person has
            for each row of the floor (see Perspective below) and divide
            it out. What is left is a number near 1.0 that means "taller
            than average for this camera" or "shorter", and it does not
            change when they walk.

  BUILD     How broad they are for their height - width divided by
            height. This is the "weight" cue: a heavy person and a slight
            person of the same stature have clearly different numbers. It
            is noisier than stature, because arms move and shoulders get
            occluded, so it is kept as a rolling MEDIAN and weighted
            less.

  POSITION  Where they are and how fast they were going. Not stored
            here - the tracker's Kalman filter already predicts it - but
            it is the third leg of every re-identification decision made
            against this module.

None of the three can be confused by two people wearing the same shirt,
and none of them changes when somebody turns their back.

What this is honestly NOT
-------------------------
It is not a person re-identification network, and it will not tell two
similarly built colleagues apart on its own. It is not meant to. Names
come from FACES. Stature and build exist to hold the right name on the
right body through an occlusion, and to refuse a name that is about to
land on a body of obviously the wrong size - both of which they do well,
because they are measured from the same box the tracker is already
following.

Which is why an unusable measurement returns None rather than zero
everywhere below. "I cannot tell" and "definitely not them" are
different answers, and treating the first as the second is what makes a
tracker invent a new person out of somebody it simply could not measure.
"""
from collections import deque
from statistics import median

# A box smaller than this is not a measurement, it is noise.
MIN_BOX_HEIGHT = 24
MIN_BOX_WIDTH = 10

# STATURE TOLERANCE. Two measurements of one person differ by a few per
# cent - posture, a foot placed forward, the detector clipping the hair.
# Two different people usually differ by more. 0.88 means "within 12% is
# perfectly consistent"; below that the score falls away.
STATURE_FLOOR = 0.88

# BUILD TOLERANCE, deliberately much looser. Width is the noisy
# measurement: an arm out, a bag, half a shoulder behind a door frame.
BUILD_FLOOR = 0.68

W_STATURE = 0.70
W_BUILD = 0.30

# How many observations before the perspective fit is trusted, and how
# much of the frame's height they must span. Fitting a line through
# people who all stood in the same place gives a confident, meaningless
# answer.
FIT_MIN_SAMPLES = 80
FIT_MIN_SPREAD = 0.12
FIT_REFRESH = 40           # re-fit after this many new samples


def _clamp01(value):
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _ratio_score(a, b, floor):
    """How consistent are two positive measurements? 1.0 = identical."""
    if not a or not b or a <= 0 or b <= 0:
        return None
    ratio = min(a, b) / max(a, b)
    return _clamp01((ratio - floor) / (1.0 - floor))


class Perspective:
    """What pixel height a standing person has, per row of the floor.

    One of these per camera. It learns from the tracker's own boxes: for
    a fixed camera the relationship between "where the feet are" and
    "how tall the box is" is close to linear, so a plain least-squares
    line through a few hundred sightings is enough to turn an apparent
    height into a real one.

    It is learned rather than configured because it depends on the lens,
    the mounting height and the tilt of each camera, and nobody is going
    to measure those. Until it has enough evidence it says so - see
    expected() - and callers fall back to comparing people who are
    standing at a similar distance, which needs no model at all.
    """

    __slots__ = ("_samples", "_a", "_b", "_since_fit")

    def __init__(self):
        self._samples = deque(maxlen=1200)   # (foot_row, height), fractions
        self._a = None
        self._b = None
        self._since_fit = 0

    def observe(self, foot_row, height_fraction):
        if not (0.0 <= foot_row <= 1.0) or not (0.0 < height_fraction <= 1.5):
            return
        self._samples.append((foot_row, height_fraction))
        self._since_fit += 1
        if (len(self._samples) >= FIT_MIN_SAMPLES
                and (self._a is None or self._since_fit >= FIT_REFRESH)):
            self._fit()

    def _fit(self):
        rows = [r for r, _h in self._samples]
        if max(rows) - min(rows) < FIT_MIN_SPREAD:
            return                       # everybody stood in one place
        self._since_fit = 0

        # Trim the tallest and shortest tenth before fitting. A merged box
        # around two overlapping people, or a person cut off by the edge
        # of the frame, is a wild outlier, and least squares would follow
        # it. The bulk of the samples are ordinary sightings and they are
        # what should set the line.
        heights = sorted(h for _r, h in self._samples)
        low = heights[len(heights) // 10]
        high = heights[-max(1, len(heights) // 10)]
        kept = [(r, h) for r, h in self._samples if low <= h <= high]
        n = len(kept)
        if n < FIT_MIN_SAMPLES // 2:
            return

        mean_r = sum(r for r, _h in kept) / n
        mean_h = sum(h for _r, h in kept) / n
        numerator = sum((r - mean_r) * (h - mean_h) for r, h in kept)
        denominator = sum((r - mean_r) ** 2 for r, _h in kept)
        if denominator <= 1e-9:
            return
        b = numerator / denominator
        a = mean_h - b * mean_r
        # A fit that predicts a non-positive height anywhere people
        # actually walk is not a perspective model, it is a bad line.
        if a + b * 0.1 <= 0.01 or a + b * 1.0 <= 0.01:
            return
        self._a, self._b = a, b

    @property
    def ready(self):
        return self._a is not None

    def expected(self, foot_row):
        """Height fraction a typical person has with their feet there,
        or None while the camera is still being learned."""
        if self._a is None:
            return None
        value = self._a + self._b * _clamp01(foot_row)
        return value if value > 0.01 else None

    def summary(self):
        return {"ready": self.ready, "samples": len(self._samples),
                "intercept": self._a, "slope": self._b}


class Body:
    """One frame's measurement of one person."""

    __slots__ = ("foot_row", "height", "width", "stature", "build",
                 "cx", "cy")

    def __init__(self, foot_row, height, width, stature, build, cx, cy):
        self.foot_row = foot_row      # 0 = top of frame, 1 = bottom
        self.height = height          # fraction of frame height
        self.width = width            # fraction of frame width
        self.stature = stature        # height / expected height, or None
        self.build = build            # width / height
        self.cx = cx
        self.cy = cy

    def __repr__(self):
        stature = "?" if self.stature is None else f"{self.stature:.2f}"
        return f"<Body stature={stature} build={self.build:.2f}>"


def measure(box, frame_shape, perspective=None, learn=True):
    """Measure the person in `box`. Returns a Body, or None.

    `learn` is False while the person overlaps somebody else: the box
    around two merged people is taller and much wider than either of
    them, and feeding that to the perspective model bends the line for
    every future measurement.
    """
    if box is None or frame_shape is None:
        return None
    frame_h, frame_w = frame_shape[0], frame_shape[1]
    if frame_h <= 0 or frame_w <= 0:
        return None

    x1, y1, x2, y2 = [float(v) for v in box]
    height_px = y2 - y1
    width_px = x2 - x1
    if height_px < MIN_BOX_HEIGHT or width_px < MIN_BOX_WIDTH:
        return None

    foot_row = _clamp01(y2 / float(frame_h))
    height = height_px / float(frame_h)
    width = width_px / float(frame_w)
    build = width_px / height_px

    stature = None
    if perspective is not None:
        if learn:
            perspective.observe(foot_row, height)
        expected = perspective.expected(foot_row)
        if expected:
            stature = height / expected

    return Body(foot_row, height, width, stature, build,
                (x1 + x2) * 0.5, (y1 + y2) * 0.5)


def similarity(a, b):
    """Could these two measurements be the same person? None = cannot say.

    NOT a probability that they ARE the same person - two colleagues of
    similar size score highly and always will. It answers the question
    the tracker actually asks after an occlusion: of the two or three
    people who could be this box, which ones are the right SIZE for it.
    """
    if a is None or b is None:
        return None
    a_stature, a_build = a.stature, a.build
    b_stature, b_build = b.stature, b.build

    parts, weights = [], []

    if a_stature and b_stature:
        score = _ratio_score(a_stature, b_stature, STATURE_FLOOR)
        if score is not None:
            parts.append(score)
            weights.append(W_STATURE)
    elif abs(a.foot_row - b.foot_row) < 0.12:
        # No perspective model yet, but they are standing at a similar
        # distance - so raw apparent height IS comparable, and it is the
        # strongest thing available.
        score = _ratio_score(a.height, b.height, STATURE_FLOOR)
        if score is not None:
            parts.append(score)
            weights.append(W_STATURE)

    score = _ratio_score(a_build, b_build, BUILD_FLOOR)
    if score is not None:
        parts.append(score)
        weights.append(W_BUILD)

    if not parts:
        return None
    return sum(p * w for p, w in zip(parts, weights)) / sum(weights)


class BodyMemory:
    """The running measurement of one tracked person.

    Medians, not averages, and not the latest value either. Every so
    often a box is wrong - clipped at the edge of the frame, merged with
    a passing colleague, or half a person behind a door - and one bad
    frame must not be able to change who this track looks like. The
    median of the last few dozen good frames is a stable number that a
    single outlier cannot move.

    Only measurements taken while the person was NOT overlapping anybody
    are added; the caller enforces that, for the reason in measure().
    """

    __slots__ = ("_statures", "_builds", "_heights", "_last", "_seen")

    def __init__(self, limit=60):
        self._statures = deque(maxlen=limit)
        self._builds = deque(maxlen=limit)
        self._heights = deque(maxlen=limit)     # (foot_row, height)
        self._last = None
        self._seen = 0

    def add(self, body):
        if body is None:
            return
        self._last = body
        self._seen += 1
        if body.stature:
            self._statures.append(body.stature)
        if body.build:
            self._builds.append(body.build)
        self._heights.append((body.foot_row, body.height))

    @property
    def stature(self):
        return median(self._statures) if self._statures else None

    @property
    def build(self):
        return median(self._builds) if self._builds else None

    @property
    def samples(self):
        return self._seen

    def typical(self):
        """This person's settled measurements, as a Body to compare with."""
        if not self._heights:
            return None
        foot = median(f for f, _h in self._heights)
        height = median(h for _f, h in self._heights)
        build = self.build or 0.0
        return Body(foot, height, height * build, self.stature, build,
                    0.0, 0.0)

    def score(self, body):
        """How consistent is this measurement with the person we have been
        watching? None = not enough evidence to say."""
        if body is None or self._seen < 3:
            return None
        return similarity(self.typical(), body)

    def __len__(self):
        return self._seen

    def __bool__(self):
        return self._seen > 0
