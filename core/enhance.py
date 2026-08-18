"""
core/enhance.py
===============
Recover detail in badly lit scenes before detection.

The problem
-----------
The workspace camera has windows down one side. Measured on a real frame:
the far side sits at mean brightness 62 with its darkest 5% crushed to
value 1, while the window side sits at 134. No single camera exposure
suits both, so the dark half loses exactly the edge detail a detector
needs to separate a seated person from a black chair.

The fix
-------
CLAHE - contrast limited adaptive histogram equalisation. It works on
small tiles rather than the whole image, so it lifts the dark corner
without blowing out the bright side. It is applied only to the L
(lightness) channel in LAB space, leaving colour untouched, which matters
because the body measurements used for re-identification depend on
stable colour.

A mild gamma lift can be applied first for scenes that are dark overall
rather than merely uneven.

This runs on CPU in about a millisecond for a typical frame, and only on
cameras configured to need it.
"""
import cv2
import numpy as np

_GAMMA_CACHE = {}


def _gamma_table(gamma):
    if gamma not in _GAMMA_CACHE:
        inv = 1.0 / max(0.05, gamma)
        _GAMMA_CACHE[gamma] = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return _GAMMA_CACHE[gamma]


def apply_gamma(image, gamma=1.4):
    """gamma > 1 brightens shadows; < 1 darkens."""
    return cv2.LUT(image, _gamma_table(gamma))


def clahe(image, clip=3.0, tiles=8):
    """Local contrast equalisation on lightness only."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return cv2.cvtColor(cv2.merge([op.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def lowlight(image, gamma=1.35, clip=3.0, tiles=8):
    """The recommended combination for an unevenly lit room."""
    return clahe(apply_gamma(image, gamma), clip=clip, tiles=tiles)


PROFILES = {
    None: lambda img: img,
    "none": lambda img: img,
    "clahe": clahe,
    "gamma": apply_gamma,
    "lowlight": lowlight,
}


def enhance(image, profile):
    """Apply a named enhancement profile. Unknown names pass through."""
    fn = PROFILES.get(profile)
    if fn is None:
        return image
    try:
        return fn(image)
    except Exception:
        return image


def detail_score(image):
    """A rough measure of how much usable edge detail an image carries.
    Useful for checking whether an enhancement actually helped."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())
