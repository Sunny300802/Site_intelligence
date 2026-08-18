"""
core/ultra.py
=============
Small compatibility helpers for Ultralytics.

Two jobs:

1. Quieten the library. Newer versions emit a deprecation warning on
   EVERY predict() call. In a loop running 20+ times a second that
   floods the console and wastes real time on I/O.

2. Work out how to ask for FP16. Ultralytics changed this: older builds
   take `half=True`, newer ones take `quantize="fp16"` (and reject
   `quantize=True` outright). Rather than guess from a version number,
   we simply TRY each form once at startup and keep the first that
   actually runs without raising. Anything that raises - for any reason -
   is rejected, which is the only safe rule: a form that errors during a
   probe will error in the hot loop too.
"""
import logging

# In preference order. The value matters, not just the name:
# quantize expects a precision like "fp16", never a boolean.
_CANDIDATES = (
    {"quantize": "fp16"},
    {"half": True},
)


def quiet():
    """Stop per-call warnings from flooding the console."""
    try:
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(logging.ERROR)
    except Exception:
        pass
    for name in ("ultralytics", "ultralytics.utils"):
        logging.getLogger(name).setLevel(logging.ERROR)


def resolve_precision_kwarg(model, imgsz, device, want_half):
    """Return the kwargs that request FP16 from THIS Ultralytics build.

    Returns {} if neither form works, which means full precision - a bit
    slower, but correct and stable.
    """
    if not want_half:
        return {}
    import numpy as np
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    for candidate in _CANDIDATES:
        try:
            model.predict(blank, imgsz=imgsz, device=device, verbose=False,
                          **candidate)
        except Exception:
            continue          # unknown name OR invalid value - unusable
        return candidate
    return {}
