"""
core/face_library.py
====================
The system builds its own reference photographs, from its own cameras.

The problem this solves
-----------------------
Enrollment photos are taken on a phone. They are well lit, close up and
frontal. The cameras see something entirely different: sixty pixels of
face, off angle, compressed, lit from one side. Asking a recognition
model to bridge that gap is asking it to do the hardest version of the
job, and on this site it does not: live faces score 0.25-0.31 against
phone-photo references while a genuine match needs 0.50.

A reference captured FROM THE CAMERA has no gap to bridge. It looks like
what the camera will see tomorrow, because it is what the camera saw
today. Twenty of those per person is worth more than any model swap or
threshold change available here - which is why this file exists.

How a sample is earned
----------------------
Two ways in, and both end in the same place:

  AUTOMATICALLY  when the temporal vote has CONFIRMED an identity. That
                 is not one frame's guess - it is several good frames
                 agreeing, over the recognition threshold, with a clear
                 margin over everybody else. Strong enough to file
                 without asking.

  BY CONFIRMATION when a human answers a question on the dashboard, the
                 crop that question was asked about is filed too.

Why samples are filtered rather than hoarded
--------------------------------------------
Twenty frames of somebody standing still are twenty copies of one
reference. They teach nothing new and they drag that person's average
toward a single pose, which makes them HARDER to recognise from any
other angle. So a new sample must be meaningfully different from every
sample already held (FACE_LIBRARY_MAX_SIMILARITY) before it is kept.
What fills up is a set of genuinely different looks: turned left,
turned right, near, far, morning, evening.

Where they go
-------------
Into that person's own enrollment folder, alongside their original
photos:

    data/faces/12219-Rajesh Palamangalam/
        Photo_1.jpg                  <- the phone photo
        cctv_reception_20260810_1432.jpg   <- captured here
        cctv_server_rm_psg_20260810_1611.jpg

so `python tools/enroll_faces.py` picks them up like any other
photograph, and a human can look through the folder and delete anything
wrong. That last part matters: an automatic system that files pictures
somewhere opaque cannot be audited, and this one must be.
"""
import os
import re
import glob
import time
import threading
from datetime import datetime

import numpy as np
import cv2

from config.settings import (FACE_DIR, FACE_LIBRARY_ENABLED,
                             FACE_LIBRARY_TARGET, FACE_LIBRARY_MIN_QUALITY,
                             FACE_LIBRARY_MAX_SIMILARITY,
                             FACE_LIBRARY_MIN_INTERVAL,
                             FACE_LIBRARY_CROP_PAD)

# Files this module wrote, as opposed to the operator's own photographs.
CAPTURE_PREFIX = "cctv_"

_SAFE = re.compile(r"[^A-Za-z0-9 _.-]+")


def _safe(text):
    return _SAFE.sub("", str(text or "")).strip()


class FaceLibrary:
    """Per-person camera-captured reference photographs.

    Thread-safe: the camera workers all call into this, and two cameras
    seeing the same person at the same moment is normal, not an edge
    case.
    """

    def __init__(self, root=None, target=None):
        self.root = root or FACE_DIR
        self.target = int(target or FACE_LIBRARY_TARGET)
        self._lock = threading.RLock()
        self._vectors = {}         # name -> [embedding, ...] of captures
        self._last_capture = {}    # name -> when
        self._folders = {}         # name -> folder path
        self.captured = 0
        self.rejected_similar = 0
        self.rejected_full = 0

    # ------------------------------------------------------- folders
    def folder_for(self, name, code="", create=True):
        """That person's enrollment folder, creating it if needed.

        Matches an EXISTING folder first, by the name embedded in it, so
        captures land next to the person's original photographs instead
        of creating a near-duplicate folder that enrollment would then
        treat as a second person.
        """
        name = _safe(name)
        if not name:
            return None
        with self._lock:
            cached = self._folders.get(name)
            if cached and os.path.isdir(cached):
                return cached

            wanted = name.lower()
            for path in sorted(glob.glob(os.path.join(self.root, "*"))):
                if not os.path.isdir(path):
                    continue
                base = os.path.basename(path)
                label = base.split("-", 1)[1] if "-" in base else base
                if label.replace("_", " ").strip().lower() == wanted:
                    self._folders[name] = path
                    return path

            if not create:
                return None
            code = _safe(code)
            folder = os.path.join(self.root,
                                  f"{code}-{name}" if code else name)
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as exc:
                print(f"[LIBRARY] cannot create {folder}: {exc}")
                return None
            self._folders[name] = folder
            return folder

    # -------------------------------------------------------- counts
    def count(self, name, code=""):
        """How many camera-captured references this person has."""
        folder = self.folder_for(name, code, create=False)
        if not folder:
            return 0
        return len(glob.glob(os.path.join(folder, f"{CAPTURE_PREFIX}*")))

    def total_photos(self, name, code=""):
        """Every usable photograph they have, captured or enrolled."""
        folder = self.folder_for(name, code, create=False)
        if not folder:
            return 0
        return len([p for p in glob.glob(os.path.join(folder, "*"))
                    if os.path.isfile(p)])

    def is_full(self, name, code=""):
        return self.count(name, code) >= self.target

    def needs_samples(self, name, code=""):
        """Is this person still worth collecting - and asking about?"""
        if not FACE_LIBRARY_ENABLED:
            return False
        return self.count(name, code) < self.target

    # ------------------------------------------------------ decisions
    def _known_vectors(self, name):
        """Embeddings of this person's captures, loaded lazily.

        Only captures made in THIS process are held in memory; the
        similarity test is therefore approximate across restarts, which
        is fine - its job is to stop bursts of near-identical frames,
        and those all happen within one run.
        """
        return self._vectors.setdefault(name, [])

    def should_capture(self, name, embedding, quality, code="", now=None):
        """Is this face worth keeping? Returns (yes, why_not).

        The reason is returned rather than swallowed so the console can
        say "nothing new to learn about Rajesh" instead of going quiet
        and leaving somebody wondering whether the feature works.
        """
        if not FACE_LIBRARY_ENABLED:
            return False, "library disabled"
        if not name or name == "Unknown":
            return False, "no name"
        if quality is not None and quality < FACE_LIBRARY_MIN_QUALITY:
            return False, f"quality {quality:.2f} below "\
                          f"{FACE_LIBRARY_MIN_QUALITY}"

        now = now or time.time()
        with self._lock:
            last = self._last_capture.get(name, 0.0)
            if now - last < FACE_LIBRARY_MIN_INTERVAL:
                return False, "too soon after the last one"

            if self.is_full(name, code):
                self.rejected_full += 1
                return False, f"already has {self.target} camera photos"

            if embedding is not None:
                vector = np.asarray(embedding, dtype=np.float32).ravel()
                if vector.size and np.isfinite(vector).all():
                    vector = vector / (np.linalg.norm(vector) + 1e-9)
                    for held in self._known_vectors(name):
                        if float(held @ vector) >= FACE_LIBRARY_MAX_SIMILARITY:
                            self.rejected_similar += 1
                            return False, "too like one already kept"
        return True, ""

    # -------------------------------------------------------- saving
    @staticmethod
    def crop_face(frame, face_box, pad=None):
        """A padded face crop at NATIVE resolution, ready to write.

        Padded rather than tight, deliberately - see
        FACE_LIBRARY_CROP_PAD. A tight crop is unusable to the detector
        that will re-read this file and to the human who may check it.
        """
        if frame is None or face_box is None:
            return None
        pad_fraction = FACE_LIBRARY_CROP_PAD if pad is None else float(pad)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_box]
        pad_px = max(10, int(pad_fraction * max(x2 - x1, y2 - y1)))
        cx1, cy1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
        cx2, cy2 = min(width, x2 + pad_px), min(height, y2 + pad_px)
        if cx2 - cx1 < 24 or cy2 - cy1 < 24:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        return crop if crop.size else None

    def save(self, name, frame, face_box, embedding=None, code="",
             camera="", quality=None, source="auto"):
        """Write one camera reference into that person's folder.

        Returns the path written, or None. The embedding is also handed
        to the live search index and the database by the caller, so the
        new reference is usable immediately rather than at the next
        enrollment run.
        """
        folder = self.folder_for(name, code)
        if not folder:
            return None
        crop = self.crop_face(frame, face_box)
        if crop is None:
            return None

        # Millisecond stamp, then a collision check. Two cameras can see
        # the same person in the same second, and a second-resolution
        # name silently OVERWRITES the earlier file - so the folder
        # stays at one photograph however many are captured, and the
        # target is never reached.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        prefix = _safe(camera).replace(" ", "_") or "cam"
        path = os.path.join(folder, f"{CAPTURE_PREFIX}{prefix}_{stamp}.jpg")
        attempt = 1
        while os.path.exists(path):
            path = os.path.join(
                folder, f"{CAPTURE_PREFIX}{prefix}_{stamp}_{attempt}.jpg")
            attempt += 1
        try:
            if not cv2.imwrite(path, crop):
                return None
        except Exception as exc:
            print(f"[LIBRARY] could not write {path}: {exc}")
            return None

        with self._lock:
            self._last_capture[name] = time.time()
            if embedding is not None:
                vector = np.asarray(embedding, dtype=np.float32).ravel()
                if vector.size and np.isfinite(vector).all():
                    self._known_vectors(name).append(
                        vector / (np.linalg.norm(vector) + 1e-9))
            self.captured += 1

        held = self.count(name, code)
        print(f"[LIBRARY] kept a camera photo of {name} "
              f"({held}/{self.target}"
              f"{f', quality {quality:.2f}' if quality is not None else ''}"
              f", {source}) -> {os.path.basename(path)}")
        return path

    # ------------------------------------------------------- reporting
    def summary(self):
        """Per-person counts, for the dashboard and the console."""
        out = []
        for path in sorted(glob.glob(os.path.join(self.root, "*"))):
            if not os.path.isdir(path):
                continue
            base = os.path.basename(path)
            label = base.split("-", 1)[1] if "-" in base else base
            captured = len(glob.glob(os.path.join(path, f"{CAPTURE_PREFIX}*")))
            total = len([p for p in glob.glob(os.path.join(path, "*"))
                         if os.path.isfile(p)])
            out.append({"name": label.replace("_", " ").strip(),
                        "captured": captured,
                        "enrolled": total - captured,
                        "total": total,
                        "full": captured >= self.target})
        return sorted(out, key=lambda r: (r["captured"], r["name"]))

    def stats(self):
        with self._lock:
            return {"captured_this_run": self.captured,
                    "skipped_as_duplicate": self.rejected_similar,
                    "skipped_as_full": self.rejected_full,
                    "target_per_person": self.target}


# One library shared by every camera - the same person seen at reception
# and in the work area must land in one folder, not two.
LIBRARY = FaceLibrary()
