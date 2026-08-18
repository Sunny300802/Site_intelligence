"""
core/vlm_verify.py
==================
The vision model as an ARBITER: one background thread that answers, off
to one side, the two questions this system gets wrong in ways nothing
else in the pipeline can catch.

    "is that box really a person?"       reflections in the glass, a
                                         person on a monitor, a coat over
                                         a chair - all of which become
                                         tracks, then presence sessions,
                                         then somebody's attendance
    "is that really who you say it is?"  a name contradicted by something
                                         plainly visible

WHY THIS EXISTS AT ALL
----------------------
Every existing guard in this system judges a face against OTHER FACE
VECTORS. The threshold, the runner-up margin, the temporal vote, the
body-size veto, agrees_with_enrollment - all of them are arguments
inside one embedding space. When that space is wrong about somebody,
every one of those guards is wrong in the same direction at once, and
they agree with each other confidently. That is exactly what a
persistent wrong name looks like, and it is why raising thresholds has
not fixed it: there is no independent witness.

A vision model is that independent witness. It has never seen the
embedding, it does not know what the tracker decided, and it is looking
at pixels a human would look at.

WHAT IT IS ALLOWED TO DO
------------------------
Refuse. Nothing else.

It cannot name anybody, it cannot raise a confidence, it cannot promote
a guess. A 7B model looking at a 40-pixel CCTV face cannot separate two
colleagues and would invent an answer if asked to - so it is never
asked. It is asked only whether it can SEE A CONTRADICTION, and a
contradiction is enough to take a name off somebody. The result of a
veto is an honest Unknown, which is the outcome this system already
prefers over a confident wrong name.

AND IT IS NEVER ASKED TO COMPARE ANYTHING
-----------------------------------------
Not even the contradiction. The model is asked one question - "describe
the person in this photograph" - and the comparison is arithmetic in
core/vlm.py, where the rules are readable and testable.

This is not fastidiousness. Asked "is this the same person?" directly,
qwen2.5vl answered "same, confidence 0.95" to a woman's CCTV crop
against a bearded man's portraits, reason "similar hair style and
beard". Asked to describe those two pictures separately it got both
right, repeatably. The comparison is the part a vision model is bad at,
so the vision model does not do it.

A person's reference description is computed ONCE from their enrollment
photographs and cached on disk, so the steady-state cost of checking a
name is a single question about the live crop.

HOW IT STAYS AFFORDABLE
-----------------------
  * ONE question in flight at a time, on ONE thread, with a minimum gap
    between questions (VLM_MIN_CALL_GAP).
  * Each question is asked ONCE - once per track for "is this a person",
    once per (track, name) for "is this really them". A confirmed answer
    is not re-asked while the track lives.
  * The queue is bounded. A backlog is dropped rather than answered,
    because an answer about somebody who left two minutes ago is not
    late, it is wrong.
  * Identity questions outrank person questions: a wrong NAME is in the
    attendance record, a phantom track is only a number.

If Ollama is not running, every answer is None, nothing is vetoed, and
the pipeline behaves exactly as it did before this file existed.
"""
import os
import json
import glob
import time
import threading

import cv2

from config.settings import (VLM_ENABLED, VLM_MODEL, VLM_MIN_CALL_GAP,
                             VLM_QUEUE_MAX, VLM_VERDICT_TTL,
                             VLM_RETRY_SECONDS, VLM_PERSON_MIN_CONFIDENCE,
                             VLM_IDENTITY_MIN_CONFIDENCE,
                             VLM_IDENTITY_REFERENCES,
                             VLM_IDENTITY_ENROLLMENT_ONLY,
                             VLM_APPEARANCE_FILE, VLM_REJECT_SHOWS,
                             VLM_SHORTLIST_MAX, VLM_CANDIDATE_POOL,
                             VLM_DEBUG, VLM_DEBUG_EVERY_SECONDS)
from core import vlm
from core.face_library import LIBRARY, CAPTURE_PREFIX

PERSON = "person"
IDENTITY = "identity"
FACE = "face"

# Identity questions are answered first: a wrong name is written into
# somebody's attendance record. The face screen comes next, because it
# is what stands between a human reviewer and a queue full of pictures
# of doors. A phantom track is only a count, so it waits.
PRIORITY = {IDENTITY: 0, FACE: 1, PERSON: 2}


class Verdict:
    """One answer from the vision model.

    `ok` is True (nothing contradicts it), False (the model saw a
    contradiction) or None (it could not tell / could not be reached).
    Only False is ever acted on.
    """

    __slots__ = ("kind", "ok", "confidence", "reason", "when", "attributes",
                 "candidates")

    def __init__(self, kind, ok, confidence=0.0, reason="", when=None,
                 attributes=None, candidates=None):
        self.kind = kind
        self.ok = ok
        self.confidence = float(confidence)
        self.reason = reason or ""
        self.when = when or time.time()
        # Only a FACE verdict carries these: what the model saw, and
        # which enrolled people that description does not rule out.
        self.attributes = attributes
        self.candidates = candidates or []

    @property
    def refuses(self):
        """Is this a veto we should act on?"""
        if self.ok is not False:
            return False
        if self.kind == FACE:
            # A face verdict is a CLASSIFICATION, not a score - the
            # decision was already made against VLM_REJECT_SHOWS by the
            # time it got here, so there is no threshold to re-apply.
            return True
        floor = {IDENTITY: VLM_IDENTITY_MIN_CONFIDENCE,
                 PERSON: VLM_PERSON_MIN_CONFIDENCE}.get(self.kind, 1.1)
        return self.confidence >= floor

    def stale(self, now=None):
        # A refusal never goes stale: it is a decision, and re-asking
        # would only give the wrong name another chance to come back.
        # A pass DOES, so a track that has quietly become somebody else
        # can be caught on a later look.
        if self.ok is False:
            return False
        return (now or time.time()) - self.when > VLM_VERDICT_TTL

    def __repr__(self):
        state = {True: "ok", False: "REFUSED", None: "unsure"}[self.ok]
        return f"<Verdict {self.kind} {state} {self.confidence:.2f} " \
               f"{self.reason!r}>"


class _Job:
    __slots__ = ("key", "kind", "image", "name", "code", "label", "queued")

    def __init__(self, key, kind, image, name="", code="", label=""):
        self.key = key
        self.kind = kind
        self.image = image
        self.name = name
        self.code = code
        self.label = label
        self.queued = time.time()


class VlmArbiter:
    """One vision model, one thread, shared by every camera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queue = []                 # [_Job]
        self._waiting = set()            # keys queued or in flight
        self._verdicts = {}              # key -> Verdict
        self._thread = None
        self._running = False

        self._available = None           # None = not checked yet
        self._checked_at = 0.0
        self._last_call = 0.0
        self._report_t0 = time.time()

        # counters, for the periodic line that says whether this is
        # doing anything worth its GPU time
        self.asked = 0
        self.refused_person = 0
        self.refused_identity = 0
        self.refused_face = 0
        self.passed = 0
        self.unsure = 0
        self.dropped = 0
        self.last_ms = 0.0

    # ---------------------------------------------------- availability
    def usable(self):
        """Is the vision model reachable? Re-checked periodically."""
        if not VLM_ENABLED:
            return False
        now = time.time()
        if self._available is None or (
                not self._available and now - self._checked_at > VLM_RETRY_SECONDS):
            self._checked_at = now
            ok, names = vlm.available()
            was = self._available
            self._available = bool(ok)
            if ok and was is not True:
                print(f"[VLM] arbiter on: '{VLM_MODEL}' will be asked to "
                      f"double-check people and names")
            elif not ok and was is None:
                print(f"[VLM] '{VLM_MODEL}' is not available at "
                      f"{vlm.VLM_HOST} - nothing will be double-checked. "
                      f"Start Ollama and run: ollama pull {VLM_MODEL}")
                if names:
                    print(f"[VLM] Ollama has: {', '.join(names)}")
        return self._available

    def _ensure_thread(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="vlm-arbiter")
        self._thread.start()

    # ---------------------------------------------------------- keys
    @staticmethod
    def person_key(camera, track_id):
        return (PERSON, camera, int(track_id))

    @staticmethod
    def identity_key(camera, track_id, name):
        return (IDENTITY, camera, int(track_id), str(name))

    @staticmethod
    def face_key(camera, track_id):
        return (FACE, camera, int(track_id))

    # --------------------------------------------------------- asking
    def _submit(self, job):
        """Queue a question, unless it is already asked or answered."""
        if not self.usable():
            return False
        with self._lock:
            if job.key in self._waiting:
                return False
            held = self._verdicts.get(job.key)
            if held is not None and not held.stale():
                return False
            self._queue.append(job)
            self._waiting.add(job.key)
            self._queue.sort(key=lambda j: (PRIORITY.get(j.kind, 9), j.queued))
            # Drop from the BACK - the lowest priority, longest waiting.
            # An answer about a track that has since left is not a late
            # answer, it is a wrong one.
            while len(self._queue) > VLM_QUEUE_MAX:
                stale = self._queue.pop()
                self._waiting.discard(stale.key)
                self.dropped += 1
        self._ensure_thread()
        return True

    def check_person(self, camera, track_id, crop, label=""):
        """Ask whether a tracked box is really a human being."""
        if crop is None or getattr(crop, "size", 0) == 0:
            return False
        return self._submit(_Job(self.person_key(camera, track_id), PERSON,
                                 crop.copy(), label=label))

    def check_identity(self, camera, track_id, name, face_crop,
                       code="", label=""):
        """Ask whether this face could be the person we have named.

        The reference side is not passed in: it is a DESCRIPTION of that
        person, computed once from their enrollment photographs and
        cached on disk. Only the live crop needs a question asked about
        it, which is what keeps this affordable.
        """
        if face_crop is None or getattr(face_crop, "size", 0) == 0:
            return False
        if not has_references(name, code):
            return False
        return self._submit(_Job(self.identity_key(camera, track_id, name),
                                 IDENTITY, face_crop.copy(), name=name,
                                 code=code,
                                 label=label or f"{name} #{track_id}"))

    def check_face(self, camera, track_id, crop, label=""):
        """Is this crop worth putting in front of a human at all?

        Answered before a question is created, not after. The queue used
        to fill with doors, floors and shoulders because everything
        upstream of the question is a number and a patch of door texture
        scores well on all of them.
        """
        if crop is None or getattr(crop, "size", 0) == 0:
            return False
        return self._submit(_Job(self.face_key(camera, track_id), FACE,
                                 crop.copy(), label=label))

    # -------------------------------------------------------- answers
    def verdict(self, key):
        with self._lock:
            held = self._verdicts.get(key)
        if held is None or held.stale():
            return None
        return held

    def person_verdict(self, camera, track_id):
        return self.verdict(self.person_key(camera, track_id))

    def identity_verdict(self, camera, track_id, name):
        return self.verdict(self.identity_key(camera, track_id, name))

    def face_verdict(self, camera, track_id):
        return self.verdict(self.face_key(camera, track_id))

    def pending(self, key):
        with self._lock:
            return key in self._waiting

    def forget(self, camera, track_id):
        """Drop everything remembered about a track that has gone."""
        with self._lock:
            for key in [k for k in self._verdicts
                        if len(k) > 2 and k[1] == camera and k[2] == track_id]:
                del self._verdicts[key]

    # --------------------------------------------------------- worker
    def _loop(self):
        while self._running:
            with self._lock:
                job = self._queue.pop(0) if self._queue else None
            if job is None:
                time.sleep(0.2)
                continue

            gap = VLM_MIN_CALL_GAP - (time.time() - self._last_call)
            if gap > 0:
                time.sleep(min(gap, 5.0))

            started = time.time()
            try:
                verdict = self._answer(job)
            except Exception as exc:
                verdict = Verdict(job.kind, None, 0.0, f"error: {exc}")
                if VLM_DEBUG:
                    print(f"[VLM] {job.kind} check failed: {exc}")
            self._last_call = time.time()
            self.last_ms = (self._last_call - started) * 1000.0

            with self._lock:
                self._verdicts[job.key] = verdict
                self._waiting.discard(job.key)
                self.asked += 1
                if verdict.ok is None:
                    self.unsure += 1
                elif verdict.ok:
                    self.passed += 1
                elif job.kind == IDENTITY:
                    self.refused_identity += 1
                elif job.kind == FACE:
                    self.refused_face += 1
                else:
                    self.refused_person += 1

            if VLM_DEBUG and verdict.refuses:
                print(f"[VLM] REFUSED {job.kind} for {job.label or job.key}: "
                      f"{verdict.reason} (confidence {verdict.confidence:.2f}, "
                      f"{self.last_ms:.0f}ms)")
            self._maybe_report()

    def _answer(self, job):
        if job.kind == PERSON:
            ok, confidence, reason = vlm.is_person(job.image)
            return Verdict(PERSON, ok, confidence, reason)

        if job.kind == FACE:
            label, note = vlm.shows(job.image)
            if label is None:
                return Verdict(FACE, None, 0.0, note)
            if label in VLM_REJECT_SHOWS:
                # Nothing here anybody could answer a question about, so
                # there is nothing to describe either.
                return Verdict(FACE, False, 1.0, f"{label} - {note}")
            # Worth asking about, so describe it. The shortlist is built
            # from this description LATER, by the camera, because it also
            # needs the recogniser's ranking and the face embedding -
            # neither of which belongs on this thread.
            attrs = vlm.attributes(job.image)
            return Verdict(FACE, True, 1.0, f"{label} - {note}",
                           attributes=attrs)

        # What this person is SUPPOSED to look like, computed once from
        # the photographs a human took of them and then cached.
        reference = reference_attributes(job.name, job.code)
        if not reference:
            return Verdict(IDENTITY, None, 0.0, "no reference description")
        live = vlm.attributes(job.image)
        if not live:
            return Verdict(IDENTITY, None, 0.0, "could not describe the crop")
        ok, confidence, reason = vlm.same_person(live, reference)
        return Verdict(IDENTITY, ok, confidence, reason)

    # ---------------------------------------------------- diagnostics
    def _maybe_report(self):
        if not VLM_DEBUG:
            return
        now = time.time()
        if now - self._report_t0 < VLM_DEBUG_EVERY_SECONDS:
            return
        self._report_t0 = now
        if not self.asked:
            return
        with self._lock:
            waiting = len(self._queue)
        print(f"[VLM] {self.asked} question(s) asked | "
              f"{self.refused_identity} wrong name(s) refused, "
              f"{self.refused_person} phantom(s) refused, "
              f"{self.refused_face} faceless crop(s) kept out of the "
              f"queue | {self.passed} agreed, {self.unsure} could not "
              f"tell | {waiting} waiting, {self.dropped} dropped | "
              f"last {self.last_ms:.0f}ms")

    def stats(self):
        with self._lock:
            return {"available": bool(self._available),
                    "model": VLM_MODEL,
                    "asked": self.asked,
                    "refused_identity": self.refused_identity,
                    "refused_person": self.refused_person,
                    "refused_face": self.refused_face,
                    "agreed": self.passed,
                    "could_not_tell": self.unsure,
                    "queued": len(self._queue),
                    "dropped": self.dropped,
                    "last_ms": round(self.last_ms)}

    def stop(self):
        self._running = False


# One arbiter for the whole process. The vision model is a single GPU
# resource; giving each camera its own worker would just queue them
# against each other inside Ollama, where nothing can prioritise them.
ARBITER = VlmArbiter()


# ------------------------------------------------- reference pictures
# Cached, because this is read every time somebody is named and the
# folders change only when a photograph is added.
_REF_CACHE = {}
_REF_CACHE_TTL = 300.0
_REF_LOCK = threading.Lock()


def reference_images(name, code="", limit=None, enrollment_only=None):
    """Photographs of one person, for the comparison picture.

    ENROLLMENT PHOTOGRAPHS BY DEFAULT - the ones a human deliberately
    took. The system's own camera captures are exactly what a wrong name
    poisons (see core/face_library.py), so checking a suspect name
    against them would be asking the mistake to audit itself. Same
    reasoning as agrees_with_enrollment().
    """
    limit = int(VLM_IDENTITY_REFERENCES if limit is None else limit)
    if enrollment_only is None:
        enrollment_only = VLM_IDENTITY_ENROLLMENT_ONLY
    if not name or name == "Unknown" or limit <= 0:
        return []

    cache_key = (name, bool(enrollment_only), limit)
    now = time.time()
    with _REF_LOCK:
        held = _REF_CACHE.get(cache_key)
        if held is not None and now - held[0] < _REF_CACHE_TTL:
            return held[1]

    folder = LIBRARY.folder_for(name, code, create=False)
    images = []
    if folder:
        files = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                       if os.path.isfile(p)
                       and p.lower().endswith((".jpg", ".jpeg", ".png",
                                               ".bmp", ".webp")))
        enrolled = [p for p in files
                    if not os.path.basename(p).startswith(CAPTURE_PREFIX)]
        captured = [p for p in files
                    if os.path.basename(p).startswith(CAPTURE_PREFIX)]
        chosen = enrolled if enrollment_only else (enrolled + captured)
        # Spread the choice across the folder rather than taking the
        # first N, which on a folder of burst captures would be three
        # copies of one moment.
        if len(chosen) > limit:
            step = len(chosen) / float(limit)
            chosen = [chosen[int(i * step)] for i in range(limit)]
        for path in chosen[:limit]:
            image = cv2.imread(path)
            if image is not None and image.size:
                images.append(image)

    with _REF_LOCK:
        _REF_CACHE[cache_key] = (now, images)
    return images


def has_references(name, code=""):
    return bool(reference_images(name, code))


# ------------------------------------------ what a person looks like
# A DESCRIPTION of each person, derived once from the photographs a
# human took of them, and kept on disk.
#
# This is what makes checking a name affordable: without it, every check
# would have to describe the reference photographs again, doubling the
# GPU cost of a question that is asked hundreds of times a day about the
# same few dozen people. It is derived from the ENROLLMENT photographs
# only, for the same reason agrees_with_enrollment() is - the system's
# own captures are exactly what a wrong name poisons, and a description
# built from those would describe the mistake.
#
# The file is plain JSON on purpose. If the system refuses somebody's
# name, "what does it think this person looks like?" has to be a
# question anybody can answer by opening a file.
_ATTR_LOCK = threading.RLock()
_ATTR_CACHE = None


def _load_appearances():
    global _ATTR_CACHE
    with _ATTR_LOCK:
        if _ATTR_CACHE is not None:
            return _ATTR_CACHE
        _ATTR_CACHE = {}
        try:
            if os.path.exists(VLM_APPEARANCE_FILE):
                with open(VLM_APPEARANCE_FILE, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    _ATTR_CACHE = loaded
        except Exception as exc:
            print(f"[VLM] could not read {VLM_APPEARANCE_FILE}: {exc}")
        return _ATTR_CACHE


def _save_appearances():
    with _ATTR_LOCK:
        try:
            os.makedirs(os.path.dirname(VLM_APPEARANCE_FILE), exist_ok=True)
            with open(VLM_APPEARANCE_FILE, "w", encoding="utf-8") as fh:
                json.dump(_ATTR_CACHE, fh, indent=2, sort_keys=True)
        except Exception as exc:
            print(f"[VLM] could not write {VLM_APPEARANCE_FILE}: {exc}")


def ranked_candidates(embedding, live_attributes=None, limit=None, pool=None):
    """Who could this unknown face be? Best first.

    THE RECOGNISER RANKS; THE VISION MODEL ONLY STRIKES OUT. Each half
    does the thing it is actually good at, and the reason that split is
    necessary was measured on this site rather than assumed.

    Ranking by DESCRIPTION alone was built first and does not work here.
    Once all 64 enrolled people were described, they fell into three
    buckets: 26 are "male, short hair, beard", 19 are "female, long
    hair, clean-shaven", 11 are "male, beard". Those attributes are
    stable enough to REFUSE a name - which is what they are used for
    elsewhere - but on a staff of this size they cannot pick anybody
    out, so a description-only shortlist was either empty or the whole
    directory.

    The face embeddings CAN separate two colleagues; that is the one
    thing they are for. What they cannot do is notice that their best
    guess is a man when the picture is plainly a woman - every score is
    a distance inside one vector space with no idea what a face is. So
    the recogniser proposes an ordered handful, and any of them whose
    own photographs contradict what is on screen is struck out.

    Still NARROWING, not identifying: a name survives only because
    nothing rules it out. Two colleagues of similar build and haircut
    will both appear, and they are meant to. Only the human answers.
    """
    limit = int(VLM_SHORTLIST_MAX if limit is None else limit)
    pool = int(VLM_CANDIDATE_POOL if pool is None else pool)
    if embedding is None or limit <= 0:
        return []

    try:
        from core.facebank import FACE_BANK
        ranked = FACE_BANK.per_person_scores(embedding)
    except Exception:
        return []
    if not ranked:
        return []
    ranked = ranked[:max(limit, pool)]

    # No usable description means no grounds to strike anybody out, so
    # the recogniser's order stands on its own.
    if not live_attributes or not vlm.says_anything(live_attributes):
        return [name for name, _code, _score in ranked[:limit]]

    store = _load_appearances()
    kept, struck = [], []
    for name, _code, _score in ranked:
        reference = (store.get(name) or {}).get("attributes") \
            if isinstance(store.get(name), dict) else None
        if reference:
            contradiction, _why = vlm.contradictions(live_attributes,
                                                     reference)
            if contradiction > 0:
                struck.append(name)
                continue
        kept.append(name)
        if len(kept) >= limit:
            break
    if VLM_DEBUG and struck:
        print(f"[VLM] struck {', '.join(struck[:4])} off the shortlist - "
              f"the picture shows {vlm.describe(live_attributes)}")
    return kept


def known_people_described():
    """How many people we have a reference description for."""
    return len(_load_appearances())


def reference_attributes(name, code="", refresh=False):
    """How this person's own photographs describe them. None if unknown.

    Computed on first use and cached. Recomputed when the number of
    enrollment photographs changes or the vision model changes, because
    either makes the stored description something other than what the
    current setup would produce.
    """
    if not name or name == "Unknown":
        return None
    images = reference_images(name, code)
    if not images:
        return None

    store = _load_appearances()
    held = store.get(name)
    if (held and not refresh and held.get("photos") == len(images)
            and held.get("model") == VLM_MODEL
            and isinstance(held.get("attributes"), dict)):
        return held["attributes"]

    # One contact sheet of their own photographs, described once. They
    # are all the same person, so there is no contrast bias to worry
    # about here - which is the whole reason the live crop is described
    # on its own instead.
    sheet = images[0] if len(images) == 1 else vlm.montage(
        images[0], images[1:], live_caption="PHOTO 1",
        reference_caption="PHOTO")
    described = vlm.attributes(sheet if sheet is not None else images[0])
    if not described:
        return None

    with _ATTR_LOCK:
        store[name] = {"attributes": described, "photos": len(images),
                       "model": VLM_MODEL, "when": int(time.time()),
                       "source": "enrollment photographs"}
    _save_appearances()
    if VLM_DEBUG:
        summary = " ".join(f"{k}={v}" for k, v in described.items())
        print(f"[VLM] learned what {name} looks like from "
              f"{len(images)} photo(s): {summary}")
    return described
