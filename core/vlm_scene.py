"""
core/vlm_scene.py
=================
"What is happening in the server room?" - answered by looking.

    >>> from core.vlm_scene import look
    >>> report = look("server_rm_psg")
    >>> report.sentence()
    '3 people in Server Rm Psg right now. Pavan Kumar is working at a
     desk (28 min), Ayesha Fatima is on the phone (1h 4m), plus 1
     unidentified person moving around the room.'

WHO AND WHAT COME FROM DIFFERENT PLACES, AND THAT IS THE WHOLE DESIGN
---------------------------------------------------------------------
Everywhere else in this system the vision model is a VETO - it may
refuse a name and never grant one - because a 7B model looking at a
40-pixel CCTV face cannot tell two colleagues apart and will invent an
answer if asked to. Nothing here changes that. What changes is the
question:

    WHO is that?        the tracker and the face vote. Always.
    WHAT are they doing? the vision model, one person crop at a time.

The model is handed a crop of ONE person and asked what that person is
doing. It is never told a name, never asked for one, and never shown the
annotated frame with names drawn on it (see SCENE_SNAPSHOT_ENABLED). The
answer is joined to a name in code, here, where the join can be read.

That division is what makes the feature safe. Posture survives a bad
camera in a way a face does not: whether somebody is facing a monitor
with their hands at a keyboard, holding a phone to their ear, or walking
across a room is carried by exactly the coarse shape a CCTV frame still
has. So the model is asked only about the part of the picture it can
actually see.

WHAT IT COSTS, AND WHY IT IS NOWHERE NEAR THE FRAME LOOP
--------------------------------------------------------
One question per visible person, plus one for the whole frame, at
1.5-5 s each. A busy room is therefore tens of seconds - fine for a
question somebody typed, catastrophic at 25 fps. So:

  * the pipeline never calls this. It only publishes snapshots.
  * an answer is cached for VLM_SCENE_CACHE_SECONDS, so the second
    person to ask the same question pays nothing.
  * a wall-clock budget stops the run, and the people who were not
    looked at are REPORTED as not looked at rather than dropped.

If Ollama is not running, or the pipeline is not running, every path
here degrades to the deterministic half - who the tracker says is there,
with no activity - and says so.
"""
import base64
import json
import time
import threading
import urllib.request
import urllib.error

import cv2
import numpy as np

from config.settings import (STREAM_PORT, VLM_ENABLED, VLM_SCENE_ENABLED,
                             VLM_SCENE_MAX_PEOPLE, VLM_SCENE_BUDGET_SECONDS,
                             VLM_SCENE_TIMEOUT, VLM_SCENE_CACHE_SECONDS,
                             VLM_SCENE_STALE_SECONDS,
                             VLM_SCENE_MIN_HEIGHT_FRAC,
                             VLM_SCENE_RETURN_IMAGE, VLM_DEBUG)
from core import vlm


# ------------------------------------------------------------- snapshots
def _local_scene(camera_key):
    """The snapshot, if we are running inside the pipeline process."""
    try:
        from streaming.mjpeg import STORE
    except Exception:
        return None
    scene = STORE.scene(camera_key)
    if scene is None:
        return None
    return {"jpeg": scene["jpeg"], "people": scene["people"],
            "at": scene["at"]}


def _remote_scene(camera_key, timeout=4.0):
    """The snapshot over HTTP, which is the normal case.

    The dashboard and the pipeline are separate processes, so the
    assistant reaches the live frame the same way the browser reaches
    the video: through the little MJPEG server the pipeline runs.
    """
    base = f"http://127.0.0.1:{STREAM_PORT}"
    try:
        with urllib.request.urlopen(f"{base}/scene/{camera_key}",
                                    timeout=timeout) as r:
            meta = json.loads(r.read())
    except Exception:
        return None
    if not isinstance(meta, dict) or "people" not in meta:
        return None
    try:
        with urllib.request.urlopen(f"{base}/snapshot/{camera_key}",
                                    timeout=timeout) as r:
            jpeg = r.read()
    except Exception:
        return None
    # The age travels with the picture rather than being recomputed from
    # a local clock: the two processes are on one machine today, and an
    # assumption like that is exactly the kind that stops being true
    # quietly.
    return {"jpeg": jpeg, "people": meta["people"],
            "at": time.time() - float(meta.get("age") or 0.0)}


def snapshot(camera_key):
    """The newest frame for a camera and who was in it, or None."""
    return _local_scene(camera_key) or _remote_scene(camera_key)


def _decode(jpeg):
    if not jpeg:
        return None
    buf = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ------------------------------------------------------------- the report
class Person:
    """One person in the frame: a name from the tracker, an activity from
    the vision model, and never the other way round."""

    __slots__ = ("person", "track_id", "seconds", "box", "counted",
                 "activity", "at_desk", "_note", "checked")

    def __init__(self, entry):
        self.person = (entry.get("person") or "").strip()
        self.track_id = entry.get("track_id")
        self.seconds = int(entry.get("seconds") or 0)
        self.box = [int(v) for v in (entry.get("box") or [0, 0, 0, 0])]
        self.counted = bool(entry.get("counted", True))
        self.activity = ""        # "" until the model has been asked
        self.at_desk = ""
        self._note = ""
        self.checked = False      # was the model asked about this one?

    @property
    def note(self):
        return self._note

    @note.setter
    def note(self, value):
        # The model ends its note with a full stop about half the time,
        # and every caller puts this in the middle of a sentence. Two
        # full stops in a row is the sort of thing that makes an answer
        # look machine-made when everything else about it is careful.
        self._note = (value or "").strip().rstrip(".").strip()

    @property
    def known(self):
        return bool(self.person)

    @property
    def label(self):
        return self.person or f"Unidentified #{self.track_id}"

    @property
    def doing(self):
        """The activity in words, for a sentence somebody has to read."""
        if not self.checked:
            return "not looked at"
        return vlm.ACTIVITY_WORDS.get(self.activity, "not clear")

    @property
    def working(self):
        """True / False / None. None means we do not know, and it is
        reported as not knowing rather than as a no."""
        if not self.checked or self.activity in ("", "unclear"):
            return None
        if self.activity in vlm.ACTIVITY_WORKING:
            return True
        if self.activity in vlm.ACTIVITY_NOT_WORKING:
            return False
        return None               # on the phone, talking - neither

    def row(self):
        return {"person": self.label, "doing": self.doing,
                "activity": self.activity or "not checked",
                "for": _humanise(self.seconds),
                "note": self.note, "track_id": self.track_id}


class SceneReport:
    """What one look at one camera found."""

    def __init__(self, camera_key, camera_name):
        self.camera_key = camera_key
        self.camera_name = camera_name
        self.at = 0.0
        self.age = 0.0
        self.people = []
        self.description = ""      # the whole-frame sentence, if asked for
        self.described = False     # ...and whether we have tried to get one
        self.model_count = None    # the model's own headcount
        # The picture this report describes, kept so a cached look can be
        # topped up with the whole-frame sentence later WITHOUT taking a
        # fresh snapshot. A newer frame would describe a room that the
        # per-person rows below no longer match.
        self.frame_jpeg = b""
        self.skipped = []          # people the budget did not reach
        self.error = ""            # why there is nothing to report, in words
        # ...and the same thing as a code, so a caller looking at
        # several cameras can tell "they all failed for the same reason"
        # from "one of them failed differently" without matching on
        # English. "" while nothing has gone wrong.
        self.reason = ""           # "no_snapshot" | "unreadable"
        self.image = b""           # annotated frame, for showing the reader

    @property
    def ok(self):
        return not self.error

    @property
    def stale(self):
        return self.age > VLM_SCENE_STALE_SECONDS

    def named(self):
        return [p for p in self.people if p.known]

    def find(self, name):
        """The person with this name, if they are in the frame."""
        wanted = (name or "").strip().lower()
        for person in self.people:
            if person.person.lower() == wanted:
                return person
        return None

    def rows(self):
        return [p.row() for p in self.people]

    def freshness(self):
        """How current this is, in words - only when it is worth saying."""
        if self.age < 15:
            return ""
        if self.stale:
            return (f" (this is the last frame the pipeline published, "
                    f"{_humanise(int(self.age))} ago - it may not be "
                    f"running)")
        return f" (as of {int(self.age)}s ago)"

    def disagreement(self):
        """Said out loud when the model and the tracker count differently.

        The description is the model's own words about the whole frame,
        so it can say "three individuals" over a report that names two -
        which without this reads as the assistant contradicting itself.

        It is worth more than tidying away. The count reported is always
        the TRACKER's, because that is the one attached to names and to
        the attendance record; the model's is an independent second
        opinion on the same picture, and the two disagreeing is the
        clearest signal available that a seated person behind a monitor
        is being missed. That is a known, documented weakness of this
        camera (see min_person_height_frac in config/cameras.py), and
        this is the only place a person would ever see it happen.
        """
        if self.model_count is None or self.model_count == len(self.people):
            return ""
        if self.model_count > len(self.people):
            return (f"(The vision model counts {self.model_count} people in "
                    f"the same picture, so somebody may not be being "
                    f"tracked - the count above is the tracker's.)")
        return (f"(The vision model counts only {self.model_count} people in "
                f"the same picture; the count above is the tracker's.)")

    def sentence(self):
        """The answer itself, as a paragraph somebody can read."""
        if self.error:
            return self.error
        where = self.camera_name or self.camera_key
        if not self.people:
            return f"Nobody is in view on {where} right now.{self.freshness()}"

        parts = [f"{len(self.people)} person(s) in view on {where}"
                 f"{self.freshness()}."]
        if self.description:
            parts.append(self.description)
        parts.append(self.disagreement())

        told = []
        for person in self.people:
            bit = f"{person.label} is {person.doing}"
            if person.seconds:
                bit += f" ({_humanise(person.seconds)} present)"
            if not person.counted:
                bit += ", outside the monitored area"
            told.append(bit)
        if told:
            parts.append("; ".join(told) + ".")
        if self.skipped:
            parts.append(f"{len(self.skipped)} more person(s) were not "
                         f"looked at - the vision model ran out of time.")
        return " ".join(p for p in parts if p)


# ------------------------------------------------------------- helpers
def _humanise(seconds):
    seconds = int(seconds or 0)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60} min"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _priority(person, focus):
    """Who gets looked at first when the budget is tight.

    The people SOMEBODY ASKED ABOUT come first - a question about Pavan
    that spends its budget on four strangers and then reports that Pavan
    was not looked at is the one outcome worth designing against. After
    that, identified people, then whoever has been here longest, because
    a name and a long stay are what make an answer worth reading.
    """
    asked = person.person.lower() in focus if person.person else False
    return (not asked, not person.known, -person.seconds)


def _crop(frame, box, pad=0.06):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px = int(pad * max(1, x2 - x1))
    py = int(pad * max(1, y2 - y1))
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


_COLOURS = {                      # BGR, matching the live stream's palette
    "working": (80, 200, 120),
    "on_phone": (0, 170, 255),
    "talking": (0, 170, 255),
    "walking": (200, 200, 0),
    "idle": (80, 130, 240),
}


def _annotate(frame, people):
    """Draw the answer onto the frame it was read off.

    Not decoration. Every other route the assistant has shows its
    working - the deterministic handlers return their rows, a generated
    query shows its SQL - and a described scene has to do the same or it
    is the one answer nobody can check. This is also where a wrong join
    becomes obvious at a glance: if the box labelled Pavan is around
    somebody else, you can see it.
    """
    canvas = frame.copy()
    for person in people:
        x1, y1, x2, y2 = person.box
        colour = _COLOURS.get(person.activity, (160, 160, 160))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        text = f"{person.label}: {person.doing}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 8 if y1 - 8 > th else y2 + th + 8
        tx = max(0, min(x1, canvas.shape[1] - tw - 8))
        cv2.rectangle(canvas, (tx - 4, ty - th - 6), (tx + tw + 6, ty + 6),
                      (30, 30, 30), -1)
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


# ---------------------------------------------------------------- cache
# One question about a room usually arrives with three more behind it -
# "what is happening in there", "is Pavan working", "who is not working"
# - and re-running thirty seconds of GPU for each of them would make the
# feature unusable. So a look is kept for VLM_SCENE_CACHE_SECONDS and
# every question in that window reads the same one.
#
# Keyed by camera only, and deliberately: the cached report describes
# EVERYBODY in the frame, so it answers a question about one person just
# as well as the question that produced it.
_CACHE = {}
_CACHE_LOCK = threading.RLock()


def _cached(camera_key):
    with _CACHE_LOCK:
        entry = _CACHE.get(camera_key)
    if entry is None:
        return None
    report, when = entry
    if time.time() - when > VLM_SCENE_CACHE_SECONDS:
        return None
    # The age has to keep moving while the report sits in the cache, or
    # a 25-second-old answer keeps claiming to be 2 seconds old.
    report.age = time.time() - report.at
    return report


def _remember(camera_key, report):
    with _CACHE_LOCK:
        _CACHE[camera_key] = (report, time.time())


def forget(camera_key=None):
    """Drop cached looks. For tests, and for a forced refresh."""
    with _CACHE_LOCK:
        if camera_key is None:
            _CACHE.clear()
        else:
            _CACHE.pop(camera_key, None)


def _describe_into(report):
    """Add the whole-frame sentence to a report that has none.

    Works off the picture the report was built from, not a fresh one:
    the per-person rows describe that frame, and a sentence about a
    different one would quietly contradict them.
    """
    report.described = True             # tried, whatever comes back
    if not (VLM_ENABLED and VLM_SCENE_ENABLED) or not report.frame_jpeg:
        return
    frame = _decode(report.frame_jpeg)
    if frame is None:
        return
    count, description = vlm.scene(frame, timeout=VLM_SCENE_TIMEOUT)
    if description:
        report.description = description
        report.model_count = count


# ----------------------------------------------------------------- look
def look(camera_key, camera_name="", focus=(), describe_scene=True,
         refresh=False, budget=None):
    """Look at a camera and report who is there and what they are doing.

    Never raises. Every failure - no pipeline, no Ollama, no people -
    comes back as a report that says so, because this is called from a
    chat handler and an exception there is a blank answer.

    `focus` is the names somebody asked about; they are looked at first
    when the budget is tight. `describe_scene` adds the whole-frame
    sentence, which costs one extra call and is not worth it for "is
    Pavan working?".

    `budget` overrides VLM_SCENE_BUDGET_SECONDS for this call, and a
    caller looking at several cameras MUST divide the budget between
    them. Somebody typed this question and is watching a spinner: the
    limit that matters is the total wait, not the wait per camera, and
    the default applied twice is a ninety-second answer.
    """
    if not refresh:
        cached = _cached(camera_key)
        if cached is not None:
            # "Who is working" does not need the room described, so it
            # does not pay for it - but the answer it caches is then the
            # one "what is happening in there" reads a moment later, and
            # that question is mostly the description. So the cached
            # look is topped up with the one missing call rather than
            # being either re-run in full or answered short.
            if describe_scene and not cached.described:
                _describe_into(cached)
            return cached

    report = SceneReport(camera_key, camera_name)

    scene = snapshot(camera_key)
    if scene is None:
        report.reason = "no_snapshot"
        report.error = (
            f"I cannot see {camera_name or camera_key} right now - the "
            f"pipeline is not publishing frames for it. Start "
            f"run_pipeline.py, or check that camera is enabled.")
        return report

    report.at = scene["at"]
    report.age = max(0.0, time.time() - scene["at"])

    frame = _decode(scene["jpeg"])
    if frame is None:
        report.reason = "unreadable"
        report.error = (f"The last frame from "
                        f"{camera_name or camera_key} could not be read.")
        return report

    height = frame.shape[0]
    report.frame_jpeg = scene["jpeg"]
    everyone = [Person(e) for e in scene["people"]]
    report.people = everyone
    if not everyone:
        return report

    if not (VLM_ENABLED and VLM_SCENE_ENABLED):
        # The deterministic half still works and is still worth having:
        # who is there, for how long. Only the "doing" is missing, and
        # it says so rather than pretending.
        if VLM_SCENE_RETURN_IMAGE:
            report.image = _annotate(frame, everyone)
        _remember(camera_key, report)
        return report

    started = time.time()
    focus = {n.strip().lower() for n in (focus or []) if n}
    budget = VLM_SCENE_BUDGET_SECONDS if budget is None else max(0.0, budget)

    if describe_scene:
        report.described = True
        count, description = vlm.scene(frame, timeout=VLM_SCENE_TIMEOUT)
        if description:
            report.description = description
            report.model_count = count

    order = sorted(everyone, key=lambda p: _priority(p, focus))

    asked = 0
    for person in order:
        if asked >= VLM_SCENE_MAX_PEOPLE:
            report.skipped.append(person.label)
            continue
        if time.time() - started >= budget:
            report.skipped.append(person.label)
            continue
        box_height = person.box[3] - person.box[1]
        if box_height < VLM_SCENE_MIN_HEIGHT_FRAC * height:
            # Too small to say anything about, and the model would say
            # something anyway. Same bar as VLM_PERSON_MIN_HEIGHT_FRAC.
            #
            # The REASON is recorded, not just the omission: "not looked
            # at" with nothing beside it reads as the system having run
            # out of time, when in fact this person will never be
            # described however long it is given.
            person.note = "too small in frame to judge"
            continue
        crop = _crop(frame, person.box)
        if crop is None:
            person.note = "no usable crop"
            continue
        label, at_desk, note = vlm.activity(crop, timeout=VLM_SCENE_TIMEOUT)
        asked += 1
        if label is None:
            person.note = note
            continue
        person.activity = label
        person.at_desk = at_desk
        person.note = note
        person.checked = True

    if VLM_DEBUG:
        print(f"[SCENE] {camera_name or camera_key}: {len(everyone)} person(s), "
              f"{asked} asked about, {len(report.skipped)} skipped, "
              f"{time.time() - started:.1f}s")

    if VLM_SCENE_RETURN_IMAGE:
        report.image = _annotate(frame, everyone)
    _remember(camera_key, report)
    return report


def image_data_uri(report):
    """The annotated frame as a data: URI, or "" - for the chat page."""
    if not report or not report.image:
        return ""
    return ("data:image/jpeg;base64,"
            + base64.b64encode(report.image).decode("ascii"))


def available():
    """Can this answer questions at all? (snapshots + a vision model)"""
    from config.cameras import enabled_cameras
    cameras = [c["key"] for c in enabled_cameras()]
    live = [k for k in cameras if snapshot(k) is not None]
    reachable, _models = vlm.available()
    return {"enabled": bool(VLM_ENABLED and VLM_SCENE_ENABLED),
            "cameras_with_snapshots": live,
            "vision_model": "reachable" if reachable else "unreachable"}
