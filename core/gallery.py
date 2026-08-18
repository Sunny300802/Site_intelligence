"""
core/gallery.py
===============
Remember people the way a person would.

The problem with what we had
----------------------------
Everything a track knew about somebody died with the track. If detection
dropped them for long enough, or they walked out and came back, the
system met a stranger. On a camera where faces are rarely readable that
meant almost everybody stayed "Unknown", and a name that did appear
flickered away a frame later.

What this does instead
----------------------
A gallery of PEOPLE, independent of any track. Each person accumulates
evidence about what they look like, and a new track is matched against
everyone we know rather than only against tracks we happened to be
holding a moment ago.

ONLY A FACE CAN NAME SOMEBODY
-----------------------------
This gallery used to accept CLOTHING as evidence of who somebody is: a
colour-and-pattern signature of what they were wearing that day. It was
the biggest single source of wrong names on this site. Two colleagues in
similar clothes score almost the same; the signature of a person who is
half behind somebody else is a blend of the two; and the moment either
of those happened the name visibly moved to the wrong body.

So the clothes are gone, and what is left is:

  FACE      The only thing allowed to put a name on anybody. Enrollment
            photographs, plus any capture a human has confirmed.

  BODY      How tall this person is for where they are standing, and how
            broad they are for their height (core/body.py). It can never
            name anybody - two people of similar build measure the
            same - but it can REFUSE a name that is about to land on a
            body of obviously the wrong size, which is what stops a face
            misread during an overlap from renaming somebody.

  RECENCY   Somebody seen thirty seconds ago is a better bet than
            somebody last seen yesterday. A supporting cue only, for the
            same reason.

What this costs, honestly
-------------------------
A camera that cannot see faces can no longer name people by their
clothes. On the work-area camera that means somebody is named when their
face is briefly readable, or when the tracker carries a name over from
where it WAS readable, or by their desk (core/seats.py) - and stays
"Unknown #12" otherwise. An honest Unknown is worth more than a
confident wrong name in an attendance record, which is the whole
argument for this change.

Honest uncertainty
------------------
When the best candidate is not clearly better than the runner-up, the
answer is "not sure" rather than a guess. Those cases go to a review
queue for a human to confirm, and a confirmation is learned permanently.
"""
import time
import threading
from datetime import datetime, date

import numpy as np

from core.body import similarity as body_similarity
from core.facebank import combine as facebank_combine
from config.settings import (GALLERY_FACE_CERTAIN, GALLERY_FACE_USELESS,
                             GALLERY_ACCEPT_PROBABILITY,
                             GALLERY_REVIEW_PROBABILITY, GALLERY_MIN_MARGIN,
                             GALLERY_BODY_MIN, BODY_IDENTITY_MIN)


# ----------------------------------------------------------- weighting
# The face is the answer. The other two adjust confidence in it; they
# cannot produce one.
W_FACE = 0.86
W_BODY = 0.09
W_RECENCY = 0.05

# Every threshold below now lives in config/settings.py, so identification
# can be tuned in the one place all other settings are - and so the
# "a face is proof on its own" bar cannot drift above
# FACE_MATCH_THRESHOLD, which is what silently sent every real match
# between the two numbers to the review queue instead of the dashboard.

# A face this good is treated as proof on its own.
FACE_CERTAIN = GALLERY_FACE_CERTAIN
# Below this, a face contributes nothing rather than misleading us.
FACE_USELESS = GALLERY_FACE_USELESS

# How badly the sizes have to disagree before a face match is refused
# outright. Deliberately low: this is the "that name is landing on
# somebody a head shorter" test, not a measurement of who anybody is.
BODY_MIN = GALLERY_BODY_MIN

# Decisions
ACCEPT_PROBABILITY = GALLERY_ACCEPT_PROBABILITY   # name them
REVIEW_PROBABILITY = GALLERY_REVIEW_PROBABILITY   # not sure - ask a human
MIN_MARGIN = GALLERY_MIN_MARGIN     # ...and only if clearly ahead of second

FACE_MEMORY = 40               # face embeddings kept per person
BODY_SAMPLES = 60              # size measurements kept per person


def _valid_vector(vec, minimum=16):
    """Return a clean 1-D float32 vector, or None if it is unusable.

    Anything that is not a proper vector - None, a scalar, an empty crop,
    a truncated database row, a vector full of zeros - is rejected here
    rather than being allowed to poison later comparisons.
    """
    if vec is None:
        return None
    try:
        arr = np.asarray(vec, dtype=np.float32).ravel()
    except Exception:
        return None
    if arr.ndim != 1 or arr.size < minimum:
        return None
    if not np.isfinite(arr).all():
        return None
    if float(np.abs(arr).sum()) <= 1e-6:
        return None
    return arr


def _norm(vec):
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


class PersonProfile:
    """Everything we know about one person."""

    def __init__(self, name, code="", gender=""):
        self.name = name
        self.code = str(code or "")
        self.gender = (gender or "").lower()

        self._faces = []          # normalised embeddings
        self._faces_permanent = 0  # how many came from enrollment/confirmed
        # How big this person is - see core/body.py. Kept across days,
        # unlike the clothing this replaced, because a person's height
        # and build do not change overnight.
        self._bodies = []          # Body measurements

        self.last_seen = None
        self.last_camera = None
        self.times_seen = 0
        self.times_confirmed = 0

    # ------------------------------------------------------- learning
    def add_face(self, embedding, permanent=False):
        embedding = _valid_vector(embedding, minimum=64)
        if embedding is None:
            return
        vec = _norm(embedding)
        self._faces.append(vec)
        if permanent:
            self._faces_permanent += 1
        # keep permanent ones, trim the learned tail
        if len(self._faces) > FACE_MEMORY:
            keep = self._faces[:self._faces_permanent]
            learned = self._faces[self._faces_permanent:]
            self._faces = keep + learned[-(FACE_MEMORY - len(keep)):]

    def add_body(self, body):
        """Remember how big this person measured.

        Only ever called for somebody a FACE identified, so this is a
        record of a known person's size rather than a guess that could
        then be used to guess again.
        """
        if body is None or not getattr(body, "build", None):
            return
        self._bodies.append(body)
        if len(self._bodies) > BODY_SAMPLES:
            self._bodies.pop(0)

    def mark_seen(self, camera=None):
        self.last_seen = time.time()
        self.last_camera = camera
        self.times_seen += 1

    # ------------------------------------------------------- evidence
    def face_score(self, embedding):
        """How well does this face match THIS person's references?

        Combined by core/facebank.combine so the gallery and the face
        bank answer this question identically. They used to disagree -
        the gallery took a plain maximum while the bank blends the
        centroid with the best few - and two different answers to the
        same question is how a name gets accepted by one path and
        refused by the other.
        """
        embedding = _valid_vector(embedding, minimum=64)
        if embedding is None or not self._faces:
            return None
        vec = _norm(embedding)
        usable = [f for f in self._faces if f.shape == vec.shape]
        if not usable:
            return None
        stack = np.stack(usable)
        centroid = stack.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        return facebank_combine(stack @ vec,
                                centroid_similarity=float(centroid @ vec))

    def body_score(self, body):
        """Is this the right size to be this person? None = cannot say.

        The BEST of the remembered measurements, not the average. People
        are measured standing, walking, half behind a desk and carrying
        things, and the spread of that is real; asking "does this match
        any way we have seen them" is the question that has a useful
        answer.
        """
        if body is None or not self._bodies:
            return None
        scores = [s for s in (body_similarity(known, body)
                              for known in self._bodies) if s is not None]
        return max(scores) if scores else None

    def recency_score(self, within_seconds=900.0):
        if self.last_seen is None:
            return 0.0
        age = time.time() - self.last_seen
        if age >= within_seconds:
            return 0.0
        return 1.0 - (age / within_seconds)

    def summary(self):
        return {
            "name": self.name, "code": self.code, "gender": self.gender,
            "faces": len(self._faces),
            "faces_permanent": self._faces_permanent,
            "body_samples": len(self._bodies),
            "times_seen": self.times_seen,
            "times_confirmed": self.times_confirmed,
            "last_seen": (datetime.utcfromtimestamp(self.last_seen).isoformat()
                          if self.last_seen else None),
            "last_camera": self.last_camera,
        }


class Match:
    """The outcome of asking the gallery 'who is this?'"""

    __slots__ = ("name", "code", "probability", "runner_up", "margin",
                 "evidence", "decision", "needed_margin")

    def __init__(self, name, code, probability, runner_up, margin,
                 evidence, decision, needed_margin=MIN_MARGIN):
        self.name = name
        self.code = code
        self.probability = probability
        self.runner_up = runner_up
        self.margin = margin
        self.needed_margin = needed_margin
        self.evidence = evidence
        self.decision = decision     # "accept" | "review" | "reject"

    def __repr__(self):
        return (f"<Match {self.name!r} p={self.probability:.2f} "
                f"margin={self.margin:.2f} {self.decision}>")

    def why(self):
        """A plain-English reason, so a wrong answer can be diagnosed
        instead of guessed at."""
        cues = ", ".join(f"{k} {v:.2f}" for k, v in self.evidence.items()
                         if isinstance(v, float) and v > 0)
        if self.decision == "accept":
            return f"{self.name} ({cues})"
        if self.decision == "review":
            if self.margin < self.needed_margin and self.runner_up:
                return (f"could be {self.name} or {self.runner_up} - only "
                        f"{self.margin:.2f} between them, "
                        f"{self.needed_margin} needed ({cues})")
            return (f"probably {self.name} at {self.probability:.2f}, "
                    f"below the {ACCEPT_PROBABILITY} needed ({cues})")
        body = self.evidence.get("body")
        if body is not None and body < BODY_MIN:
            return (f"the face said {self.name}, but this body is the wrong "
                    f"size for them (fits {body:.2f}, {BODY_MIN} needed) "
                    f"- refused")
        return f"no confident match ({cues or 'no face was readable'})"


class PersonGallery:
    """Everyone we know, shared across cameras."""

    def __init__(self):
        self.people = {}          # name -> PersonProfile
        self._lock = threading.Lock()

    # ------------------------------------------------------- building
    def add_person(self, name, code="", gender=""):
        with self._lock:
            if name not in self.people:
                self.people[name] = PersonProfile(name, code, gender)
            return self.people[name]

    def load_enrollment(self, embeddings, names, codes=None):
        """Seed the gallery from enrolled photos."""
        codes = codes or [""] * len(names)
        for emb, name, code in zip(embeddings, names, codes):
            self.add_person(name, code).add_face(emb, permanent=True)
        return len(self.people)

    # ------------------------------------------------------- matching
    def identify(self, face=None, body=None, exclude=()):
        """Who is this? Returns a Match, or None if we know nobody.

        NO FACE, NO ANSWER. If nothing readable came off this person's
        face, the honest reply is None and every caller treats that as
        "still Unknown". Guessing from the rest is what this system used
        to do, and what it was wrong about.

        `exclude` lets a camera rule out people it is already showing
        elsewhere in the same frame - one person cannot be in two places.
        """
        if face is None:
            return None

        with self._lock:
            candidates = []
            for name, profile in self.people.items():
                if name in exclude:
                    continue
                prob, evidence = self._score(profile, face, body)
                if prob > 0:
                    candidates.append((prob, name, profile, evidence))

        if not candidates:
            return None

        candidates.sort(reverse=True, key=lambda c: c[0])
        best_p, best_name, best_profile, evidence = candidates[0]
        runner_p = candidates[1][0] if len(candidates) > 1 else 0.0
        runner_name = candidates[1][1] if len(candidates) > 1 else None
        margin = best_p - runner_p
        needed_margin = MIN_MARGIN

        if best_p >= ACCEPT_PROBABILITY and margin >= needed_margin:
            decision = "accept"
        elif best_p >= REVIEW_PROBABILITY:
            decision = "review"       # plausible but not certain - ask
        else:
            decision = "reject"

        # THE SIZE VETO.
        #
        # The face said this person; the body says it cannot be. That
        # happens when a face is read off the wrong body - two people
        # overlapping at the door, one face visible between them - and it
        # is the last remaining route by which a name could jump to
        # somebody else. A measurement this far out is not a close call,
        # so it is refused rather than sent for review: there is nothing
        # a human could usefully confirm about a face that was never on
        # this body.
        fits = evidence.get("body")
        if fits is not None and fits < BODY_MIN:
            decision = "reject"

        return Match(best_name, best_profile.code, best_p, runner_name,
                     margin, evidence, decision, needed_margin)

    @staticmethod
    def _score(profile, face, body):
        """Combine whatever evidence we have into one probability."""
        f = profile.face_score(face)
        b = profile.body_score(body)
        r = profile.recency_score()

        evidence = {"face": f, "body": b, "recency": r}

        # No face score means this person has no references to compare
        # against - not that they are somebody else.
        if f is None:
            return 0.0, evidence

        # A face that is present but poor is evidence AGAINST, not noise
        # to be ignored: it means we looked and it did not match.
        if f < FACE_USELESS:
            return 0.0, evidence

        # A clearly good face is proof by itself - no other cue should be
        # able to talk us out of it. (The size veto in identify() still
        # applies: that is not a cue disagreeing, it is a statement that
        # this face cannot have come off this body.)
        if f >= FACE_CERTAIN:
            span = max(1e-6, 1.0 - FACE_CERTAIN)
            return min(1.0, 0.75 + 0.25 * (f - FACE_CERTAIN) / span), evidence

        parts, weights = [f], [W_FACE]
        if b is not None:
            parts.append(b)
            weights.append(W_BODY)
        parts.append(r)
        weights.append(W_RECENCY)

        return float(np.average(parts, weights=weights)), evidence

    # ------------------------------------------------------- learning
    def learn(self, name, face=None, body=None, camera=None,
              permanent=False):
        """Add evidence for somebody we are confident about."""
        with self._lock:
            profile = self.people.get(name)
            if profile is None:
                return False
            if face is not None:
                profile.add_face(face, permanent=permanent)
            if body is not None:
                profile.add_body(body)
            profile.mark_seen(camera)
            if permanent:
                profile.times_confirmed += 1
            return True

    def size_allows(self, name, body):
        """Could this name belong to a body of this size?

        Asked before a name is put on a track. True when we have no
        measurements for that person yet, or when the sizes are
        compatible; False only when they clearly are not. This is what
        stops a name moving to the wrong person after two people who
        overlapped separate again.
        """
        if body is None:
            return True
        with self._lock:
            profile = self.people.get(name)
        if profile is None:
            return True
        fits = profile.body_score(body)
        return fits is None or fits >= BODY_IDENTITY_MIN

    def names(self):
        with self._lock:
            return sorted(self.people)

    def summary(self):
        with self._lock:
            return [p.summary() for p in self.people.values()]

    def stats(self):
        with self._lock:
            return {
                "people": len(self.people),
                "with_size_measured": sum(1 for p in self.people.values()
                                          if p._bodies),
                "total_faces": sum(len(p._faces) for p in self.people.values()),
            }


# ---------------------------------------------------------------------
# One gallery shared by every camera. This is deliberate: a person whose
# face is readable at the reception door teaches the system what they are
# wearing, and the workspace camera - which can almost never see a face -
# can then recognise them by those clothes for the rest of the day.
# ---------------------------------------------------------------------
GALLERY = PersonGallery()
