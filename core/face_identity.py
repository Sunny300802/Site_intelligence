"""
core/face_identity.py
=====================
Per-track identity state: which frame to recognise, and how the answers
add up to a name.

Two problems are solved here, and they are the two that actually cause
wrong names on a CCTV dashboard.

1. RECOGNISING THE WRONG FRAME
   The old pipeline recognised whatever face happened to be in the frame
   when the worker was free. Most of those frames are poor - somebody
   mid-stride, half turned, backlit at the door - and a poor frame does
   not produce "no answer", it produces a confident wrong one.

   Instead, each track keeps the BEST face it has produced so far,
   scored by core/face_quality.py, and recognition runs on that. A new
   face only displaces the incumbent if it is better by
   BEST_FRAME_IMPROVEMENT, so an essentially identical frame does not
   trigger pointless GPU work, and a genuinely better look jumps the
   queue immediately rather than waiting for the next interval - waiting
   eight frames to use the best view of somebody all day is exactly the
   wrong trade.

2. DECIDING FROM ONE FRAME
   A single frame was allowed to name somebody, and the next frame was
   allowed to rename them. That is what identity switching IS: not a bad
   model, a decision made from one sample.

   Instead every recognition is a VOTE, weighted by how confident the
   match was AND how good the face was:

       Track 25
         frame 1 -> A 0.87    frame 4 -> A 0.89
         frame 2 -> A 0.91    frame 5 -> A 0.93
         frame 3 -> B 0.54
       final identity: A

   A name is only assigned once it holds enough of the recent window,
   by enough of a margin, over enough separate observations. And once
   assigned, it is DEFENDED: a contradicting name has to build up
   VOTE_OVERRIDE_FACTOR times the confirmed name's weight before it can
   take over, so one strong wrong frame cannot rename anybody - while a
   genuine tracking error, where every frame afterwards says somebody
   else, still corrects itself within a few seconds.

Nothing here touches the GPU or the database. It is pure bookkeeping, so
it can be reasoned about and tested on its own - which matters, because
this is the layer that decides what ends up in the attendance record.
"""
import time
import threading
from collections import deque

from config.settings import (VOTING_WINDOW, MIN_VOTES, VOTE_MIN_SHARE,
                             VOTE_MIN_MARGIN, VOTE_MAX_AGE_SECONDS,
                             VOTE_OVERRIDE_FACTOR, VOTE_OVERRIDE_MIN_VOTES,
                             VOTE_UNKNOWN_BELOW, FACE_QUALITY_THRESHOLD,
                             BEST_FRAME_QUALITY_THRESHOLD,
                             BEST_FRAME_IMPROVEMENT,
                             BEST_FRAME_MAX_AGE_SECONDS,
                             BEST_FRAME_FORCES_RECOGNITION,
                             RECOGNITION_INTERVAL,
                             RECOGNITION_INTERVAL_CONFIRMED)

UNKNOWN = "Unknown"

# A vote's weight is its similarity scaled by the quality of the face it
# came from. Quality never zeroes a vote outright - a 0.45-quality face
# that clears every gate is still real evidence - it just counts for
# less than a clean one. These two numbers are that floor and that span.
_QUALITY_FLOOR = 0.40
_QUALITY_SPAN = 0.60

# What an "I recognised nobody" observation is worth as evidence against
# every name. Deliberately substantial: a track that keeps failing to
# match anybody should not slowly accumulate its way onto a name from
# the handful of frames that did.
_UNKNOWN_WEIGHT = 0.50


class Vote:
    __slots__ = ("name", "score", "quality", "when", "weight")

    def __init__(self, name, score, quality, when):
        self.name = name or UNKNOWN
        self.score = float(score)
        self.quality = float(quality)
        self.when = float(when)
        base = _QUALITY_FLOOR + _QUALITY_SPAN * max(0.0, min(1.0, quality))
        strength = _UNKNOWN_WEIGHT if self.name == UNKNOWN else max(0.0, self.score)
        self.weight = base * strength

    def __repr__(self):
        return f"<Vote {self.name} {self.score:.2f}@q{self.quality:.2f}>"


class Verdict:
    """What the votes currently say about one track."""

    __slots__ = ("name", "code", "score", "confirmed", "votes", "share",
                 "margin", "runner_up", "reason")

    def __init__(self, name=UNKNOWN, code="", score=0.0, confirmed=False,
                 votes=0, share=0.0, margin=0.0, runner_up="", reason=""):
        self.name = name
        self.code = code
        self.score = float(score)
        self.confirmed = bool(confirmed)
        self.votes = int(votes)
        self.share = float(share)
        self.margin = float(margin)
        self.runner_up = runner_up
        self.reason = reason

    @property
    def known(self):
        return bool(self.name) and self.name != UNKNOWN

    def __repr__(self):
        return (f"<Verdict {self.name!r} {self.score:.2f} "
                f"{'CONFIRMED' if self.confirmed else 'provisional'} "
                f"{self.votes} votes, share {self.share:.2f}>")


class TrackIdentity:
    """Everything the face stage knows about one tracking id."""

    def __init__(self, track_id):
        self.track_id = track_id
        self.created = time.time()
        self.last_seen = self.created

        # ---- best frame ---------------------------------------------
        self.best_quality = 0.0
        self.best_crop = None            # aligned 112x112 BGR
        self.best_box = None             # face box, frame coordinates
        self.best_landmarks = None
        self.best_det_score = 0.0
        self.best_at = 0.0
        self.best_embedding = None       # from the last recognition of it

        # ---- scheduling ---------------------------------------------
        self.last_recognised_frame = -1
        self.recognitions = 0
        self.faces_seen = 0
        self.faces_rejected = 0
        # Set when a look disagrees with the confirmed name. While it is
        # set, this track goes back to the fast recognition interval.
        self.contested = False

        # ---- voting --------------------------------------------------
        self.votes = deque(maxlen=max(1, VOTING_WINDOW))
        self.confirmed_name = ""
        self.confirmed_code = ""
        self.confirmed_score = 0.0
        self.confirmed_at = 0.0
        self.last_verdict = Verdict()

    # -------------------------------------------------- best frame
    def best_is_stale(self, now=None):
        """Has the held best frame stopped describing this track?

        See BEST_FRAME_MAX_AGE_SECONDS. A best frame that is never
        allowed to expire turns into a cached picture of whoever this
        track used to be following.
        """
        if self.best_crop is None:
            return True
        return (now or time.time()) - self.best_at > BEST_FRAME_MAX_AGE_SECONDS

    def offer(self, quality, crop, box, landmarks, det_score, now=None):
        """Offer a face as this track's best. True if it took the place.

        The improvement margin is what stops a stationary person at a
        desk re-triggering recognition on every pass with a frame that
        is not meaningfully different from the one already held. It only
        protects a RECENT incumbent - an old one is replaced outright.
        """
        now = now or time.time()
        self.last_seen = now
        self.faces_seen += 1
        if quality < BEST_FRAME_QUALITY_THRESHOLD:
            return False
        if (self.best_crop is not None and not self.best_is_stale(now)
                and quality < self.best_quality + BEST_FRAME_IMPROVEMENT):
            return False
        self.best_quality = float(quality)
        self.best_crop = crop
        self.best_box = box
        self.best_landmarks = landmarks
        self.best_det_score = float(det_score)
        self.best_at = now
        return True

    def touch(self, now=None):
        self.last_seen = now or time.time()

    # --------------------------------------------------- scheduling
    def should_recognise(self, frame_index, quality, is_new_best=False):
        """Is this face worth spending a recognition pass on?

        Returns (yes, why) - the reason is carried through to the debug
        line, because "why is the GPU busy" and "why has nobody been
        recognised for a minute" are the same question asked twice.
        """
        if quality < FACE_QUALITY_THRESHOLD:
            return False, "below quality bar"

        if self.last_recognised_frame < 0:
            return True, "first look"

        if is_new_best and BEST_FRAME_FORCES_RECOGNITION and \
                quality >= BEST_FRAME_QUALITY_THRESHOLD:
            return True, "better frame"

        # A CONTESTED track drops back to the fast interval. Once an
        # identity is confirmed we deliberately stop looking often -
        # that is most of the GPU saving - but the one thing worth
        # spending it on is a look that DISAGREED with the name already
        # on the person. Without this the disagreement would be
        # re-examined at the confirmed interval, so a genuine tracking
        # error (two people swapping boxes) would carry the wrong name
        # for half a minute before the vote could overturn it.
        interval = RECOGNITION_INTERVAL
        if self.confirmed_name and not self.contested:
            interval = RECOGNITION_INTERVAL_CONFIRMED
        if frame_index - self.last_recognised_frame >= interval:
            return True, "contested" if self.contested else "interval"
        return False, "too soon"

    def mark_recognised(self, frame_index, embedding=None):
        self.last_recognised_frame = frame_index
        self.recognitions += 1
        if embedding is not None:
            self.best_embedding = embedding

    # ------------------------------------------------------- voting
    def record(self, name, score, quality, code="", now=None,
               unknown_below=None):
        """Add one recognition result to the window and re-decide.

        `unknown_below` is passed in by the face stage rather than read
        from settings, because it has to track the recognition
        threshold of the model that actually loaded - which is not
        always the one configured.
        """
        now = now or time.time()
        self.last_seen = now
        floor = VOTE_UNKNOWN_BELOW if unknown_below is None else unknown_below
        if not name or score < floor:
            name = UNKNOWN
        self.votes.append(Vote(name, score, quality, now))
        if self.confirmed_name:
            # Only a positive identification of somebody ELSE counts as a
            # contest. An Unknown means we could not tell, which is not
            # evidence against the name - treating it as one would have
            # every occluded frame drag a confirmed track back to full
            # recognition cost.
            self.contested = (name != UNKNOWN and name != self.confirmed_name)
        if code and name != UNKNOWN:
            # remember the employee code that came with a real name, so a
            # later Unknown observation cannot blank it
            self._codes = getattr(self, "_codes", {})
            self._codes[name] = code
        return self.decide(now)

    def _tally(self, now):
        """{name: [weight, count, best_score]} over the live window."""
        totals = {}
        for vote in self.votes:
            if now - vote.when > VOTE_MAX_AGE_SECONDS:
                continue
            entry = totals.setdefault(vote.name, [0.0, 0, 0.0])
            entry[0] += vote.weight
            entry[1] += 1
            entry[2] = max(entry[2], vote.score)
        return totals

    def decide(self, now=None):
        """The current verdict. Confirms, defends, and only rarely moves."""
        now = now or time.time()
        totals = self._tally(now)
        if not totals:
            self.last_verdict = Verdict(reason="no recent observations")
            return self.last_verdict

        total_weight = sum(entry[0] for entry in totals.values()) or 1e-9
        named = [(name, entry) for name, entry in totals.items()
                 if name != UNKNOWN]
        named.sort(key=lambda item: -item[1][0])

        if not named:
            verdict = Verdict(reason="every look said nobody")
            self.last_verdict = self._defend(verdict, totals, total_weight, now)
            return self.last_verdict

        leader_name, leader = named[0]
        runner_name, runner_weight = ("", 0.0)
        if len(named) > 1:
            runner_name, runner = named[1]
            runner_weight = runner[0]
        # "Unknown" is a rival too - if most looks matched nobody, the
        # one look that did should not be allowed to name the person.
        unknown_weight = totals.get(UNKNOWN, [0.0, 0, 0.0])[0]
        rival_weight = max(runner_weight, unknown_weight)
        if unknown_weight > runner_weight:
            runner_name = UNKNOWN

        share = leader[0] / total_weight
        margin = (leader[0] - rival_weight) / total_weight
        codes = getattr(self, "_codes", {})

        enough = (leader[1] >= MIN_VOTES and share >= VOTE_MIN_SHARE
                  and margin >= VOTE_MIN_MARGIN)

        if not enough:
            reason = (f"{leader_name} leading with {leader[1]} vote(s), "
                      f"share {share:.2f}/{VOTE_MIN_SHARE}, "
                      f"margin {margin:.2f}/{VOTE_MIN_MARGIN}")
            verdict = Verdict(leader_name, codes.get(leader_name, ""),
                              leader[2], False, leader[1], share, margin,
                              runner_name, reason)
            self.last_verdict = self._defend(verdict, totals, total_weight, now)
            return self.last_verdict

        verdict = Verdict(leader_name, codes.get(leader_name, ""), leader[2],
                          True, leader[1], share, margin, runner_name,
                          "confirmed by vote")
        self.last_verdict = self._defend(verdict, totals, total_weight, now)
        return self.last_verdict

    def _defend(self, verdict, totals, total_weight, now):
        """Protect an identity that is already confirmed.

        This is the rule that stops identity switching. Everything above
        decides what the RECENT WINDOW says; this decides whether that is
        enough to overturn a decision already made and already written
        into somebody's attendance record.
        """
        if not self.confirmed_name:
            if verdict.confirmed and verdict.known:
                self.confirmed_name = verdict.name
                self.confirmed_code = verdict.code
                self.confirmed_score = verdict.score
                self.confirmed_at = now
            return verdict

        held = totals.get(self.confirmed_name, [0.0, 0, 0.0])

        if verdict.known and verdict.name == self.confirmed_name:
            # same answer - refresh the score, keep the confirmation, and
            # stand the track back down to the slow interval
            self.confirmed_score = max(self.confirmed_score, verdict.score)
            self.confirmed_code = verdict.code or self.confirmed_code
            self.contested = False
            verdict.confirmed = True
            return verdict

        challenger = totals.get(verdict.name, [0.0, 0, 0.0]) \
            if verdict.known else [0.0, 0, 0.0]
        needed = held[0] * VOTE_OVERRIDE_FACTOR

        if verdict.known and verdict.confirmed and \
                challenger[1] >= VOTE_OVERRIDE_MIN_VOTES and \
                challenger[0] >= needed:
            # A sustained, better-supported disagreement. Almost always a
            # genuine tracking error - two people swapped boxes - and
            # leaving the wrong name on somebody is worse than moving it.
            self.confirmed_name = verdict.name
            self.confirmed_code = verdict.code
            self.confirmed_score = verdict.score
            self.confirmed_at = now
            self.contested = False
            verdict.reason = (f"identity changed to {verdict.name} - "
                              f"{challenger[1]} observations at "
                              f"{challenger[0]:.2f} weight vs "
                              f"{held[0]:.2f} for the previous name")
            return verdict

        # Not enough to overturn: keep the confirmed identity. This is
        # the common case, and it is the whole point of the layer.
        return Verdict(self.confirmed_name, self.confirmed_code,
                       self.confirmed_score, True,
                       held[1], verdict.share, verdict.margin,
                       verdict.name if verdict.known else "",
                       f"holding {self.confirmed_name} "
                       f"(challenge from {verdict.name or 'nobody'} at "
                       f"{challenger[0]:.2f}, needs {needed:.2f})")

    # ------------------------------------------------------ external
    def force(self, name, code="", score=1.0, now=None):
        """Pin an identity from outside the vote - a human confirmation.

        A person's answer outranks anything the models can produce, so it
        is recorded as confirmed immediately and the window is cleared:
        leaving the old votes in place would let the model argue with the
        human for the next few seconds.
        """
        now = now or time.time()
        self.votes.clear()
        self.confirmed_name = name
        self.confirmed_code = code
        self.confirmed_score = float(score)
        self.confirmed_at = now
        self.contested = False
        self._codes = getattr(self, "_codes", {})
        if name:
            self._codes[name] = code
        self.last_verdict = Verdict(name, code, score, True, MIN_VOTES,
                                    1.0, 1.0, "", "set by a human")
        return self.last_verdict

    def summary(self):
        verdict = self.last_verdict
        return {"track": self.track_id, "name": verdict.name,
                "confirmed": verdict.confirmed, "score": round(verdict.score, 3),
                "votes": verdict.votes, "share": round(verdict.share, 3),
                "best_quality": round(self.best_quality, 3),
                "recognitions": self.recognitions,
                "faces_seen": self.faces_seen,
                "reason": verdict.reason}


class IdentityBook:
    """Every track's identity state for one camera.

    Lives on the face worker's side of the thread boundary but is read
    from the camera thread, so every entry point takes the lock. The
    contents are small - a few dozen tracks at most - so this is never
    a contention point.
    """

    def __init__(self, forget_after=120.0):
        self._lock = threading.RLock()
        self._tracks = {}
        self.forget_after = float(forget_after)

    def get(self, track_id, create=True):
        with self._lock:
            state = self._tracks.get(track_id)
            if state is None and create:
                state = TrackIdentity(track_id)
                self._tracks[track_id] = state
            return state

    def verdict(self, track_id):
        """The current answer for a track, or None if it has none."""
        with self._lock:
            state = self._tracks.get(track_id)
            return state.last_verdict if state is not None else None

    def force(self, track_id, name, code="", score=1.0):
        with self._lock:
            return self.get(track_id).force(name, code, score)

    def drop(self, track_id):
        with self._lock:
            self._tracks.pop(track_id, None)

    def prune(self, alive_ids=None, now=None):
        """Forget tracks that are gone.

        Bounded on both sides: an id the tracker no longer reports is
        dropped once it has been quiet for forget_after seconds. The
        delay matters - a track that briefly coasts must come back to
        its own votes, not to an empty window, or every occlusion would
        re-open a settled identity.
        """
        now = now or time.time()
        with self._lock:
            for track_id in list(self._tracks):
                state = self._tracks[track_id]
                if now - state.last_seen > self.forget_after:
                    del self._tracks[track_id]
                elif alive_ids is not None and track_id not in alive_ids \
                        and now - state.last_seen > self.forget_after * 0.5:
                    del self._tracks[track_id]

    def summary(self):
        with self._lock:
            return [state.summary() for state in self._tracks.values()]

    def confirmed_names(self):
        with self._lock:
            return {state.confirmed_name for state in self._tracks.values()
                    if state.confirmed_name}
