"""
cameras/reception.py
====================
CAMERA 1 - Reception Lobby.

Purpose
-------
Count everyone who comes INTO the reception area and identify them.

Why entry only: a person leaving walks away from the camera, so we only
ever see the back of their head. Their face cannot be read, so an "exit"
could never be attributed to a named person. Rather than record
half-truths, this camera records arrivals only.

The rule for an entry
---------------------
COMING TOWARD THE CAMERA IS AN ENTRY. NOTHING ELSE IS.

A person is counted ONCE, and only if we have watched them walk toward
the camera - growing in the frame and moving down it, which is what
somebody coming in through the door does and what nobody else does. A
person walking away from the camera is leaving. A person crossing the
lobby without approaching is passing through. Neither is an arrival, and
neither is counted.

That single entry belongs to them for their whole stay - walking around
afterwards, standing at the desk or sitting down never produces a second
count.

The decision is deliberately DEFERRED for ENTRY_MIN_OBSERVATIONS frames.
You cannot tell an arrival from a departure at the first sighting, and
waiting costs nothing: the arrival time recorded is when they were FIRST
seen, not when we made up our minds.

Late identification (the important part)
----------------------------------------
People enter at the far end of the lobby, behind the glass door, where a
face is only a few pixels wide. At that moment we honestly do not know
who they are, so the visit is written as "Unknown" - but it IS written,
with the correct arrival time.

As the person walks toward the camera their face grows. The moment it is
big enough to trust, the recogniser returns a name, we attach it to the
track, and we UPDATE the visit row that was already created. The arrival
time stays exactly as it was; only the name changes. That is why the log
ends up saying "Rajesh arrived at 10:45" even though at 10:45 the system
could not yet see who it was.
"""
import time
from datetime import datetime

import cv2

from cameras.base import BaseCamera, GREEN, AMBER, GREY, WHITE
from core.tracker import APPROACHING, RECEDING, STEADY, UNCLEAR
from database import repository as repo
from config.settings import (VERBOSE, ENTRY_MIN_OBSERVATIONS,
                             ENTRY_NEAR_HEIGHT_FRAC, ENTRY_BLOCK_RECEDING,
                             ENTRY_REQUIRE_APPROACH, ENTRY_DEDUP_SECONDS)


class ReceptionCamera(BaseCamera):
    handler_name = "reception"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.zone = self.options.get("zone")            # None = whole frame
        self.near_fraction = self.options.get(
            "near_height_fraction", ENTRY_NEAR_HEIGHT_FRAC)
        self.block_receding = self.options.get(
            "block_receding", ENTRY_BLOCK_RECEDING)
        self.require_approach = self.options.get(
            "require_approach", ENTRY_REQUIRE_APPROACH)
        self.not_approaching = 0        # seen, but not coming in
        self._passed_by = set()         # track ids already counted as that
        self.min_observations = self.options.get(
            "min_observations", ENTRY_MIN_OBSERVATIONS)
        self.entries_today = 0
        self._zone_px = None

    # ------------------------------------------------------------ zone
    def _zone_pixels(self, w, h):
        """Convert the fractional zone to pixels once we know the size."""
        if self.zone is None:
            return None
        if self._zone_px is None:
            x1, y1, x2, y2 = self.zone
            self._zone_px = (
                int(x1 * w if x1 <= 1 else x1),
                int(y1 * h if y1 <= 1 else y1),
                int(x2 * w if x2 <= 1 else x2),
                int(y2 * h if y2 <= 1 else y2),
            )
        return self._zone_px

    def _in_zone(self, box, w, h):
        z = self._zone_pixels(w, h)
        if z is None:
            return True
        zx1, zy1, zx2, zy2 = z
        # use the person's feet - more reliable than the centre when
        # someone is only half visible at the edge of frame
        fx = (box[0] + box[2]) // 2
        fy = box[3]
        return zx1 <= fx <= zx2 and zy1 <= fy <= zy2

    # --------------------------------------------------------- process
    def process(self, detections):
        frame = self.frame
        if frame is None:
            return
        h, w = frame.shape[:2]

        kept = [d for d in detections if self._in_zone(d["box"], w, h)]
        boxes = [d["box"] for d in kept]
        scores = [d.get("conf", 1.0) for d in kept]
        # The frame is passed so each detection can be measured -
        # height, build, where the feet are. ByteTrack uses that only to
        # break ties between geometrically plausible pairings, which is
        # what keeps two people crossing the lobby from swapping tracks
        # - and swapping identities with them.
        live, finished = self.tracker.update(boxes, frame=frame, scores=scores)

        # Faces are submitted AFTER tracking, so every face can be
        # attributed to the tracking id it belongs to.
        self.submit_faces(live)
        faces = self.face.get() if self.face else []
        self._assign_faces(live, faces)
        # An independent look at what we just decided. Asked in the
        # background, answered on some later frame, and only ever able
        # to take a name off somebody - see cameras/base.py.
        self.vlm_review(live, faces, frame)

        for track in live:
            self._maybe_log_entry(track, h)

        for track in finished:
            self._close_track(track)

        self._draw(frame, live, faces, w, h)
        self.annotated = frame

    # ------------------------------------------------- identity linking
    def _assign_faces(self, tracks, faces):
        """Work out who each person is.

        Reception is where faces are most likely to be readable, so this
        camera does most of the teaching: it files camera photographs of
        people it is sure about, and it measures them, so a later face
        match that lands on somebody of the wrong size can be refused.
        """
        frame = self.frame
        if frame is None:
            return

        for track in tracks:
            if not track.confirmed(self.tracker.min_hits):
                continue

            # 1. THE FACE VOTE FIRST.
            # A confirmed identity from the temporal vote is the
            # strongest evidence any model here can produce: several
            # good frames of this person's face agreeing with each
            # other. It is applied before the gallery gets a chance to
            # propose anybody else.
            applied, verdict = self.apply_voted_identity(track)
            if applied and track.visit_id is not None:
                repo.identify_visit(track.visit_id, track.name,
                                    track.emp_code, track.score)

            (match, face_emb, body, face_box) = self.resolve_identity(
                track, faces, frame)

            # MEASURE ANYBODY WE ARE SURE ABOUT, every frame, face or no
            # face. This is not identification - it is building the
            # record of how big this person is, which is what a later
            # face match gets checked against. Doing it only when a face
            # happened to be readable would leave most people with no
            # measurements at all, and the check would never fire.
            if track.identified:
                self.learn_from(track, face_emb, body)

            if match is None:
                # No face this frame, or nobody in the gallery scored at
                # all. The second case is the one worth asking about: it
                # is exactly the person whose reference photographs are
                # missing or unusable.
                if not track.identified:
                    self.ask_who_this_is(track, faces, frame)
                continue

            if match.decision == "accept":
                was_unknown = not track.identified
                if (track.would_accept_identity(match.name, match.probability)
                        and self.claim_identity(track, match.name, match.code,
                                                match.probability)):
                    if track.visit_id is not None:
                        repo.identify_visit(track.visit_id, track.name,
                                            track.emp_code, track.score)
                    if VERBOSE and was_unknown:
                        cues = ", ".join(
                            f"{k} {v:.2f}" for k, v in match.evidence.items()
                            if isinstance(v, float) and v > 0)
                        print(f"[{self.name}] track #{track.id} = "
                              f"{match.name} (p={match.probability:.2f}; "
                              f"{cues}) - visit updated")
                self.learn_from(track, face_emb, body)

            elif not track.identified:
                self.explain_occasionally(track, match)

            # Keep a camera photograph of anybody we are now sure about,
            # so tomorrow's recognition has a reference that looks like
            # what this camera actually sees.
            if track.identified:
                self.capture_reference(track, faces, frame)

            if match.decision == "review" and not track.identified:
                self.queue_for_review(track, match, frame, face_box,
                                      face_emb, faces)
            elif not track.identified:
                # Nothing matched at all - ask outright rather than stay
                # silent, or the people with no usable reference photo
                # never get one.
                self.ask_who_this_is(track, faces, frame)

    # ----------------------------------------------------- entry rules
    def _entry_verdict(self, track, frame_h):
        """Decide whether this track is a genuine arrival.

        Returns (should_log, reason). Four things have to be true, and
        the third is the one that matters most: they have to have walked
        TOWARD the camera.
        """
        if track.entry_logged:
            return False, "already logged"
        if not track.confirmed(self.tracker.min_hits):
            return False, "not confirmed"
        # A reflection in the lobby glass that survives the tracker is
        # recorded as somebody arriving, and nothing downstream can tell
        # it from a colleague walking in.
        if not self.is_real_person(track):
            return False, "not a person (vision model)"
        if track.hits < self.min_observations:
            return False, "gathering evidence"

        motion = track.motion()

        # someone walking AWAY from the camera is a departure, not an
        # arrival
        if self.block_receding and motion == RECEDING:
            return False, "walking away"

        # ...AND WALKING PAST IS NOT AN ARRIVAL EITHER.
        #
        # This is the rule that was missing. Blocking only the people
        # who were clearly receding still counted everybody who crossed
        # the lobby, walked along the far wall, or stood about near the
        # door - anyone whose size did not happen to shrink. On a camera
        # pointed down a corridor that is most of the traffic, and every
        # one of them was recorded as somebody arriving.
        #
        # An arrival now has to LOOK like an arrival: growing in the
        # frame and moving down it (core/tracker.motion). Somebody who
        # never does that is not counted, however long they are in view.
        if self.require_approach and motion != APPROACHING:
            return False, ("still deciding" if motion == UNCLEAR
                           else "not coming toward the camera")

        # they must actually come reasonably close at some point.
        # people crossing the far background never reach this size, so
        # they are never recorded as entering the reception.
        if track.peak_fraction(frame_h) < self.near_fraction:
            return False, "too far away"

        return True, motion

    def _maybe_log_entry(self, track, frame_h):
        ok, reason = self._entry_verdict(track, frame_h)
        if not ok:
            # Count the people we watched and decided against, once each.
            # Without it the only visible number is the entry count, and
            # "why did it only see three arrivals this morning" has no
            # answer on the screen.
            if (reason in ("walking away", "not coming toward the camera")
                    and track.id not in self._passed_by):
                self._passed_by.add(track.id)
                self.not_approaching += 1
                if VERBOSE:
                    print(f"[{self.name}] not an entry: track #{track.id} "
                          f"{track.label} - {reason}")
            return

        # last guard against duplicates: if this person is already known
        # and we opened a visit for them moments ago, reuse it instead of
        # creating a second arrival.
        if track.identified:
            existing = repo.find_recent_visit(
                self.key, track.name, ENTRY_DEDUP_SECONDS)
            if existing:
                track.visit_id = existing
                track.entry_logged = True
                if VERBOSE:
                    print(f"[{self.name}] duplicate suppressed for "
                          f"{track.name} (reusing visit {existing})")
                return

        track.visit_id = repo.open_visit(self.key, self.name, track.id,
                                         when=track.entered_at)
        track.entry_logged = True
        self.entries_today += 1

        if track.identified:
            repo.identify_visit(track.visit_id, track.name,
                                track.emp_code, track.score)
        if VERBOSE:
            print(f"[{self.name}] ENTRY  track #{track.id}  {track.label}  "
                  f"({reason}, {track.peak_fraction(frame_h):.0%} of frame)")

    def on_identity_refused(self, track, name):
        """The vision model took a name off this body.

        The visit row has to be corrected too, or an arrival stays filed
        under somebody who never came in - which is the part of a wrong
        name that actually costs somebody. The visit is kept: a person
        DID arrive, we simply no longer claim to know who.
        """
        if track.visit_id is None:
            return
        repo.unidentify_visit(track.visit_id, was_name=name)
        if VERBOSE:
            print(f"[{self.name}] visit {track.visit_id} no longer credited "
                  f"to {name} - the record now says Unknown rather than the "
                  f"wrong person")

    def _close_track(self, track):
        self.release_identity(track)
        if track.visit_id is None:
            return
        repo.close_visit(track.visit_id)
        if VERBOSE:
            secs = (datetime.utcnow() - track.entered_at).total_seconds()
            print(f"[{self.name}] LEFT   track #{track.id}  {track.label} "
                  f"after {secs:.0f}s")

    # ------------------------------------------------------------ draw
    def _draw(self, frame, tracks, faces, w, h):
        self.draw_detect_roi(frame)
        z = self._zone_pixels(w, h)
        if z:
            cv2.rectangle(frame, (z[0], z[1]), (z[2], z[3]), (90, 90, 90), 1)
            cv2.putText(frame, "count zone", (z[0] + 6, z[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)

        for t in tracks:
            if not self.is_real_person(t):
                # Drawn, and visibly not counted. Hiding it would make
                # the reason somebody is missing from the tally
                # invisible, which is how a wrong veto goes unnoticed.
                self.label_box(frame, t.box, "not a person (vision model)",
                               GREY)
                continue
            motion = t.motion()
            arrow = {APPROACHING: "v", RECEDING: "^"}.get(motion, "")
            if t.identified:
                colour, text = GREEN, t.name
            elif t.confirmed(self.tracker.min_hits):
                colour, text = AMBER, f"Unknown #{t.id}"
            else:
                colour, text = GREY, "..."      # not yet confirmed
            if t.entry_logged:
                text = f"{text} {arrow}".strip()
            else:
                # not counted - say why in one word, so somebody watching
                # the stream can see the rule being applied rather than
                # wondering why a person did not appear in the count
                why = {APPROACHING: "coming in", RECEDING: "away",
                       STEADY: "passing"}.get(motion, "watching")
                text = f"{text} [{why}]"
            self.label_box(frame, t.box, text, colour)

        # faint boxes on faces the system judged too small to trust
        for f in faces:
            if f.get("too_small"):
                x1, y1, x2, y2 = f["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

        confirmed = sum(1 for t in tracks
                        if t.confirmed(self.tracker.min_hits))
        self.draw_header(frame, f"FPS {self.fps:.1f}")
        cv2.rectangle(frame, (0, h - 34), (w, h), (20, 20, 20), -1)
        cv2.putText(frame,
                    f"ENTRIES {self.entries_today}    "
                    f"PASSED BY {self.not_approaching}    "
                    f"IN VIEW {confirmed}",
                    (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)


HANDLER = ReceptionCamera
