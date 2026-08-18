"""
cameras/workspace.py
====================
CAMERA 2 - Server Rm Psg (and any other work-area camera).

Purpose
-------
Not entries and exits. This camera answers a different question:
**how long was each person in this area?**

What makes this scene hard
--------------------------
People are seated at desks and heavily occluded - monitors, chair backs,
partitions. For most of the time only a head and shoulders are visible,
and a given person may be undetectable for seconds at a stretch. A naive
approach either loses them constantly (undercounting their hours) or
re-creates them as somebody new (splitting one person into many).

Three things are done about that:

1. POLYGON AREA. Only the region you marked is considered. Everything
   outside is blacked out before detection, so the far desks, the
   corridor and the glass frontage cannot generate detections at all.
   The frame is then cropped to the polygon's bounding box, which also
   makes the people inside proportionally larger to the detector.

2. RE-IDENTIFICATION BY SIZE. Each person carries their measured
   stature and build (core/body.py). When somebody vanishes behind a
   monitor and reappears, they are matched on how big they are as well
   as where they were. This used to be a colour signature of their
   clothes, which is what let presence time - and names - end up
   charged to the wrong person whenever two people overlapped.

3. GAP-TOLERANT PRESENCE. A gap in detection is not treated as leaving.
   Someone is considered still present until they have been absent for
   PRESENCE_GAP_TOLERANCE. Crucially, the accumulated seconds count only
   time they were actually seen, so a tolerated gap never inflates
   anybody's hours.

Identity is by face, applied retroactively: the moment somebody's face is
recognised, the time they have already accumulated is relabelled to them.
"""
import time
from datetime import datetime

import cv2

from cameras.base import BaseCamera, GREEN, AMBER, GREY, WHITE
from core.enhance import enhance
from core.geometry import (to_pixels, point_inside, build_mask, bounding_box,
                           apply_mask, draw_polygon, polygon_area_fraction)
from core.seats import SeatMap
from database import repository as repo
from config.settings import (VERBOSE, PRESENCE_GAP_TOLERANCE,
                             PRESENCE_MIN_SECONDS, PRESENCE_SAVE_EVERY,
                             PRESENCE_MIN_HITS, SEAT_DWELL_SECONDS,
                             SEAT_IDENTIFY_SCORE)


class WorkspaceCamera(BaseCamera):
    handler_name = "workspace"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.polygon = self.options.get("area")          # list of (x, y)
        self.mask_outside = self.options.get("mask_outside", True)
        self.crop_to_area = self.options.get("crop_to_area", True)

        self._poly_px = None
        self._mask = None
        self._bbox = None
        self._reported_area = False

        # identify people by which desk they are sitting at - the only
        # method that works when faces are not visible (see core/seats.py)
        self.seatmap = SeatMap(self.options.get("seats"),
                               dwell_seconds=SEAT_DWELL_SECONDS)
        if self.seatmap:
            print(f"[{self.name}] {len(self.seatmap.seats)} seat(s) mapped "
                  f"to people")

    # -------------------------------------------------------- geometry
    def _prepare_area(self, w, h):
        if self.polygon is None or self._poly_px is not None:
            return
        self._poly_px = to_pixels(self.polygon, w, h)
        self._mask = build_mask(self._poly_px, w, h)
        self._bbox = bounding_box(self._poly_px)
        if not self._reported_area:
            frac = polygon_area_fraction(self._poly_px, w, h)
            print(f"[{self.name}] monitored area covers {frac:.0%} of the "
                  f"frame; the rest is ignored")
            self._reported_area = True

    def frame_for_detection(self):
        """The image the detector sees.

        The marked area is used for masking ONLY if you ask for it. By
        default the whole frame is detected, so somebody who steps out of
        the work zone and back is still the same tracked person. Whatever
        the masking choice, the camera's image enhancement is always
        applied - previously this path returned the raw frame and quietly
        skipped it, which cost this camera its low-light correction.
        """
        if self.frame is None:
            return None
        h, w = self.frame.shape[:2]
        self._prepare_area(w, h)

        if self._mask is not None and self.mask_outside:
            bbox = self._bbox if self.crop_to_area else None
            out = apply_mask(self.frame, self._mask, bbox)
        elif self._mask is not None and self.crop_to_area:
            x1, y1, x2, y2 = self._bbox
            out = self.frame[y1:y2, x1:x2]
        else:
            out = self.frame

        if self.enhance_profile:
            out = enhance(out, self.enhance_profile)
        return out

    def map_detections(self, detections):
        """Shift boxes from the cropped area back to full-frame space."""
        if self._bbox is None or not self.crop_to_area:
            return detections
        ox, oy = self._bbox[0], self._bbox[1]
        if ox == 0 and oy == 0:
            return detections
        return [{**d, "box": (d["box"][0] + ox, d["box"][1] + oy,
                              d["box"][2] + ox, d["box"][3] + oy)}
                for d in detections]

    def _inside_area(self, box):
        """Judge by the person's lower-centre - roughly where they stand
        or where their chair is - which is far more stable than the box
        centre when they are half hidden behind a desk."""
        if self._poly_px is None:
            return True
        x = (box[0] + box[2]) // 2
        y = int(box[1] + 0.85 * (box[3] - box[1]))
        return point_inside(self._poly_px, x, y)

    # --------------------------------------------------------- process
    def process(self, detections):
        frame = self.frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        self._prepare_area(w, h)

        # Track EVERYONE the camera can see. The marked area decides who
        # counts as being in the work area for reporting, but restricting
        # tracking to it meant a person who stepped outside for a moment
        # came back as a stranger.
        boxes = [d["box"] for d in detections]
        scores = [d.get("conf", 1.0) for d in detections]
        # The confidences matter here more than anywhere else on the
        # site: a seated person behind a monitor scores low, and it is
        # ByteTrack's second association pass over exactly that low band
        # that keeps their track - and therefore their identity and
        # their accumulated presence time - alive through the occlusion.
        live, finished = self.tracker.update(boxes, frame=frame, scores=scores)

        self.submit_faces(live)
        faces = self.face.get() if self.face else []
        self._assign_faces(live, faces)
        if self.seatmap:
            self._assign_seats(live, w, h)
        # An independent look at what we just decided. Asked in the
        # background, answered on some later frame, and only ever able
        # to take a name off somebody - see cameras/base.py.
        self.vlm_review(live, faces, frame)

        now = time.time()
        for track in live:
            self._update_presence(track, now)
        for track in finished:
            self._close_presence(track)
            self.release_identity(track)
            if self.seatmap:
                self.seatmap.forget(track.id)

        self._draw(frame, live, w, h)
        self.annotated = frame

    # -------------------------------------------------------- identity
    def _assign_faces(self, tracks, faces):
        """Work out who each tracked person is.

        A FACE, or the desk they are sitting at (core/seats.py), or
        nothing. On this camera a face is rarely readable, so a lot of
        people stay Unknown - and that is the intended outcome of
        removing clothing matching, which used to fill those gaps with
        confident guesses that moved from person to person.

        What still gets people named here: a face caught in the moment
        they turn toward the camera, a name the tracker carried in from
        the reception door, and the seat map.
        """
        frame = self.frame
        if frame is None:
            return

        for track in tracks:
            if track.hits < PRESENCE_MIN_HITS:
                continue

            # The face vote first, for the same reason as at reception:
            # several agreeing looks at a face is the strongest evidence
            # this system produces, and nothing else may take that name
            # off them.
            applied, verdict = self.apply_voted_identity(
                track, self._revert_name)
            if applied and track.presence_id is not None:
                repo.identify_presence(track.presence_id, track.name,
                                       track.emp_code, track.score)

            (match, face_emb, body, face_box) = self.resolve_identity(
                track, faces, frame)

            # Measure anybody we are sure about, every frame, face or no
            # face - see the same step in cameras/reception.py. On this
            # camera it matters more: faces are rare here, so almost all
            # of a seated person's measurements come from frames where
            # there was no face to be had.
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
                if (track.would_accept_identity(match.name, match.probability)
                        and self.claim_identity(track, match.name, match.code,
                                                match.probability,
                                                self._revert_name)):
                    if track.presence_id is not None:
                        repo.identify_presence(track.presence_id, track.name,
                                               track.emp_code, track.score)
                    if VERBOSE:
                        cues = ", ".join(
                            f"{k} {v:.2f}" for k, v in match.evidence.items()
                            if isinstance(v, float) and v > 0)
                        mins = track.present_seconds / 60.0
                        print(f"[{self.name}] track #{track.id} = "
                              f"{match.name} (p={match.probability:.2f}; "
                              f"{cues}) - {mins:.1f} min credited")
                self.learn_from(track, face_emb, body)

            elif not track.identified:
                self.explain_occasionally(track, match)

            if track.identified:
                self.capture_reference(track, faces, frame)

            if match.decision == "review" and not track.identified:
                # plausible but not certain - ask a human rather than guess
                self.queue_for_review(track, match, frame, face_box,
                                      face_emb, faces)
            elif not track.identified:
                self.ask_who_this_is(track, faces, frame)

    def scene_people(self, scale=1.0):
        """As BaseCamera, but say who is actually in the work area.

        People are tracked across the whole frame here and only counted
        inside the marked polygon, and the assistant has to be able to
        make the same distinction: somebody standing in the corridor
        behind the glass is visible, is not at a desk, and should not be
        reported as being in the room.
        """
        people = super().scene_people(scale)
        inside = {t.id: self._inside_area(t.box)
                  for t in self.tracker.all_tracks()}
        for entry in people:
            entry["counted"] = bool(inside.get(entry["track_id"], True))
        return people

    def _revert_name(self, track):
        """A track just lost its name to a better claim - correct the
        database row too, so the record does not keep the wrong name."""
        if track.presence_id is not None:
            repo.update_presence(track.presence_id, track.present_seconds,
                                 active=True)

    def on_identity_refused(self, track, name):
        """The vision model took a name off this body.

        The presence row has to be corrected too, or the minutes already
        credited stay filed under somebody who was never here - which is
        the part of a wrong name that actually matters. It goes back to
        being an unattributed session rather than being deleted: the
        person WAS present, we simply no longer claim to know who.
        """
        if track.presence_id is None:
            return
        # Only the NAME is taken back. The active flag is deliberately
        # untouched: this can run as a track is being closed, and forcing
        # active=True there would reopen a session that has just ended.
        repo.unidentify_presence(track.presence_id, was_name=name)
        if VERBOSE:
            mins = track.present_seconds / 60.0
            print(f"[{self.name}] {mins:.1f} min un-credited from {name} "
                  f"- the record now says Unknown rather than the wrong "
                  f"person")

    # ------------------------------------------------------- by desk
    def _assign_seats(self, tracks, w, h):
        """Credit a person with their desk once they have settled into it.

        Face recognition wins if it ever produces a confident match; a
        seat is used when it does not, which on this camera is nearly
        always.
        """
        for t in tracks:
            if t.identified and t.score >= SEAT_IDENTIFY_SCORE:
                continue                     # a real face match beats a seat
            seat = self.seatmap.resolve(t, w, h)
            if seat is None:
                continue
            if t.identified and t.name == seat.employee:
                continue
            if not t.would_accept_identity(seat.employee,
                                           SEAT_IDENTIFY_SCORE):
                continue
            if not self.claim_identity(t, seat.employee, seat.code,
                                       SEAT_IDENTIFY_SCORE,
                                       self._revert_name):
                continue          # somebody else has a better claim
            if t.presence_id is not None:
                repo.identify_presence(t.presence_id, t.name, t.emp_code,
                                       t.score)
            if VERBOSE:
                mins = t.present_seconds / 60.0
                print(f"[{self.name}] track #{t.id} identified as "
                      f"{seat.employee} by desk position - "
                      f"{mins:.1f} min credited to them")

    # -------------------------------------------------------- presence
    def _update_presence(self, track, now):
        if track.hits < PRESENCE_MIN_HITS:
            return
        # tracked everywhere, but only counted while inside the area
        if not self._inside_area(track.box):
            return
        # ...and not counted at all if the vision model says this is not
        # a person. A reflection in the glass frontage that survives the
        # tracker accumulates presence minutes exactly like a colleague
        # does, and nothing downstream can tell the difference.
        if not self.is_real_person(track):
            return

        if track.presence_id is None:
            # if this person was here moments ago, continue that session
            existing = None
            if track.identified:
                existing = repo.find_open_presence(
                    self.key, track.name, PRESENCE_GAP_TOLERANCE)
            if existing:
                track.presence_id = existing
                if VERBOSE:
                    print(f"[{self.name}] {track.name} resumed an existing "
                          f"presence session")
            else:
                track.presence_id = repo.open_presence(
                    self.key, self.name, track.id,
                    when=datetime.utcfromtimestamp(track.created)
                    if track.created < 1e12 else None)
                if VERBOSE:
                    print(f"[{self.name}] PRESENT  track #{track.id}  "
                          f"{track.label}")
                if track.identified:
                    repo.identify_presence(track.presence_id, track.name,
                                           track.emp_code, track.score)

        # write the running total periodically, not every frame
        if track.present_seconds - track.presence_saved >= PRESENCE_SAVE_EVERY:
            repo.update_presence(track.presence_id, track.present_seconds,
                                 active=True)
            track.presence_saved = track.present_seconds

    def _close_presence(self, track):
        if track.presence_id is None:
            return
        if track.present_seconds < PRESENCE_MIN_SECONDS:
            # too brief to be a real stay - close it out quietly
            repo.update_presence(track.presence_id, track.present_seconds,
                                 active=False)
            return
        repo.update_presence(track.presence_id, track.present_seconds,
                             active=False)
        if VERBOSE:
            mins = track.present_seconds / 60.0
            print(f"[{self.name}] LEFT     track #{track.id}  {track.label}  "
                  f"total {mins:.1f} min")

    # ------------------------------------------------------------ draw
    def _draw(self, frame, tracks, w, h):
        if self._poly_px is not None:
            draw_polygon(frame, self._poly_px, (60, 90, 220), 2,
                         "monitored area")
        if self.seatmap:
            for seat, px in self.seatmap.all_pixels(w, h):
                cv2.polylines(frame, [px], True, (90, 110, 130), 1)
                x, y = px[0]
                cv2.putText(frame, seat.label, (int(x) + 4, int(y) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 110, 130), 1)

        for t in tracks:
            inside = self._inside_area(t.box)
            real = self.is_real_person(t)
            if not real:
                # Drawn, and visibly not counted. Hiding it would make
                # the reason a person is missing from the tally
                # invisible, which is how a wrong veto goes unnoticed.
                self.label_box(frame, t.box, "not a person (vision model)",
                               GREY)
                continue
            if t.identified:
                colour, who = GREEN, t.name
            elif t.hits >= PRESENCE_MIN_HITS:
                colour, who = AMBER, f"Unknown #{t.id}"
            else:
                colour, who = GREY, "..."
            if not inside:
                colour = GREY          # tracked, but outside the work area
            mins = t.present_seconds / 60.0
            label = (f"{who}  {mins:.0f}m" if mins >= 1
                     else f"{who}  {t.present_seconds:.0f}s")
            self.label_box(frame, t.box, label, colour)

        counted = sum(1 for t in tracks
                      if t.hits >= PRESENCE_MIN_HITS and self.is_real_person(t))
        named = sum(1 for t in tracks if t.identified)
        self.draw_header(frame, f"FPS {self.fps:.1f}")
        cv2.rectangle(frame, (0, h - 34), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"IN AREA {counted}    IDENTIFIED {named}",
                    (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)


HANDLER = WorkspaceCamera
