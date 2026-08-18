"""
core/tracker.py
===============
Follows each person across the whole camera view, carries their identity
(including identity discovered late), and works out whether they are
walking toward the camera or away from it.

Three jobs:

1. ONE IDENTITY PER PERSON, across the whole coverage area.
   Matching is by box overlap first, centre distance second. Overlap
   survives the big frame-to-frame jumps of a wide-angle camera far
   better than distance alone.

2. NO DUPLICATES. Two things cause a person to be counted twice: a track
   dying and restarting, and the detector blinking for a moment. So a
   track coasts for several seconds while unseen, AND dead tracks are
   remembered for a short while - if a new detection appears where a
   dead track just was, the old track is REVIVED (same id, same database
   visit) instead of a new person being invented.

3. DIRECTION OF TRAVEL. A person walking toward the camera grows in the
   frame; someone leaving shrinks. Comparing a person's size now against
   when they were first seen tells us which is happening, which is how
   we avoid recording someone's departure as an arrival.

WHICH ASSOCIATOR RUNS (TRACKER_BACKEND)
---------------------------------------
The three jobs above are the CONTRACT and are unchanged. What can be
swapped is the machinery that decides which detection belongs to which
track from one frame to the next:

  "bytetrack"  (default) core/bytetrack.py - Kalman prediction plus
               ByteTrack's two-pass association. The second pass offers
               the LOW-confidence detections to tracks the first pass
               could not match, which is what keeps a half-occluded
               person at a desk alive instead of killing their track
               and, with it, their identity and their presence session.
  "geometry"   the original IoU + centre-distance + body-size matcher
               below, kept so the two can be compared on real footage.

Everything outside this file sees the same Track objects either way -
same ids, same identity, same visit_id, same body measurements, same
ghost/re-identification layer.

WHAT A PERSON IS MATCHED ON
---------------------------
Position, size and build - see core/body.py. It used to be the colour
of their clothes, and that is exactly what put one person's name on
another person's body: while two people overlap, each one's crop
contains the other, so their colour signatures drift together, and when
they separate the name follows whichever body the blended descriptor
now fits better. Stature and build cannot drift that way. They are
measured from the box the tracker is already following, they do not
change when somebody turns round, and measurements taken while two
people overlap are DISCARDED rather than learned.
"""
import time
from collections import deque
from datetime import datetime

from core.boxes import iou, centre, height
from core.body import BodyMemory, Perspective, measure
from config.settings import (TRACK_IOU_THRESHOLD, TRACK_MAX_DISTANCE,
                             TRACK_MIN_HITS, TRACK_MAX_AGE,
                             REID_MEMORY_SECONDS, REID_MAX_DISTANCE_FACTOR,
                             REID_SIZE_RATIO, FACE_UPGRADE_MARGIN,
                             ENTRY_APPROACH_RATIO, ENTRY_RECEDE_RATIO,
                             BODY_ENABLED, BODY_REID_MIN, TRACKER_BACKEND,
                             BYTETRACK_USE_BODY, BYTETRACK_BODY_WEIGHT,
                             RECLAIM_COASTING, RECLAIM_MAX_GAP_SECONDS,
                             RECLAIM_DISTANCE_PER_SECOND,
                             BODY_RECLAIM_MIN, OCCLUSION_IOU,
                             VERBOSE_IDENTITY, REID_DISTANCE_PER_SECOND)

# How the three matching signals are weighted. Geometry dominates; body
# size is a tie-breaker worth about a third of the decision.
W_IOU = 0.50
W_DIST = 0.20
W_BODY = 0.30
# Any geometrically plausible pair stays in contention; the weights above
# only decide which pairing wins when several are possible.
MIN_PLAUSIBLE = 0.10
MATCH_ACCEPT = 0.10

APPROACHING = "approaching"
RECEDING = "receding"
STEADY = "steady"
UNCLEAR = "unclear"


class Track:
    """One person, for as long as they are in this camera's view."""

    def __init__(self, track_id, box, now):
        self.id = track_id
        self.box = box
        self.cx, self.cy = centre(box)
        self.created = now
        self.last_seen = now
        self.hits = 1

        # size history drives the approaching / walking-away decision,
        # and the feet history seconds it - see motion()
        self.sizes = deque(maxlen=60)
        self.sizes.append(height(box))
        self.feet = deque(maxlen=60)
        self.feet.append(box[3])
        self.peak_height = height(box)

        # identity - starts unknown, may arrive much later
        self.name = "Unknown"
        self.emp_code = ""
        self.score = 0.0
        self.identified = False

        # How tall and how broad this person is, so they can be picked
        # out again after an occlusion (see core/body.py)
        self.body = BodyMemory()
        # True while another person's box overlaps this one - see
        # observe(). Measuring is paused while it is set, because the box
        # around two merged people describes neither of them.
        self.occluded = False
        # When the overlap started and what we knew just before it. A
        # name is not allowed to move while this is set, and when the
        # group breaks up the people in it are re-matched against these
        # measurements rather than against whatever the merged boxes
        # said in the middle.
        self.occluded_since = 0.0

        # accumulated time actually present, used by area cameras
        self.present_seconds = 0.0
        self._presence_mark = now

        # link to the database row, so a late name can be written back
        self.visit_id = None
        self.entry_logged = False
        self.entered_at = datetime.utcnow()

        # area/presence camera bookkeeping
        self.presence_id = None
        self.presence_saved = 0.0

    # ------------------------------------------------------- geometry
    @property
    def height(self):
        return height(self.box)

    def height_fraction(self, frame_height):
        return self.height / max(1, frame_height)

    def peak_fraction(self, frame_height):
        return self.peak_height / max(1, frame_height)

    def observe(self, box, now, body=None, occluded=False):
        # accumulate time present, ignoring long gaps where we lost them
        gap = now - self._presence_mark
        if 0 < gap < 2.0:
            self.present_seconds += gap
        self._presence_mark = now

        self.box = box
        self.cx, self.cy = centre(box)
        self.last_seen = now
        self.hits += 1
        occluded = bool(occluded)
        if occluded and not self.occluded:
            self.occluded_since = now
        elif not occluded:
            self.occluded_since = 0.0
        self.occluded = occluded
        h = height(box)
        self.sizes.append(h)
        self.feet.append(box[3])
        self.peak_height = max(self.peak_height, h)
        # While overlapping somebody else, this box contains part of
        # THEM: it is taller than this person and much wider. Measuring
        # it would drag this track's stature and build toward the merged
        # pair, which is the mechanism behind an identity ending up on
        # the wrong body after the two separate. So the measurement is
        # thrown away and the settled one from before the overlap is
        # what the tracker keeps comparing against.
        if body is not None and not occluded:
            self.body.add(body)

    @property
    def stature(self):
        """How tall this person is, corrected for distance. None until
        the camera's perspective has been learned."""
        return self.body.stature

    @property
    def build(self):
        """How broad they are for their height - the 'weight' cue."""
        return self.body.build

    def confirmed(self, min_hits=TRACK_MIN_HITS):
        return self.hits >= min_hits

    def contains_point(self, px, py):
        x1, y1, x2, y2 = self.box
        return x1 <= px <= x2 and y1 <= py <= y2

    # ------------------------------------------------------- movement
    def motion(self):
        """approaching / receding / steady / unclear.

        TWO measurements have to agree about which way somebody is
        walking, because either one alone is wrong often enough to matter
        on a reception door where the answer decides whether an arrival
        is recorded:

          SIZE  a person walking toward the camera grows in the frame.
                This is the primary cue, but a box that clips the top of
                the frame, or a person who raises their arms, changes
                size without anybody moving.

          FEET  a person walking toward the camera also moves DOWN the
                image - their feet land on lower and lower rows of the
                floor - and away from it, up. This is unaffected by the
                box being clipped and is the cue that separates somebody
                genuinely coming to the desk from somebody crossing the
                lobby who merely appears to grow.

        Medians over the first and last third, so one bad box cannot flip
        the verdict, and the foot movement is expressed as a fraction of
        the person's own height so it means the same for somebody at the
        door and somebody by the desk.
        """
        n = len(self.sizes)
        if n < 6:
            return UNCLEAR
        third = max(2, n // 3)
        early = sorted(list(self.sizes)[:third])
        late = sorted(list(self.sizes)[-third:])
        e = early[len(early) // 2]
        l = late[len(late) // 2]
        if e <= 0:
            return UNCLEAR
        ratio = l / e

        # how far the feet travelled, in body heights (+ = toward camera)
        travel = 0.0
        if len(self.feet) >= 6:
            feet_early = sorted(list(self.feet)[:third])
            feet_late = sorted(list(self.feet)[-third:])
            travel = ((feet_late[len(feet_late) // 2]
                       - feet_early[len(feet_early) // 2]) / e)

        grew = ratio >= ENTRY_APPROACH_RATIO
        shrank = ratio <= ENTRY_RECEDE_RATIO
        came_down = travel >= 0.30
        went_up = travel <= -0.30

        # Either cue on its own is enough when it is emphatic; when only
        # one is, the other must at least not contradict it.
        if (grew and not went_up) or (came_down and ratio > 1.02):
            return APPROACHING
        if (shrank and not came_down) or (went_up and ratio < 0.98):
            return RECEDING
        return STEADY

    # ------------------------------------------------------- identity
    def would_accept_identity(self, name, score):
        """Would this track accept that name? A PURE CHECK - it changes
        nothing.

        This has to stay side-effect free. The camera asks this first and
        only then asks the name registry whether the name is free. An
        earlier version applied the name here and asked the registry
        afterwards, so a refused claim still left the name stuck on the
        track - which is how two people ended up labelled with the same
        name on screen.
        """
        if not name or name == "Unknown":
            return False
        if self.identified and score < self.score + FACE_UPGRADE_MARGIN:
            return False        # already have a name, and this is no better
        return True

    def apply_identity(self, name, code, score):
        """Actually attach the name. Only the camera calls this, and only
        after the registry has granted the claim."""
        self.name = name
        self.emp_code = code
        self.score = float(score)
        self.identified = True

    def clear_identity(self):
        self.name = "Unknown"
        self.emp_code = ""
        self.score = 0.0
        self.identified = False

    @property
    def label(self):
        return self.name if self.identified else f"Unknown #{self.id}"


class _Ghost:
    """A track that just died, kept briefly in case the same person
    reappears (detector blink, brief occlusion)."""

    __slots__ = ("track", "died_at")

    def __init__(self, track, died_at):
        self.track = track
        self.died_at = died_at


class PersonTracker:
    def __init__(self, iou_threshold=TRACK_IOU_THRESHOLD,
                 max_distance=TRACK_MAX_DISTANCE,
                 min_hits=TRACK_MIN_HITS, max_age=TRACK_MAX_AGE,
                 backend=None):
        self.iou_threshold = iou_threshold
        self.max_dist2 = max_distance * max_distance
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks = {}
        self.ghosts = []
        self._next_id = 1

        # What a standing person's pixel height is at each row of THIS
        # camera's floor. Learned from the tracker's own boxes, and the
        # thing that turns "300 pixels tall" into "this person is tall".
        self.perspective = Perspective()

        self.backend = (backend or TRACKER_BACKEND or "bytetrack").lower()
        self._byte = None
        self._by_byte = {}         # ByteTrack id -> our Track
        if self.backend == "bytetrack":
            from core.bytetrack import ByteTracker
            self._byte = ByteTracker(
                body_weight=(BYTETRACK_BODY_WEIGHT
                             if BYTETRACK_USE_BODY else 0.0))

    # --------------------------------------------------------- match
    def _pair_score(self, box, body, track):
        """How well does this detection fit this track?

        The rule that matters here:

            GEOMETRY decides what is POSSIBLE.
            SIZE only decides WHICH of the possible candidates.

        Size must never reject a candidate on its own. A person half
        behind a door frame measures short and narrow for a few frames,
        and a veto on that throws away the only correct match and invents
        a new, unnamed person - the "turns round and becomes Unknown"
        fault, which was never really about turning round.

        So: if the body is plausibly in the right place, the pair is
        eligible. Stature and build then rank the eligible pairs, which
        is what keeps two people who pass close together from swapping
        tracks.

        Returns 0.0 when geometrically impossible, otherwise > 0.
        """
        overlap = iou(box, track.box)
        cx, cy = centre(box)
        dist2 = (cx - track.cx) ** 2 + (cy - track.cy) ** 2

        # ---- plausibility: geometry alone ----
        if overlap <= 0.0 and dist2 > self.max_dist2:
            return 0.0

        closeness = 1.0 - min(1.0, (dist2 / self.max_dist2) ** 0.5)
        fits = track.body.score(body) if body is not None else None

        # ---- ranking among plausible pairs ----
        score = W_IOU * overlap + W_DIST * closeness
        score += W_BODY * (fits if fits is not None
                           else max(overlap, closeness))

        # never let a plausible pair fall out of contention entirely
        return max(score, MIN_PLAUSIBLE)

    def _assign(self, detections, bodies):
        """Match all detections to all tracks at once, best pair first.

        Assigning globally rather than one detection at a time is what
        stops two people who pass close together from stealing each
        other's track.
        """
        pairs = []
        for di, box in enumerate(detections):
            for tid, track in self.tracks.items():
                score = self._pair_score(box, bodies[di], track)
                if score >= MATCH_ACCEPT:
                    pairs.append((score, di, tid))
        pairs.sort(reverse=True)

        det_to_track, used_dets, used_tracks = {}, set(), set()
        for score, di, tid in pairs:
            if di in used_dets or tid in used_tracks:
                continue
            det_to_track[di] = tid
            used_dets.add(di)
            used_tracks.add(tid)
        return det_to_track

    def _match_coasting(self, box, now, body, taken):
        """Is this a person we are STILL HOLDING but did not see just now?

        This is the gap that produced most of the "he turned round and
        became Unknown" reports. A person hidden behind a colleague or a
        pillar for a second or two is dropped by ByteTrack and comes
        back under a NEW id. Their original track has not died - it
        coasts for TRACK_MAX_AGE - so the ghost layer, which only looks
        at DEAD tracks, never sees it. Nothing matched, a new nameless
        track was created, and the real one sat invisible still holding
        the name, the visit row and the accumulated presence time. Two
        tracks, one person, and the wrong one on screen.

        The question asked here is a much easier one than the gallery's.
        It is not "who is this out of sixty-four people", it is "which of
        the two or three people who vanished from this camera in the last
        few seconds is this" - a small candidate set over a short
        window, where being the right height and build is strong
        evidence rather than a guess.
        """
        if not RECLAIM_COASTING:
            return None

        cx, cy = centre(box)
        h = height(box)
        best, best_score = None, 0.0

        for track_id, track in self.tracks.items():
            if track_id in taken or track.last_seen >= now:
                continue                      # already matched this frame
            gap = now - track.last_seen
            if gap <= 0 or gap > RECLAIM_MAX_GAP_SECONDS:
                continue

            ratio = max(h, track.height) / max(1, min(h, track.height))
            if ratio > REID_SIZE_RATIO:
                continue                      # very different size

            # How far could they have walked while out of sight? About
            # one body height a second, so the allowance grows with the
            # length of the gap instead of being a fixed radius.
            allowed = max(h, track.height) * (
                REID_MAX_DISTANCE_FACTOR + RECLAIM_DISTANCE_PER_SECOND * gap)
            distance = ((cx - track.cx) ** 2 + (cy - track.cy) ** 2) ** 0.5
            if distance > allowed:
                continue

            closeness = 1.0 - min(1.0, distance / max(1.0, allowed))
            fits = track.body.score(body) if body is not None else None

            if fits is not None and fits >= BODY_RECLAIM_MIN:
                # The right size AND somewhere they could have got to.
                # This is the strong case.
                score = 0.55 + 0.45 * fits
            elif fits is not None and fits < BODY_RECLAIM_MIN * 0.6:
                # Clearly a different size. Size is never allowed to VETO
                # on its own elsewhere, but here it is deciding between
                # candidates rather than rejecting a detection outright,
                # so a plain mismatch is a reason to prefer another one.
                continue
            else:
                # No usable measurement - fall back to geometry, and
                # demand that the person be close to where we lost them.
                if distance > allowed * 0.5:
                    continue
                score = 0.30 * closeness

            score += 0.10 * closeness
            if score > best_score:
                best, best_score = track, score

        return best

    def _match_ghost(self, box, now, body=None):
        """Is this detection somebody we recently lost? If so, revive them
        with their original id, name and accumulated time.

        Size carries most of the weight here: after an occlusion the
        person may have moved a long way, so position is weak evidence,
        while their height and build are exactly what they were. It is
        still not a veto - a close body of plausible size that we could
        not measure can still be revived on geometry alone.
        """
        cx, cy = centre(box)
        h = height(box)
        best, best_score = None, None
        for g in self.ghosts:
            if now - g.died_at > REID_MEMORY_SECONDS:
                continue
            t = g.track
            ratio = max(h, t.height) / max(1, min(h, t.height))
            if ratio > REID_SIZE_RATIO:
                continue                       # very different size
            gap = max(0.0, now - g.died_at)
            allowed = REID_MAX_DISTANCE_FACTOR * max(h, t.height)
            dist = ((cx - t.cx) ** 2 + (cy - t.cy) ** 2) ** 0.5

            fits = t.body.score(body) if body is not None else None
            if fits is not None and fits >= BODY_REID_MIN:
                # The right size, so the question is only whether they
                # could have GOT there. The allowance grows with the time
                # they were out of sight, because a fixed radius cannot
                # express "gone for ten seconds, so could be anywhere" -
                # and refusing a 0.75 size match on distance alone is
                # what turned a returning colleague into a stranger.
                reach = allowed + (max(h, t.height)
                                   * REID_DISTANCE_PER_SECOND * gap)
                if dist > reach:
                    continue
                score = 0.5 + 0.5 * fits
            else:
                if dist > allowed:
                    continue
                score = 0.5 * (1.0 - dist / max(1.0, allowed))

            if best_score is None or score > best_score:
                best, best_score = g, score
        return best

    # -------------------------------------------------------- update
    def update(self, boxes, now=None, frame=None, scores=None):
        """boxes: person detections this frame.
        scores: detector confidences, aligned with boxes. Required for
                the ByteTrack backend to do anything useful - without
                them every detection looks equally confident and the
                second association pass has nothing to work with.
        frame:  optional image. Only its SHAPE is needed now that people
                are measured rather than described by colour, but it is
                still accepted so callers did not have to change.

        Returns (live_tracks, finished_tracks). A track is reported as
        FINISHED only when it is genuinely gone - when its ghost finally
        expires - so a momentary dropout never closes anybody.
        """
        now = now or time.time()
        finished = self._expire_ghosts(now)

        bodies = self._measure_all(boxes, frame)

        if self._byte is not None:
            return self._update_bytetrack(boxes, scores, bodies, now,
                                          finished)

        assignment = self._assign(boxes, bodies)

        live, taken = [], set()
        for di, box in enumerate(boxes):
            overlapped = self._overlaps_another(di, boxes)
            tid = assignment.get(di)
            if tid is not None:
                track = self.tracks[tid]
                track.observe(box, now, bodies[di], occluded=overlapped)
                taken.add(tid)
                live.append(track)
                continue

            ghost = self._match_ghost(box, now, bodies[di])
            if ghost is not None:
                self.ghosts.remove(ghost)
                track = ghost.track
                track.observe(box, now, bodies[di], occluded=overlapped)
                self.tracks[track.id] = track
                taken.add(track.id)
                live.append(track)
                continue

            new_id = self._next_id
            self._next_id += 1
            track = Track(new_id, box, now)
            if not overlapped:
                track.body.add(bodies[di])
            self.tracks[new_id] = track
            taken.add(new_id)
            live.append(track)

        # unmatched tracks coast, then become ghosts - still not finished
        for tid in list(self.tracks):
            if tid in taken:
                continue
            track = self.tracks[tid]
            if now - track.last_seen > self.max_age:
                del self.tracks[tid]
                self.ghosts.append(_Ghost(track, now))

        return live, finished

    # --------------------------------------------- measuring
    def _measure_all(self, boxes, frame):
        """Measure every detection - height, build, where the feet are.

        A box that overlaps another box is NOT measured. It is not one
        person: it is a box drawn around two, taller than either of them
        and far wider, and both the perspective model and the track's own
        running measurement are poisoned by it. Returning None says "I
        could not measure this one", and every caller below treats that
        as no evidence rather than as evidence against.
        """
        if not BODY_ENABLED or frame is None:
            return [None] * len(boxes)
        shape = frame.shape
        out = []
        for i, box in enumerate(boxes):
            if self._overlaps_another(i, boxes):
                out.append(None)
                continue
            out.append(measure(box, shape, self.perspective, learn=True))
        return out

    # --------------------------------------------- ByteTrack backend
    def _bind(self, byte_id, track):
        """Point one ByteTrack id at one of our Tracks, exclusively.

        A revived ghost can arrive under a NEW ByteTrack id while its old
        one is still in the map. Leaving both would let two ByteTrack
        tracks feed the same Track - two boxes, one person's history -
        so any stale binding to this Track is dropped first.
        """
        for existing_id, existing in list(self._by_byte.items()):
            if existing is track and existing_id != byte_id:
                del self._by_byte[existing_id]
        self._by_byte[byte_id] = track

    def _update_bytetrack(self, boxes, scores, bodies, now, finished):
        """Association by ByteTrack, everything else exactly as before.

        The two layers do different jobs and both are worth keeping:
        ByteTrack holds a person together WITHIN a continuous sighting,
        including through the low-confidence frames that used to break
        them; the ghost layer below reconnects them ACROSS a real gap -
        somebody who sat down, disappeared for twenty seconds and came
        back - using how big they are, which is evidence ByteTrack's
        motion model does not have.
        """
        live_byte, removed_byte = self._byte.update(boxes, scores, bodies)

        # WHO IS OVERLAPPING WHOM, this frame.
        #
        # While two people overlap there is no honest measurement of
        # either of them: one box contains part of the other person, so
        # it is taller and much broader than the person it belongs to.
        # Recording that is how a track's stature drifts toward its
        # neighbour's - and that is precisely why, after they separate,
        # the wrong body could keep the identity. Tracking continues as
        # normal; only the MEASURING is paused, and the settled figures
        # from before the overlap are what the two are told apart by
        # when they come out of it.
        occluded = {}
        byte_boxes = [t.box for t in live_byte]
        for i, st in enumerate(live_byte):
            for j in range(len(live_byte)):
                if i != j and iou(byte_boxes[i], byte_boxes[j]) >= OCCLUSION_IOU:
                    occluded[id(st)] = True
                    break

        live, seen = [], set()
        for st in live_byte:
            box = tuple(int(round(v)) for v in st.box)
            track = self._by_byte.get(st.track_id)

            if track is None or track.id not in self.tracks:
                # Order matters: the shortest gap first. A person we are
                # still holding is a better explanation for a new box
                # than a person we gave up on, and a person we gave up on
                # is a better explanation than a stranger.
                reclaimed = self._match_coasting(box, now, st.descriptor, seen)
                if reclaimed is not None:
                    track = reclaimed
                    if VERBOSE_IDENTITY and track.identified:
                        print(f"[TRACK] {track.name} (#{track.id}) came back "
                              f"after {now - track.last_seen:.1f}s out of "
                              f"sight - kept their identity")
                else:
                    ghost = self._match_ghost(box, now, st.descriptor)
                    if ghost is not None:
                        self.ghosts.remove(ghost)
                        track = ghost.track
                        self.tracks[track.id] = track
                    elif self._is_duplicate_of_live(box, st.descriptor, seen):
                        # A second box on somebody we are ALREADY tracking
                        # this very frame. Not a new person - a duplicate
                        # detection. Creating a track for it invents a
                        # phantom colleague standing on top of a real one,
                        # and the phantom then competes for the name.
                        continue
                    else:
                        self._report_new_track(box, now, st.descriptor)
                        track = Track(self._next_id, box, now)
                        self._next_id += 1
                        if not occluded.get(id(st), False):
                            track.body.add(st.descriptor)
                        self.tracks[track.id] = track
                self._bind(st.track_id, track)

            track.observe(box, now, st.descriptor,
                          occluded=occluded.get(id(st), False))
            seen.add(track.id)
            live.append(track)

        for st in removed_byte:
            self._by_byte.pop(st.track_id, None)

        # Our own coasting is kept on TOP of ByteTrack's. ByteTrack stops
        # REPORTING a track the moment it is unmatched; this project has
        # always let a track survive a short dropout without any change
        # downstream, and TRACK_MAX_AGE is what entry counting and
        # presence sessions are tuned against.
        for track_id in list(self.tracks):
            if track_id in seen:
                continue
            track = self.tracks[track_id]
            if now - track.last_seen > self.max_age:
                del self.tracks[track_id]
                self.ghosts.append(_Ghost(track, now))
                for byte_id, bound in list(self._by_byte.items()):
                    if bound is track:
                        del self._by_byte[byte_id]

        return live, finished

    def _is_duplicate_of_live(self, box, body, seen):
        """Is this box a second detection of somebody already tracked?

        Measured on this site: a detection 67px from a live track, and
        the same size as it, was becoming a separate person. That is not
        two people standing very close - it is one body detected twice.
        The detector-side de-duplication (core/detector.py) removes most
        of these; this is the backstop for the ones whose boxes are
        offset rather than nested.
        """
        for track_id in seen:
            track = self.tracks.get(track_id)
            if track is None:
                continue
            overlap = iou(box, track.box)
            if overlap < 0.45:
                continue
            fits = track.body.score(body) if body is not None else None
            # Heavy overlap alone is enough when there is nothing to
            # measure; with a measurement, it has to agree as well, so
            # two people genuinely standing close together still get
            # their own tracks.
            if fits is None or fits >= BODY_RECLAIM_MIN + 0.20:
                return True
        return False

    def _report_new_track(self, box, now, body):
        """Say when a new person is invented next to a known one.

        This is the diagnostic for "he walked behind a pillar and came
        back as Unknown". When that happens there is almost always an
        identified track sitting right there that we declined to
        reclaim - and the only thing worth knowing is WHY we declined:
        too far, too different, or too long ago. Guessing at that from
        the outside is impossible, which is why it is printed.
        """
        if not VERBOSE_IDENTITY:
            return
        cx, cy = centre(box)
        h = height(box)
        nearby = []
        for track in list(self.tracks.values()) + [g.track for g in self.ghosts]:
            if not track.identified:
                continue
            distance = ((cx - track.cx) ** 2 + (cy - track.cy) ** 2) ** 0.5
            # Only report a candidate the reclaim logic could plausibly
            # have taken. A wider net turns this diagnostic into noise:
            # it starts naming somebody standing right across the room as
            # though we had declined to reclaim them.
            reach = max(h, track.height) * (
                REID_MAX_DISTANCE_FACTOR +
                RECLAIM_DISTANCE_PER_SECOND * max(0.0, now - track.last_seen))
            if distance > reach:
                continue
            fits = track.body.score(body) if body is not None else None
            nearby.append((distance, track, fits, now - track.last_seen))
        if not nearby:
            return
        nearby.sort(key=lambda row: row[0])
        distance, track, fits, gap = nearby[0]
        print(f"[TRACK] new person created near {track.name} (#{track.id}): "
              f"{distance:.0f}px away, {gap:.1f}s since seen, size fits "
              f"{'n/a' if fits is None else f'{fits:.2f}'} "
              f"(reclaim needs {BODY_RECLAIM_MIN}, within "
              f"{RECLAIM_MAX_GAP_SECONDS}s) - if this is the same person, "
              f"that is the setting to relax")

    @staticmethod
    def _overlaps_another(index, boxes):
        """Is this detection sitting on top of another one?"""
        for other in range(len(boxes)):
            if other != index and iou(boxes[index], boxes[other]) >= OCCLUSION_IOU:
                return True
        return False

    def _expire_ghosts(self, now):
        """Drop ghosts that have waited too long and report them finished
        - this is the single point at which a person is considered gone."""
        still_waiting, gone = [], []
        for g in self.ghosts:
            if now - g.died_at <= REID_MEMORY_SECONDS:
                still_waiting.append(g)
            else:
                gone.append(g.track)
        self.ghosts = still_waiting
        return gone

    def all_tracks(self):
        return list(self.tracks.values())
