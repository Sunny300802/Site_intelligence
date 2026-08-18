"""
cameras/base.py
===============
Shared plumbing for every camera handler.

Each camera in the system gets its own handler class (reception.py is the
first). A handler owns: its frame grabber, its tracker, its face worker,
and its drawing. The pipeline just asks every handler for a frame,
detects people for all of them in one batched GPU call, then hands each
handler its own detections back.

Adding camera 2 later means writing one new handler file - nothing in the
pipeline or the dashboard has to change.
"""
import os
import time

import cv2

from core.frame_grabber import FrameGrabber
from core.tracker import PersonTracker
from core.face import FaceRecognizer
from core.identity import NameRegistry
from core.enhance import enhance
from core.body import measure
from core.gallery import GALLERY
from config.settings import (FACE_ENABLED, FACE_EVERY_N_FRAMES, IMG_SIZE,
                             PERSON_CONF, ADAPTIVE_DETECTION,
                             ADAPTIVE_MAX_SKIP, REVIEW_DIR,
                             REVIEW_COOLDOWN, REVIEW_NAME_COOLDOWN,
                             LEARN_BODY_SIZE, VERBOSE_IDENTITY,
                             NAME_STEAL_MARGIN, FACE_RECOGNITION_THRESHOLD,
                             VERBOSE, FACE_LIBRARY_AUTO_CAPTURE,
                             ASK_ABOUT_UNKNOWN, ASK_UNKNOWN_MIN_SECONDS,
                             ASK_UNKNOWN_MIN_QUALITY, ASK_AGAIN_SECONDS,
                             MIN_PERSON_HEIGHT_FRAC, VLM_ENABLED,
                             VLM_VERIFY_TRACKS, VLM_VERIFY_MIN_HITS,
                             VLM_PERSON_MIN_HEIGHT_FRAC, VLM_VERIFY_IDENTITY,
                             VLM_IDENTITY_MIN_FACE_WIDTH,
                             BEST_FRAME_QUALITY_THRESHOLD,
                             VLM_SCREEN_QUESTIONS)
from core.face import learn_face, agrees_with_enrollment
from core.face_library import LIBRARY
from core import vlm
from core.vlm_verify import ARBITER, has_references, ranked_candidates


# a readable palette (BGR)
GREEN = (80, 200, 120)
AMBER = (0, 170, 255)
GREY = (160, 160, 160)
WHITE = (255, 255, 255)
CYAN = (200, 200, 0)
DARK = (30, 30, 30)


class BaseCamera:
    handler_name = "base"

    def __init__(self, cfg):
        self.key = cfg["key"]
        self.name = cfg["name"]
        self.url = cfg["url"]
        self.options = cfg.get("options", {})

        self.grabber = FrameGrabber(self.url, name=self.name).start()
        self.tracker = PersonTracker()
        # one name can only belong to one person at a time on this camera
        self.names = NameRegistry(steal_margin=NAME_STEAL_MARGIN)
        self._review_dir = os.path.join(REVIEW_DIR, self.key)
        self._last_review = {}     # track id -> when we last queued it
        self._asked_recently = set()
        self._confirmed_today = set()
        self._asked_at = {}        # name -> when we last asked about them
        self._asked_unknown = {}   # track id -> when we asked "who is this?"
        self._name_guard_t0 = 0.0
        self._why_t0 = 0.0
        # Names the vision model has REFUSED for a track (see
        # vlm_review below). Kept per track rather than per name: the
        # refusal is about this body, not about the person - the same
        # name may be perfectly correct on somebody else in the frame.
        self._vlm_denied = {}      # track id -> {name, ...}

        self.face = None
        if FACE_ENABLED and self.options.get("recognise_faces", True):
            # min_face_size is per camera: a 720p feed and a 1440p feed
            # produce faces of completely different pixel sizes, and one
            # global number is either far too strict for one or far too
            # loose for the other. The [FACE] debug line reports the
            # measured widths so this can be set from evidence.
            self.face = FaceRecognizer(
                name=self.name,
                min_face_size=self.options.get("min_face_size"))

        # Optional DETECTION REGION. Cropping the frame before detection
        # does two things at once: the model no longer wastes pixels on
        # areas nobody walks through (a blank wall, the ceiling), and the
        # people who ARE there become proportionally larger in the image
        # the model sees, so distant ones get detected. Boxes are mapped
        # back to full-frame coordinates afterwards, so everything
        # downstream is unaffected.
        self.detect_roi = self.options.get("detect_roi")
        self._roi_px = None

        # ---- per-camera detection settings ----------------------
        # Different cameras need different treatment. At a reception,
        # people move fast, so detection must run on every frame or they
        # are past the door before we see them. In a workspace people are
        # seated and barely move, so detecting a few times a second is
        # plenty - and spending that saved time on a larger input size
        # and image enhancement buys far more accuracy than raw rate.
        self.imgsz = self.options.get("imgsz", IMG_SIZE)
        self.conf = self.options.get("conf", PERSON_CONF)
        self.detect_every = max(1, int(self.options.get("detect_every", 1)))
        self.enhance_profile = self.options.get("enhance")
        self.min_person_height_frac = self.options.get(
            "min_person_height_frac", MIN_PERSON_HEIGHT_FRAC)
        self._skip = 0             # extra skipping added automatically
                                   # when the machine cannot keep up

        self._last_frame_id = -1
        self.frame = None          # frame currently being processed
        self.annotated = None      # frame with drawings, for the stream
        self.fps = 0.0
        self.frame_count = 0

    # ------------------------------------------------------- lifecycle
    def grab(self):
        """Fetch the newest frame. Returns True if it is new."""
        fid, frame = self.grabber.read()
        if frame is None or fid == self._last_frame_id:
            return False
        self._last_frame_id = fid
        self.frame = frame
        self.frame_count += 1
        return True

    def _roi_pixels(self, w, h):
        if self.detect_roi is None:
            return None
        if self._roi_px is None:
            x1, y1, x2, y2 = self.detect_roi
            self._roi_px = (
                max(0, int(x1 * w if x1 <= 1 else x1)),
                max(0, int(y1 * h if y1 <= 1 else y1)),
                min(w, int(x2 * w if x2 <= 1 else x2)),
                min(h, int(y2 * h if y2 <= 1 else y2)),
            )
        return self._roi_px

    def frame_for_detection(self):
        """The image the detector should look at: cropped to the region
        of interest, then enhanced if this camera needs it."""
        if self.frame is None:
            return None
        h, w = self.frame.shape[:2]
        roi = self._roi_pixels(w, h)
        if roi is None:
            out = self.frame
        else:
            x1, y1, x2, y2 = roi
            out = (self.frame if (x2 - x1 < 32 or y2 - y1 < 32)
                   else self.frame[y1:y2, x1:x2])
        if self.enhance_profile:
            out = enhance(out, self.enhance_profile)
        return out

    def map_detections(self, detections):
        """Shift detection boxes from crop coordinates back to the full
        frame, so drawing and tracking stay in one coordinate system."""
        if self.frame is None or self.detect_roi is None:
            return detections
        h, w = self.frame.shape[:2]
        roi = self._roi_pixels(w, h)
        if roi is None:
            return detections
        ox, oy = roi[0], roi[1]
        if ox == 0 and oy == 0:
            return detections
        out = []
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            out.append({**d, "box": (x1 + ox, y1 + oy, x2 + ox, y2 + oy)})
        return out

    def draw_detect_roi(self, frame):
        h, w = frame.shape[:2]
        roi = self._roi_pixels(w, h)
        if roi is None:
            return
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]),
                      (70, 90, 110), 1)
        cv2.putText(frame, "detection region", (roi[0] + 6, roi[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 90, 110), 1)

    # --------------------------------------------------- scheduling
    def detect_key(self):
        """Cameras sharing these settings can be batched together.

        The minimum person height is part of the key because it is
        applied inside the detector, so two cameras with different
        values cannot share one call.
        """
        return (self.imgsz, self.conf, self.enhance_profile,
                self.min_person_height_frac)

    @property
    def effective_interval(self):
        return self.detect_every + self._skip

    def due_for_detection(self, cycle):
        return cycle % self.effective_interval == 0

    def ease_off(self):
        """Detect less often - called when the machine is falling behind."""
        if ADAPTIVE_DETECTION and self._skip < ADAPTIVE_MAX_SKIP:
            self._skip += 1
            print(f"[{self.name}] running behind - detecting every "
                  f"{self.effective_interval} frames")
            return True
        return False

    def catch_up(self):
        """Return toward the configured rate once there is headroom."""
        if self._skip > 0:
            self._skip -= 1
            print(f"[{self.name}] recovered - detecting every "
                  f"{self.effective_interval} frames")

    def health(self):
        """Stream status for the dashboard and the console summary."""
        h = self.grabber.health()
        h.update({"key": self.key, "name": self.name,
                  "detect_interval": self.effective_interval})
        return h

    def wants_face_pass(self):
        return (self.face is not None
                and self.frame_count % FACE_EVERY_N_FRAMES == 0)

    def submit_faces(self, live_tracks):
        """Hand the face worker this frame AND the tracks in it.

        Called from each handler's process(), immediately after tracking,
        rather than from the pipeline loop before it. That ordering is
        the point: the face stage associates every face with a tracking
        id, and it can only do that if it is told which ids were live in
        the frame it is being given. Submitting a frame before the
        tracker has seen it means associating this frame's faces with
        last frame's boxes, and on a reception camera - where people
        cross quickly - that is enough to put one person's face on
        another person's track.

        The box snapshot is copied here, on the camera thread, because
        the worker reads it on its own thread while the tracker is
        already rewriting the live Track objects for the next frame.
        """
        if self.face is None or self.frame is None:
            return False
        if not self.wants_face_pass():
            return False
        snapshot = [(t.id, tuple(t.box)) for t in live_tracks]
        return self.face.submit(self.frame, snapshot)

    def submit_face_frame(self):
        """Compatibility shim for anything still calling the old name."""
        return self.submit_faces(self.tracker.all_tracks())

    def process(self, detections):
        """Override: handle this camera's detections."""
        raise NotImplementedError

    def claim_identity(self, track, name, code, score, on_evict=None,
                       trusted=False):
        """Give a track a name, but only if no better claim exists.

        A person cannot be in two places at once, so a name is exclusive.
        If another track already holds it with a WEAKER score, that track
        loses the name and goes back to Unknown - better one correct label
        and one honest Unknown than two confident labels where one must be
        wrong.

        Returns True if this track now holds the name.
        """
        # TWO REFUSALS BEFORE ANY OF THAT, both of them about the moment
        # two people are on top of each other - which is when every
        # complaint about a name moving to the wrong person begins.
        #
        # WHILE OVERLAPPING, A GUESS CHANGES NOTHING HANDS. The boxes are
        # merged, the face between two heads could belong to either of
        # them, and a name applied now cannot be checked. It is not lost:
        # the moment they separate, the same evidence arrives again
        # against measurements that mean something.
        #
        # `trusted` is the exception, and it is only ever the CONFIRMED
        # face vote - several frames of a face that the ownership check
        # in core/face_pipeline.py agreed belongs to this body, not the
        # one behind it. Without that exception two colleagues who walk
        # in side by side would stay Unknown for as long as they stayed
        # together, which on a reception door is most of their visit.
        if track.occluded and not track.identified and not trusted:
            return False

        # AND THE VISION MODEL MUST NOT HAVE ALREADY REFUSED THIS NAME
        # FOR THIS BODY.
        #
        # This guard is what makes the veto stick. Without it the face
        # vote simply re-applies the same name on the next pass - it has
        # no idea the name was taken off - and the dashboard shows the
        # wrong name flickering back every few seconds. `trusted` is NOT
        # an exception here: a confirmed face vote is precisely the
        # thing that was refused.
        if name in self._vlm_denied.get(track.id, ()):
            return False

        # AND THE BODY HAS TO BE THE RIGHT SIZE FOR THE NAME. If we have
        # watched this person before and they are visibly not this
        # height and build, the face was read off somebody else.
        if not GALLERY.size_allows(name, track.body.typical()):
            if VERBOSE_IDENTITY:
                print(f"[{self.name}] refused '{name}' on track #{track.id}: "
                      f"the body is the wrong size for them")
            return False

        granted, evicted_id = self.names.claim(name, track.id, score)
        if not granted:
            return False

        if evicted_id is not None:
            for other in self.tracker.all_tracks():
                if other.id == evicted_id:
                    other.clear_identity()
                    if on_evict:
                        on_evict(other)
                    print(f"[{self.name}] '{name}' reassigned from track "
                          f"#{evicted_id} to #{track.id} (better match)")
                    break

        track.apply_identity(name, code, score)
        return True

    def release_identity(self, track):
        # LAST CHANCE to act on a late answer. Everything about this
        # track is about to be forgotten, and an answer that arrived
        # after it was last seen would go with it - leaving the wrong
        # name on a finished record permanently, which is the one place
        # nothing downstream can ever correct it.
        if VLM_ENABLED and VLM_VERIFY_IDENTITY and track.identified:
            self._vlm_apply_identity_verdict(track)
        self.names.release(track.id)
        self._last_review.pop(track.id, None)
        self._vlm_denied.pop(track.id, None)
        ARBITER.forget(self.key, track.id)
        if self.face is not None:
            self.face.forget(track.id)

    # ------------------------------------------- the vision model veto
    #
    # Everything else in this system judges a face against OTHER FACE
    # VECTORS - the threshold, the runner-up margin, the temporal vote,
    # the body-size veto, agrees_with_enrollment. They are all arguments
    # inside one embedding space, so when that space is wrong about
    # somebody they are all wrong together, confidently, and raising any
    # of them does not help. That is what a name that survives a human
    # confirmation looks like.
    #
    # The vision model is the only independent witness available: it has
    # not seen the embedding, it does not know what the tracker decided,
    # and it is looking at the pixels a person would look at. It is
    # allowed to REFUSE and nothing else - see core/vlm_verify.py for
    # why it is never allowed to name anybody.

    def vlm_review(self, tracks, faces, frame):
        """Ask the vision model to double-check this camera's people.

        Called once per processed frame by each handler. Nothing here
        blocks: questions are queued for one background thread and the
        answers are picked up on a later frame. Costs nothing at all
        when Ollama is not running.
        """
        if not VLM_ENABLED or frame is None or not ARBITER.usable():
            return
        height = frame.shape[0]

        # ACT ON ANSWERS FIRST, AND ACROSS EVERY TRACK WE ARE STILL
        # HOLDING - not only the ones detected in this frame.
        #
        # An answer takes about five seconds to come back, and on a
        # work-area camera a seated person is regularly undetected for
        # longer than that. Applying refusals only to the tracks in
        # THIS frame therefore lost them exactly when they mattered:
        # measured here, a wrong name was refused while its track was
        # coasting, the track was never re-examined before it ended, and
        # the person kept the 1.9 minutes credited to them. Reading a
        # stored verdict costs nothing, so it is done for everybody.
        if VLM_VERIFY_IDENTITY:
            for track in self.tracker.all_tracks():
                if track.identified:
                    self._vlm_apply_identity_verdict(track)

        # ...then ask new questions, only about what we can actually see.
        for track in tracks:
            if VLM_VERIFY_TRACKS:
                self._vlm_check_person(track, frame, height)
            if VLM_VERIFY_IDENTITY and track.identified:
                self._vlm_check_identity(track, faces, frame)
            if VLM_SCREEN_QUESTIONS and not track.identified:
                self._vlm_screen_question(track, faces, frame)

    def _vlm_screen_question(self, track, faces, frame):
        """Is this crop worth asking a human about?

        Screened BEFORE the question exists, not after. Everything
        upstream of the review queue is a number - detector confidence,
        sharpness, contrast, face quality - and a patch of door texture
        or the top of somebody's head scores perfectly well on all of
        them. That is why the queue filled with pictures of doors,
        floors, shoulders and backs of heads, each one asking "who is
        this?" about something nobody could answer.

        A verdict here also carries the DESCRIPTION of the face and the
        shortlist of people it does not rule out, so the card the
        reviewer eventually sees is a question they can actually answer.
        """
        if ARBITER.face_verdict(self.key, track.id) is not None:
            return                       # already screened
        if track.hits < self.tracker.min_hits:
            return
        if (time.time() - track.created) < ASK_UNKNOWN_MIN_SECONDS:
            return
        crop, _width = self._vlm_face_crop(track, faces, frame)
        if crop is None:
            return
        ARBITER.check_face(self.key, track.id, crop,
                           label=f"#{track.id} ({self.name})")

    def question_screening(self, track):
        """(may_ask, verdict) for this track's review card.

        Three states, and the middle one is the point:
          * not screened yet  -> WAIT. A question delayed by a few
            seconds is free; an unanswerable one is not.
          * screened, no face -> never ask.
          * screened, face    -> ask, and use the description.
        """
        if not VLM_ENABLED or not VLM_SCREEN_QUESTIONS \
                or not ARBITER.usable():
            return True, None
        verdict = ARBITER.face_verdict(self.key, track.id)
        if verdict is None:
            return False, None           # still being looked at
        if verdict.ok is None:
            return True, verdict         # could not tell - behave as before
        if verdict.refuses:
            return False, verdict        # no face in it: never ask
        return True, verdict

    def _vlm_check_person(self, track, frame, frame_height):
        """Is this box a human being, or a reflection of one?"""
        verdict = ARBITER.person_verdict(self.key, track.id)
        if verdict is not None:
            return                       # already answered
        if track.hits < VLM_VERIFY_MIN_HITS or track.occluded:
            return
        x1, y1, x2, y2 = [int(v) for v in track.box]
        if (y2 - y1) < VLM_PERSON_MIN_HEIGHT_FRAC * frame_height:
            # Too small to judge. Asking anyway would have the model
            # guessing about thirty pixels of blur, and a wrong veto
            # costs a real person their attendance.
            return
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return
        ARBITER.check_person(self.key, track.id, crop,
                             label=f"{self.name} #{track.id}")

    def _vlm_apply_identity_verdict(self, track):
        """Act on an answer already in hand. Asks nothing, costs nothing.

        Separate from _vlm_check_identity so it can be run over every
        track we are holding, including ones not seen this frame - see
        vlm_review for why that distinction was a bug and not a detail.
        Returns True if an answer had already arrived for this name.
        """
        verdict = ARBITER.identity_verdict(self.key, track.id, track.name)
        if verdict is None:
            return False
        if verdict.refuses:
            self._vlm_refuse_name(track, track.name, verdict)
        return True

    def _vlm_check_identity(self, track, faces, frame):
        """Is the name on this body contradicted by what we can see?"""
        name = track.name
        if self._vlm_apply_identity_verdict(track):
            return                       # already answered

        crop, width = self._vlm_face_crop(track, faces, frame)
        if crop is None or width < VLM_IDENTITY_MIN_FACE_WIDTH:
            return
        if not has_references(name, track.emp_code):
            # Nothing a human took to compare against. Staying quiet is
            # right: the alternative is checking a suspect name against
            # the very captures a wrong name poisons.
            return
        ARBITER.check_identity(self.key, track.id, name, crop,
                               code=track.emp_code,
                               label=f"{name} on #{track.id} ({self.name})")

    def _vlm_face_crop(self, track, faces, frame):
        """The best picture of this track's face, and how wide it is.

        The face found THIS pass, at native resolution with margin, is
        the first choice - it is the sharpest thing available and it
        keeps the hair, glasses and jawline the model is being asked
        about. On a work-area camera a face is rarely readable in any
        given frame, so the aligned crop the recogniser actually matched
        on is used instead: if the answer came off that picture, that
        picture is what should be argued about.

        EITHER WAY IT MUST CLEAR BEST_FRAME_QUALITY_THRESHOLD, and that
        gate is the difference between this check working and it doing
        harm. Measured with tools/vlm_check.py on this site's own files,
        every case where the vision model refused a CORRECT name was a
        crop with no face in it - the top of somebody's head, a
        shoulder, a downward blur. The model does not answer "I cannot
        see a face"; it describes what it thinks it sees, and on the top
        of a head it says "male, short hair" whoever that is. So it is
        only ever shown a face this system itself judged good enough to
        base an identity on.
        """
        record = self.face_record_for(track, faces)
        if record is not None and record.get("box") and \
                float(record.get("quality") or 0.0) >= \
                BEST_FRAME_QUALITY_THRESHOLD:
            box = record["box"]
            width = int(box[2] - box[0])
            crop = LIBRARY.crop_face(frame, box)
            if crop is not None:
                return crop, width

        if self.face is None:
            return None, 0
        best = self.face.best_face(track.id)
        if best is None:
            return None, 0
        crop, _quality, _age = best
        # The aligned crop is a fixed 112px square regardless of how big
        # the face was in the frame, so it carries no width of its own -
        # it only exists because a face good enough to recognise was
        # found, which is the bar this check needs anyway.
        return crop, VLM_IDENTITY_MIN_FACE_WIDTH

    def _vlm_refuse_name(self, track, name, verdict):
        """Take a refused name off a track, and keep it off."""
        denied = self._vlm_denied.setdefault(track.id, set())
        first_time = name not in denied
        denied.add(name)

        if track.identified and track.name == name:
            track.clear_identity()
            self.names.release_name(name, track.id)
            # Clear the face stage's vote for this track as well.
            # Otherwise the confirmed verdict sits there and re-offers
            # the same name on the next pass - claim_identity would
            # refuse it, but the track would spend the rest of its life
            # arguing with a decision already made.
            if self.face is not None:
                self.face.forget(track.id)
            self.on_identity_refused(track, name)

        if first_time and VERBOSE:
            print(f"[{self.name}] track #{track.id} is NOT {name} - "
                  f"the vision model says: {verdict.reason} "
                  f"(confidence {verdict.confidence:.2f}). Back to Unknown.")

    def on_identity_refused(self, track, name):
        """Hook: a name was taken off this track. Override to fix rows."""

    def clear_vlm_denial(self, track, name=None):
        """Forget a refusal, because a HUMAN has now answered.

        A person's explicit confirmation outranks a 7B model looking at
        a blurry crop, and it has to: the veto exists to protect the
        record from a machine that is confidently wrong, not to argue
        with the operator. So a confirmation clears the refusal AND the
        stored verdict, rather than being silently ignored - which is
        what would otherwise happen, and would look exactly like the
        dashboard doing nothing.
        """
        denied = self._vlm_denied.get(track.id)
        if denied is None:
            return
        if name is None:
            self._vlm_denied.pop(track.id, None)
        else:
            denied.discard(name)
            if not denied:
                self._vlm_denied.pop(track.id, None)
        ARBITER.forget(self.key, track.id)

    def is_real_person(self, track):
        """False only when the vision model said this is not a person."""
        if not VLM_ENABLED or not VLM_VERIFY_TRACKS:
            return True
        verdict = ARBITER.person_verdict(self.key, track.id)
        return not (verdict is not None and verdict.refuses)

    # -------------------------------------------- the voted identity
    def face_probability(self, score):
        """Put a face similarity on the gallery's 0-1 probability scale.

        A CONFIRMED face vote deliberately lands just below a human's
        answer (CONFIRMED_SCORE, 1.0) and above every automatic match,
        so nothing can take a name off somebody whose face we have
        actually seen and voted on.
        """
        # The EFFECTIVE bar, from the model that loaded - the same one
        # the vote was judged against. Using the configured value here
        # while the face stage used a different one would map scores
        # onto the wrong part of the probability scale.
        bar = (self.face.threshold if self.face is not None
               else FACE_RECOGNITION_THRESHOLD)
        span = max(1e-6, 1.0 - bar)
        above = max(0.0, float(score) - bar) / span
        return min(0.99, 0.90 + 0.09 * above)

    def voted_identity(self, track):
        """The face stage's verdict for this track, or None."""
        if self.face is None:
            return None
        return self.face.verdict(track.id)

    def apply_voted_identity(self, track, on_evict=None):
        """Put a CONFIRMED voted identity onto the track.

        Returns (applied, verdict). Only a confirmed verdict is used
        here - a provisional one is still evidence, and it reaches the
        gallery through the embedding like any other cue, but it is not
        allowed to put a name on screen on its own. That distinction is
        the whole reason the voting layer exists.
        """
        verdict = self.voted_identity(track)
        if verdict is None or not verdict.confirmed or not verdict.known:
            return False, verdict

        probability = self.face_probability(verdict.score)
        if track.identified and track.name == verdict.name:
            return False, verdict          # already wearing it
        if not track.would_accept_identity(verdict.name, probability):
            return False, verdict
        if not self.claim_identity(track, verdict.name, verdict.code,
                                   probability, on_evict, trusted=True):
            return False, verdict

        if VERBOSE:
            print(f"[{self.name}] track #{track.id} = {verdict.name} "
                  f"(face vote: {verdict.votes} observation(s), "
                  f"similarity {verdict.score:.2f}, "
                  f"share {verdict.share:.2f})")
        return True, verdict

    # ------------------------------------- building the face library
    def face_record_for(self, track, faces):
        """The face the stage attributed to this track, or None."""
        for f in faces:
            if f.get("track_id") == track.id:
                return f
        return None

    def capture_reference(self, track, faces, frame, source="auto"):
        """Keep a camera photograph of somebody we are sure about.

        Called only when the temporal vote has CONFIRMED the identity -
        several good frames agreeing, over the threshold, clear of
        everybody else. That is a far stronger statement than one
        frame's match, which is why this can file a reference without
        asking anybody.

        The picture goes into that person's own enrollment folder, and
        the embedding goes straight into the live search index, so the
        next person through the door is already matched against it.
        """
        if not FACE_LIBRARY_AUTO_CAPTURE or not track.identified:
            return None

        # THE VOTE MUST AGREE, not merely the track's current label.
        #
        # A track can carry a name for reasons that are good enough to
        # display but not good enough to teach from for ever - a seat,
        # or a name the tracker carried in from another part of the
        # room. Filing a
        # photograph under a name that turns out to be wrong poisons
        # that person's references permanently and is very hard to
        # notice afterwards, so capture insists on the strongest thing
        # this system produces - several good frames of THIS face
        # agreeing on THIS name.
        #
        # A human confirmation also qualifies: force() marks the verdict
        # confirmed, which is exactly what an answer on the dashboard
        # should count as.
        verdict = self.voted_identity(track)
        if verdict is None or not verdict.confirmed or not verdict.known:
            return None
        if verdict.name != track.name:
            return None

        record = self.face_record_for(track, faces)
        if record is None or record.get("embedding") is None:
            return None

        quality = float(record.get("quality") or 0.0)
        ok, why = LIBRARY.should_capture(track.name, record["embedding"],
                                         quality, code=track.emp_code)
        if not ok:
            return None

        # DOES THIS FACE ACTUALLY LOOK LIKE THE PERSON IT IS BEING FILED
        # UNDER? Checked against the ENROLLMENT photographs - the ones a
        # human deliberately took - and not against the live index.
        #
        # Without this the capture loop can amplify its own mistakes. One
        # wrong sample makes the wrong match slightly more likely, which
        # captures another wrong sample, which makes it likelier still.
        # Measured on this site after a few hours of running: 28 of 79
        # stored samples disagreed with their own person's enrollment
        # photos, several at NEGATIVE similarity - a different person
        # entirely - and 13 of Srikanth's 16 were somebody else. That is
        # a feedback loop, not a run of bad luck, and a confirmed vote is
        # not sufficient protection against it because the vote is taken
        # against the very index the loop is polluting.
        #
        # The same guard already protects human confirmations. It should
        # always have protected automatic ones, which are the ones nobody
        # is watching.
        agrees, score = agrees_with_enrollment(track.name, record["embedding"])
        if not agrees:
            if VERBOSE_IDENTITY:
                print(f"[{self.name}] NOT keeping a photo of {track.name}: "
                      f"it scores only {score:.2f} against their enrollment "
                      f"photos, so this is probably not them")
            return None

        path = LIBRARY.save(track.name, frame, record["box"],
                            embedding=record["embedding"],
                            code=track.emp_code, camera=self.key,
                            quality=quality, source=source)
        if not path:
            return None

        # Usable immediately: into this process's search index, and into
        # the database so the OTHER process (and the next restart) has it
        # too. Writing only the file would mean the reference did
        # nothing until somebody re-ran enrollment.
        try:
            learn_face(track.name, record["embedding"], track.emp_code)
            GALLERY.add_person(track.name, track.emp_code).add_face(
                record["embedding"], permanent=True)
            from database import repository as repo
            repo.add_learned_face(track.name, record["embedding"],
                                  emp_code=track.emp_code,
                                  source="camera", camera_key=self.key,
                                  image_path=path)
        except Exception as exc:
            print(f"[{self.name}] captured {track.name} but could not "
                  f"record it: {exc}")
        return path

    # ------------------------------------------------ who is this?
    def resolve_identity(self, track, faces, frame):
        """Ask the gallery who this person is.

        Returns (match, face_embedding, body, face_box). A FACE is the
        only thing that can produce a name; the body measurement rides
        along so the gallery can refuse a name that is landing on
        somebody of the wrong size, and so a confirmed person's size can
        be learned.
        """
        h, w = frame.shape[:2]

        # The face belonging to THIS body. The face stage now tells us
        # which track each face was associated with, so this is an exact
        # answer rather than "whichever face's centre happens to land
        # inside this box" - which was wrong roughly half the time when
        # one person stood behind another. The geometric test is kept
        # only as a fallback for faces the stage could not attribute.
        face_embedding = None
        face_box = None
        mine = [f for f in faces if f.get("track_id") == track.id]
        if not mine:
            mine = [f for f in faces if f.get("track_id") is None
                    and track.contains_point((f["box"][0] + f["box"][2]) // 2,
                                             (f["box"][1] + f["box"][3]) // 2)]
        for f in mine:
            # The BOX is kept even for a face too poor to name from, so
            # an uncertain sighting still saves a picture of the face for
            # the human to look at. The EMBEDDING stays gated on
            # QUALITY, so a blurry face can never put a name on screen by
            # itself.
            face_box = f["box"]
            if f.get("embedding") is not None:
                face_embedding = f["embedding"]
                break

        # What this track has settled at, not this one frame: a single
        # frame can catch somebody mid-stride or half behind a door, and
        # the running median is the number that describes the person.
        body = track.body.typical() or measure(
            track.box, frame.shape, self.tracker.perspective, learn=False)

        # a name already shown on another body in this frame is not
        # available - one person cannot be in two places
        taken = {n for n in self.names.names_in_use()
                 if self.names.holder(n) != track.id}

        match = GALLERY.identify(face=face_embedding, body=body,
                                 exclude=taken)
        return match, face_embedding, body, face_box

    def explain_occasionally(self, track, match):
        """Every 30s, say why one unidentified person is still unnamed.

        Without this, "it is not recognising anybody" is impossible to
        act on - it could be a face nobody has enrolled, a face too
        small to read, or two people too similar to separate.
        """
        if not VERBOSE_IDENTITY:
            return
        now = time.time()
        if now - self._why_t0 < 30:
            return
        self._why_t0 = now
        reason = match.why() if match else "nobody in the gallery scored at all"
        print(f"[WHY:{self.name}] track #{track.id} unnamed - {reason}")

    def learn_from(self, track, face_embedding, body):
        """Record how big a confidently identified person measures.

        Not so the size can name them later - it cannot, and the gallery
        will not let it. It is so that when a face match later tries to
        put this name on a body of a completely different size, there is
        something to check it against.

        Nothing is learned while this person is overlapping somebody
        else: that box is around two people and describes neither.
        """
        if not LEARN_BODY_SIZE or not track.identified or track.occluded:
            return
        GALLERY.learn(track.name,
                      face=None,          # only confirmed faces are stored
                      body=body,
                      camera=self.name)

    def queue_for_review(self, track, match, frame, face_box,
                         face_embedding=None, faces=()):
        """Save an uncertain sighting for a human to confirm.

        Guarded three ways, because a queue that asks the same question
        twenty times is worse than useless:
          * once per track, on a cooldown
          * not at all if we asked about this PERSON recently, even under
            a different track id - a track breaking and restarting used
            to reset the cooldown and start the questions again
          * never for somebody already confirmed today
        """
        now = time.time()
        if now - self._last_review.get(track.id, 0) < REVIEW_COOLDOWN:
            return

        # NEVER ASK A QUESTION THE ANSWER CANNOT TEACH.
        #
        # Measured on this site: of the last 120 questions, 93% carried
        # no face vector and 68% showed only a body crop - yet every one
        # of them asked "is this <name>?", where the name came from a
        # CLOTHING guess. That is the source of every complaint about
        # this queue at once:
        #
        #   * the same person asked over and over, because the clothing
        #     guess kept changing and each new name looked like a new
        #     question;
        #   * a different name every time - one track was asked about as
        #     Divya, Ravi Teja, Sony and Sujana;
        #   * pictures of a hand, a leg or the back of a head, because
        #     with no face detected the card fell back to the body crop;
        #   * and worst, a confirmation that taught nothing, or taught
        #     the wrong person's face vector under the name a human had
        #     just vouched for.
        #
        # So a question now requires a face we could actually learn
        # from. Without one there is nothing to ask and nothing to
        # learn, and the honest thing is to stay quiet.
        record = self.face_record_for(track, faces)
        if record is None or record.get("embedding") is None:
            return
        quality = float(record.get("quality") or 0.0)
        if quality < ASK_UNKNOWN_MIN_QUALITY:
            return
        face_box = record["box"] or face_box
        face_embedding = record.get("embedding")

        # AND SOMETHING HAS TO HAVE LOOKED AT THE PICTURE. Every test
        # above this line is a number, and a crop of a door passes all
        # of them.
        may_ask, screening = self.question_screening(track)
        if not may_ask:
            return

        from database import repository as repo
        # refresh the "already asked / already answered" lists periodically
        if now - self._name_guard_t0 > 30:
            self._name_guard_t0 = now
            try:
                self._asked_recently = repo.recently_reviewed_names(
                    self.key, REVIEW_NAME_COOLDOWN)
                self._confirmed_today = repo.confirmed_names_today()
            except Exception:
                pass

        # WHETHER TO ASK AGAIN ABOUT SOMEBODY ALREADY CONFIRMED.
        #
        # This used to be an outright no: one confirmation muted that
        # person for the rest of the day. It is why the queue went quiet
        # and why the system never accumulated more than one camera
        # photograph of anybody - the goal is twenty, and a rule that
        # stops after the first makes that impossible.
        #
        # Now the library decides. While somebody still needs reference
        # photographs we keep asking, at ASK_AGAIN_SECONDS intervals;
        # once they have enough, the questions stop for good.
        wants_more = LIBRARY.needs_samples(match.name, match.code)
        if not wants_more:
            if match.name in self._confirmed_today:
                return      # already told us, and their library is full
            if match.name in self._asked_recently:
                return      # question already waiting for an answer
        else:
            asked_at = self._asked_at.get(match.name, 0.0)
            if now - asked_at < ASK_AGAIN_SECONDS:
                return

        self._last_review[track.id] = now
        self._asked_recently.add(match.name)
        self._asked_at[match.name] = now

        os.makedirs(self._review_dir, exist_ok=True)
        stamp = f"{int(now)}_{track.id}"
        body_path = os.path.join(self._review_dir, f"{stamp}_body.jpg")
        face_path = ""
        x1, y1, x2, y2 = track.box
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size:
            cv2.imwrite(body_path, crop)
        else:
            return
        face_width = 0
        if face_box:
            fx1, fy1, fx2, fy2 = face_box
            face_width = fx2 - fx1
            # Save the face with MARGIN around it, not a tight box.
            # A tight crop is unusable twice over: a face detector cannot
            # find a face in it (measured: 0 of 25 saved crops), and a
            # human squinting at 48 pixels of cheek cannot tell who it is
            # either. Padding costs nothing and fixes both.
            pad = max(8, int(0.45 * max(fx2 - fx1, fy2 - fy1)))
            fh, fw = frame.shape[:2]
            px1, py1 = max(0, fx1 - pad), max(0, fy1 - pad)
            px2, py2 = min(fw, fx2 + pad), min(fh, fy2 + pad)
            fcrop = frame[py1:py2, px1:px2]
            if fcrop.size:
                face_path = os.path.join(self._review_dir, f"{stamp}_face.jpg")
                cv2.imwrite(face_path, fcrop)

        ev = ", ".join(f"{k}={v:.2f}" for k, v in match.evidence.items()
                       if isinstance(v, float))
        # The suggestion is only shown when a FACE actually supports it.
        # A clothing-only guess is presented as an open question instead
        # - "who is this?" with a name box - because naming a person the
        # system did not really recognise invites a rubber-stamped
        # answer, and a rubber-stamped answer is filed as ground truth.
        # ...and never when the vision model has already refused that
        # name for this body. Putting it in front of a human as "is this
        # Rajesh?" invites the rubber-stamped yes that this whole guard
        # exists to prevent - so the card becomes an open "who is this?"
        # instead, which is what we actually want answered.
        face_backed = (match.evidence.get("face") is not None
                       and match.name not in self._vlm_denied.get(track.id, ()))
        suggested = match.name if face_backed else ""
        repo.add_review(self.key, self.name, track.id, suggested,
                        match.runner_up if face_backed else "",
                        match.probability if face_backed else 0.0,
                        match.margin if face_backed else 0.0, ev,
                        body_path, face_path,
                        face_embedding=face_embedding,
                        face_width=face_width,
                        looks_like=self._screening_note(screening),
                        candidates=self._screening_candidates(
                            screening, face_embedding, match.name))

    @staticmethod
    def _screening_note(screening):
        """What the vision model said this face looks like, in words."""
        if screening is None:
            return ""
        return vlm.describe(getattr(screening, "attributes", None))

    @staticmethod
    def _screening_candidates(screening, embedding, suggested=""):
        """Who this could be: the recogniser's ranking, minus the people
        the picture contradicts. See ranked_candidates() for why it takes
        both - neither half is enough on its own."""
        attrs = getattr(screening, "attributes", None) if screening else None
        names = ranked_candidates(embedding, attrs)
        if suggested and suggested in names:
            names.remove(suggested)
            names.insert(0, suggested)
        return names

    def ask_who_this_is(self, track, faces, frame):
        """Ask about somebody the system cannot name AT ALL.

        The gap this fills: a question was only ever raised when there
        was already a candidate to confirm. So the people the system had
        no idea about - precisely the ones whose reference photographs
        are missing or unusable - were the only ones it never asked
        about. It could not get better at exactly the people it was
        worst at.

        Guarded on time-present and face quality so that somebody
        crossing a corner of the frame does not generate a question, and
        on one question per track.
        """
        if not ASK_ABOUT_UNKNOWN or track.identified:
            return False
        if track.id in self._asked_unknown:
            return False
        if (time.time() - track.created) < ASK_UNKNOWN_MIN_SECONDS:
            return False

        record = self.face_record_for(track, faces)
        if record is None:
            return False
        quality = float(record.get("quality") or 0.0)
        if quality < ASK_UNKNOWN_MIN_QUALITY:
            return False
        # No face vector means nothing could be learned from the answer,
        # so the question would cost somebody's attention and teach
        # nothing.
        if record.get("embedding") is None:
            return False

        # AND SOMETHING HAS TO HAVE LOOKED AT THE PICTURE. This is the
        # path that produced "who is this?" over a picture of a door: it
        # fires precisely when nothing matched, which is exactly what a
        # door does.
        may_ask, screening = self.question_screening(track)
        if not may_ask:
            return False

        now = time.time()
        if now - self._last_review.get(track.id, 0) < REVIEW_COOLDOWN:
            return False
        self._last_review[track.id] = now
        self._asked_unknown[track.id] = now

        from database import repository as repo
        os.makedirs(self._review_dir, exist_ok=True)
        stamp = f"{int(now)}_{track.id}"
        body_path = os.path.join(self._review_dir, f"{stamp}_body.jpg")
        x1, y1, x2, y2 = track.box
        body = frame[max(0, y1):y2, max(0, x1):x2]
        if not body.size:
            return False
        cv2.imwrite(body_path, body)

        face_path = ""
        crop = LIBRARY.crop_face(frame, record["box"])
        if crop is not None:
            face_path = os.path.join(self._review_dir, f"{stamp}_face.jpg")
            cv2.imwrite(face_path, crop)

        width = record["box"][2] - record["box"][0]
        # An EMPTY suggested name is the signal to the dashboard that
        # this is an open question - "who is this?" rather than "is this
        # Rajesh?" - so the reviewer is asked to name them instead of
        # being nudged toward a guess the system did not actually make.
        looks_like = self._screening_note(screening)
        candidates = self._screening_candidates(screening,
                                                record["embedding"])
        repo.add_review(self.key, self.name, track.id, "", "",
                        0.0, 0.0, f"unidentified, face quality {quality:.2f}",
                        body_path, face_path,
                        face_embedding=record["embedding"],
                        face_width=width,
                        looks_like=looks_like, candidates=candidates)
        if VERBOSE:
            extra = f", looks like {looks_like}" if looks_like else ""
            if candidates:
                extra += f", one of: {', '.join(candidates[:4])}"
            print(f"[{self.name}] asking who track #{track.id} is "
                  f"(face quality {quality:.2f}, {width}px{extra})")
        return True

    # ------------------------------------------------- scene snapshots
    def scene_people(self, scale=1.0):
        """Who this camera has in view, as plain data for another process.

        This is the WHO half of "what is happening in the server room".
        The vision model that answers the other half is handed a picture
        and asked what the person in it is doing; it is never told a
        name and never asked for one. Every name in the reply comes from
        here - from the face vote, the tracker, or nothing at all.

        `scale` maps the live box coordinates onto the shrunk snapshot
        the crops will be cut from, so the two cannot drift apart.

        Not-a-person verdicts are honoured, for the same reason presence
        honours them: a reflection in the glass that survived the
        tracker would otherwise be described to somebody as a colleague
        standing at a desk.
        """
        people = []
        for track in self.tracker.all_tracks():
            if track.hits < self.tracker.min_hits:
                continue
            if not self.is_real_person(track):
                continue
            x1, y1, x2, y2 = [int(round(v * scale)) for v in track.box]
            people.append({
                "track_id": track.id,
                # Empty, not "Unknown": the assistant needs to tell "we
                # know this is Pavan" from "somebody is there" without
                # parsing a display label.
                "person": track.name if track.identified else "",
                "box": [x1, y1, x2, y2],
                "seconds": int(getattr(track, "present_seconds", 0) or 0),
                "occluded": bool(getattr(track, "occluded", False)),
                "counted": True,
            })
        return people

    def close(self):
        self.grabber.stop()
        if self.face:
            self.face.stop()

    # ------------------------------------------------------- drawing
    @staticmethod
    def label_box(frame, box, text, colour, filled_bg=True):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        if not text:
            return
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = y1 - 8 if y1 - 8 > th else y2 + th + 8
        tx = max(0, min(x1, frame.shape[1] - tw - 8))
        if filled_bg:
            cv2.rectangle(frame, (tx - 4, ty - th - 6),
                          (tx + tw + 6, ty + 6), DARK, -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 2)

    def draw_header(self, frame, extra=""):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.putText(frame, f"{self.name}", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        if extra:
            (tw, _), _ = cv2.getTextSize(extra, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.55, 2)
            cv2.putText(frame, extra, (w - tw - 10, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 2)
