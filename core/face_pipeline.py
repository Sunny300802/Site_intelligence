"""
core/face_pipeline.py
=====================
The face stage, end to end, on its own thread.

    person tracks (ByteTrack)
          |
          v
    SCRFD face detection      <- on head crops, batched, or full frame
          |
          v
    5-point alignment          core/face_align.py
          |
          v
    quality filtering          core/face_quality.py     <- most faces stop here
          |
          v
    best-frame selection       core/face_identity.py
          |
          v
    AdaFace embedding          core/face_embed.py       <- batched
          |
          v
    vector similarity search   core/facebank.py
          |
          v
    confidence-weighted vote   core/face_identity.py
          |
          v
    final identity, per track id

Three properties this file exists to guarantee
----------------------------------------------
1. IT NEVER BLOCKS THE VIDEO LOOP. The camera drops its newest frame in
   and carries on. Whatever the face stage costs, it costs it on this
   thread; if it cannot keep up it simply skips frames, which loses a
   little latency in naming somebody and nothing else. A pending frame
   is always DISCARDED rather than queued - a face result about a frame
   from four seconds ago is not late, it is wrong.

2. IT DOES NOT RE-RECOGNISE PEOPLE. Work is scheduled per TRACK, not per
   frame: a track that has been confirmed is looked at every
   RECOGNITION_INTERVAL_CONFIRMED frames rather than constantly, and a
   track that has already been recognised recently is skipped entirely.
   The GPU cost of a busy room is therefore roughly the cost of the
   people who are NEW in it.

3. ITS COST PER PASS IS BOUNDED. At most FACE_MAX_RECOGNITIONS_PER_PASS
   faces are embedded in one go, chosen worst-known-first, so a sudden
   crowd delays some namings instead of stalling the pipeline.
"""
import os
import csv
import time
import threading
from collections import Counter

import numpy as np
import cv2

from config.settings import (FACE_DET_SIZE, FACE_INPUT_MAX_WIDTH,
                             FACE_DETECT_ON_PERSON_CROPS,
                             FACE_CROP_HEAD_FRACTION, FACE_CROP_PAD,
                             FACE_CROP_DET_SIZE, FACE_CROP_BATCH,
                             FACE_QUALITY_THRESHOLD, MIN_FACE_SIZE,
                             FACE_MAX_RECOGNITIONS_PER_PASS,
                             FACE_RECOGNITION_THRESHOLD, FACE_MATCH_MARGIN,
                             FACE_DEBUG, FACE_DEBUG_EVERY_SECONDS,
                             FACE_TRACE, FACE_TRACE_FILE, FACE_EMBED_BACKEND)
from core.face_detect import get_detector
from core.face_embed import get_embedder
from core.face_align import align, align_from_box
from core.face_quality import assess
from core.face_identity import IdentityBook, UNKNOWN
from core.facebank import FACE_BANK


class FaceStageStats:
    """What the face stage did, so the console can explain itself.

    Kept as a class rather than a pile of counters because the periodic
    summary is the primary diagnostic tool for this whole subsystem -
    "nobody is being recognised" is answerable from it in one line, and
    unanswerable without it.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.passes = 0
        self.faces = 0
        self.rejected = 0
        self.reasons = Counter()
        self.recognitions = 0
        self.matched = 0
        self.unmatched = 0
        self.ambiguous = 0
        self.best_unmatched = 0.0
        self.last_ambiguous = None
        self.quality_bins = Counter()
        self.widths = []
        self.confirmed = 0

    def note_quality(self, score):
        self.quality_bins[min(9, int(max(0.0, score) * 10))] += 1

    def note_width(self, width):
        # Bounded, so a busy hour cannot grow this without limit.
        if width and len(self.widths) < 4000:
            self.widths.append(int(width))

    def width_summary(self, minimum=None):
        """How big the faces actually are, against the bar they must clear.

        This one line answers the most common question this subsystem
        produces - "why is everything rejected as too small?" - which is
        otherwise only answerable by writing a separate measuring script.
        """
        if not self.widths:
            return ""
        minimum = MIN_FACE_SIZE if minimum is None else int(minimum)
        values = sorted(self.widths)
        middle = values[len(values) // 2]
        clearing = 100.0 * sum(1 for v in values if v >= minimum) / len(values)
        return (f"face width px: min {values[0]} median {middle} "
                f"max {values[-1]} | {clearing:.0f}% reach the "
                f"{minimum}px minimum")

    def histogram(self):
        if not self.quality_bins:
            return ""
        total = sum(self.quality_bins.values())
        parts = []
        for bucket in range(10):
            count = self.quality_bins.get(bucket, 0)
            if count:
                parts.append(f"{bucket / 10:.1f}:{100 * count / total:.0f}%")
        return " ".join(parts)


class FacePipeline:
    """One camera's face stage. Owns a thread; shares the models."""

    def __init__(self, name="face", identity_forget=120.0,
                 min_face_size=None):
        self.name = name
        # Per CAMERA, because face size is a property of the camera, not
        # of the system. A 720p reception feed and a 1440p work-area feed
        # cannot share one pixel bar: measured here, reception faces are
        # 26-41px and the work area's are 32-78px, so a single number is
        # either far too strict for one or far too loose for the other.
        self.min_face_size = int(min_face_size or MIN_FACE_SIZE)
        self.ready = False
        self.detector = get_detector()
        self.embedder = get_embedder()
        self.bank = FACE_BANK

        # The similarity bar belongs to the model that LOADED, not the
        # one that was configured. See thresholds_for_loaded_model().
        from core.face_embed import thresholds_for_loaded_model
        (self.threshold, self.learn_agree_min,
         adjusted) = thresholds_for_loaded_model()
        # Everything below the recognition bar by this much is recorded
        # as "nobody matched" rather than as a weak vote for the closest
        # person, so it has to follow the bar.
        self.vote_unknown_below = max(0.0, self.threshold - 0.04)
        if adjusted:
            print(f"[FACE:{name}] *** the configured backend is "
                  f"'{FACE_EMBED_BACKEND}' but '{self.embedder.name}' is "
                  f"what loaded, so the recognition threshold has been "
                  f"set to {self.threshold} - the value that belongs to "
                  f"the model actually running. Applying one model's "
                  f"threshold to another is how confident WRONG names "
                  f"appear. Set FACE_RECOGNITION_THRESHOLD explicitly to "
                  f"override. ***")
        self.identities = IdentityBook(forget_after=identity_forget)
        self.stats = FaceStageStats()

        self._lock = threading.Lock()
        self._pending = None
        self._latest = []
        self._running = True
        self._busy = False
        self._frame_index = 0
        self.last_ms = 0.0
        self._report_t0 = time.time()
        self._trace_file = None
        self._trace_writer = None

        if self.detector is None:
            print(f"[FACE:{self.name}] no face detector - everybody stays "
                  f"Unknown")
        elif self.embedder is None:
            print(f"[FACE:{self.name}] no recognition model - faces will be "
                  f"found and drawn, but never named")
        else:
            self.ready = True
            if not len(self.bank):
                print(f"[FACE:{self.name}] no enrolled faces - everyone will "
                      f"be 'Unknown'. Run: python tools/enroll_faces.py")

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---------------------------------------------------------- inbox
    def busy(self):
        return self._busy

    def submit(self, frame, tracks=None, frame_index=None):
        """Hand the worker the newest frame and the tracks in it.

        `tracks` is a snapshot - [(track_id, (x1, y1, x2, y2)), ...] -
        taken on the camera thread immediately after tracking, so the
        face stage associates faces with the ids that were live in THAT
        frame. Passing the live Track objects instead would mean reading
        boxes that the camera thread is concurrently rewriting, and a
        face would occasionally be filed under the wrong person.
        """
        if not self.ready or frame is None:
            return False
        if self._busy:
            return False               # still working: drop, never queue
        with self._lock:
            self._frame_index += 1
            self._pending = (frame, list(tracks or []),
                             self._frame_index if frame_index is None
                             else int(frame_index))
        return True

    def get(self):
        """The latest per-face results, in the shape the cameras expect."""
        with self._lock:
            return list(self._latest)

    def verdict(self, track_id):
        """The voted identity for one track, or None."""
        return self.identities.verdict(track_id)

    def force_identity(self, track_id, name, code="", score=1.0):
        """Pin an identity from outside - a human confirmation."""
        return self.identities.force(track_id, name, code, score)

    def forget(self, track_id):
        self.identities.drop(track_id)

    def best_face(self, track_id):
        """The best aligned face this track has produced, or None.

        Returned as (crop, quality, age_seconds). The crop is the very
        image recognition was run on, which is what makes it the right
        thing to show an independent checker: if the answer came off
        this picture, this picture is what should be argued about.
        """
        state = self.identities.get(track_id, create=False)
        if state is None or state.best_crop is None:
            return None
        return (state.best_crop, state.best_quality,
                time.time() - state.best_at)

    # ------------------------------------------------------ detection
    def _head_region(self, box, width, height):
        """Where to look for this person's face.

        Only the top of the body box, widened a little. Cropping is what
        turns a 22-pixel face (after the detector shrinks a full frame to
        640) back into a 180-pixel one, and that difference is the whole
        reason distant people can be recognised at all.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        pad_x = int(w * FACE_CROP_PAD)
        pad_y = int(h * FACE_CROP_PAD * 0.5)
        hx1 = max(0, x1 - pad_x)
        hy1 = max(0, y1 - pad_y)
        hx2 = min(width, x2 + pad_x)
        hy2 = min(height, y1 + int(h * FACE_CROP_HEAD_FRACTION) + pad_y)
        if hx2 - hx1 < 16 or hy2 - hy1 < 16:
            return None
        return hx1, hy1, hx2, hy2

    def _detect_on_crops(self, frame, tracks):
        """Batched SCRFD over each track's head region.

        Returns [(track_id, face_dict)] with boxes and landmarks already
        mapped back into full-frame coordinates.
        """
        height, width = frame.shape[:2]
        regions, owners = [], []
        for track_id, box in tracks:
            region = self._head_region(box, width, height)
            if region is None:
                continue
            hx1, hy1, hx2, hy2 = region
            crop = frame[hy1:hy2, hx1:hx2]
            if crop.size == 0:
                continue
            regions.append(crop)
            owners.append((track_id, hx1, hy1))

        found = []
        for start in range(0, len(regions), max(1, FACE_CROP_BATCH)):
            chunk = regions[start:start + FACE_CROP_BATCH]
            chunk_owners = owners[start:start + FACE_CROP_BATCH]
            results = self.detector.detect_many(
                chunk, det_size=FACE_CROP_DET_SIZE, max_faces=1)
            for faces, (track_id, ox, oy) in zip(results, chunk_owners):
                if not faces:
                    continue
                face = faces[0]
                x1, y1, x2, y2 = face["box"]
                face = {"box": (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                        "score": face["score"],
                        "landmarks": (None if face["landmarks"] is None
                                      else face["landmarks"] +
                                      np.array([ox, oy], dtype=np.float32))}
                found.append((track_id, face))
        return found

    def _detect_full_frame(self, frame, tracks):
        """One SCRFD pass over the whole frame, then associate to tracks."""
        scale = 1.0
        work = frame
        if FACE_INPUT_MAX_WIDTH and frame.shape[1] > FACE_INPUT_MAX_WIDTH:
            scale = FACE_INPUT_MAX_WIDTH / frame.shape[1]
            work = cv2.resize(frame, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        inverse = 1.0 / max(scale, 1e-9)

        faces = self.detector.detect(work, det_size=FACE_DET_SIZE)
        out = []
        for face in faces:
            x1, y1, x2, y2 = [v * inverse for v in face["box"]]
            landmarks = (None if face["landmarks"] is None
                         else face["landmarks"] * inverse)
            mapped = {"box": (int(round(x1)), int(round(y1)),
                              int(round(x2)), int(round(y2))),
                      "score": face["score"], "landmarks": landmarks}
            out.append((self._owner_of(mapped["box"], tracks), mapped))
        return out

    @staticmethod
    def _owner_of(face_box, tracks):
        """Which tracked person does this face belong to?

        Containment of the face centre, then - among the tracks that
        contain it - the one where the face sits where a HEAD should sit.
        Without that second test a face is handed to whichever body box
        happens to be listed first, and in a room where somebody stands
        behind somebody else that is the wrong one about half the time,
        which puts one person's face on another person's track.
        """
        fx = (face_box[0] + face_box[2]) * 0.5
        fy = (face_box[1] + face_box[3]) * 0.5
        best, best_error = None, None
        for track_id, box in tracks:
            x1, y1, x2, y2 = box
            if not (x1 <= fx <= x2 and y1 <= fy <= y2):
                continue
            height = max(1.0, float(y2 - y1))
            error = abs((fy - y1) / height - 0.12)
            if best_error is None or error < best_error:
                best, best_error = track_id, error
        return best

    @staticmethod
    def _confirm_owners(detections, tracks):
        """Check that each face really belongs to the track it came from.

        Faces are looked for inside each tracked person's HEAD REGION,
        which is a fast way to find small faces - but when one person
        stands behind another, the region cropped for the person in
        front contains the face of the person behind, and SCRFD dutifully
        returns it. The face is then attributed to the wrong body, and
        one person is named as another. Everything downstream trusts
        that attribution, so this is where a swap has to be caught.

        Each detected face is re-offered to every track, and kept only
        if the region it was found in is still the best owner - the box
        containing it, with the face where a head should be. A face two
        tracks disagree about is DROPPED rather than guessed at: a
        missed identification costs one frame, a wrong one costs a name.
        """
        out = []
        for track_id, face in detections:
            owner = FacePipeline._owner_of(face["box"], tracks)
            if owner is None or owner == track_id:
                # Nobody else claims it, or the same track does.
                out.append((track_id, face))
        return out

    # -------------------------------------------------------- the pass
    def _process(self, frame, tracks, frame_index):
        """One complete face pass over one frame."""
        self.stats.passes += 1
        track_boxes = {tid: box for tid, box in tracks}

        if tracks and FACE_DETECT_ON_PERSON_CROPS:
            detections = self._detect_on_crops(frame, tracks)
            detections = self._confirm_owners(detections, tracks)
        else:
            detections = self._detect_full_frame(frame, tracks)

        results, candidates = [], []
        for track_id, face in detections:
            self.stats.faces += 1
            box = face["box"]
            landmarks = face["landmarks"]

            crop = (align(frame, landmarks) if landmarks is not None
                    else align_from_box(frame, box))
            quality = assess(crop, box, landmarks, face["score"],
                             min_size=self.min_face_size)
            self.stats.note_quality(quality.score)
            self.stats.note_width(quality.width)

            record = {"box": box, "score": 0.0, "name": UNKNOWN, "code": "",
                      "embedding": None, "track_id": track_id,
                      "landmarks": landmarks, "quality": quality.score,
                      "quality_reason": quality.reason,
                      "det_score": face["score"], "width": quality.width,
                      # kept for the existing drawing code, which greys
                      # out faces it was told not to trust
                      "too_small": not quality.ok,
                      "confirmed": False}
            results.append(record)

            if not quality.ok:
                self.stats.rejected += 1
                for reason in quality.reasons:
                    self.stats.reasons[reason.split(" (")[0]] += 1
                if track_id is not None:
                    state = self.identities.get(track_id)
                    state.touch()
                    state.faces_rejected += 1
                continue

            if track_id is None:
                # A face with no track behind it (the camera passed no
                # tracks, or nobody's body box contains it). Still worth
                # recognising, but there is nothing to vote on, so it is
                # answered per-frame and never confirmed.
                candidates.append((None, record, crop, quality, False))
                continue

            state = self.identities.get(track_id)
            is_new_best = state.offer(quality.score, crop, box, landmarks,
                                      face["score"])
            wanted, why = state.should_recognise(frame_index, quality.score,
                                                 is_new_best)
            record["schedule"] = why
            if wanted:
                # Recognise the BEST frame this track has, which may be
                # an earlier one than this - that is the entire point.
                #
                # ...unless that best frame has gone stale, in which case
                # the face in FRONT OF US wins. A held best frame that is
                # never allowed to expire is a cached picture of whoever
                # the track used to be following, and re-recognising it
                # would keep confirming that person no matter who is
                # actually standing there now.
                use_crop = crop
                if state.best_crop is not None and not state.best_is_stale():
                    use_crop = state.best_crop
                candidates.append((track_id, record, use_crop, quality, True))

        self._recognise(candidates, frame_index)
        self._apply_verdicts(results, track_boxes)
        self.identities.prune(alive_ids=set(track_boxes))
        return results

    def _recognise(self, candidates, frame_index):
        """Embed and search, in one batch, bounded in size."""
        if not candidates or self.embedder is None:
            return

        # Worst-known first: a track nobody has named yet is worth far
        # more than another look at somebody already confirmed.
        def priority(item):
            track_id = item[0]
            state = (self.identities.get(track_id, create=False)
                     if track_id is not None else None)
            already = 1 if (state is not None and state.confirmed_name) else 0
            return (already, -item[3].score)

        candidates = sorted(candidates, key=priority)
        if FACE_MAX_RECOGNITIONS_PER_PASS > 0:
            dropped = len(candidates) - FACE_MAX_RECOGNITIONS_PER_PASS
            if dropped > 0:
                candidates = candidates[:FACE_MAX_RECOGNITIONS_PER_PASS]
                # Say so rather than silently covering fewer people: a
                # cap that is being hit constantly is a real finding.
                self.stats.reasons[f"deferred (over {FACE_MAX_RECOGNITIONS_PER_PASS}/pass)"] += dropped

        embeddings = self.embedder.embed([item[2] for item in candidates])
        now = time.time()

        for (track_id, record, _crop, quality, votes), embedding in zip(
                candidates, embeddings):
            if embedding is None or float(np.abs(embedding).sum()) < 1e-6:
                continue
            self.stats.recognitions += 1
            record["embedding"] = embedding

            # A name confirmed on ANOTHER track right now is not
            # available: one person cannot be in two places at once. The
            # camera enforces this too (core/identity.py), but doing it
            # here as well means the losing track never even votes for
            # the taken name - so it goes on collecting evidence for who
            # it really is, instead of building a confident case for
            # somebody who is standing elsewhere in the same frame.
            taken = self.identities.confirmed_names() - (
                {self.identities.get(track_id).confirmed_name}
                if track_id is not None else set())
            match = self.bank.search(embedding, exclude=taken,
                                     threshold=self.threshold)
            record["similarity"] = match.score
            record["runner_up"] = match.runner_up
            record["match_reason"] = match.reason

            if match.accepted:
                self.stats.matched += 1
            else:
                self.stats.unmatched += 1
                self.stats.best_unmatched = max(self.stats.best_unmatched,
                                                match.score)
                if match.runner_up and match.score >= self.threshold:
                    self.stats.ambiguous += 1
                    self.stats.last_ambiguous = (
                        match.reason, match.score, match.runner_up,
                        match.runner_score)

            if track_id is None or not votes:
                # No track: answer this face on its own, honestly, with
                # no confirmation possible.
                record["name"] = match.name if match.accepted else UNKNOWN
                record["code"] = match.code
                record["score"] = match.score
                continue

            state = self.identities.get(track_id)
            state.mark_recognised(frame_index, embedding)
            verdict = state.record(match.name if match.accepted else UNKNOWN,
                                   match.score, quality.score, match.code, now,
                                   unknown_below=self.vote_unknown_below)
            self._trace(track_id, quality, match, verdict)

    def _apply_verdicts(self, results, track_boxes):
        """Stamp each face record with its track's voted identity."""
        for record in results:
            track_id = record.get("track_id")
            if track_id is None:
                continue
            state = self.identities.get(track_id, create=False)
            if state is None:
                continue

            # Carry the track's most recent embedding onto every face
            # record, not just the ones a recognition ran on this pass.
            #
            # Recognition is now deliberately intermittent, but the
            # GALLERY (core/gallery.py) is asked who this person is on
            # EVERY processed frame and weighs the face against size,
            # height and recency. Handing it a face only on recognition
            # passes would quietly remove the face cue from most of its
            # decisions - and it is also the vector stored with a review
            # question, so a confirmation would have nothing to learn
            # from.
            if record.get("embedding") is None and \
                    state.best_embedding is not None:
                record["embedding"] = state.best_embedding

            verdict = state.last_verdict
            if verdict is None:
                continue
            if verdict.known:
                record["name"] = verdict.name
                record["code"] = verdict.code
                record["score"] = verdict.score
                record["confirmed"] = verdict.confirmed
                if verdict.confirmed:
                    self.stats.confirmed += 1
            record["verdict_reason"] = verdict.reason

    # ------------------------------------------------------ the thread
    def _loop(self):
        while self._running:
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                time.sleep(0.005)
                continue

            frame, tracks, frame_index = job
            self._busy = True
            started = time.time()
            try:
                results = self._process(frame, tracks, frame_index)
                with self._lock:
                    self._latest = results
                self._maybe_report()
            except Exception as exc:
                print(f"[FACE:{self.name}] {exc}")
                if FACE_DEBUG:
                    import traceback
                    traceback.print_exc()
                time.sleep(0.05)
            finally:
                self.last_ms = (time.time() - started) * 1000.0
                self._busy = False

    # ----------------------------------------------------- diagnostics
    def _trace(self, track_id, quality, match, verdict):
        """One CSV row per recognition, for offline comparison.

        This is what makes "is AdaFace actually better here?" a question
        with an answer: run the old pipeline and the new one over the
        same recording, and compare the two files with
        tools/eval_recognition.py rather than comparing impressions.
        """
        if not FACE_TRACE:
            return
        try:
            if self._trace_writer is None:
                new_file = not os.path.exists(FACE_TRACE_FILE)
                os.makedirs(os.path.dirname(FACE_TRACE_FILE), exist_ok=True)
                self._trace_file = open(FACE_TRACE_FILE, "a", newline="",
                                        encoding="utf-8")
                self._trace_writer = csv.writer(self._trace_file)
                if new_file:
                    self._trace_writer.writerow(
                        ["time", "camera", "backend", "track", "quality",
                         "width", "sharpness", "brightness", "contrast",
                         "yaw", "roll", "occlusion", "det", "match",
                         "similarity", "runner_up", "runner_score",
                         "accepted", "verdict", "confirmed", "votes",
                         "share"])
            self._trace_writer.writerow(
                [f"{time.time():.3f}", self.name, FACE_EMBED_BACKEND,
                 track_id, f"{quality.score:.4f}", quality.width,
                 f"{quality.sharpness:.1f}", f"{quality.brightness:.1f}",
                 f"{quality.contrast:.1f}", f"{quality.yaw:.3f}",
                 f"{quality.roll:.1f}", f"{quality.occlusion:.3f}",
                 f"{quality.det_score:.3f}", match.name,
                 f"{match.score:.4f}", match.runner_up,
                 f"{match.runner_score:.4f}", int(match.accepted),
                 verdict.name, int(verdict.confirmed), verdict.votes,
                 f"{verdict.share:.3f}"])
            self._trace_file.flush()
        except Exception as exc:
            print(f"[FACE:{self.name}] trace disabled ({exc})")
            self._trace_writer = None

    def _maybe_report(self):
        """Periodically explain WHY faces are or are not being named."""
        if not FACE_DEBUG:
            return
        now = time.time()
        if now - self._report_t0 < FACE_DEBUG_EVERY_SECONDS:
            return
        self._report_t0 = now
        stats = self.stats
        if stats.faces == 0:
            return

        print(f"[FACE:{self.name}] {stats.faces} face(s) in {stats.passes} "
              f"pass(es) | {stats.rejected} rejected on quality | "
              f"{stats.recognitions} recognised | {stats.matched} matched, "
              f"{stats.unmatched} no match "
              f"(best {stats.best_unmatched:.2f} vs "
              f"{self.threshold:.2f})")
        widths = stats.width_summary(self.min_face_size)
        if widths:
            print(f"[FACE:{self.name}]   {widths}")
        histogram = stats.histogram()
        if histogram:
            print(f"[FACE:{self.name}]   quality: {histogram}")
        if stats.reasons:
            top = ", ".join(f"{reason} x{count}" for reason, count
                            in stats.reasons.most_common(4))
            print(f"[FACE:{self.name}]   rejected for: {top}")
        if stats.last_ambiguous:
            reason, score, rival, rival_score = stats.last_ambiguous
            print(f"[FACE:{self.name}]   too close to call: {reason}. "
                  f"If this pair keeps appearing, THOSE TWO need better "
                  f"enrollment photos - that is the real fix.")
        if stats.recognitions and stats.matched == 0:
            print(f"[FACE:{self.name}]   -> faces ARE passing quality but "
                  f"nobody matches. If the best score "
                  f"({stats.best_unmatched:.2f}) is close to "
                  f"{self.threshold:.2f}, lower "
                  f"FACE_RECOGNITION_THRESHOLD; if it is far below, the "
                  f"enrollment was probably made with a different model - "
                  f"re-run tools/enroll_faces.py.")
        elif stats.faces and stats.recognitions == 0:
            print(f"[FACE:{self.name}]   -> every face was rejected before "
                  f"recognition. Lower FACE_QUALITY_THRESHOLD "
                  f"({FACE_QUALITY_THRESHOLD}) or MIN_FACE_SIZE "
                  f"({self.min_face_size}px), or the camera is too far "
                  f"away. The face-width line above says which.")
        stats.reset()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        if self._trace_file is not None:
            try:
                self._trace_file.close()
            except Exception:
                pass
