"""
core/pipeline.py
================
The engine loop.

Each cycle:
    1. ask every camera for its newest frame
    2. run ONE batched person-detection pass on the GPU for all of them
    3. hand each camera its own detections to process
    4. publish the annotated frames to the MJPEG server

Because step 2 is a single batched call, adding cameras increases GPU
utilisation efficiently rather than multiplying per-call overhead.
"""
import time
import signal
import cv2

from collections import defaultdict
from datetime import datetime

from config.settings import (STREAM_ENABLED, STREAM_PORT, JPEG_QUALITY,
                             SHOW_LOCAL_WINDOWS, STREAM_MAX_WIDTH,
                             PROFILE, PROFILE_EVERY_SECONDS,
                             ADAPTIVE_DETECTION, ADAPTIVE_TARGET_FPS,
                             ADAPTIVE_GOOD_FPS, ADAPTIVE_PATIENCE,
                             LEARN_POLL_SECONDS, CONFIRM_APPLY_WINDOW,
                             CONFIRMED_SCORE, SCENE_SNAPSHOT_ENABLED,
                             SCENE_SNAPSHOT_INTERVAL,
                             SCENE_SNAPSHOT_MAX_WIDTH,
                             SCENE_SNAPSHOT_QUALITY)
from config.cameras import enabled_cameras
from cameras.registry import build_camera
from core.detector import PersonDetector
from core.ultra import quiet as quiet_ultralytics
from core.gallery import GALLERY
from core.face import load_encodings, init_face_bank, learn_face
from core.facebank import FACE_BANK
from streaming.mjpeg import STORE, start_stream_server
from database.db import init_db
from database import repository as repo


class Pipeline:
    def __init__(self):
        self.cameras = []
        self.detector = None
        self.running = False
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._fps = 0.0
        # per-stage timing, so a slow pipeline can be diagnosed instead
        # of guessed at
        self._stage = {"grab": 0.0, "detect": 0.0, "process": 0.0,
                       "encode": 0.0}
        self._stage_n = 0
        self._profile_t0 = time.time()
        self._cycle = 0
        self._slow_since = None
        self._fast_since = None
        self._health_t0 = time.time()
        self._learned_t0 = time.time()
        self._scene_t0 = {}        # camera key -> when we last snapshotted
        self._learned_last_id = 0
        # review items whose confirmation we have already applied to a
        # live track. Review ids are NOT monotonic in the order people
        # answer them - somebody can confirm an old card long after a
        # newer one - so a "last id seen" cursor would silently skip
        # answers. We remember what we applied instead.
        self._applied_reviews = set()
        # when this run began. Track ids only mean anything within one
        # run, so an answer to a question asked before this moment cannot
        # safely be matched to a track by id.
        self._started_at = datetime.utcnow()

    @staticmethod
    def _compatible_faces(rows):
        """Split confirmed face rows into (usable, rejected_count).

        A face confirmed under one recognition model says nothing under
        another. These vectors are the right SHAPE either way, so
        nothing would error - they would simply sit at meaningless
        positions in the new model's space and be free to match the
        wrong person. That is the quietest possible way to produce wrong
        names, so it is checked rather than assumed.
        """
        from core.face_embed import tags_compatible, model_tag
        current = model_tag()
        usable, rejected = [], 0
        for row in rows:
            if tags_compatible(row.get("model", ""), current):
                usable.append(row)
            else:
                rejected += 1
        return usable, rejected

    @staticmethod
    def _print_face_settings():
        """Say what the face stage is actually configured to do.

        Every one of these can be overridden from the environment, so
        "which thresholds was this run using?" must be answerable from
        the log rather than from whatever config/settings.py says today.
        That is also what makes an A/B run against the old pipeline
        meaningful afterwards.
        """
        from config import settings as s
        print("[FACE] configuration -----------------------------------")
        print(f"[FACE]   detector      SCRFD @ {s.FACE_DET_SIZE}px, "
              f"conf>={s.FACE_DETECTION_THRESHOLD}, "
              f"crops={'on' if s.FACE_DETECT_ON_PERSON_CROPS else 'off'}"
              f"@{s.FACE_CROP_DET_SIZE}px x{s.FACE_CROP_BATCH}")
        # The EFFECTIVE threshold, resolved from the model that actually
        # loaded. Printing the configured one would be worse than
        # printing nothing: it is exactly the number somebody would
        # later use to explain a wrong name.
        try:
            from core.face_embed import (loaded_backend,
                                         thresholds_for_loaded_model)
            backend = loaded_backend()
            threshold, agree, adjusted = thresholds_for_loaded_model()
            note = (f"  (configured {s.FACE_EMBED_BACKEND}, "
                    f"threshold corrected from "
                    f"{s.FACE_RECOGNITION_THRESHOLD})" if adjusted else "")
        except Exception:
            backend, threshold, note = s.FACE_EMBED_BACKEND, \
                s.FACE_RECOGNITION_THRESHOLD, ""
        print(f"[FACE]   recognition   {backend}, "
              f"threshold {threshold}, "
              f"margin {s.FACE_MATCH_MARGIN}, "
              f"bank scoring '{s.FACE_BANK_SCORING}'{note}")
        print(f"[FACE]   quality       >= {s.FACE_QUALITY_THRESHOLD}, "
              f"min face {s.MIN_FACE_SIZE}px, "
              f"best-frame >= {s.BEST_FRAME_QUALITY_THRESHOLD}")
        print(f"[FACE]   scheduling    every {s.FACE_EVERY_N_FRAMES} frame(s), "
              f"recognise 1/{s.RECOGNITION_INTERVAL} per track "
              f"(1/{s.RECOGNITION_INTERVAL_CONFIRMED} once confirmed), "
              f"max {s.FACE_MAX_RECOGNITIONS_PER_PASS}/pass")
        print(f"[FACE]   voting        window {s.VOTING_WINDOW}, "
              f"min votes {s.MIN_VOTES}, share {s.VOTE_MIN_SHARE}, "
              f"override x{s.VOTE_OVERRIDE_FACTOR}")
        print(f"[FACE]   tracking      {s.TRACKER_BACKEND}, "
              f"buffer {s.TRACK_BUFFER} frames")
        print("[FACE] -------------------------------------------------")

    def setup(self):
        quiet_ultralytics()
        init_db()
        self._print_face_settings()
        cleared = repo.close_all_active()
        if cleared:
            print(f"[DB] closed {cleared} stale visit(s) from a previous run")
        cleared = repo.close_all_presence()
        if cleared:
            print(f"[DB] closed {cleared} stale presence session(s)")

        configs = enabled_cameras()
        if not configs:
            raise SystemExit("No enabled cameras in config/cameras.py")

        # Seed the shared gallery from the enrolled photos. From here on
        # it keeps learning: every confident identification records how
        # big that person measures, which is what lets a later face
        # match be refused when it lands on a body of the wrong size.
        # anything already confirmed by a human, from previous sessions
        try:
            previously = repo.learned_faces_since(0)
        except Exception:
            previously = []

        embeddings, names, codes = load_encodings()
        if embeddings is not None and len(embeddings):
            n = GALLERY.load_enrollment(embeddings, names, codes)
            # The same references also seed the face bank, which is what
            # the face stage searches directly. Two indexes, one source:
            # the gallery weighs a face against the person's measured
            # size and how recently they were seen, while the bank
            # answers the face-only question the vote needs. Filling
            # only one of them is how a person could be recognised by
            # one path and unknown to the other.
            init_face_bank(embeddings, names, codes)
            print(f"[GALLERY] {n} people loaded from enrollment "
                  f"({len(embeddings)} face samples)")
        else:
            init_face_bank(None, [], [])
            print("[GALLERY] no enrollment - run tools/enroll_faces.py")

        # Nothing to restore for clothing any more - it is not used, and
        # body measurements are re-learned from the cameras within a
        # minute of starting, so there is nothing worth persisting.

        if previously:
            usable, wrong_model = self._compatible_faces(previously)
            for row in usable:
                prof = GALLERY.add_person(row["name"], row.get("code", ""))
                prof.add_face(row["embedding"], permanent=True)
                learn_face(row["name"], row["embedding"], row.get("code", ""))
            # The cursor advances past the rejected rows too. They will
            # never become usable - they were made by a model that is no
            # longer loaded - and re-reading them every poll would just
            # reprint the same warning for ever.
            for row in previously:
                self._learned_last_id = max(self._learned_last_id, row["id"])
            if usable:
                who = sorted({r["name"] for r in usable})
                print(f"[GALLERY] plus {len(usable)} confirmed sample(s) "
                      f"from earlier sessions: {', '.join(who[:6])}"
                      f"{' ...' if len(who) > 6 else ''}")
            if wrong_model:
                print(f"[GALLERY] ignored {wrong_model} confirmed sample(s) "
                      f"made by a different recognition model - they are "
                      f"not comparable with the one loaded now. They will "
                      f"be re-learned the next time those people are "
                      f"confirmed.")

        self.detector = PersonDetector()

        for cfg in configs:
            print(f"[PIPELINE] starting camera '{cfg['name']}'")
            cam = build_camera(cfg)
            self.cameras.append(cam)
            STORE.register(cam.key, cam.name)

        if STREAM_ENABLED:
            start_stream_server(STREAM_PORT)

        self._announce_vlm()

        repo.log_system("START", f"{len(self.cameras)} camera(s)")
        pending = repo.review_counts().get("pending", 0)
        if pending:
            print(f"[REVIEW] {pending} uncertain sighting(s) waiting for "
                  f"confirmation on the dashboard")
        print(f"[PIPELINE] ready with {len(self.cameras)} camera(s). "
              f"Ctrl+C to stop.")

    @staticmethod
    def _announce_vlm():
        """Say whether the independent check is actually running.

        It matters that this is visible at startup. Everything the
        arbiter does is a REFUSAL, so when it is switched off - or when
        Ollama simply is not running - the system looks and behaves
        exactly as it did before. That is the correct failure mode and
        the worst possible thing to have to guess about.
        """
        from config import settings as s
        if not s.VLM_ENABLED:
            print("[VLM] off (VLM_ENABLED=0) - names and boxes are not "
                  "independently checked")
            return
        from core.vlm_verify import ARBITER
        checks = []
        if s.VLM_VERIFY_TRACKS:
            checks.append("is this a person")
        if s.VLM_VERIFY_IDENTITY:
            checks.append("is this really them")
        if not checks:
            print("[VLM] enabled but no checks are turned on")
            return
        asking = " and ".join(f"'{c}'" for c in checks)
        print(f"[VLM] second opinion: {s.VLM_MODEL} answering {asking}")
        print("[VLM]   veto only - it can refuse a name, never grant one")
        print(f"[VLM]   at most one question every {s.VLM_MIN_CALL_GAP}s, "
              f"asked once per track; confidence needed to refuse: "
              f"person {s.VLM_PERSON_MIN_CONFIDENCE}, "
              f"name {s.VLM_IDENTITY_MIN_CONFIDENCE}")
        ARBITER.usable()          # prints whether Ollama answered

    def run(self):
        self.running = True
        signal.signal(signal.SIGINT, self._stop_signal)
        try:
            while self.running:
                self._tick()
        finally:
            self.shutdown()

    def _tick(self):
        self._cycle += 1
        t0 = time.time()
        ready = [c for c in self.cameras if c.grab()]
        t_grab = time.time() - t0
        if not ready:
            time.sleep(0.005)
            return

        # Face frames are NOT submitted here any more. The face stage
        # attributes every face to a tracking id, so it has to be given
        # the frame together with the tracks that were found in it -
        # which only exists after the camera has tracked. Each handler
        # therefore calls submit_faces() itself, immediately after
        # tracking. See cameras/base.py.

        # Only the cameras that are due this cycle are detected. The rest
        # keep streaming their last annotated frame, so the video stays
        # live while the GPU work is spread out.
        due = [c for c in ready if c.due_for_detection(self._cycle)]

        # Cameras with identical settings are batched into one GPU call;
        # different settings mean separate calls, which is the price of
        # letting each camera be tuned for its own scene.
        groups = defaultdict(list)
        for cam in due:
            groups[cam.detect_key()].append(cam)

        t1 = time.time()
        detections = {}
        for (imgsz, conf, _enh, min_height), cams in groups.items():
            frames = [c.frame_for_detection() for c in cams]
            frames = [f for f in frames if f is not None]
            if not frames:
                continue
            results = self.detector.detect_people(
                frames, imgsz=imgsz, conf=conf, min_height_frac=min_height)
            for cam, dets in zip(cams, results):
                detections[cam.key] = dets
        t_detect = time.time() - t1

        self._tick_fps()
        t2 = time.time()
        for cam in due:
            cam.fps = self._fps
            dets = detections.get(cam.key)
            if dets is not None:
                cam.process(cam.map_detections(dets))
        t_process = time.time() - t2

        t3 = time.time()
        for cam in ready:
            self._publish(cam)
        t_encode = time.time() - t3

        self._record_stages(t_grab, t_detect, t_process, t_encode)
        self._adapt()
        self._preview(ready)
        self._check_health()
        self._pull_confirmations()

    def _pull_confirmations(self):
        """Pick up faces a human confirmed on the dashboard.

        The dashboard runs as a separate process, so confirmations reach
        us through the database rather than directly. Polling every few
        seconds means a confirmation starts helping almost immediately,
        without restarting anything.
        """
        if time.time() - self._learned_t0 < LEARN_POLL_SECONDS:
            return
        self._learned_t0 = time.time()
        learned_names = set()

        # confirmed FACES - permanent, useful every day
        try:
            rows = repo.learned_faces_since(self._learned_last_id)
        except Exception:
            rows = []
        usable, wrong_model = self._compatible_faces(rows)
        for row in rows:
            self._learned_last_id = max(self._learned_last_id, row["id"])
        if wrong_model:
            print(f"[GALLERY] ignored {wrong_model} confirmed face(s) made "
                  f"by a different recognition model")
        for row in usable:
            profile = GALLERY.add_person(row["name"], row.get("code", ""))
            profile.add_face(row["embedding"], permanent=True)
            profile.times_confirmed += 1
            # ...and into the face bank, so the very next frame searches
            # against it. This is what makes confirming somebody on the
            # dashboard take effect in seconds instead of at the next
            # restart.
            learn_face(row["name"], row["embedding"], row.get("code", ""))
            learned_names.add(row["name"])
        rows = usable

        # A confirmation used to teach clothing as well. It no longer
        # does: clothing is not evidence of who anybody is, and the one
        # thing worth learning from a confirmation - the face vector -
        # is handled above.

        if learned_names:
            print(f"[GALLERY] learned from {len(rows)} face "
                  f"confirmation(s): {', '.join(sorted(learned_names))}")

        self._apply_confirmations()

    def _apply_confirmations(self):
        """Put the confirmed name straight onto the person on screen.

        Learning what somebody looks like helps the NEXT time we see
        them. It does not rename the sighting that was asked about: that
        track is still labelled Unknown, its presence/visit row still
        says Unknown, and the answer appears to have done nothing - which
        is exactly what confirming felt like before.

        So a confirmation is also treated as an identification: the track
        the question came from is named directly, and its database row is
        back-filled, so the time already accumulated belongs to the right
        person.
        """
        try:
            answers = repo.recent_confirmations(CONFIRM_APPLY_WINDOW)
        except Exception as e:
            print(f"[REVIEW] could not read confirmations: {e}")
            return

        for answer in answers:
            if answer["id"] in self._applied_reviews:
                continue
            self._applied_reviews.add(answer["id"])

            name = answer["name"]
            if not name:
                continue
            # Only questions THIS run asked can be matched by track id -
            # ids restart at 1 every time the pipeline starts, so an
            # answer to a question from the previous run would otherwise
            # be pinned onto whoever happens to be track #7 now. Those
            # sightings are already corrected in the database by the
            # dashboard; here we would only be renaming a stranger.
            created = answer.get("created_at")
            if created is not None and created < self._started_at:
                continue
            cam = next((c for c in self.cameras
                        if c.key == answer["camera_key"]), None)
            if cam is None:
                continue
            track = next((t for t in cam.tracker.all_tracks()
                          if t.id == answer["track_id"]), None)
            if track is None:
                continue          # they have left; the row was back-filled
                                  # by the dashboard already

            code = answer.get("code") or ""
            # A human answer outranks anything the models can produce -
            # including the vision model's veto. If it had refused this
            # name on this body, the refusal is dropped rather than
            # silently overruling the person who just answered, which
            # would look exactly like the dashboard doing nothing.
            cam.clear_vlm_denial(track, name)
            # ...claimed at full score, so no later frame can quietly
            # rename them.
            if not cam.claim_identity(track, name, code, CONFIRMED_SCORE):
                continue
            # Pin it in the face stage too, and clear that track's vote
            # window. Without this the model spends the next few seconds
            # arguing with the person who just answered the question.
            if cam.face is not None:
                cam.face.force_identity(track.id, name, code, CONFIRMED_SCORE)
            if getattr(track, "presence_id", None) is not None:
                repo.identify_presence(track.presence_id, name, code,
                                       CONFIRMED_SCORE)
            if getattr(track, "visit_id", None) is not None:
                repo.identify_visit(track.visit_id, name, code,
                                    CONFIRMED_SCORE)
            print(f"[REVIEW] {cam.name}: track #{track.id} is {name} "
                  f"(confirmed by a human) - record updated")

        # keep the "already applied" set from growing forever
        if len(self._applied_reviews) > 2000:
            self._applied_reviews = set(list(self._applied_reviews)[-500:])

    def _check_health(self):
        """Say clearly when a camera is offline, without spamming.

        A stream that dies at 2am and never comes back should be obvious
        in the morning, not buried in a wall of retry lines.
        """
        if time.time() - self._health_t0 < 60:
            return
        self._health_t0 = time.time()
        down = [c for c in self.cameras if not c.grabber.connected]
        if not down:
            return
        for cam in down:
            h = cam.grabber.health()
            mins = h["down_seconds"] / 60.0
            print(f"[HEALTH] {cam.name} OFFLINE for {mins:.0f} min "
                  f"({h['failures']} attempts)")

    def health(self):
        return [cam.health() for cam in self.cameras]

    def _adapt(self):
        """Ease off detection when we cannot keep up, recover when we can."""
        if not ADAPTIVE_DETECTION or self._fps <= 0:
            return
        now = time.time()

        if self._fps < ADAPTIVE_TARGET_FPS:
            self._fast_since = None
            if self._slow_since is None:
                self._slow_since = now
            elif now - self._slow_since >= ADAPTIVE_PATIENCE:
                # ease off the camera that can best tolerate it: the one
                # already configured to detect least often
                candidates = sorted(self.cameras,
                                    key=lambda c: -c.detect_every)
                for cam in candidates:
                    if cam.ease_off():
                        break
                self._slow_since = now
        elif self._fps > ADAPTIVE_GOOD_FPS:
            self._slow_since = None
            if self._fast_since is None:
                self._fast_since = now
            elif now - self._fast_since >= ADAPTIVE_PATIENCE:
                for cam in self.cameras:
                    cam.catch_up()
                self._fast_since = now
        else:
            self._slow_since = self._fast_since = None

    def _preview(self, ready):
        """Optional OpenCV windows on the pipeline machine itself.

        This belongs to the frame loop, not to adaptive scheduling - it
        used to sit at the end of _adapt(), where `ready` does not exist,
        so turning SHOW_LOCAL_WINDOWS on crashed the pipeline with a
        NameError on the first cycle.
        """
        if not SHOW_LOCAL_WINDOWS:
            return
        for cam in ready:
            if cam.annotated is not None:
                cv2.imshow(cam.name, cam.annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.running = False

    def _publish_scene(self, cam):
        """Publish the RAW frame and who is in it, for the assistant.

        Two things make this cheap enough to sit in the frame loop.
        First it is throttled to SCENE_SNAPSHOT_INTERVAL, which is
        seconds rather than frames - nobody types a question ten times a
        second. Second it is only ever one JPEG encode; the reasoning
        about the picture happens in another process, when somebody
        actually asks.

        cam.frame, not cam.annotated: see SCENE_SNAPSHOT_ENABLED in
        config/settings.py for why the drawn frame is the wrong input.
        """
        if not SCENE_SNAPSHOT_ENABLED or cam.frame is None:
            return
        now = time.time()
        if now - self._scene_t0.get(cam.key, 0.0) < SCENE_SNAPSHOT_INTERVAL:
            return
        self._scene_t0[cam.key] = now

        frame = cam.frame
        scale = 1.0
        if SCENE_SNAPSHOT_MAX_WIDTH and frame.shape[1] > SCENE_SNAPSHOT_MAX_WIDTH:
            scale = SCENE_SNAPSHOT_MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY),
                            SCENE_SNAPSHOT_QUALITY])
        if not ok:
            return
        try:
            people = cam.scene_people(scale)
        except Exception as exc:
            # A snapshot is a convenience. Losing the frame loop over one
            # would not be.
            print(f"[SCENE] {cam.name}: could not list people: {exc}")
            return
        STORE.publish_scene(cam.key, buf.tobytes(), people)

    def _publish(self, cam):
        self._publish_scene(cam)
        if not STREAM_ENABLED or cam.annotated is None:
            return
        frame = cam.annotated
        # shrink for the browser - encoding a full 4MP frame every cycle
        # costs far more CPU than the dashboard tile can even show
        if STREAM_MAX_WIDTH and frame.shape[1] > STREAM_MAX_WIDTH:
            scale = STREAM_MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            STORE.publish(cam.key, buf.tobytes())

    def _record_stages(self, grab, detect, process, encode):
        if not PROFILE:
            return
        self._stage["grab"] += grab
        self._stage["detect"] += detect
        self._stage["process"] += process
        self._stage["encode"] += encode
        self._stage_n += 1

        if time.time() - self._profile_t0 < PROFILE_EVERY_SECONDS:
            return
        n = max(1, self._stage_n)
        parts = " ".join(f"{k} {v / n * 1000:5.1f}ms"
                         for k, v in self._stage.items())
        rates = " ".join(f"{c.key}:1/{c.effective_interval}"
                         for c in self.cameras)
        face_ms = ""
        for cam in self.cameras:
            if cam.face is not None and cam.face.last_ms:
                face_ms = f" | face {cam.face.last_ms:5.1f}ms (own thread)"
                break
        print(f"[PERF] {self._fps:4.1f} fps | {parts}{face_ms} | {rates}")
        # ONLY the profiling counters are reset here. Nothing else: this
        # function runs every PROFILE_EVERY_SECONDS (5s), and it used to
        # also reset the confirmation poll timer, the health timer, the
        # adaptive-scheduling timers and the learned-row cursors. Because
        # 5s < LEARN_POLL_SECONDS (10s), that meant _pull_confirmations
        # could never reach its own interval, so confirmations made on
        # the dashboard were NEVER picked up while PROFILE was on.
        for k in self._stage:
            self._stage[k] = 0.0
        self._stage_n = 0
        self._profile_t0 = time.time()

    def _tick_fps(self):
        self._fps_n += 1
        dt = time.time() - self._fps_t0
        if dt >= 1.0:
            self._fps = self._fps_n / dt
            self._fps_n = 0
            self._fps_t0 = time.time()

    def _stop_signal(self, *_):
        print("\n[PIPELINE] stopping...")
        self.running = False

    def shutdown(self):
        for cam in self.cameras:
            # close any visit still open so nobody is stuck "in view"
            for t in cam.tracker.all_tracks():
                if t.visit_id is not None:
                    repo.close_visit(t.visit_id)
                if getattr(t, "presence_id", None) is not None:
                    repo.update_presence(t.presence_id, t.present_seconds,
                                         active=False)
            cam.close()
        cv2.destroyAllWindows()
        repo.log_system("STOP", "pipeline stopped")
        print("[PIPELINE] stopped cleanly")
