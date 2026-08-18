"""
tools/benchmark.py
==================
Measure where the time actually goes, so tuning is based on numbers
rather than guesswork.

It times each stage separately on real frames from your camera:

    decode      reading a frame from the camera
    detect      YOLO person detection
    face-empty  face recognition with NO face in view
    face-person face recognition WITH a face in view   <-- the important one
    encode      JPEG encoding for the dashboard

The gap between "face-empty" and "face-person" is what causes the
pipeline to stall when somebody walks up to the camera. With the default
settings that gap should now be small; if it is large, the notes printed
at the end tell you what to change.

USAGE
-----
    python tools/benchmark.py
    python tools/benchmark.py --source "D:\\clips\\reception.mp4"
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from core.sampling import sample_frames, video_info
from config.settings import (DETECT_MODEL, DEVICE, USE_HALF, IMG_SIZE,
                             PERSON_CONF, PERSON_CLASS_ID,
                             FACE_DET_SIZE, FACE_CROP_DET_SIZE,
                             FACE_CROP_BATCH, FACE_EMBED_BACKEND,
                             FACE_INPUT_MAX_WIDTH, STREAM_MAX_WIDTH,
                             JPEG_QUALITY, TILED_DETECTION)
from config.cameras import enabled_cameras, camera_by_key


from core.ultra import quiet as quiet_ultralytics


def timed(fn, runs=10):
    fn()                                  # warm up, ignore
    t0 = time.time()
    for _ in range(runs):
        fn()
    return (time.time() - t0) / runs * 1000.0


def main(source, camera_key):
    quiet_ultralytics()
    if source is None:
        cams = enabled_cameras()
        cfg = camera_by_key(camera_key) if camera_key else (cams[0] if cams else None)
        if cfg is None:
            print("[BENCH] no camera configured; use --source")
            return 1
        source = cfg["url"]
        print(f"[BENCH] camera: {cfg['name']}")

    info = video_info(source)
    if info and info["seconds"]:
        print(f"[BENCH] source: {info['width']}x{info['height']}, "
              f"{info['seconds']:.1f}s")
    # sample across the WHOLE clip so we can find a frame containing a face
    sampled = sample_frames(source, count=12)
    if not sampled:
        print(f"[BENCH] could not read frames from {source}")
        return 1
    frames = [f for _label, f in sampled]
    h, w = frames[0].shape[:2]
    print(f"[BENCH] {len(frames)} frames sampled across the clip, {w}x{h}")
    print(f"[BENCH] settings: imgsz={IMG_SIZE} conf={PERSON_CONF} "
          f"tiled={TILED_DETECTION} face_modules={list(FACE_MODULES)}\n")

    results = {}

    # ---- detection ----
    from ultralytics import YOLO
    model = YOLO(DETECT_MODEL, task="detect")
    frame = frames[0]

    def do_detect():
        model.predict(frame, imgsz=IMG_SIZE, conf=PERSON_CONF, device=DEVICE,
                      half=USE_HALF and DEVICE != "cpu",
                      classes=[PERSON_CLASS_ID], verbose=False)
    results["detect"] = timed(do_detect)

    # ---- face stage, measured as three separate costs ----------
    # Detection, alignment+quality, and embedding are now distinct
    # stages, and they scale differently: detection is per FRAME,
    # embedding is per FACE THAT PASSES QUALITY. Reporting one combined
    # "face" number - as this tool used to - hides which of them is
    # actually the constraint, and they have opposite fixes.
    face_empty = face_person = face_embed = None
    try:
        from core.face_detect import get_detector
        from core.face_embed import get_embedder
        from core.face_align import align, align_from_box
        from core.face_quality import assess

        detector = get_detector()
        embedder = get_embedder()
        if detector is None:
            raise RuntimeError("no SCRFD detector found")

        def prep(img):
            if FACE_INPUT_MAX_WIDTH and img.shape[1] > FACE_INPUT_MAX_WIDTH:
                s = FACE_INPUT_MAX_WIDTH / img.shape[1]
                img = cv2.resize(img, None, fx=s, fy=s,
                                 interpolation=cv2.INTER_AREA)
            return img

        with_face, without_face = None, None
        for f in frames:
            n = len(detector.detect(prep(f), det_size=FACE_DET_SIZE))
            if n and with_face is None:
                with_face = f
            if not n and without_face is None:
                without_face = f
        blank = np.zeros_like(frames[0])
        without_face = without_face if without_face is not None else blank

        empty = prep(without_face)
        results["scrfd-empty"] = timed(
            lambda: detector.detect(empty, det_size=FACE_DET_SIZE), runs=6)
        face_empty = results["scrfd-empty"]

        if with_face is not None:
            busy = prep(with_face)
            results["scrfd-person"] = timed(
                lambda: detector.detect(busy, det_size=FACE_DET_SIZE), runs=6)
            face_person = results["scrfd-person"]

            # a batch of head crops, the way the pipeline detects them
            found = detector.detect(busy, det_size=FACE_DET_SIZE)
            crops = []
            for f in found[:FACE_CROP_BATCH]:
                x1, y1, x2, y2 = f["box"]
                pad = max(8, int(0.4 * max(x2 - x1, y2 - y1)))
                h_, w_ = busy.shape[:2]
                crop = busy[max(0, y1 - pad):min(h_, y2 + pad),
                            max(0, x1 - pad):min(w_, x2 + pad)]
                if crop.size:
                    crops.append(crop)
            if crops:
                batch = (crops * FACE_CROP_BATCH)[:FACE_CROP_BATCH]
                results[f"scrfd-crops x{len(batch)}"] = timed(
                    lambda: detector.detect_many(
                        batch, det_size=FACE_CROP_DET_SIZE, max_faces=1),
                    runs=6)

            aligned = []
            for f in found:
                crop = (align(busy, f["landmarks"])
                        if f["landmarks"] is not None
                        else align_from_box(busy, f["box"]))
                if crop is not None:
                    aligned.append(crop)
            if aligned:
                results["align+quality"] = timed(
                    lambda: [assess(c, found[i]["box"],
                                    found[i]["landmarks"], found[i]["score"])
                             for i, c in enumerate(aligned)], runs=6)
            if aligned and embedder is not None:
                batch = (aligned * FACE_CROP_BATCH)[:FACE_CROP_BATCH]
                results[f"embed x{len(batch)}"] = timed(
                    lambda: embedder.embed(batch), runs=6)
                face_embed = results[f"embed x{len(batch)}"]
        else:
            print("[BENCH] note: no face was found anywhere in this clip, so")
            print("        the per-face costs could not be measured. Use a")
            print("        clip where somebody walks up to the camera.")
            print()
    except Exception as e:
        print(f"[BENCH] face stage unavailable: {e}")
        print()

    # ---- JPEG encode ----
    def do_encode():
        img = frame
        if STREAM_MAX_WIDTH and img.shape[1] > STREAM_MAX_WIDTH:
            s = STREAM_MAX_WIDTH / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    results["encode"] = timed(do_encode, runs=20)

    # ---- report ----
    print(f"{'stage':<16}{'time':>12}")
    print("-" * 28)
    for k, v in results.items():
        print(f"{k:<16}{v:>9.1f} ms")
    print("-" * 28)

    per_frame = results["detect"] + results["encode"]
    print(f"\nMain loop (detect + encode): {per_frame:.1f} ms "
          f"-> about {1000/max(per_frame,0.1):.1f} fps ceiling")

    print("\nNotes")
    print("-----")
    print("* The face stage runs on its own thread, so none of these times")
    print("  are on the main loop. They decide how QUICKLY somebody gets")
    print("  named, not the frame rate.")
    if face_empty is not None and face_person is not None:
        gap = face_person - face_empty
        print(f"* SCRFD costs {face_empty:.0f} ms on an empty view and "
              f"{face_person:.0f} ms with a person ({gap:+.0f} ms).")
        if face_person > 40:
            print(f"  Lower FACE_DET_SIZE (currently {FACE_DET_SIZE}) if that "
                  f"is too slow - detection on head crops does not need a "
                  f"large full-frame pass.")
    if face_embed is not None:
        print(f"* Embedding {FACE_CROP_BATCH} faces costs {face_embed:.0f} ms "
              f"({face_embed / max(1, FACE_CROP_BATCH):.1f} ms each). This is "
              f"per FACE THAT PASSES QUALITY, and it is bounded per pass by "
              f"FACE_MAX_RECOGNITIONS_PER_PASS.")
        print(f"  It is also the cost that RECOGNITION_INTERVAL and the "
              f"best-frame logic exist to avoid paying repeatedly for the "
              f"same person.")
    if results["detect"] > 40:
        print(f"* Detection is {results['detect']:.0f} ms. Lower IMG_SIZE "
              f"(currently {IMG_SIZE}) or turn TILED_DETECTION off to speed up.")
    if results["encode"] > 15:
        print(f"* JPEG encoding is {results['encode']:.0f} ms. Lower "
              f"STREAM_MAX_WIDTH (currently {STREAM_MAX_WIDTH}) or JPEG_QUALITY.")
    print()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure pipeline stage timings")
    ap.add_argument("--source", default=None)
    ap.add_argument("--camera", default=None)
    args = ap.parse_args()
    raise SystemExit(main(args.source, args.camera))
