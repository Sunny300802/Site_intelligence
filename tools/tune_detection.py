"""
tools/tune_detection.py
=======================
Find the right detection settings for YOUR camera, by measuring.

It samples frames across the WHOLE clip (not just the opening seconds),
runs detection with several combinations of input size / confidence /
tiling, reports how many people each setting finds, and saves annotated
images so you can see exactly what was detected.

USAGE
-----
    # a recorded clip - best option
    python tools/tune_detection.py --source "D:\\clips\\reception.mp4"

    # the live camera from config/cameras.py
    python tools/tune_detection.py

    # diagnostic: show EVERYTHING the model sees, not just people.
    # Use this if the people count is zero and you expected otherwise.
    python tools/tune_detection.py --source clip.mp4 --diagnose

IF EVERY SETTING REPORTS ZERO
-----------------------------
That usually means the sampled frames genuinely contain no people, not
that detection is broken. Run with --diagnose: it reports what the model
DOES see in each frame, which immediately tells you whether the clip has
people in it at all.
"""
import os
import sys
import logging
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from config.settings import (DETECT_MODEL, DEVICE, USE_HALF, PERSON_CLASS_ID,
                             DATA_DIR, TILE_GRID, TILE_OVERLAP, NMS_IOU,
                             ENTRY_NEAR_HEIGHT_FRAC, MIN_PERSON_HEIGHT_FRAC)
from config.cameras import enabled_cameras, camera_by_key
from core.boxes import nms, tile_windows, choose_tile_grid
from core.sampling import sample_frames, video_info

OUT_DIR = os.path.join(DATA_DIR, "tuning")
IMG_SIZES = [640, 960, 1280]
CONFS = [0.25, 0.15]


from core.ultra import quiet as quiet_ultralytics


PRECISION = {}


def predict(model, image, imgsz, conf, classes=None):
    kwargs = dict(imgsz=imgsz, conf=conf, device=DEVICE, verbose=False)
    kwargs.update(PRECISION)
    if classes is not None:
        kwargs["classes"] = classes
    res = model.predict(image, **kwargs)
    boxes, scores, names = [], [], []
    for r in res:
        label_map = r.names
        if r.boxes is not None and len(r.boxes) > 0:
            for b, c, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.conf.cpu().numpy(),
                                 r.boxes.cls.cpu().numpy()):
                boxes.append([float(b[0]), float(b[1]),
                              float(b[2]), float(b[3])])
                scores.append(float(c))
                names.append(label_map.get(int(cls), str(int(cls))))
    return boxes, scores, names


def predict_tiled(model, frame, imgsz, conf):
    h, w = frame.shape[:2]
    boxes, scores, _ = predict(model, frame, imgsz, conf, [PERSON_CLASS_ID])
    grid = choose_tile_grid(w, h, TILE_GRID)
    for (x1, y1, x2, y2) in tile_windows(w, h, grid, TILE_OVERLAP):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        b, s, _ = predict(model, crop, imgsz, conf, [PERSON_CLASS_ID])
        for bb, ss in zip(b, s):
            boxes.append([bb[0] + x1, bb[1] + y1, bb[2] + x1, bb[3] + y1])
            scores.append(ss)
    keep = nms(boxes, scores, NMS_IOU)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def annotate(frame, boxes, scores, title):
    img = frame.copy()
    h = img.shape[0]
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v) for v in b]
        frac = (y2 - y1) / h
        colour = (80, 200, 120) if frac >= ENTRY_NEAR_HEIGHT_FRAC else (0, 170, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, f"{s:.2f} {frac:.0%}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    return img


def diagnose(model, frames):
    """Report EVERYTHING the model sees. Answers 'is anyone even here?'"""
    print("\nDIAGNOSTIC - every object detected (imgsz=1280, conf=0.10)")
    print("-" * 68)
    seen = Counter()
    any_person = False
    for label, frame in frames:
        _b, _s, names = predict(model, frame, 1280, 0.10, None)
        counts = Counter(names)
        seen.update(counts)
        top = (", ".join(f"{n} x{c}" for n, c in counts.most_common(6))
               if counts else "nothing at all")
        mark = " <-- PERSON" if counts.get("person") else ""
        if counts.get("person"):
            any_person = True
        print(f"  {label:>8}  {top}{mark}")
    print("-" * 68)
    print(f"  totals: {dict(seen)}\n")

    if any_person:
        print("  People ARE present, so detection works. If the table below")
        print("  shows low numbers, it is a settings problem.")
    else:
        print("  NO people found anywhere in this clip. Detection is not the")
        print("  problem - the footage has nobody in it. Record a clip where")
        print("  somebody walks through reception, or point --source at the")
        print("  live camera while someone is in view.")
        if seen:
            print("  (Other objects WERE detected, so the model is running.)")
    return any_person


def crop_roi(frame, roi):
    if not roi:
        return frame, (0, 0)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, int(x1 * w if x1 <= 1 else x1))
    y1 = max(0, int(y1 * h if y1 <= 1 else y1))
    x2 = min(w, int(x2 * w if x2 <= 1 else x2))
    y2 = min(h, int(y2 * h if y2 <= 1 else y2))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return frame, (0, 0)
    return frame[y1:y2, x1:x2], (x1, y1)


def parse_roi(text):
    if not text:
        return None
    parts = [float(v) for v in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("--roi needs 4 numbers: x1,y1,x2,y2")
    return parts


def main(source, camera_key, do_diagnose, samples, roi):
    quiet_ultralytics()

    if source is None:
        cams = enabled_cameras()
        cfg = camera_by_key(camera_key) if camera_key else (cams[0] if cams else None)
        if cfg is None:
            print("[TUNE] no camera configured. Use --source with a file.")
            return 1
        source = cfg["url"]
        print(f"[TUNE] using camera '{cfg['name']}'")

    if not os.path.exists(DETECT_MODEL):
        print(f"[TUNE] model missing: {DETECT_MODEL}")
        return 1

    info = video_info(source)
    if info and info["seconds"]:
        print(f"[TUNE] source: {info['width']}x{info['height']}, "
              f"{info['seconds']:.1f}s, {info['fps']:.0f} fps")

    print(f"[TUNE] sampling {samples} frames across the WHOLE clip...")
    frames = sample_frames(source, count=samples)
    if not frames:
        print("[TUNE] no frames captured - check the path/URL")
        return 1
    h, w = frames[0][1].shape[:2]
    print(f"[TUNE] got {len(frames)} frames at {w}x{h}")

    os.makedirs(OUT_DIR, exist_ok=True)
    from ultralytics import YOLO
    from core.ultra import resolve_precision_kwarg
    model = YOLO(DETECT_MODEL, task="detect")
    global PRECISION
    PRECISION = resolve_precision_kwarg(model, 640, DEVICE,
                                        USE_HALF and DEVICE != "cpu")

    if do_diagnose:
        diagnose(model, frames)

    if roi:
        cropped, _ = crop_roi(frames[0][1], roi)
        print(f"[TUNE] detection region {roi} -> "
              f"{cropped.shape[1]}x{cropped.shape[0]} "
              f"(aspect {cropped.shape[1]/cropped.shape[0]:.2f})")

    grid = choose_tile_grid(w, h, TILE_GRID)
    print(f"[TUNE] tile grid for this frame shape: {grid} "
          f"({len(tile_windows(w, h, grid, TILE_OVERLAP)) + 1} passes)")

    # only frames that actually contain someone are meaningful for the
    # 'per frame' figure - averaging over empty frames hides everything
    if MIN_PERSON_HEIGHT_FRAC > 0:
        print(f"[TUNE] applying the production size filter: detections under "
              f"{MIN_PERSON_HEIGHT_FRAC:.0%} of frame height are discarded "
              f"({int(MIN_PERSON_HEIGHT_FRAC * h)}px here)")

    print(f"\n{'setting':<32}{'per busy frame':>15}{'busy frames':>13}"
          f"{'max':>6}{'dropped':>9}{'smallest':>10}{'largest':>9}")
    print("-" * 94)

    table = []
    for imgsz in IMG_SIZES:
        for conf in CONFS:
            for tiled in (False, True):
                total, with_people, max_n, dropped = 0, 0, 0, 0
                smallest, largest = 1.0, 0.0
                best_n, best_img, best_lbl = -1, None, ""
                for label, full in frames:
                    frame, (ox, oy) = crop_roi(full, roi)
                    if tiled:
                        b, s = predict_tiled(model, frame, imgsz, conf)
                    else:
                        b, s, _ = predict(model, frame, imgsz, conf,
                                          [PERSON_CLASS_ID])
                    b = [[bb[0] + ox, bb[1] + oy, bb[2] + ox, bb[3] + oy]
                         for bb in b]
                    # apply the SAME minimum-size filter the pipeline uses,
                    # so this table shows what production would really see
                    if MIN_PERSON_HEIGHT_FRAC > 0:
                        floor = MIN_PERSON_HEIGHT_FRAC * h
                        kept_pairs = [(bb, ss) for bb, ss in zip(b, s)
                                      if (bb[3] - bb[1]) >= floor]
                        dropped += len(b) - len(kept_pairs)
                        b = [bb for bb, _ in kept_pairs]
                        s = [ss for _, ss in kept_pairs]
                    total += len(b)
                    max_n = max(max_n, len(b))
                    if b:
                        with_people += 1
                    for bb in b:
                        frac = (bb[3] - bb[1]) / h
                        smallest = min(smallest, frac)
                        largest = max(largest, frac)
                    if len(b) > best_n:
                        best_n, best_lbl = len(b), label
                        best_img = (full, b, s)

                per_busy = total / with_people if with_people else 0.0
                name = f"imgsz={imgsz} conf={conf} {'tiled' if tiled else 'full'}"
                small_txt = f"{smallest:.0%}" if total else "-"
                large_txt = f"{largest:.0%}" if total else "-"
                print(f"{name:<32}{per_busy:>15.1f}"
                      f"{with_people:>9}/{len(frames)}{max_n:>6}"
                      f"{dropped:>9}{small_txt:>10}{large_txt:>9}")
                table.append((per_busy, max_n, imgsz, conf, tiled))

                if best_img is not None and best_n > 0:
                    tag = f"{imgsz}_conf{conf}_{'tiled' if tiled else 'full'}"
                    img = annotate(
                        best_img[0], best_img[1], best_img[2],
                        f"imgsz={imgsz} conf={conf} "
                        f"{'TILED' if tiled else 'full'} @{best_lbl} "
                        f"-> {best_n} people")
                    cv2.imwrite(os.path.join(OUT_DIR, f"{tag}.jpg"), img)

    print("-" * 94)
    print("\n[TUNE] 'dropped' = detections too small to ever become an entry,")
    print("       removed by the production size filter (mostly reflections).")
    print("[TUNE] 'per busy frame' counts only frames that contain someone,")
    print("       so empty footage no longer hides the real numbers.")
    print(f"[TUNE] annotated images: {OUT_DIR}")
    print("[TUNE] 'smallest'/'largest' = body height as a % of frame height.")
    print(f"[TUNE] green box = close enough to count as an entry "
          f"(>= {ENTRY_NEAR_HEIGHT_FRAC:.0%}); amber = too far.")

    if all(row[0] == 0 for row in table):
        print("\n[TUNE] Every setting found zero people.")
        print("[TUNE] Re-run with --diagnose to check whether the clip")
        print("       contains any people before changing settings.\n")
        return 0

    print("\nHOW TO READ THIS")
    print("-" * 60)
    print("More detections is NOT automatically better. A setting that")
    print("reports people in frames where the lobby is empty is finding")
    print("reflections, and those become phantom entries.")
    print()
    print("Run with --diagnose first to learn which frames really contain")
    print("people, then pick the setting whose 'busy frames' count matches")
    print("that number most closely. Prefer the CHEAPEST such setting")
    print("(smallest imgsz, no tiling) - it will also be the fastest.")
    print()
    print(f"Finally, open {OUT_DIR} and confirm the boxes are on people.\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure detection settings")
    ap.add_argument("--source", default=None,
                    help="video file or RTSP url (default: camera config)")
    ap.add_argument("--camera", default=None, help="camera key to test")
    ap.add_argument("--diagnose", action="store_true",
                    help="show every object the model sees")
    ap.add_argument("--samples", type=int, default=12,
                    help="frames to sample across the clip")
    ap.add_argument("--roi", default=None,
                    help="test a detection region: x1,y1,x2,y2 as "
                         "fractions, e.g. 0,0,0.5,1")
    args = ap.parse_args()
    raise SystemExit(main(args.source, args.camera, args.diagnose,
                          args.samples, parse_roi(args.roi)))
