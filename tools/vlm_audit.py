"""
tools/vlm_audit.py
==================
Use a local vision-language model as GROUND TRUTH for tuning detection.

The problem this solves
-----------------------
Every time we tuned detection we compared settings against each other
with nothing to check them against, so "found more boxes" looked like
"better" - even when the extra boxes were reflections. A vision model
gives an independent opinion on how many people are really in a frame,
which turns that guesswork into a measurement.

It runs offline on sampled frames. It is far too slow for live use and is
never part of the pipeline.

USAGE
-----
    # audit a clip against the model's own count
    python tools/vlm_audit.py --source "D:\\clips\\office.mp4" --camera server_rm_psg

    # fewer frames = faster (each frame costs a few seconds)
    python tools/vlm_audit.py --source clip.mp4 --samples 6

WHAT IT PRINTS
--------------
For each sampled frame: the vision model's count, and what each YOLO
setting found. Then, per setting, the average error against the vision
model. The setting with the lowest error is the one to use - not the one
that finds the most boxes.
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from config.settings import (DETECT_MODEL, DEVICE, USE_HALF, PERSON_CLASS_ID,
                             DATA_DIR, TILE_GRID, TILE_OVERLAP, NMS_IOU,
                             MIN_PERSON_HEIGHT_FRAC, VLM_MODEL)
from config.cameras import enabled_cameras, camera_by_key
from core.boxes import nms, tile_windows, choose_tile_grid
from core.sampling import sample_frames, video_info
from core.geometry import to_pixels, build_mask, bounding_box, apply_mask
from core.ultra import quiet as quiet_ultralytics, resolve_precision_kwarg
from core import vlm

OUT_DIR = os.path.join(DATA_DIR, "vlm_audit")

# keep the sweep small - every frame costs a VLM call plus N detections
SETTINGS = [
    (640, 0.25, False),
    (960, 0.25, False),
    (960, 0.15, False),
    (1280, 0.25, False),
    (1280, 0.25, True),
]

PRECISION = {}


def detect(model, image, imgsz, conf):
    kwargs = dict(imgsz=imgsz, conf=conf, device=DEVICE, verbose=False,
                  classes=[PERSON_CLASS_ID])
    kwargs.update(PRECISION)
    res = model.predict(image, **kwargs)
    boxes, scores = [], []
    for r in res:
        if r.boxes is not None and len(r.boxes) > 0:
            for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy()):
                boxes.append([float(v) for v in b])
                scores.append(float(c))
    return boxes, scores


def detect_tiled(model, frame, imgsz, conf):
    h, w = frame.shape[:2]
    boxes, scores = detect(model, frame, imgsz, conf)
    grid = choose_tile_grid(w, h, TILE_GRID)
    for (x1, y1, x2, y2) in tile_windows(w, h, grid, TILE_OVERLAP):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        b, s = detect(model, crop, imgsz, conf)
        for bb, ss in zip(b, s):
            boxes.append([bb[0] + x1, bb[1] + y1, bb[2] + x1, bb[3] + y1])
            scores.append(ss)
    keep = nms(boxes, scores, NMS_IOU)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def size_filter(boxes, scores, frame_h):
    if MIN_PERSON_HEIGHT_FRAC <= 0:
        return boxes, scores
    floor = MIN_PERSON_HEIGHT_FRAC * frame_h
    pairs = [(b, s) for b, s in zip(boxes, scores) if (b[3] - b[1]) >= floor]
    return [b for b, _ in pairs], [s for _, s in pairs]


def area_prepare(frame, camera_cfg):
    """Apply the camera's monitored area, exactly as the pipeline does."""
    if not camera_cfg:
        return frame, (0, 0)
    opts = camera_cfg.get("options", {})
    poly = opts.get("area")
    roi = opts.get("detect_roi")
    h, w = frame.shape[:2]
    if poly:
        px = to_pixels(poly, w, h)
        mask = build_mask(px, w, h)
        bbox = bounding_box(px)
        return apply_mask(frame, mask, bbox), (bbox[0], bbox[1])
    if roi:
        x1 = int(roi[0] * w if roi[0] <= 1 else roi[0])
        y1 = int(roi[1] * h if roi[1] <= 1 else roi[1])
        x2 = int(roi[2] * w if roi[2] <= 1 else roi[2])
        y2 = int(roi[3] * h if roi[3] <= 1 else roi[3])
        return frame[y1:y2, x1:x2], (x1, y1)
    return frame, (0, 0)


def annotate(frame, boxes, title):
    img = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 120), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(img, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return img


def main(source, camera_key, samples):
    quiet_ultralytics()

    cfg = camera_by_key(camera_key) if camera_key else None
    if source is None:
        cams = enabled_cameras()
        cfg = cfg or (cams[0] if cams else None)
        if cfg is None:
            print("[VLM] no camera configured; use --source")
            return 1
        source = cfg["url"]
    if cfg:
        print(f"[VLM] camera settings: {cfg['name']}")

    ok, names = vlm.available()
    if not ok:
        print(f"[VLM] vision model '{VLM_MODEL}' not available at "
              f"{vlm.VLM_HOST}")
        if names:
            print(f"[VLM] models Ollama has: {', '.join(names)}")
        print("[VLM] install one with:  ollama pull qwen2.5vl:7b")
        print("[VLM] (or set VLM_MODEL in config/settings.py)")
        return 1
    print(f"[VLM] using vision model: {VLM_MODEL}")

    info = video_info(source)
    if info and info["seconds"]:
        print(f"[VLM] source: {info['width']}x{info['height']}, "
              f"{info['seconds']:.0f}s")

    frames = sample_frames(source, count=samples)
    if not frames:
        print("[VLM] could not read frames")
        return 1
    print(f"[VLM] {len(frames)} frames sampled across the clip\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(DETECT_MODEL, task="detect")
    global PRECISION
    PRECISION = resolve_precision_kwarg(model, 640, DEVICE,
                                        USE_HALF and DEVICE != "cpu")

    header = f"{'frame':>8}{'vision model':>14}" + "".join(
        f"{f'{i}/{c}{chr(84) if t else chr(70)}':>10}"
        for i, c, t in SETTINGS)
    print(header)
    print("-" * len(header))

    truth, results = [], {s: [] for s in SETTINGS}
    t_start = time.time()

    for label, full in frames:
        area_frame, _offset = area_prepare(full, cfg)
        n_true, note = vlm.count_people(area_frame)
        truth.append(n_true)

        row = f"{label:>8}{(str(n_true) if n_true is not None else '?'):>14}"
        for setting in SETTINGS:
            imgsz, conf, tiled = setting
            if tiled:
                b, s = detect_tiled(model, area_frame, imgsz, conf)
            else:
                b, s = detect(model, area_frame, imgsz, conf)
            b, s = size_filter(b, s, area_frame.shape[0])
            results[setting].append(len(b))
            row += f"{len(b):>10}"
        print(row)

        # keep a picture of the best setting for eyeballing
        imgsz, conf, tiled = SETTINGS[-1]
        b, _ = (detect_tiled(model, area_frame, imgsz, conf) if tiled
                else detect(model, area_frame, imgsz, conf))
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{label.replace('.', '_')}.jpg"),
                    annotate(area_frame, b,
                             f"{label}  vision={n_true}  yolo={len(b)}  {note[:40]}"))

    print("-" * len(header))
    valid = [(i, t) for i, t in enumerate(truth) if t is not None]
    if not valid:
        print("\n[VLM] the vision model did not return usable counts.")
        return 1

    print(f"\n{'setting':<26}{'avg error':>11}{'over':>7}{'under':>7}"
          f"{'verdict':>22}")
    print("-" * 73)
    scored = []
    for setting in SETTINGS:
        imgsz, conf, tiled = setting
        errs, over, under = [], 0, 0
        for i, t in valid:
            got = results[setting][i]
            errs.append(abs(got - t))
            if got > t:
                over += 1
            if got < t:
                under += 1
        avg = sum(errs) / len(errs)
        if over > under:
            verdict = "finds phantoms"
        elif under > over:
            verdict = "misses people"
        else:
            verdict = "balanced"
        name = (f"imgsz={imgsz} conf={conf} "
                f"{'tiled' if tiled else 'full'}")
        print(f"{name:<26}{avg:>11.2f}{over:>7}{under:>7}{verdict:>22}")
        scored.append((avg, imgsz, conf, tiled))

    best = min(scored)
    print("-" * 73)
    print(f"\n[VLM] closest to the vision model: imgsz={best[1]} "
          f"conf={best[2]} {'tiled' if best[3] else 'full frame'} "
          f"(average error {best[0]:.2f} people)")
    print(f"[VLM] annotated frames: {OUT_DIR}")
    print(f"[VLM] took {time.time() - t_start:.0f}s\n")

    print("HOW TO USE THIS")
    print("-" * 62)
    print("The vision model is a second opinion, not gospel - it can miss a")
    print("heavily hidden person too. Open the saved frames and check a")
    print("couple by eye before adopting a setting.")
    print()
    print("If EVERY setting 'misses people', the detector itself is the")
    print("limit: download yolov8m.pt (or yolov8l.pt) into data/models/ and")
    print("point DETECT_MODEL at it, then run this again.\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Audit detection against a local vision model")
    ap.add_argument("--source", default=None)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--samples", type=int, default=8,
                    help="frames to check (each costs a few seconds)")
    args = ap.parse_args()
    raise SystemExit(main(args.source, args.camera, args.samples))
