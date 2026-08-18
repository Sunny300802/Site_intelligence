"""
core/detector.py
================
One shared YOLO model for every camera, with optional tiled inference.

Why tiling exists
-----------------
YOLO shrinks the whole frame to IMG_SIZE before looking at it. On a
wide-angle CCTV view, a person at the far end can end up only ~20 pixels
tall after that shrink, which is below what the model can detect - so
they are missed regardless of which model you use.

Tiling cuts the frame into overlapping pieces and detects each piece at
full resolution. A distant person then occupies a much larger share of
the image the model actually sees, so they get found. The tile results
and the full-frame results are merged with non-maximum suppression so a
person straddling two tiles is not counted twice.

Tiling costs roughly (rows*cols + 1) times the GPU work, so it is off by
default. Use tools/tune_detection.py to decide whether you need it.
"""
import os
import numpy as np
import torch
from ultralytics import YOLO

from config.settings import (DEVICE, USE_HALF, IMG_SIZE, DETECT_MODEL,
                             DETECT_ENGINE, USE_TENSORRT, PERSON_CONF,
                             PERSON_CLASS_ID, TILED_DETECTION, TILE_GRID,
                             TILE_OVERLAP, NMS_IOU, MIN_PERSON_HEIGHT_FRAC)


from core.boxes import nms, tile_windows, choose_tile_grid
from core.ultra import quiet as quiet_ultralytics, resolve_precision_kwarg


class PersonDetector:
    def __init__(self):
        quiet_ultralytics()
        self.device = DEVICE
        self.half = USE_HALF and DEVICE != "cpu"
        self.tiled = TILED_DETECTION
        self.precision = {}        # resolved at warmup

        weights, self.is_engine = self._choose_weights()
        print(f"[DETECT] loading {os.path.basename(weights)} "
              f"(device={self.device}, fp16={self.half}, imgsz={IMG_SIZE}, "
              f"conf={PERSON_CONF}, tiled={self.tiled})")
        self.model = YOLO(weights, task="detect")

        if not self.is_engine and self.device != "cpu":
            try:
                self.model.to(f"cuda:{self.device}")
            except Exception as e:
                print(f"[DETECT] could not move model to GPU: {e}")
        self._warmup()

    def _choose_weights(self):
        if USE_TENSORRT and os.path.exists(DETECT_ENGINE):
            return DETECT_ENGINE, True
        if not os.path.exists(DETECT_MODEL):
            raise FileNotFoundError(
                f"Detection model not found: {DETECT_MODEL}\n"
                f"Download yolov8n.pt into data/models/ (see the manual).")
        return DETECT_MODEL, False

    def _warmup(self):
        # figure out what this Ultralytics build calls the FP16 flag, so
        # we neither spam deprecation warnings nor crash on an unknown
        # argument
        self.precision = resolve_precision_kwarg(
            self.model, IMG_SIZE, self.device, self.half)
        if self.precision:
            key, val = next(iter(self.precision.items()))
            print(f"[DETECT] precision arg: {key}={val!r}")
        elif self.half:
            print("[DETECT] FP16 not supported by this build; using FP32")
        try:
            blank = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            self.model.predict(blank, imgsz=IMG_SIZE, device=self.device,
                               verbose=False, **self.precision)
            print("[DETECT] warmup done")
        except Exception as e:
            print(f"[DETECT] warmup skipped: {e}")

    # ------------------------------------------------------------ raw
    @torch.no_grad()
    def _predict(self, images, imgsz=None, conf=None):
        """Run the model on a list of images -> list of (boxes, scores)."""
        if not images:
            return []
        results = self.model.predict(
            images, imgsz=imgsz or IMG_SIZE, conf=conf or PERSON_CONF,
            device=self.device, classes=[PERSON_CLASS_ID], verbose=False,
            **self.precision)
        out = []
        for r in results:
            boxes, scores = [], []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                for b, c in zip(xyxy, conf):
                    boxes.append([float(b[0]), float(b[1]),
                                  float(b[2]), float(b[3])])
                    scores.append(float(c))
            out.append((boxes, scores))
        return out

    # -------------------------------------------------------- public
    def detect_people(self, frames, imgsz=None, conf=None,
                      min_height_frac=None):
        """frames: list of BGR images, all sharing the same settings.
        Returns a list (aligned with frames) of lists of:
            {"box": (x1, y1, x2, y2), "conf": float}

        Detections smaller than MIN_PERSON_HEIGHT_FRAC of the frame are
        dropped here, at the source. They can never become an entry (that
        needs a much larger person), so passing them on would only feed
        reflections and distant movement into the tracker.
        """
        if not frames:
            return []
        if not self.tiled:
            raw = self._predict(frames, imgsz, conf)
            return [self._dedupe(
                        self._filter(self._pack(b, s), f.shape[0],
                                     min_height_frac))
                    for (b, s), f in zip(raw, frames)]
        return [self._dedupe(
                    self._filter(self._detect_tiled(f, imgsz, conf),
                                 f.shape[0], min_height_frac))
                for f in frames]

    @staticmethod
    def _dedupe(detections, contained=0.72):
        """Drop a detection that sits almost entirely inside a bigger one.

        YOLO regularly returns a person's upper body AND their whole
        body as two boxes. Ordinary NMS keeps both, because their
        intersection over UNION is low - the small box is only a third
        of the big one. Measured on this site the survivors then became
        separate tracks: a phantom second person 67px from a real one,
        appearance 0.99 identical, which is exactly the "after they
        separate, the other one is a person and the real one is Unknown"
        report.
        
        So the test is intersection over the SMALLER area, which is what
        actually expresses "this box is inside that one".
        """
        if len(detections) < 2:
            return detections
        order = sorted(range(len(detections)),
                       key=lambda i: -((detections[i]["box"][2] -
                                        detections[i]["box"][0]) *
                                       (detections[i]["box"][3] -
                                        detections[i]["box"][1])))
        keep, boxes = [], []
        for index in order:
            box = detections[index]["box"]
            area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            swallowed = False
            for other in boxes:
                ix1, iy1 = max(box[0], other[0]), max(box[1], other[1])
                ix2, iy2 = min(box[2], other[2]), min(box[3], other[3])
                overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if overlap / area >= contained:
                    swallowed = True
                    break
            if not swallowed:
                keep.append(detections[index])
                boxes.append(box)
        return keep

    @staticmethod
    def _filter(detections, frame_height, min_height_frac=None):
        """Drop detections too short to be a person worth tracking.

        PER CAMERA, because "too short" depends entirely on the view. A
        reception camera sees people standing, so a short box really is
        a reflection. A work-area camera sees people SEATED behind
        desks, where a short box is the normal appearance of somebody
        who is definitely there - measured on this site, one global
        value of 0.12 discarded 55% of all detections on that camera,
        which is exactly the "they are in the area but not detected"
        complaint.
        """
        floor_frac = (MIN_PERSON_HEIGHT_FRAC if min_height_frac is None
                      else float(min_height_frac))
        if floor_frac <= 0:
            return detections
        floor = floor_frac * frame_height
        return [d for d in detections
                if (d["box"][3] - d["box"][1]) >= floor]

    def _detect_tiled(self, frame, imgsz=None, conf=None):
        """Full-frame pass plus overlapping tiles, merged with NMS."""
        h, w = frame.shape[:2]
        images = [frame]
        offsets = [(0, 0)]
        grid = choose_tile_grid(w, h, TILE_GRID)
        for (x1, y1, x2, y2) in tile_windows(w, h, grid, TILE_OVERLAP):
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                images.append(crop)
                offsets.append((x1, y1))

        raw = self._predict(images, imgsz, conf)

        all_boxes, all_scores = [], []
        for (boxes, scores), (ox, oy) in zip(raw, offsets):
            for b, s in zip(boxes, scores):
                all_boxes.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy])
                all_scores.append(s)

        keep = nms(all_boxes, all_scores, NMS_IOU)
        return self._pack([all_boxes[i] for i in keep],
                          [all_scores[i] for i in keep])

    @staticmethod
    def _pack(boxes, scores):
        return [{"box": (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                 "conf": float(s)} for b, s in zip(boxes, scores)]
