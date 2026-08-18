"""
core/face_detect.py
===================
SCRFD face detection, driven directly rather than through a wrapper.

What changed and why
--------------------
The old pipeline called InsightFace's FaceAnalysis, which bundles
detection and recognition behind one .get() call. That was convenient
and cost us three things that matter on CCTV:

1. THE LANDMARKS WERE THROWN AWAY. FaceAnalysis aligns internally and
   hands back only a box and an embedding, so there was no way to align
   a face ourselves, no way to judge POSE, and no way to tell a frontal
   face from a profile before deciding whether to trust the match. Every
   piece of quality filtering in this upgrade needs those five points.

2. ONE FIXED INPUT SIZE, ONE IMAGE AT A TIME. A reception camera and a
   workspace camera want different detector sizes, and detecting eight
   person crops in one batched call is many times cheaper than eight
   separate calls. Neither was expressible.

3. NO CONTROL OVER THE CONFIDENCE FLOOR. The wrapper's 0.5 default
   discards exactly the faces this system cares about - the poor ones -
   before anything else has a chance to judge them.

So we load the SCRFD graph ourselves. The weights are the ones already
on the machine: InsightFace's model packs ship SCRFD-10GF as their
detector (antelopev2/scrfd_10g_bnkps.onnx, buffalo_l/det_10g.onnx), so
this needs no new download.

How SCRFD's outputs are read
----------------------------
SCRFD is anchor-free and predicts, at each of three feature strides
(8, 16, 32), a score, a distance-to-each-edge box, and (on the *_bnkps
/ det_* variants) five keypoint offsets. Distances are in units of the
stride, so decoding is: take the anchor centre for that cell, push out
by the predicted distances x stride.

The number of output tensors tells us the variant, which is how one
implementation covers scrfd_500m through scrfd_10g without a config
file per model:
      6 outputs  3 strides, 1 anchor,  no keypoints
      9 outputs  3 strides, 2 anchors, keypoints      <- the 10g packs
     10 outputs  5 strides, 1 anchor,  no keypoints
     15 outputs  5 strides, 1 anchor,  keypoints
"""
import os
import glob
import threading

import numpy as np
import cv2

from config.settings import (SCRFD_MODEL, MODEL_DIR, FACE_DET_SIZE,
                             FACE_DETECTION_THRESHOLD, FACE_NMS_IOU,
                             DEVICE, USE_TENSORRT)
from core.gpu import onnx_session


# Where to look for a detector if SCRFD_MODEL is not set. Ordered by
# preference: a copy inside the project first (so a deployment can be
# self-contained), then the InsightFace cache the old pipeline populated.
_SEARCH = [
    os.path.join(MODEL_DIR, "scrfd_10g_bnkps.onnx"),
    os.path.join(MODEL_DIR, "det_10g.onnx"),
    os.path.join(MODEL_DIR, "scrfd_2.5g_bnkps.onnx"),
    os.path.join(MODEL_DIR, "scrfd_500m_bnkps.onnx"),
    os.path.expanduser("~/.insightface/models/antelopev2/scrfd_10g_bnkps.onnx"),
    os.path.expanduser("~/.insightface/models/buffalo_l/det_10g.onnx"),
    os.path.expanduser("~/.insightface/models/buffalo_s/det_500m.onnx"),
]


def find_model(explicit=""):
    """The SCRFD weights to use, or None with nothing found."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for path in _SEARCH:
        if os.path.exists(path):
            return path
    # last resort: any scrfd/det_*.onnx anywhere in the insightface cache
    for path in sorted(glob.glob(os.path.expanduser(
            "~/.insightface/models/*/*.onnx"))):
        base = os.path.basename(path).lower()
        if base.startswith("scrfd") or base.startswith("det_"):
            return path
    return None


def _nms(boxes, scores, threshold):
    """Standard NMS over (x1, y1, x2, y2) boxes. Returns kept indices."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = (np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1))
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = rest[iou <= threshold]
    return keep


class SCRFD:
    """One SCRFD graph, shared by every camera on this process.

    Thread-safe: the face workers run one per camera and all call detect
    on the same session, which onnxruntime does not guarantee is safe for
    concurrent Run() on every provider. A lock costs nothing next to the
    inference itself and removes a whole class of intermittent crash.
    """

    def __init__(self, model_path=None, det_threshold=None, nms_iou=None,
                 use_tensorrt=None):
        self.model_path = model_path or find_model(SCRFD_MODEL)
        if not self.model_path:
            raise FileNotFoundError(
                "No SCRFD face detector found.\n"
                "Set SCRFD_MODEL in config/settings.py, or put one of\n"
                "  scrfd_10g_bnkps.onnx / det_10g.onnx\n"
                f"into {MODEL_DIR}.\n"
                "Both ship inside the InsightFace model packs this machine\n"
                "already has (~/.insightface/models/).")

        self.det_threshold = (FACE_DETECTION_THRESHOLD if det_threshold is None
                              else float(det_threshold))
        self.nms_iou = FACE_NMS_IOU if nms_iou is None else float(nms_iou)

        trt = USE_TENSORRT if use_tensorrt is None else bool(use_tensorrt)
        self.session, self.provider = onnx_session(
            self.model_path,
            use_tensorrt=trt and DEVICE != "cpu",
            cache_dir=os.path.join(MODEL_DIR, "trt_cache"),
            device_id=0 if DEVICE == "cpu" else int(DEVICE))

        self._lock = threading.Lock()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        shape = self.session.get_inputs()[0].shape
        # A fixed batch dimension means we may not stack images; most of
        # the shipped packs are exported with batch 1.
        self.batchable = not isinstance(shape[0], int) or shape[0] <= 0
        # A fixed H/W means the graph only accepts one input size, so the
        # per-camera det_size settings have to be ignored for this file.
        self.fixed_hw = None
        if isinstance(shape[2], int) and shape[2] > 0 and \
                isinstance(shape[3], int) and shape[3] > 0:
            self.fixed_hw = (int(shape[2]), int(shape[3]))

        self._configure_from_outputs()
        self._anchor_cache = {}

        print(f"[SCRFD] {os.path.basename(self.model_path)} "
              f"({'GPU' if 'CUDA' in self.provider or 'Tensorrt' in self.provider else 'CPU'}"
              f" via {self.provider}) - strides {self.strides}, "
              f"landmarks {'yes' if self.use_kps else 'NO'}, "
              f"conf>={self.det_threshold}")
        if "CUDA" not in self.provider and "Tensorrt" not in self.provider:
            print("[SCRFD] *** running on the CPU - face detection will be "
                  "far too slow to keep up. See core/gpu.py ***")
        if not self.use_kps:
            print("[SCRFD] *** this detector has no landmark outputs, so "
                  "faces cannot be aligned properly and pose cannot be "
                  "judged. Use scrfd_10g_bnkps.onnx or det_10g.onnx. ***")

    def _configure_from_outputs(self):
        """Work out the variant from how many tensors it returns."""
        n = len(self.output_names)
        if n == 6:
            self.fmc, self.strides, self.num_anchors, self.use_kps = \
                3, [8, 16, 32], 1, False
        elif n == 9:
            self.fmc, self.strides, self.num_anchors, self.use_kps = \
                3, [8, 16, 32], 2, True
        elif n == 10:
            self.fmc, self.strides, self.num_anchors, self.use_kps = \
                5, [8, 16, 32, 64, 128], 1, False
        elif n == 15:
            self.fmc, self.strides, self.num_anchors, self.use_kps = \
                5, [8, 16, 32, 64, 128], 1, True
        else:
            raise RuntimeError(
                f"{os.path.basename(self.model_path)} returns {n} tensors, "
                f"which is not a layout SCRFD is known to produce. Use one "
                f"of the standard scrfd_*/det_* exports.")

    # ------------------------------------------------------ pre-process
    def _letterbox(self, image, det_size):
        """Resize keeping aspect ratio, pad into a det_size square.

        Padding rather than stretching matters: SCRFD's anchors assume
        faces are roughly square, and squashing a 16:9 frame into a
        square makes every face wide and short, which measurably costs
        both detections and landmark accuracy.
        """
        height, width = image.shape[:2]
        target_h, target_w = det_size
        scale = min(target_h / max(1, height), target_w / max(1, width))
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(image, (max(1, new_w), max(1, new_h)),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        return canvas, scale

    def _det_size(self, requested):
        if self.fixed_hw is not None:
            return self.fixed_hw
        size = int(requested or FACE_DET_SIZE)
        # SCRFD's largest stride is 32 (or 128 on the 5-stride variants);
        # a size that is not a multiple produces a fractional feature map
        # and the anchor grid stops lining up with the outputs.
        step = 32 * (4 if self.fmc == 5 else 1)
        size = max(step, int(round(size / step)) * step)
        return (size, size)

    # ---------------------------------------------------------- anchors
    def _anchors(self, height, width, stride):
        key = (height, width, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        ys, xs = np.mgrid[:height, :width]
        centres = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
        centres = centres.reshape(-1, 2)
        if self.num_anchors > 1:
            centres = np.repeat(centres, self.num_anchors, axis=0)
        if len(self._anchor_cache) > 64:
            self._anchor_cache.clear()      # cameras only use a few sizes
        self._anchor_cache[key] = centres
        return centres

    # -------------------------------------------------------- inference
    def _run(self, blob):
        with self._lock:
            return self.session.run(self.output_names, {self.input_name: blob})

    @staticmethod
    def _slot(tensor, index):
        """One image's slice of an output tensor.

        SCRFD exports differ here in a way that silently corrupts
        results if ignored. A batch-1 export flattens the batch away and
        returns (N, C); a dynamic-batch export returns (B, N, C).
        Indexing the first as if it were the second does not raise - it
        just reads row `index`, so every box comes out in the wrong
        place. Checking the rank is the whole fix.
        """
        arr = np.asarray(tensor)
        return arr[index] if arr.ndim == 3 else arr

    def _decode(self, outputs, index, det_size, threshold):
        """Turn one image's raw outputs into (boxes, scores, keypoints)."""
        det_h, det_w = det_size
        boxes, scores, points = [], [], []

        for level, stride in enumerate(self.strides):
            score = self._slot(outputs[level], index)
            bbox = self._slot(outputs[level + self.fmc], index) * stride
            score = score.reshape(-1)
            bbox = bbox.reshape(-1, 4)

            keep = np.where(score >= threshold)[0]
            if keep.size == 0:
                continue

            centres = self._anchors(det_h // stride, det_w // stride, stride)
            if centres.shape[0] != bbox.shape[0]:
                # Defensive: an exported graph whose feature map does not
                # match the anchor grid would otherwise index garbage and
                # produce boxes in the wrong place, silently.
                continue
            c = centres[keep]
            d = bbox[keep]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1],
                                   c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]],
                                  axis=-1))
            scores.append(score[keep])

            if self.use_kps:
                kps = self._slot(outputs[level + self.fmc * 2], index) * stride
                kps = kps.reshape(-1, 5, 2)[keep]
                points.append(kps + c[:, None, :])

        if not boxes:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                    np.zeros((0, 5, 2), np.float32))

        boxes = np.concatenate(boxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        points = (np.concatenate(points, axis=0) if points
                  else np.zeros((len(boxes), 5, 2), np.float32))
        return boxes, scores, points

    def detect(self, image, det_size=None, threshold=None, max_faces=0):
        """Faces in one BGR image.

        Returns a list of dicts, biggest-and-most-confident first:
            {"box": (x1, y1, x2, y2), "score": float,
             "landmarks": np.ndarray(5, 2) or None}
        Coordinates are in the coordinate system of the image passed in.
        """
        results = self.detect_many([image], det_size=det_size,
                                   threshold=threshold, max_faces=max_faces)
        return results[0] if results else []

    def detect_many(self, images, det_size=None, threshold=None, max_faces=0):
        """Faces in several BGR images, batched into one call when the
        graph allows it.

        This is the path used for person crops: eight heads in one GPU
        launch instead of eight launches. Images may be different sizes -
        each is letterboxed into the same square canvas first.
        """
        images = [im for im in images]
        if not images:
            return []

        size = self._det_size(det_size)
        thr = self.det_threshold if threshold is None else float(threshold)

        blobs, scales, valid = [], [], []
        for i, image in enumerate(images):
            if image is None or image.size == 0 or image.shape[0] < 8 \
                    or image.shape[1] < 8:
                continue
            canvas, scale = self._letterbox(image, size)
            blobs.append(canvas)
            scales.append(scale)
            valid.append(i)

        out = [[] for _ in images]
        if not blobs:
            return out

        chunks = [(blobs, scales, valid)] if self.batchable else [
            ([b], [s], [v]) for b, s, v in zip(blobs, scales, valid)]

        for chunk_blobs, chunk_scales, chunk_valid in chunks:
            blob = cv2.dnn.blobFromImages(
                chunk_blobs, 1.0 / 128.0, (size[1], size[0]),
                (127.5, 127.5, 127.5), swapRB=True)
            try:
                outputs = self._run(blob.astype(np.float32))
            except Exception as exc:
                print(f"[SCRFD] inference failed: {exc}")
                continue

            for slot, (index, scale) in enumerate(
                    zip(chunk_valid, chunk_scales)):
                boxes, scores, points = self._decode(outputs, slot, size, thr)
                if len(boxes) == 0:
                    continue
                keep = _nms(boxes, scores, self.nms_iou)
                boxes, scores = boxes[keep], scores[keep]
                points = points[keep] if len(points) else points

                # back to the coordinates of the image we were handed
                inv = 1.0 / max(scale, 1e-9)
                boxes = boxes * inv
                if self.use_kps and len(points):
                    points = points * inv

                found = []
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j]
                    found.append({
                        "box": (int(round(x1)), int(round(y1)),
                                int(round(x2)), int(round(y2))),
                        "score": float(scores[j]),
                        "landmarks": (points[j].astype(np.float32)
                                      if self.use_kps and len(points) else None),
                    })
                # Rank by area x confidence, not by confidence alone: on a
                # CCTV frame the most confident face is often a poster or a
                # reflection near the lens, while the person we care about
                # is the biggest real one.
                found.sort(key=lambda f: (f["box"][2] - f["box"][0]) *
                           (f["box"][3] - f["box"][1]) * f["score"],
                           reverse=True)
                if max_faces:
                    found = found[:max_faces]
                out[index] = found
        return out


# ---------------------------------------------------------------------
# One detector per process. Every camera's face worker shares it, which
# keeps a single copy of the weights in GPU memory however many RTSP
# streams are running - the difference between 4 cameras costing 4x the
# detector memory and costing none.
# ---------------------------------------------------------------------
_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()
_DETECTOR_ERROR = None


def get_detector():
    """The shared SCRFD, or None if it could not be loaded."""
    global _DETECTOR, _DETECTOR_ERROR
    if _DETECTOR is not None or _DETECTOR_ERROR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None and _DETECTOR_ERROR is None:
            try:
                _DETECTOR = SCRFD()
            except Exception as exc:
                _DETECTOR_ERROR = exc
                print(f"[SCRFD] face detection disabled: {exc}")
    return _DETECTOR
