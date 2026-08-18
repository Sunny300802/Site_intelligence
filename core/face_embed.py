"""
core/face_embed.py
==================
Turn an aligned 112x112 face into the vector that identifies it.

Why AdaFace replaced ArcFace/glintr100 here
-------------------------------------------
Not because it is "a newer model". Because of what it does with BAD
INPUT, which on a CCTV site is most input.

ArcFace applies the same angular margin to every training sample. A
blurry, half-lit, 40-pixel face is pushed just as hard toward its
identity cluster as a studio portrait, so the model learns to answer
confidently from whatever it can see - and at inference time it keeps
doing that, which is precisely the behaviour that produces a confident
wrong name.

AdaFace makes the margin a function of the sample's image quality,
approximated by the feature norm. Low-quality samples are de-emphasised
during training, so the model is not forced to squeeze an identity out
of pixels that do not contain one. The practical effect at inference is
that poor faces land nearer the middle of the space - lower similarity
to everybody - instead of confidently near somebody. That is exactly
the failure mode this upgrade is trying to remove, and it is why
AdaFace is worth the swap on a camera where faces are visible but poor.

That is the ARGUMENT, not the evidence. Do not take it on trust:
    python tools/eval_recognition.py --dataset data/faces --compare
runs both backends over the same photographs and prints the genuine and
impostor score distributions, so the change can be measured on this
site's own people rather than assumed.

Two backends live here
----------------------
  "adaface"  the new default.
  "arcface"  InsightFace's glintr100 (antelopev2) or w600k_r50
             (buffalo_l), driven directly rather than through
             FaceAnalysis. This is the OLD system's recognition model,
             kept solely as the baseline the comparison needs. It is
             never selected by default.

Both take the same input - an aligned BGR 112x112 crop from
core/face_align.py - and both return L2-normalised float32 vectors, so
everything downstream (the face bank, the gallery, the voting) is
identical whichever is loaded. Only the THRESHOLDS differ between them,
which is why FACE_RECOGNITION_THRESHOLD defaults differently per backend
and why the enrollment file records which model produced it.
"""
import os
import glob
import threading

import numpy as np
import cv2

from config.settings import (FACE_EMBED_BACKEND, ADAFACE_MODEL,
                             ADAFACE_INPUT_BGR, FACE_EMBED_FALLBACK,
                             FACE_EMBED_BATCH, FACE_EMBED_FLIP,
                             FACE_ALIGN_SIZE, FACE_MODEL, MODEL_DIR,
                             DEVICE, USE_TENSORRT)
from core.gpu import onnx_session


# Where an AdaFace ONNX might be, most specific first.
_ADAFACE_NAMES = ("adaface_ir101_webface12m.onnx",
                  "adaface_ir101_ms1mv3.onnx",
                  "adaface_ir101_webface4m.onnx",
                  "adaface_ir50_ms1mv2.onnx",
                  "adaface_ir50_webface4m.onnx",
                  "adaface_ir18_webface4m.onnx",
                  "adaface.onnx")

# The InsightFace recognition graph inside each pack.
_ARCFACE_FILES = {
    "antelopev2": "glintr100.onnx",
    "buffalo_l": "w600k_r50.onnx",
    "buffalo_s": "w600k_mbf.onnx",
}


def find_adaface_model(explicit=""):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for name in _ADAFACE_NAMES:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            return path
    matches = sorted(glob.glob(os.path.join(MODEL_DIR, "adaface*.onnx")))
    return matches[0] if matches else None


# What the OLD pipeline called the model that produced an embedding. It
# recorded the InsightFace PACK name, and a pack is not a model - both of
# these packs' recognition graphs are what this file now calls
# "arcface:<file>.onnx". Same weights, different label.
#
# Without this map the model guard rejects an enrollment that is in fact
# perfectly usable, purely because the naming convention changed, and the
# site loses every enrolled face for no reason.
_LEGACY_TAGS = {
    "antelopev2": "arcface:glintr100.onnx",
    "buffalo_l": "arcface:w600k_r50.onnx",
    "buffalo_s": "arcface:w600k_mbf.onnx",
}


def normalise_tag(tag):
    """Put an old pack name onto the current tag scheme."""
    if not tag:
        return ""
    return _LEGACY_TAGS.get(str(tag).strip(), str(tag).strip())


def tags_compatible(stored, current):
    """Were these two tags produced by the same recognition weights?

    An UNTAGGED sample (stored == "") is treated as compatible. Those
    rows predate model tracking, and refusing them would throw away
    every face a human has ever confirmed. That is a deliberate,
    bounded risk: it only applies to rows already in the database, and
    everything written from now on carries its tag.
    """
    if not stored:
        return True
    return normalise_tag(stored) == normalise_tag(current)


def find_arcface_model(pack=None):
    pack = pack or FACE_MODEL
    candidates = []
    filename = _ARCFACE_FILES.get(pack)
    if filename:
        candidates.append(os.path.join(MODEL_DIR, filename))
        candidates.append(os.path.expanduser(
            f"~/.insightface/models/{pack}/{filename}"))
    candidates += sorted(glob.glob(os.path.expanduser(
        f"~/.insightface/models/{pack}/*.onnx")))
    for path in candidates:
        base = os.path.basename(path).lower()
        # skip the detector / landmark / age models that share the folder
        if base.startswith(("scrfd", "det_", "1k3d", "2d106", "genderage")):
            continue
        if os.path.exists(path):
            return path
    return None


class _OnnxEmbedder:
    """Shared plumbing for both backends.

    Subclasses only have to say how a crop becomes a blob; everything
    else - batching, the flipped second pass, normalisation, the session
    lock - is the same either way.
    """

    name = "onnx"

    def __init__(self, model_path, use_tensorrt=None):
        self.model_path = model_path
        trt = USE_TENSORRT if use_tensorrt is None else bool(use_tensorrt)
        self.session, self.provider = onnx_session(
            model_path,
            use_tensorrt=trt and DEVICE != "cpu",
            cache_dir=os.path.join(MODEL_DIR, "trt_cache"),
            device_id=0 if DEVICE == "cpu" else int(DEVICE))
        self._lock = threading.Lock()

        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.batchable = not isinstance(shape[0], int) or shape[0] <= 0
        self.input_size = FACE_ALIGN_SIZE
        if isinstance(shape[2], int) and shape[2] > 0:
            self.input_size = int(shape[2])
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.dims = 512
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) >= 2 and isinstance(out_shape[-1], int) \
                and out_shape[-1] > 0:
            self.dims = int(out_shape[-1])

        self.on_gpu = "CUDA" in self.provider or "Tensorrt" in self.provider

    @property
    def model_tag(self):
        """Identifies the exact model that produced an embedding.

        Written into the enrollment file and checked at load. Two
        different recognition models produce equally valid 512-number
        descriptions of the same face which are NOT comparable with each
        other; without this tag a mismatch does not fail, it just
        silently matches nobody, and looks exactly like a broken
        threshold.
        """
        return f"{self.name}:{os.path.basename(self.model_path)}"

    # -------------------------------------------------------- blobbing
    def _blob(self, crops):
        raise NotImplementedError

    def _prepare(self, crops):
        """Resize anything that is not already the model's input size."""
        out = []
        for crop in crops:
            if crop is None or crop.size == 0:
                return None
            if crop.shape[0] != self.input_size or \
                    crop.shape[1] != self.input_size:
                crop = cv2.resize(crop, (self.input_size, self.input_size),
                                  interpolation=cv2.INTER_LINEAR)
            out.append(crop)
        return out

    def _run(self, blob):
        with self._lock:
            outputs = self.session.run(self.output_names,
                                       {self.input_name: blob})
        # AdaFace exports return (feature, norm); the feature is first.
        features = np.asarray(outputs[0], dtype=np.float32)
        return features.reshape(features.shape[0], -1)

    # ---------------------------------------------------------- public
    def embed(self, crops, flip=None):
        """Aligned BGR crops -> L2-normalised embeddings [N, dims].

        Rows for crops that could not be processed come back as zeros,
        so the caller's indexing is never disturbed - a silently dropped
        row is how a face ends up matched against somebody else's
        embedding.
        """
        if crops is None or len(crops) == 0:
            return np.zeros((0, self.dims), dtype=np.float32)

        do_flip = FACE_EMBED_FLIP if flip is None else bool(flip)
        results = np.zeros((len(crops), self.dims), dtype=np.float32)

        usable = [i for i, c in enumerate(crops)
                  if c is not None and getattr(c, "size", 0) > 0]
        if not usable:
            return results

        step = max(1, FACE_EMBED_BATCH if self.batchable else 1)
        for start in range(0, len(usable), step):
            index = usable[start:start + step]
            prepared = self._prepare([crops[i] for i in index])
            if prepared is None:
                continue
            try:
                features = self._run(self._blob(prepared))
                if do_flip:
                    mirrored = [cv2.flip(c, 1) for c in prepared]
                    features = features + self._run(self._blob(mirrored))
            except Exception as exc:
                print(f"[EMBED] inference failed: {exc}")
                continue
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.maximum(norms, 1e-9)
            for row, i in enumerate(index):
                if row < len(features):
                    results[i] = features[row]
        return results

    def embed_one(self, crop, flip=None):
        """Convenience for a single face. Returns a vector or None."""
        out = self.embed([crop], flip=flip)
        if len(out) == 0:
            return None
        vector = out[0]
        return None if float(np.abs(vector).sum()) < 1e-6 else vector


class AdaFaceEmbedder(_OnnxEmbedder):
    """AdaFace.

    Pre-processing is part of the model contract, not a preference:
    AdaFace's reference inference converts the image to BGR and scales
    it to [-1, 1] with (x/255 - 0.5) / 0.5. Feeding RGB instead does not
    fail - it produces a consistent but WRONG embedding space, so
    enrollment and recognition still agree with each other and accuracy
    quietly drops by a large margin with nothing in the logs to show for
    it. ADAFACE_INPUT_BGR exists only for an ONNX exported with the
    channel swap already baked in.
    """

    name = "adaface"

    def _blob(self, crops):
        batch = np.stack(crops).astype(np.float32)          # N,H,W,C (BGR)
        if not ADAFACE_INPUT_BGR:
            batch = batch[..., ::-1]
        batch = (batch / 255.0 - 0.5) / 0.5
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


class ArcFaceEmbedder(_OnnxEmbedder):
    """InsightFace glintr100 / w600k_r50 - the BASELINE, not the default.

    Present so the upgrade can be measured against the system it
    replaces on identical inputs. Its pre-processing is RGB scaled to
    [-1, 1] by (x - 127.5) / 127.5, which is what InsightFace's own
    ArcFaceONNX does.
    """

    name = "arcface"

    def _blob(self, crops):
        batch = np.stack(crops).astype(np.float32)[..., ::-1]   # BGR -> RGB
        batch = (batch - 127.5) / 127.5
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


_ADAFACE_HELP = """
AdaFace weights were not found.

AdaFace ships as a PyTorch checkpoint, so it needs converting once:

  1. download a checkpoint from the AdaFace project, e.g.
       adaface_ir101_webface12m.ckpt      (best accuracy)
       adaface_ir50_ms1mv2.ckpt           (lighter, faster)
  2. convert it:
       python tools/export_adaface_onnx.py --ckpt <the .ckpt file>
     which writes data/models/<name>.onnx
  3. re-enroll, because embeddings from two different models are not
     comparable:
       python tools/enroll_faces.py

Set ADAFACE_MODEL in config/settings.py (or the environment) if you keep
the file somewhere else.
""".strip()


def build_embedder(backend=None, model_path=None, quiet=False):
    """Load the configured recognition model. Returns an embedder.

    Raises RuntimeError only when nothing at all can be loaded. If
    AdaFace is missing and FACE_EMBED_FALLBACK is on, the old ArcFace
    model is used instead and said so LOUDLY - the site keeps running,
    but nobody is left thinking they are measuring AdaFace when they are
    not.
    """
    backend = (backend or FACE_EMBED_BACKEND or "adaface").strip().lower()

    if backend == "adaface":
        path = model_path or find_adaface_model(ADAFACE_MODEL)
        if path:
            embedder = AdaFaceEmbedder(path)
            if not quiet:
                _announce(embedder)
            return embedder
        print("[EMBED] " + _ADAFACE_HELP.replace("\n", "\n[EMBED] "))
        if not FACE_EMBED_FALLBACK:
            raise RuntimeError("AdaFace model not found and fallback is off")
        print("[EMBED] *** falling back to the OLD ArcFace model so the "
              "site keeps recognising people. This is NOT AdaFace - the "
              "accuracy improvement is not in effect. ***")
        backend = "arcface"

    if backend == "arcface":
        path = model_path or find_arcface_model()
        if not path:
            raise RuntimeError(
                f"No recognition model found for pack '{FACE_MODEL}'. "
                f"Expected {_ARCFACE_FILES.get(FACE_MODEL, '*.onnx')} in "
                f"{MODEL_DIR} or ~/.insightface/models/{FACE_MODEL}/.")
        embedder = ArcFaceEmbedder(path)
        if not quiet:
            _announce(embedder)
        return embedder

    raise RuntimeError(f"Unknown FACE_EMBED_BACKEND: {backend!r} "
                       f"(expected 'adaface' or 'arcface')")


def _announce(embedder):
    print(f"[EMBED] {embedder.model_tag} "
          f"({'GPU' if embedder.on_gpu else 'CPU'} via {embedder.provider}) "
          f"- {embedder.dims}-d, batch {'yes' if embedder.batchable else 'no'}, "
          f"flip-average {'on' if FACE_EMBED_FLIP else 'off'}")
    if not embedder.on_gpu:
        print("[EMBED] *** running on the CPU - recognition will be far too "
              "slow to keep up with the cameras. See core/gpu.py ***")


# ---------------------------------------------------------------------
# One embedder per process, shared by every camera - the recognition
# model is the largest thing the face stage puts on the GPU, and four
# cameras must not mean four copies of it.
# ---------------------------------------------------------------------
_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_ERROR = None


def get_embedder():
    """The shared embedder, or None if none could be loaded."""
    global _EMBEDDER, _EMBEDDER_ERROR
    if _EMBEDDER is not None or _EMBEDDER_ERROR is not None:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None and _EMBEDDER_ERROR is None:
            try:
                _EMBEDDER = build_embedder()
            except Exception as exc:
                _EMBEDDER_ERROR = exc
                print(f"[EMBED] face recognition disabled: {exc}")
    return _EMBEDDER


def model_tag():
    """The tag of the model currently loaded, or the configured one."""
    embedder = get_embedder()
    if embedder is not None:
        return embedder.model_tag
    return f"{FACE_EMBED_BACKEND}:unloaded"


def loaded_backend():
    """Which model is REALLY running - 'adaface' or 'arcface'."""
    embedder = get_embedder()
    return embedder.name if embedder is not None else FACE_EMBED_BACKEND


def thresholds_for_loaded_model():
    """The similarity bars that belong to the model that actually loaded.

    This exists because of a specific, silent failure. The threshold is
    a property of the MODEL, but it was being chosen from
    FACE_EMBED_BACKEND - what was asked for - rather than from what was
    loaded. When AdaFace weights are missing the system falls back to
    ArcFace and keeps running, and it was then applying AdaFace's bar
    (0.38) to ArcFace, which needs 0.50. Every ArcFace score between the
    two was accepted as an identification when ArcFace considers it
    noise, and the result on the dashboard is a confident wrong name.

    An explicitly pinned value always wins - if somebody set the number
    by hand they meant it, whichever model came up.

    Returns (recognition_threshold, learn_agree_min, adjusted).
    """
    from config.settings import (FACE_BACKEND_THRESHOLDS,
                                 FACE_BACKEND_LEARN_AGREE,
                                 FACE_RECOGNITION_THRESHOLD,
                                 FACE_RECOGNITION_THRESHOLD_PINNED,
                                 FACE_LEARN_AGREE_MIN,
                                 FACE_LEARN_AGREE_MIN_PINNED)

    backend = loaded_backend()
    threshold = FACE_RECOGNITION_THRESHOLD
    agree = FACE_LEARN_AGREE_MIN
    adjusted = False

    if not FACE_RECOGNITION_THRESHOLD_PINNED:
        proper = FACE_BACKEND_THRESHOLDS.get(backend)
        if proper is not None and abs(proper - threshold) > 1e-9:
            threshold, adjusted = proper, True
    if not FACE_LEARN_AGREE_MIN_PINNED:
        proper = FACE_BACKEND_LEARN_AGREE.get(backend)
        if proper is not None:
            agree = proper
    return threshold, agree, adjusted
