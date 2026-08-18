"""
core/face.py
============
The face subsystem's front door.

Everything outside core/ talks to face recognition through this module
and nothing else. That was true before this upgrade and it is still
true, which is why replacing the entire recognition stack -
InsightFace/AntelopeV2 out, SCRFD + AdaFace + ByteTrack + quality
filtering + temporal voting in - did not require the cameras, the
dashboard, the database or the streaming to change shape.

What lives behind here now
--------------------------
    core/face_detect.py     SCRFD, driven directly, with landmarks
    core/face_align.py      the 112x112 five-point warp
    core/face_quality.py    what is worth recognising at all
    core/face_embed.py      AdaFace (ArcFace kept as the baseline)
    core/facebank.py        multi-reference vector similarity search
    core/face_identity.py   best-frame selection + confidence voting
    core/face_pipeline.py   the worker thread that runs the above

Two ideas still do the heavy lifting, and they are the same two:

1. ASYNC. Recognition never blocks the detection loop. The pipeline
   drops the newest frame in; whenever the worker finishes it publishes
   the latest result. A slow face pass slows nothing else down.

2. GATING. A face that is too small, too blurry, too side-on, too dark
   or partly covered is detected but NOT matched. This is deliberate,
   and it is now measured properly rather than judged on width alone: a
   confident wrong name is far worse than an honest "Unknown". The
   person is named a moment later from a better frame, and the tracker
   back-fills that name over their whole visit.
"""
import os
import pickle

import numpy as np

from config.settings import (ENCODINGS_FILE, FACE_LEARN_AGREE_MIN,
                             FACE_EMBED_BACKEND)
from core.facebank import FACE_BANK, FaceBank

# The ENROLLMENT PHOTOGRAPHS ONLY - the pictures a human deliberately
# took of each person - kept separate from the live search index.
#
# This exists because of a specific, self-inflicted failure. The guard
# that decides whether a new sample really looks like the person it is
# being filed under was scoring it against FACE_BANK, which holds the
# enrollment photos AND every sample learned since. So the moment one
# wrong sample got in, the next wrong sample agreed with IT and passed;
# the guard was validating new evidence against evidence it had already
# failed to reject. Measured here after a few hours: 12 of 46 stored
# samples disagreed with their own person's enrollment photos, and most
# of them arrived through the guarded confirmation path.
#
# A reference set that never grows cannot be talked into anything. That
# is the whole point of it.
from core.face_pipeline import FacePipeline


ENROLLMENT_BANK = FaceBank()


def FACE_APP():
    """Deprecated. Kept so any local script that imported it still runs.

    There is no InsightFace FaceAnalysis object any more - detection and
    recognition are separate, explicit stages now. Anything that wants
    to embed a face should call embed_face() below, which goes through
    the same detect -> align -> quality -> AdaFace path the cameras use
    and therefore produces vectors that are actually comparable with the
    enrolled ones.
    """
    return None


# ---------------------------------------------------------------------
#                        the enrollment file
# ---------------------------------------------------------------------
def current_model_tag():
    """Which recognition model this process is actually running."""
    from core.face_embed import model_tag
    return model_tag()


def load_encodings(path=ENCODINGS_FILE, check_model=True):
    """Returns (matrix[N, dims], names[N], codes[N]) or (None, [], []).

    Encodings are only comparable with the model that made them. This is
    not a formality: AdaFace and ArcFace both produce perfectly valid
    512-number descriptions of the same face, and they have nothing to
    do with each other. Loading one set against the other model does not
    error - it matches nobody at all, which looks exactly like
    "recognition is broken" and sends people off tuning thresholds that
    were never the problem. So the file records its model and this
    refuses to use it with a different one.
    """
    if not os.path.exists(path):
        return None, [], []
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    made_with = data.get("embed_model") or data.get("face_model")
    if check_model:
        from core.face_embed import tags_compatible
        current = current_model_tag()
        if made_with and not tags_compatible(made_with, current):
            print(f"[FACE] *** the enrollment file was built with "
                  f"'{made_with}' but this process is running "
                  f"'{current}' ***")
            print(f"[FACE] these are not comparable - nobody would ever "
                  f"match.")
            print(f"[FACE] re-run:  python tools/enroll_faces.py")
            return None, [], []
        if not made_with:
            print(f"[FACE] the enrollment file predates model tracking. "
                  f"If nobody is recognised, re-run tools/enroll_faces.py")

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    return embeddings, list(data["names"]), list(data.get("codes", []))


def init_face_bank(embeddings=None, names=None, codes=None):
    """Fill the shared face bank. Returns how many people it holds.

    Called once at startup by core/pipeline.py. Separate from
    load_encodings on purpose - loading a file and mutating global state
    are two different things, and the enrollment and evaluation tools
    want the first without the second.
    """
    if embeddings is None:
        embeddings, names, codes = load_encodings()
    tag = current_model_tag()
    FACE_BANK.model_tag = tag
    ENROLLMENT_BANK.model_tag = tag
    if embeddings is None or not len(embeddings):
        return 0
    # The same photographs seed both, and only FACE_BANK grows from here.
    ENROLLMENT_BANK.load(embeddings, names, codes)
    return FACE_BANK.load(embeddings, names, codes)


def learn_face(name, embedding, code=""):
    """Add one confirmed reference to the live search index.

    Called whenever a human confirms somebody on the dashboard, so that
    confirmation starts helping within seconds rather than at the next
    restart.
    """
    return FACE_BANK.add(name, embedding, code)


def agrees_with_enrollment(name, embedding, minimum=None):
    """Does this sample actually look like the person it is labelled as?

    A guard on LEARNING, not on matching. A human confirming a suggestion
    is normally the best evidence there is - but a confirmation made
    while the system was suggesting the wrong person just records that
    wrong person, and a camera-realistic sample carries more weight than
    an enrollment photo, so one rubber-stamped answer can poison a name
    permanently.

    Returns (ok, score). Unknown or unenrolled names pass, because there
    is nothing to contradict.

    NOTE ON THE THRESHOLD: the numbers that justified 0.32 were measured
    on ArcFace/glintr100. AdaFace does not put genuine pairs in the same
    place, so FACE_LEARN_AGREE_MIN now defaults per backend and should be
    re-measured on this site's own confirmations:
        python tools/eval_recognition.py --dataset data/faces
    """
    if embedding is None or not name:
        return False, 0.0
    if minimum is None:
        # from the model that LOADED, not the one configured
        from core.face_embed import thresholds_for_loaded_model
        minimum = thresholds_for_loaded_model()[1]
    if not len(ENROLLMENT_BANK):
        init_face_bank()
    # Judged against the ENROLLMENT photographs alone. Scoring against
    # the live index would let one bad sample vouch for the next.
    score = ENROLLMENT_BANK.score_against(name, embedding)
    if score is None:
        return True, 0.0            # nothing enrolled to contradict it
    return score >= minimum, float(score)


# ---------------------------------------------------------------------
#                       one-off embedding
# ---------------------------------------------------------------------
def embed_face(image, box=None, min_quality=None, strict=False):
    """Detect, align, quality-check and embed the biggest face in an image.

    Used by the dashboard when a human confirms a saved crop, and by the
    enrollment and evaluation tools. It runs the SAME stages as the live
    cameras, which is the point: a vector produced here is directly
    comparable with the ones the pipeline produces, and one produced by
    a different code path silently is not.

    Returns (embedding, quality) - either may be None.
    """
    from core.face_detect import get_detector
    from core.face_embed import get_embedder
    from core.face_align import align, align_from_box
    from core.face_quality import assess
    from config.settings import (FACE_LEARN_QUALITY_MIN,
                                 FACE_LEARN_MIN_WIDTH)

    if image is None or image.size == 0:
        return None, None
    detector, embedder = get_detector(), get_embedder()
    if detector is None or embedder is None:
        return None, None

    faces = detector.detect(image)
    if not faces:
        return None, None
    if box is not None:
        # prefer a face inside the region the caller cares about
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        inside = [f for f in faces
                  if f["box"][0] <= cx <= f["box"][2]
                  and f["box"][1] <= cy <= f["box"][3]]
        faces = inside or faces

    face = faces[0]
    crop = (align(image, face["landmarks"]) if face["landmarks"] is not None
            else align_from_box(image, face["box"]))
    bar = FACE_LEARN_QUALITY_MIN if min_quality is None else float(min_quality)
    # The LEARNING size bar, not the live-recognition one. This path is
    # answering "is this picture worth keeping", and a human has already
    # said who it is - so it must not inherit MIN_FACE_SIZE, which is
    # tuned for the very different question of whether a face is big
    # enough to IDENTIFY somebody from unaided.
    quality = assess(crop, face["box"], face["landmarks"], face["score"],
                     threshold=bar, strict=strict,
                     min_size=FACE_LEARN_MIN_WIDTH)
    if not quality.ok:
        return None, quality
    return embedder.embed_one(crop), quality


# ---------------------------------------------------------------------
#                    what the cameras actually use
# ---------------------------------------------------------------------
class FaceRecognizer(FacePipeline):
    """The camera-facing face worker.

    Same name and same lifecycle as before (construct, submit, get,
    stop), so cameras/base.py did not have to be rebuilt around a new
    object. What changed is the shape of what submit() is given: the
    tracks in the frame, so that every face can be associated with a
    tracking id and voted on over time rather than judged alone.
    """

    def __init__(self, name="face", min_face_size=None):
        if not len(FACE_BANK):
            init_face_bank()
        super().__init__(name=name, min_face_size=min_face_size)
        if self.ready:
            summary = FACE_BANK.summary()
            print(f"[FACE:{name}] ready - {summary['people']} people, "
                  f"{summary['references']} reference face(s), "
                  f"backend {FACE_EMBED_BACKEND}")
