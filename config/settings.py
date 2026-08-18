"""
config/settings.py
==================
Every global setting lives here. Per-camera settings are in cameras.py.

Anything you might reasonably want to change is in this one file, grouped
and commented. Nothing else in the project hard-codes these values.

ENVIRONMENT OVERRIDES
---------------------
Every setting in the FACE RECOGNITION PIPELINE section below can also be
set from the environment, using the same name. That is what makes it
possible to A/B two thresholds on the same footage without editing code:

    set FACE_RECOGNITION_THRESHOLD=0.38 && python run_pipeline.py

The environment always wins. Anything not set falls back to the value
written here.
"""
import os

# ------------------------------------------------- environment helpers
# Small, deliberate, and used only by the tunable blocks below, so that
# "what is this system actually running with" is answerable from one
# place - the console banner prints the resolved values at startup.


def _env_str(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_int(name, default):
    try:
        return int(str(_env_str(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(str(_env_str(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
FACE_DIR = os.path.join(DATA_DIR, "faces")          # enrollment photos
DB_PATH = os.path.join(DATA_DIR, "site.db")
DB_URL = f"sqlite:///{DB_PATH}"
ENCODINGS_FILE = os.path.join(DATA_DIR, "face_encodings.pkl")

# ------------------------------------------------------------------ GPU
DEVICE = 0                 # CUDA device index. "cpu" only for debugging.
USE_HALF = True            # FP16 - roughly 2x on RTX Ada tensor cores

# INPUT RESOLUTION - the single most important setting for detecting
# people far from the camera.
#
# YOLO shrinks the whole frame to IMG_SIZE before looking at it. On a
# 2560x1440 camera, IMG_SIZE=640 turns a distant 90px-tall person into
# about 22px, which is below what the model can reliably see - so they
# are simply missed, no matter which model you use.
#   640  = fastest; plenty when a detection region is configured
#   960  = good balance (default)
#   1280 = only if measurement shows you need it
#
# NOTE: a detect_roi (see config/cameras.py) is usually better than a
# bigger IMG_SIZE. Cropping to the area people actually use makes them
# proportionally larger to the model AND costs less, instead of more.
IMG_SIZE = 960

# Detection model. Bigger models help with hard cases, but resolution
# (above) helps small distant people far more.
# For a DENSE, OCCLUDED scene (an open-plan office with people seated
# behind monitors) a heavier model is worth real accuracy:
#   yolov8n.pt  fast, fine for a clear reception view
#   yolov8m.pt  recommended once you add the workspace camera
#   yolov8l.pt  best accuracy, if the GPU has room
# Download into data/models/ and change this line.
DETECT_MODEL = os.path.join(MODEL_DIR, "yolov8n.pt")
DETECT_ENGINE = os.path.join(MODEL_DIR, "yolov8n.engine")   # TensorRT (optional)
USE_TENSORRT = True        # use the .engine file when it exists

# Person detection confidence.
# 0.25 is deliberate. Dropping it lower does NOT find more real people on
# a reception camera - it finds reflections in the glass and calls them
# people, which then become phantom entries. Measured on real footage,
# conf 0.15 + tiling produced detections in frames that contained nobody
# at all. Raise to 0.30 if you still see phantoms; lower only if you can
# prove (with tools/tune_detection.py --diagnose) that real people are
# being missed.
PERSON_CONF = 0.25
PERSON_CLASS_ID = 0        # COCO class 0 = person

# MINIMUM PERSON SIZE, as a fraction of frame height.
# Anything smaller than this is discarded before it ever reaches the
# tracker. This is the single most effective filter against reflections
# and distant background movement.
#
# It is safe because of how entries work: a person must reach
# ENTRY_NEAR_HEIGHT_FRAC (0.22) to be counted as having entered, so a
# detection that never gets above 12% of the frame could never have
# produced an entry anyway - it could only add noise and churn the
# tracker. Set to 0 to disable.
MIN_PERSON_HEIGHT_FRAC = 0.12

# TILED DETECTION - a last resort, not a default.
# Measured on real reception footage, tiling at low confidence produced
# detections in frames containing nobody. Only enable it if
# tools/tune_detection.py --diagnose proves real people are being missed.
# The frame is split into overlapping tiles and each is detected at full
# resolution, which makes distant people effectively much larger to the
# model. Results are merged with the full-frame pass.
# Cost: roughly (rows*cols + 1) times the GPU work, so enable only if
# you need it. Use tools/tune_detection.py to see whether it helps you.
TILED_DETECTION = False

# "auto" picks the grid from the frame shape, which matters a lot:
# a wide CCTV frame (say 2.3:1) split LEFT/RIGHT gives tiles that are
# nearly square, delivering the same zoom benefit as a 2x2 grid with only
# 3 passes instead of 5 - about 40% less GPU work for the same result.
# Override with an explicit (rows, cols) tuple if you prefer.
TILE_GRID = "auto"
TILE_OVERLAP = 0.2         # fraction of tile size shared with neighbours
NMS_IOU = 0.55             # merge duplicate boxes between tiles

# ======================================================================
#                     FACE RECOGNITION PIPELINE
# ======================================================================
# RTSP -> person detection -> ByteTrack -> SCRFD face detection ->
# 5-point alignment -> quality filtering -> best-frame selection ->
# AdaFace embedding -> vector similarity search -> confidence-weighted
# temporal voting -> final identity.
#
# Everything in this section can be overridden from the environment
# (same name), so two configurations can be compared on the same footage
# without editing code.
# ======================================================================

FACE_ENABLED = _env_bool("FACE_ENABLED", True)

# ------------------------------------------------------- 1. DETECTION
# SCRFD (Sample and Computation Redistribution for Face Detection).
# Replaces the InsightFace FaceAnalysis wrapper: we drive the ONNX graph
# directly, which lets us control the input size per camera, batch face
# detection over person crops, and - crucially - keep the five facial
# LANDMARKS, which the wrapper threw away and which alignment needs.
#
# The weights already exist on this machine: InsightFace's model packs
# ship SCRFD-10GF as the detector. Either file works.
#   ~/.insightface/models/antelopev2/scrfd_10g_bnkps.onnx
#   ~/.insightface/models/buffalo_l/det_10g.onnx
# Leave SCRFD_MODEL empty to let core/face_detect.py find one of those,
# or point it at your own copy (e.g. a lighter scrfd_2.5g for speed).
SCRFD_MODEL = _env_str("SCRFD_MODEL", "")

# Detector input size, in pixels (square). This is the single biggest
# speed/accuracy lever on the face stage.
#   320  fastest, only finds faces that are already large
#   640  balanced (default)
#   960  finds smaller/more distant faces, ~2x the cost
FACE_DET_SIZE = _env_int("FACE_DET_SIZE", 640)

# Minimum SCRFD confidence for a box to be considered a face at all.
#
# What this does and does NOT fix: it decides whether a patch of pixels
# is a face, not whose face it is. Raising it removes spurious boxes -
# a poster, a chair back, a pattern on a monitor - which do produce
# wrong names, because a "face" that is not a face still gets embedded
# and still lands nearest SOMEBODY. It does not make a genuine but
# ambiguous face any less ambiguous; FACE_RECOGNITION_THRESHOLD and
# FACE_MATCH_MARGIN are what govern that.
#
# Raised from 0.45. On these cameras the quality filter was rejecting
# nearly everything under 0.55 anyway, so the extra strictness costs
# almost no real faces and removes the low-confidence detections that
# reach recognition on the workspace camera.
FACE_DETECTION_THRESHOLD = _env_float("FACE_DETECTION_THRESHOLD", 0.55)

# NMS overlap for merging duplicate face boxes.
FACE_NMS_IOU = _env_float("FACE_NMS_IOU", 0.4)

# Frames wider than this are shrunk before FULL-FRAME face detection.
# SCRFD resizes to FACE_DET_SIZE internally anyway, so handing it a full
# 4MP frame only burns memory bandwidth. Face boxes are scaled back to
# the original coordinates afterwards, so nothing else is affected.
FACE_INPUT_MAX_WIDTH = _env_int("FACE_INPUT_MAX_WIDTH", 1280)

# DETECT FACES INSIDE PERSON CROPS instead of on the whole frame.
#
# This is the accuracy option for CCTV. A person 200px tall in a 1280px
# frame has a ~45px face; after the detector shrinks the frame to 640
# that face is ~22px and is simply missed. Cropping the person's head
# region and detecting THAT at 640 makes the same face ~180px, so it is
# found, and its landmarks are far more accurate - which matters more
# than the detection itself, because bad landmarks mean bad alignment
# and bad alignment is the largest single source of recognition error.
#
# Cost is controlled: crops are batched into one GPU call, only tracks
# that need recognition are cropped, and the head region is a fraction
# of the body box. Set False to go back to one full-frame pass.
FACE_DETECT_ON_PERSON_CROPS = _env_bool("FACE_DETECT_ON_PERSON_CROPS", True)

# Fraction of the person box (from the top) searched for a face, and how
# much padding is added around it. 0.42 comfortably contains the head
# and shoulders of a standing person and of somebody seated at a desk.
FACE_CROP_HEAD_FRACTION = _env_float("FACE_CROP_HEAD_FRACTION", 0.42)
FACE_CROP_PAD = _env_float("FACE_CROP_PAD", 0.15)

# Detector input size used for person crops. Smaller than the full-frame
# size because the crop is already "zoomed in"; 320 is plenty and keeps
# a batch of 8 people cheap.
FACE_CROP_DET_SIZE = _env_int("FACE_CROP_DET_SIZE", 320)

# How many person crops go into one SCRFD call. Batching is what keeps
# per-crop detection affordable - one GPU launch instead of N.
FACE_CROP_BATCH = _env_int("FACE_CROP_BATCH", 8)

# --------------------------------------------------------- 2. ALIGNMENT
# Faces are warped to a fixed 112x112 template using the five landmarks
# (both eyes, nose tip, both mouth corners) before embedding. This is
# not optional: AdaFace, like every ArcFace-family model, is trained on
# this exact geometry and degrades badly on an unaligned crop.
FACE_ALIGN_SIZE = _env_int("FACE_ALIGN_SIZE", 112)

# ----------------------------------------------------- 3. EMBEDDING
# AdaFace. Chosen over ArcFace/glintr100 for one specific reason that
# matters here: AdaFace's loss adapts its margin to IMAGE QUALITY, so it
# is trained to give a low-quality face a less confident answer instead
# of a confidently wrong one. On CCTV, where most faces are poor, that
# is exactly the failure mode we are trying to remove.
#
#   "adaface"  AdaFace ONNX (default). See tools/export_adaface_onnx.py
#   "arcface"  InsightFace glintr100 / w600k_r50 - kept ONLY as the
#              BASELINE for tools/eval_recognition.py, so the upgrade can
#              be measured against the old system on the same dataset
#              rather than assumed to be better.
FACE_EMBED_BACKEND = _env_str("FACE_EMBED_BACKEND", "adaface")

# Path to the AdaFace ONNX file. Leave empty to look for the usual names
# in data/models/ (adaface_ir101_webface12m.onnx, adaface_ir50_*.onnx...).
#   python tools/export_adaface_onnx.py --ckpt adaface_ir101_webface12m.ckpt
ADAFACE_MODEL = _env_str(
    "ADAFACE_MODEL",
    "data/models/adaface_ir101_webface12m.onnx"
)

# AdaFace was trained on BGR input scaled to [-1, 1]. Do not change these
# unless you exported the ONNX graph with different pre-processing baked
# in - a mismatch here produces embeddings that look fine and match
# nobody.
ADAFACE_INPUT_BGR = _env_bool("ADAFACE_INPUT_BGR", True)

# If AdaFace cannot be loaded, fall back to the InsightFace recogniser
# rather than leaving the site with no face recognition at all. The
# fallback is announced loudly on the console and recorded in the model
# tag, so results can never be silently mixed between the two.
FACE_EMBED_FALLBACK = _env_bool("FACE_EMBED_FALLBACK", True)

# How many aligned faces are embedded in one call. The recognition model
# is the expensive part of the face stage; batching amortises the launch
# overhead across everybody visible in the frame.
FACE_EMBED_BATCH = _env_int("FACE_EMBED_BATCH", 8)

# Feed each face to the model twice - once as-is, once mirrored - and
# average the two embeddings. Costs one extra forward pass per face and
# measurably steadies scores on off-angle CCTV faces, which is precisely
# where identity switching happens. Turn off if the GPU is tight.
FACE_EMBED_FLIP = _env_bool("FACE_EMBED_FLIP", True)

# Legacy InsightFace pack, used by the "arcface" baseline backend and by
# tools/eval_recognition.py.
#   "antelopev2"  ResNet100 (glintr100) - what this system used before
#   "buffalo_l"   ResNet50  (w600k_r50)
FACE_MODEL = _env_str("FACE_MODEL", "antelopev2")
FACE_MODULES = ("detection", "recognition")

# How often the face stage runs, in PROCESSED frames. The face worker is
# asynchronous, so this only decides how often it is handed new work.
FACE_EVERY_N_FRAMES = _env_int("FACE_EVERY_N_FRAMES", 3)

# ------------------------------------------------- 4. QUALITY FILTERING
# The gate that stops the system recognising faces it should not be
# looking at. A poor face does not produce "no answer" from a recognition
# model - it produces a CONFIDENT WRONG ANSWER, which is worse. Every
# rejection below is one fewer chance of putting the wrong name on
# screen.
#
# Each metric produces a 0-1 sub-score. A face must pass every hard gate
# AND reach FACE_QUALITY_THRESHOLD overall before it is embedded at all.

# Minimum face width in pixels, in the ORIGINAL frame. A 40px face on a
# compressed CCTV stream carries about 25px between the eyes, which is
# below what any recognition model needs to tell two colleagues apart.
MIN_FACE_SIZE = _env_int("MIN_FACE_SIZE", 56)

# The width at which size stops being a limiting factor.
FACE_GOOD_SIZE = _env_int("FACE_GOOD_SIZE", 120)

# BLUR. Variance of the Laplacian over the aligned crop. Motion blur and
# heavy H.264 compression both collapse this number.
FACE_SHARPNESS_MIN = _env_float("FACE_SHARPNESS_MIN", 28.0)
FACE_SHARPNESS_GOOD = _env_float("FACE_SHARPNESS_GOOD", 160.0)

# ILLUMINATION. Mean and spread of brightness over the aligned crop.
# A face lit from behind (silhouette) or blown out by a window is
# unusable however sharp it is.
FACE_BRIGHTNESS_MIN = _env_float("FACE_BRIGHTNESS_MIN", 40.0)
FACE_BRIGHTNESS_MAX = _env_float("FACE_BRIGHTNESS_MAX", 225.0)
FACE_CONTRAST_MIN = _env_float("FACE_CONTRAST_MIN", 18.0)
FACE_CONTRAST_GOOD = _env_float("FACE_CONTRAST_GOOD", 55.0)

# POSE, measured from the five landmarks rather than a separate model,
# and expressed in DEGREES because that is the only form in which the
# number can be sanity-checked.
#
# The head-turn angle is estimated from where the nose tip sits between
# the eyes. Under a yaw of theta the nose moves off centre by roughly
#     (nose protrusion / interocular distance) * tan(theta)
# so the angle can be recovered from that offset. FACE_NOSE_PROJECTION
# is that ratio - about 0.5 for an average adult face.
#
# WHY DEGREES AND NOT A 0-1 SCORE. The first version of this scored the
# offset directly and gated at 0.30, which sounded reasonable and in
# fact rejected everything past about 35 degrees - the score saturates
# at 0 by 45 degrees, so a perfectly usable three-quarter view scored
# the same as a full profile. Two thirds of this site's enrollment
# photographs were being thrown away by it. An angle cannot hide that
# kind of mistake: 60 means 60.
FACE_NOSE_PROJECTION = _env_float("FACE_NOSE_PROJECTION", 0.5)

# Full marks at or below this turn; the score falls to zero at the hard
# gate. 60 degrees is about where the far eye disappears and an
# ArcFace-family model genuinely starts to lose the identity.
FACE_GOOD_YAW_DEG = _env_float("FACE_GOOD_YAW_DEG", 20.0)
FACE_MAX_YAW_DEG = _env_float("FACE_MAX_YAW_DEG", 60.0)

# Head TILT, from the eye line. Kept separate from turn on purpose: a
# person leaning their head on their hand is perfectly recognisable,
# and alignment removes tilt entirely, so this bar is generous.
FACE_MAX_ROLL_DEG = _env_float("FACE_MAX_ROLL_DEG", 35.0)

# Superseded by the degree-based gate above. Retained so an existing
# override or local edit does not raise on import.
FACE_POSE_YAW_MIN = _env_float("FACE_POSE_YAW_MIN", 0.30)
FACE_POSE_YAW_GOOD = _env_float("FACE_POSE_YAW_GOOD", 0.72)

# OCCLUSION. Measured as local detail energy across the eye, nose and
# mouth bands of the aligned crop: a hand, a mask, a monitor edge or a
# lanyard flattens whichever band it covers. The weakest band decides.
FACE_OCCLUSION_MIN = _env_float("FACE_OCCLUSION_MIN", 0.18)
FACE_OCCLUSION_GOOD = _env_float("FACE_OCCLUSION_GOOD", 0.55)

# How the sub-scores combine into one number. They are weighted, then
# multiplied by the detector's own confidence, so a face the detector
# was unsure about cannot reach a high quality score on sharpness alone.
FACE_QUALITY_WEIGHTS = {
    "size": _env_float("FACE_QUALITY_W_SIZE", 0.28),
    "sharpness": _env_float("FACE_QUALITY_W_SHARPNESS", 0.26),
    "pose": _env_float("FACE_QUALITY_W_POSE", 0.22),
    "illumination": _env_float("FACE_QUALITY_W_ILLUMINATION", 0.12),
    "occlusion": _env_float("FACE_QUALITY_W_OCCLUSION", 0.12),
}

# The overall bar. Below this a face is DETECTED and drawn, but never
# embedded and never allowed to influence an identity.
#
# Start at 0.45 and read the [FACE] quality histogram in the console: if
# almost everything is rejected the cameras cannot support this bar and
# it should come down; if wrong names still appear, raise it.
FACE_QUALITY_THRESHOLD = _env_float("FACE_QUALITY_THRESHOLD", 0.50)

# A separate, LOWER bar for learning from a face a human has confirmed.
# The risk is different: matching a blurry face risks the wrong name,
# whereas learning from one carries no such risk because a person has
# already told us who it is. The only question is whether the picture
# describes a face well enough to be worth keeping.
FACE_LEARN_QUALITY_MIN = _env_float("FACE_LEARN_QUALITY_MIN", 0.50)
# Must sit BELOW the smallest face your cameras actually produce, or
# confirmations teach nothing. Measured here: reception delivers 26-41px
# faces, so a 44px bar silently discarded every answer given about that
# camera - the person confirmed, the record updated, and nothing was
# learned.
#
# Setting it this low is safe because width is not what protects this
# path. agrees_with_enrollment() does: a confirmed sample that does not
# look like the person it was filed under is refused however large it
# is, and that is the guard that actually stops a misclick poisoning
# somebody's references.
FACE_LEARN_MIN_WIDTH = _env_int("FACE_LEARN_MIN_WIDTH", 26)

# ------------------------------------------- 5. BEST-FRAME SELECTION
# Recognition does NOT run on every frame of a track. It runs on the
# frames worth running it on.
#
# Each track keeps the best face it has produced so far. A new face
# replaces it only if it is better by BEST_FRAME_IMPROVEMENT, so an
# essentially identical frame does not trigger pointless GPU work.

# A face must reach this quality before it is even considered as a
# track's "best frame". Above FACE_QUALITY_THRESHOLD on purpose: the
# lower bar decides what may be recognised at all, this one decides what
# is good enough to base a CONFIRMED identity on.
BEST_FRAME_QUALITY_THRESHOLD = _env_float("BEST_FRAME_QUALITY_THRESHOLD", 0.58)

# How much better a face must be to displace the track's current best.
BEST_FRAME_IMPROVEMENT = _env_float("BEST_FRAME_IMPROVEMENT", 0.05)

# HOW LONG A BEST FRAME STAYS AUTHORITATIVE, in seconds.
#
# This one is not a tuning knob, it is a correctness guard. Without it
# the best frame is kept for the life of the track, and recognition
# keeps being run on a crop captured minutes ago. In the normal case
# that is merely stale; in the case that matters - a tracking error, two
# people swapping boxes - it is much worse than stale, because the
# system keeps re-confirming the ORIGINAL person from a cached picture
# of them and can never notice that the body it is following is somebody
# else. The wrong name then survives indefinitely, which is the exact
# failure the voting layer exists to prevent.
#
# Past this age the incumbent stops being defended: the next face good
# enough to be a best frame replaces it outright, however slightly
# worse it is.
BEST_FRAME_MAX_AGE_SECONDS = _env_float("BEST_FRAME_MAX_AGE_SECONDS", 5.0)

# Minimum gap, in processed frames, between two recognition passes on
# the SAME track. This is the setting that stops the GPU re-recognising
# a person who has been standing at the desk for two minutes.
RECOGNITION_INTERVAL = _env_int("RECOGNITION_INTERVAL", 8)

# Once a track's identity is CONFIRMED by the vote, recognition slows
# right down - we only keep checking at all so that a genuine tracking
# error can eventually be caught. This is where most of the GPU saving
# on a busy room comes from: a person standing at the desk for two
# minutes is looked at a handful of times, not two thousand.
#
# The saving is not paid for in correction time. The moment a look
# DISAGREES with the confirmed name, that track is marked contested and
# drops straight back to RECOGNITION_INTERVAL until the disagreement is
# settled one way or the other - so a genuine identity swap is resolved
# in seconds while a settled one still costs almost nothing.
RECOGNITION_INTERVAL_CONFIRMED = _env_int("RECOGNITION_INTERVAL_CONFIRMED", 45)

# A face that is a clear new best for the track jumps the interval
# queue: waiting eight frames to use the best look we have had all day
# is exactly the wrong trade.
BEST_FRAME_FORCES_RECOGNITION = _env_bool("BEST_FRAME_FORCES_RECOGNITION", True)

# Total number of aligned faces the whole face stage will embed in one
# pass, across all tracks. A hard ceiling on the GPU cost of one frame,
# so a sudden crowd cannot stall the pipeline.
FACE_MAX_RECOGNITIONS_PER_PASS = _env_int("FACE_MAX_RECOGNITIONS_PER_PASS", 8)

# ---------------------------------------------- 6. SIMILARITY SEARCH
# Cosine similarity needed to accept an identity (0-1, higher =
# stricter).
#
# *** THIS VALUE IS MODEL-SPECIFIC. *** AdaFace and ArcFace do not put
# genuine pairs in the same place, so the old 0.50 (tuned for glintr100)
# is meaningless for AdaFace. Measure it on your own people:
#
#     python tools/eval_recognition.py --dataset data/faces
#
# which prints the genuine/impostor distributions and the threshold that
# gives the false-positive rate you ask for. The default below is a
# conservative starting point for AdaFace ir101, not a measured answer
# for this site.
# The bar is PER MODEL, and the table below is the authority.
#
# Read this before changing a number in it. AdaFace and ArcFace put
# genuine pairs in completely different places, so one threshold cannot
# serve both - and the model that is CONFIGURED is not always the model
# that LOADS. When AdaFace weights are missing the system falls back to
# ArcFace, and applying AdaFace's much lower bar to ArcFace accepts
# matches ArcFace considers meaningless, which shows up as confident
# wrong names. core/face_pipeline.py therefore resolves the effective
# threshold from the model that actually loaded, not from
# FACE_EMBED_BACKEND, unless you pin one explicitly.
FACE_BACKEND_THRESHOLDS = {
    # AdaFace: MEASURED on this site's own 67 people with
    #     python tools/eval_recognition.py --dataset data/faces
    # (310 photos, 188 usable, 158 leave-one-out identification queries):
    #
    #   thresh  correct  WRONG  recall
    #     0.40      144      0   91.1%   <- the tool's recommendation
    #     0.50      140      0   88.6%
    #     0.55      132      0   83.5%   <- set here, deliberately
    #     0.60      111      0   70.3%
    #
    # READ THIS BEFORE CHANGING IT BACK OR PUSHING IT HIGHER.
    #
    # The measurement found NO wrong names at any threshold between 0.40
    # and 0.65 - precision was 100% throughout - so raising this number
    # buys nothing measurable and costs recall. It is set to 0.55 as a
    # deliberate operational choice: fewer names shown, more honest
    # Unknowns, on the judgement that a missing name is cheaper here than
    # a wrong one.
    #
    # WHAT IT COSTS, measured on live frames from these cameras the same
    # day (41 samples): face similarity runs min 0.30, median 0.45, max
    # 0.77 - so 0.55 leaves about 83% of what these cameras actually
    # produce below the bar. Expect a lot of Unknowns on the workspace
    # camera and a fuller review queue. That is the intended trade, not a
    # fault.
    #
    # AND WHAT IT DOES NOT FIX. The wrong names seen on this site were
    # not weak matches. Pavan Kumar landed on a woman at 0.56, Ayesha
    # Fatima on somebody with short hair at 0.80 - individually
    # confident, comfortably clear of the runner-up. No threshold in a
    # usable range separates those, which is why the vision model veto
    # (core/vlm_verify.py) exists and why the real fix is the reference
    # photographs: 30 of 67 people have only ONE usable photo, and 122 of
    # 310 were rejected as too side-on.
    #
    # To revert without editing code:  set ADAFACE_RECOGNITION_THRESHOLD=0.50
    "adaface": _env_float("ADAFACE_RECOGNITION_THRESHOLD", 0.55),
    # ArcFace/glintr100: measured on this site's own 64 people.
    # Genuine pairs sit at 0.66 and above (5th percentile); different
    # people reach 0.22 (95th percentile). Anything under 0.44 produced
    # wrong names in tools/eval_recognition.py, and 0.50 was the value
    # this system ran in production for good measured reasons.
    "arcface": _env_float("ARCFACE_RECOGNITION_THRESHOLD", 0.50),
}

FACE_RECOGNITION_THRESHOLD = _env_float(
    "FACE_RECOGNITION_THRESHOLD",
    FACE_BACKEND_THRESHOLDS.get(FACE_EMBED_BACKEND, 0.50))

# Did somebody pin the threshold by hand? If so it wins over the table
# above, even when the loaded model is not the configured one - an
# explicit instruction is never silently overridden.
FACE_RECOGNITION_THRESHOLD_PINNED = bool(
    os.environ.get("FACE_RECOGNITION_THRESHOLD"))

# ...AND the winner must beat the best DIFFERENT person by this much.
#
# Threshold alone was the original defect: only the single highest score
# was looked at, so 0.53-for-one-person and 0.51-for-another printed the
# first name with full confidence when the honest reading is "these two
# are indistinguishable in this frame". Several reference photos of the
# SAME person are corroboration, not rivals, so this costs nothing for
# well-enrolled people.
#
# This is the most effective single setting against wrong names, and it
# is a cheaper instrument than the threshold: it refuses only the cases
# that were a coin toss, rather than refusing everything below a line.
# Measured on this site with tools/eval_recognition.py, a margin of 0.10
# removed every wrong name even at thresholds low enough to produce a
# dozen of them without it.
FACE_MATCH_MARGIN = _env_float("FACE_MATCH_MARGIN", 0.25)

# HOW MULTIPLE REFERENCE IMAGES PER PERSON ARE COMBINED.
#   "max"       best single reference wins (most sensitive, most prone
#               to one bad enrollment photo matching everybody)
#   "centroid"  compare against the person's mean embedding (most
#               robust, loses genuinely different angles)
#   "topk"      mean of that person's best FACE_BANK_TOPK references
#   "blend"     default. FACE_BANK_CENTROID_WEIGHT of the centroid score
#               plus the rest from top-k. Keeps the robustness of the
#               centroid without discarding the profile shot that is the
#               only reference matching somebody turned away.
FACE_BANK_SCORING = _env_str("FACE_BANK_SCORING", "blend")
FACE_BANK_TOPK = _env_int("FACE_BANK_TOPK", 2)
FACE_BANK_CENTROID_WEIGHT = _env_float("FACE_BANK_CENTROID_WEIGHT", 0.35)

# ------------------------------------------------- 7. TEMPORAL VOTING
# No single frame is allowed to decide who somebody is.
#
# Each track accumulates (name, similarity, quality) observations. The
# winner is the name with the largest CONFIDENCE-WEIGHTED share of the
# recent window - weight = similarity x quality, so a sharp frontal
# 0.62 counts for far more than a blurry side-on 0.36. This is what
# turns the example
#     A 0.87, A 0.91, B 0.54, A 0.89, A 0.93   ->   A
# into an answer instead of four right answers and one wrong one.

# How many recent observations are kept per track.
VOTING_WINDOW = _env_int("VOTING_WINDOW", 7)

# The winner needs at least this many separate observations. One good
# frame is never enough, however confident it looks.
MIN_VOTES = _env_int("MIN_VOTES", 3)

# ...and at least this share of the total weight in the window, so a
# name that keeps changing never confirms.
VOTE_MIN_SHARE = _env_float("VOTE_MIN_SHARE", 0.60)

# ...and this much more weight than the runner-up name.
VOTE_MIN_MARGIN = _env_float("VOTE_MIN_MARGIN", 0.15)

# Observations older than this stop counting, so a person who leaves and
# a different person who inherits the track cannot pool their votes.
VOTE_MAX_AGE_SECONDS = _env_float("VOTE_MAX_AGE_SECONDS", 25.0)

# WHAT PROTECTS A CONFIRMED IDENTITY.
#
# Once a track is confirmed, a contradicting name must build up this
# multiple of the confirmed name's weight before it is allowed to take
# over. At 2.5 a single strong wrong frame cannot rename anybody; a
# genuine tracking error, where every subsequent frame says somebody
# else, still corrects itself within a few seconds.
VOTE_OVERRIDE_FACTOR = _env_float("VOTE_OVERRIDE_FACTOR", 2.5)

# ...and it needs at least this many observations of its own.
VOTE_OVERRIDE_MIN_VOTES = _env_int("VOTE_OVERRIDE_MIN_VOTES", 4)

# Below this similarity an observation is recorded as "Unknown" rather
# than as a weak vote for the closest person. Deliberately a little
# under FACE_RECOGNITION_THRESHOLD: scores in the gap are real evidence
# that nobody matched, and counting them as Unknown votes is what stops
# a track drifting onto a name it never really earned.
VOTE_UNKNOWN_BELOW = _env_float(
    "VOTE_UNKNOWN_BELOW",
    max(0.0, FACE_RECOGNITION_THRESHOLD - 0.04))

# ------------------------------------------- 8. SELF-TRAINING LIBRARY
# The system builds its own reference photographs from the cameras.
#
# WHY THIS IS THE HIGHEST-VALUE PART OF THE WHOLE PIPELINE.
# Enrollment photos are taken on a phone: well lit, close up, frontal.
# The cameras see something else entirely - 60 pixels of face, off
# angle, compressed, lit from one side. A model comparing the two is
# being asked to bridge that gap on its own, and that gap is where the
# accuracy goes. Measured on this site, live faces scored 0.25-0.31
# against phone-photo references while genuine matches need 0.50.
#
# A reference captured FROM THE CAMERA has none of that gap. It looks
# exactly like what the camera will see tomorrow. Twenty of them per
# person, taken at different angles, distances and times of day, is
# worth more than any model change or threshold tuning available here.
#
# So every time the system is sure who somebody is, it keeps the best
# picture of their face and files it under their name. When it is not
# sure, it asks - and the answer is filed the same way.
FACE_LIBRARY_ENABLED = _env_bool("FACE_LIBRARY_ENABLED", True)

# How many camera-captured references to collect per person. Once a
# person has this many, capture stops for them and the questions stop
# too - they are done.
FACE_LIBRARY_TARGET = _env_int("FACE_LIBRARY_TARGET", 25)

# Quality a face must reach before it is worth KEEPING as a reference.
# Higher than the bar for recognising from one: a poor frame is worth
# answering a question with, and not worth teaching from for ever.
FACE_LIBRARY_MIN_QUALITY = _env_float("FACE_LIBRARY_MIN_QUALITY", 0.60)

# HOW DIFFERENT A NEW SAMPLE MUST BE FROM THE ONES ALREADY HELD.
#
# This is what makes the library worth having. Twenty frames of
# somebody standing still are twenty copies of one reference - they add
# nothing, and they drag that person's average toward one pose. A new
# sample is only kept if it is at most this similar to every sample
# already stored for them, so the library fills up with genuinely
# different looks: turned left, turned right, closer, further, morning
# light, evening light.
FACE_LIBRARY_MAX_SIMILARITY = _env_float("FACE_LIBRARY_MAX_SIMILARITY", 0.92)

# Least time between two captures of the same person, in seconds. Even
# a moving person produces near-identical frames a second apart, and
# the similarity test alone would burn CPU rejecting them.
FACE_LIBRARY_MIN_INTERVAL = _env_float("FACE_LIBRARY_MIN_INTERVAL", 20.0)

# Capture automatically when the temporal vote has CONFIRMED an
# identity, without asking. Safe because a confirmed vote already
# required MIN_VOTES agreeing observations over the threshold - it is a
# far stronger statement than one frame's match. Turn off to make every
# single sample pass through a human first.
FACE_LIBRARY_AUTO_CAPTURE = _env_bool("FACE_LIBRARY_AUTO_CAPTURE", True)

# Padding around the face box when the picture is written to disk, as a
# fraction of the face size. NOT cosmetic: tools/enroll_faces.py re-runs
# face detection over these files, and a detector cannot find a face in
# a tight crop of one - measured, 0 of 25 tight crops were detectable.
# A human squinting at 48 pixels of cheek cannot identify it either.
FACE_LIBRARY_CROP_PAD = _env_float("FACE_LIBRARY_CROP_PAD", 0.55)

# --------------------------------------------------- asking for help
# When the system does NOT know somebody, it should say so and ask,
# rather than staying quiet. These control how often it does.

# Ask about a person the system cannot name at all, not only about ones
# it has a guess for. Previously a question was only ever raised when
# there was already a candidate, which meant the people most in need of
# reference photos - the ones nothing matches - were exactly the ones
# never asked about.
ASK_ABOUT_UNKNOWN = _env_bool("ASK_ABOUT_UNKNOWN", True)

# A track must have been watched this long, with a good enough face,
# before it is worth asking about. Stops somebody walking past the
# corner of the frame generating a question.
ASK_UNKNOWN_MIN_SECONDS = _env_float("ASK_UNKNOWN_MIN_SECONDS", 3.0)
ASK_UNKNOWN_MIN_QUALITY = _env_float("ASK_UNKNOWN_MIN_QUALITY",
                                     FACE_QUALITY_THRESHOLD)

# Once somebody's library is full (FACE_LIBRARY_TARGET) the system stops
# asking about them. Until then it may keep asking, at this interval,
# even if they were confirmed earlier today - that is the whole point:
# one confirmation gives one reference, and the goal is twenty.
#
# The OLD behaviour muted a person for the entire day after a single
# confirmation, which is why the queue went quiet and the library never
# grew.
ASK_AGAIN_SECONDS = _env_float("ASK_AGAIN_SECONDS", 240.0)

# ------------------------------------------------------- 9. DIAGNOSTICS
# Print a periodic summary explaining why faces are / are not being
# named: how many were seen, what they were rejected for, the quality
# histogram, and the best score that failed to match. This is what makes
# "it is not recognising anybody" an answerable question, and it is also
# the log the A/B comparison against the old pipeline reads.
FACE_DEBUG = _env_bool("FACE_DEBUG", True)
FACE_DEBUG_EVERY_SECONDS = _env_float("FACE_DEBUG_EVERY_SECONDS", 15.0)

# Write one CSV row per recognition attempt (track, quality, every
# sub-score, similarity, runner-up, the vote's verdict). This is the
# file tools/eval_recognition.py reads to compare pipelines on the same
# footage. Off by default - it is a few hundred KB an hour.
FACE_TRACE = _env_bool("FACE_TRACE", False)
FACE_TRACE_FILE = _env_str("FACE_TRACE_FILE",
                           os.path.join(DATA_DIR, "face_trace.csv"))

# A face must be at least this wide before we trust a match. Kept as an
# alias of MIN_FACE_SIZE so older tools and any local edits keep working.
FACE_MIN_WIDTH = MIN_FACE_SIZE
# Legacy alias: the old name for FACE_RECOGNITION_THRESHOLD.
FACE_MATCH_THRESHOLD = FACE_RECOGNITION_THRESHOLD

# A confirmed sample must still LOOK LIKE the person it is filed
# under, or we do not keep it.
#
# This exists because a human confirmation is not automatically true. If
# the system suggests the wrong person and somebody clicks confirm
# without really checking, that answer records the wrong face - and
# because a camera-realistic sample outweighs an enrollment photo, one
# rubber-stamped answer can poison a name permanently.
#
# Measured on this site's own 134 confirmations under the previous
# (ArcFace) model: genuine samples scored 0.40-0.48 against the person's
# enrollment photos, mis-confirmed ones 0.05-0.28. AdaFace puts these
# numbers somewhere else entirely, so re-measure with
#     python tools/eval_recognition.py --dataset data/faces
# after switching. The default below is scaled for AdaFace.
#
# A person with no enrollment photos has nothing to contradict, so their
# confirmations are always kept.
FACE_BACKEND_LEARN_AGREE = {
    "adaface": _env_float("ADAFACE_LEARN_AGREE_MIN", 0.24),
    "arcface": _env_float("ARCFACE_LEARN_AGREE_MIN", 0.32),
}
FACE_LEARN_AGREE_MIN = _env_float(
    "FACE_LEARN_AGREE_MIN",
    FACE_BACKEND_LEARN_AGREE.get(FACE_EMBED_BACKEND, 0.28))
FACE_LEARN_AGREE_MIN_PINNED = bool(os.environ.get("FACE_LEARN_AGREE_MIN"))

# Once a track is named, only replace the name if a later face scores at
# least this much better. Stops a single bad frame from renaming someone.
FACE_UPGRADE_MARGIN = _env_float("FACE_UPGRADE_MARGIN", 0.06)

# How much better a SECOND body's claim must be before it takes a name off
# the body already wearing it.
#
# A name belongs to one person at a time, so when two tracks claim it the
# better claim wins. But "better" used to mean better by any amount at
# all, so two bodies scoring 0.74 and 0.75 traded the name every few
# frames and the loser was blanked to "Unknown" each time - which is
# exactly what a correct name flickering away looks like from the
# dashboard. A name now only moves when the evidence is clearly better,
# not merely different.
NAME_STEAL_MARGIN = 0.10

# ------------------------------------------------------------ TRACKING
# How the same person is followed from frame to frame.
#
# WHICH ASSOCIATOR
#   "bytetrack"  (default) Kalman motion prediction plus ByteTrack's
#                two-stage association: high-confidence detections are
#                matched first, then the LOW-confidence ones - which are
#                usually a real person who is half occluded - are offered
#                to whatever is still unmatched. That second pass is the
#                whole point of ByteTrack, and it is exactly the case
#                this site has: somebody behind a monitor scores 0.2 and
#                used to be thrown away, killing the track and, with it,
#                the identity attached to it.
#   "geometry"   the previous IoU + centre-distance + appearance matcher,
#                kept so the two can be compared on the same footage.
#
# Either way the surrounding behaviour is unchanged: the same Track
# objects, the same appearance memory, the same ghost/re-identification
# layer, the same visit and presence bookkeeping.
TRACKER_BACKEND = _env_str("TRACKER_BACKEND", "bytetrack")

TRACK_IOU_THRESHOLD = _env_float("TRACK_IOU_THRESHOLD", 0.20)
                               # box-overlap match (geometry backend)
TRACK_MAX_DISTANCE = _env_int("TRACK_MAX_DISTANCE", 160)
                               # px, centre-distance match (fallback)
TRACK_MIN_HITS = _env_int("TRACK_MIN_HITS", 3)
                               # frames before a track is "real" (kills blips)
TRACK_MAX_AGE = _env_float("TRACK_MAX_AGE", 5.0)
                               # seconds a track survives while unseen.
                               # Generous on purpose: a track that dies and
                               # restarts is the main cause of DOUBLE COUNTS.

# --- ByteTrack ---------------------------------------------------------
# How many FRAMES a lost track is kept alive for re-association. This is
# ByteTrack's own memory and is separate from TRACK_MAX_AGE (seconds) and
# from REID_MEMORY_SECONDS (the appearance-based revival below). At ~10
# processed fps, 60 frames is about six seconds of occlusion tolerance.
TRACK_BUFFER = _env_int("TRACK_BUFFER", 60)

# Detections at or above this confidence go into the FIRST association
# pass; everything between BYTETRACK_LOW_THRESH and this goes into the
# second. Note the first pass threshold is deliberately well above
# PERSON_CONF - the gap between them is the "low" band that ByteTrack
# exists to exploit.
BYTETRACK_HIGH_THRESH = _env_float("BYTETRACK_HIGH_THRESH", 0.50)
BYTETRACK_LOW_THRESH = _env_float("BYTETRACK_LOW_THRESH", 0.10)

# A brand-new track is only started from a detection this confident, so
# the low band can keep existing people alive without inventing new ones.
BYTETRACK_NEW_TRACK_THRESH = _env_float("BYTETRACK_NEW_TRACK_THRESH", 0.55)

# IoU-distance ceilings for the two association passes (lower = stricter).
BYTETRACK_MATCH_THRESH = _env_float("BYTETRACK_MATCH_THRESH", 0.80)
BYTETRACK_SECOND_MATCH_THRESH = _env_float("BYTETRACK_SECOND_MATCH_THRESH", 0.50)

# Let BODY SIZE break ties in the first pass. Geometry still decides
# what is POSSIBLE; size only decides which of the possible pairings
# wins, which is what keeps two people who walk past each other from
# swapping tracks - and swapping identities with them.
BYTETRACK_USE_BODY = _env_bool("BYTETRACK_USE_BODY", True)
BYTETRACK_BODY_WEIGHT = _env_float("BYTETRACK_BODY_WEIGHT", 0.25)

# BODY MEASUREMENT (core/body.py)
#
# THIS REPLACED CLOTHING MATCHING, AND THAT WAS THE POINT.
#
# People used to be re-identified by the colour and pattern of what they
# were wearing. It failed in exactly the situation it was written for:
# while two people overlap, each one's crop contains the other, both
# signatures drift together, and when they separate the name follows the
# wrong body. Two colleagues in similar clothes were never separable
# either.
#
# What is compared now is how TALL somebody is (corrected for how far
# away they are, using a perspective model each camera learns from its
# own footage) and how BROAD they are for that height. Neither changes
# when a person turns round, and measurements taken while two people
# overlap are thrown away rather than learned.
BODY_ENABLED = _env_bool("BODY_ENABLED", True)

# How well the sizes must agree to REVIVE a person the tracker had given
# up on. The candidate set there is everybody who recently vanished from
# this camera, so the bar is the higher one.
BODY_REID_MIN = _env_float("BODY_REID_MIN", 0.55)

# ...and to reclaim somebody we are still holding but did not see this
# frame. Lower, because the candidate set is tiny - the two or three
# people who disappeared in the last few seconds.
BODY_RECLAIM_MIN = _env_float("BODY_RECLAIM_MIN", 0.45)

# THE ONE THAT STOPS A NAME MOVING TO THE WRONG PERSON.
#
# Before a name is put on a track, the track's measured size is checked
# against that person's remembered size. Below this the claim is
# refused: the face was read off somebody else, which is what happens
# when one face is visible between two overlapping people.
#
# Low on purpose. It is a "that is not the same size of human" test, not
# an identification - raising it starts refusing correct names when
# somebody is half behind a desk.
BODY_IDENTITY_MIN = _env_float("BODY_IDENTITY_MIN", 0.35)

# How much further a person may be from where we lost them, per second
# of absence, when their SIZE agrees. Somebody out of sight for ten
# seconds can be right across the room, and a fixed radius measured in
# body heights cannot express that - it refused a good match at 1223px
# simply because 1223 is a big number.
REID_DISTANCE_PER_SECOND = _env_float("REID_DISTANCE_PER_SECOND", 1.5)

# RE-IDENTIFICATION - the other half of the duplicate fix.
# When a track does die and a new one appears in almost the same place a
# moment later, it is almost always the same person (the detector
# blinked, or they were briefly hidden). We remember dead tracks for a
# few seconds and revive them instead of creating a new person.
REID_MEMORY_SECONDS = 30.0     # how long a dead track is remembered.
                               # Generous because people who SIT DOWN stop
                               # being detected reliably; a short memory
                               # makes them reappear as a brand new person
                               # and produces a second entry.
REID_MAX_DISTANCE_FACTOR = 1.4 # allowed gap, as a multiple of body height
REID_SIZE_RATIO = 2.0          # sizes must be within this ratio

# RECLAIMING A PERSON WHO WAS ONLY BRIEFLY LOST.
#
# The gap this closes was the single biggest cause of "he turned round
# and became Unknown". When somebody is occluded - by a colleague, a
# pillar, a monitor - ByteTrack drops them and, on their reappearance a
# second or two later, gives them a NEW id. The old track was still
# being held (it coasts for TRACK_MAX_AGE) but was not a ghost yet, so
# nothing tried to match against it: a brand-new, nameless track was
# created while the original sat invisible holding the name, the visit
# row and the accumulated presence time.
#
# So a new id is now offered to the tracks we are still holding but did
# not see this frame, BEFORE a new person is invented. That is a much
# safer question than "who is this out of everybody enrolled": the
# candidates are the two or three people who vanished from THIS camera
# in the last few seconds.
RECLAIM_COASTING = _env_bool("RECLAIM_COASTING", True)

# How long a coasting track stays reclaimable. Beyond this the ghost
# layer takes over, which is stricter.
# Measured on this site: a person walked out of view and returned after
# 11.2 seconds, 1223px away, looking 0.75 like their remembered
# appearance - and was refused, purely because the window was 8s. They
# came back as a new, nameless person. Twenty seconds covers somebody
# crossing a room, stepping out of shot and returning; beyond that the
# ghost layer's stricter test takes over.
RECLAIM_MAX_GAP_SECONDS = _env_float("RECLAIM_MAX_GAP_SECONDS", 20.0)

# How far somebody may have moved while out of sight, as a multiple of
# their height PER SECOND of absence. A person walks about one body
# height per second, so this is generous but not unbounded.
RECLAIM_DISTANCE_PER_SECOND = _env_float("RECLAIM_DISTANCE_PER_SECOND", 1.6)

# OCCLUSION: when two people overlap this much, neither one is measured
# and neither one's name can change.
#
# A box drawn around two overlapping people is taller than either of
# them and far wider. Recording that pulls a track's stature and build
# toward its neighbour's, which is how the WRONG person ends up keeping
# the identity after they separate. So while the overlap lasts: the
# boxes still track normally, the measurements are discarded, and no
# name is granted or moved. When they come apart, each of them is
# matched against what they measured BEFORE they met.
OCCLUSION_IOU = _env_float("OCCLUSION_IOU", 0.22)

# --------------------------------------------------------- ENTRY RULES
# What counts as "this person entered".
#
# The problem these solve: someone walking AWAY from the camera, or just
# passing across the lobby, must NOT be recorded as an arrival.
#
# A person is only counted once we have watched them long enough to be
# sure they came in:
#   * they must be observed for at least ENTRY_MIN_OBSERVATIONS frames
#   * they must get reasonably close (ENTRY_NEAR_HEIGHT_FRAC)
#   * they must be walking TOWARD the camera (ENTRY_REQUIRE_APPROACH)
# The entry is then recorded with their FIRST-SEEN time, so the arrival
# time stays correct even though the decision was made a moment later.
ENTRY_MIN_OBSERVATIONS = 15    # frames of evidence before deciding.
                               # At ~25 fps this is about 0.6s, enough for
                               # the approaching/receding verdict to mean
                               # something. (The recorded arrival time is
                               # still the person's first-seen time.)

# Body height as a fraction of frame height that means "close enough to
# have actually come in". People wandering in the far background never
# reach this, so they are never counted.
# Lower it if real arrivals are being missed; raise it if background
# people are still counted.
ENTRY_NEAR_HEIGHT_FRAC = 0.22

# Movement classification, comparing the person's size AND the row their
# feet are on now against when they were first seen. Growing and moving
# down the frame = walking toward the camera. Shrinking and moving up it
# = walking away. Both cues are used, because either one alone is wrong
# often enough to matter here - see core/tracker.Track.motion.
ENTRY_APPROACH_RATIO = 1.15    # >= this much bigger  -> approaching
ENTRY_RECEDE_RATIO = 0.87      # <= this much smaller -> walking away
ENTRY_BLOCK_RECEDING = True    # never count someone who is walking away

# ONLY PEOPLE COMING TOWARD THE CAMERA ARE ENTRIES.
#
# Blocking the obvious departures was not enough. Anybody whose size
# merely held steady - crossing the lobby, standing near the door,
# walking along the far wall - was still recorded as arriving, and on a
# camera pointed down a corridor that is most of the traffic.
#
# With this on, a track has to actually approach before it counts.
# Somebody who never does is shown on the stream as "passing" and
# tallied under PASSED BY, so they are visibly not-counted rather than
# silently dropped.
#
# Turn it off (per camera: "require_approach": False in config/cameras.py)
# on a camera where people arrive from the side and never grow.
ENTRY_REQUIRE_APPROACH = _env_bool("ENTRY_REQUIRE_APPROACH", True)

# Do not open a second entry for the same identified person on the same
# camera within this many seconds (catches any duplicate that slips past
# the tracker).
ENTRY_DEDUP_SECONDS = 90

# ------------------------------------------------- WORKSPACE / PRESENCE
# Settings for area cameras that measure HOW LONG people are present
# (as opposed to counting arrivals).
#
# People at desks are static and heavily occluded by monitors and chair
# backs, so detection is intermittent by nature. These values are tuned
# for that: a person is treated as continuously present through gaps,
# and only considered to have left after a long absence.
PRESENCE_GAP_TOLERANCE = 90.0     # s. Absence shorter than this does not
                                  # end someone's presence session.
PRESENCE_MIN_SECONDS = 20.0       # ignore blips shorter than this
PRESENCE_SAVE_EVERY = 15.0        # how often live totals are written
PRESENCE_MIN_HITS = 10            # frames before a person is "really there"

# SEAT-BASED IDENTIFICATION (core/seats.py)
# OFF unless you actually configure "seats" on a camera.
#
# Use with care: it assumes people sit in their own place. If anybody
# ever uses a colleague's desk it will confidently report the wrong
# person, which is worse than reporting Unknown. Sticky tracking (a name
# learned once and carried by the body) is the safer primary method.
SEAT_DWELL_SECONDS = 6.0          # must stay put this long before being
                                  # credited with a desk (so people
                                  # walking past are never mistaken for
                                  # its occupant)
SEAT_IDENTIFY_SCORE = 0.45        # deliberately LOW, so any real face
                                  # match outranks a seat guess.

# ------------------------------------------------------------ STREAMING
STREAM_ENABLED = True
STREAM_PORT = 8001             # pipeline serves annotated MJPEG here
JPEG_QUALITY = 70

# Shrink frames before JPEG-encoding them for the browser. Encoding a
# full 4MP frame 20 times a second is a surprisingly large CPU cost, and
# the dashboard tile is only ~700px wide anyway. This does NOT affect
# detection - only what the browser receives.
STREAM_MAX_WIDTH = 1280

SHOW_LOCAL_WINDOWS = False     # OpenCV preview windows on the pipeline PC

# ------------------------------------------------- VISION MODEL (VLM)
# A local vision-language model via Ollama, used as an INDEPENDENT SECOND
# OPINION - never in the real-time loop.
#
# Be clear about why: at 25 fps each frame has ~40 ms; a 7B vision model
# needs 1.5-5 s per image, so it is 20-100x too slow to detect or track.
# What it is genuinely good for is telling us how many people are really
# in a frame, which is the ground truth we otherwise had to guess at when
# tuning, plus occasional per-person checks that run at most once each.
VLM_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen2.5vl:7b")
VLM_TIMEOUT = _env_float("VLM_TIMEOUT", 120)   # vision models are not fast
VLM_MAX_IMAGE_WIDTH = _env_int("VLM_MAX_IMAGE_WIDTH", 1024)

# Keep the model resident in VRAM between questions.
#
# Not a micro-optimisation. Left to Ollama's default the vision model is
# unloaded a few minutes after each call, so the NEXT question pays to
# push 6GB of weights back onto a GPU that is already running YOLO,
# SCRFD and AdaFace - several seconds, every time, and a visible stall
# in the detector while it happens. Held resident, a question costs only
# its own inference.
#
# Set to "0" to unload immediately after each call if VRAM is tight.
VLM_KEEP_ALIVE = _env_str("VLM_KEEP_ALIVE", "30m")

# Hard cap on the reply length. Every one of these answers is a single
# short JSON object; without a cap the model narrates, and each token of
# narration is GPU time taken from the detector.
VLM_NUM_PREDICT = _env_int("VLM_NUM_PREDICT", 96)

# ---------------------------------------------- THE VLM AS AN ARBITER
# core/vlm_verify.py runs the vision model on ONE background thread and
# uses its answers as a VETO on two decisions the rest of the system
# gets wrong in ways nothing else can catch:
#
#   1. "that box is not a person"   - a reflection in the glass, a
#      person shown on a monitor, a coat over a chair. These become
#      tracks, then presence sessions, then somebody's attendance.
#
#   2. "that is not who you say it is" - the name on a body contradicted
#      by something plainly visible: a beard against a clean-shaven
#      reference photo, glasses against none, an obviously different
#      person. This is the one that survives a human confirmation,
#      because a confirmation pins a name to a TRACK and the track can
#      later be following somebody else.
#
# WHY ONLY A VETO. A 7B vision model looking at a 40-pixel CCTV face
# cannot tell two colleagues apart, and asked to name somebody it will
# invent an answer. It is never allowed to name anybody here. What it
# CAN see is a contradiction, and a contradiction is enough to refuse.
# Refusing produces an honest Unknown; accepting a wrong name produces a
# wrong attendance record.
#
# COST. One question at a time, at most one every VLM_MIN_CALL_GAP
# seconds, and each question is asked ONCE per track (or once per
# track+name). A busy room costs a handful of questions a minute, not
# one per frame. If Ollama is not running, every answer is None and the
# pipeline behaves exactly as it did before this existed.
VLM_ENABLED = _env_bool("VLM_ENABLED", True)

# Ask "is this really a person?" once per track.
VLM_VERIFY_TRACKS = _env_bool("VLM_VERIFY_TRACKS", True)
VLM_VERIFY_MIN_HITS = _env_int("VLM_VERIFY_MIN_HITS", 12)
# ...and only act on a NO this confident. Below it the answer is
# recorded and ignored, which is the right default for a veto: the cost
# of a wrong veto is a real person going uncounted.
VLM_PERSON_MIN_CONFIDENCE = _env_float("VLM_PERSON_MIN_CONFIDENCE", 0.70)
# A body box smaller than this (fraction of frame height) is not worth
# asking about - the model would be judging 30 pixels of blur.
VLM_PERSON_MIN_HEIGHT_FRAC = _env_float("VLM_PERSON_MIN_HEIGHT_FRAC", 0.08)

# Ask "is this really <name>?" once per (track, name).
#
# HOW THAT QUESTION IS ACTUALLY ASKED. Not as "is this the same
# person?" - measured on this site's own photographs, qwen2.5vl answers
# "same, 0.95" to a woman's CCTV crop against a bearded man's portraits.
# The model is asked only to DESCRIBE the live crop; that description is
# compared in code against a description of the person's own enrollment
# photographs, computed once and cached in VLM_APPEARANCE_FILE. See
# core/vlm.py for which attributes are trusted and why.
VLM_VERIFY_IDENTITY = _env_bool("VLM_VERIFY_IDENTITY", True)

# The confidence a contradiction has to reach before a name is refused.
# The scale is not arbitrary - it is set by how much the contradicting
# attributes are worth:
#     0.80   facial hair disagrees and nothing else   (NO veto by default)
#     0.95   hair length disagrees                    (veto)
#     0.99   apparent sex disagrees                   (veto)
# So the default refuses on the two attributes that repeat identically
# call after call, and not on the one that moves. Raise to 0.96 to
# refuse only on sex; lower to 0.75 to include facial hair, which is
# not recommended - it moves between "beard" and "moustache" on the
# same person.
VLM_IDENTITY_MIN_CONFIDENCE = _env_float("VLM_IDENTITY_MIN_CONFIDENCE", 0.90)

# Where each person's reference description is cached. Plain JSON on
# purpose: if the system refuses somebody's name, "what does it think
# this person looks like?" must be answerable by opening a file.
VLM_APPEARANCE_FILE = _env_str("VLM_APPEARANCE_FILE",
                               os.path.join(DATA_DIR, "vlm_appearance.json"))

# ------------------------------------ SCREENING THE CONFIRMATION QUEUE
# Never ask a human "who is this?" about a picture with no face in it.
#
# The queue was full of doors, floors, backs of heads and shoulders, and
# every guard upstream of it had already passed them - because every one
# of those guards is a NUMBER. A patch of door texture scores perfectly
# well on detector confidence, on sharpness, on contrast and on face
# quality; there is nothing in a number that says "this is a door". So
# the crop is now looked at before the question is asked.
#
# This is also the guard on LEARNING. All three of the poisoned
# reference photographs found on this site by tools/vlm_check.py came in
# through a confirmation, and none of them contained a face - so the
# answer a person gave was filed against a picture of the top of
# somebody's head, which then taught the recogniser that shape.
VLM_SCREEN_QUESTIONS = _env_bool("VLM_SCREEN_QUESTIONS", True)

# WHICH ANSWERS COUNT AS "do not ask about this".
#
# The model is asked to CLASSIFY the crop, not to judge it - "is there a
# face here?" was tried first and answered false to everything, real
# faces included. See SHOWS_PROMPT in core/vlm.py for that measurement.
# It returns one of: face, head_back, body, place.
#
# Only two of those are trusted, and the reason is measured rather than
# assumed:
#   place       no person at all - a wall, a door, a floor, a desk.
#               Correct on background patches. This is the "why is it
#               asking me who a door is" case.
#   body        a person, but no usable face - a seated colleague behind
#               a monitor, a shoulder, a torso. Correct on 4 of 4 review
#               body crops, and the important one, because the body crop
#               is what the review card was built around.
#
# head_back is deliberately NOT here. The model uses it as a dumping
# ground for anything it is unsure about: it applied it to people merely
# looking DOWN, and to a patch of floor. Trusting it would suppress most
# of a work-area camera's genuine questions.
VLM_REJECT_SHOWS = tuple(
    s.strip() for s in _env_str("VLM_REJECT_SHOWS", "place,body").split(",")
    if s.strip())

# Offer the reviewer a shortlist of who an unknown face could be.
#
# THE RECOGNISER RANKS AND THE VISION MODEL STRIKES OUT. The face bank
# proposes its nearest VLM_CANDIDATE_POOL people, and any of them whose
# own photographs contradict what is on screen is removed; the first
# VLM_SHORTLIST_MAX survivors are shown.
#
# Ranking by DESCRIPTION alone was tried first and measured: with all 64
# people described, 26 came out "male, short hair, beard" and 19
# "female, long hair, clean-shaven". Those attributes are stable enough
# to REFUSE a name but nowhere near specific enough to pick one, so the
# shortlist was either empty or the entire staff directory. The
# embeddings are what can separate two colleagues; the vision model is
# what can notice that the best embedding guess is a man when the
# picture is plainly a woman.
#
# It still NARROWS rather than identifies - a name survives only
# because nothing rules it out. Only the human answers.
VLM_SHORTLIST_MAX = _env_int("VLM_SHORTLIST_MAX", 6)
VLM_CANDIDATE_POOL = _env_int("VLM_CANDIDATE_POOL", 12)
# Do not ask about a face this small. Below roughly 40px there is
# nothing for a vision model to contradict, and its answer would be
# noise applied to a real name.
VLM_IDENTITY_MIN_FACE_WIDTH = _env_int("VLM_IDENTITY_MIN_FACE_WIDTH", 40)
# How many reference photographs go into the comparison picture.
VLM_IDENTITY_REFERENCES = _env_int("VLM_IDENTITY_REFERENCES", 3)
# Compare against the ENROLLMENT photographs only - the ones a human
# deliberately took - rather than against the system's own camera
# captures. The captures are exactly what a wrong name poisons, so
# checking a suspect name against them would be asking the mistake to
# audit itself. Same reasoning as agrees_with_enrollment().
VLM_IDENTITY_ENROLLMENT_ONLY = _env_bool("VLM_IDENTITY_ENROLLMENT_ONLY", True)

# Least time between two questions, in seconds. This is the only real
# throttle on GPU contention, so it is the number to raise first if the
# detector's FPS drops after turning this on.
VLM_MIN_CALL_GAP = _env_float("VLM_MIN_CALL_GAP", 2.0)
# How many questions may be waiting. Beyond this the oldest low-priority
# one is dropped rather than letting a backlog answer questions about
# people who left minutes ago.
VLM_QUEUE_MAX = _env_int("VLM_QUEUE_MAX", 24)
# How long a verdict is trusted before the same question may be asked
# again. Mostly relevant to identity: a track that keeps being renamed
# should keep being checked.
VLM_VERDICT_TTL = _env_float("VLM_VERDICT_TTL", 300.0)
# How often to retry reaching Ollama once it has been found missing.
VLM_RETRY_SECONDS = _env_float("VLM_RETRY_SECONDS", 120.0)

VLM_DEBUG = _env_bool("VLM_DEBUG", True)
VLM_DEBUG_EVERY_SECONDS = _env_float("VLM_DEBUG_EVERY_SECONDS", 60.0)

VLM_DESCRIBE_PEOPLE = _env_bool("VLM_DESCRIBE_PEOPLE", False)
                               # ask for a clothing description, so an
                               # unidentified person can be shown as
                               # "grey patterned top" not "Unknown #13"

# ================================================================
#            "WHAT IS HAPPENING ON THAT CAMERA RIGHT NOW?"
# ================================================================
# The vision model answering a QUESTION A PERSON ASKED, rather than
# arbitrating a decision the pipeline already made. See core/vlm_scene.py.
#
# WHAT IS DIFFERENT ABOUT THIS PATH, AND WHY IT IS SAFE.
#
# Everything else the VLM does here is a veto, because the failure it
# guards against is a wrong NAME on an attendance record. This path never
# touches the record: it answers a question on a chat page and writes
# nothing. So it is allowed to be descriptive - but it still may not
# name anybody. The division of labour is absolute:
#
#     WHO   comes from the tracker and the face pipeline. Always.
#     WHAT  comes from the vision model, per person crop.
#
# The model is shown one person at a time and asked what that person is
# DOING. It is never told a name, never asked for one, and its answer is
# joined to a name in code. A model that cannot tell two colleagues apart
# at 40 pixels of face can still tell typing from talking, and this is
# the only arrangement in which that is worth anything.
#
# COST. One question per visible person plus one for the whole frame, at
# roughly 1.5-5 s each, so a full look at a busy room takes tens of
# seconds. That is fine for a question somebody typed and unacceptable
# anywhere near the frame loop, which is why it lives behind a cache and
# a wall-clock budget and is never called by the pipeline.
VLM_SCENE_ENABLED = _env_bool("VLM_SCENE_ENABLED", True)

# How many people are described in one answer. The people the tracker is
# most sure about come first (identified, then longest present), so the
# cap drops the ones nobody asked about rather than a random few.
VLM_SCENE_MAX_PEOPLE = _env_int("VLM_SCENE_MAX_PEOPLE", 6)

# TOTAL wall-clock allowed for one question, in seconds. When it runs
# out the remaining people are reported as "not looked at" rather than
# being quietly dropped - a partial answer that says which part is
# missing is usable; one that silently omits three people is not.
VLM_SCENE_BUDGET_SECONDS = _env_float("VLM_SCENE_BUDGET_SECONDS", 45.0)

# Per-call timeout for these questions. Lower than VLM_TIMEOUT because
# somebody is sitting waiting for the reply.
VLM_SCENE_TIMEOUT = _env_float("VLM_SCENE_TIMEOUT", 30.0)

# How long an answer is reused. Two people asking "what is happening in
# the server room" within the cache window get the same look at the same
# frame instead of two 30-second GPU runs. Short enough that "right now"
# stays honest.
VLM_SCENE_CACHE_SECONDS = _env_float("VLM_SCENE_CACHE_SECONDS", 25.0)

# A snapshot older than this is not "right now" and is reported with its
# age rather than presented as current. Mostly this means the pipeline
# has stopped or the camera is offline.
VLM_SCENE_STALE_SECONDS = _env_float("VLM_SCENE_STALE_SECONDS", 90.0)

# Do not ask about a person smaller than this fraction of the frame
# height - the same reasoning as VLM_PERSON_MIN_HEIGHT_FRAC. Thirty
# pixels of blur produces a confident sentence about nothing.
VLM_SCENE_MIN_HEIGHT_FRAC = _env_float("VLM_SCENE_MIN_HEIGHT_FRAC", 0.06)

# Include the frame the answer was read off in the reply, as a data URI.
# The whole reason the deterministic handlers show their rows and the
# generated queries show their SQL is that an answer nobody can check is
# not worth having; a described scene is no different.
VLM_SCENE_RETURN_IMAGE = _env_bool("VLM_SCENE_RETURN_IMAGE", True)

# ------------------------------------------------- SCENE SNAPSHOTS
# The pipeline publishes an UNANNOTATED frame plus the boxes and names
# it has for that moment, so another process can ask questions about it.
#
# Unannotated on purpose. The MJPEG stream carries the drawn frame -
# boxes, names, headers - and asking a vision model what is happening in
# a picture with "Pavan Kumar 42m" written across somebody's chest
# invites it to read the label and repeat it back as an observation. The
# model must see what the camera saw.
SCENE_SNAPSHOT_ENABLED = _env_bool("SCENE_SNAPSHOT_ENABLED", True)

# Least time between two snapshots of one camera, in seconds. This is a
# JPEG encode on the pipeline thread, so it is throttled hard: nobody
# asks a question ten times a second.
SCENE_SNAPSHOT_INTERVAL = _env_float("SCENE_SNAPSHOT_INTERVAL", 2.0)

# Snapshots are shrunk to this width. Larger than STREAM_MAX_WIDTH
# because person crops are cut out of it - a 1280px frame leaves a
# seated colleague about 120px tall, which is the bottom of what a
# vision model can say anything useful about.
SCENE_SNAPSHOT_MAX_WIDTH = _env_int("SCENE_SNAPSHOT_MAX_WIDTH", 1600)
SCENE_SNAPSHOT_QUALITY = _env_int("SCENE_SNAPSHOT_QUALITY", 88)

# ------------------------------------------------- ADAPTIVE SCHEDULING
# Keep the system responsive when it cannot keep up.
#
# Detection cost grows with model size, input size and camera count, and
# a GPU under sustained load will also throttle. Rather than let the
# whole pipeline crawl, cameras that can tolerate it start detecting less
# often, and return to their configured rate once there is headroom
# again. A seated-workspace camera loses almost nothing from detecting 3
# times a second instead of 10; a reception camera is protected because
# you set its detect_every to 1.
ADAPTIVE_DETECTION = True
ADAPTIVE_TARGET_FPS = 8.0      # below this, start easing off
ADAPTIVE_GOOD_FPS = 16.0       # above this, start catching up again
ADAPTIVE_MAX_SKIP = 4          # never skip more than this many extra frames
ADAPTIVE_PATIENCE = 5.0        # seconds of sustained slowness before acting

# ------------------------------------------------- LEARNING & REVIEW
# The system learns what people look like as it watches them, and asks
# for help when it is unsure - the way a photo app asks "is this the same
# person?" rather than silently guessing.
LEARN_BODY_SIZE = True            # measure people we have identified,
                                  # so a name landing on the wrong-sized
                                  # body can be refused later
REVIEW_DIR = os.path.join(DATA_DIR, "review")
REVIEW_COOLDOWN = 120.0           # seconds between review items per person,
                                  # so one uncertain person cannot flood
                                  # the queue
REVIEW_MAX_PENDING = 300          # stop queueing beyond this
LEARN_POLL_SECONDS = 10.0         # how often the pipeline checks for new
                                  # confirmations made on the dashboard
REVIEW_NAME_COOLDOWN = 900        # do not ask about the same PERSON again
                                  # for this long, even if their track
                                  # broke and restarted under a new id

# How far back the pipeline looks for answers it has not applied yet.
# Must comfortably exceed LEARN_POLL_SECONDS; anything older has already
# been written to the database by the dashboard itself.
CONFIRM_APPLY_WINDOW = 900        # seconds

# The score a human answer is recorded with. Deliberately higher than
# anything the models can produce (they top out at 1.0 cosine but never
# reach it), so a later frame cannot quietly rename somebody a person
# has already identified.
CONFIRMED_SCORE = 1.0

# ------------------------------------------------- IDENTITY DECISIONS
# When is the gallery (core/gallery.py) sure enough to put a name on
# screen, when should it ask, and when should it stay quiet?
#
# These were hard-coded, and one of them mattered a lot: the "a face this
# good is proof on its own" bar sat at 0.55 while FACE_MATCH_THRESHOLD
# above - the documented CCTV bar - is 0.42. Every genuine match between
# those two numbers was therefore parked in the review queue instead of
# being accepted, which is why the queue kept suggesting the RIGHT person
# while the dashboard still said Unknown.
#
# The bars below have since been raised together to stop the opposite
# failure: names being shown for the WRONG person. The principle is that
# every accept now needs two things, not one - a high enough score AND a
# clear gap to the next best candidate. A close call is a question for a
# human, never a label on screen.
GALLERY_FACE_CERTAIN = FACE_MATCH_THRESHOLD   # face alone is enough here
                                 # (deliberately tied, never a fixed number:
                                 #  if this drifts ABOVE the face threshold
                                 #  every genuine match in between goes to
                                 #  review instead of the dashboard)

# Below this, a face is treated as evidence AGAINST this candidate.
# Raised from 0.22: with the accept bar at 0.50, a face scoring 0.25
# against somebody is not weak support for them, it is a look at their
# face that says no. Letting clothing outvote it is how the wrong name
# survived on people whose face WAS visible.
GALLERY_FACE_USELESS = 0.30

GALLERY_ACCEPT_PROBABILITY = 0.70   # name them (was 0.62)
GALLERY_REVIEW_PROBABILITY = 0.38   # ask a human - left low on purpose, so
                                    # everything no longer auto-accepted
                                    # becomes a question rather than silence
GALLERY_MIN_MARGIN = 0.18        # ...and only if clearly ahead of second place

# HOW BIG THE BODY VETO IS (core/gallery.py, core/body.py).
#
# A face match is refused outright when the body it is landing on is
# this far from the size we have measured that person to be. It is the
# last route by which a name could move to the wrong person: one face
# visible between two overlapping people gets read, and without this it
# would be attached to whichever body the pipeline picked.
#
# Deliberately low. This is "that is not the same size of human", not an
# identification - two colleagues of similar build score well above it,
# and they are meant to. Raise it only if names still land on visibly
# wrong people; lower it if correct names are being refused for somebody
# who is usually half behind a desk.
GALLERY_BODY_MIN = _env_float("GALLERY_BODY_MIN", 0.30)

# WHAT HAPPENED TO THE CLOTHING SETTINGS
#
# GALLERY_NO_FACE_MIN_MARGIN, GALLERY_NO_FACE_CAP,
# GALLERY_ALLOW_CLOTHING_ONLY, GALLERY_CLOTHING_MIN and
# GALLERY_CLOTHING_STRONG are gone, along with the colour matching they
# tuned. There is no longer any route by which a person can be named
# without a face, so there is nothing left for them to control - see
# core/body.py for what replaced it and why.

# ---------------------------------------------------------- PERFORMANCE
# Print a timing breakdown every few seconds so you can see exactly which
# stage is slow instead of guessing. Turn off once you are happy.
PROFILE = True
PROFILE_EVERY_SECONDS = 5.0

# ------------------------------------------------------------- WEB APP
WEB_HOST = "0.0.0.0"
WEB_PORT = 8000
WEB_SECRET = os.environ.get("WEB_SECRET", "change-this-secret-in-production")
SESSION_HOURS = 8

# ---------------------------------------------------------- ASSISTANT
# Ask the system questions in plain English.
#
#     "who is in the reception now?"
#     "what time did Deepthi first arrive today?"
#     "how many people are in cam2 and who are they?"
#
# HOW IT ANSWERS, AND WHY IT IS BUILT THIS WAY
# --------------------------------------------
# Two routes, tried in order, and the order is the whole design:
#
#   1. A DETERMINISTIC HANDLER. The questions people actually ask are a
#      small, knowable set - who is here, when did X arrive, summarise
#      this camera. Those are answered by real functions against
#      database/repository.py. They cannot hallucinate, they cannot
#      invent a name, and they return the same answer every time.
#
#   2. A GENERATED SQL QUERY, only for questions no handler covers. The
#      model writes SELECT, it is validated, and it runs against a
#      READ-ONLY connection.
#
# The temptation is to send everything to the model and let it write SQL.
# That is a bad trade here: this database is an attendance record. A
# confidently wrong answer about who was present is worse than "I don't
# know", and an 8-billion-parameter model writing unsupervised SQL over
# staff records is not something to put in front of anybody. So the
# model handles the long tail and never the common case.
CHAT_ENABLED = _env_bool("CHAT_ENABLED", True)

# Ollama, shared with the VLM settings above.
CHAT_HOST = _env_str("CHAT_HOST", os.environ.get("OLLAMA_HOST",
                                                 "http://localhost:11434"))

# The model that writes SQL. A CODE model is the right tool - qwen2.5
# coder is markedly better at SQL than general models of the same size,
# and faster than reaching for a 14B.
CHAT_SQL_MODEL = _env_str("CHAT_SQL_MODEL", "qwen2.5-coder:7b")

# The model that turns a result table into a sentence. Kept separate
# because it is a different job: this one needs to read well, not to be
# precise about syntax. Set to "" to skip it and get plain formatted
# output, which is faster and completely predictable.
# OFF by default, and that is a considered choice.
#
# This step exists only to turn a result table into a sentence. It is
# cosmetic, and measured here it was actively harmful: asked which
# camera had the most visits, the query returned reception correctly and
# llama3.1 then reported "the data does not provide enough information".
# It also added 30-60 seconds to a round trip whose query took under a
# second.
#
# With this empty the rows are formatted by code: instant, and it cannot
# contradict the data it was given. Set it to qwen3:14b if you would
# rather have prose and can accept the latency and the risk.
CHAT_REPLY_MODEL = _env_str("CHAT_REPLY_MODEL", "")

CHAT_TIMEOUT = _env_float("CHAT_TIMEOUT", 60.0)

# May the model write SQL at all? Turn off to restrict the assistant to
# its deterministic handlers - it will then say plainly when it cannot
# answer something rather than improvising.
CHAT_ALLOW_SQL = _env_bool("CHAT_ALLOW_SQL", True)

# Hard ceiling on rows returned by a generated query, injected into the
# SQL itself. Stops "list everything" turning into a megabyte of JSON.
CHAT_MAX_ROWS = _env_int("CHAT_MAX_ROWS", 200)

# Print the generated SQL to the console. On by default: a query written
# by a model against staff records should be visible to whoever runs the
# system, not hidden behind a friendly sentence.
CHAT_DEBUG = _env_bool("CHAT_DEBUG", True)

# ------------------------------------------------------------ TIME ZONE
# The database stores UTC. The dashboard shows local time using this.
# India Standard Time = 5.5
TZ_OFFSET_HOURS = 5.5

# ------------------------------------------------------------- LOGGING
VERBOSE = True                 # print events to the pipeline console
VERBOSE_IDENTITY = True        # every 30s, explain why one person is
                               # still unnamed - turn off once happy
