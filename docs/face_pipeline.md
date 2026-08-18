# The face recognition pipeline

SCRFD → alignment → quality filtering → best-frame selection → AdaFace →
vector search → confidence-weighted temporal voting.

This document covers the recognition stack only. Everything else — RTSP
handling, person detection, the dashboard, the database, entry counting,
presence sessions, the review queue — is unchanged.

---

## 1. What runs, in order

```
RTSP camera                       core/frame_grabber.py
      ↓
person detection (YOLO)           core/detector.py        batched across cameras
      ↓
ByteTrack                         core/bytetrack.py       stable track ids
      ↓                           core/tracker.py         + size-based re-ID
SCRFD face detection              core/face_detect.py     on head crops, batched
      ↓
5-point alignment                 core/face_align.py      112×112 ArcFace template
      ↓
face quality filtering            core/face_quality.py    ← most faces stop here
      ↓
best-quality frame selection      core/face_identity.py
      ↓
AdaFace embedding                 core/face_embed.py      batched, flip-averaged
      ↓
vector similarity search          core/facebank.py        multi-reference
      ↓
confidence-weighted voting        core/face_identity.py
      ↓
final identity                    cameras/base.py         → visit / presence row
```

`core/face_pipeline.py` drives all of it on one background thread per
camera. `core/face.py` is the only module the rest of the project talks
to.

---

## 2. Why each stage is there

**SCRFD instead of the InsightFace wrapper.** The wrapper aligned faces
internally and returned only a box and an embedding. Keeping the five
landmarks ourselves is what makes alignment, pose measurement and
quality filtering possible at all — they all need those points. Driving
the graph directly also allows per-camera detector sizes and batched
detection over head crops.

**Head crops rather than one full-frame pass.** A person 200 px tall in
a 1280 px frame has a ~45 px face. After the detector shrinks the frame
to 640 that face is ~22 px and is simply missed. Cropping their head
region and detecting *that* makes the same face ~180 px, and the
landmarks are far more accurate — which matters more than the detection
itself, because bad landmarks mean bad alignment, and bad alignment is
the largest single source of recognition error. Crops are batched into
one GPU call. Set `FACE_DETECT_ON_PERSON_CROPS=0` to go back to one
full-frame pass.

**Alignment is not optional.** AdaFace, like every ArcFace-family model,
expects the eyes at (38.3, 51.7) and (73.5, 51.5) in a 112×112 grid.
Give it an unaligned crop and it still returns a confident 512-number
answer — about a different geometry. A *similarity* transform is used
rather than an affine one, so a profile face stays a profile and the
quality filter gets an honest picture to reject.

**Quality filtering is the false-positive fix.** A recognition model
never says "I cannot see this well enough"; it returns a confident wrong
answer. Refusing to ask the question is the only reliable remedy. Faces
are rejected for being too small, blurry, occluded, side-on or badly lit
before any embedding is computed.

**AdaFace instead of ArcFace/glintr100.** ArcFace applies the same
angular margin to every training sample, so it learns to answer
confidently from whatever it can see. AdaFace makes the margin a
function of image quality, so poor faces land nearer the middle of the
space — lower similarity to *everybody* — instead of confidently near
somebody. On CCTV, where most faces are poor, that is exactly the
failure mode being removed.

That is the argument, not the evidence. See §6.

**Voting instead of per-frame decisions.** Identity switching is not a
model problem, it is a decision-from-one-sample problem. Every
recognition is a vote weighted by similarity × quality; a name is
assigned only when it holds enough of the recent window, and once
assigned it is defended.

---

## 3. Configuration

Everything below lives in `config/settings.py` and can be overridden by
an environment variable of the same name:

```
set FACE_RECOGNITION_THRESHOLD=0.38 && python run_pipeline.py
```

The resolved values are printed at startup under `[FACE] configuration`,
so any run can be reproduced from its own log.

| Setting | Default | What it does |
|---|---|---|
| `FACE_DETECTION_THRESHOLD` | 0.45 | SCRFD confidence floor. Low on purpose — quality decides, not the detector |
| `FACE_RECOGNITION_THRESHOLD` | 0.34 | cosine similarity needed to accept a name. **Model-specific — measure it** |
| `FACE_MATCH_MARGIN` | 0.05 | …and the winner must beat the best *different* person by this |
| `FACE_QUALITY_THRESHOLD` | 0.45 | overall quality bar for recognising a face at all |
| `MIN_FACE_SIZE` | 56 | minimum face width in pixels, in the original frame |
| `TRACK_BUFFER` | 60 | frames a lost ByteTrack track is kept for re-association |
| `RECOGNITION_INTERVAL` | 8 | minimum face passes between recognitions of the same track |
| `VOTING_WINDOW` | 7 | observations kept per track |
| `MIN_VOTES` | 3 | separate observations needed before a name is confirmed |
| `BEST_FRAME_QUALITY_THRESHOLD` | 0.58 | quality needed to become a track's best frame |

Supporting settings worth knowing about:

| Setting | Default | What it does |
|---|---|---|
| `FACE_EMBED_BACKEND` | `adaface` | `adaface` or `arcface` (baseline only) |
| `FACE_DET_SIZE` / `FACE_CROP_DET_SIZE` | 640 / 320 | detector input size, full frame / head crop |
| `FACE_EVERY_N_FRAMES` | 3 | how often the face stage is handed a frame |
| `RECOGNITION_INTERVAL_CONFIRMED` | 45 | interval once a track is confirmed |
| `BEST_FRAME_MAX_AGE_SECONDS` | 5.0 | when a held best frame stops being authoritative |
| `FACE_MAX_RECOGNITIONS_PER_PASS` | 8 | hard ceiling on embeddings per pass |
| `VOTE_MIN_SHARE` / `VOTE_MIN_MARGIN` | 0.60 / 0.15 | share of window weight needed to confirm |
| `VOTE_OVERRIDE_FACTOR` | 2.5 | how much stronger a challenger must be to rename a confirmed track |
| `FACE_BANK_SCORING` | `blend` | `max` / `centroid` / `topk` / `blend` |
| `TRACKER_BACKEND` | `bytetrack` | or `geometry` for the previous matcher |
| `FACE_TRACE` | off | one CSV row per recognition, for offline comparison |

---

## 4. Installing AdaFace

AdaFace ships as a PyTorch checkpoint, so it is converted once:

```
python tools\export_adaface_onnx.py --ckpt adaface_ir101_webface12m.ckpt
python tools\enroll_faces.py
```

The export is verified numerically against PyTorch and refuses to keep a
file that does not match. Recommended checkpoints:

* `adaface_ir101_webface12m` — best accuracy, largest and most varied
  training set, which is what matters for poor-quality faces
* `adaface_ir50_ms1mv2` — about half the cost, slightly weaker

**Until AdaFace weights are present the system falls back to the old
ArcFace model and says so loudly.** It keeps running; the accuracy
upgrade is simply not in effect. `python tools\check_setup.py` reports
this.

SCRFD needs no download — the InsightFace model packs already on the
machine contain it (`scrfd_10g_bnkps.onnx`, `det_10g.onnx`).

---

## 5. Re-enrolling

**Changing the recognition model requires re-enrolling everyone.**
AdaFace and ArcFace produce equally valid 512-number descriptions of the
same face that have nothing to do with each other. The enrollment file
records which model made it and the pipeline refuses to load it under a
different one, so a mismatch fails loudly instead of silently matching
nobody.

```
python tools\enroll_faces.py
```

Every usable photo becomes a separate reference. Near-profile photos are
skipped by default — the cameras will never accept a profile face
(`FACE_POSE_YAW_MIN`), so a profile reference can never be matched and
only pulls the person's average away from the frontal views that will
be. `--allow-profiles` overrides this.

---

## 5b. The self-training library

The system builds its own reference photographs, from its own cameras.

**Why this matters more than anything else here.** Enrollment photos are
taken on a phone: well lit, close, frontal. The cameras see 30-70 pixels
of face, off angle, compressed, lit from one side. Measured on this
site, live faces score 0.25-0.31 against phone references when a genuine
match needs 0.50. A reference captured *from the camera* has no such gap
to bridge, so twenty of them per person is worth more than any model or
threshold change available here.

Two ways a reference is earned, both ending in the person's own folder:

| | |
|---|---|
| **Automatically** | when the temporal vote **confirms** an identity — several good frames agreeing, over threshold, clear of everyone else. Not one frame's guess. |
| **By confirmation** | when you answer a question on the dashboard, the crop you answered about is filed too. |

```
data/faces/12219-Rajesh Palamangalam/
    Photo_1.jpg                          <- the phone photo
    cctv_reception_20260811_143207_412.jpg    <- captured automatically
    cctv_server_rm_psg_confirmed_1786419731_5_face.jpg   <- from your answer
```

`tools/enroll_faces.py` picks them up like any other photograph, and you
can open the folder and delete anything wrong — which is the point of
using files rather than an opaque table.

**Samples are filtered, not hoarded.** Twenty frames of somebody
standing still are twenty copies of one reference; they teach nothing
and drag that person's average toward a single pose. A new sample must
be at most `FACE_LIBRARY_MAX_SIMILARITY` (0.92) similar to every sample
already held, so what accumulates is genuinely different looks.

| Setting | Default | |
|---|---|---|
| `FACE_LIBRARY_ENABLED` | on | the whole feature |
| `FACE_LIBRARY_TARGET` | 25 | photos per person; capture and questions stop there |
| `FACE_LIBRARY_MIN_QUALITY` | 0.60 | quality needed to be worth keeping |
| `FACE_LIBRARY_MAX_SIMILARITY` | 0.92 | how different a new sample must be |
| `FACE_LIBRARY_AUTO_CAPTURE` | on | off = every sample passes a human first |
| `ASK_ABOUT_UNKNOWN` | on | ask "who is this?", not only "is this X?" |
| `ASK_AGAIN_SECONDS` | 240 | keep asking until their library is full |

### Asking for help

Three separate faults used to stop the queue asking anything:

1. one confirmation **muted that person for the whole day**, so nobody
   ever accumulated more than one camera photo;
2. a question was only raised when there was already a *candidate*, so
   the people with no usable reference — exactly the ones who need
   photographs — were never asked about;
3. when the gallery scored **nobody at all**, the code returned before
   either branch, which is the most valuable case of all.

All three are fixed. The queue now asks openly ("Who is this?", with a
name box) and keeps asking about a person until their library is full.

---

## 6. Measuring it — do not assume the model improved anything

```
python tools\eval_recognition.py --dataset data\faces --compare
```

Runs **both** backends over the same photographs through the same
detection, alignment and quality stages, and reports:

* **Verification** — how far apart genuine pairs and impostor pairs sit.
  That gap *is* the accuracy; the threshold only decides where in it you
  stand.
* **Identification** — each photo held out, the face bank built from the
  rest, searched exactly as a live frame would be. This reports the
  number that matters: how often somebody gets the **wrong name**.
* **A threshold sweep**, with a recommended
  `FACE_RECOGNITION_THRESHOLD` / `FACE_MATCH_MARGIN` chosen to produce
  no wrong names.

It also reports how many people have only one usable reference photo.
That number is usually the real limit on accuracy, and no threshold or
model change fixes it — more photographs do.

For the CCTV half, which enrollment photos cannot tell you about:

```
set FACE_TRACE=1
:: run each pipeline over the same recording, then
python tools\eval_recognition.py --trace data\trace_adaface.csv data\trace_arcface.csv
```

which reports quality distributions, acceptance and confirmation rates,
and the count of identity changes after a name was assigned — the
identity-stability number the voting layer exists to drive to zero.

---

## 7. Reading the console

Every `FACE_DEBUG_EVERY_SECONDS` the face stage explains itself:

```
[FACE:Reception Lobby] 214 face(s) in 96 pass(es) | 138 rejected on quality |
                       19 recognised | 14 matched, 5 no match (best 0.31 vs 0.34)
[FACE:Reception Lobby]   quality: 0.1:12% 0.2:28% 0.3:19% 0.5:14% 0.6:18% 0.7:9%
[FACE:Reception Lobby]   rejected for: too small x71, blurry x38, too side-on x21
```

How to read it:

* **almost everything rejected** → the cameras cannot support the
  current bar. Lower `FACE_QUALITY_THRESHOLD` or `MIN_FACE_SIZE`, or
  accept that people are only named closer to the camera.
* **recognised but nothing matched, best score near the threshold** →
  lower `FACE_RECOGNITION_THRESHOLD`.
* **recognised but nothing matched, best score far below** → the
  enrollment was almost certainly made with a different model. Re-run
  `tools\enroll_faces.py`.
* **"too close to call" for one particular pair, repeatedly** → those
  two people need better enrollment photos. That is the real fix, and
  the margin gate is what makes the problem visible instead of turning
  it into a wrong name on the dashboard.

---

## 8. Performance

* One SCRFD and one AdaFace instance per process, shared by every
  camera — four cameras do not mean four copies on the GPU.
* Head crops and face embeddings are batched.
* Work is scheduled **per track, not per frame**: a confirmed person is
  looked at every `RECOGNITION_INTERVAL_CONFIRMED` passes rather than
  constantly, so the cost of a busy room is roughly the cost of the
  people who are *new* in it.
* `FACE_MAX_RECOGNITIONS_PER_PASS` caps the GPU cost of any single pass,
  so a sudden crowd delays some namings instead of stalling the
  pipeline. When the cap bites it is reported, not hidden.
* The face stage never blocks the video loop. A pending frame is
  *discarded*, never queued — a face result about a frame from four
  seconds ago is not late, it is wrong.
* TensorRT is used when its provider actually loads. The provider ladder
  in `core/gpu.py` steps down TensorRT → CUDA → CPU one rung at a time,
  because onnxruntime abandons the whole provider list when TensorRT
  fails to load, and would otherwise drop face recognition onto the CPU
  while reporting success.

Measure the stages on your own footage:

```
python tools\benchmark.py
```

---

## 9. What did not change

The person detector, RTSP handling, the tracker's public contract, the
entry counting, presence
sessions, the review queue, the learning loop, the dashboard, the MJPEG
stream, the database schema and every API. `core/face.py` keeps its
`load_encodings` / `agrees_with_enrollment` / `FaceRecognizer` surface;
`PersonTracker.update()` keeps returning `(live, finished)` with the
same `Track` objects.
