# Site_intelligence

## Create a Python environment

A virtual environment keeps these packages separate from the rest of your
system, so nothing else on the PC can break it.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Your prompt should now start with `(venv)`. **Any time you open a new
terminal, run the activate line again** before running project commands.

> If PowerShell blocks the script, run this once, then retry:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

---

## Install PyTorch (GPU) — do this first

**This step matters more than any other.** Installing PyTorch the plain
way gives you a CPU-only build. Everything will still *run* — just
without the GPU, at a few frames per second. Install the CUDA build
explicitly:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify immediately:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You need to see `True`. If it says `False`, do not continue — see
[Troubleshooting](#troubleshooting).

> **Remember this for later:** any time you install a package that
> depends on PyTorch, re-run the check above. Some packages quietly
> replace your CUDA build with the CPU one.

---

## Install everything else

```powershell
pip install -r requirements.txt
```

This installs the detector (Ultralytics), the ONNX runtime that face
detection and recognition run on, SciPy (ByteTrack's association step),
the database layer, and the web dashboard.

Now re-check PyTorch, because installing other packages can overwrite it:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Still `True`? Good. If it flipped to `False`, just re-run Step 3.

---

## Download the detection model

Download **yolov8n.pt** and place it in `data\models\`:

<https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt>

```powershell
# or from PowerShell:
curl.exe -L -o data\models\yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

You should end up with `data\models\yolov8n.pt` (about 6 MB).

> **If people at the far end of the lobby are missed later,** download
> `yolov8s.pt` the same way and point `DETECT_MODEL` in
> `config\settings.py` at it. It is slower but much better at small,
> distant people — which matters on this camera.

### Face models

Two are needed.

**SCRFD (face detection)** downloads itself the first time you run
enrollment, inside the InsightFace model packs (about 300 MB, one time,
needs internet).

**AdaFace (face recognition)** ships as a PyTorch checkpoint and is
converted once:

```powershell
python tools\export_adaface_onnx.py --ckpt adaface_ir101_webface12m.ckpt
```

Download `adaface_ir101_webface12m.ckpt` from the AdaFace project first.
The conversion is verified against PyTorch and refuses to keep a file
that does not match.

**Until AdaFace is installed the system falls back to the older ArcFace
model and says so loudly on startup.** Everything keeps working; the
accuracy improvement is simply not in effect. `tools\check_setup.py`
reports it.


## Check the setup

Before starting anything, let the system inspect itself:

```powershell
python tools\check_setup.py
```

It checks Python, every package, whether the GPU is visible, whether the
model file exists, whether photos are present, and prints your camera
list. Anything missing is listed with the exact command to fix it.

Fix anything marked **FAIL** before continuing. **WARN** items are fine
for now (they clear as you finish the remaining steps).

---

## Create the database and login

```powershell
python tools\init_database.py --username admin --password "ChooseAStrongPassword#1"
(venv) PS D:\HHCL CCTV access\site_intelligence> python tools\init_database.py --username admin --password Hetero@1
```

This creates `data\site.db` and your dashboard login. Use a real
password — the dashboard is reachable from other machines on your
network.

---

##  Enroll the faces

```powershell
python tools\enroll_faces.py
```

You will see a line per employee:

```
  [    ok]    12219  Rajesh Palamangalam          3 photo(s)
  [    ok]    13381  Eswar Kumar                  3 photo(s)

[ENROLL] 6 embeddings for 2 people
[ENROLL] saved -> data\face_encodings.pkl
```

- `ok` — enrolled properly.
- `thin` — only one usable photo; add more angles.
- `FAILED` — no face was found; replace those photos.

Re-run this whenever you add or remove an employee.

---

## Start the system

You need **two terminals**, both with `(venv)` activated.

**Terminal 1 — the vision pipeline:**

```powershell
cd D:\site_intelligence
.\venv\Scripts\Activate.ps1
python run_pipeline.py
```

Expect to see:

```
[FACE] configuration -----------------------------------
[FACE]   detector      SCRFD @ 640px, conf>=0.45, crops=on@320px x8
[FACE]   recognition   adaface, threshold 0.34, margin 0.05, bank scoring 'blend'
[FACE]   quality       >= 0.45, min face 56px, best-frame >= 0.58
[FACE]   scheduling    every 3 frame(s), recognise 1/8 per track (1/45 once confirmed), max 8/pass
[FACE]   voting        window 7, min votes 3, share 0.6, override x2.5
[FACE]   tracking      bytetrack, buffer 60 frames
[FACE] -------------------------------------------------
[DETECT] loading yolov8n.pt (device=0, half=True, imgsz=640)
[DETECT] warmup done
[SCRFD] scrfd_10g_bnkps.onnx (GPU via CUDAExecutionProvider) - landmarks yes
[EMBED] adaface:adaface_ir101_webface12m.onnx (GPU via CUDAExecutionProvider)
[GALLERY] 2 people loaded from enrollment (6 face samples)
[GRABBER:Reception Lobby] connected
[STREAM] video server on port 8001
[PIPELINE] ready with 1 camera(s). Ctrl+C to stop.
```

Then, as people arrive:

```
[Reception Lobby] ENTRY  track #1  Unknown #1
[Reception Lobby] track #1 identified as Rajesh Palamangalam (0.61) - visit updated
[Reception Lobby] LEFT   track #1  Rajesh Palamangalam after 95s
```

Those three lines are the whole design in action: logged on arrival,
named when close enough, closed on leaving.

**Terminal 2 — the dashboard:**

```powershell
cd D:\site_intelligence
.\venv\Scripts\Activate.ps1
python run_dashboard.py
```

Open <http://localhost:8000> and sign in.

To stop either one, press **Ctrl+C** in its terminal.

---

## Reading the dashboard

**Top row (KPI cards)**

| Card | Meaning |
|---|---|
| In view now | People on camera at this moment |
| Total entries | Arrivals in the selected time window |
| Identified | Entries matched to an enrolled employee |
| Unidentified | Entries where no clear face was ever captured |

**Live cameras** — the video with boxes drawn on:

| Box colour | Meaning |
|---|---|
| Grey | Just appeared, not yet confirmed (ignored on purpose) |
| Amber | Confirmed person, not yet identified |
| Green | Face recognised — name shown |

A person normally goes grey → amber → green as they walk toward the
camera. That colour change *is* the late-identification working.

**Currently in view** — who is on camera right now and for how long.

**Entry log** — every arrival, newest first. When a name is filled in
retroactively, the row briefly flashes green. Note that "Arrived" always
shows the true first-seen time, even when the name arrived seconds later.

The **camera** and **time window** dropdowns at the top filter everything.

---

## Asking the system questions

The **Assistant** page (`/chat`) answers questions about the cameras in
plain English. Type a question, or tap one of the chips above the box —
those change after every answer to follow whatever you just asked about.

Every answer says which route produced it, and shows its working: a
table of the rows it used, the query if one was generated, and — for the
live questions below — the actual frame it was read off.

### Questions answered from the record

These come from real functions over the database. They are instant, and
they cannot invent a name.

| Ask | You get |
|---|---|
| Who is present right now? | Everybody in view, on both cameras |
| Is Akhila here right now? | Yes/no, which camera, how long |
| What time did Deepthi first enter today? | Her arrival, from the visit record |
| How long has Akhila been present today? | Her total presence time |
| Who has not come in today? | Enrolled people not seen |
| How many people entered today? | The entry count |
| What is the last one hour summary of cam2? | Totals for that window |

### Questions answered by LOOKING at the camera

Three questions are not in the database — there is no column for what
somebody is doing — so they are answered by taking the live frame and
asking the vision model about each person in it.

| Ask | You get |
|---|---|
| What is happening in Server Rm Psg? | A description of the room, then each person and what they are doing |
| What is Pavan Kumar doing? | That one person's activity, and where |
| Pavan Ayeesha what are they doing? | Both of them (several names in one question is fine) |
| Who is working and who is not? | Split into working / not working / cannot say |

**The names never come from the vision model.** Who somebody is comes
from the tracker and the face pipeline exactly as it does everywhere
else; the model is shown one anonymous person at a time and asked only
what that person is doing. A model that cannot tell two colleagues apart
at 40 pixels of face can still tell typing from walking, and that split
is what makes these answers safe to read.

Activities are one of: **working** (at a desk and engaged with it),
**on the phone**, **talking**, **moving around**, **idle** (at a desk but
not working), or **not clear enough to tell**. That last one is a real
answer, not a failure — CCTV crops are often genuinely unreadable, and
"cannot say" is listed separately from "not working" for that reason.

**What to expect:**

- The first such question takes **10–40 seconds** — one call per visible
  person. Follow-up questions about the same room are instant for the
  next 25 seconds (`VLM_SCENE_CACHE_SECONDS`), because they reuse the
  same look.
- It needs **both** the pipeline running (for the frame) and Ollama with
  a vision model (for the answer). If either is down you get the
  deterministic half — who is there, for how long — and a line saying
  what is missing. Check with `/api/chat/health`.
- Naming a camera makes it faster and the answer narrower. "What's
  happening in server_psg_cam" works — the camera does not have to be
  spelled the way `config\cameras.py` spells it.

**Settings** (all in `config\settings.py`):

| Setting | Default | What it does |
|---|---|---|
| `VLM_SCENE_ENABLED` | on | Turn the live questions off entirely |
| `VLM_SCENE_MAX_PEOPLE` | 6 | People described per camera |
| `VLM_SCENE_BUDGET_SECONDS` | 45 | Total wait allowed for one question |
| `VLM_SCENE_CACHE_SECONDS` | 25 | How long an answer is reused |
| `VLM_SCENE_RETURN_IMAGE` | on | Show the frame the answer came from |
| `SCENE_SNAPSHOT_INTERVAL` | 2.0 | How often the pipeline publishes a frame |

Check the routing without needing any of it up:

```powershell
python tools\test_chat.py
python tools\test_chat.py --ask "what is happening in server_psg_cam"
```
