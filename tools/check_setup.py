"""
tools/check_setup.py
====================
Run this FIRST, and any time something misbehaves. It checks every part
of the installation and tells you exactly what is missing.

    python tools/check_setup.py
"""
import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, WARN, BAD = "  OK  ", " WARN ", " FAIL "
problems = []


def line(status, label, detail=""):
    print(f"[{status}] {label:<34} {detail}")


def check_package(mod, label=None, required=True, hint=""):
    label = label or mod
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "")
        line(OK, label, v)
        return True
    except Exception:
        line(BAD if required else WARN, label, hint)
        if required:
            problems.append(f"{label} missing - {hint}")
        return False


print("\n=== Python ===")
line(OK if sys.version_info >= (3, 9) else BAD, "Python version",
     sys.version.split()[0])

print("\n=== Core packages ===")
check_package("cv2", "opencv-python", hint="pip install opencv-python")
check_package("numpy", hint="pip install numpy")
check_package("ultralytics", hint="pip install ultralytics")
check_package("sqlalchemy", hint="pip install sqlalchemy")
check_package("fastapi", hint="pip install fastapi")
check_package("uvicorn", hint="pip install uvicorn[standard]")
check_package("jinja2", hint="pip install jinja2")
check_package("itsdangerous", hint="pip install itsdangerous")
check_package("multipart", "python-multipart",
              hint="pip install python-multipart")

print("\n=== Face recognition ===")
check_package("onnxruntime", required=True,
              hint="pip install onnxruntime-gpu")
check_package("scipy", required=False,
              hint="pip install scipy  (ByteTrack falls back to a greedy "
                   "assignment without it)")
check_package("insightface", required=False,
              hint="only needed for the old ArcFace baseline in "
                   "tools/eval_recognition.py")

# The two models the new pipeline actually runs. Neither goes through
# InsightFace's wrapper any more, so what matters is whether the ONNX
# files are findable - not whether a package is installed.
try:
    from core.face_detect import find_model as find_scrfd
    from core.face_embed import find_adaface_model, find_arcface_model
    from config.settings import SCRFD_MODEL, ADAFACE_MODEL, FACE_EMBED_BACKEND

    scrfd = find_scrfd(SCRFD_MODEL)
    if scrfd:
        line(OK, "SCRFD face detector", os.path.basename(scrfd))
    else:
        line(BAD, "SCRFD face detector", "not found")
        problems.append(
            "No SCRFD detector. Put scrfd_10g_bnkps.onnx or det_10g.onnx\n"
            "      into data/models/ - both ship inside the InsightFace\n"
            "      model packs (~/.insightface/models/).")

    adaface = find_adaface_model(ADAFACE_MODEL)
    if adaface:
        line(OK, "AdaFace recognition", os.path.basename(adaface))
    elif FACE_EMBED_BACKEND == "adaface":
        line(WARN, "AdaFace recognition", "not found - see the hint below")
        problems.append(
            "AdaFace weights missing, so recognition falls back to the OLD\n"
            "      ArcFace model and the accuracy upgrade is NOT in effect.\n"
            "      Download an AdaFace checkpoint, then:\n"
            "        python tools/export_adaface_onnx.py --ckpt <file.ckpt>\n"
            "        python tools/enroll_faces.py")
    else:
        line(OK, "AdaFace recognition", "not required (backend=arcface)")

    arcface = find_arcface_model()
    line(OK if arcface else WARN, "ArcFace baseline (optional)",
         os.path.basename(arcface) if arcface
         else "not found - tools/eval_recognition.py --compare needs it")
except Exception as e:
    line(WARN, "face models", f"could not be checked ({e})")

print("\n=== GPU ===")
try:
    import torch
    line(OK, "torch", torch.__version__)
    if torch.cuda.is_available():
        line(OK, "CUDA available", torch.cuda.get_device_name(0))
        line(OK, "CUDA version", str(torch.version.cuda))
    else:
        line(BAD, "CUDA available", "False - torch cannot see the GPU")
        problems.append(
            "torch has no CUDA. Reinstall the CUDA build, e.g.:\n"
            "      pip uninstall -y torch torchvision\n"
            "      pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    line(BAD, "torch", "pip install torch --index-url "
                       "https://download.pytorch.org/whl/cu121")
    problems.append("torch missing")

try:
    import tensorrt  # noqa: F401
    line(OK, "tensorrt (optional)", "installed")
except Exception:
    line(WARN, "tensorrt (optional)", "not installed - system runs fine")

# Does onnxruntime's CUDA provider actually LOAD? get_available_providers()
# only reports what it was built with, so it happily lists CUDA while
# falling back to the CPU at session time - which makes face recognition
# roughly twenty times slower and leaves everybody "Unknown". This checks
# the real thing by creating a session.
try:
    from core.gpu import enable_onnx_cuda
    enable_onnx_cuda(verbose=False)
    import numpy as np
    import onnxruntime as ort
    from onnx import helper, TensorProto

    g = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])], "t",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 10          # onnx writes a newer IR than ORT accepts
    so = ort.SessionOptions()
    so.log_severity_level = 3          # the fallback is noisy on stderr
    s = ort.InferenceSession(m.SerializeToString(), so,
                             providers=["CUDAExecutionProvider",
                                        "CPUExecutionProvider"])
    used = s.get_providers()[0]
    if "CUDA" in used:
        line(OK, "onnxruntime CUDA", "loads - face recognition on the GPU")
    else:
        line(BAD, "onnxruntime CUDA", f"FELL BACK to {used}")
        problems.append(
            "onnxruntime cannot load its CUDA provider, so face\n"
            "      recognition runs on the CPU - far too slow to keep up,\n"
            "      and almost everybody stays 'Unknown'. The CUDA DLLs it\n"
            "      needs ship with torch; core/gpu.py puts them on the\n"
            "      path. If this still fails, check that onnxruntime-gpu\n"
            "      matches your CUDA/cuDNN major versions.")
except Exception as e:
    line(WARN, "onnxruntime CUDA", f"could not be checked ({e})")

print("\n=== Project files ===")
from config.settings import (DETECT_MODEL, ENCODINGS_FILE, DB_PATH,
                             FACE_DIR, MODEL_DIR)

if os.path.exists(DETECT_MODEL):
    line(OK, "detection model", os.path.basename(DETECT_MODEL))
else:
    line(BAD, "detection model", f"missing -> put yolov8n.pt in {MODEL_DIR}")
    problems.append("yolov8n.pt missing from data/models/")

if os.path.isdir(FACE_DIR):
    folders = [d for d in os.listdir(FACE_DIR)
               if os.path.isdir(os.path.join(FACE_DIR, d))]
    if folders:
        line(OK, "employee photo folders", f"{len(folders)} found")
    else:
        line(WARN, "employee photo folders",
             "none yet - add data/faces/<id>-<name>/")
else:
    line(WARN, "employee photo folders", "data/faces/ not created yet")

if os.path.exists(ENCODINGS_FILE):
    line(OK, "face encodings", "enrolled")
else:
    line(WARN, "face encodings",
         "not built - run: python tools/enroll_faces.py")

if os.path.exists(DB_PATH):
    line(OK, "database", DB_PATH)
else:
    line(WARN, "database",
         "not created - run: python tools/init_database.py")

print("\n=== Cameras ===")
from config.cameras import enabled_cameras
cams = enabled_cameras()
if cams:
    for c in cams:
        url = str(c["url"])
        shown = url if len(url) < 60 else url[:57] + "..."
        line(OK, c["name"], shown)
else:
    line(BAD, "cameras", "none enabled in config/cameras.py")
    problems.append("no cameras configured")

print("\n" + "=" * 62)
if problems:
    print("Issues to fix:\n")
    for p in problems:
        print(f"  * {p}")
    print()
    sys.exit(1)
print("Everything needed is in place. You can start the system.")
print("=" * 62 + "\n")
