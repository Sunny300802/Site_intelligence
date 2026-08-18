"""
tools/export_tensorrt.py
========================
OPTIONAL speed-up. Converts the YOLO model to a TensorRT engine, which
typically runs noticeably faster than the .pt file.

    python tools/export_tensorrt.py

Notes
-----
* TensorRT must be installed and must match your CUDA version. If you are
  on CUDA 12.x install the CUDA-12 build explicitly:
      pip install tensorrt-cu12
  A bare "pip install tensorrt" may pull a CUDA-13 build that will not work.
* Build the engine on the SAME machine and GPU that will run it - an
  engine file is tied to that specific hardware and driver.
* Everything works fine without this. The pipeline falls back to the .pt
  file automatically when no .engine exists.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DETECT_MODEL, IMG_SIZE, USE_HALF


def main():
    if not os.path.exists(DETECT_MODEL):
        print(f"[TRT] model not found: {DETECT_MODEL}")
        return 1
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("[TRT] the 'tensorrt' package is not installed.")
        print("[TRT] for CUDA 12.x run:  pip install tensorrt-cu12")
        print("[TRT] (this step is optional - the system runs without it)")
        return 1

    from ultralytics import YOLO
    print(f"[TRT] exporting {DETECT_MODEL} at {IMG_SIZE}px, half={USE_HALF}")
    model = YOLO(DETECT_MODEL)
    model.export(format="engine", imgsz=IMG_SIZE, half=USE_HALF,
                 device=0, workspace=4, verbose=False)
    print("[TRT] done - the pipeline will use the .engine file automatically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
