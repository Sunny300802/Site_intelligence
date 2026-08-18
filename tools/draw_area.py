"""
tools/draw_area.py
==================
Draw a camera's monitored area by clicking on a real frame.

Typing polygon coordinates by hand is guesswork. This grabs a frame from
the camera (or a file), lets you click the boundary, and prints the
result ready to paste into config/cameras.py.

USAGE
-----
    python tools/draw_area.py --camera server_rm_psg
    python tools/draw_area.py --source "D:\\clips\\office.mp4"
    python tools/draw_area.py --source frame.png

CONTROLS
--------
    left click    add a point
    right click   remove the last point
    r             reset
    m             preview what the detector will actually see
    s             save + print the coordinates
    q / Esc       quit without saving

The preview matters: it shows the masked, cropped image the detector
receives, which is the real test of whether your boundary is right.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from config.cameras import enabled_cameras, camera_by_key
from core.geometry import (to_pixels, build_mask, bounding_box, apply_mask,
                           polygon_area_fraction)
from core.sampling import sample_frames

WINDOW = "Draw monitored area  (click boundary | r reset | m preview | s save | q quit)"


class AreaDrawer:
    def __init__(self, frame, existing=None):
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.points = []
        if existing:
            self.points = [(int(x * self.w if x <= 1 else x),
                            int(y * self.h if y <= 1 else y))
                           for x, y in existing]

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def render(self):
        img = self.frame.copy()
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(img, [pts], len(self.points) >= 3, (0, 230, 255), 2)
        for i, (x, y) in enumerate(self.points):
            cv2.circle(img, (x, y), 5, (0, 230, 255), -1)
            cv2.putText(img, str(i + 1), (x + 7, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        bar = f"{len(self.points)} points"
        if len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            frac = polygon_area_fraction(pts, self.w, self.h)
            bar += f"   covers {frac:.0%} of the frame"
        cv2.rectangle(img, (0, 0), (self.w, 26), (20, 20, 20), -1)
        cv2.putText(img, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        return img

    def preview(self):
        if len(self.points) < 3:
            print("[DRAW] need at least 3 points to preview")
            return
        pts = np.array(self.points, dtype=np.int32)
        mask = build_mask(pts, self.w, self.h)
        bbox = bounding_box(pts)
        shown = apply_mask(self.frame, mask, bbox)
        cv2.imshow("Detector input preview  (any key to close)", shown)
        print(f"[DRAW] detector would receive {shown.shape[1]}x{shown.shape[0]} "
              f"(frame is {self.w}x{self.h})")
        cv2.waitKey(0)
        cv2.destroyWindow("Detector input preview  (any key to close)")

    def as_fractions(self):
        return [(round(x / self.w, 3), round(y / self.h, 3))
                for x, y in self.points]


def print_config(points):
    print("\n" + "=" * 62)
    print("Paste this into config/cameras.py, as the camera's \"area\":")
    print("=" * 62)
    print('            "area": [')
    for i in range(0, len(points), 3):
        chunk = points[i:i + 3]
        line = " ".join(f"({x:.3f}, {y:.3f})," for x, y in chunk)
        print(f"                {line}")
    print("            ],")
    print("=" * 62 + "\n")


def main(source, camera_key):
    existing = None
    if source is None:
        cams = enabled_cameras()
        cfg = camera_by_key(camera_key) if camera_key else (cams[0] if cams else None)
        if cfg is None:
            print("[DRAW] no camera configured; use --source")
            return 1
        source = cfg["url"]
        existing = cfg.get("options", {}).get("area")
        print(f"[DRAW] camera '{cfg['name']}'")
        if existing:
            print(f"[DRAW] loaded existing area ({len(existing)} points) - "
                  f"adjust it, or press r to start over")

    if os.path.isfile(source) and source.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp")):
        frame = cv2.imread(source)
    else:
        frames = sample_frames(source, count=3)
        frame = frames[len(frames) // 2][1] if frames else None

    if frame is None:
        print(f"[DRAW] could not read a frame from {source}")
        return 1
    print(f"[DRAW] frame {frame.shape[1]}x{frame.shape[0]}")

    drawer = AreaDrawer(frame, existing)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, drawer.on_mouse)

    while True:
        cv2.imshow(WINDOW, drawer.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            print("[DRAW] cancelled")
            break
        if key == ord("r"):
            drawer.points = []
        elif key == ord("m"):
            drawer.preview()
        elif key == ord("s"):
            if len(drawer.points) < 3:
                print("[DRAW] need at least 3 points")
                continue
            print_config(drawer.as_fractions())
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draw a camera's monitored area")
    ap.add_argument("--source", default=None,
                    help="camera url, video file, or image")
    ap.add_argument("--camera", default=None, help="camera key from config")
    args = ap.parse_args()
    raise SystemExit(main(args.source, args.camera))
