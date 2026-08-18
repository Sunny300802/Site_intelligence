"""
core/sampling.py
================
Grab frames spread across a WHOLE video, not just the beginning.

Sampling only the opening seconds is a classic way to test on an empty
room and conclude, wrongly, that detection is broken. These helpers walk
the entire clip so the sample actually contains the people you care
about.
"""
import cv2


def video_info(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"frames": total, "fps": fps, "width": w, "height": h,
            "seconds": (total / fps) if fps else 0}


def sample_frames(source, count=12, live_seconds=20):
    """Return up to `count` frames spread over the whole source.

    For a FILE: seeks evenly from start to end.
    For a LIVE stream (no frame count): reads for a few seconds and keeps
    an even spread of what arrives.
    Returns a list of (position_label, frame).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []

    if total > 1:
        # a file: seek evenly across the whole thing
        step = max(1, total // (count + 1))
        for i in range(1, count + 1):
            pos = min(total - 1, i * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if ok:
                secs = pos / fps if fps else 0
                frames.append((f"{secs:0.1f}s", frame))
    else:
        # a live stream: just collect over a window of time
        import time
        t_end = time.time() + live_seconds
        keep_every = max(1, int(fps / 2))
        i = 0
        while time.time() < t_end and len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            if i % keep_every == 0:
                frames.append((f"#{i}", frame))
            i += 1

    cap.release()
    return frames
