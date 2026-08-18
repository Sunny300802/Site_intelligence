"""
core/frame_grabber.py
=====================
Reads an RTSP stream (or a video file) on its own thread and always hands
back the NEWEST frame.

Why a thread
------------
cv2.VideoCapture.read() blocks. If the main GPU loop called it directly,
decoding would stall inference and a delay would build up that never
recovers. This grabber keeps draining the stream and simply overwrites
the stored frame, so the pipeline always sees live video rather than a
growing backlog.

Surviving a long run
--------------------
Cameras drop connections - overnight, on a network blip, when the NVR
restarts. Reconnecting correctly matters more than it sounds:

  * The old capture is RELEASED before a new one is opened. Replacing it
    without releasing leaks the ffmpeg context and, on most cameras,
    leaves the RTSP session open at the far end. Cameras allow only a
    handful of concurrent sessions, so a night of leaked retries can
    exhaust them - after which nothing can connect until the camera
    itself times them out. That is what turns a brief blip into a dead
    stream by morning.

  * Transport is forced to TCP. UDP silently loses packets over a busy
    network and many cameras handle it worse.

  * Open and read timeouts are set to a few seconds instead of the
    thirty-second default, so a failed attempt fails fast.

  * Retries back off (3s, 6s, 12s ... up to a minute) rather than
    hammering a camera that is rebooting, and the log says how long the
    stream has actually been down instead of repeating one line forever.
"""
import os
import time
import threading

# These options are read by FFMPEG when a capture is opened, so they must
# be set before the first VideoCapture is created.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|reorder_queue_size;0",
)

import cv2  # noqa: E402  (import after the env var above)


class FrameGrabber:
    def __init__(self, source, name="camera", reconnect_delay=3.0,
                 max_backoff=60.0, open_timeout_ms=8000,
                 read_timeout_ms=8000):
        self.source = source
        self.name = name
        self.base_delay = reconnect_delay
        self.max_backoff = max_backoff
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms

        self._cap = None
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # health
        self.connected = False
        self.down_since = None
        self.failures = 0
        self.reconnects = 0
        self._last_report = 0.0

        self.is_file = (isinstance(source, str)
                        and not str(source).startswith(
                            ("rtsp://", "rtmp://", "http://", "https://")))

    # ------------------------------------------------------- lifecycle
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _close(self):
        """Release the capture properly. Skipping this is what leaks RTSP
        sessions on the camera and eventually blocks all reconnection."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _open(self):
        self._close()                      # always release before reopening
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        for prop, value in ((cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms),
                            (cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms),
                            (cv2.CAP_PROP_BUFFERSIZE, 1)):
            try:
                cap.set(prop, value)
            except Exception:
                pass                       # not all builds expose these
        return cap

    def _backoff(self):
        delay = min(self.base_delay * (2 ** min(self.failures, 5)),
                    self.max_backoff)
        return delay

    def _report_down(self, reason):
        """Say something useful, occasionally - not the same line forever."""
        now = time.time()
        if self.down_since is None:
            self.down_since = now
        down_for = now - self.down_since
        # first failure, then every 60s
        if self.failures <= 1 or now - self._last_report >= 60:
            self._last_report = now
            mins = down_for / 60.0
            when = (f"{down_for:.0f}s" if down_for < 120
                    else f"{mins:.0f} min")
            print(f"[GRABBER:{self.name}] {reason}; down for {when}, "
                  f"attempt {self.failures}, next try in "
                  f"{self._backoff():.0f}s")
            if mins >= 10 and self.failures % 10 == 0:
                print(f"[GRABBER:{self.name}] still unreachable after "
                      f"{mins:.0f} min - check the camera is powered and "
                      f"on the network, and that the URL/credentials are "
                      f"still valid")

    def _mark_up(self):
        was_down = not self.connected
        self.connected = True
        if was_down:
            self.reconnects += 1
            if self.down_since:
                down = time.time() - self.down_since
                print(f"[GRABBER:{self.name}] reconnected after "
                      f"{down:.0f}s (reconnect #{self.reconnects})")
            else:
                print(f"[GRABBER:{self.name}] connected")
        self.down_since = None
        self.failures = 0

    # ------------------------------------------------------------ loop
    def _loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    self.connected = False
                    self.failures += 1
                    self._report_down("cannot open stream")
                    self._close()          # do not leave a dead handle open
                    time.sleep(self._backoff())
                    continue
                self._mark_up()

            ok, frame = self._cap.read()
            if not ok:
                if self.is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop files
                    continue
                self.connected = False
                self.failures += 1
                self._report_down("stream dropped")
                self._close()
                time.sleep(self._backoff())
                continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1

            if self.is_file:
                time.sleep(1 / 30)         # play files at a sane speed

    # ------------------------------------------------------------ read
    def read(self):
        """Return (frame_id, frame). frame is None until the first arrives."""
        with self._lock:
            if self._frame is None:
                return -1, None
            return self._frame_id, self._frame.copy()

    def health(self):
        return {
            "connected": self.connected,
            "down_seconds": (0 if self.down_since is None
                             else round(time.time() - self.down_since, 1)),
            "failures": self.failures,
            "reconnects": self.reconnects,
        }

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._close()
