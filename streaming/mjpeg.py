"""
streaming/mjpeg.py
==================
A tiny dependency-free MJPEG server.

The pipeline pushes each camera's annotated frame in here; the dashboard
embeds them with a plain <img> tag. No websockets, no browser plugins -
an <img> pointed at a multipart stream just works.

It runs on its own port (default 8001) because the vision pipeline and
the web dashboard are separate processes.

SCENES - THE SECOND THING THIS CARRIES
--------------------------------------
The stream above is for a person to look at. A SCENE is for a program to
look at: the same moment, but as the UNANNOTATED frame plus the boxes,
names and dwell times the pipeline had for it. That pair is what lets
the assistant answer "what is happening in the server room" from another
process (core/vlm_scene.py) - the picture for the vision model, the
names from the tracker, joined in code and never by the model.

Unannotated matters. Handed the drawn frame, a vision model reads
"Pavan Kumar 42m" off the box and reports it back as an observation,
which is the one thing the whole identity design exists to prevent.
"""
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = {}
        self._meta = {}
        self._scene = {}

    def register(self, key, name):
        with self._lock:
            self._meta[key] = name

    def publish(self, key, jpeg_bytes):
        with self._lock:
            self._jpeg[key] = jpeg_bytes

    def get(self, key):
        with self._lock:
            return self._jpeg.get(key)

    def cameras(self):
        with self._lock:
            return [{"key": k, "name": n} for k, n in self._meta.items()]

    # ------------------------------------------------------- scenes
    def publish_scene(self, key, jpeg_bytes, people):
        """One unannotated frame and who the pipeline had in it.

        `people` is a list of plain dicts - track id, name (empty when
        the person is not identified), box in the coordinates of THIS
        jpeg, and seconds present. Plain data on purpose: it crosses a
        process boundary as JSON, and a Track object would not.
        """
        with self._lock:
            self._scene[key] = {"jpeg": jpeg_bytes,
                                "people": list(people or []),
                                "at": time.time()}

    def scene(self, key):
        """The newest scene for a camera, or None."""
        with self._lock:
            return self._scene.get(key)

    def scene_meta(self, key):
        """Everything about a scene except the picture itself."""
        scene = self.scene(key)
        if scene is None:
            return None
        return {"key": key, "name": self._meta.get(key, key),
                "age": round(time.time() - scene["at"], 2),
                "people": scene["people"]}


STORE = FrameStore()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass                      # keep the pipeline console readable

    def handle_one_request(self):
        # a client disappearing mid-stream is routine; don't dump a
        # traceback into the pipeline console for it
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError,
                ConnectionAbortedError):
            self.close_connection = True

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/cameras"):
            self._json(STORE.cameras())
            return

        # ONE frame, unannotated, for a program to reason about - not a
        # stream. Its age is in a header rather than only in /scene, so a
        # caller that took the picture alone can still tell whether the
        # pipeline is actually running.
        if self.path.startswith("/snapshot/"):
            key = self.path.split("/snapshot/", 1)[1].strip("/")
            scene = STORE.scene(key)
            if scene is None:
                self._json({"error": f"no snapshot for {key!r}"}, status=404)
                return
            jpeg = scene["jpeg"]
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("X-Snapshot-Age",
                             f"{time.time() - scene['at']:.2f}")
            self._cors()
            self.end_headers()
            self.wfile.write(jpeg)
            return

        # ...and who the pipeline had in that frame. Kept separate from
        # the picture so a caller can check the age and the people
        # cheaply before deciding whether the picture is worth fetching.
        if self.path.startswith("/scene/"):
            key = self.path.split("/scene/", 1)[1].strip("/")
            meta = STORE.scene_meta(key)
            if meta is None:
                self._json({"error": f"no scene for {key!r}"}, status=404)
                return
            self._json(meta)
            return

        if self.path.startswith("/video/"):
            key = self.path.split("/video/", 1)[1].strip("/")
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self._cors()
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = STORE.get(key)
                    if jpeg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)          # cap the wire at ~20 fps
            except (BrokenPipeError, ConnectionResetError,
                    ConnectionAbortedError, OSError):
                # the browser closed the tab or navigated away - normal
                return
            return

        body = (b"MJPEG server. Try /cameras, /video/<key>, "
                b"/snapshot/<key> or /scene/<key>")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)


def start_stream_server(port):
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[STREAM] video server on port {port}  (/cameras, /video/<key>, "
          f"/snapshot/<key>, /scene/<key>)")
    return server
