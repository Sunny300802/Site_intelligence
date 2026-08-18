"""
config/cameras.py
=================
The camera registry. ONE entry per physical camera.

This is the file you edit as we add each new camera to the system.
Right now it holds Camera 1 (Reception Lobby).

Each entry:
    key           internal id, lowercase, no spaces (used in URLs/DB)
    name          human label shown on the dashboard and in logs
    url           RTSP link, a video file path, or 0 for a webcam
    handler       which camera logic module runs for this feed
    enabled       set False to temporarily switch a camera off
    options       handler-specific settings (see the handler's docstring)
"""
import os

# Camera URLs come from the environment so no real credential is ever
# committed. The strings below are PLACEHOLDERS - set the real links as
# environment variables before starting the pipeline:
#
#   PowerShell:
#     $env:RECEPTION_URL = "rtsp://user:pass@10.0.0.5:554/chID=6&streamType=sub&linkType=tcp"
#     $env:SERVER_RM_URL = "rtsp://user:pass@10.0.0.5:554/chID=19&streamType=main&linkType=tcp"
#
# If the password contains @ or :, percent-encode it (@ -> %40, : -> %3A).
# A local .mp4 path also works in place of an RTSP link, which is handy
# for testing.
RECEPTION_URL = os.environ.get(
    "RECEPTION_URL",
    "rtsp://USER:PASSWORD@CAMERA_IP:554/chID=6&streamType=sub&linkType=tcp",
)

SERVER_RM_URL = os.environ.get(
    "SERVER_RM_URL",
    "rtsp://USER:PASSWORD@CAMERA_IP:554/chID=19&streamType=main&linkType=tcp",
)

CAMERAS = [
    {
        "key": "reception",
        "name": "Reception Lobby",
        "url": RECEPTION_URL,
        "handler": "reception",
        "enabled": True,
        "options": {
            # --- speed vs care ---------------------------------------
            # People cross a reception FAST. Detection must run on EVERY
            # frame or somebody is through the door before we see them,
            # so this camera gets priority and a modest input size to
            # keep it quick.
            "detect_every": 1,
            "imgsz": 1280,
            "conf": 0.25,
            "enhance": None,        # reception is evenly lit

            # --- what this camera does -------------------------------
            # Entry only. People leaving show their back to the camera,
            # so exits are deliberately NOT tracked.
            "count_entries": True,

            # --- where to LOOK for people ----------------------------
            # A detection region, as fractions of the frame
            # [x1, y1, x2, y2]. The detector only sees this area, which
            # both skips scenery nobody walks through (the wooden wall on
            # the right of the reception view) AND makes the people who
            # are there proportionally bigger to the model, so distant
            # ones get found. Boxes are mapped back to the full frame, so
            # the video and logs are unaffected.
            #
            # Set to None to scan the whole frame. Use
            #   python tools/tune_detection.py --roi 0,0,0.5,1
            # to compare regions before committing to one.
            # Whole frame. Restricting the area was costing us people who
            # walked in outside it, which matters more than the compute it
            # saved.
            "detect_roi": None,

            # --- where an entry COUNTS -------------------------------
            # Optional extra filter applied after detection. None = the
            # whole detection region counts.
            "zone": None,

            # --- identity --------------------------------------------
            # Recognise faces and back-fill the name onto the whole visit.
            "recognise_faces": True,

            # MINIMUM FACE WIDTH, in pixels, for THIS camera.
            #
            # Measured on this feed: faces arrive at 26-41px, median 32.
            # The global MIN_FACE_SIZE of 56 is therefore unreachable
            # here - it rejected 100% of them, which is why reception
            # recognised nobody and never asked about anybody either.
            #
            # 28 is set to match what this camera can actually deliver.
            # It does NOT make wrong names more likely: everything
            # downstream still has to clear FACE_RECOGNITION_THRESHOLD,
            # the runner-up margin and the temporal vote. It only stops
            # the pipeline throwing every face away before those gates
            # get a chance to judge it.
            #
            # THE REAL FIX IS THE STREAM. This camera is on the SUB
            # stream at 1280x720. Its main stream is 2560x1440, which
            # would put faces at 55-80px and make all of this
            # comfortable. Change streamType=sub to streamType=main in
            # RECEPTION_URL above if the bandwidth and GPU allow it -
            # the workspace camera already runs that way.
            "min_face_size": 28,
        },
    },

    {
        "key": "server_rm_psg",
        "name": "Server Rm Psg",
        "url": SERVER_RM_URL,
        "handler": "workspace",
        "enabled": True,
        "options": {
            # --- speed vs care ---------------------------------------
            # The opposite trade to reception. People here are SEATED and
            # barely move, so detecting 3 times a second loses nothing -
            # and the time saved is spent on a bigger input size and low
            # light enhancement, which is where the accuracy actually is.
            "detect_every": 3,
            "imgsz": 1280,          # small, half-hidden, seated people

            # PERSON detection confidence. Lowered back from 0.34.
            #
            # This number has nothing to do with wrong NAMES - it decides
            # whether a shape is a person at all, not who that person is.
            # Raising it cannot reduce misidentification, and it does
            # lose people: a seated person half behind a monitor scores
            # low by nature, which is exactly the "they are in the area
            # but not detected" report. Wrong names are governed by
            # FACE_RECOGNITION_THRESHOLD and FACE_MATCH_MARGIN.
            "conf": 0.50,

            # MINIMUM PERSON HEIGHT for this camera, as a fraction of the
            # frame. Measured here: the global 0.12 (about 173px on this
            # 1440-tall feed) threw away 55% of all detections - 110
            # detected, 50 kept - because a SEATED person behind a desk
            # is simply not that tall in frame. The single biggest cause
            # of people in the area going undetected.
            #
            # People here are counted for PRESENCE, not entries, so there
            # is no size requirement to protect: the tracker's min-hits
            # and PRESENCE_MIN_SECONDS already filter noise.
            "min_person_height_frac": 0.05,
            # Measured on this camera: the far side sits at mean
            # brightness 62 with blacks crushed to 1, the window side at
            # 134. Local contrast equalisation more than DOUBLES usable
            # detail on the dark side (941 -> 2091) while slightly
            # REDUCING highlight clipping.
            "enhance": "lowlight",

            # --- the monitored area (your red boundary) --------------
            # Points are FRACTIONS of the frame, so they survive a
            # resolution change. Everything outside is blacked out
            # before detection, so the far desks, the corridor and the
            # glass frontage cannot produce detections at all.
            #
            # These are traced from your screenshot - fine-tune them by
            # clicking on a real frame:
            #     python tools/draw_area.py --camera server_rm_psg
            "area": [
                (0.010, 0.560), (0.048, 0.410), (0.105, 0.370),
                (0.200, 0.330), (0.290, 0.230), (0.420, 0.170),
                (0.520, 0.160), (0.620, 0.185), (0.700, 0.215),
                (0.790, 0.235), (0.860, 0.245), (0.905, 0.235),
                (0.930, 0.250), (0.955, 0.330), (0.962, 0.480),
                (0.958, 0.640), (0.930, 0.800), (0.880, 0.940),
                (0.560, 0.985), (0.230, 0.980), (0.130, 0.930),
                (0.030, 0.800),
            ],
            # Detect across the WHOLE frame. The area below is still used
            # to decide who counts as being in the work area, but people
            # are now tracked everywhere the camera can see - so somebody
            # who steps out of the zone and back is the same person, not a
            # new one.
            "mask_outside": False,
            "crop_to_area": False,

            # --- identity --------------------------------------------
            # Primary method: STICKY TRACKING. A name learned once (from
            # a face, when one is briefly visible) stays attached to that
            # body for as long as it is tracked - including when they
            # turn away from the camera.
            "recognise_faces": True,

            # Measured on this feed (2560x1440, main stream): faces
            # arrive at 32-78px, median 68, and 91% already clear 56.
            # Lowered a little so the smaller ones - somebody at the far
            # desk - are judged on quality rather than discarded on size
            # alone.
            "min_face_size": 44,

            # Optional: map desks to people. Left empty on purpose -
            # it assumes nobody ever sits in a colleague's place, and a
            # confident wrong name is worse than an honest Unknown.
            # Build one with: python tools/draw_seats.py --camera server_rm_psg
            # "seats": [],
        },
    },
]


def enabled_cameras():
    return [c for c in CAMERAS if c.get("enabled", True)]


def camera_by_key(key):
    for c in CAMERAS:
        if c["key"] == key:
            return c
    return None
