"""
web/app.py
==========
The dashboard web application.

It only READS the database - all writing is done by the pipeline. That
separation means you can restart either process without disturbing the
other.
"""
import os

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from config.settings import (WEB_SECRET, SESSION_HOURS, STREAM_PORT,
                             TZ_OFFSET_HOURS, REVIEW_DIR, CONFIRMED_SCORE,
                             FACE_LEARN_MIN_WIDTH)
from config.cameras import enabled_cameras
from database import repository as repo
from web.auth import verify_password, current_user

HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Site Intelligence")
app.add_middleware(SessionMiddleware, secret_key=WEB_SECRET,
                   max_age=SESSION_HOURS * 3600)
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")),
          name="static")
# the crops the system was unsure about, so they can be shown for review
os.makedirs(REVIEW_DIR, exist_ok=True)
app.mount("/review-images", StaticFiles(directory=REVIEW_DIR),
          name="review_images")
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))


def _cameras():
    return [{"key": c["key"], "name": c["name"]} for c in enabled_cameras()]


def _guard(request: Request):
    return current_user(request) is not None


# ------------------------------------------------------------- pages
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if current_user(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": ""})


@app.post("/login")
def login_submit(request: Request, username: str = Form(...),
                 password: str = Form(...)):
    user = repo.get_user(username)
    if user and verify_password(password, user.password_hash):
        request.session["user"] = username
        request.session["role"] = user.role
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(
        request, "login.html",
        {"error": "Incorrect username or password."}, status_code=401)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    if not _guard(request):
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request, "dashboard.html", {
        "user": current_user(request),
        "cameras": _cameras(),
        "has_workspace": any(c.get("handler") == "workspace"
                             for c in enabled_cameras()),
        "stream_port": STREAM_PORT,
        "tz_offset": TZ_OFFSET_HOURS,
    })


# --------------------------------------------------------------- API
@app.get("/api/cameras")
def api_cameras(request: Request):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    return _cameras()


@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request):
    """The 'is this the same person?' queue."""
    if not _guard(request):
        return RedirectResponse("/login", status_code=302)
    from core.gallery import GALLERY
    # Everyone we could possibly mean. The gallery in THIS process is
    # empty (the pipeline holds the live one), so the enrolled employees
    # are the real source here - union rather than "or", so a name that
    # only exists in the gallery is still offered.
    people = sorted({e["name"] for e in repo.list_employees()}
                    | set(GALLERY.names()))
    return templates.TemplateResponse(request, "review.html", {
        "user": current_user(request),
        "known_people": people,
    })


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    """Ask the system questions in plain English."""
    if not _guard(request):
        return RedirectResponse("/login", status_code=302)
    from core.chat import ASSISTANT, starters
    return templates.TemplateResponse(request, "chat.html", {
        "user": current_user(request),
        # Built from the cameras that are actually running, and replaced
        # after every answer by whatever follows on from it.
        "chips": starters(),
        "suggestions": ASSISTANT.suggestions,
    })


@app.post("/api/chat")
async def api_chat(request: Request):
    """One question in, one answer out.

    Read-only by construction: the deterministic handlers only read, and
    a generated query runs over a connection opened mode=ro. Nothing
    reachable from here can modify the record.
    """
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    body = await request.json()
    question = (body.get("question") or "").strip()
    from core.chat import ASSISTANT
    return ASSISTANT.ask(question).as_dict()


@app.get("/api/chat/health")
def api_chat_health(request: Request):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    from core.chat import ASSISTANT
    return ASSISTANT.health()


@app.get("/api/reviews")
def api_reviews(request: Request, limit: int = 60):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    return {"pending": repo.pending_reviews(limit),
            "counts": repo.review_counts()}


@app.post("/api/review/{review_id}")
async def api_resolve_review(review_id: int, request: Request):
    """Confirm or reject a suggestion - and teach the system from it."""
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    body = await request.json()
    action = body.get("action")          # confirm | reject | reassign
    name = (body.get("name") or "").strip()

    status = "confirmed" if action in ("confirm", "reassign") else "rejected"
    row = repo.resolve_review(review_id, name, status)
    if row is None:
        return JSONResponse({"error": "not found"}, status_code=404)

    # Use the spelling as ENROLLED from here on. Whatever was typed,
    # everything downstream - the gallery profile, today's clothing, the
    # permanent face sample - must hang off one key per person, or one
    # person quietly becomes two, each holding half the evidence.
    name = row.get("name") or ""

    learned = False
    updated = {"presence": 0, "visits": 0}
    if status == "confirmed" and name:
        # Two separate jobs, and only the first one used to happen:
        #
        #   1. TEACH the system, so this person is recognised next time.
        #   2. CORRECT THE RECORD for the sighting we just asked about.
        #
        # Skipping (2) is why confirming appeared to do nothing: the
        # dashboard row kept saying "Unknown" no matter how many times
        # you answered, because nothing ever wrote the name back to it.
        learned = _learn_from_review(row, name)
        updated = repo.apply_confirmed_identity(
            row.get("camera_key", ""), row.get("track_id"), name,
            emp_code=repo.code_for_name(name),
            when=row.get("created_at"), score=CONFIRMED_SCORE)
        if not any(updated.values()):
            print(f"[REVIEW] confirmed {name}, but no open session/entry "
                  f"matched track #{row.get('track_id')} on "
                  f"{row.get('camera_key')} - only the gallery was taught")

    return {"ok": True, "status": status, "learned": learned,
            "updated": updated}


def _file_reference_photo(row, name):
    """Copy the answered crop into that person's own photo folder.

    This is what turns one confirmation into a permanent, camera-real
    reference photograph rather than just a vector in a table. The file
    lands beside their enrollment photos, so:

      * tools/enroll_faces.py picks it up like any other photograph, and
      * a human can open the folder and delete anything that is wrong.

    That second point is the reason it is a file at all. An automatic
    system that accumulates evidence somewhere opaque cannot be audited,
    and one wrong confirmation would otherwise be invisible for ever.
    """
    try:
        import shutil
        from core.face_library import LIBRARY, CAPTURE_PREFIX

        source = row.get("face_image") or row.get("body_image")
        if not source or not os.path.exists(source):
            return None
        if not LIBRARY.needs_samples(name, repo.code_for_name(name)):
            return None            # they already have enough

        folder = LIBRARY.folder_for(name, repo.code_for_name(name))
        if not folder:
            return None
        stamp = os.path.splitext(os.path.basename(source))[0]
        camera = (row.get("camera_key") or "cam").replace(" ", "_")
        target = os.path.join(folder,
                              f"{CAPTURE_PREFIX}{camera}_confirmed_{stamp}.jpg")
        shutil.copyfile(source, target)
        held = LIBRARY.count(name, repo.code_for_name(name))
        print(f"[REVIEW] filed a camera photo of {name} "
              f"({held}/{LIBRARY.target}) -> {os.path.basename(target)}")
        return target
    except Exception as exc:
        print(f"[REVIEW] could not file the photo for {name}: {exc}")
        return None


def _crop_shows_a_face(row):
    """Is there a face in the picture we are about to file as a reference?

    Runs in the dashboard process, on the click, and takes a few seconds.
    That is the right trade: it happens once per confirmation, and the
    alternative is a permanent reference photograph of a door.

    Returns True when the vision model is off or unreachable, so this can
    never be the reason a confirmation stops working - it only ever
    removes a specific, demonstrated failure.
    """
    from config.settings import VLM_ENABLED, VLM_SCREEN_QUESTIONS
    if not VLM_ENABLED or not VLM_SCREEN_QUESTIONS:
        return True
    path = row.get("face_image") or row.get("body_image") or ""
    if not path or not os.path.exists(path):
        return True
    try:
        import cv2
        from core import vlm
        from core.vlm_verify import ARBITER
        from config.settings import VLM_REJECT_SHOWS
        if not ARBITER.usable():
            return True
        crop = cv2.imread(path)
        if crop is None:
            return True
        label, note = vlm.shows(crop)
        if label in VLM_REJECT_SHOWS:
            print(f"[REVIEW] NOT learning from this crop - it shows "
                  f"{label} ({note}). The sighting HAS still been renamed "
                  f"and the attendance row corrected; what is refused is "
                  f"keeping this picture as a reference photograph, "
                  f"because a picture with no recognisable face in it "
                  f"cannot teach one and would make future recognition "
                  f"worse. {os.path.basename(path)}")
            return False
    except Exception as exc:
        print(f"[REVIEW] could not screen the crop ({exc}); continuing")
    return True


def _learn_from_review(row, name):
    """Turn a human confirmation into a permanent face sample.

    IMPORTANT: this runs in the DASHBOARD process, which shares no memory
    with the running pipeline. So the embedding is written to the
    database rather than into a local gallery - the pipeline polls for
    new rows and folds them in within seconds. Writing it only to memory
    here is why earlier confirmations appeared to do nothing.
    """
    learned = []

    # DOES THE PICTURE BEING FILED ACTUALLY CONTAIN A FACE?
    #
    # This is the guard that was missing, and its absence is measurable.
    # Every one of the three poisoned reference photographs found on this
    # site by tools/vlm_check.py arrived through this function: the top
    # of somebody's head filed under Pavan Kumar, a shoulder and a blur
    # filed twice under Shivajyothi. Each is now used as a reference on
    # every frame, which is what a name that stays wrong AFTER being
    # confirmed actually looks like.
    #
    # Nothing already here could have caught them. The embedding checks
    # are all arguments inside one vector space, and the embedding of the
    # top of a head is a perfectly valid vector that lands near
    # somebody - so agrees_with_enrollment could and did wave them
    # through. Only looking at the picture catches it.
    #
    # The answer is still recorded either way: the sighting is renamed
    # and the attendance row corrected. What is refused is TEACHING from
    # a picture with nothing in it to teach.
    if not _crop_shows_a_face(row):
        return False

    # A FACE, OR NOTHING.
    #
    # This used to learn the clothing in the crop as well, and treat that
    # as the important half - on a work-area camera a confirmation rarely
    # contains a readable face, so clothing was the only thing an answer
    # could teach. It was also the mechanism by which one person's name
    # ended up on another person's body an hour later, so it is gone.
    # A confirmation with no usable face now teaches nothing, and says so.
    #
    # Preferred source: the embedding the PIPELINE already computed and
    # stored with the question. It needs no model here and cannot fail
    # the way re-detection did.
    try:
        emb, width = repo.review_face_embedding(row.get("id"))
        if emb is not None and width >= FACE_LEARN_MIN_WIDTH:
            from core.face import agrees_with_enrollment
            ok, score = agrees_with_enrollment(name, emb)
            if not ok:
                print(f"[REVIEW] {name}: the face in this sighting scores "
                      f"only {score:.2f} against their enrollment photos, "
                      f"so it was NOT learned. Please check the crop - if "
                      f"it really is them, their enrollment photos need "
                      f"redoing; if not, this answer was a misclick.")
                return bool(learned)
            kept = _file_reference_photo(row, name)
            repo.add_learned_face(name, emb, source="review",
                                  camera_key=row.get("camera_key", ""),
                                  image_path=kept or "")
            learned.append("face photo" if kept else "face")
            print(f"[REVIEW] confirmed {name}: learned "
                  f"{' + '.join(learned)} ({width}px face, agrees "
                  f"{score:.2f}) - the pipeline will pick it up in seconds")
            return True
    except Exception as e:
        print(f"[REVIEW] stored embedding unusable ({e}); falling back")

    # Fallback for questions saved before the embedding was kept: re-embed
    # from the BODY crop, NOT the face crop. The face crop is a tight box
    # and a detector finds nothing in it; the body crop has the context a
    # detector needs (measured: 0/25 from face crops, 25/25 from bodies).
    #
    # This now runs the SAME stages as the cameras - SCRFD, five-point
    # alignment, quality filtering, AdaFace - through core.face.embed_face.
    # That is not tidiness: a vector produced by a different code path is
    # not comparable with the enrolled ones, so learning from it would
    # add a reference that never matches anybody.
    body_path = row.get("body_image")
    if not body_path or not os.path.exists(body_path):
        print(f"[REVIEW] confirmed {name}, but the saved crop gave us "
              f"nothing usable to learn from")
        return False
    try:
        import cv2
        from core.face import embed_face, agrees_with_enrollment

        img = cv2.imread(body_path)
        if img is None:
            return bool(learned)

        embedding, quality = embed_face(img)
        if embedding is None:
            # A human has told us WHO this is, so the name is safe - but
            # the picture is still too poor to describe a face with, and
            # a vague sample in the gallery makes everybody harder to
            # tell apart, not easier.
            why = quality.reason if quality is not None else "no face found"
            print(f"[REVIEW] confirmed {name}: learned "
                  f"{' + '.join(learned) or 'nothing'} "
                  f"(the saved crop gave no usable face - {why})")
            return bool(learned)

        ok, score = agrees_with_enrollment(name, embedding)
        if not ok:
            print(f"[REVIEW] {name}: the face in this sighting scores only "
                  f"{score:.2f} against their enrollment photos, so it was "
                  f"NOT learned - this answer looks like a misclick.")
            return bool(learned)
        kept = _file_reference_photo(row, name)
        repo.add_learned_face(name, embedding, source="review",
                              camera_key=row.get("camera_key", ""),
                              image_path=kept or "")
        learned.append("face photo" if kept else "face")
        print(f"[REVIEW] confirmed {name}: learned {' + '.join(learned)} "
              f"({quality.width}px face, quality {quality.score:.2f}, "
              f"agrees {score:.2f}) - the pipeline will pick it up in "
              f"seconds")
        return True
    except Exception as e:
        print(f"[REVIEW] face part of the confirmation failed ({e}); "
              f"learned {' + '.join(learned) or 'nothing'}")
        return bool(learned)


@app.get("/api/gallery")
def api_gallery(request: Request):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    from core.gallery import GALLERY
    return {"stats": GALLERY.stats(), "people": GALLERY.summary()}


@app.get("/api/camera_status")
def api_camera_status(request: Request):
    """Whether each camera stream is actually alive.

    The dashboard reads this from the video server, so it reflects the
    pipeline's real state rather than the config file.
    """
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    import json as _json
    import urllib.request as _url
    try:
        with _url.urlopen(f"http://127.0.0.1:{STREAM_PORT}/cameras",
                          timeout=2) as r:
            live = {c["key"] for c in _json.loads(r.read())}
        running = True
    except Exception:
        live, running = set(), False
    return {"pipeline_running": running,
            "cameras": [{**c, "streaming": c["key"] in live}
                        for c in _cameras()]}


@app.get("/api/summary")
def api_summary(request: Request, hours: int = 24, camera: str = None):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    return repo.summary(hours=hours, camera_key=camera)


@app.get("/api/visits")
def api_visits(request: Request, limit: int = 100, camera: str = None):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    return repo.recent_visits(limit=limit, camera_key=camera)


@app.get("/api/present")
def api_present(request: Request, camera: str = None):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    return repo.active_visits(camera_key=camera)


@app.get("/api/presence")
def api_presence(request: Request, hours: int = 24, camera: str = None):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    return repo.presence_rows(camera_key=camera, hours=hours)


@app.get("/api/presence_totals")
def api_presence_totals(request: Request, hours: int = 24, camera: str = None):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    return repo.presence_totals(camera_key=camera, hours=hours)


@app.get("/api/hourly")
def api_hourly(request: Request, hours: int = 24, camera: str = None):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    return repo.hourly_entries(hours=hours, camera_key=camera)


@app.get("/api/employees")
def api_employees(request: Request):
    if not _guard(request):
        return JSONResponse({"error": "auth"}, status_code=401)
    return repo.list_employees()
