"""
database/repository.py
======================
All database reads and writes live here, so no other module writes SQL.

The three functions that matter for the reception camera:

    open_visit()      person confirmed in view  -> create the row (Unknown)
    identify_visit()  face finally recognised   -> BACK-FILL the name
    close_visit()     person left coverage      -> stamp the end time

open_visit + identify_visit is the retroactive naming: the arrival time
never moves, but the name is corrected the moment the person is close
enough for their face to be read.
"""
import os
from datetime import datetime, timedelta
from sqlalchemy import select, func, desc

from database.db import SessionLocal
from database.models import (Visit, User, Employee, SystemEvent,
                            Presence, ReviewItem, LearnedFace,
                            LearnedAppearance)


# ------------------------------------------------------------ visits
def open_visit(camera_key, camera_name, track_id, when=None):
    """Create a visit for a newly confirmed person. Returns the row id."""
    when = when or datetime.utcnow()
    with SessionLocal() as s:
        v = Visit(camera_key=camera_key, camera_name=camera_name,
                  track_id=track_id, person_name="Unknown",
                  entered_at=when, last_seen=when, active=True)
        s.add(v)
        s.commit()
        return v.id


def identify_visit(visit_id, person_name, emp_code, score):
    """Back-fill the identity onto an existing visit.

    Only overwrites an existing name if the new score is better, so a
    single poor frame can't rename someone who was already recognised.
    Returns True if the name was applied.
    """
    with SessionLocal() as s:
        v = s.get(Visit, visit_id)
        if v is None:
            return False
        if v.identified and score <= v.match_score:
            return False
        v.person_name = person_name
        v.emp_code = emp_code or ""
        v.match_score = float(score)
        v.identified = True
        if v.identified_at is None:
            v.identified_at = datetime.utcnow()
        s.commit()
        return True


def touch_visit(visit_id, when=None):
    """Keep the visit alive while the person is still on screen."""
    when = when or datetime.utcnow()
    with SessionLocal() as s:
        v = s.get(Visit, visit_id)
        if v:
            v.last_seen = when
            v.duration_seconds = (when - v.entered_at).total_seconds()
            s.commit()


def find_recent_visit(camera_key, person_name, within_seconds):
    """Is there already a visit for this person on this camera, started
    within the last N seconds? Used as the final duplicate guard: if the
    tracker lost someone and picked them up as a new track, we reuse the
    original arrival instead of recording a second one.

    Returns the visit id, or None.
    """
    if not person_name or person_name == "Unknown":
        return None
    cutoff = datetime.utcnow() - timedelta(seconds=within_seconds)
    with SessionLocal() as s:
        row = s.execute(
            select(Visit.id)
            .where(Visit.camera_key == camera_key,
                   Visit.person_name == person_name,
                   Visit.entered_at >= cutoff)
            .order_by(desc(Visit.entered_at))
            .limit(1)
        ).scalar_one_or_none()
    return row


def close_visit(visit_id, when=None):
    when = when or datetime.utcnow()
    with SessionLocal() as s:
        v = s.get(Visit, visit_id)
        if v:
            v.left_at = when
            v.last_seen = when
            v.duration_seconds = (when - v.entered_at).total_seconds()
            v.active = False
            s.commit()


def close_all_active():
    """Called at pipeline start so a previous crash can't leave people
    permanently 'in view'."""
    with SessionLocal() as s:
        n = s.query(Visit).filter(Visit.active.is_(True)).update(
            {Visit.active: False, Visit.left_at: datetime.utcnow()})
        s.commit()
        return n


# ---------------------------------------------------- learned faces
def add_learned_face(person_name, embedding, emp_code="", source="review",
                     camera_key="", model_tag=None, image_path=""):
    """Persist a confirmed face so the pipeline process can pick it up.

    The model tag is stamped automatically from whatever recognition
    model this process has loaded, because the caller getting it wrong
    is exactly the failure this column exists to prevent.
    """
    import numpy as np
    if embedding is None:
        return None
    vec = np.asarray(embedding, dtype=np.float32).ravel()
    if vec.ndim != 1 or vec.size < 64 or not np.isfinite(vec).all():
        return None
    if model_tag is None:
        try:
            from core.face_embed import model_tag as current_tag
            model_tag = current_tag()
        except Exception:
            model_tag = ""
    person_name = canonical_name(person_name)     # one spelling per person
    with SessionLocal() as s:
        row = LearnedFace(person_name=person_name, emp_code=emp_code or "",
                          embedding=vec.tobytes(), dims=int(vec.size),
                          model_tag=model_tag or "",
                          image_path=image_path or "",
                          source=source, camera_key=camera_key or "")
        s.add(row)
        s.commit()
        return row.id


def learned_faces_since(last_id=0):
    """New confirmed faces the pipeline has not loaded yet."""
    import numpy as np
    with SessionLocal() as s:
        rows = s.execute(
            select(LearnedFace)
            .where(LearnedFace.id > last_id)
            .order_by(LearnedFace.id.asc())
        ).scalars().all()
    out = []
    for r in rows:
        try:
            vec = np.frombuffer(r.embedding, dtype=np.float32)
            if vec.size == r.dims:
                out.append({"id": r.id, "name": r.person_name,
                            "code": r.emp_code, "embedding": vec,
                            "model": r.model_tag or "",
                            "source": r.source})
        except Exception:
            continue
    return out


def learned_face_count():
    with SessionLocal() as s:
        return int(s.execute(
            select(func.count()).select_from(LearnedFace)).scalar_one())


def add_learned_appearance(person_name, descriptor, emp_code="",
                           camera_key="", source="review", for_day=None):
    """Persist a confirmed clothing signature for a given day."""
    import numpy as np
    from datetime import date
    if descriptor is None:
        return None
    vec = np.asarray(descriptor, dtype=np.float32).ravel()
    if vec.ndim != 1 or vec.size < 16 or not np.isfinite(vec).all():
        return None          # never store a signature we cannot compare
    person_name = canonical_name(person_name)     # one spelling per person
    day = for_day or date.today().isoformat()
    with SessionLocal() as s:
        row = LearnedAppearance(person_name=person_name,
                                emp_code=emp_code or "",
                                descriptor=vec.tobytes(), dims=int(vec.size),
                                for_day=day, camera_key=camera_key or "",
                                source=source)
        s.add(row)
        s.commit()
        return row.id


def learned_appearance_since(last_id=0, for_day=None):
    """New confirmed clothing signatures for today."""
    import numpy as np
    from datetime import date
    day = for_day or date.today().isoformat()
    with SessionLocal() as s:
        rows = s.execute(
            select(LearnedAppearance)
            .where(LearnedAppearance.id > last_id,
                   LearnedAppearance.for_day == day)
            .order_by(LearnedAppearance.id.asc())
        ).scalars().all()
    out = []
    for r in rows:
        try:
            vec = np.frombuffer(r.descriptor, dtype=np.float32)
            if vec.size == r.dims and vec.size:
                out.append({"id": r.id, "name": r.person_name,
                            "code": r.emp_code, "descriptor": vec,
                            "camera_key": r.camera_key})
        except Exception:
            continue
    return out


def recently_reviewed_names(camera_key, within_seconds=600):
    """Names we have already asked about on this camera recently.

    Stops the queue filling with the same question over and over while
    the answer is still working its way through.
    """
    cutoff = datetime.utcnow() - timedelta(seconds=within_seconds)
    with SessionLocal() as s:
        rows = s.execute(
            select(ReviewItem.suggested_name)
            .where(ReviewItem.camera_key == camera_key,
                   ReviewItem.created_at >= cutoff)
        ).scalars().all()
    return set(rows)


def confirmed_names_today():
    """People a human has already confirmed today - no need to ask again."""
    start = datetime.utcnow().replace(hour=0, minute=0, second=0,
                                      microsecond=0)
    with SessionLocal() as s:
        rows = s.execute(
            select(ReviewItem.confirmed_name)
            .where(ReviewItem.status == "confirmed",
                   ReviewItem.reviewed_at >= start)
        ).scalars().all()
    return {n for n in rows if n}


# ------------------------------------------------------- review queue
def add_review(camera_key, camera_name, track_id, suggested, runner_up,
               probability, margin, evidence, body_image="", face_image="",
               face_embedding=None, face_width=0, looks_like="",
               candidates=()):
    """Save a question for a human.

    `face_embedding` is the vector the recogniser produced from the live
    frame. Storing it here is what lets a confirmation teach a FACE:
    re-detecting one inside the saved crop does not work, because the
    crop is a tight box and a detector needs context around a face.

    `looks_like` and `candidates` come from the vision model's screening
    of the crop (core/vlm_verify.py) - the description it produced, and
    the people that description does not rule out.
    """
    import numpy as np
    blob, dims = None, 0
    if face_embedding is not None:
        vec = np.asarray(face_embedding, dtype=np.float32).ravel()
        if vec.size >= 64 and np.isfinite(vec).all():
            blob, dims = vec.tobytes(), int(vec.size)
    with SessionLocal() as s:
        row = ReviewItem(camera_key=camera_key, camera_name=camera_name,
                         track_id=track_id, suggested_name=suggested,
                         runner_up=runner_up or "", probability=probability,
                         margin=margin, evidence=evidence[:250],
                         body_image=body_image, face_image=face_image,
                         face_embedding=blob, face_dims=dims,
                         face_width=int(face_width or 0),
                         looks_like=(looks_like or "")[:120],
                         candidates=",".join(candidates or ())[:500])
        s.add(row)
        s.commit()
        return row.id


def review_face_embedding(review_id):
    """The stored embedding for one review item, or (None, 0)."""
    import numpy as np
    with SessionLocal() as s:
        row = s.get(ReviewItem, review_id)
        if row is None or not row.face_embedding or not row.face_dims:
            return None, 0
        vec = np.frombuffer(row.face_embedding, dtype=np.float32)
        if vec.size != row.face_dims:
            return None, 0
        return vec, int(row.face_width or 0)


def pending_reviews(limit=60):
    with SessionLocal() as s:
        rows = s.execute(
            select(ReviewItem)
            .where(ReviewItem.status == "pending")
            .order_by(desc(ReviewItem.probability))
            .limit(limit)
        ).scalars().all()
    return [{"id": r.id, "camera": r.camera_name,
             "camera_key": r.camera_key, "track_id": r.track_id,
             "suggested": r.suggested_name, "runner_up": r.runner_up,
             "probability": round(r.probability, 2),
             "margin": round(r.margin, 2), "evidence": r.evidence,
             "body_image": os.path.basename(r.body_image or ""),
             "face_image": os.path.basename(r.face_image or ""),
             "looks_like": r.looks_like or "",
             "candidates": [n for n in (r.candidates or "").split(",") if n],
             "created_at": r.created_at.isoformat()} for r in rows]


def resolve_review(review_id, confirmed_name, status="confirmed"):
    confirmed_name = canonical_name(confirmed_name)
    with SessionLocal() as s:
        row = s.get(ReviewItem, review_id)
        if row is None:
            return None
        row.status = status
        row.confirmed_name = confirmed_name or ""
        row.reviewed_at = datetime.utcnow()
        s.commit()
        return {"id": row.id, "status": row.status,
                "name": row.confirmed_name,
                "body_image": row.body_image,
                "face_image": row.face_image,
                "camera_key": row.camera_key,
                "track_id": row.track_id,
                "created_at": row.created_at}


def recent_confirmations(within_seconds=600):
    """Answers a human gave recently, for the pipeline to act on.

    Deliberately time-based rather than id-based. Review ids are handed
    out when a QUESTION is created, but people answer the cards in
    whatever order they like - an old card confirmed now has a lower id
    than a card created a minute ago. An "id > last seen" cursor would
    therefore skip exactly the answers we care about.
    """
    cutoff = datetime.utcnow() - timedelta(seconds=within_seconds)
    with SessionLocal() as s:
        rows = s.execute(
            select(ReviewItem)
            .where(ReviewItem.status == "confirmed",
                   ReviewItem.reviewed_at.is_not(None),
                   ReviewItem.reviewed_at >= cutoff)
            .order_by(ReviewItem.reviewed_at.asc())
        ).scalars().all()
        out = []
        for r in rows:
            name = r.confirmed_name or r.suggested_name
            out.append({"id": r.id, "camera_key": r.camera_key,
                        "track_id": r.track_id, "name": name,
                        "code": _code_for_name(s, name),
                        "created_at": r.created_at,
                        "reviewed_at": r.reviewed_at})
    return out


def _code_for_name(session, person_name):
    if not person_name:
        return ""
    return session.execute(
        select(Employee.emp_code)
        .where(Employee.name == person_name).limit(1)
    ).scalar_one_or_none() or ""


def code_for_name(person_name):
    """The employee code we have on file for a name, or ""."""
    with SessionLocal() as s:
        return _code_for_name(s, canonical_name(person_name))


def canonical_name(person_name):
    """The ONE official spelling of a name, as enrolled.

    A name typed on the dashboard is the key everything else hangs off:
    the gallery profile, the clothing learned today, the permanent face
    samples. So "shivajyothi" and "Shivajyothi" do not become two
    spellings of one person - they become two PEOPLE, each holding half
    the evidence, and each too weak to be recognised.

    Everything a human types is therefore snapped back to the enrolled
    spelling. Unknown names are only trimmed, never invented or altered,
    so a genuinely new name still passes through unchanged.
    """
    name = (person_name or "").strip()
    if not name:
        return ""
    key = " ".join(name.split()).lower()
    with SessionLocal() as s:
        rows = s.execute(select(Employee.name)).scalars().all()
    for official in rows:
        if " ".join((official or "").split()).lower() == key:
            return official
    return " ".join(name.split())


def apply_confirmed_identity(camera_key, track_id, person_name, emp_code="",
                             when=None, score=1.0, tolerance_seconds=1800):
    """Back-fill a human confirmation onto the row the dashboard shows.

    Without this a confirmation only taught the gallery, so the sighting
    that was actually asked about kept its "Unknown" label for ever and
    the answer looked as though it had been ignored.

    `when` is the moment the QUESTION was created, and it is what makes
    this safe: track ids restart from 1 every time the pipeline starts,
    so matching on camera + track id alone could rename a stranger from
    another day. We only touch a row whose session was actually open
    around that moment.

    Returns {"presence": n, "visits": n} - how many rows were corrected.
    """
    updated = {"presence": 0, "visits": 0}
    if not person_name or track_id is None:
        return updated

    when = when or datetime.utcnow()
    early = when - timedelta(seconds=tolerance_seconds)
    late = when + timedelta(seconds=tolerance_seconds)
    score = float(score)

    with SessionLocal() as s:
        row = s.execute(
            select(Presence)
            .where(Presence.camera_key == camera_key,
                   Presence.track_id == track_id,
                   Presence.first_seen <= late,
                   Presence.last_seen >= early)
            .order_by(desc(Presence.last_seen)).limit(1)
        ).scalar_one_or_none()
        if row is not None:
            row.person_name = person_name
            row.emp_code = emp_code or row.emp_code or ""
            row.match_score = max(score, row.match_score or 0.0)
            row.identified = True
            updated["presence"] = 1

        visit = s.execute(
            select(Visit)
            .where(Visit.camera_key == camera_key,
                   Visit.track_id == track_id,
                   Visit.entered_at <= late,
                   Visit.last_seen >= early)
            .order_by(desc(Visit.last_seen)).limit(1)
        ).scalar_one_or_none()
        if visit is not None:
            visit.person_name = person_name
            visit.emp_code = emp_code or visit.emp_code or ""
            visit.match_score = max(score, visit.match_score or 0.0)
            visit.identified = True
            if visit.identified_at is None:
                visit.identified_at = datetime.utcnow()
            updated["visits"] = 1

        s.commit()
    return updated


def review_counts():
    with SessionLocal() as s:
        rows = s.execute(
            select(ReviewItem.status, func.count())
            .group_by(ReviewItem.status)).all()
    return {status: int(n) for status, n in rows}


# ---------------------------------------------------------- presence
def open_presence(camera_key, camera_name, track_id, when=None):
    when = when or datetime.utcnow()
    with SessionLocal() as s:
        row = Presence(camera_key=camera_key, camera_name=camera_name,
                       track_id=track_id, first_seen=when, last_seen=when,
                       seconds_present=0.0, active=True)
        s.add(row)
        s.commit()
        return row.id


def update_presence(presence_id, seconds, when=None, active=True):
    when = when or datetime.utcnow()
    with SessionLocal() as s:
        row = s.get(Presence, presence_id)
        if row:
            row.seconds_present = float(seconds)
            row.last_seen = when
            row.active = bool(active)
            s.commit()


def identify_presence(presence_id, person_name, emp_code, score):
    """Attach a recognised identity - including retroactively, so the
    time already accumulated belongs to the right person."""
    with SessionLocal() as s:
        row = s.get(Presence, presence_id)
        if row is None:
            return False
        if row.identified and score <= row.match_score:
            return False
        row.person_name = person_name
        row.emp_code = emp_code or ""
        row.match_score = float(score)
        row.identified = True
        s.commit()
        return True


def unidentify_presence(presence_id, was_name=""):
    """Take a name back off a presence row.

    identify_presence() deliberately refuses to lower a score, because
    almost everything that calls it is a weaker piece of evidence
    arriving late. A RETRACTION is the one case that is not: something
    has established the name was wrong, and leaving the minutes filed
    under a person who was never here is the part of a wrong name that
    actually costs somebody.

    The row is not deleted. Somebody WAS present - we simply no longer
    claim to know who, which is the same honesty the rest of the system
    applies to an unrecognised face.
    """
    with SessionLocal() as s:
        row = s.get(Presence, presence_id)
        if row is None:
            return False
        if was_name and row.person_name != was_name:
            return False        # already moved on to somebody else
        row.person_name = "Unknown"
        row.emp_code = ""
        row.match_score = 0.0
        row.identified = False
        s.commit()
        return True


def unidentify_visit(visit_id, was_name=""):
    """Take a name back off a visit row. See unidentify_presence()."""
    with SessionLocal() as s:
        v = s.get(Visit, visit_id)
        if v is None:
            return False
        if was_name and v.person_name != was_name:
            return False
        v.person_name = "Unknown"
        v.emp_code = ""
        v.match_score = 0.0
        v.identified = False
        v.identified_at = None
        s.commit()
        return True


def find_open_presence(camera_key, person_name, within_seconds):
    """A recent presence row for this person, so somebody who steps out
    and comes back continues their session instead of starting a new one."""
    if not person_name or person_name == "Unknown":
        return None
    cutoff = datetime.utcnow() - timedelta(seconds=within_seconds)
    with SessionLocal() as s:
        return s.execute(
            select(Presence.id)
            .where(Presence.camera_key == camera_key,
                   Presence.person_name == person_name,
                   Presence.last_seen >= cutoff)
            .order_by(desc(Presence.last_seen)).limit(1)
        ).scalar_one_or_none()


def close_all_presence():
    with SessionLocal() as s:
        n = s.query(Presence).filter(Presence.active.is_(True)).update(
            {Presence.active: False})
        s.commit()
        return n


def presence_rows(camera_key=None, hours=24, limit=200):
    with SessionLocal() as s:
        q = (select(Presence)
             .where(Presence.first_seen >= _since(hours))
             .order_by(desc(Presence.active), desc(Presence.last_seen))
             .limit(limit))
        if camera_key:
            q = q.where(Presence.camera_key == camera_key)
        rows = s.execute(q).scalars().all()
    return [{"id": r.id, "camera": r.camera_name, "camera_key": r.camera_key,
             "person": r.person_name, "emp_code": r.emp_code,
             "identified": bool(r.identified), "track_id": r.track_id,
             "first_seen": r.first_seen.isoformat(),
             "last_seen": r.last_seen.isoformat(),
             "seconds": round(r.seconds_present or 0, 1),
             "active": bool(r.active)} for r in rows]


def presence_totals(camera_key=None, hours=24):
    """Total time per person - the headline number for a work area."""
    with SessionLocal() as s:
        q = (select(Presence.person_name, Presence.emp_code,
                    func.sum(Presence.seconds_present),
                    func.count(), func.max(Presence.last_seen))
             .where(Presence.first_seen >= _since(hours)))
        if camera_key:
            q = q.where(Presence.camera_key == camera_key)
        rows = s.execute(q.group_by(Presence.person_name,
                                    Presence.emp_code)).all()
    out = [{"person": n, "emp_code": c or "", "seconds": round(float(t or 0), 1),
            "sessions": int(k), "last_seen": (m.isoformat() if m else None)}
           for n, c, t, k, m in rows]
    out.sort(key=lambda r: r["seconds"], reverse=True)
    return out


# ------------------------------------------------------------ queries
def _since(hours):
    return datetime.utcnow() - timedelta(hours=hours)


def summary(hours=24, camera_key=None):
    with SessionLocal() as s:
        q = select(func.count()).select_from(Visit).where(
            Visit.entered_at >= _since(hours))
        if camera_key:
            q = q.where(Visit.camera_key == camera_key)
        total = s.execute(q).scalar_one()

        qi = q.where(Visit.identified.is_(True))
        identified = s.execute(qi).scalar_one()

        qa = select(func.count()).select_from(Visit).where(
            Visit.active.is_(True))
        if camera_key:
            qa = qa.where(Visit.camera_key == camera_key)
        present = s.execute(qa).scalar_one()

    return {
        "window_hours": hours,
        "total_entries": int(total),
        "identified": int(identified),
        "unidentified": int(total) - int(identified),
        "present_now": int(present),
    }


def recent_visits(limit=100, camera_key=None):
    with SessionLocal() as s:
        q = select(Visit).order_by(desc(Visit.entered_at)).limit(limit)
        if camera_key:
            q = q.where(Visit.camera_key == camera_key)
        rows = s.execute(q).scalars().all()
    return [_visit_dict(r) for r in rows]


def active_visits(camera_key=None):
    with SessionLocal() as s:
        q = select(Visit).where(Visit.active.is_(True)).order_by(
            desc(Visit.entered_at))
        if camera_key:
            q = q.where(Visit.camera_key == camera_key)
        rows = s.execute(q).scalars().all()
    return [_visit_dict(r) for r in rows]


def hourly_entries(hours=24, camera_key=None):
    with SessionLocal() as s:
        q = (select(func.strftime("%Y-%m-%d %H:00", Visit.entered_at),
                    func.count())
             .where(Visit.entered_at >= _since(hours)))
        if camera_key:
            q = q.where(Visit.camera_key == camera_key)
        rows = s.execute(
            q.group_by(func.strftime("%Y-%m-%d %H:00", Visit.entered_at))
        ).all()
    return [{"hour": h, "count": int(c)} for h, c in sorted(rows)]


def _visit_dict(v):
    return {
        "id": v.id,
        "camera_key": v.camera_key,
        "camera": v.camera_name,
        "track_id": v.track_id,
        "person": v.person_name,
        "emp_code": v.emp_code,
        "identified": bool(v.identified),
        "score": round(v.match_score or 0, 3),
        "entered_at": v.entered_at.isoformat() if v.entered_at else None,
        "identified_at": (v.identified_at.isoformat()
                          if v.identified_at else None),
        "left_at": v.left_at.isoformat() if v.left_at else None,
        "duration_seconds": round(v.duration_seconds or 0, 1),
        "active": bool(v.active),
    }


# ------------------------------------------------------------ users
def get_user(username):
    with SessionLocal() as s:
        return s.execute(
            select(User).where(User.username == username)
        ).scalar_one_or_none()


def create_user(username, password_hash, role="admin"):
    with SessionLocal() as s:
        s.add(User(username=username, password_hash=password_hash, role=role))
        s.commit()


# ------------------------------------------------------------ employees
def replace_employees(people):
    """people: list of (emp_code, name, photo_count)."""
    with SessionLocal() as s:
        s.query(Employee).delete()
        for code, name, count in people:
            s.add(Employee(emp_code=code, name=name, photo_count=count))
        s.commit()


def list_employees():
    with SessionLocal() as s:
        rows = s.execute(select(Employee).order_by(Employee.name)).scalars().all()
    return [{"emp_code": r.emp_code, "name": r.name,
             "photos": r.photo_count} for r in rows]


def log_system(kind, message):
    with SessionLocal() as s:
        s.add(SystemEvent(kind=kind, message=message))
        s.commit()
