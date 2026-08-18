"""
database/models.py
==================
The tables.

The important design choice is the Visit table. One row = one person's
single stay in front of one camera, from the moment they are first seen
until they leave coverage.

We create the row as soon as the person is confirmed (name unknown), and
UPDATE that same row later when their face is recognised. That is what
makes retroactive naming work: the entry keeps its original arrival time
but gains the correct name once the person walks close enough to be seen.
"""
from datetime import datetime
from sqlalchemy import (Column, Integer, String, Float, DateTime,
                        Boolean, Index, LargeBinary)
from database.db import Base


class User(Base):
    """Dashboard login."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(32), default="operator")
    created_at = Column(DateTime, default=datetime.utcnow)


class Employee(Base):
    """People enrolled from data/faces/<id>-<name>/."""
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    emp_code = Column(String(32), index=True)     # e.g. "12219"
    name = Column(String(128), index=True)        # e.g. "Rajesh Palamangalam"
    photo_count = Column(Integer, default=0)
    enrolled_at = Column(DateTime, default=datetime.utcnow)


class Visit(Base):
    """One person's stay in one camera's view.

    identified_at is NULL while the person is still Unknown. When their
    face is finally recognised we fill in person_name / emp_code /
    identified_at WITHOUT changing entered_at - so the arrival time stays
    honest while the identity is corrected.
    """
    __tablename__ = "visits"
    id = Column(Integer, primary_key=True)

    camera_key = Column(String(64), index=True)
    camera_name = Column(String(128))
    track_id = Column(Integer)                 # per-camera track number

    person_name = Column(String(128), default="Unknown", index=True)
    emp_code = Column(String(32), default="")
    identified = Column(Boolean, default=False)
    match_score = Column(Float, default=0.0)   # best face similarity so far

    entered_at = Column(DateTime, default=datetime.utcnow, index=True)
    identified_at = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, default=datetime.utcnow, index=True)
    left_at = Column(DateTime, nullable=True)

    active = Column(Boolean, default=True, index=True)   # still in view
    duration_seconds = Column(Float, default=0.0)


Index("ix_visit_camera_time", Visit.camera_key, Visit.entered_at)


class Presence(Base):
    """How long one person was present in an AREA camera's zone.

    Different from Visit (which records an arrival). Here we care about
    duration, not events: one row per person per continuous stay, where
    "continuous" tolerates the long gaps you get when somebody is seated
    behind a monitor and only intermittently detectable.

    seconds_present counts time the person was ACTUALLY detected, so a
    long unexplained gap does not inflate their hours.
    """
    __tablename__ = "presence"
    id = Column(Integer, primary_key=True)

    camera_key = Column(String(64), index=True)
    camera_name = Column(String(128))
    track_id = Column(Integer)

    person_name = Column(String(128), default="Unknown", index=True)
    emp_code = Column(String(32), default="")
    identified = Column(Boolean, default=False)
    match_score = Column(Float, default=0.0)

    first_seen = Column(DateTime, default=datetime.utcnow, index=True)
    last_seen = Column(DateTime, default=datetime.utcnow, index=True)
    seconds_present = Column(Float, default=0.0)
    active = Column(Boolean, default=True, index=True)


Index("ix_presence_camera_time", Presence.camera_key, Presence.first_seen)


class ReviewItem(Base):
    """A sighting the system was not sure about, saved for a human.

    This is the "is this you?" queue. When the evidence points at
    somebody but not strongly enough to be trusted, we save the crop and
    the suggestion instead of guessing. Confirming one teaches the system
    permanently, which is how it gets better at the people it sees most.
    """
    __tablename__ = "review_items"
    id = Column(Integer, primary_key=True)

    camera_key = Column(String(64), index=True)
    camera_name = Column(String(128))
    track_id = Column(Integer)

    suggested_name = Column(String(128), index=True)
    runner_up = Column(String(128), default="")
    probability = Column(Float, default=0.0)
    margin = Column(Float, default=0.0)
    evidence = Column(String(256), default="")   # which cues, how strong

    body_image = Column(String(256), default="")  # file path
    face_image = Column(String(256), default="")

    # WHAT THE VISION MODEL SAW, in words, and who that does not rule
    # out. Both come from core/vlm_verify.py and exist for one reason:
    # the reviewer's problem was never that the system stayed silent, it
    # was that answering meant picking one name out of sixty-odd from a
    # crop they could barely make out.
    #
    # looks_like  "male, short hair, beard/moustache" - the description
    #             the screening produced. Also the audit trail: if this
    #             says "female, long hair" and the card is asking about a
    #             man, the screening is what needs looking at.
    # candidates  comma-separated names whose own photographs do not
    #             contradict that description. NARROWING, never
    #             identification - see shortlist() for why the
    #             distinction is enforced rather than described.
    looks_like = Column(String(128), default="")
    candidates = Column(String(512), default="")

    # The face embedding AS THE PIPELINE SAW IT, kept with the question.
    #
    # Confirming used to re-detect a face inside the saved face crop, and
    # that never worked: the crop is a tight box around the face, and a
    # face detector needs context around a face to find one at all
    # (measured: 0 of 25 saved crops). So every confirmation silently
    # taught clothing only, learned_faces stayed empty for ever, and the
    # system never got better at recognising anybody.
    #
    # Keeping the embedding here removes the guesswork - it is the exact
    # vector the recogniser produced from the live frame, so confirming
    # needs no second detection pass and cannot fail this way again.
    face_embedding = Column(LargeBinary, nullable=True)
    face_dims = Column(Integer, default=0)
    face_width = Column(Integer, default=0)      # px, in the original frame

    status = Column(String(16), default="pending", index=True)  # pending/confirmed/rejected
    confirmed_name = Column(String(128), default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    reviewed_at = Column(DateTime, nullable=True)


class LearnedFace(Base):
    """A face sample confirmed by a human, stored so the PIPELINE can use it.

    The dashboard and the pipeline are separate processes and share no
    memory, so a confirmation made in the browser cannot reach the
    running vision engine directly. It goes through the database instead:
    the dashboard writes the embedding here, the pipeline polls for new
    rows and folds them into its gallery within seconds.

    These samples matter more than enrollment photos, because they come
    from the actual cameras rather than a phone.
    """
    __tablename__ = "learned_faces"
    id = Column(Integer, primary_key=True)
    person_name = Column(String(128), index=True)
    emp_code = Column(String(32), default="")
    embedding = Column(LargeBinary)        # float32 vector

    # WHICH MODEL produced this vector, e.g. "adaface:adaface_ir101_*.onnx".
    #
    # Without it these rows are the one path by which incomparable
    # embeddings can reach the live search index. The enrollment FILE is
    # checked at load and refused on a mismatch - but these rows were
    # loaded unconditionally, so switching the recognition model would
    # quietly mix the old model's vectors into the new model's bank.
    # They are the right number of dimensions, so nothing errors; they
    # simply sit at meaningless positions in the new space and can match
    # the wrong person. Rows written before this column existed are
    # blank and are still accepted (see tags_compatible).
    model_tag = Column(String(64), default="")
    dims = Column(Integer, default=512)

    # Where the photograph this vector came from lives on disk, when it
    # came from the camera-captured library (core/face_library.py).
    # Kept so a bad automatic capture can be traced back to a file and
    # deleted, rather than being an anonymous vector nobody can audit.
    image_path = Column(String(256), default="")
    source = Column(String(32), default="review")   # review / enrollment
    camera_key = Column(String(64), default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class LearnedAppearance(Base):
    """NO LONGER WRITTEN OR READ. Kept only so an old database opens.

    This held a colour-and-pattern signature of what a confirmed person
    was wearing that day, and the pipeline used it to name people on a
    camera that could not see their faces. It was the mechanism behind
    the complaint this system was rebuilt to fix: two people overlap,
    their signatures blend, and one person's name ends up on the other
    person's body. Nothing identifies anybody by clothing now - see
    core/body.py for what replaced it.

    The original description follows, for anyone reading an old row.

    What a confirmed person is wearing TODAY, stored so the pipeline
    can use it.

    This is the one that matters in a work area. Faces there are 30-55
    pixels wide and people face their monitors, so a confirmation almost
    never contains a usable face - and if a confirmation can only teach
    faces, confirming a work-area sighting teaches nothing at all and the
    same person gets asked about again a minute later.

    Clothing is different: it is readable from behind, at distance and in
    poor light, and within a single day it identifies somebody almost as
    well as a face. So a confirmation stores the body signature here, the
    pipeline picks it up within seconds, and that person is recognised on
    sight for the rest of the day.

    Rows are scoped to a DAY, because people change clothes.
    """
    __tablename__ = "learned_appearance"
    id = Column(Integer, primary_key=True)
    person_name = Column(String(128), index=True)
    emp_code = Column(String(32), default="")
    descriptor = Column(LargeBinary)       # float32 vector
    dims = Column(Integer, default=0)
    for_day = Column(String(10), index=True)   # YYYY-MM-DD
    camera_key = Column(String(64), default="")
    source = Column(String(32), default="review")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class SystemEvent(Base):
    """Pipeline start/stop and other notable events - useful for auditing."""
    __tablename__ = "system_events"
    id = Column(Integer, primary_key=True)
    kind = Column(String(32))          # START / STOP / ERROR
    message = Column(String(512))
    ts = Column(DateTime, default=datetime.utcnow, index=True)
