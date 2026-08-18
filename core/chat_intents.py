"""
core/chat_intents.py
====================
The questions this system can answer WITHOUT a language model.

Why these are hand-written
--------------------------
The obvious way to build a "ask your CCTV anything" feature is to hand
every question to a model and let it write SQL. That is the wrong trade
for this database, which is an attendance record: a confidently wrong
answer about who was present is worse than an honest "I don't know",
and nobody audits a sentence that reads well.

In practice the questions people ask are a small, knowable set - who is
here, when did somebody arrive, summarise this camera, how many people
are in that room. Those are answered here, by real functions over
database/repository.py. They cannot hallucinate a name, they cannot
misread a timestamp, and they give the same answer twice.

The model is still useful, but for the long tail only - see
core/chat_sql.py. This file is what it falls back FROM, not to.

Adding an intent
----------------
Write a handler that returns an Answer, and list its trigger words in
INTENTS. Matching is deliberately simple and inspectable: a score over
required and optional keywords, with the highest score winning. No
embedding, no model, nothing that can drift between runs - if a question
routes to the wrong handler you can see exactly why by reading the
table.
"""
import re
import difflib
from datetime import datetime, timedelta

from config.settings import TZ_OFFSET_HOURS
from config.cameras import enabled_cameras, CAMERAS
from database import repository as repo


class Answer:
    """What the assistant says back, and what it is based on.

    `rows` is kept alongside the sentence so the dashboard can show a
    table, and so the reasoning behind an answer is inspectable rather
    than being a claim in prose.
    """

    __slots__ = ("text", "rows", "columns", "source", "sql", "chips",
                 "image")

    def __init__(self, text, rows=None, columns=None, source="", sql="",
                 image=""):
        self.text = text
        self.rows = rows or []
        self.columns = columns or []
        self.source = source        # which handler answered
        self.sql = sql              # set only when a query was generated
        # The frame an answer was READ OFF, as a data: URI, when the
        # vision model was involved. Same purpose as `sql` above: an
        # answer about what somebody is doing is the one kind this
        # system produces that cannot be checked against a row, so the
        # picture it came from travels with it.
        self.image = image
        # What to offer next. Filled in by core/chat.py once it knows
        # what this answer turned out to be about, so a handler never
        # has to think about the user interface.
        self.chips = []

    def as_dict(self):
        return {"answer": self.text, "rows": self.rows,
                "columns": self.columns, "source": self.source,
                "sql": self.sql, "chips": self.chips, "image": self.image}

    def __repr__(self):
        return f"<Answer {self.source}: {self.text[:60]}>"


# ---------------------------------------------------------------- time
def local(when):
    """UTC out of the database -> the site's local wall clock."""
    if when is None:
        return None
    if isinstance(when, str):
        try:
            when = datetime.fromisoformat(when)
        except ValueError:
            return None
    return when + timedelta(hours=TZ_OFFSET_HOURS)


def clock(when):
    value = local(when)
    return value.strftime("%H:%M") if value else "unknown"


def stamp(when):
    value = local(when)
    return value.strftime("%d %b %H:%M") if value else "unknown"


def humanise(seconds):
    seconds = int(seconds or 0)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60} min"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def hours_from(question, default=1.0):
    """Pull a time window out of the wording."""
    text = question.lower()
    match = re.search(r"(?:last|past|previous)\s+(\d+)\s*(min|hour|day|week)",
                      text)
    if match:
        value, unit = int(match.group(1)), match.group(2)
        return {"min": value / 60.0, "hour": float(value),
                "day": value * 24.0, "week": value * 168.0}[unit]
    if re.search(r"last\s+(one\s+)?hour|past hour", text):
        return 1.0
    if "half an hour" in text or "30 min" in text:
        return 0.5
    if "today" in text or "so far" in text:
        # since local midnight, expressed as hours back from now
        now = datetime.utcnow() + timedelta(hours=TZ_OFFSET_HOURS)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return max(0.25, (now - midnight).total_seconds() / 3600.0)
    if "yesterday" in text:
        return 48.0
    if "week" in text:
        return 168.0
    return default


# -------------------------------------------------------------- lookup
def camera_catalogue():
    """[(key, name, handler, enabled)] for every configured camera."""
    return [(c["key"], c["name"], c.get("handler", ""), c.get("enabled", True))
            for c in CAMERAS]


def resolve_camera(question):
    """Which camera is being asked about? Returns (key, name) or None.

    Understands the key, the display name, and the "cam2" shorthand
    people actually use, where the number is the position in the
    configured list.
    """
    text = (question or "").lower()
    cameras = camera_catalogue()

    # "cam2", "camera 2", "cam 2"
    match = re.search(r"\bcam(?:era)?\s*[-_ ]?(\d+)\b", text)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(cameras):
            return cameras[index][0], cameras[index][1]

    best, best_score = None, 0.0
    for key, name, _handler, _enabled in cameras:
        for candidate in (key, key.replace("_", " "), name.lower()):
            if candidate and candidate in text:
                score = len(candidate)
                if score > best_score:
                    best, best_score = (key, name), score
    if best:
        return best

    # NOBODY TYPES THE KEY. The camera configured as `server_rm_psg` /
    # "Server Rm Psg" gets asked about as "server psg cam",
    # "server_psg_cam", "the server room" - none of which contain the
    # key or the name as a substring, so the exact test above found
    # nothing and the question fell through to the query generator.
    #
    # So the camera's own words are matched against the question's
    # words: a camera wins if the question uses at least half of the
    # distinctive words in its name, and the most words wins. "server"
    # and "psg" both appear, "rm" does not, and that is a match. Short
    # noise words are dropped so "the" and "cam" cannot carry it.
    asked = set(re.findall(r"[a-z0-9]+", text)) - _CAMERA_NOISE
    for key, name, _handler, _enabled in cameras:
        words = {w for w in re.findall(r"[a-z0-9]+", f"{key} {name}".lower())
                 if len(w) >= 3} - _CAMERA_NOISE
        if not words:
            continue
        hit = words & asked
        if len(hit) * 2 >= len(words) and len(hit) >= 1:
            score = len(hit)
            if score > best_score:
                best, best_score = (key, name), score
    return best


# Words that appear in a camera's name AND in half the questions asked
# about it, so they cannot be evidence of which camera is meant.
_CAMERA_NOISE = {"cam", "cams", "camera", "cameras", "the", "room", "rm",
                 "area", "feed", "view", "lobby"}


# A name-part shared by more than this many enrolled people identifies
# nobody. "Kumar" is in six of the sixty-eight names on this site, so a
# question mentioning it has not named anybody - and answering about one
# of the six is worse than saying no name was recognised, because it
# reads as an answer.
_MAX_AMBIGUOUS = 3


def _name_parts(people):
    """{name-part: [everybody whose name contains it]}."""
    index = {}
    for name in people:
        for part in name.lower().split():
            if len(part) >= 3:
                index.setdefault(part, []).append(name)
    return index


def _best_for_part(candidates, part):
    """Which of several people a name-part most likely means.

    Deliberately simple and in this order: somebody's FIRST name beats
    the same word appearing later in a longer name (people are called by
    their first names, so "pavan" is Pavan Kumar rather than Sudha
    Pavan), then the shorter full name.
    """
    def rank(name):
        words = name.lower().split()
        return (0 if words and words[0] == part else 1, len(name))
    return min(candidates, key=rank)


def resolve_people(question):
    """EVERY enrolled person named in a question, in the order asked.

    resolve_person() returns the single best match, which is right for
    "when did Deepthi arrive?". It is wrong for "pavan ayeesha what are
    they doing", where the whole point of the question is that there are
    two of them - and that phrasing is common enough here to be worth
    handling rather than quietly answering about one of the pair.

    THE HARD PART IS NOT FINDING NAMES, IT IS NOT FINDING TOO MANY.
    Sixty-eight enrolled people share a small pool of name-parts: six of
    them are a "Kumar", two are a "Pavan". Matching every part of every
    name against the question - which is what finding names the obvious
    way does - turns "pavan kumar and ayesha fatima" into a list of
    seven people, most of whom were never mentioned.

    So the question is consumed left to right:
      1. a FULL NAME in the text wins outright and takes those
         characters with it, so "kumar" in "Pavan Kumar" can never go on
         to match five other people;
      2. each remaining word resolves to at most one person, and to
         nobody at all if it is shared too widely to mean anything;
      3. a word that matched nothing exactly gets one fuzzy attempt, so
         "ayeesha" still finds Ayesha Fatima.
    """
    text = (question or "").lower()
    people = known_people()
    if not people:
        return []

    found = []              # (position in the question, name)
    consumed = []           # spans already explained by a full name

    for name in people:
        lowered = name.lower()
        at = text.find(lowered)
        if at >= 0:
            found.append((at, name))
            consumed.append((at, at + len(lowered)))

    parts = _name_parts(people)
    taken = {name for _at, name in found}
    ambiguous = False

    for match in re.finditer(r"[a-z]{3,}", text):
        word = match.group()
        if word in _STOPWORDS:
            continue
        if any(start <= match.start() and match.end() <= end
               for start, end in consumed):
            continue

        part = word if word in parts else ""
        if not part and len(word) >= 4:
            close = difflib.get_close_matches(word, list(parts), n=1,
                                              cutoff=0.85)
            part = close[0] if close else ""
        if not part:
            continue

        # The ambiguity test counts EVERYBODY sharing the part, not only
        # the ones still available: "kumar" is no more informative
        # because one of the six has already been named.
        if len(parts[part]) > _MAX_AMBIGUOUS:
            ambiguous = True
            continue
        candidates = [n for n in parts[part] if n not in taken]
        if not candidates:
            continue
        best = _best_for_part(candidates, part)
        found.append((match.start(), best))
        taken.add(best)

    if found:
        found.sort()
        return [name for _at, name in found]

    # Nothing matched at all. resolve_person's fuzzy pass is more
    # generous than the one above and is worth one last try - a single
    # misspelt name is by far the commonest case.
    #
    # NOT when the question leant on a name-part shared by half the
    # office, though. resolve_person will happily return one of the six
    # Kumars for "what is kumar doing", and the whole point of the
    # guard above is that answering that question about a particular
    # person is worse than admitting no name was recognised.
    if ambiguous:
        return []
    one = resolve_person(question)
    return [one] if one else []


def known_people():
    try:
        return [e["name"] for e in repo.list_employees() if e.get("name")]
    except Exception:
        return []


def resolve_person(question):
    """Which person is being asked about? Returns the enrolled name.

    Matches on any word of their name, so "deepthi" finds "Deepthi
    Gangireddy Gari", and falls back to fuzzy matching for a misspelling.
    """
    text = (question or "").lower()
    people = known_people()
    if not people:
        return None

    best, best_score = None, 0.0
    for name in people:
        lowered = name.lower()
        if lowered in text:
            score = len(lowered) + 10          # full name wins outright
        else:
            score = 0.0
            for part in lowered.split():
                if len(part) >= 3 and re.search(rf"\b{re.escape(part)}\b", text):
                    score = max(score, len(part))
        if score > best_score:
            best, best_score = name, score
    if best:
        return best

    # nothing matched outright - try a fuzzy match on the longest word,
    # so "deepti" still finds "Deepthi"
    words = [w for w in re.findall(r"[a-z]{4,}", text)
             if w not in _STOPWORDS]
    for word in sorted(words, key=len, reverse=True):
        close = difflib.get_close_matches(
            word, [p.lower() for p in people], n=1, cutoff=0.8)
        if close:
            for name in people:
                if name.lower() == close[0]:
                    return name
            # matched a first name inside a full name
        for name in people:
            for part in name.lower().split():
                if difflib.SequenceMatcher(None, word, part).ratio() >= 0.85:
                    return name
    return None


_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "many", "much",
    "time", "first", "last", "entry", "entries", "today", "yesterday", "now",
    "present", "camera", "cameras", "running", "there", "here", "show",
    "list", "tell", "give", "summary", "people", "person", "members",
    "member", "count", "total", "hours", "hour", "minutes", "cams", "cam",
    "into", "from", "about", "have", "been", "does", "area", "room", "the",
}


# ------------------------------------------------------------ handlers
def answer_cameras(question, **_):
    cameras = camera_catalogue()
    live = [c for c in cameras if c[3]]
    lines = [f"{name} ({key}, {handler or 'camera'})"
             for key, name, handler, enabled in cameras if enabled]
    off = [name for _k, name, _h, enabled in cameras if not enabled]
    text = (f"{len(live)} camera(s) running: " + "; ".join(lines)) if live \
        else "No cameras are enabled."
    if off:
        text += f". Disabled: {', '.join(off)}"
    return Answer(text,
                  rows=[{"camera": name, "key": key, "type": handler,
                         "enabled": enabled}
                        for key, name, handler, enabled in cameras],
                  columns=["camera", "key", "type", "enabled"],
                  source="cameras")


def _present_rows(camera_key=None):
    """Everyone currently in view, from both kinds of camera.

    Reception records ARRIVALS (visits); a work area records PRESENCE.
    A question like "who is here" means both, so both are read and
    merged rather than the caller having to know which camera is which.
    """
    people = {}
    try:
        for visit in repo.active_visits(camera_key):
            name = visit.get("person") or "Unknown"
            people.setdefault(name, {
                "person": name, "camera": visit.get("camera", ""),
                "since": stamp(visit.get("entered_at")), "for": None})
    except Exception:
        pass
    try:
        for row in repo.presence_rows(camera_key, hours=24, limit=400):
            if not row.get("active"):
                continue
            name = row.get("person") or "Unknown"
            entry = people.setdefault(name, {
                "person": name, "camera": row.get("camera", ""),
                "since": stamp(row.get("first_seen")), "for": None})
            entry["for"] = humanise(row.get("seconds"))
            if not entry.get("camera"):
                entry["camera"] = row.get("camera", "")
    except Exception:
        pass
    return list(people.values())


def answer_who_present(question, camera=None, **_):
    key, name = camera if camera else (None, None)
    rows = _present_rows(key)
    named = [r for r in rows if r["person"] != "Unknown"]
    unknown = len(rows) - len(named)
    where = f" in {name}" if name else ""

    if not rows:
        return Answer(f"Nobody is in view{where} right now.",
                      source="who_present")

    parts = []
    for row in named:
        detail = f" (for {row['for']})" if row.get("for") else ""
        parts.append(f"{row['person']}{detail}")
    if named:
        text = (f"{len(named)} identified person(s){where} right now: "
                + ", ".join(parts))
    else:
        text = f"Somebody is{where} in view, but nobody is identified yet"
    if unknown:
        text += f", plus {unknown} unidentified"
    return Answer(text + ".", rows=rows,
                  columns=["person", "camera", "since", "for"],
                  source="who_present")


def answer_count_present(question, camera=None, **_):
    key, name = camera if camera else (None, None)
    rows = _present_rows(key)
    named = [r for r in rows if r["person"] != "Unknown"]
    where = f" in {name}" if name else ""
    text = (f"{len(rows)} person(s){where} right now"
            f"{f' - {len(named)} identified: ' + ', '.join(r['person'] for r in named) if named else ''}.")
    return Answer(text, rows=rows,
                  columns=["person", "camera", "since", "for"],
                  source="count_present")


def _entries_for(person, hours=24.0):
    """Every time this person was SEEN, from both kinds of camera.

    Reception records arrivals in `visits`; a work area records
    `presence` sessions. Reading only visits meant somebody who works in
    the server room and never passes reception had "no recorded entry"
    however many hours the cameras had watched them - which is wrong in
    the most confusing way, because the dashboard was showing them at
    the same moment.
    """
    wanted = (person or "").lower()
    sightings = []
    try:
        for visit in repo.recent_visits(limit=500):
            if (visit.get("person") or "").lower() == wanted:
                sightings.append({
                    "entered_at": visit.get("entered_at"),
                    "camera_name": visit.get("camera", ""),
                    "duration_seconds": visit.get("duration_seconds") or 0,
                    "kind": "entry"})
    except Exception:
        pass
    try:
        for row in repo.presence_rows(None, hours=max(hours, 24.0), limit=500):
            if (row.get("person") or "").lower() == wanted:
                sightings.append({
                    "entered_at": row.get("first_seen"),
                    "camera_name": row.get("camera", ""),
                    "duration_seconds": row.get("seconds") or 0,
                    "kind": "presence"})
    except Exception:
        pass
    return [s for s in sightings if s.get("entered_at")]


def answer_first_entry(question, person=None, **_):
    if not person:
        return Answer("Tell me whose entry you mean - I did not recognise a "
                      "name in that question.", source="first_entry")
    visits = _entries_for(person, hours_from(question, 24.0))
    if not visits:
        return Answer(f"No recorded entry for {person}.", source="first_entry")
    first = min(visits, key=lambda v: v.get("entered_at") or "")
    return Answer(
        f"{person} was first seen at {stamp(first.get('entered_at'))} "
        f"on {first.get('camera_name', 'a camera')}.",
        rows=[{"person": person, "first_entry": stamp(first.get("entered_at")),
               "camera": first.get("camera_name", "")}],
        columns=["person", "first_entry", "camera"], source="first_entry")


def answer_last_entry(question, person=None, **_):
    if not person:
        return Answer("Tell me whose entry you mean - I did not recognise a "
                      "name in that question.", source="last_entry")
    visits = _entries_for(person, hours_from(question, 24.0))
    if not visits:
        return Answer(f"No recorded entry for {person}.", source="last_entry")
    last = max(visits, key=lambda v: v.get("entered_at") or "")
    return Answer(
        f"{person} was last seen entering at {stamp(last.get('entered_at'))} "
        f"on {last.get('camera_name', 'a camera')}.",
        rows=[{"person": person, "last_entry": stamp(last.get("entered_at")),
               "camera": last.get("camera_name", "")}],
        columns=["person", "last_entry", "camera"], source="last_entry")


def answer_all_entries(question, person=None, **_):
    if not person:
        return Answer("Whose entries would you like?", source="entries")
    visits = sorted(_entries_for(person, hours_from(question, 24.0)),
                    key=lambda v: v.get("entered_at") or "")
    if not visits:
        return Answer(f"No recorded entry for {person}.", source="entries")
    rows = [{"time": stamp(v.get("entered_at")),
             "camera": v.get("camera_name", ""),
             "duration": humanise(v.get("duration_seconds"))} for v in visits]
    return Answer(
        f"{person} has {len(rows)} recorded entry(ies): "
        f"{', '.join(r['time'] for r in rows[:8])}"
        f"{' ...' if len(rows) > 8 else ''}.",
        rows=rows, columns=["time", "camera", "duration"], source="entries")


def answer_camera_summary(question, camera=None, **_):
    hours = hours_from(question, 1.0)
    key, name = camera if camera else (None, "all cameras")
    try:
        data = repo.summary(hours=hours, camera_key=key)
    except Exception as exc:
        return Answer(f"Could not read the summary: {exc}",
                      source="camera_summary")

    totals = repo.presence_totals(key, hours=hours) or []
    window = (f"{int(hours * 60)} minutes" if hours < 1
              else f"{hours:.0f} hour(s)")
    bits = [f"In the last {window} on {name}:"]
    for label, value in data.items():
        bits.append(f"{str(label).replace('_', ' ')} {value}")
    text = " ".join(bits[:1]) + " " + ", ".join(bits[1:]) + "."
    rows = [{"person": t.get("person", "Unknown"),
             "time_present": humanise(t.get("seconds"))}
            for t in totals[:20]]
    if rows:
        text += (" People present: "
                 + ", ".join(f"{r['person']} ({r['time_present']})"
                             for r in rows[:8]) + ".")
    return Answer(text, rows=rows, columns=["person", "time_present"],
                  source="camera_summary")


def answer_person_time(question, person=None, camera=None, **_):
    if not person:
        return Answer("Whose time would you like?", source="person_time")
    hours = hours_from(question, 24.0)
    key = camera[0] if camera else None
    totals = repo.presence_totals(key, hours=hours) or []
    wanted = person.lower()
    mine = [t for t in totals
            if (t.get("person") or "").lower() == wanted]
    if not mine:
        return Answer(f"No presence recorded for {person} in that period.",
                      source="person_time")
    seconds = sum(t.get("seconds") or 0 for t in mine)
    return Answer(
        f"{person} has been present for {humanise(seconds)}"
        f"{f' on {camera[1]}' if camera else ''}.",
        rows=[{"person": person, "time_present": humanise(seconds)}],
        columns=["person", "time_present"], source="person_time")


def answer_is_here(question, person=None, **_):
    if not person:
        return Answer("Who would you like me to look for?", source="is_here")
    rows = _present_rows()
    for row in rows:
        if row["person"].lower() == person.lower():
            detail = f" for {row['for']}" if row.get("for") else ""
            return Answer(f"Yes - {person} is in view on "
                          f"{row['camera']}{detail}.",
                          rows=[row],
                          columns=["person", "camera", "since", "for"],
                          source="is_here")
    return Answer(f"No, {person} is not in view on any camera right now.",
                  source="is_here")


def answer_top_presence(question, camera=None, **_):
    hours = hours_from(question, 24.0)
    key = camera[0] if camera else None
    totals = repo.presence_totals(key, hours=hours) or []
    named = [t for t in totals
             if (t.get("person") or "Unknown") != "Unknown"]
    if not named:
        return Answer("No identified presence recorded in that period.",
                      source="top_presence")
    named.sort(key=lambda t: -(t.get("seconds") or 0))
    rows = [{"person": t["person"],
             "time_present": humanise(t.get("seconds"))}
            for t in named[:10]]
    return Answer(
        "Most time present: "
        + ", ".join(f"{r['person']} ({r['time_present']})" for r in rows[:5])
        + ".", rows=rows, columns=["person", "time_present"],
        source="top_presence")


def answer_absent(question, **_):
    hours = hours_from(question, 24.0)
    totals = repo.presence_totals(None, hours=hours) or []
    seen = {(t.get("person") or "").lower() for t in totals}
    try:
        for visit in repo.recent_visits(limit=500):
            seen.add((visit.get("person") or "").lower())
    except Exception:
        pass
    missing = [n for n in known_people() if n.lower() not in seen]
    if not missing:
        return Answer("Everybody enrolled has been seen in that period.",
                      source="absent")
    return Answer(
        f"{len(missing)} enrolled person(s) not seen: "
        f"{', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''}.",
        rows=[{"person": n} for n in missing], columns=["person"],
        source="absent")


def answer_employees(question, **_):
    people = known_people()
    return Answer(f"{len(people)} people are enrolled: "
                  f"{', '.join(people[:12])}"
                  f"{' ...' if len(people) > 12 else ''}.",
                  rows=[{"person": n} for n in people], columns=["person"],
                  source="employees")


def answer_reviews(question, **_):
    counts = repo.review_counts() or {}
    pending = counts.get("pending", 0)
    return Answer(
        f"{pending} sighting(s) waiting to be confirmed "
        f"({counts.get('confirmed', 0)} confirmed, "
        f"{counts.get('rejected', 0)} rejected so far).",
        rows=[{"status": k, "count": v} for k, v in counts.items()],
        columns=["status", "count"], source="reviews")


def answer_entries_count(question, camera=None, **_):
    hours = hours_from(question, 24.0)
    key = camera[0] if camera else None
    try:
        data = repo.summary(hours=hours, camera_key=key)
    except Exception as exc:
        return Answer(f"Could not read the entries: {exc}",
                      source="entries_count")
    where = f" on {camera[1]}" if camera else ""
    window = ("today" if "today" in question.lower()
              else f"in the last {hours:.0f} hour(s)")
    total = data.get("total_entries", 0)
    return Answer(f"{total} entry(ies) recorded{where} {window}.",
                  rows=[{"metric": k, "value": v} for k, v in data.items()],
                  columns=["metric", "value"], source="entries_count")


def answer_status(question, **_):
    cameras = [c for c in enabled_cameras()]
    counts = repo.review_counts() or {}
    people = known_people()
    return Answer(
        f"{len(cameras)} camera(s) configured and enabled, "
        f"{len(people)} people enrolled, "
        f"{counts.get('pending', 0)} confirmation(s) waiting.",
        rows=[{"item": "cameras", "value": len(cameras)},
              {"item": "enrolled people", "value": len(people)},
              {"item": "pending confirmations",
               "value": counts.get("pending", 0)}],
        columns=["item", "value"], source="status")


def answer_recent_activity(question, camera=None, **_):
    hours = hours_from(question, 1.0)
    key = camera[0] if camera else None
    visits = repo.recent_visits(limit=60, camera_key=key)
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    recent = []
    for visit in visits:
        when = visit.get("entered_at")
        if isinstance(when, str):
            try:
                when = datetime.fromisoformat(when)
            except ValueError:
                when = None
        if when and when >= cutoff:
            recent.append({"time": stamp(visit.get("entered_at")),
                           "person": visit.get("person") or "Unknown",
                           "camera": visit.get("camera", "")})
    if not recent:
        return Answer("Nothing recorded in that period.",
                      source="recent_activity")
    return Answer(
        f"{len(recent)} sighting(s): "
        + ", ".join(f"{r['person']} at {r['time']}" for r in recent[:8])
        + ".", rows=recent, columns=["time", "person", "camera"],
        source="recent_activity")


def answer_entry_list(question, camera=None, **_):
    """Who entered, on which camera, within a window - with details.

    Distinct from answer_entries_count, which returns a number. This is
    the "give me the list" question, and it was previously answered by
    the CAMERA LIST handler because the word "cam" appeared in it.
    """
    hours = hours_from(question, 1.0)
    key, name = camera if camera else (None, "all cameras")
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    rows = []
    try:
        for visit in repo.recent_visits(limit=500, camera_key=key):
            when = visit.get("entered_at")
            if isinstance(when, str):
                try:
                    when = datetime.fromisoformat(when)
                except ValueError:
                    when = None
            if not when or when < cutoff:
                continue
            rows.append({
                "person": visit.get("person") or "Unknown",
                "entered": stamp(visit.get("entered_at")),
                "camera": visit.get("camera", ""),
                "stayed": humanise(visit.get("duration_seconds")),
            })
    except Exception as exc:
        return Answer(f"Could not read the entries: {exc}",
                      source="entry_list")

    if not rows:
        return Answer(f"Nobody entered {name} in that period.",
                      source="entry_list")

    named = [r for r in rows if r["person"] != "Unknown"]
    unknown = len(rows) - len(named)
    window = (f"{int(hours * 60)} minutes" if hours < 1
              else f"{hours:.0f} hour(s)")
    who = ", ".join(sorted({r["person"] for r in named}))
    text = (f"{len(rows)} entry(ies) on {name} in the last {window}"
            f"{f' - {who}' if named else ''}"
            f"{f', plus {unknown} unidentified' if unknown else ''}.")
    # identified first, most recent first - the useful ones at the top
    rows.sort(key=lambda r: (r["person"] == "Unknown", r["entered"]),
              reverse=False)
    return Answer(text, rows=rows,
                  columns=["person", "entered", "camera", "stayed"],
                  source="entry_list")


# ------------------------------------------- what are they DOING
# The three handlers below are the only ones in this file that do not
# answer from the database. They answer from a LIVE FRAME, by asking a
# vision model what each tracked person is doing - see core/vlm_scene.py
# for the division of labour that makes that safe (who from the tracker,
# what from the model, joined in code).
#
# They are still handlers, not a fallback to the query generator, for
# exactly the reason the rest of this file exists: "who is working" has
# a right answer that can be produced deterministically apart from the
# one word the model contributes, and a generated SELECT cannot produce
# it at all - there is no column for what somebody is doing.
#
# EVERY ONE OF THEM WORKS WITHOUT THE VISION MODEL. With Ollama down
# they degrade to who is in view and for how long, and say that the
# activity is not available. That is the same trade the rest of the VLM
# integration makes: the feature going quiet is acceptable, the feature
# inventing something is not.

def _look_targets(camera):
    """Which cameras a live question should look at.

    A named camera wins. Without one every enabled camera is looked at,
    which is affordable because a look is cached per camera for
    VLM_SCENE_CACHE_SECONDS and this site has two.
    """
    if camera:
        return [camera]
    return [(c["key"], c["name"]) for c in enabled_cameras()]


def _share_of_budget(targets):
    """The wall-clock each camera gets when several are being looked at.

    The person waiting is waiting for the WHOLE answer, so the budget is
    the total and the cameras divide it. Applied per camera instead, two
    cameras would take twice VLM_SCENE_BUDGET_SECONDS - a minute and a
    half - and the request would time out somewhere between here and
    the browser with nothing to show for the GPU time.
    """
    from config.settings import VLM_SCENE_BUDGET_SECONDS
    return VLM_SCENE_BUDGET_SECONDS / max(1, len(targets))


def _camera_holding(names):
    """Where these people are right now: ((key, name) or None, looked).

    Checked from the SNAPSHOT, which costs nothing - no vision model is
    involved in finding out WHERE somebody is, only in finding out what
    they are doing. So a question about one person does not pay to have
    every other camera described first.

    `looked` is the second half of the answer and is not a detail. "They
    are not in view" and "I could not look" are completely different
    replies, and returning only the camera makes them indistinguishable
    - which is how a stopped pipeline reports that everybody has gone
    home.
    """
    from core import vlm_scene
    wanted = {n.lower() for n in names if n}
    looked = False
    for key, name in _look_targets(None):
        scene = vlm_scene.snapshot(key)
        if not scene:
            continue
        looked = True
        for entry in scene.get("people") or []:
            if (entry.get("person") or "").lower() in wanted:
                return (key, name), True
    return None, looked


def _no_view_answer(reports, source):
    """One message for "I could not look at any camera".

    Said once, not once per camera. Two cameras with no snapshots is one
    fault - the pipeline is not running - and printing the same sentence
    twice with different camera names in it reads as two.

    Compared on the REASON rather than on the wording, because the
    wording names the camera and so is never equal.
    """
    if not reports:
        return Answer("No cameras are enabled.", source=source)
    reasons = {r.reason for r in reports}
    if len(reports) > 1 and reasons == {"no_snapshot"}:
        return Answer(
            "I cannot look at any camera right now - the pipeline is not "
            "publishing frames. Start run_pipeline.py, then ask again.",
            source=source)
    return Answer(" ".join(r.error for r in reports if r.error),
                  source=source)


def _blind_spots(reports):
    """A note about the cameras that could NOT be looked at.

    Only reached when at least one camera answered, and it matters most
    then: "nobody is working" is a very different statement depending on
    whether it covers both cameras or one. Silently answering from
    whatever happened to be available is how a half-answer gets read as
    a whole one.
    """
    down = [r.camera_name or r.camera_key for r in reports if not r.ok]
    if not down:
        return ""
    return (f" I could not look at {', '.join(down)}, so this does not "
            f"cover {'it' if len(down) == 1 else 'them'}.")


def _activity_rows(reports):
    rows = []
    for report in reports:
        for person in report.people:
            row = person.row()
            row["camera"] = report.camera_name
            rows.append(row)
    return rows


def answer_scene(question, camera=None, **_):
    """What is happening on a camera right now - by looking at it."""
    from core import vlm_scene

    targets = _look_targets(camera)
    if not targets:
        return Answer("No cameras are enabled.", source="scene")

    share = _share_of_budget(targets)
    reports = [vlm_scene.look(key, name, describe_scene=True, budget=share)
               for key, name in targets]
    working = [r for r in reports if r.ok]
    if not working:
        return _no_view_answer(reports, "scene")

    text = " ".join(r.sentence() for r in working) + _blind_spots(reports)
    image = ""
    for report in working:
        image = vlm_scene.image_data_uri(report)
        if image:
            break
    return Answer(text, rows=_activity_rows(working),
                  columns=["person", "doing", "for", "camera", "note"],
                  source="scene", image=image)


def answer_person_activity(question, person=None, **_):
    """What is this person - or these people - doing right now?"""
    from core import vlm_scene

    # resolve_people, NOT the `person` the router already found. The
    # router's version answers "who is this question mostly about",
    # which is the right question for "when did Deepthi arrive" and the
    # wrong one here twice over: it drops the second of two people, and
    # it returns one of the six Kumars for a question that named none of
    # them. Falling back to it would undo both guards.
    names = resolve_people(question)
    if not names:
        return Answer("Tell me who you mean - I did not recognise an "
                      "enrolled name in that question. If you used just a "
                      "surname, try the full name.",
                      source="person_activity")

    found, looked = _camera_holding(names)
    if not looked:
        return Answer("I cannot look at any camera right now - the "
                      "pipeline is not publishing frames. Start "
                      "run_pipeline.py, then ask again.",
                      source="person_activity")
    if found is None:
        # DETERMINISTIC, and worth being careful about: not being in
        # view is a fact about the tracker, not something the vision
        # model was asked. Saying "I could not see them" when the answer
        # is "they are not there" - or the reverse, which is what a
        # stopped pipeline used to produce here - is the difference
        # between a useful answer and a misleading one.
        who = ", ".join(names)
        return Answer(f"{who} is not in view on any camera right now, so "
                      f"there is nothing to look at."
                      if len(names) == 1 else
                      f"None of {who} are in view on any camera right now.",
                      source="person_activity")

    key, camera_name = found
    report = vlm_scene.look(key, camera_name, focus=names,
                            describe_scene=False)
    if not report.ok:
        return Answer(report.error, source="person_activity")

    said, missing = [], []
    for name in names:
        who = report.find(name)
        if who is None:
            missing.append(name)
            continue
        bit = f"{name} is {who.doing} on {camera_name}"
        if who.seconds:
            bit += f" ({humanise(who.seconds)} present)"
        if who.note and who.checked:
            bit += f" - {who.note}"
        said.append(bit)

    text = ". ".join(said)
    if text:
        text += "."
    if missing:
        text += (f" {', '.join(missing)} is not in view right now."
                 if len(missing) == 1
                 else f" {', '.join(missing)} are not in view right now.")
    rows = [p.row() for p in report.people
            if p.person.lower() in {n.lower() for n in names}]
    return Answer(text.strip(), rows=rows or _activity_rows([report]),
                  columns=["person", "doing", "for", "note"],
                  source="person_activity",
                  image=vlm_scene.image_data_uri(report))


def answer_working(question, camera=None, **_):
    """Who is working right now, and who is not.

    THE HONEST ANSWER HAS THREE PARTS, NOT TWO. "Working" and "not
    working" are what somebody asks for; "cannot tell" is what a CCTV
    camera very often supports, and folding it into either of the other
    two is how this kind of feature ends up making claims about people
    that the picture never justified. Somebody on the phone is doing
    neither, and is reported as being on the phone.
    """
    from core import vlm_scene

    targets = _look_targets(camera)
    share = _share_of_budget(targets)
    reports = [vlm_scene.look(key, name, describe_scene=False, budget=share)
               for key, name in targets]
    live = [r for r in reports if r.ok]
    if not live:
        return _no_view_answer(reports, "working")

    busy, idle, other = [], [], []
    for report in live:
        for who in report.people:
            entry = (who.label, who.doing, report.camera_name)
            if who.working is True:
                busy.append(entry)
            elif who.working is False:
                idle.append(entry)
            else:
                other.append(entry)

    if not (busy or idle or other):
        return Answer("Nobody is in view on any camera right now."
                      + _blind_spots(reports), source="working")

    parts = []
    if busy:
        parts.append(f"Working ({len(busy)}): "
                     + ", ".join(f"{n} on {c}" for n, _d, c in busy))
    if idle:
        parts.append(f"Not working ({len(idle)}): "
                     + ", ".join(f"{n} - {d}" for n, d, _c in idle))
    if other:
        parts.append(f"Cannot say ({len(other)}): "
                     + ", ".join(f"{n} - {d}" for n, d, _c in other))
    text = ". ".join(parts) + "." + _blind_spots(reports)

    image = ""
    for report in live:
        image = vlm_scene.image_data_uri(report)
        if image:
            break
    return Answer(text, rows=_activity_rows(live),
                  columns=["person", "doing", "for", "camera", "note"],
                  source="working", image=image)


# ------------------------------------------------------------ routing
# (handler, required groups, helpful words, disqualifying words,
#  needs_person, needs_camera)
INTENTS = [
    (answer_cameras,
     [["camera", "cameras", "cams", "cam"]],
     ["how many", "what are", "running", "list", "configured"],
     # A question that merely MENTIONS a camera is usually about the
     # people on it. "last one hour entered persons details in reception
     # lobby cam" was being answered with the camera list, because "cam"
     # was the only word this intent needed.
     ["most", "least", "busiest", "highest", "lowest", "ever", "total",
      "entered", "person", "persons", "people", "present", "who",
      "details", "summary", "entries", "entry"],
     False, False),

    (answer_entry_list,
     [["entered", "entries", "entry", "arrivals", "came in", "visitors"],
      ["details", "list", "who", "persons", "people", "show", "names"]],
     ["last", "hour", "today", "cam", "camera"],
     ["how many", "count"], False, False),

    # ---- the three that LOOK at a live frame -----------------------
    # Listed above the presence handlers on purpose. "Who is working"
    # and "who is here" are different questions, and the words that
    # separate them are in these entries' required groups and in the
    # older entries' avoid lists - so the routing table itself says
    # which is which, rather than the answer depending on list order.
    (answer_scene,
     [["happening", "going on", "scene", "look like", "looks like",
       "situation", "on camera", "on the camera", "in the frame",
       "live view", "describe"]],
     ["what", "now", "right now", "currently", "camera", "cam", "there"],
     ["happened", "how many", "how long", "entered", "entries", "enrolled",
      "yesterday", "summary", "last hour", "who has"],
     False, True),

    (answer_person_activity,
     [["doing", "up to", "working", "busy", "activity", "what is",
       "what's", "whats", "what are"]],
     ["now", "right now", "currently", "they", "them", "at the moment"],
     ["entered", "entries", "enrolled", "how many", "how long", "first",
      "last entry", "yesterday", "summary", "longest"],
     True, False),

    (answer_working,
     [["working", "work", "busy", "idle", "active"],
      ["who", "which", "anybody", "anyone", "everybody", "everyone"]],
     ["now", "right now", "currently", "not", "camera", "cam"],
     ["how long", "entered", "entries", "enrolled", "yesterday",
      "last hour", "hours", "longest", "most"],
     False, False),

    (answer_is_here,
     [["is", "has", "was"],
      ["here", "present", "around", "still", "in view", "in office"]],
     ["right now", "currently", "today"],
     ["most", "how many", "which camera", "doing", "working"], True, False),

    (answer_who_present,
     [["who"]], ["present", "now", "currently", "in", "there", "view"],
     # "who is working" is a question about what people are DOING, and
     # it used to land here - "in" is a substring of "working", which
     # was enough to score it. It is a different handler now.
     ["most", "longest", "not come", "absent", "enrolled", "working",
      "busy", "idle", "doing"], False, False),

    (answer_count_present,
     [["how many"], ["present", "now", "currently", "there", "inside",
                     "in view", "in cam", "in the"]],
     ["members", "people", "persons"],
     ["entered", "entries", "enrolled", "pending"], False, False),

    (answer_first_entry,
     [["first"]], ["entry", "entered", "arrive", "arrived", "time", "came"],
     [], True, False),

    (answer_last_entry,
     [["last", "latest", "recent"]],
     ["entry", "entered", "arrive", "arrived", "time", "seen"],
     ["summary", "minutes"], True, False),

    (answer_all_entries,
     [["entries", "entry"]], ["all", "list", "show", "every", "times"],
     [], True, False),

    (answer_camera_summary,
     [["summary", "summarise", "summarize", "report", "overview"]],
     ["hour", "today", "last", "camera", "cam"],
     [], False, False),

    (answer_person_time,
     [["how long", "how much time", "duration", "total time"]],
     ["present", "been", "spent", "stayed", "today"],
     [], True, False),

    (answer_top_presence,
     [["most", "longest", "top"]],
     ["time", "present", "spent", "hours", "stayed"],
     ["camera", "cams", "visits"], False, False),

    (answer_absent,
     [["absent", "not come", "missing", "not seen", "did not come"]],
     ["today", "who", "anybody", "anyone"],
     [], False, False),

    (answer_employees,
     [["enrolled", "employees", "staff", "registered"]],
     ["how many", "list", "who", "people", "all"],
     [], False, False),

    (answer_reviews,
     [["confirm", "confirmation", "confirmations", "review", "reviews",
       "pending"]],
     ["how many", "waiting", "queue"],
     [], False, False),

    (answer_entries_count,
     [["how many"], ["entered", "entries", "entry", "arrivals", "came in",
                     "visited", "arrived", "visits"]],
     ["today", "people", "total"],
     ["most", "which camera"], False, False),

    (answer_status,
     [["status", "everything", "system", "health", "working"]],
     ["is", "running", "ok", "fine"],
     # "is everything working" is about the system; "who is working" is
     # about people, and this entry answered both because of the one
     # shared word.
     ["who", "which", "doing", "busy", "idle"], False, False),

    (answer_recent_activity,
     [["happened", "activity", "going on", "events"]],
     ["last", "recent", "minutes", "hour"],
     [], False, False),
]


def score_intent(question, required, helpful, avoid=()):
    """How well does this question fit this intent? 0 = not at all.

    A REQUIRED GROUP is an OR - the question must contain one of its
    words - and the groups are ANDed. That distinction is what separates
    "how many people entered today" from "how many people are here":
    both open with "how many", and leaving the tie to the optional words
    made the answer depend on which intent happened to be listed first.

    AVOID words disqualify outright, which is how "which camera recorded
    the most visits" stops being answered with the camera list and falls
    through to a real aggregate query.
    """
    text = " " + (question or "").lower() + " "
    for group in required:
        if not any(word in text for word in group):
            return 0.0
    for word in avoid:
        if word in text:
            return 0.0
    score = 1.0
    for word in helpful:
        if word in text:
            score += 1.0
    return score


def match(question):
    """Route a question. Returns (handler, kwargs) or (None, {})."""
    if not question or not question.strip():
        return None, {}

    person = resolve_person(question)
    camera = resolve_camera(question)

    best, best_score = None, 0.0
    for (handler, required, helpful, avoid,
         needs_person, needs_camera) in INTENTS:
        score = score_intent(question, required, helpful, avoid)
        if not score:
            continue
        # An intent that needs a name and did not find one is almost
        # certainly the wrong intent - "how many people are here" is not
        # a question about somebody's first entry.
        if needs_person and not person:
            score *= 0.25
        if person and needs_person:
            score += 1.5
        if camera and needs_camera:
            score += 1.0
        if score > best_score:
            best, best_score = handler, score

    if best is None:
        return None, {}
    return best, {"person": person, "camera": camera}


def answer(question):
    """Answer a question deterministically, or return None."""
    handler, kwargs = match(question)
    if handler is None:
        return None
    return handler(question, **kwargs)
