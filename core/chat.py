"""
core/chat.py
============
Ask the system questions in plain English.

    >>> from core.chat import ASSISTANT
    >>> ASSISTANT.ask("who is in the reception now?").text
    '2 identified person(s) in Reception Lobby right now: ...'

Two routes, and the order matters
---------------------------------
    1. core/chat_intents.py - a real function over the repository.
       Deterministic, instant, cannot invent a name.
    2. core/chat_sql.py - the model writes a SELECT, which is validated
       and run read-only.

Route 1 handles the questions people actually ask; route 2 exists for
the ones nobody anticipated. Sending everything to the model would be
simpler to write and worse to rely on: this is an attendance record, and
a fluent wrong answer about who was present is more damaging than an
honest "I could not work that out".

AND ONE QUESTION NEITHER ROUTE CAN ANSWER
-----------------------------------------
"What is happening in the server room?" is not in the database. There is
no column for what somebody is doing, so no query - written by a model
or by hand - can produce it. Three handlers in route 1 therefore answer
by LOOKING: they take the live frame the pipeline publishes and ask a
vision model what each tracked person is doing.

The rule that makes that safe is in core/vlm_scene.py and is worth
repeating here, because it is the difference between this feature and
the reason the rest of the file exists: the model is never asked WHO
anybody is. Names come from the tracker and the face vote exactly as
before; the model is shown one anonymous crop and asked only what that
person is doing, and the two are joined in code. A model that cannot
tell two colleagues apart can still tell typing from walking.

Those answers carry the frame they were read off, for the same reason a
generated query carries its SQL.

Every answer says which route produced it, and a generated query is
always shown. If the assistant tells you something surprising you can
see exactly where it came from.
"""
import time

from config.settings import (CHAT_ENABLED, CHAT_ALLOW_SQL, CHAT_REPLY_MODEL,
                             CHAT_SQL_MODEL, CHAT_HOST, CHAT_DEBUG)
from core.chat_intents import Answer, answer as intent_answer


SUGGESTIONS = [
    "What is happening in Server Rm Psg right now?",
    "Who is working and who is not?",
    "What is Pavan Kumar doing?",
    "How many cameras are running and what are they?",
    "Who is present in the reception now?",
    "How many people are in cam2 and who are they?",
    "What is the last one hour summary of cam2?",
    "What time did Deepthi first enter today?",
    "When was Deepthi's last entry?",
    "Who has been present the longest today?",
    "Who has not come in today?",
    "Is Akhila here right now?",
    "How many people entered today?",
    "How many confirmations are pending?",
    "How many people are enrolled?",
]


# ------------------------------------------------------------- chips
# The quick chips sitting above the input box.
#
# The first version was this whole SUGGESTIONS list, printed once and
# never changed: twelve full sentences wrapping over three rows, naming
# people who might not be enrolled and cameras that might be switched
# off. It read as documentation, not as something to tap.
#
# A chip is now a short LABEL and the QUESTION it actually asks - the
# label has to fit on a button, the question has to be one the handlers
# in chat_intents recognise. Only three or four are shown, and which
# ones depends on the moment: on load they are built from the cameras
# that are really running, and after an answer they follow whoever or
# whatever that answer was about. Asking "is Akhila here?" should leave
# "when did Akhila arrive?" one tap away.

def chip(label, question):
    return {"label": label, "q": question}


# NAMED, not indexed. These were referred to as GENERAL_CHIPS[1] and
# GENERAL_CHIPS[6] throughout the follow-up rules, so adding one chip
# silently re-pointed every reference below it at the wrong question.
HERE_NOW = chip("Who's here now", "Who is present right now?")
WHO_WORKING = chip("Who's working", "Who is working and who is not?")
ENTRIES_TODAY = chip("Entries today", "How many people entered today?")
LONGEST_TODAY = chip("Longest today", "Who has been present the longest today?")
NOT_IN_TODAY = chip("Not in today", "Who has not come in today?")
LAST_HOUR = chip("Last hour", "What happened in the last hour?")
CAMERAS = chip("Cameras", "How many cameras are running and what are they?")
PENDING = chip("Pending confirmations", "How many confirmations are pending?")
ENROLLED = chip("Enrolled", "How many people are enrolled?")

GENERAL_CHIPS = [HERE_NOW, WHO_WORKING, ENTRIES_TODAY, LONGEST_TODAY,
                 NOT_IN_TODAY, LAST_HOUR, CAMERAS, PENDING, ENROLLED]


def _cameras():
    """(key, name) for the cameras that are switched on."""
    try:
        from config.cameras import enabled_cameras
        return [(c["key"], c["name"]) for c in enabled_cameras()]
    except Exception:
        return []


def _short_camera(name):
    """"Reception Lobby" -> "Reception". A chip has one line to work with."""
    return (name or "").split()[0] if name else ""


def _camera_chips(name):
    if not name:
        return []
    short = _short_camera(name)
    return [
        chip(f"Who's in {short}", f"Who is present in {name} now?"),
        # The live look, one tap from every camera. It is the slowest
        # thing the assistant does and the one people ask for first, so
        # it is offered rather than left to be typed exactly right.
        chip(f"{short}: what's happening",
             f"What is happening in {name} right now?"),
        chip(f"{short}: last hour",
             f"What is the last one hour summary of {name}?"),
        chip(f"{short}: arrivals",
             f"Last one hour entered persons details in {name}"),
    ]


def _person_chips(person, skip=""):
    """Follow-ups about one person, minus the one just asked."""
    first = (person or "").split()[0] if person else ""
    if not first:
        return []
    options = [
        ("person_activity", chip(f"What is {first} doing?",
                                 f"What is {person} doing right now?")),
        ("is_here", chip(f"Is {first} here?", f"Is {person} here right now?")),
        ("first_entry", chip(f"{first}: arrival",
                             f"What time did {person} first enter today?")),
        ("last_entry", chip(f"{first}: last entry",
                            f"When was {person}'s last entry?")),
        ("person_time", chip(f"{first}: time today",
                             f"How long has {person} been present today?")),
        ("entries", chip(f"{first}: all entries",
                         f"Show all entries for {person}")),
    ]
    return [c for source, c in options if source != skip]


def _person_in(answer):
    """The person an answer is about, if it is about one."""
    for row in (answer.rows or [])[:4]:
        if not isinstance(row, dict):
            continue
        for key in ("person", "name", "employee"):
            value = str(row.get(key) or "")
            if not value or value.lower() in ("unknown", "none", "—"):
                continue
            # A live look labels the people it could not name
            # "Unidentified #7". Offering "What is Unidentified #7
            # doing?" as a follow-up is a chip that cannot work - no
            # handler can resolve that to anybody.
            if value.lower().startswith(("unidentified", "unknown #")):
                continue
            return value
    return ""


def _trim(chips, count=4):
    """First `count`, no duplicates, nothing empty."""
    seen, out = set(), []
    for c in chips:
        if not c or not c.get("q") or c["q"].lower() in seen:
            continue
        seen.add(c["q"].lower())
        out.append(c)
        if len(out) >= count:
            break
    return out


def starters():
    """The chips a freshly opened page shows."""
    cameras = _cameras()
    chips = [HERE_NOW]
    if cameras:
        chips.append(_camera_chips(cameras[0][1])[1])   # what's happening
    if len(cameras) > 1:
        chips.append(_camera_chips(cameras[1][1])[1])
    chips += [WHO_WORKING, ENTRIES_TODAY, LONGEST_TODAY, CAMERAS]
    return _trim(chips)


# Which handler answered -> what somebody usually wants next.
_PERSON_SOURCES = {"is_here", "first_entry", "last_entry", "person_time",
                   "entries", "person_activity"}


def followups(question, answer):
    """Three or four things worth asking after this answer."""
    source = answer.source or ""
    cameras = _cameras()
    chips = []

    if source in _PERSON_SOURCES:
        # about one person - stay with that person
        person = _person_in(answer)
        if not person:
            try:
                # resolve_people, which returns nothing when the question
                # named nobody it could pin down. resolve_person would
                # answer "Anil Kumar Sahu" to a question about "kumar"
                # that the handler itself had just declined to answer -
                # so the chips would offer follow-ups about a person who
                # was never mentioned.
                from core.chat_intents import resolve_people
                found = resolve_people(question)
                person = found[0] if found else ""
            except Exception:
                person = ""
        chips += _person_chips(person, skip=source)
        chips.append(HERE_NOW)

    elif source in ("scene", "working"):
        # A LIVE LOOK. What follows one is almost always another live
        # question - about one of the people it just named, or about the
        # other camera - so those come first and the database questions
        # come after.
        person = _person_in(answer)
        chips += _person_chips(person, skip="")[:1]
        try:
            from core.chat_intents import resolve_camera
            found = resolve_camera(question)
        except Exception:
            found = None
        asked_about = found[0] if found else ""
        for key, name in cameras:
            if key != asked_about:
                chips += _camera_chips(name)[1:2]
        chips += [WHO_WORKING, HERE_NOW]

    elif source in ("who_present", "count_present", "entry_list",
                    "camera_summary", "recent_activity", "top_presence"):
        # about a place or a moment - offer the people in it, then the
        # camera the question was already looking at
        person = _person_in(answer)
        chips += _person_chips(person)[:2]
        try:
            from core.chat_intents import resolve_camera
            found = resolve_camera(question)
        except Exception:
            found = None
        name = found[1] if found else (cameras[0][1] if cameras else "")
        chips += [c for c in _camera_chips(name)[1:]]
        chips.append(ENTRIES_TODAY)

    elif source == "cameras":
        for _key, name in cameras[:2]:
            chips += _camera_chips(name)[:1]
        chips += [HERE_NOW, WHO_WORKING, ENTRIES_TODAY]

    elif source in ("absent", "employees", "reviews", "status"):
        chips += [HERE_NOW, ENTRIES_TODAY, LONGEST_TODAY, PENDING]

    else:
        # a generated query, a failure, or nothing recognised at all -
        # fall back to the questions that always work
        chips += starters()

    # `question` is dropped from the follow-ups: re-offering what was
    # just asked is the one chip nobody taps.
    asked = (question or "").strip().lower()
    chips = [c for c in chips if c["q"].strip().lower() != asked]
    return _trim(chips + GENERAL_CHIPS)


def _table(columns, rows, limit=12):
    """Format a result set as plain text - no model involved."""
    if not rows:
        return ""
    columns = columns or list(rows[0].keys())
    widths = {c: max(len(str(c)),
                     max(len(str(r.get(c, ""))) for r in rows[:limit]))
              for c in columns}
    head = "  ".join(str(c).ljust(widths[c]) for c in columns)
    lines = [head, "  ".join("-" * widths[c] for c in columns)]
    for row in rows[:limit]:
        lines.append("  ".join(str(row.get(c, "")).ljust(widths[c])
                               for c in columns))
    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more")
    return "\n".join(lines)


def _phrase(question, columns, rows):
    """Ask the reply model to put the result into a sentence.

    Strictly cosmetic, and given ONLY the rows that were actually
    returned. It is told not to add anything, because the failure mode
    of this step is a model helpfully explaining what the numbers mean -
    which is how "3 rows" becomes "a quiet afternoon" with nothing
    behind it.
    """
    if not CHAT_REPLY_MODEL or not rows:
        return ""
    from core.chat_sql import ollama
    sample = rows[:15]
    prompt = (
        "Answer the question in ONE short sentence using ONLY the data "
        "below. Do not add any fact that is not in the data. Do not "
        "speculate or interpret. If the data does not answer it, say so.\n\n"
        f"QUESTION: {question}\n"
        f"DATA ({len(rows)} row(s)): {sample}\n\nANSWER:")
    try:
        reply = ollama(prompt, CHAT_REPLY_MODEL, temperature=0.1)
    except Exception:
        return ""
    # qwen3 emits a <think> block; keep only what follows it
    if "</think>" in reply:
        reply = reply.split("</think>")[-1]
    return " ".join(reply.split())[:400]


class Assistant:
    """Answers questions about what the cameras have seen."""

    def __init__(self, allow_sql=None, phrase=True):
        self.allow_sql = CHAT_ALLOW_SQL if allow_sql is None else allow_sql
        self.phrase = phrase
        self.suggestions = list(SUGGESTIONS)

    def ask(self, question):
        """Returns an Answer, with follow-up chips attached. Never raises."""
        answer = self._resolve(question)
        try:
            answer.chips = followups(question, answer)
        except Exception as exc:              # chips are decoration, never
            answer.chips = GENERAL_CHIPS[:4]  # a reason to lose the answer
            if CHAT_DEBUG:
                print(f"[CHAT] chips failed: {exc}")
        return answer

    def _resolve(self, question):
        """The answer itself - which route can handle this question."""
        started = time.time()
        if not CHAT_ENABLED:
            return Answer("The assistant is switched off "
                          "(CHAT_ENABLED=0).", source="disabled")
        question = (question or "").strip()
        if not question:
            return Answer("Ask me something about the cameras, who is "
                          "present, or when somebody arrived.",
                          source="empty")

        # ---- 1. a handler that knows this question -------------------
        try:
            answer = intent_answer(question)
        except Exception as exc:
            answer = None
            if CHAT_DEBUG:
                print(f"[CHAT] handler failed: {exc}")
        if answer is not None:
            if CHAT_DEBUG:
                print(f"[CHAT] {answer.source} in "
                      f"{(time.time() - started) * 1000:.0f}ms")
            return answer

        # ---- 2. let the model write a query -------------------------
        if not self.allow_sql:
            return Answer(
                "I do not have a built-in answer for that, and query "
                "generation is switched off (CHAT_ALLOW_SQL=0). Try one "
                "of: " + "; ".join(self.suggestions[:4]),
                source="no_handler")

        from core.chat_sql import ask as sql_ask
        columns, rows, sql, error = sql_ask(question)
        if error:
            return Answer(
                f"I could not work that one out - {error}. "
                f"Try rephrasing, or ask something like: "
                f"{self.suggestions[1]}",
                source="sql_failed", sql=sql)
        if not rows:
            return Answer("Nothing in the records matches that.",
                          columns=columns, source="sql", sql=sql)

        text = _phrase(question, columns, rows) if self.phrase else ""
        if not text:
            # A SENTENCE, not a rendered table. The caller already has
            # the rows and knows how to display them - the console
            # prints a table, the dashboard draws one - and putting a
            # second copy inside the text meant the web page showed
            # everything twice.
            text = (f"{len(rows)} result(s) - see the table below."
                    if len(rows) > 1 else "1 result:")
        if CHAT_DEBUG:
            print(f"[CHAT] sql in {(time.time() - started) * 1000:.0f}ms, "
                  f"{len(rows)} row(s)")
        return Answer(text, rows=rows, columns=columns, source="sql", sql=sql)

    def health(self):
        """Is everything the assistant needs actually reachable?"""
        from core.chat_sql import ollama
        status = {"enabled": CHAT_ENABLED, "host": CHAT_HOST,
                  "sql_model": CHAT_SQL_MODEL,
                  "reply_model": CHAT_REPLY_MODEL or "(none)",
                  "sql_allowed": self.allow_sql}
        try:
            ollama("Reply with the single word: ok", CHAT_SQL_MODEL,
                   timeout=20)
            status["ollama"] = "reachable"
        except Exception as exc:
            status["ollama"] = f"unreachable: {exc}"
        # ...and whether the LIVE questions can be answered, which needs
        # two more things than the rest: a running pipeline publishing
        # snapshots, and a vision model. Reported separately because
        # they fail separately - "what is happening" going quiet while
        # "who is here" still works is a pipeline problem, not a chat
        # one, and that has to be visible.
        try:
            from core.vlm_scene import available
            status["live_questions"] = available()
        except Exception as exc:
            status["live_questions"] = f"unavailable: {exc}"
        return status


ASSISTANT = Assistant()


def ask(question):
    """Convenience for scripts and the dashboard."""
    return ASSISTANT.ask(question)
