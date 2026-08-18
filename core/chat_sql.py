"""
core/chat_sql.py
================
The fallback: let a language model write SQL for questions the
deterministic handlers do not cover.

The whole point is the guard rails
----------------------------------
A model writing SQL against a staff attendance database is a genuinely
risky thing to build, and it is worth being explicit about why this one
is acceptable:

  READ-ONLY AT THE CONNECTION. The query runs over a SQLite connection
  opened with mode=ro. Not "we checked the SQL first" - the operating
  system will not let this connection write. Even a query that gets past
  every check below physically cannot modify or delete anything.

  ONE STATEMENT, AND IT MUST BE A SELECT. Anything with a second
  statement, a semicolon in the middle, or a write verb is refused
  before it reaches the database.

  ONLY TABLES WE NAMED. The model is given a curated schema and the
  query is checked against that list, so it cannot read the users table
  and hand somebody a password hash.

  A ROW CEILING. A LIMIT is injected if the model did not write one.

  AND IT IS PRINTED. The generated SQL goes to the console and is
  returned to the dashboard with the answer. A query written by a model
  over staff records should be visible to whoever runs the system, not
  hidden behind a friendly sentence.

What the model is NOT asked to do
---------------------------------
It never decides what the data MEANS. It writes a query; the numbers
that come back are formatted by code. Asking a model to both fetch and
interpret is how "3 people were present" becomes "it was a quiet
afternoon" with no basis.
"""
import os
import re
import json
import sqlite3
import urllib.error
import urllib.request

from config.settings import (DB_PATH, CHAT_HOST, CHAT_SQL_MODEL,
                             CHAT_TIMEOUT, CHAT_MAX_ROWS, CHAT_DEBUG,
                             TZ_OFFSET_HOURS)

# The tables the assistant may read, and what they mean. Deliberately a
# HAND-WRITTEN description rather than a dump of the schema: the model
# needs to know that `presence.seconds_present` counts only time somebody
# was actually detected, which no column type can tell it.
SCHEMA = """
TABLE visits            one person's stay in front of one camera (an ARRIVAL)
  id INTEGER
  camera_key TEXT       e.g. 'reception', 'server_rm_psg'
  camera_name TEXT      e.g. 'Reception Lobby'
  track_id INTEGER
  person_name TEXT      'Unknown' until recognised
  emp_code TEXT
  identified BOOLEAN
  match_score REAL
  entered_at DATETIME   UTC, when they arrived
  identified_at DATETIME
  last_seen DATETIME    UTC
  left_at DATETIME      UTC, NULL while still in view
  active BOOLEAN        1 = still in view now
  duration_seconds REAL

TABLE presence          how long one person was in an AREA camera's zone
  id INTEGER
  camera_key TEXT
  camera_name TEXT
  track_id INTEGER
  person_name TEXT      'Unknown' until recognised
  emp_code TEXT
  identified BOOLEAN
  first_seen DATETIME   UTC
  last_seen DATETIME    UTC
  seconds_present REAL  time ACTUALLY detected, not wall-clock elapsed
  active BOOLEAN        1 = present now

TABLE employees         people enrolled from their photographs
  id INTEGER
  emp_code TEXT
  name TEXT
  photo_count INTEGER
  enrolled_at DATETIME

TABLE review_items      sightings a human was asked to confirm
  id INTEGER
  camera_key TEXT
  camera_name TEXT
  track_id INTEGER
  suggested_name TEXT   empty when the system had no idea
  confirmed_name TEXT
  status TEXT           'pending' | 'confirmed' | 'rejected'
  probability REAL
  created_at DATETIME   UTC
  reviewed_at DATETIME

TABLE system_events     pipeline start/stop and notable events
  id INTEGER
  kind TEXT             'START' | 'STOP' | 'ERROR'
  message TEXT
  created_at DATETIME   UTC
""".strip()

ALLOWED_TABLES = {"visits", "presence", "employees", "review_items",
                  "system_events"}

# Never readable through this path, whatever the model writes.
FORBIDDEN_TABLES = {"users", "learned_faces", "learned_appearance",
                    "sqlite_master", "sqlite_temp_master"}

FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|"
    r"detach|pragma|vacuum|reindex|grant|revoke)\b", re.IGNORECASE)

PROMPT = """You write SQLite queries for a CCTV attendance database.

{schema}

RULES
- Answer with ONE SQLite SELECT statement and nothing else.
- No markdown, no explanation, no trailing semicolon.
- EVERY constraint in the question must appear in the query. If it names
  a camera, filter on it. If it names a time window, filter on it. If it
  names a person, filter on it. A query that ignores part of the
  question is wrong even if it runs.
- Times are stored in UTC. Local time is UTC+{offset} hours.
- "now" is datetime('now'). "the last N hours" is
  entered_at >= datetime('now', '-N hours').
- "today" means date(entered_at, '+{offset} hours') = date('now', '+{offset} hours').
- person_name is 'Unknown' for anybody never identified. Exclude those
  unless the question is about unidentified people.
- Select the few columns that answer the question, not every column.
  Never select internal ids unless asked.
- Always include a LIMIT of at most {limit}.
- GROUP BY camera_key, never camera_name. The display name has changed
  over time, so the same camera appears under several names in older
  rows and grouping by it splits one camera into three.

EXAMPLES

Q: who entered the reception in the last hour
SELECT person_name AS person, datetime(entered_at, '+{offset} hours') AS entered
FROM visits
WHERE camera_key = 'reception'
  AND entered_at >= datetime('now', '-1 hours')
  AND person_name != 'Unknown'
ORDER BY entered_at DESC LIMIT {limit}

Q: which camera recorded the most visits
SELECT camera_key AS camera, COUNT(*) AS visits
FROM visits GROUP BY camera_key ORDER BY visits DESC LIMIT {limit}

Q: how long was each person in the server room today
SELECT person_name AS person, SUM(seconds_present) AS seconds
FROM presence
WHERE camera_key = 'server_rm_psg'
  AND date(first_seen, '+{offset} hours') = date('now', '+{offset} hours')
  AND person_name != 'Unknown'
GROUP BY person_name ORDER BY seconds DESC LIMIT {limit}

CAMERAS AVAILABLE
{cameras}

QUESTION: {question}

SELECT"""


def camera_hint():
    """Tell the model which camera keys actually exist.

    Without this it invents plausible-looking keys - 'reception_lobby',
    'cam2' - which match no rows, and the query returns nothing with no
    indication that the filter was the problem.
    """
    try:
        from config.cameras import CAMERAS
        return "\n".join(
            f"  camera_key '{c['key']}' is {c['name']}" for c in CAMERAS)
    except Exception:
        return "  (unknown)"


def _read_only_connection():
    """A connection the OS will not let us write through."""
    uri = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def ollama(prompt, model, host=None, timeout=None, temperature=0.0):
    """One completion from Ollama. Returns text, or raises."""
    host = (host or CHAT_HOST).rstrip("/")
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 512},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request,
                                timeout=timeout or CHAT_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8"))
    return (body.get("response") or "").strip()


def clean_sql(text):
    """Pull a single SELECT out of whatever the model replied with."""
    if not text:
        return ""
    # models like to wrap things in fences even when told not to
    fence = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    text = text.strip()
    if not re.match(r"(?is)^\s*select\b", text):
        # the prompt ends with "SELECT", so the model often continues it
        text = "SELECT " + text.lstrip()
    # keep only the first statement
    text = text.split(";")[0].strip()
    return re.sub(r"\s+", " ", text)


def validate(sql):
    """Is this query safe to run? Returns (ok, reason)."""
    if not sql:
        return False, "the model returned nothing"
    if not re.match(r"(?is)^select\b", sql.strip()):
        return False, "only SELECT statements are allowed"
    if FORBIDDEN_SQL.search(sql):
        return False, "the query contains a statement that could modify data"
    lowered = sql.lower()
    for table in FORBIDDEN_TABLES:
        if re.search(rf"\b{re.escape(table)}\b", lowered):
            return False, f"the table '{table}' is not readable here"

    # every table it names must be one we published
    referenced = set(re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)",
                                lowered))
    unknown = referenced - ALLOWED_TABLES
    if unknown:
        return False, f"unknown table(s): {', '.join(sorted(unknown))}"
    if not referenced:
        return False, "the query does not read any known table"
    return True, ""


def enforce_limit(sql, maximum=None):
    maximum = int(maximum or CHAT_MAX_ROWS)
    if re.search(r"\blimit\s+\d+", sql, re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {maximum}"


def run(sql):
    """Execute a validated query. Returns (columns, rows)."""
    connection = _read_only_connection()
    try:
        cursor = connection.execute(sql)
        fetched = cursor.fetchmany(CHAT_MAX_ROWS)
        columns = [d[0] for d in (cursor.description or [])]
        rows = [{c: r[c] for c in columns} for r in fetched]
        return columns, rows
    finally:
        connection.close()


def ask(question, model=None):
    """Question -> (columns, rows, sql, error).

    Any failure is returned rather than raised: a chatbot that throws a
    stack trace at somebody who asked how many people are in the room is
    worse than one that says it could not work the question out.
    """
    model = model or CHAT_SQL_MODEL
    prompt = PROMPT.format(schema=SCHEMA, question=question.strip(),
                           offset=TZ_OFFSET_HOURS, limit=CHAT_MAX_ROWS,
                           cameras=camera_hint())
    try:
        raw = ollama(prompt, model)
    except urllib.error.URLError as exc:
        return [], [], "", (f"could not reach Ollama at {CHAT_HOST} ({exc}). "
                            f"Is it running?")
    except Exception as exc:
        return [], [], "", f"the model failed: {exc}"

    sql = clean_sql(raw)
    ok, reason = validate(sql)
    if CHAT_DEBUG:
        print(f"[CHAT] {'SQL' if ok else 'REFUSED'}: {sql}")
        if not ok:
            print(f"[CHAT] reason: {reason}")
    if not ok:
        return [], [], sql, reason

    sql = enforce_limit(sql)
    try:
        columns, rows = run(sql)
    except sqlite3.Error as exc:
        return [], [], sql, f"the query did not run: {exc}"
    return columns, rows, sql, ""
