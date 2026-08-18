"""
core/vlm.py
===========
A client for a local vision-language model served by Ollama
(qwen2.5vl and similar).

WHAT THIS IS FOR - AND WHAT IT IS NOT
-------------------------------------
A VLM cannot do the tracking. Detection has a ~40 ms budget per frame at
25 fps; a 7B vision model needs 1.5-5 seconds per image, so it is 20-100x
too slow, and it would be competing for the same GPU as YOLO and
InsightFace. Nothing about that is fixable by prompting.

What it CAN do is answer, slowly and off to one side, the two questions
this system keeps getting wrong:

    "is that box really a person?"        <- reflections, posters, chairs
    "is that really who you say it is?"   <- the wrong name on a body

Both are used as a VETO and never as an identifier. The VLM is not
allowed to name anybody, because a 7B model looking at a 40-pixel CCTV
face cannot tell two colleagues apart and would invent an answer if
asked to. What it CAN see reliably is a contradiction - a beard where
the reference photo is clean-shaven, glasses where there are none, an
obviously different age or build - and a contradiction is enough to
refuse a name. Refusing produces an honest Unknown; accepting a wrong
one produces a wrong attendance record.

Everything here degrades gracefully: if Ollama is not running, or the
model is missing, callers get None and the pipeline carries on exactly
as it did before.

The scheduling, rate-limiting and application of these answers lives in
core/vlm_verify.py. This file is only the client and the prompts.
"""
import base64
import json
import re
import urllib.request
import urllib.error

import cv2
import numpy as np

from config.settings import (VLM_HOST, VLM_MODEL, VLM_TIMEOUT,
                             VLM_MAX_IMAGE_WIDTH, VLM_KEEP_ALIVE,
                             VLM_NUM_PREDICT)


def _encode(image, max_width=None):
    """BGR image -> base64 JPEG, shrunk to keep the request small."""
    max_width = max_width or VLM_MAX_IMAGE_WIDTH
    if image is None or getattr(image, "size", 0) == 0:
        return None
    if max_width and image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def available(host=None, model=None):
    """Is the vision model installed and reachable?"""
    host = host or VLM_HOST
    model = model or VLM_MODEL
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
            tags = json.loads(r.read())
        names = [m.get("name", "") for m in tags.get("models", [])]
        exact = model in names
        loose = any(n.split(":")[0] == model.split(":")[0] for n in names)
        return exact or loose, names
    except Exception:
        return False, []


def ask(image, prompt, host=None, model=None, timeout=None,
        json_mode=False, images=None, num_predict=None, keep_alive=None):
    """Send one or more images + a prompt. Returns the reply text, or None.

    `keep_alive` is not a detail: without it Ollama unloads the vision
    model between calls, and reloading 6GB of weights onto a GPU that is
    already running YOLO, SCRFD and AdaFace costs several seconds EVERY
    question. Held resident, a question costs only its own inference.
    """
    host = host or VLM_HOST
    model = model or VLM_MODEL
    timeout = timeout or VLM_TIMEOUT

    if images is None:
        images = [image] if image is not None else []
    encoded = [b for b in (_encode(i) for i in images) if b]
    if not encoded:
        return None

    payload = {
        "model": model,
        "prompt": prompt,
        "images": encoded,
        "stream": False,
        "keep_alive": VLM_KEEP_ALIVE if keep_alive is None else keep_alive,
        # deterministic: we want a measurement, not creative writing
        "options": {
            "temperature": 0,
            # These answers are one short JSON object. Left unbounded the
            # model will happily narrate, and every token of that is GPU
            # time taken from the detector.
            "num_predict": int(VLM_NUM_PREDICT if num_predict is None
                               else num_predict),
        },
    }
    if json_mode:
        payload["format"] = "json"

    try:
        req = urllib.request.Request(
            f"{host}/api/generate", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "").strip()
    except urllib.error.URLError:
        return None
    except Exception:
        return None


# ------------------------------------------------------------ montage
# WHY ONE COMPOSED IMAGE RATHER THAN SEVERAL ATTACHMENTS.
#
# Ollama will accept a list of images, but the model is then told
# nothing about which is which, and qwen2.5vl answers about "the image"
# without being reliable about ordering. Composing ONE picture with the
# live capture on the left, the references on the right, and captions
# drawn on both, removes the ambiguity entirely - the prompt can then
# talk about "the left panel" and mean it.
_CAPTION_H = 22


def _panel(image, caption, height):
    """One captioned panel, scaled to a common height."""
    if image is None or getattr(image, "size", 0) == 0:
        return None
    h, w = image.shape[:2]
    scale = height / max(1, h)
    width = max(1, int(round(w * scale)))
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC
                       if scale > 1 else cv2.INTER_AREA)
    strip = np.full((_CAPTION_H, width, 3), 20, dtype=np.uint8)
    cv2.putText(strip, caption[:28], (4, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, panel])


def montage(live, references, live_caption="LIVE (CCTV)",
            reference_caption="REFERENCE", panel_height=256):
    """Live crop on the left, reference photographs on the right.

    A divider is drawn between the two halves so the model cannot read
    the strip as one continuous scene.
    """
    panels = [_panel(live, live_caption, panel_height)]
    for i, ref in enumerate(references or []):
        panels.append(_panel(ref, f"{reference_caption} {i + 1}",
                             panel_height))
    panels = [p for p in panels if p is not None]
    if len(panels) < 2:
        return None

    divider = np.full((panels[0].shape[0], 6, 3), 200, dtype=np.uint8)
    out = [panels[0], divider]
    for i, panel in enumerate(panels[1:]):
        if i:
            out.append(np.full((panel.shape[0], 3, 3), 60, dtype=np.uint8))
        out.append(panel)
    return np.hstack(out)


# --------------------------------------------------------------- tasks
COUNT_PROMPT = (
    "Look at this office CCTV image carefully. Count how many PEOPLE are "
    "visible, including people who are seated, partly hidden behind "
    "monitors or chairs, or turned away from the camera. Do not count "
    "empty chairs, jackets, bags or reflections.\n"
    'Reply with JSON only: {"people": <number>, "note": "<short reason>"}'
)

# Deliberately enumerates the things that DO get detected as people on
# this site, because "is there a human here?" is answered yes for a
# poster of one and the old prompt asked exactly that.
IS_PERSON_PROMPT = (
    "This crop is what an automatic detector claims is ONE person, seen "
    "by an office CCTV camera.\n"
    "Is the main subject a REAL, physically present human being?\n"
    "Answer false if it is: a reflection in glass or a screen, a "
    "photograph, poster or person shown on a monitor, a mannequin, a "
    "coat or bag on a chair, an empty chair, or furniture.\n"
    "Answer true if it is a real person, even if they are seated, "
    "partly hidden, blurry or facing away.\n"
    'Reply with JSON only: {"person": true or false, '
    '"confidence": 0.0 to 1.0, "reason": "<6 words max>"}'
)

# THE IDENTITY CHECK - AND WHY IT IS NOT A YES/NO QUESTION.
#
# The obvious design is to show the model the live face beside the
# reference photographs and ask "same person?". It was built that way
# first and MEASURED on this site's own photographs, and it does not
# work: asked to be lenient about CCTV quality, qwen2.5vl answers
# "same, confidence 0.95" to everything - including a woman's CCTV crop
# against a bearded man's portraits, with the reason "similar hair style
# and beard". The leading question stops it looking.
#
# Asking it to be strict instead fails the other way, and worse: it
# starts refusing correct names because the crop is blurry, which trades
# wrong names for missing ones.
#
# So the model is never asked to compare anything. It is asked only to
# DESCRIBE one photograph, which is what vision models are actually
# reliable at, and the comparison is done in code below where the rules
# are visible and testable. Measured on this site, solo descriptions
# repeat identically call after call, and describing the two sides
# SEPARATELY also removes a contrast bias that was clearly present when
# both appeared in one picture - the same live crop was read as "female,
# long hair" beside one reference and "male, moustache" beside another.
ATTRIBUTES_PROMPT = (
    "Look at this photograph of ONE person and describe them factually.\n"
    "It may be a sharp portrait or a small, blurry frame from a CCTV "
    "camera - describe whichever you are given.\n"
    "Use \"unclear\" whenever you genuinely cannot tell. Saying unclear "
    "is correct and useful; guessing is not.\n"
    "\n"
    "Reply with JSON only, exactly these keys:\n"
    '{"sex":"male|female|unclear",'
    '"facial_hair":"none|moustache|beard|unclear",'
    '"glasses":"yes|no|unclear",'
    '"hair":"bald|short|medium|long|unclear",'
    '"age":"young_adult|middle_aged|older|unclear"}'
)

# WHAT IS IN THIS PICTURE?
#
# The gate on the review queue. Its job is to stop the system asking a
# human "who is this?" about a door, a floor or somebody's shoulder -
# which it was doing, because every test upstream of the question is a
# NUMBER (a detector score, a sharpness score, a face-quality score) and
# a patch of door texture scores perfectly well on all of them.
#
# ASKED AS A CLASSIFICATION, NOT AS A JUDGEMENT, and that is the third
# time in this file that distinction has decided whether something
# works. "Is there a face here?" was tried and measured first: the model
# answered false to EVERYTHING - real faces included - at a flat
# confidence of 0.2. Asked instead to say what the picture shows, it
# gets the useful cases right.
#
# Measured on this site's own crops, which is also where the trust
# boundary below comes from:
#   "place"     correct on background patches. Reliable.
#   "body"      correct on 4 of 4 review body crops - a seated person
#               whose face is not visible. Reliable, and the important
#               one: the body crop is what filled the review card.
#   "face"      correct on clear faces.
#   "head_back" NOT reliable. The model uses it for anything it is
#               unsure about - a person merely looking DOWN, and even a
#               patch of floor. So it is treated as "cannot tell" and
#               allowed through, because suppressing every downward-
#               facing crop would suppress most of a work-area camera.
SHOWS_PROMPT = (
    "What does this picture MAINLY show? Look carefully, then choose "
    "exactly one label.\n"
    "\n"
    '  "face"      a person\'s face, from the front or the side, where '
    "you can make out the eyes, nose and mouth\n"
    '  "head_back" a person\'s head from behind or from above - you can '
    "see hair or a scalp, but no face\n"
    '  "body"      a person is present, but their face is turned away, '
    "covered, cut off by the edge, or too small to make out\n"
    '  "place"     no person at all - a wall, a door, a floor, a desk, a '
    "chair, a screen, equipment, an empty corridor\n"
    "\n"
    "This is a CCTV frame, so it may be blurry or dark. Blurry is fine - "
    'label it "face" if you can still make out a face in it.\n'
    "\n"
    'Reply with JSON only: {"shows": "face|head_back|body|place", '
    '"note": "<5 words max>"}'
)

SHOWS_LABELS = ("face", "head_back", "body", "place")

DESCRIBE_PROMPT = (
    "Describe ONLY the clothing and hair of the person in this crop, in "
    "at most 6 words, so they can be told apart from other people. "
    'Reply with JSON only: {"description": "<words>"}'
)

# ---------------------------------------------------- WHAT ARE THEY DOING
#
# The one question in this file that is NOT a veto, and the reason it is
# allowed to exist: it asks about an ACTIVITY, not an IDENTITY.
#
# The whole argument against letting a 7B model name anybody is that a
# 40-pixel CCTV face does not contain the information the question needs,
# so the model fills the gap with something plausible. Posture does not
# have that problem. Whether somebody is facing a monitor with their
# hands at a keyboard, holding a phone to their ear, or walking across a
# room, is carried by exactly the coarse shape that survives a bad
# camera. The model is shown ONE person, is never told who they are, and
# its answer is joined to a name in core/vlm_scene.py.
#
# A CLOSED VOCABULARY, for the same reason SHOWS_PROMPT has one. Asked
# in open prose the model writes a paragraph per person - slow, and
# impossible to total up into "4 working, 1 on the phone". The labels
# are the ones an office camera can actually distinguish; anything else
# is "other", which is honest, and "unclear", which is the answer when
# the crop does not support a verdict.
#
# WORKING IS DEFINED BY WHAT IS VISIBLE, not by whether somebody is
# being productive - a camera cannot see that, and a system that claims
# to would be worse than useless. "at_desk" is asked separately for the
# same reason: leaning back at a desk is a different observation from
# standing in a corridor, and merging them into one judgement throws
# away the part that can be checked.
ACTIVITY_PROMPT = (
    "This is a crop of ONE person from an office CCTV camera.\n"
    "What is this person DOING? Look at their posture, their hands, "
    "where they are facing, and what is in front of them, then choose "
    "exactly one label.\n"
    "\n"
    '  "working"    at a desk or workstation and engaged with it - '
    "facing a monitor, hands at a keyboard or mouse, writing, reading "
    "papers, or handling equipment\n"
    '  "on_phone"   holding a phone to their ear, or looking down at a '
    "phone in their hand\n"
    '  "talking"    turned toward another person, standing or leaning '
    "in conversation\n"
    '  "walking"    moving through the room, standing away from any '
    "desk, entering or leaving\n"
    '  "idle"       at a desk but not engaged with it - leaning back, '
    "turned away from the screen, resting, eating, on their own\n"
    '  "other"      clearly doing something none of these describe\n'
    '  "unclear"    you genuinely cannot tell from this crop\n'
    "\n"
    "This is CCTV, so the crop may be small, dark or blurry. Say "
    '"unclear" rather than guessing - an honest "cannot tell" is useful '
    "and a guess is not.\n"
    "Do NOT guess the person's name, job or identity.\n"
    "\n"
    'Reply with JSON only: {"activity": '
    '"working|on_phone|talking|walking|idle|other|unclear", '
    '"at_desk": "yes|no|unclear", "note": "<6 words max, what you see>"}'
)

ACTIVITY_LABELS = ("working", "on_phone", "talking", "walking", "idle",
                   "other", "unclear")

# Which labels count as somebody being at work on something. Used only
# to phrase an answer - the label itself is always reported, so "is
# Pavan working?" can be checked against "on_phone" rather than being
# silently folded into a yes or a no.
ACTIVITY_WORKING = ("working",)
ACTIVITY_NOT_WORKING = ("idle", "walking")

ACTIVITY_WORDS = {
    "working": "working at a desk",
    "on_phone": "on the phone",
    "talking": "talking to somebody",
    "walking": "moving around the room",
    "idle": "at a desk but not working",
    "other": "doing something else",
    "unclear": "not clear enough to tell",
}

# THE WHOLE-FRAME QUESTION.
#
# Deliberately narrow. Asked "what is happening here?" a vision model
# writes an office-life narrative - who looks busy, who seems to be
# supervising, what the mood is - none of which is in the pixels. So it
# is asked for a factual description of the room and told, twice,
# that it must not identify anybody: the per-person labels above carry
# the detail, and this only sets the scene around them.
SCENE_PROMPT = (
    "This is a single frame from an office CCTV camera.\n"
    "Describe FACTUALLY what is visible: roughly how many people, where "
    "they are (at desks, standing, walking), and what the room is - "
    "desks, monitors, equipment, doorways.\n"
    "\n"
    "Rules:\n"
    "- Describe only what you can actually see in the picture.\n"
    "- Do NOT name anybody, guess who they are, or guess their job.\n"
    "- Do NOT speculate about what anybody is thinking or feeling, or "
    "whether they are being productive.\n"
    "- Two sentences at most.\n"
    "\n"
    'Reply with JSON only: {"people": <number>, '
    '"description": "<two sentences at most>"}'
)

ATTRIBUTE_KEYS = ("sex", "facial_hair", "glasses", "hair", "age")

# WHICH ATTRIBUTES ARE ALLOWED TO REFUSE A NAME, AND WHY THESE.
#
# Measured on this site by describing the same crops repeatedly and by
# describing people against their own enrollment photographs
# (tools/vlm_check.py reproduces it):
#
#   sex          identical on every repeat, and different on every pair
#                of different people. The single most useful signal.
#   hair         long vs short is stable and discriminative. "medium" is
#                not - it is what the model says when it is hedging - so
#                it is normalised to "unclear" and carries no weight.
#   facial_hair  stable in the presence/absence sense, but the model
#                moves between "beard" and "moustache" on the same
#                person, so only presence is compared - and on its own
#                it is not enough to refuse a name.
#   glasses      FLIPPED between two calls on the SAME image. Recorded
#                for the log and never acted on. This is exactly the
#                sort of plausible-looking cue that would have produced
#                confident wrong vetoes.
#   age          "young_adult" and "middle_aged" swap freely. Ignored.
_ATTRIBUTE_WEIGHT = {"sex": 1.0, "hair": 0.9, "facial_hair": 0.6}
_IGNORED_ATTRIBUTES = ("glasses", "age")


def _extract_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _as_bool(value):
    """Ollama's JSON mode returns real booleans; models sometimes do not."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "1", "y"):
            return True
        if text in ("false", "no", "0", "n"):
            return False
    return None


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def count_people(image, **kw):
    """Independent second opinion on how many people are in a frame.
    Returns (count, note) or (None, reason)."""
    reply = ask(image, COUNT_PROMPT, json_mode=True, num_predict=64, **kw)
    if reply is None:
        return None, "vision model unreachable"
    data = _extract_json(reply)
    if data is None:
        # fall back to the first number in the text
        m = re.search(r"\d+", reply)
        if m:
            return int(m.group()), reply[:60]
        return None, f"could not parse: {reply[:60]}"
    try:
        return int(data.get("people")), str(data.get("note", ""))[:80]
    except (TypeError, ValueError):
        return None, f"bad value: {reply[:60]}"


def is_person(crop, **kw):
    """Verify a detection really is a person.

    Returns (True/False, confidence, reason), or (None, 0.0, why) when
    the model could not be reached or did not answer usefully. None is
    NOT a rejection - every caller treats it as "carry on unchanged".
    """
    reply = ask(crop, IS_PERSON_PROMPT, json_mode=True, **kw)
    if reply is None:
        return None, 0.0, "vision model unreachable"
    data = _extract_json(reply)
    if data is None:
        return None, 0.0, f"could not parse: {reply[:40]}"
    value = _as_bool(data.get("person"))
    if value is None:
        return None, 0.0, f"no answer: {reply[:40]}"
    return value, _as_float(data.get("confidence")), \
        str(data.get("reason", ""))[:60]


def shows(crop, **kw):
    """What is in this picture? Returns (label, note).

    label is one of SHOWS_LABELS, or None when the model could not be
    reached or answered with something unrecognised. None means "carry
    on as before" and is never treated as a rejection.
    """
    reply = ask(crop, SHOWS_PROMPT, json_mode=True,
                num_predict=kw.pop("num_predict", 60), **kw)
    if reply is None:
        return None, "vision model unreachable"
    data = _extract_json(reply)
    if data is None:
        return None, f"could not parse: {reply[:40]}"
    label = str(data.get("shows", "")).strip().lower()
    note = str(data.get("note", ""))[:50]
    if label not in SHOWS_LABELS:
        return None, f"unrecognised label {label[:20]!r}"
    return label, note


def describe(attrs):
    """The short human-readable form of a description."""
    if not attrs:
        return ""
    parts = []
    if attrs.get("sex") not in (None, "", "unclear"):
        parts.append(attrs["sex"])
    hair = _normalise("hair", attrs.get("hair"))
    if hair:
        parts.append(f"{hair} hair")
    facial = _normalise("facial_hair", attrs.get("facial_hair"))
    if facial == "some":
        parts.append("beard/moustache")
    elif facial == "none":
        parts.append("clean-shaven")
    if attrs.get("glasses") == "yes":
        parts.append("glasses?")
    return ", ".join(parts)


def attributes(image, **kw):
    """Describe ONE person's durable features. None if it could not.

    One image, one person, no comparison. Everything that decides
    whether two of these describe the same person happens in
    contradictions() below, in ordinary code that can be read and
    tested.
    """
    reply = ask(image, ATTRIBUTES_PROMPT, json_mode=True,
                num_predict=kw.pop("num_predict", 120), **kw)
    if reply is None:
        return None
    data = _extract_json(reply)
    if not isinstance(data, dict):
        return None
    out = {key: str(data.get(key, "unclear")).strip().lower()[:12]
           for key in ATTRIBUTE_KEYS}
    # An answer where the model told us nothing at all is not an answer.
    if all(out[key] in ("", "unclear", "unknown") for key in ATTRIBUTE_KEYS):
        return None
    return out


def _normalise(key, value):
    """Fold an attribute onto the values that are actually stable.

    Returns None for "no usable signal", which is treated as agreement
    everywhere - the whole design refuses only on a positive
    contradiction, never on an absence.
    """
    value = (value or "").strip().lower()
    if value in ("", "unclear", "unknown", "none_visible"):
        return None
    if key == "hair":
        if value in ("bald", "short"):
            return "short"
        if value == "long":
            return "long"
        return None            # "medium" is the model hedging
    if key == "facial_hair":
        if value == "none":
            return "none"
        if value in ("moustache", "beard", "stubble", "goatee"):
            return "some"      # it moves between these on one person
        return None
    if key == "sex":
        return value if value in ("male", "female") else None
    return value


def says_anything(attrs):
    """Does this description rule anything in or out?

    A description made entirely of "unclear" is a valid answer - it is
    the model correctly declining to guess - but it carries no
    information, and code that treats it as information will conclude
    that everybody matches.
    """
    return any(_normalise(key, (attrs or {}).get(key))
               for key in _ATTRIBUTE_WEIGHT)


def contradictions(live, reference):
    """What, if anything, rules out these two being the same person?

    Returns (score, [reasons]). Zero means nothing contradicts - which
    is NOT evidence that they match, and is never treated as any.
    """
    if not live or not reference:
        return 0.0, []
    score, reasons = 0.0, []
    for key, weight in _ATTRIBUTE_WEIGHT.items():
        a = _normalise(key, live.get(key))
        b = _normalise(key, reference.get(key))
        if a is None or b is None or a == b:
            continue
        score += weight
        reasons.append(f"{key} {a} vs {b}")
    return score, reasons


def same_person(live_attributes, reference_attributes):
    """Judge two descriptions. Returns (ok, confidence, reason).

    `ok` is False only on a positive contradiction; True means nothing
    contradicts, which is worth nothing on its own and is treated that
    way by every caller.
    """
    if not live_attributes or not reference_attributes:
        return None, 0.0, "no description"
    score, reasons = contradictions(live_attributes, reference_attributes)
    if score <= 0:
        return True, 0.0, "nothing contradicts it"
    # 0.6 (facial hair alone) -> 0.80, below the default bar and
    # deliberately so. 0.9 (hair) -> 0.95. 1.0 (sex) -> 0.99.
    confidence = min(0.99, 0.5 + 0.5 * score)
    # Long enough for all three reasons to survive intact. A reason cut
    # off mid-word ("facial_hair none vs ") is what somebody reads when
    # they are trying to work out why a correct name was refused.
    return False, confidence, "; ".join(reasons)[:120]


def activity(crop, **kw):
    """What is this ONE person doing? Returns (label, at_desk, note).

    label is one of ACTIVITY_LABELS. None means the model could not be
    reached or answered with something unrecognised - reported as "not
    checked" rather than folded into "unclear", because a model that is
    switched off and a model that looked and could not tell are
    different facts about the answer.
    """
    reply = ask(crop, ACTIVITY_PROMPT, json_mode=True,
                num_predict=kw.pop("num_predict", 80), **kw)
    if reply is None:
        return None, "", "vision model unreachable"
    data = _extract_json(reply)
    if data is None:
        return None, "", f"could not parse: {reply[:40]}"
    label = str(data.get("activity", "")).strip().lower()
    at_desk = str(data.get("at_desk", "")).strip().lower()
    note = str(data.get("note", ""))[:60]
    if label not in ACTIVITY_LABELS:
        return None, at_desk, f"unrecognised activity {label[:20]!r}"
    if at_desk not in ("yes", "no", "unclear"):
        at_desk = "unclear"
    return label, at_desk, note


def scene(image, **kw):
    """Describe a whole frame. Returns (count, description) or (None, why).

    `count` is the model's own reading of how many people are visible
    and is NOT what the assistant reports - the tracker's count is. It
    is kept so the two can be shown side by side when they disagree,
    which is a genuinely useful signal about a frame the pipeline may be
    misreading.
    """
    reply = ask(image, SCENE_PROMPT, json_mode=True,
                num_predict=kw.pop("num_predict", 200), **kw)
    if reply is None:
        return None, "vision model unreachable"
    data = _extract_json(reply)
    if data is None:
        return None, f"could not parse: {reply[:60]}"
    description = " ".join(str(data.get("description", "")).split())[:400]
    try:
        count = int(data.get("people"))
    except (TypeError, ValueError):
        count = None
    if not description:
        return count, ""
    return count, description


def describe_person(crop, **kw):
    """A few words about clothing/hair, for labelling and re-identification."""
    reply = ask(crop, DESCRIBE_PROMPT, json_mode=True, **kw)
    data = _extract_json(reply)
    if data is None:
        return None
    text = str(data.get("description", "")).strip()
    return text[:60] or None
