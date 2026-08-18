"""
tools/vlm_check.py
==================
Measure whether the vision model's veto is worth switching on, using
THIS site's own photographs.

Why measure rather than assume
------------------------------
core/vlm_verify.py lets a vision model refuse a name. That is a real
power over the attendance record, and it has two ways to be wrong:

  A FALSE VETO   it refuses a CORRECT name. The person goes back to
                 Unknown and their minutes stop being credited. This is
                 the expensive mistake, and it is the one to measure
                 first.

  A MISSED VETO  it accepts a WRONG name. No worse than today - that is
                 the situation this was added to improve - but it tells
                 you how much of the problem the check actually catches.

Neither number can be guessed at from a model card. They depend on how
big your faces are, how good your enrollment photographs are, and how
similar your people look to each other. So this runs the exact question
the pipeline asks, over pairs built from data/faces, and reports both.

WHAT IT DOES
------------
Exactly what the pipeline does, on the same code path: the vision model
DESCRIBES each picture on its own, and the descriptions are compared in
code (core/vlm.py). It never asks the model "is this the same person?" -
see the comment on ATTRIBUTES_PROMPT for the measurement that settled
that question.

For each person who has BOTH enrollment photographs (taken by a human)
and camera captures (cctv_*.jpg, filed by the system):

    GENUINE   their own camera capture  vs  their own enrollment photos
              -> nothing should contradict. A refusal here is a FALSE
                 VETO: a correct name taken off somebody.

    IMPOSTOR  their camera capture      vs  somebody ELSE's enrollment
              -> something should contradict. A pass here is a miss.

A THIRD THING IT FINDS, WHICH MAY MATTER MORE
---------------------------------------------
A genuine pair that contradicts is not always the model being wrong. It
can equally mean the camera capture filed under that person is NOT that
person - which is the failure core/face_library.py warns about and the
reason tools/audit_learned_faces.py exists. The report says which of
the two it looks like, and the saved pictures let you settle it in
seconds by eye.

USAGE
-----
    python tools/vlm_check.py                    # 12 pairs of each
    python tools/vlm_check.py --pairs 30         # slower, more reliable
    python tools/vlm_check.py --person "Rajesh"  # one person only
    python tools/vlm_check.py --save             # write the comparison
                                                 # pictures for eyeballing

    # find camera photographs filed under the WRONG person - these are
    # live recognition references, so a wrong one teaches a wrong face
    python tools/vlm_check.py --audit-library --save

    # describe everybody once, so the confirmation queue can shortlist
    # who an unknown face could be (run once after enrolling people)
    python tools/vlm_check.py --learn-people

    # separately: run the "is this a person?" check over a folder of
    # crops (e.g. data/review/server_rm_psg) and print the answers
    python tools/vlm_check.py --person-crops data/review/server_rm_psg

READING THE RESULT
------------------
The table at the end prints, at each confidence threshold, how many
correct names would have been refused and how many wrong ones caught.
Set VLM_IDENTITY_MIN_CONFIDENCE to the lowest threshold where the false
veto count is zero (or as near as you can accept). If it is zero at no
threshold, your faces are too small for this check - leave
VLM_VERIFY_IDENTITY off and spend the effort on better enrollment
photographs instead, which helps everything else at the same time.
"""
import os
import sys
import glob
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from config.settings import (FACE_DIR, DATA_DIR, VLM_MODEL,
                             VLM_IDENTITY_REFERENCES,
                             VLM_IDENTITY_MIN_CONFIDENCE,
                             VLM_PERSON_MIN_CONFIDENCE)
from core import vlm
from core.face_library import CAPTURE_PREFIX

OUT_DIR = os.path.join(DATA_DIR, "vlm_check")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# The thresholds the table is reported at. VLM_IDENTITY_MIN_CONFIDENCE
# is one of them so the current setting is always visible in the result.
THRESHOLDS = sorted({0.5, 0.6, 0.7, 0.8, 0.9,
                     round(float(VLM_IDENTITY_MIN_CONFIDENCE), 2)})


def label_of(folder):
    base = os.path.basename(folder)
    label = base.split("-", 1)[1] if "-" in base else base
    return label.replace("_", " ").strip()


def collect_people(root=FACE_DIR):
    """{name: {"enrolled": [paths], "captured": [paths]}}"""
    people = {}
    for path in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(path):
            continue
        files = [p for p in sorted(glob.glob(os.path.join(path, "*")))
                 if os.path.isfile(p) and p.lower().endswith(IMAGE_EXT)]
        enrolled = [p for p in files
                    if not os.path.basename(p).startswith(CAPTURE_PREFIX)]
        captured = [p for p in files
                    if os.path.basename(p).startswith(CAPTURE_PREFIX)]
        if enrolled or captured:
            people[label_of(path)] = {"enrolled": enrolled,
                                      "captured": captured}
    return people


def load(paths, limit=None):
    out = []
    for path in paths[:limit] if limit else paths:
        image = cv2.imread(path)
        if image is not None and image.size:
            out.append(image)
    return out


def build_pairs(people, wanted, rng, only=None):
    """(kind, subject, reference_owner, live_path, reference_paths)"""
    usable = {name: data for name, data in people.items()
              if data["captured"] and data["enrolled"]}
    if only:
        usable = {n: d for n, d in usable.items()
                  if only.lower() in n.lower()}
    names = sorted(usable)
    if not names:
        return []

    others = sorted(n for n in people if people[n]["enrolled"])
    pairs = []
    for i in range(wanted):
        subject = names[i % len(names)]
        live = rng.choice(usable[subject]["captured"])
        pairs.append(("genuine", subject, subject, live,
                      usable[subject]["enrolled"][:VLM_IDENTITY_REFERENCES]))

        rivals = [n for n in others if n != subject]
        if rivals:
            rival = rng.choice(rivals)
            pairs.append(("impostor", subject, rival, live,
                          people[rival]["enrolled"][:VLM_IDENTITY_REFERENCES]))
    return pairs


def describe_reference(name, ref_paths, cache):
    """One description per person's enrollment photographs, reused."""
    if name in cache:
        return cache[name]
    references = load(ref_paths)
    if not references:
        cache[name] = None
        return None
    sheet = references[0] if len(references) == 1 else vlm.montage(
        references[0], references[1:], live_caption="PHOTO 1",
        reference_caption="PHOTO")
    cache[name] = vlm.attributes(sheet if sheet is not None else references[0])
    return cache[name]


def summarise(attrs):
    if not attrs:
        return "(no description)"
    return (f"{attrs.get('sex','?')}/{attrs.get('hair','?')}/"
            f"{attrs.get('facial_hair','?')}")


def run_identity(pairs, save=False):
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)

    print(f"{'kind':<10}{'live face':<20}{'described as':<22}"
          f"{'reference':<18}{'described as':<22}{'verdict':>10}{'conf':>6}")
    print("-" * 110)

    reference_cache = {}
    live_cache = {}
    results = []
    for index, (kind, subject, owner, live_path, ref_paths) in enumerate(pairs):
        live = cv2.imread(live_path)
        if live is None:
            continue
        # The live crop is described once and reused across both the
        # genuine and the impostor pair it appears in - exactly as the
        # pipeline caches it per track, and it halves the run time.
        if live_path not in live_cache:
            live_cache[live_path] = vlm.attributes(live)
        live_attrs = live_cache[live_path]
        ref_attrs = describe_reference(owner, ref_paths, reference_cache)

        ok, confidence, reason = vlm.same_person(live_attrs, ref_attrs)
        verdict = {True: "agrees", False: "REFUSES"}.get(ok, "unsure")
        print(f"{kind:<10}{subject[:18]:<20}{summarise(live_attrs):<22}"
              f"{owner[:16]:<18}{summarise(ref_attrs):<22}"
              f"{verdict:>10}{confidence:>6.2f}")
        if ok is False:
            print(f"{'':<10}-> {reason}")
        results.append((kind, subject, owner, ok, confidence, live_path,
                        live_attrs, ref_attrs))

        if save:
            picture = vlm.montage(live, load(ref_paths))
            if picture is not None:
                cv2.imwrite(os.path.join(
                    OUT_DIR, f"{index:03d}_{kind}_{subject[:14]}"
                             f"_vs_{owner[:14]}.jpg".replace(" ", "_")),
                    picture)
    return results


def report(results):
    genuine = [r for r in results if r[0] == "genuine"]
    impostor = [r for r in results if r[0] == "impostor"]
    if not genuine and not impostor:
        print("\n[VLM] no usable pairs - nothing to measure")
        return 1

    unsure = sum(1 for r in results if r[3] is None)
    if unsure:
        print(f"\n[VLM] {unsure} of {len(results)} answers were unusable "
              f"(model unreachable or unparseable). They are counted as "
              f"'no veto', which is how the pipeline treats them too.")

    print(f"\n{'confidence needed':<20}{'correct names refused':>24}"
          f"{'wrong names caught':>22}{'verdict':>26}")
    print("-" * 92)
    best = None
    for threshold in THRESHOLDS:
        # a veto fires only on a confident "different"
        false_vetoes = sum(1 for r in genuine
                           if r[3] is False and r[4] >= threshold)
        caught = sum(1 for r in impostor
                     if r[3] is False and r[4] >= threshold)
        if false_vetoes == 0 and caught > 0:
            verdict = "safe and useful"
            if best is None:
                best = threshold
        elif false_vetoes == 0:
            verdict = "safe, catches nothing"
        elif caught > false_vetoes:
            verdict = "helps, but costs names"
        else:
            verdict = "does more harm than good"
        current = "  <- current" if abs(
            threshold - float(VLM_IDENTITY_MIN_CONFIDENCE)) < 1e-9 else ""
        print(f"{threshold:<20.2f}"
              f"{f'{false_vetoes} of {len(genuine)}':>24}"
              f"{f'{caught} of {len(impostor)}':>22}"
              f"{verdict:>26}{current}")
    print("-" * 92)

    refused_genuine = [r for r in genuine if r[3] is False]

    print("\nWHAT TO DO WITH THIS")
    print("-" * 68)
    if best is not None and not refused_genuine:
        print(f"Set VLM_IDENTITY_MIN_CONFIDENCE={best} in "
              f"config/settings.py (or in the environment).")
        print("That is the lowest bar at which no CORRECT name would have")
        print("been refused on this data, while wrong ones still are.")
    elif refused_genuine:
        print("Some pairs that SHOULD match were refused. Before treating")
        print("that as the vision model being wrong, check the other")
        print("possibility - it is the more likely one on this site:")
        print("")
        for kind, subject, owner, _ok, conf, path, live_a, ref_a in refused_genuine:
            print(f"  {subject}")
            print(f"    enrollment photos say : {summarise(ref_a)}")
            print(f"    this capture says     : {summarise(live_a)}")
            print(f"    {os.path.basename(path)}")
        print("")
        print("If the capture really is that person, the check is too")
        print("strict - raise VLM_IDENTITY_MIN_CONFIDENCE or leave")
        print("VLM_VERIFY_IDENTITY off.")
        print("")
        print("If it is NOT that person, you have just found a camera")
        print("photograph filed under the wrong name. That file is being")
        print("used as a reference every time the recogniser runs, so it")
        print("is actively teaching the wrong face - which is what a name")
        print("that stays wrong after being confirmed looks like. Delete")
        print("it, then re-run:  python tools/enroll_faces.py")
        print("           and:  python tools/audit_learned_faces.py")
    else:
        print("Nothing was contradicted, on genuine or impostor pairs. On")
        print("this data the check is harmless but useless: the faces are")
        print("probably too small for the model to see a contradiction.")
        print("Leave VLM_VERIFY_IDENTITY off.")
    return 0


def audit_library(only=None, limit_per_person=None, save=False):
    """Find camera photographs filed under the WRONG person.

    This is the other half of the same measurement, and on this site it
    is the more valuable half. Every cctv_*.jpg in data/faces is used as
    a recognition reference: it is compared against every live face, and
    it teaches the recogniser what that person looks like. A file filed
    under the wrong name therefore does not just sit there being wrong -
    it actively pulls other people onto that name, which is what a name
    that stays wrong even after somebody confirms it looks like.

    tools/audit_learned_faces.py already checks these by embedding. This
    checks them by DESCRIPTION, which catches the case that one cannot:
    a picture with no face in it at all. An embedding of the top of
    somebody's head is a perfectly valid vector that lands near
    somebody; a description of it says "short hair, no beard" and
    contradicts the woman it is filed under.
    """
    people = collect_people()
    if only:
        people = {n: d for n, d in people.items() if only.lower() in n.lower()}
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)

    reference_cache = {}
    suspect, checked, people_checked = [], 0, 0

    for name in sorted(people):
        data = people[name]
        if not data["enrolled"] or not data["captured"]:
            continue
        reference = describe_reference(name, data["enrolled"][:VLM_IDENTITY_REFERENCES],
                                       reference_cache)
        if not reference:
            print(f"{name:<28} could not describe the enrollment photos - "
                  f"skipped")
            continue
        people_checked += 1
        print(f"\n{name}  ({summarise(reference)}, "
              f"{len(data['captured'])} capture(s))")
        for path in data["captured"][:limit_per_person]:
            image = cv2.imread(path)
            if image is None:
                continue
            checked += 1
            live = vlm.attributes(image)
            ok, confidence, reason = vlm.same_person(live, reference)
            flag = "  <-- SUSPECT" if ok is False else ""
            print(f"    {summarise(live):<24}"
                  f"{os.path.basename(path)[:44]:<46}{flag}")
            if ok is False:
                print(f"        {reason}")
                suspect.append((name, path, live, reference, confidence))
                if save:
                    picture = vlm.montage(
                        image, load(data["enrolled"][:VLM_IDENTITY_REFERENCES]))
                    if picture is not None:
                        cv2.imwrite(os.path.join(
                            OUT_DIR,
                            f"suspect_{name[:16]}_"
                            f"{os.path.basename(path)}".replace(" ", "_")),
                            picture)

    print("\n" + "-" * 78)
    print(f"{checked} camera photograph(s) checked across {people_checked} "
          f"person(s); {len(suspect)} contradict the person they are filed "
          f"under")
    if not suspect:
        print("\nNothing to clean up. Every camera photograph agrees with "
              "the\nenrollment photos of the person it is filed under.")
        return 0

    print("\nSUSPECT FILES")
    print("-" * 78)
    for name, path, live, reference, confidence in suspect:
        print(f"  {path}")
        print(f"      filed under {name} ({summarise(reference)}) "
              f"but looks like {summarise(live)}")
    print("\nWHAT TO DO")
    print("-" * 78)
    print("LOOK at each file before deleting anything - this is a 7B model,")
    print("not an oracle, and a face at a hard angle can be described")
    print("wrongly. In practice most of these are not faces at all: the top")
    print("of a head, a shoulder, a downward blur. Those are worth deleting")
    print("whoever they are, because nothing can be recognised from them.")
    print("")
    print("After deleting, rebuild the index so the deletions take effect:")
    print("    python tools/enroll_faces.py")
    print("    python tools/audit_learned_faces.py")
    if save:
        print(f"\nComparison pictures for each suspect: {OUT_DIR}")
    return 0


def learn_people(only=None, refresh=False):
    """Describe every enrolled person once, and cache it.

    The confirmation queue offers a shortlist of who an unknown face
    could be, built by removing everybody whose own photographs
    contradict what is on screen. That needs a description of each
    person, and they are otherwise computed lazily - one the first time
    each person is named - so on a fresh install the shortlist can only
    narrow among the handful of people already seen today, and a list of
    "everybody we happen to have described" is worse than no list.

    Run this once after enrolling people. A few seconds each, cached in
    VLM_APPEARANCE_FILE, and never recomputed unless their photographs
    or the model change.
    """
    from core.vlm_verify import reference_attributes, known_people_described
    people = collect_people()
    if only:
        people = {n: d for n, d in people.items() if only.lower() in n.lower()}
    names = [n for n, d in people.items() if d["enrolled"]]
    if not names:
        print("[VLM] nobody has enrollment photographs")
        return 1

    print(f"[VLM] describing {len(names)} person(s), a few seconds each - "
          f"about {len(names) * 6 // 60 + 1} min\n")
    done = 0
    for name in sorted(names):
        attrs = reference_attributes(name, refresh=refresh)
        if attrs:
            done += 1
        else:
            print(f"    {name:<30} could not be described")
    print(f"\n[VLM] {done} of {len(names)} described; "
          f"{known_people_described()} now cached in total")
    print("[VLM] the confirmation queue will now shortlist against all of "
          "them.")
    return 0


def run_person_crops(folder, limit):
    paths = [p for p in sorted(glob.glob(os.path.join(folder, "*")))
             if p.lower().endswith(IMAGE_EXT)][:limit]
    if not paths:
        print(f"[VLM] no images in {folder}")
        return 1
    print(f"{'file':<44}{'is a person':>14}{'conf':>7}{'sec':>6}  reason")
    print("-" * 96)
    rejected = 0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        started = time.time()
        ok, confidence, reason = vlm.is_person(image)
        took = time.time() - started
        verdict = {True: "yes", False: "NO"}.get(ok, "unsure")
        if ok is False and confidence >= VLM_PERSON_MIN_CONFIDENCE:
            rejected += 1
        print(f"{os.path.basename(path)[:42]:<44}{verdict:>14}"
              f"{confidence:>7.2f}{took:>6.1f}  {reason[:30]}")
    print("-" * 96)
    print(f"\n[VLM] {rejected} of {len(paths)} would be vetoed at "
          f"VLM_PERSON_MIN_CONFIDENCE={VLM_PERSON_MIN_CONFIDENCE}")
    print("[VLM] Open the ones marked NO. If any of them is a real person, "
          "raise\n      that threshold - a wrong veto costs somebody their "
          "attendance.")
    return 0


def main(args):
    ok, names = vlm.available()
    if not ok:
        print(f"[VLM] '{VLM_MODEL}' is not available at {vlm.VLM_HOST}")
        if names:
            print(f"[VLM] Ollama has: {', '.join(names)}")
        print(f"[VLM] start Ollama, then: ollama pull {VLM_MODEL}")
        return 1
    print(f"[VLM] using {VLM_MODEL}\n")

    if args.person_crops:
        return run_person_crops(args.person_crops, args.pairs * 2)

    if args.learn_people:
        return learn_people(only=args.person, refresh=args.refresh)

    if args.audit_library:
        return audit_library(only=args.person, limit_per_person=args.per_person,
                             save=args.save)

    people = collect_people()
    if not people:
        print(f"[VLM] no enrollment folders under {FACE_DIR}")
        return 1

    with_both = [n for n, d in people.items() if d["captured"] and d["enrolled"]]
    print(f"[VLM] {len(people)} people enrolled, {len(with_both)} of them "
          f"with camera captures to test against")
    if not with_both:
        print("[VLM] nothing to compare. This check needs at least one "
              "person who has BOTH a photograph a human took and a capture "
              "the system filed (data/faces/<person>/cctv_*.jpg).")
        return 1

    pairs = build_pairs(people, args.pairs, random.Random(args.seed),
                        only=args.person)
    if not pairs:
        print("[VLM] no pairs matched")
        return 1
    print(f"[VLM] {len(pairs)} comparisons, a few seconds each - "
          f"about {len(pairs) * 4 // 60 + 1} min\n")

    results = run_identity(pairs, save=args.save)
    code = report(results)
    if args.save:
        print(f"\n[VLM] comparison pictures written to {OUT_DIR}")
        print("[VLM] look at any row marked DIFFERENT on a genuine pair - "
              "that is\n      the picture the model would have refused a "
              "correct name from.")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Measure the vision model's identity veto on this "
                    "site's own photographs")
    ap.add_argument("--pairs", type=int, default=12,
                    help="genuine pairs to test (the same number of "
                         "impostor pairs is added)")
    ap.add_argument("--person", default=None,
                    help="test only people whose name contains this")
    ap.add_argument("--save", action="store_true",
                    help="write the comparison pictures to data/vlm_check")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--audit-library", action="store_true",
                    help="instead: check every camera photograph in "
                         "data/faces against the enrollment photos of the "
                         "person it is filed under, and list the ones that "
                         "contradict it")
    ap.add_argument("--learn-people", action="store_true",
                    help="instead: describe every enrolled person once and "
                         "cache it, so the confirmation queue can shortlist "
                         "who an unknown face could be. Run once after "
                         "enrolling.")
    ap.add_argument("--refresh", action="store_true",
                    help="with --learn-people: recompute descriptions that "
                         "are already cached")
    ap.add_argument("--per-person", type=int, default=None,
                    help="with --audit-library: check at most this many "
                         "captures per person")
    ap.add_argument("--person-crops", default=None,
                    help="instead: run the 'is this a person?' check over "
                         "every image in this folder")
    raise SystemExit(main(ap.parse_args()))
