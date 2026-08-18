"""
tools/diagnose_person.py
========================
Why is this one person being recognised wrongly?

    python tools/diagnose_person.py "Srikanth Nandigama"
    python tools/diagnose_person.py "Srikanth" --fix

"It gets Srikanth wrong" has half a dozen possible causes, and they need
opposite remedies. Guessing between them is what turns a ten-minute fix
into a week of threshold tweaking, so this checks all of them and says
which one it is:

  TOO FEW REFERENCES     one photo means one pose. Any other angle is
                         unmatched, and the person stays Unknown.
  A BAD REFERENCE        a photo of somebody else filed under their
                         name - usually an unverified camera capture.
                         This is the one that produces a WRONG name
                         rather than no name, because that reference
                         matches the other person on sight.
  A LOOK-ALIKE           somebody genuinely close to them in the
                         embedding space. Needs a bigger margin, or
                         better photos of both.
  POOR PHOTO QUALITY     blurred, side-on or dark references match
                         nothing reliably.
  CONTAMINATED LEARNING  samples stored in the database under their name
                         that disagree with their own photographs.

Everything is measured against the ENROLLMENT PHOTOGRAPHS - the ones a
human deliberately took - because those are the only images in the
folder that are certain to be that person.
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose why one person is recognised wrongly")
    parser.add_argument("person", help="their name, or part of it")
    parser.add_argument("--fix", action="store_true",
                        help="delete the camera captures that disagree with "
                             "their own photographs")
    parser.add_argument("--top", type=int, default=6,
                        help="how many look-alikes to list")
    args = parser.parse_args()

    from database.db import init_db, SessionLocal
    from database.models import LearnedFace
    from core.face import load_encodings
    from core.facebank import FaceBank, combine
    from core.face_embed import model_tag, tags_compatible, build_embedder
    from core.face_detect import get_detector
    from core.face_align import align, align_from_box
    from core.face_quality import assess
    from core.face_library import LIBRARY, CAPTURE_PREFIX
    from config.settings import (FACE_RECOGNITION_THRESHOLD,
                                 FACE_MATCH_MARGIN, FACE_LIBRARY_TARGET)

    init_db()
    embeddings, names, codes = load_encodings()
    if embeddings is None or not len(embeddings):
        print("[DIAGNOSE] no enrollment file - run tools/enroll_faces.py")
        return 1

    # who did they mean?
    wanted = args.person.strip().lower()
    matches = sorted({n for n in names if wanted in n.lower()})
    if not matches:
        print(f"[DIAGNOSE] nobody enrolled matches {args.person!r}")
        print(f"[DIAGNOSE] enrolled: {', '.join(sorted(set(names))[:20])} ...")
        return 1
    if len(matches) > 1:
        print(f"[DIAGNOSE] {args.person!r} matches several people: "
              f"{', '.join(matches)}")
        return 1
    person = matches[0]
    code = ""
    for n, c in zip(names, codes):
        if n == person and c:
            code = c
            break

    print(f"\n{'=' * 70}")
    print(f"  {person}" + (f"  (id {code})" if code else ""))
    print(f"  model: {model_tag()}")
    print(f"{'=' * 70}")

    bank = FaceBank(model_tag=model_tag())
    bank.load(embeddings, names, codes)

    # ---------------------------------------------------- 1. references
    mine = np.stack([v for v, n in zip(embeddings, names) if n == person])
    mine = mine / np.maximum(np.linalg.norm(mine, axis=1, keepdims=True), 1e-9)
    print(f"\n1. REFERENCES")
    print(f"   {len(mine)} reference face(s) in the enrollment file")
    if len(mine) < 2:
        print(f"   ** ONE reference means one pose. Any other angle will not")
        print(f"      match, and they will stay Unknown most of the time.")
        print(f"      Two or three frontal photos is the fix.")

    # how consistent are they with each other?
    if len(mine) > 1:
        pairs = mine @ mine.T
        off = pairs[~np.eye(len(mine), dtype=bool)]
        print(f"   agreement between their own references: "
              f"min {off.min():.2f}  median {np.median(off):.2f}  "
              f"max {off.max():.2f}")
        if off.min() < 0.15:
            print(f"   ** at least two of their references barely agree with")
            print(f"      each other. One of them is probably NOT them - see")
            print(f"      the photo check below.")

    # ---------------------------------------------------- 2. the photos
    folder = LIBRARY.folder_for(person, code, create=False)
    print(f"\n2. PHOTOGRAPHS ON DISK")
    if not folder:
        print(f"   no folder found under data/faces for {person}")
    else:
        files = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                       if os.path.isfile(p))
        human = [p for p in files
                 if not os.path.basename(p).lower().startswith(CAPTURE_PREFIX)]
        captured = [p for p in files if p not in human]
        print(f"   {os.path.basename(folder)}")
        print(f"   {len(human)} taken by a human, {len(captured)} captured "
              f"by the cameras")

        detector, embedder = get_detector(), build_embedder(quiet=True)
        if detector is not None and embedder is not None and human:
            # the trusted set, rebuilt from the human photos alone
            trusted = []
            print(f"\n   their own photographs:")
            for path in human:
                image = cv2.imread(path)
                if image is None:
                    continue
                faces = detector.detect(image)
                if not faces:
                    print(f"     {os.path.basename(path)[:44]:<45} no face found")
                    continue
                face = faces[0]
                crop = (align(image, face["landmarks"])
                        if face["landmarks"] is not None
                        else align_from_box(image, face["box"]))
                quality = assess(crop, face["box"], face["landmarks"],
                                 face["score"], strict=False)
                vector = embedder.embed_one(crop)
                if vector is not None:
                    trusted.append(vector)
                print(f"     {os.path.basename(path)[:44]:<45} "
                      f"quality {quality.score:.2f}  {quality.reason[:28]}")

            if trusted and captured:
                reference = np.stack(trusted)
                reference /= np.maximum(
                    np.linalg.norm(reference, axis=1, keepdims=True), 1e-9)
                print(f"\n   camera captures, checked against those:")
                suspect = []
                for path in captured:
                    image = cv2.imread(path)
                    if image is None:
                        continue
                    faces = detector.detect(image)
                    if not faces:
                        suspect.append((path, -1.0))
                        print(f"     {os.path.basename(path)[:44]:<45} "
                              f"NO FACE IN IT")
                        continue
                    face = faces[0]
                    crop = (align(image, face["landmarks"])
                            if face["landmarks"] is not None
                            else align_from_box(image, face["box"]))
                    vector = embedder.embed_one(crop)
                    if vector is None:
                        continue
                    agreement = float((reference @ vector).max())
                    flag = "" if agreement >= 0.28 else "   <-- NOT THEM"
                    if agreement < 0.28:
                        suspect.append((path, agreement))
                    print(f"     {os.path.basename(path)[:44]:<45} "
                          f"agrees {agreement:+.2f}{flag}")

                if suspect:
                    print(f"\n   ** {len(suspect)} camera photo(s) filed under "
                          f"{person} are probably somebody else.")
                    print(f"      Each one is a REFERENCE, so each one teaches")
                    print(f"      the system to answer '{person}' when it sees")
                    print(f"      that other person. This is the usual cause of")
                    print(f"      a confident wrong name.")
                    if args.fix:
                        for path, _score in suspect:
                            try:
                                os.remove(path)
                                print(f"      removed {os.path.basename(path)}")
                            except OSError as exc:
                                print(f"      could not remove {path}: {exc}")
                        print(f"      re-run tools/enroll_faces.py to rebuild.")
                    else:
                        print(f"      re-run with --fix to delete them.")

    # ---------------------------------------------------- 3. look-alikes
    print(f"\n3. WHO THEY GET CONFUSED WITH")
    centroid = mine.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-9)
    rivals = []
    for other in sorted(set(names)):
        if other == person:
            continue
        theirs = np.stack([v for v, n in zip(embeddings, names) if n == other])
        theirs = theirs / np.maximum(
            np.linalg.norm(theirs, axis=1, keepdims=True), 1e-9)
        best = float((theirs @ mine.T).max())
        rivals.append((best, other))
    rivals.sort(reverse=True)
    print(f"   threshold {FACE_RECOGNITION_THRESHOLD}, "
          f"margin {FACE_MATCH_MARGIN}")
    for score, other in rivals[:args.top]:
        warn = ""
        if score >= FACE_RECOGNITION_THRESHOLD:
            warn = "   <-- ABOVE the threshold: can be mistaken for them"
        elif score >= FACE_RECOGNITION_THRESHOLD - FACE_MATCH_MARGIN:
            warn = "   <-- within the margin: both refused, stays Unknown"
        print(f"     {score:+.2f}  {other[:40]:<41}{warn}")

    # ---------------------------------------------------- 4. learned rows
    print(f"\n4. SAMPLES LEARNED SINCE ENROLMENT")
    with SessionLocal() as session:
        rows = session.query(LearnedFace).filter(
            LearnedFace.person_name == person).all()
        current = model_tag()
        usable = bad = incompatible = 0
        for row in rows:
            if not tags_compatible(row.model_tag or "", current):
                incompatible += 1
                continue
            try:
                vector = np.frombuffer(row.embedding, dtype=np.float32)
            except Exception:
                continue
            if vector.size != row.dims:
                continue
            score = bank.score_against(person, vector)
            if score is not None and score >= 0.24:
                usable += 1
            else:
                bad += 1
        print(f"   {len(rows)} stored, {usable} agree with their photos, "
              f"{bad} do NOT, {incompatible} from a different model")
        if bad:
            print(f"   ** those {bad} are actively producing wrong matches.")
            print(f"      python tools/audit_learned_faces.py --apply "
                  f"--delete-photos")

    # ---------------------------------------------------- verdict
    print(f"\n{'=' * 70}")
    print("  WHAT TO DO")
    print(f"{'=' * 70}")
    todo = []
    if len(mine) < 3:
        todo.append(f"Add 2-3 more frontal photographs of {person} to "
                    f"data/faces/ - they have only {len(mine)}.")
    if rivals and rivals[0][0] >= FACE_RECOGNITION_THRESHOLD:
        todo.append(f"{rivals[0][1]} scores {rivals[0][0]:.2f} against them, "
                    f"above the {FACE_RECOGNITION_THRESHOLD} threshold. "
                    f"Better photos of BOTH, or raise FACE_MATCH_MARGIN.")
    if not todo:
        todo.append("Nothing obviously wrong with their references. If they "
                    "are still misidentified, capture the live scores with "
                    "FACE_TRACE=1 and check what is actually matching.")
    for i, line in enumerate(todo, 1):
        print(f"  {i}. {line}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
