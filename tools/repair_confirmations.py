#!/usr/bin/env python
"""
tools/repair_confirmations.py
=============================
Recover what past confirmations should have taught, and merge names that
were split by spelling.

Why this exists
---------------
Two faults meant months of human answers taught the system almost
nothing, and neither showed up as an error:

1. CONFIRMATIONS NEVER TAUGHT A FACE. Confirming re-detected a face
   inside the saved face crop - but that crop is a tight box around the
   face, and a face detector needs context around a face to find one.
   Measured on the saved crops: 0 of 25 worked. So every confirmation
   fell through to "clothing only", which is discarded overnight, and
   learned_faces stayed empty for ever.

   The body crop does have that context (25 of 25), so the face can be
   recovered from questions already answered.

2. NAMES WERE NOT NORMALISED. "shivajyothi" and "Shivajyothi" became two
   people in the gallery, each holding half the evidence and each too
   weak to be recognised.

Run it once after upgrading:

    python tools/repair_confirmations.py            # show what it would do
    python tools/repair_confirmations.py --apply    # actually do it
"""
import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import init_db, SessionLocal          # noqa: E402
from database import repository as repo                # noqa: E402
from database.models import (ReviewItem, LearnedFace,  # noqa: E402
                             LearnedAppearance)
from config.settings import FACE_LEARN_MIN_WIDTH   # noqa: E402


def normalise_names(apply_changes):
    """Snap every stored name to the one enrolled spelling."""
    print("\n=== 1. name spellings ===")
    changes = []
    with SessionLocal() as s:
        targets = [
            (ReviewItem, "confirmed_name"),
            (LearnedFace, "person_name"),
            (LearnedAppearance, "person_name"),
        ]
        for model, field in targets:
            for row in s.query(model).all():
                current = getattr(row, field) or ""
                if not current:
                    continue
                fixed = repo.canonical_name(current)
                if fixed != current:
                    changes.append((model.__tablename__, field,
                                    current, fixed))
                    if apply_changes:
                        setattr(row, field, fixed)
        if apply_changes:
            s.commit()

    if not changes:
        print("   every stored name already matches its enrolled spelling")
        return 0
    seen = {}
    for table, _f, was, now in changes:
        seen.setdefault((was, now), []).append(table)
    for (was, now), tables in sorted(seen.items()):
        print(f"   {was!r} -> {now!r}   ({len(tables)} row(s) in "
              f"{', '.join(sorted(set(tables)))})")
    print(f"   {len(changes)} row(s) {'updated' if apply_changes else 'would change'}")
    return len(changes)


def recover_faces(apply_changes):
    """Re-embed confirmed sightings from their BODY crop."""
    print("\n=== 2. faces from past confirmations ===")
    import cv2

    with SessionLocal() as s:
        rows = s.query(ReviewItem).filter(
            ReviewItem.status == "confirmed").all()
        items = [{"id": r.id, "name": r.confirmed_name or r.suggested_name,
                  "body": r.body_image, "camera_key": r.camera_key}
                 for r in rows]
        already = {(f.person_name, f.camera_key)
                   for f in s.query(LearnedFace).all()}

    usable = [i for i in items
              if i["name"] and i["body"] and os.path.exists(i["body"])]
    print(f"   {len(items)} confirmed answer(s), "
          f"{len(usable)} with a body crop still on disk")
    if already:
        print(f"   {len(already)} person/camera pair(s) already have a "
              f"learned face")
    if not usable:
        return 0

    if not apply_changes:
        print("   (dry run - not loading the face model)")
        return 0

    # Re-embedded through the SAME stages the cameras use - SCRFD,
    # five-point alignment, quality filtering, AdaFace. A vector made any
    # other way is not comparable with the enrolled ones, so a sample
    # recovered through a different code path would be a reference that
    # never matches anybody.
    from core.face import agrees_with_enrollment, embed_face, init_face_bank
    init_face_bank()

    added = skipped_small = skipped_noface = rejected = 0
    per_person = {}
    disagreed = {}
    for i in usable:
        img = cv2.imread(i["body"])
        if img is None:
            continue
        embedding, quality = embed_face(img)
        if embedding is None:
            if quality is None:
                skipped_noface += 1
            else:
                skipped_small += 1
            continue
        # A past answer is not automatically true. If the sighting does
        # not look like the person it was filed under, the answer was
        # given while the system was suggesting the wrong name - learning
        # it would bake that mistake in permanently.
        ok, score = agrees_with_enrollment(i["name"], embedding)
        if not ok:
            rejected += 1
            d = disagreed.setdefault(i["name"], [])
            d.append(score)
            continue
        repo.add_learned_face(i["name"], embedding, source="recovered",
                              camera_key=i["camera_key"] or "")
        added += 1
        per_person[i["name"]] = per_person.get(i["name"], 0) + 1

    print(f"   recovered  {added} face sample(s)")
    print(f"   skipped    {skipped_noface} (no face found in the crop)")
    print(f"   skipped    {skipped_small} (face found but too poor to learn "
          f"from - too small, blurry, side-on or badly lit)")
    print(f"   REJECTED   {rejected} (does not look like the person it was "
          f"confirmed as)")
    if per_person:
        print("   learned per person:")
        for n, k in sorted(per_person.items(), key=lambda kv: -kv[1]):
            print(f"      {k:3d}  {n}")
    if disagreed:
        print("\n   These answers were NOT learned - the sighting does not")
        print("   match the enrolled person. Either the answer was wrong,")
        print("   or that person's enrollment photos need redoing:")
        for n, scores in sorted(disagreed.items(), key=lambda kv: -len(kv[1])):
            import statistics
            print(f"      {len(scores):3d}  {n:28s} "
                  f"median agreement {statistics.median(scores):.2f}")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default is a dry run)")
    args = ap.parse_args()

    init_db()
    print("=" * 62)
    print(" Repair past confirmations"
          f"{'' if args.apply else '   (DRY RUN - nothing will be written)'}")
    print("=" * 62)

    normalise_names(args.apply)
    recover_faces(args.apply)

    print("\n" + "=" * 62)
    if args.apply:
        print(" done. Restart run_pipeline.py to load the recovered faces.")
    else:
        print(" dry run only. Re-run with --apply to make these changes.")
    print("=" * 62)


if __name__ == "__main__":
    main()
