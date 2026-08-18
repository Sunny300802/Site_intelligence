"""
tools/audit_learned_faces.py
============================
Find and remove face samples that were learned under the wrong name.

    python tools/audit_learned_faces.py            # report only
    python tools/audit_learned_faces.py --apply    # actually remove them

Why this is needed
------------------
Every face a human confirms becomes a permanent reference for that
person, and a camera-realistic reference outweighs an enrollment photo.
That is what makes confirmation powerful - and it is also what makes a
mistaken confirmation expensive: one wrong answer files somebody else's
face under your name, and from then on the system recognises that other
person AS you.

Two things made mistaken confirmations easy to give:

  * questions were asked about sightings with NO usable face - measured
    on this site, 93% of the last 120 questions carried no face vector
    at all - so the card showed a body, a leg or the back of a head;
  * the name on the card came from a CLOTHING guess, and that guess
    changed from one question to the next. One track was asked about as
    Divya, then Ravi Teja, then Sony, then Sujana.

Both are fixed going forward, but the samples already stored under those
answers are still in the database, still being searched, and still
capable of putting the wrong name on somebody. This tool finds them.

How a sample is judged
----------------------
Each stored sample is compared against that person's ENROLLMENT photos -
the ones a human deliberately took of them - using the same scoring the
live pipeline uses. A genuine sample of Akhila looks like Akhila's
enrollment photos. One that does not is either somebody else's face
filed under her name, or a picture with no face in it at all.

Nothing is deleted without --apply, and the report names every sample
and its score so the decision can be checked before it is made.
"""
import os
import sys
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config.settings import FACE_LEARN_AGREE_MIN


def main():
    parser = argparse.ArgumentParser(
        description="Find face samples learned under the wrong name")
    parser.add_argument("--apply", action="store_true",
                        help="remove the disagreeing samples (default: "
                             "report only)")
    parser.add_argument("--min-agree", type=float, default=None,
                        help=f"agreement below which a sample is rejected "
                             f"(default {FACE_LEARN_AGREE_MIN})")
    parser.add_argument("--delete-photos", action="store_true",
                        help="also delete the captured .jpg the sample came "
                             "from, so re-enrolling cannot bring it back")
    args = parser.parse_args()

    from database.db import SessionLocal, init_db
    from database.models import LearnedFace
    from core.face import init_face_bank, load_encodings
    from core.facebank import FaceBank
    from core.face_embed import tags_compatible, model_tag

    init_db()

    # The reference set is the ENROLLMENT file only - the photographs a
    # human deliberately took. Judging learned samples against a bank
    # that already contains those learned samples would be circular: a
    # wrong sample would happily agree with itself.
    embeddings, names, codes = load_encodings()
    if embeddings is None or not len(embeddings):
        print("[AUDIT] no enrollment file - nothing to judge against.")
        print("[AUDIT] run: python tools/enroll_faces.py")
        return 1

    reference = FaceBank(model_tag=model_tag())
    reference.load(embeddings, names, codes)
    print(f"[AUDIT] judging against {len(reference)} enrolled people "
          f"({reference.reference_count} photos), model {model_tag()}")

    bar = FACE_LEARN_AGREE_MIN if args.min_agree is None else args.min_agree
    print(f"[AUDIT] a sample must agree at least {bar:.2f} with its own "
          f"person's enrollment photos\n")

    with SessionLocal() as session:
        rows = session.query(LearnedFace).order_by(LearnedFace.id).all()
        if not rows:
            print("[AUDIT] no learned faces stored.")
            return 0

        current = model_tag()
        keep, drop, unjudgeable = [], [], []
        per_person = collections.defaultdict(lambda: [0, 0])

        for row in rows:
            try:
                vector = np.frombuffer(row.embedding, dtype=np.float32)
            except Exception:
                unjudgeable.append((row, "unreadable vector"))
                continue
            if vector.size != row.dims:
                unjudgeable.append((row, "truncated vector"))
                continue
            if not tags_compatible(row.model_tag or "", current):
                unjudgeable.append((row, f"made by {row.model_tag}"))
                continue

            score = reference.score_against(row.person_name, vector)
            if score is None:
                # nobody enrolled under that name to contradict it
                unjudgeable.append((row, "person not in the enrollment file"))
                continue

            per_person[row.person_name][1] += 1
            if score >= bar:
                keep.append((row, score))
                per_person[row.person_name][0] += 1
            else:
                drop.append((row, score))

        print(f"{'person':<28}{'kept':>6}{'REJECTED':>10}   worst")
        print("-" * 60)
        worst = collections.defaultdict(lambda: 1.0)
        for row, score in drop:
            worst[row.person_name] = min(worst[row.person_name], score)
        for name in sorted(per_person):
            good, total = per_person[name]
            bad = total - good
            flag = "   <-- check this person" if bad else ""
            print(f"{name[:27]:<28}{good:>6}{bad:>10}   "
                  f"{worst[name]:.2f}{flag}" if bad else
                  f"{name[:27]:<28}{good:>6}{bad:>10}")
        print("-" * 60)
        print(f"{'TOTAL':<28}{len(keep):>6}{len(drop):>10}")

        if unjudgeable:
            print(f"\n{len(unjudgeable)} sample(s) could not be judged:")
            reasons = collections.Counter(r for _row, r in unjudgeable)
            for reason, count in reasons.most_common():
                print(f"   {count:4d}  {reason}")

        if drop:
            print(f"\nthe {len(drop)} rejected sample(s):")
            for row, score in sorted(drop, key=lambda item: item[1])[:25]:
                where = os.path.basename(row.image_path or "") or "(no photo)"
                print(f"   {score:.2f}  {row.person_name[:24]:<25} "
                      f"{row.source:<10} {where}")
            if len(drop) > 25:
                print(f"   ... and {len(drop) - 25} more")

        if not args.apply:
            print(f"\n[AUDIT] report only. Re-run with --apply to remove the "
                  f"{len(drop)} rejected sample(s).")
            if args.delete_photos:
                print(f"[AUDIT] (--delete-photos also needs --apply)")
            return 0

        removed_photos = 0
        for row, _score in drop:
            if args.delete_photos and row.image_path and \
                    os.path.exists(row.image_path):
                try:
                    os.remove(row.image_path)
                    removed_photos += 1
                except OSError as exc:
                    print(f"[AUDIT] could not delete {row.image_path}: {exc}")
            session.delete(row)
        session.commit()
        print(f"\n[AUDIT] removed {len(drop)} sample(s)"
              f"{f' and {removed_photos} photo(s)' if removed_photos else ''}.")
        print(f"[AUDIT] restart run_pipeline.py so the search index is "
              f"rebuilt without them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
