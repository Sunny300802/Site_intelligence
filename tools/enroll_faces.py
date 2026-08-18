"""
tools/enroll_faces.py
=====================
Teach the system your employees' faces.

FOLDER LAYOUT  (data/faces/)
----------------------------
One folder per person, named "<employee id>-<name>", with as many photos
of that person inside as you have:

    data/faces/
        12219-Rajesh Palamangalam/
            Photo_1.jpg
            Photo_2.jpg
            Photo_3.jpg
        13381 - Eswar Kumar/
            Photo_1.jpg  Photo_2.jpg  Photo_3.jpg

Both "12219-Name" and "13381 - Name" work; underscores in the name become
spaces. EVERY usable photo is enrolled as a separate reference, and more
angles means more reliable recognition on a CCTV view - the pipeline
compares a live face against all of a person's references (see
core/facebank.py), so a three-quarter reference is what matches a
three-quarter camera face that the frontal ones miss.

RUN
---
    python tools/enroll_faces.py

Writes data/face_encodings.pkl, which the pipeline loads at startup.
Re-run it whenever you add or remove an employee.

*** RE-RUN IT AFTER CHANGING THE RECOGNITION MODEL. ***
AdaFace and ArcFace produce equally valid 512-number descriptions of the
same face that have nothing to do with each other. The file records
which model made it and the pipeline refuses to load it under a
different one, so a mismatch fails loudly instead of silently matching
nobody.

WHAT IT CHECKS
--------------
Each photo goes through exactly the stages the cameras use - SCRFD, the
five-point alignment, and the quality assessment - so a reference is
built from the same geometry the live faces will have. Photos are scored
and the weak ones are named, because the most common cause of "it keeps
confusing these two people" is not the threshold: it is one blurred,
side-on or group photo enrolled as somebody's reference.
"""
import os
import sys
import glob
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from config.settings import FACE_DIR, ENCODINGS_FILE, FACE_EMBED_BACKEND

IMAGE_TYPES = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# Deliberately lower than the live FACE_QUALITY_THRESHOLD. An enrollment
# photo is a chosen photograph, not a frame that happened to be grabbed:
# a three-quarter view is a reference worth keeping rather than a
# mistake to reject. What is still refused is a picture whose landmarks
# are unusable, because aligning on those samples a near-random patch of
# the image and produces a reference that is slightly close to everybody.
DEFAULT_MIN_QUALITY = 0.30

# How well a CAMERA-CAPTURED photo must agree with that person's own
# original photographs before it is allowed to become a reference.
#
# Not a quality bar - a truth bar. It asks "is this really them?", and
# the answer has to come from photographs a human chose, because those
# are the only ones in the folder that are certain.
DEFAULT_CAPTURE_AGREE = 0.28


def parse_folder_name(folder):
    """'12219-Rajesh Palamangalam' -> ('12219', 'Rajesh Palamangalam')"""
    if "-" in folder:
        code, name = folder.split("-", 1)
        code, name = code.strip(), name.strip()
        if not code.replace(" ", "").isdigit():
            # no numeric id, treat the whole thing as the name
            code, name = "", folder.strip()
    else:
        code, name = "", folder.strip()
    return code, name.replace("_", " ").strip()


def find_images(folder):
    files = []
    for pattern in IMAGE_TYPES:
        files += glob.glob(os.path.join(folder, pattern))
        files += glob.glob(os.path.join(folder, pattern.upper()))
    return sorted(set(files))


def split_sources(paths):
    """(photographs a human took, photographs the cameras captured).

    THIS DISTINCTION IS THE WHOLE SAFETY MODEL of the self-training
    loop, and leaving it out was a serious mistake.

    A photograph an operator took is ground truth: somebody chose it and
    knew who was in it. A photograph the system captured is a CLAIM -
    it is only that person if the recognition that filed it was right.
    Enrolling the two as equals means every wrong capture becomes a
    reference photo, and a reference photo is what future recognition is
    judged against. Measured on this site after that happened: the
    genuine and impostor score distributions OVERLAPPED (a gap of -0.175
    where the operator's photos alone gave +0.374), which is the
    measurable form of "it says Akhila is Divya".

    So captures are enrolled only after they agree with the operator's
    own photographs of that person - see main().
    """
    from core.face_library import CAPTURE_PREFIX
    trusted, captured = [], []
    for path in paths:
        name = os.path.basename(path).lower()
        (captured if name.startswith(CAPTURE_PREFIX) else trusted).append(path)
    return trusted, captured


def enroll_image(path, detector, embedder, min_quality,
                 allow_profile=False):
    """One photo -> (embedding, quality) or (None, quality/None)."""
    from core.face_align import align, align_from_box
    from core.face_quality import assess

    image = cv2.imread(path)
    if image is None:
        return None, None, "unreadable"

    faces = detector.detect(image)
    if not faces:
        return None, None, "no face found"

    # The biggest face is the subject of the photo. If there are several
    # comparable ones this is a group photo, and enrolling the wrong
    # person from it is a mistake that is almost impossible to find
    # later, so it is called out rather than guessed at.
    faces.sort(key=lambda f: (f["box"][2] - f["box"][0]) *
               (f["box"][3] - f["box"][1]), reverse=True)
    face = faces[0]
    note = ""
    if len(faces) > 1:
        biggest = (face["box"][2] - face["box"][0])
        second = (faces[1]["box"][2] - faces[1]["box"][0])
        if second > 0.7 * biggest:
            note = f"{len(faces)} similar-sized faces - is this a group photo?"

    crop = (align(image, face["landmarks"]) if face["landmarks"] is not None
            else align_from_box(image, face["box"]))
    quality = assess(crop, face["box"], face["landmarks"], face["score"],
                     threshold=min_quality, strict=False,
                     allow_profile=allow_profile)
    if not quality.ok:
        return None, quality, note or quality.reason

    embedding = embedder.embed_one(crop)
    if embedding is None:
        return None, quality, "embedding failed"
    return embedding, quality, note


def main(face_dir, out_file, save_to_db=True, min_quality=DEFAULT_MIN_QUALITY,
         backend=None, allow_profile=False,
         capture_agree_min=DEFAULT_CAPTURE_AGREE):
    if not os.path.isdir(face_dir):
        print(f"[ENROLL] folder not found: {face_dir}")
        print("[ENROLL] create it and add one sub-folder per employee.")
        return 1

    people_dirs = [d for d in sorted(glob.glob(os.path.join(face_dir, "*")))
                   if os.path.isdir(d)]
    if not people_dirs:
        print(f"[ENROLL] no employee folders inside {face_dir}")
        print("[ENROLL] expected e.g. data/faces/12219-Rajesh Palamangalam/")
        return 1

    from core.face_detect import get_detector
    from core.face_embed import build_embedder

    print(f"[ENROLL] loading SCRFD + {backend or FACE_EMBED_BACKEND}...")
    detector = get_detector()
    if detector is None:
        print("[ENROLL] no face detector available - cannot enroll.")
        return 1
    try:
        embedder = build_embedder(backend)
    except Exception as exc:
        print(f"[ENROLL] no recognition model: {exc}")
        return 1

    embeddings, names, codes = [], [], []
    summary, skipped, weak = [], [], []
    rejected_captures, verified_captures, skipped_unverified = [], 0, 0

    for folder in people_dirs:
        base = os.path.basename(folder)
        code, name = parse_folder_name(base)
        images = find_images(folder)
        if not images:
            skipped.append(f"{base} (no images)")
            continue

        trusted_paths, captured_paths = split_sources(images)
        good, qualities = 0, []
        mine = []          # this person's trusted vectors, for the check

        # ---- PASS 1: the photographs a human took -------------------
        for path in trusted_paths:
            embedding, quality, note = enroll_image(
                path, detector, embedder, min_quality, allow_profile)
            filename = os.path.basename(path)
            if embedding is None:
                print(f"           ! skipped {filename}: {note}")
                continue
            if note:
                print(f"           ? {filename}: {note}")
            embeddings.append(embedding.astype(np.float32))
            names.append(name)
            codes.append(code)
            qualities.append(quality.score)
            mine.append(embedding.astype(np.float32))
            good += 1

        # ---- PASS 2: photographs the cameras captured ---------------
        # Admitted only if they agree with pass 1. A capture is a claim,
        # not a fact, and an unverified claim promoted to a reference
        # photo is how one mistake becomes permanent.
        if captured_paths and not mine:
            skipped_unverified += len(captured_paths)
            print(f"           ! {len(captured_paths)} camera photo(s) "
                  f"ignored - no original photo of {name} to check against")
        elif captured_paths:
            reference = np.stack(mine)
            centroid = reference.mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-9)
            for path in captured_paths:
                embedding, quality, note = enroll_image(
                    path, detector, embedder, min_quality, allow_profile)
                filename = os.path.basename(path)
                if embedding is None:
                    continue
                vector = embedding.astype(np.float32)
                agreement = float(max(reference @ vector,
                                      default=0.0))
                agreement = max(agreement, float(centroid @ vector))
                if agreement < capture_agree_min:
                    rejected_captures.append(
                        (name, filename, agreement))
                    continue
                embeddings.append(vector)
                names.append(name)
                codes.append(code)
                qualities.append(quality.score)
                good += 1
                verified_captures += 1

        summary.append((code, name, good))
        best = max(qualities) if qualities else 0.0
        status = "ok" if good >= 2 else ("thin" if good == 1 else "FAILED")
        print(f"  [{status:>6}] {code:>8}  {name:<28} {good} photo(s)"
              f"  best quality {best:.2f}")
        if good and best < 0.55:
            weak.append(f"{name} (best {best:.2f})")

    if not embeddings:
        print("\n[ENROLL] nothing enrolled - no usable faces found.")
        print("[ENROLL] try --min-quality 0.2 to see what is being rejected.")
        return 1

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as handle:
        pickle.dump({"embeddings": np.vstack(embeddings),
                     "names": names, "codes": codes,
                     "format": "site-intelligence-v2",
                     # WHICH MODEL produced these. Embeddings from
                     # different models are not comparable, so the
                     # pipeline checks this before trusting them.
                     "embed_model": embedder.model_tag,
                     "dims": int(embeddings[0].size)}, handle)

    people = len(set(names))
    print(f"\n[ENROLL] {len(embeddings)} reference face(s) for {people} people")
    print(f"[ENROLL] model: {embedder.model_tag}")
    print(f"[ENROLL] saved -> {out_file}")

    thin = [n for _c, n, g in summary if g == 1]
    if thin:
        print(f"[ENROLL] only one usable photo for: {', '.join(thin)}")
        print("[ENROLL] add 2-3 angles each for noticeably better accuracy.")
    if weak:
        print(f"[ENROLL] weak reference photos: {', '.join(weak)}")
        print("[ENROLL] these people will be the hardest to recognise and "
              "the most likely to be confused with somebody else. Better "
              "photos for them is worth more than any threshold change.")
    if skipped:
        print(f"[ENROLL] skipped: {', '.join(skipped)}")

    if verified_captures or rejected_captures or skipped_unverified:
        print(f"\n[ENROLL] camera-captured photos: {verified_captures} "
              f"verified against the original photos, "
              f"{len(rejected_captures)} rejected"
              f"{f', {skipped_unverified} unverifiable' if skipped_unverified else ''}")
        for who, filename, score in sorted(rejected_captures,
                                           key=lambda r: r[2])[:15]:
            print(f"           - {who[:24]:<25} {filename[:44]:<45} "
                  f"agrees {score:.2f}")
        if len(rejected_captures) > 15:
            print(f"           ... and {len(rejected_captures) - 15} more")
        if rejected_captures:
            print(f"[ENROLL] those are probably somebody else. Delete them "
                  f"with:  python tools/audit_learned_faces.py --apply "
                  f"--delete-photos")

    if save_to_db:
        try:
            from database.db import init_db
            from database import repository as repo
            init_db()
            repo.replace_employees(summary)
            print("[ENROLL] employee list written to the database")
        except Exception as exc:
            print(f"[ENROLL] (database update skipped: {exc})")

    print("\n[ENROLL] done. Restart run_pipeline.py to load the new faces.")
    print("[ENROLL] then measure the right threshold for THIS set of people:")
    print("[ENROLL]     python tools/eval_recognition.py --dataset data/faces")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll employee faces")
    parser.add_argument("--dir", default=FACE_DIR,
                        help="folder holding one sub-folder per employee")
    parser.add_argument("--out", default=ENCODINGS_FILE)
    parser.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY,
                        help="reject reference photos below this quality "
                             f"(default {DEFAULT_MIN_QUALITY})")
    parser.add_argument("--backend", default=None,
                        choices=["adaface", "arcface"],
                        help="recognition model to enroll with (default: "
                             "whatever FACE_EMBED_BACKEND says)")
    parser.add_argument("--capture-agree-min", type=float,
                        default=DEFAULT_CAPTURE_AGREE,
                        help="how well a camera-captured photo must agree "
                             "with that person's original photos before it "
                             "is enrolled (default "
                             f"{DEFAULT_CAPTURE_AGREE})")
    parser.add_argument("--allow-profiles", action="store_true",
                        help="also enroll near-profile photos. Off by "
                             "default: the cameras will never accept a "
                             "profile face, so a profile reference can "
                             "never be matched and only dilutes the "
                             "person's average")
    parser.add_argument("--no-db", action="store_true",
                        help="do not update the employee table")
    args = parser.parse_args()
    raise SystemExit(main(args.dir, args.out, save_to_db=not args.no_db,
                          min_quality=args.min_quality, backend=args.backend,
                          allow_profile=args.allow_profiles,
                          capture_agree_min=args.capture_agree_min))
