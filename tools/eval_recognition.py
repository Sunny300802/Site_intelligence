"""
tools/eval_recognition.py
=========================
Measure face recognition on YOUR OWN people, and compare the new
AdaFace + SCRFD pipeline against the old AntelopeV2/Buffalo_L one on
exactly the same images.

    python tools/eval_recognition.py --dataset data/faces
    python tools/eval_recognition.py --dataset data/faces --compare

Why this exists
---------------
Changing the model does not automatically improve accuracy. It changes
where the numbers sit, which is not the same thing - and because a
recognition threshold tuned for one model is meaningless for another,
swapping models without re-measuring can make a system WORSE while every
individual component is working correctly.

So nothing in this upgrade asks to be taken on trust. This tool runs
both backends over the same photographs, through the same detection,
alignment and quality stages, and reports:

  VERIFICATION   how far apart genuine pairs (two photos of one person)
                 and impostor pairs (two people) sit. The gap between
                 those two distributions IS the accuracy of the system;
                 the threshold merely decides where in the gap you
                 stand.

  IDENTIFICATION the question the pipeline actually asks. Each photo is
                 held out, the face bank is built from everything else,
                 and the photo is searched for exactly as a live frame
                 would be - multi-reference scoring, threshold and
                 runner-up margin included. This is the only measurement
                 that reports the number that matters on a CCTV site:
                 how often somebody is given the WRONG NAME.

  THRESHOLDS     a sweep, so FACE_RECOGNITION_THRESHOLD can be set from
                 evidence rather than copied from a blog post.

A note on what this can and cannot tell you
-------------------------------------------
Enrollment photos are not CCTV frames. A model that separates your
people cleanly here can still struggle on a 60-pixel face at the far end
of the lobby. What this measurement is genuinely good for is the
comparison - two models, same images, same stages - and for the
threshold, which is the setting people most often get wrong. For the
CCTV half, turn on FACE_TRACE, run each pipeline against the same
recording, and compare the two CSVs with --trace.
"""
import os
import sys
import glob
import argparse
import itertools
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from config.settings import (FACE_DIR, FACE_RECOGNITION_THRESHOLD,
                             FACE_MATCH_MARGIN, FACE_BANK_SCORING)

IMAGE_TYPES = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


# ------------------------------------------------------------ dataset
def load_dataset(root, min_quality, limit_per_person=0):
    """[(person, path)] from a folder of per-person sub-folders."""
    items = []
    for folder in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(folder):
            continue
        person = os.path.basename(folder)
        if "-" in person:
            head, tail = person.split("-", 1)
            if head.strip().replace(" ", "").isdigit():
                person = tail.strip()
        person = person.replace("_", " ").strip()
        files = []
        for pattern in IMAGE_TYPES:
            files += glob.glob(os.path.join(folder, pattern))
            files += glob.glob(os.path.join(folder, pattern.upper()))
        files = sorted(set(files))
        if limit_per_person:
            files = files[:limit_per_person]
        items += [(person, path) for path in files]
    return items


def embed_dataset(items, backend, min_quality, verbose=True,
                  allow_profile=False):
    """Run every image through detect -> align -> quality -> embed.

    Returns (labels, embeddings, qualities, rejected) - and the SAME
    stages the cameras use, which is what makes the comparison
    meaningful. Measuring one backend through this path and the other
    through a different one would compare the plumbing, not the models.
    """
    from core.face_detect import get_detector
    from core.face_embed import build_embedder
    from core.face_align import align, align_from_box
    from core.face_quality import assess

    detector = get_detector()
    embedder = build_embedder(backend)
    if detector is None:
        raise SystemExit("no face detector available")

    labels, vectors, qualities, rejected = [], [], [], []
    crops, pending = [], []
    for person, path in items:
        image = cv2.imread(path)
        if image is None:
            rejected.append((path, "unreadable"))
            continue
        faces = detector.detect(image)
        if not faces:
            rejected.append((path, "no face found"))
            continue
        face = faces[0]
        crop = (align(image, face["landmarks"])
                if face["landmarks"] is not None
                else align_from_box(image, face["box"]))
        quality = assess(crop, face["box"], face["landmarks"], face["score"],
                         threshold=min_quality, strict=False,
                         allow_profile=allow_profile)
        if not quality.ok:
            rejected.append((path, quality.reason))
            continue
        crops.append(crop)
        pending.append((person, path, quality.score))

    # batched, exactly as the pipeline embeds a frame's worth of faces
    for start in range(0, len(crops), 32):
        chunk = embedder.embed(crops[start:start + 32])
        for row, (person, _path, score) in zip(
                chunk, pending[start:start + 32]):
            if float(np.abs(row).sum()) < 1e-6:
                continue
            labels.append(person)
            vectors.append(row)
            qualities.append(score)

    if verbose:
        print(f"  {len(vectors)} usable face(s) from {len(items)} image(s), "
              f"{len(rejected)} rejected")
    return (labels, np.asarray(vectors, dtype=np.float32),
            np.asarray(qualities, dtype=np.float32), rejected,
            embedder.model_tag)


# -------------------------------------------------------- measurements
def verification(labels, vectors, max_impostors=200000):
    """Genuine vs impostor similarity distributions."""
    similarity = vectors @ vectors.T
    genuine, impostor = [], []
    for i, j in itertools.combinations(range(len(labels)), 2):
        (genuine if labels[i] == labels[j] else impostor).append(
            float(similarity[i, j]))
        if len(impostor) > max_impostors:
            break
    return np.asarray(genuine), np.asarray(impostor)


def identification(labels, vectors, thresholds, margins):
    """Leave-one-out search against a bank built from everything else.

    This mirrors the live pipeline exactly, including the runner-up
    margin, so the counts below are the counts that would appear on the
    dashboard: correct names, wrong names, and honest Unknowns.
    """
    from core.facebank import combine

    labels = list(labels)
    by_person = defaultdict(list)
    for index, person in enumerate(labels):
        by_person[person].append(index)

    results = {}
    similarity = vectors @ vectors.T

    # A person with only ONE usable photo cannot be measured this way:
    # hold that photo out and they have no references left, so the right
    # answer is not available and the query can only come back Unknown or
    # wrong. Counting those as failures would say the model is bad when
    # what is actually thin is the enrollment. They are excluded and
    # reported separately - which is itself the more useful finding.
    measurable = [i for i, person in enumerate(labels)
                  if len(by_person[person]) > 1]
    unmeasurable = len(labels) - len(measurable)

    for threshold in thresholds:
        for margin in margins:
            correct = wrong = unknown = 0
            for index in measurable:
                truth = labels[index]
                ranked = []
                for person, members in by_person.items():
                    others = [m for m in members if m != index]
                    if not others:
                        continue          # only one photo: no references left
                    sims = similarity[index, others]
                    centroid = vectors[others].mean(axis=0)
                    centroid /= (np.linalg.norm(centroid) + 1e-9)
                    ranked.append((combine(
                        sims,
                        centroid_similarity=float(centroid @ vectors[index])),
                        person))
                if not ranked:
                    continue
                ranked.sort(reverse=True)
                best, name = ranked[0]
                rival = ranked[1][0] if len(ranked) > 1 else 0.0

                if best < threshold or (len(ranked) > 1
                                        and best - rival < margin):
                    unknown += 1
                elif name == truth:
                    correct += 1
                else:
                    wrong += 1
            total = max(1, correct + wrong + unknown)
            results[(threshold, margin)] = {
                "correct": correct, "wrong": wrong, "unknown": unknown,
                "recall": correct / total, "error": wrong / total,
                "precision": correct / max(1, correct + wrong),
                "unmeasurable": unmeasurable}
    return results


def percentiles(values, points=(1, 5, 25, 50, 75, 95, 99)):
    if len(values) == 0:
        return {p: float("nan") for p in points}
    return {p: float(np.percentile(values, p)) for p in points}


# ------------------------------------------------------------ reports
def report_backend(name, labels, vectors, qualities, model_tag,
                   thresholds, margins):
    print(f"\n{'=' * 70}")
    print(f"  {name}   ({model_tag})")
    print(f"{'=' * 70}")
    people = len(set(labels))
    print(f"  {len(labels)} face(s), {people} people, "
          f"mean quality {float(np.mean(qualities)):.2f}")
    if people < 2:
        print("  need at least two people to measure anything.")
        return None

    genuine, impostor = verification(labels, vectors)
    g = percentiles(genuine)
    i = percentiles(impostor)
    print(f"\n  VERIFICATION ({len(genuine)} genuine, "
          f"{len(impostor)} impostor pairs)")
    print(f"    {'':14s} {'1%':>7s} {'5%':>7s} {'25%':>7s} {'50%':>7s} "
          f"{'75%':>7s} {'95%':>7s} {'99%':>7s}")
    print(f"    same person   " + "".join(f"{g[p]:7.3f}" for p in
                                          (1, 5, 25, 50, 75, 95, 99)))
    print(f"    different     " + "".join(f"{i[p]:7.3f}" for p in
                                          (1, 5, 25, 50, 75, 95, 99)))
    separation = g[5] - i[95]
    print(f"    gap between the 5th percentile of genuine and the 95th of "
          f"impostor: {separation:+.3f}")
    if separation <= 0:
        print(f"    ** the distributions OVERLAP. No threshold can separate "
              f"these people cleanly - the limit here is the photographs, "
              f"not the model. **")

    results = identification(labels, vectors, thresholds, margins)
    unmeasurable = next(iter(results.values()))["unmeasurable"]

    # The single most decision-relevant number in this whole report.
    # A person with one usable reference is a person the system will
    # keep failing to recognise from any angle but that one, and no
    # threshold or model change fixes it - a second and third photo
    # does.
    counts = defaultdict(int)
    for person in labels:
        counts[person] += 1
    thin = sorted(p for p, n in counts.items() if n < 2)
    if thin:
        print(f"\n  ** {len(thin)} of {people} people have only ONE usable "
              f"reference photo. **")
        print(f"     {', '.join(thin[:8])}"
              f"{' ...' if len(thin) > 8 else ''}")
        print(f"     They cannot be measured below (holding their only "
              f"photo out leaves nothing to match), and more importantly "
              f"they will be the hardest people for the live cameras to "
              f"recognise. Two or three frontal photos each is worth more "
              f"than any threshold or model change.")

    print(f"\n  IDENTIFICATION - leave one photo out, search the rest "
          f"(scoring '{FACE_BANK_SCORING}', {unmeasurable} single-reference "
          f"face(s) excluded)")
    print(f"    {'thresh':>7s} {'margin':>7s} {'correct':>8s} {'WRONG':>7s} "
          f"{'unknown':>8s} {'recall':>7s} {'precision':>10s}")
    for (threshold, margin), row in sorted(results.items()):
        flag = "  <- current" if (
            abs(threshold - FACE_RECOGNITION_THRESHOLD) < 1e-6 and
            abs(margin - FACE_MATCH_MARGIN) < 1e-6) else ""
        print(f"    {threshold:7.2f} {margin:7.2f} {row['correct']:8d} "
              f"{row['wrong']:7d} {row['unknown']:8d} {row['recall']:7.1%} "
              f"{row['precision']:10.1%}{flag}")

    # ---- the recommendation ------------------------------------------
    #
    # HOW MANY QUERIES IS THIS BASED ON? The identification test can only
    # use people who have two or more usable photos. When almost nobody
    # does, every threshold scores a perfect zero wrong names simply
    # because there is nothing to get wrong - and "pick the lowest
    # threshold with no errors" then returns the lowest threshold in the
    # sweep. That is not a measurement, it is an artefact, and acting on
    # it would set the bar far too low.
    #
    # So the sample size decides which evidence is used.
    measurable = sum(next(iter(results.values()))[k]
                     for k in ("correct", "wrong", "unknown"))
    MIN_QUERIES = 25

    if measurable >= MIN_QUERIES:
        # Enough queries for the identification test to mean something.
        # It optimises for NOT BEING WRONG: on an attendance system a
        # missed name costs a moment, a wrong name costs the record's
        # credibility and nobody goes looking for it.
        clean = [(key, row) for key, row in results.items()
                 if row["wrong"] == 0]
        if clean:
            key, row = max(clean, key=lambda item: item[1]["recall"])
            print(f"\n    RECOMMENDED  FACE_RECOGNITION_THRESHOLD={key[0]:.2f}  "
                  f"FACE_MATCH_MARGIN={key[1]:.2f}")
            print(f"                 no wrong names over {measurable} "
                  f"queries, {row['recall']:.0%} of faces named")
        else:
            key, row = min(results.items(), key=lambda item: item[1]["error"])
            print(f"\n    BEST AVAILABLE  threshold={key[0]:.2f} "
                  f"margin={key[1]:.2f} - still {row['wrong']} wrong "
                  f"name(s). Better photographs for the confused people "
                  f"will help more than any threshold.")
        return {"genuine": genuine, "impostor": impostor, "results": results,
                "separation": separation, "model": model_tag}

    # ---- not enough queries: fall back to the IMPOSTOR distribution ---
    #
    # This is the honest alternative, and on a thin enrollment it is
    # actually the better evidence: there are thousands of impostor
    # pairs even when there are five genuine ones, because every pair of
    # different people counts. The threshold that matters is the one no
    # stranger reaches.
    print(f"\n    ** Only {measurable} query/queries could be measured, so "
          f"the table above proves almost nothing. **")
    print(f"       Every threshold shows zero wrong names because there "
          f"are barely any answers to get wrong - not because the "
          f"settings are safe. Do not read a recommendation off it.")

    if len(impostor):
        worst = float(np.max(impostor))
        p999 = float(np.percentile(impostor, 99.9))
        safe = round(min(0.95, max(worst, p999) + 0.05), 2)
        print(f"\n    RECOMMENDED FROM THE IMPOSTOR DISTRIBUTION "
              f"({len(impostor)} pairs of DIFFERENT people)")
        print(f"       highest score any two different people reached: "
              f"{worst:.3f}   (99.9th percentile {p999:.3f})")
        print(f"       FACE_RECOGNITION_THRESHOLD={safe:.2f}   "
              f"- above every impostor pair, with headroom")
        if len(genuine):
            floor = float(np.percentile(genuine, 5))
            print(f"       for reference, genuine pairs start at "
                  f"{floor:.3f} (5th percentile of only {len(genuine)} "
                  f"pairs), so this bar has room above the strangers and "
                  f"below the real matches.")
            if safe >= floor:
                print(f"       ** but {safe:.2f} is at or above where "
                      f"genuine pairs sit. These photographs cannot "
                      f"separate your people - that is a photograph "
                      f"problem, not a threshold one. **")
    print(f"\n       The real fix is more reference photos: give the "
          f"people listed above two or three frontal shots each and "
          f"re-run this. Then the table becomes meaningful.")
    return {"genuine": genuine, "impostor": impostor, "results": results,
            "separation": separation, "model": model_tag}


def compare(a_name, a, b_name, b):
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: {a_name} vs {b_name}")
    print(f"{'=' * 70}")
    if a is None or b is None:
        print("  one of the backends produced nothing to compare.")
        return

    print(f"  separation (genuine 5th pct - impostor 95th pct)")
    print(f"    {a_name:<12s} {a['separation']:+.3f}")
    print(f"    {b_name:<12s} {b['separation']:+.3f}")
    better = a_name if a["separation"] > b["separation"] else b_name
    print(f"    -> {better} separates these people better by "
          f"{abs(a['separation'] - b['separation']):.3f}")

    def best_of(result):
        clean = [(k, v) for k, v in result["results"].items()
                 if v["wrong"] == 0]
        if clean:
            return max(clean, key=lambda item: item[1]["recall"])
        return min(result["results"].items(), key=lambda item: item[1]["error"])

    for label, result in ((a_name, a), (b_name, b)):
        key, row = best_of(result)
        print(f"\n  {label} at its own best operating point "
              f"(threshold {key[0]:.2f}, margin {key[1]:.2f}):")
        print(f"    {row['correct']} correct, {row['wrong']} WRONG, "
              f"{row['unknown']} unknown "
              f"({row['recall']:.0%} named, {row['precision']:.1%} of those "
              f"correct)")

    print(f"\n  Remember what this does and does not show: these are "
          f"ENROLLMENT PHOTOS.")
    print(f"  A model that wins here has a better chance on the cameras, "
          f"not a guarantee.")
    print(f"  For the CCTV half, set FACE_TRACE=1, run each pipeline over "
          f"the same recording,")
    print(f"  and compare the two CSVs with --trace.")


def report_trace(paths):
    """Summarise face_trace.csv files from live runs."""
    import csv
    print(f"\n{'=' * 70}")
    print(f"  LIVE TRACE SUMMARY")
    print(f"{'=' * 70}")
    for path in paths:
        if not os.path.exists(path):
            print(f"  {path}: not found")
            continue
        rows = []
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
        if not rows:
            print(f"  {path}: empty")
            continue
        backend = rows[0].get("backend", "?")
        quality = np.array([float(r["quality"]) for r in rows])
        similarity = np.array([float(r["similarity"]) for r in rows])
        accepted = np.array([int(r["accepted"]) for r in rows])
        confirmed = np.array([int(r["confirmed"]) for r in rows])
        tracks = len({r["track"] for r in rows})
        named = len({r["verdict"] for r in rows if r["verdict"] != "Unknown"})
        print(f"\n  {os.path.basename(path)}  [{backend}]")
        print(f"    {len(rows)} recognition(s) over {tracks} track(s), "
              f"{named} distinct name(s) reached")
        print(f"    quality      mean {quality.mean():.2f}  "
              f"median {np.median(quality):.2f}")
        print(f"    similarity   mean {similarity.mean():.2f}  "
              f"95th {np.percentile(similarity, 95):.2f}")
        print(f"    accepted by the bank   {accepted.mean():.0%}")
        print(f"    confirmed by the vote  {confirmed.mean():.0%}")
        # How often a track changed its mind is the identity-stability
        # number: the whole voting layer exists to drive it toward zero.
        by_track = defaultdict(list)
        for row in rows:
            by_track[row["track"]].append(row["verdict"])
        switches = sum(sum(1 for a, b in zip(v, v[1:])
                           if a != b and a != "Unknown" and b != "Unknown")
                       for v in by_track.values())
        print(f"    identity changes after a name was assigned: {switches}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure face recognition accuracy on your own people")
    parser.add_argument("--dataset", default=FACE_DIR,
                        help="folder of per-person sub-folders")
    parser.add_argument("--backend", default=None,
                        choices=["adaface", "arcface"],
                        help="which model to measure (default: the "
                             "configured one)")
    parser.add_argument("--compare", action="store_true",
                        help="measure BOTH backends on the same images")
    parser.add_argument("--min-quality", type=float, default=0.25,
                        help="reject photos below this quality (default 0.25)")
    parser.add_argument("--limit-per-person", type=int, default=0)
    parser.add_argument("--allow-profiles", action="store_true",
                        help="include near-profile photos (matches "
                             "enroll_faces.py --allow-profiles)")
    parser.add_argument("--thresholds", default=None,
                        help="comma-separated sweep, e.g. 0.25,0.30,0.35")
    parser.add_argument("--margins", default=None,
                        help="comma-separated sweep, e.g. 0.0,0.05,0.10")
    parser.add_argument("--trace", nargs="*", default=None,
                        help="summarise one or more face_trace.csv files "
                             "from live runs instead")
    args = parser.parse_args()

    if args.trace is not None:
        report_trace(args.trace or [])
        return 0

    items = load_dataset(args.dataset, args.min_quality,
                         args.limit_per_person)
    if not items:
        print(f"[EVAL] no images under {args.dataset}")
        print(f"[EVAL] expected data/faces/<id>-<name>/photo.jpg")
        return 1
    print(f"[EVAL] {len(items)} image(s) under {args.dataset}")

    if args.thresholds:
        thresholds = [float(v) for v in args.thresholds.split(",")]
    else:
        centre = FACE_RECOGNITION_THRESHOLD
        thresholds = sorted({round(centre + step, 2)
                             for step in (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
                             if centre + step > 0})
    margins = ([float(v) for v in args.margins.split(",")]
               if args.margins else [0.0, FACE_MATCH_MARGIN, 0.10])

    backends = (["adaface", "arcface"] if args.compare
                else [args.backend or None])
    outcomes = {}
    for backend in backends:
        label = backend or "configured"
        print(f"\n[EVAL] embedding with {label}...")
        try:
            labels, vectors, qualities, rejected, tag = embed_dataset(
                items, backend, args.min_quality,
                allow_profile=args.allow_profiles)
        except Exception as exc:
            print(f"[EVAL] {label} unavailable: {exc}")
            outcomes[label] = None
            continue
        if len(vectors) < 2:
            print(f"[EVAL] {label}: not enough usable faces")
            outcomes[label] = None
            continue
        outcomes[label] = report_backend(label, labels, vectors, qualities,
                                         tag, thresholds, margins)
        if rejected:
            print(f"\n  rejected photos ({len(rejected)}):")
            for path, why in rejected[:10]:
                print(f"    {os.path.basename(path):<40s} {why}")
            if len(rejected) > 10:
                print(f"    ... and {len(rejected) - 10} more")

    if args.compare:
        compare("adaface", outcomes.get("adaface"),
                "arcface", outcomes.get("arcface"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
