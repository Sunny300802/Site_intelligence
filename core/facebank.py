"""
core/facebank.py
================
The vector similarity search: one embedding in, a name out.

Multiple reference images per person
------------------------------------
Everybody is enrolled from several photographs, and every face a human
later confirms on the dashboard is added to their references too. So the
question "who is this?" is never a comparison against ONE vector per
person - it is a comparison against a small set per person, and how
those are combined matters more than it looks:

  max        the best single reference wins. Most sensitive: a
             three-quarter reference will match a three-quarter CCTV
             face that the frontal ones miss. Also the most fragile -
             one bad enrollment photo (wrong person cropped, or a face
             so poor it sits near the middle of the space) becomes a
             reference that is slightly close to EVERYBODY, and that
             single row then produces wrong names for months.

  centroid   compare against the person's mean vector. Robust, because
             one bad reference is diluted by the others. Loses the
             profile shot: averaging a frontal and a profile gives a
             vector that is neither, and matches neither well.

  topk       the mean of that person's best k references. Keeps most of
             max's sensitivity while requiring TWO references to agree,
             which is what kills the single-bad-photo failure.

  blend      the default: part centroid, part top-k. The centroid term
             is a stability floor - a person whose references genuinely
             agree gets a lift from it - and the top-k term keeps the
             angles. Set FACE_BANK_CENTROID_WEIGHT to 0 for pure top-k
             or 1 for pure centroid.

Two gates, not one
------------------
Clearing FACE_RECOGNITION_THRESHOLD is necessary but not sufficient: the
winner must also be clearly ahead of the best DIFFERENT person. Taking
the argmax alone is what produced wrong names in the old pipeline.
Colleagues do resemble each other, and a mediocre CCTV face lands close
to several enrolled people at once - 0.53 for one and 0.51 for another
is not a recognition, it is a coin toss that was being printed as a
confident name. Several references belonging to the SAME person are
corroboration, never rivals, so this costs nothing for well-enrolled
people.
"""
import threading

import numpy as np

from config.settings import (FACE_RECOGNITION_THRESHOLD, FACE_MATCH_MARGIN,
                             FACE_BANK_SCORING, FACE_BANK_TOPK,
                             FACE_BANK_CENTROID_WEIGHT)


def _normalise(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def combine(similarities, mode=None, topk=None, centroid_similarity=None,
            centroid_weight=None):
    """One person's per-reference similarities -> one score.

    Shared with core/gallery.py so that the gallery and the face bank
    cannot silently disagree about what "how well does this face match
    this person" means - two different answers to that question is how a
    name ends up accepted in one place and refused in the other.
    """
    similarities = np.asarray(similarities, dtype=np.float32).ravel()
    if similarities.size == 0:
        return None

    mode = (mode or FACE_BANK_SCORING or "blend").lower()
    k = max(1, int(FACE_BANK_TOPK if topk is None else topk))
    weight = float(FACE_BANK_CENTROID_WEIGHT if centroid_weight is None
                   else centroid_weight)

    best = float(similarities.max())
    if mode == "max" or similarities.size == 1:
        top = best
    else:
        k = min(k, similarities.size)
        top = float(np.sort(similarities)[-k:].mean())

    if mode == "max":
        return best
    if mode == "topk":
        return top
    if mode == "centroid":
        return best if centroid_similarity is None else float(centroid_similarity)
    # "blend"
    if centroid_similarity is None:
        return top
    return float(weight * centroid_similarity + (1.0 - weight) * top)


class BankMatch:
    """The answer to 'who is this face?'"""

    __slots__ = ("name", "code", "score", "runner_up", "runner_score",
                 "margin", "accepted", "reason")

    def __init__(self, name="Unknown", code="", score=0.0, runner_up="",
                 runner_score=0.0, accepted=False, reason=""):
        self.name = name
        self.code = code
        self.score = float(score)
        self.runner_up = runner_up
        self.runner_score = float(runner_score)
        self.margin = float(score) - float(runner_score)
        self.accepted = bool(accepted)
        self.reason = reason

    def __repr__(self):
        return (f"<BankMatch {self.name!r} {self.score:.3f} "
                f"(margin {self.margin:+.3f}) "
                f"{'accept' if self.accepted else self.reason}>")


class FaceBank:
    """Everybody's reference embeddings, and the search over them.

    Rebuilt in place whenever new references arrive (a confirmation from
    the dashboard), so a person confirmed at 10:05 is recognised from
    10:05 rather than after the next restart.
    """

    def __init__(self, model_tag=""):
        self.model_tag = model_tag
        self._lock = threading.RLock()
        self._people = {}          # name -> {"code": str, "vectors": [...]}
        self._names = []           # person index -> name
        self._codes = []
        self._matrix = None        # [total_refs, dims], L2-normalised
        self._owner = None         # [total_refs] -> person index
        self._slices = []          # person index -> (start, stop)
        self._centroids = None     # [people, dims], L2-normalised
        self._dirty = False

    # ------------------------------------------------------- building
    def add(self, name, embedding, code=""):
        """Add one reference. Cheap; the index rebuilds lazily."""
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        if vector.size < 64 or not np.isfinite(vector).all():
            return False
        if float(np.abs(vector).sum()) <= 1e-6:
            return False
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        with self._lock:
            person = self._people.setdefault(name, {"code": "", "vectors": []})
            if code and not person["code"]:
                person["code"] = str(code)
            person["vectors"].append(vector)
            self._dirty = True
        return True

    def load(self, embeddings, names, codes=None):
        """Bulk-load the enrollment file. Returns the number of people."""
        if embeddings is None or len(embeddings) == 0:
            return 0
        codes = list(codes or []) or [""] * len(names)
        for vector, name, code in zip(embeddings, names, codes):
            self.add(name, vector, code)
        self.build()
        return len(self._people)

    def build(self):
        """Materialise the search matrices. Idempotent."""
        with self._lock:
            if not self._dirty and self._matrix is not None:
                return
            names, codes, rows, owner, slices, centroids = [], [], [], [], [], []
            cursor = 0
            for index, (name, person) in enumerate(sorted(self._people.items())):
                vectors = person["vectors"]
                if not vectors:
                    continue
                stack = np.stack(vectors).astype(np.float32)
                names.append(name)
                codes.append(person["code"])
                rows.append(stack)
                owner.append(np.full(len(stack), index, dtype=np.int32))
                slices.append((cursor, cursor + len(stack)))
                cursor += len(stack)
                mean = stack.mean(axis=0)
                centroids.append(mean / (np.linalg.norm(mean) + 1e-9))

            self._names = names
            self._codes = codes
            self._matrix = (np.concatenate(rows, axis=0) if rows else None)
            self._owner = (np.concatenate(owner) if owner else None)
            self._slices = slices
            self._centroids = (np.stack(centroids).astype(np.float32)
                               if centroids else None)
            self._dirty = False

    # -------------------------------------------------------- queries
    def __len__(self):
        with self._lock:
            return len(self._people)

    @property
    def reference_count(self):
        with self._lock:
            return sum(len(p["vectors"]) for p in self._people.values())

    def names(self):
        with self._lock:
            return sorted(self._people)

    def code_for(self, name):
        with self._lock:
            person = self._people.get(name)
            return person["code"] if person else ""

    def per_person_scores(self, embedding, exclude=()):
        """[(name, code, score)] for everybody, best first."""
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        if vector.size == 0 or not np.isfinite(vector).all():
            return []
        vector = vector / (np.linalg.norm(vector) + 1e-9)

        with self._lock:
            self.build()
            if self._matrix is None or vector.size != self._matrix.shape[1]:
                return []
            similarities = self._matrix @ vector
            centroid_similarities = (self._centroids @ vector
                                     if self._centroids is not None else None)
            names, codes, slices = self._names, self._codes, self._slices

        out = []
        for index, (start, stop) in enumerate(slices):
            name = names[index]
            if name in exclude:
                continue
            centroid = (float(centroid_similarities[index])
                        if centroid_similarities is not None else None)
            score = combine(similarities[start:stop],
                            centroid_similarity=centroid)
            if score is None:
                continue
            out.append((name, codes[index], float(score)))
        out.sort(key=lambda row: -row[2])
        return out

    def search(self, embedding, exclude=(), threshold=None, margin=None):
        """Who is this? Returns a BankMatch, accepted or not.

        `exclude` lets a camera rule out people it is already showing
        elsewhere in the same frame - one person cannot be in two places
        at once.
        """
        bar = (FACE_RECOGNITION_THRESHOLD if threshold is None
               else float(threshold))
        lead = FACE_MATCH_MARGIN if margin is None else float(margin)

        ranked = self.per_person_scores(embedding, exclude=exclude)
        if not ranked:
            return BankMatch(reason="nobody enrolled")

        name, code, score = ranked[0]
        runner_up, runner_score = ("", 0.0)
        if len(ranked) > 1:
            runner_up, _rc, runner_score = ranked[1]

        if score < bar:
            return BankMatch("Unknown", "", score, runner_up, runner_score,
                             accepted=False,
                             reason=f"{name} only {score:.2f} < {bar:.2f}")
        if runner_up and (score - runner_score) < lead:
            return BankMatch("Unknown", "", score, runner_up, runner_score,
                             accepted=False,
                             reason=(f"{name} {score:.2f} vs {runner_up} "
                                     f"{runner_score:.2f} - under the {lead} "
                                     f"lead needed"))
        return BankMatch(name, code, score, runner_up, runner_score,
                         accepted=True, reason="accept")

    def score_against(self, name, embedding):
        """How well one embedding matches ONE named person, or None.

        Used by the confirmation guard: a human answer that disagrees
        this strongly with the person's own references is a misclick, not
        a fact, and learning from it would poison that name permanently.
        """
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        with self._lock:
            person = self._people.get(name)
            if person is None or not person["vectors"]:
                return None
            stack = np.stack(person["vectors"])
        if vector.size != stack.shape[1] or not np.isfinite(vector).all():
            return None
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        similarities = stack @ vector
        centroid = stack.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        return combine(similarities,
                       centroid_similarity=float(centroid @ vector))

    def summary(self):
        with self._lock:
            return {"people": len(self._people),
                    "references": sum(len(p["vectors"])
                                      for p in self._people.values()),
                    "model": self.model_tag}


# ---------------------------------------------------------------------
# One face bank shared by every camera in the process.
#
# Deliberately a singleton, for the same reason core/gallery.py is: a
# face confirmed at the reception door must be usable by the workspace
# camera in the same second, not after a restart. core/face.py fills it
# from the enrollment file at startup and core/pipeline.py adds every
# confirmation the dashboard produces.
# ---------------------------------------------------------------------
FACE_BANK = FaceBank()
