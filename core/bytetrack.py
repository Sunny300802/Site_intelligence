"""
core/bytetrack.py
=================
ByteTrack association: a Kalman motion model plus a two-pass match.

The idea, and why it is the right one for this site
--------------------------------------------------
Every tracker throws away low-confidence detections. ByteTrack's
observation is that most of those are not noise - they are a REAL person
who is half occluded, motion blurred, or turned away. Discarding them is
what breaks the track, and on this system a broken track is not a
cosmetic problem: the identity, the visit row and the accumulated
presence time all live on the track, so losing it means a second entry
for the same arrival, or an hour of somebody's presence credited to
"Unknown #47".

So detections are used TWICE:

  PASS 1  the confident detections (>= BYTETRACK_HIGH_THRESH) are matched
          against every track, using IoU against each track's Kalman
          PREDICTION rather than its last box. Predicting first is what
          lets a fast walk across a reception still overlap.

  PASS 2  whatever is left - the low-confidence band between
          BYTETRACK_LOW_THRESH and the high threshold - is offered ONLY
          to tracks that pass 1 could not match. This is the pass that
          keeps somebody alive while they are behind a monitor. It can
          never create a new track, so the low band cannot invent people.

  PASS 3  remaining detections are offered to tracks that were only just
          created, before finally starting a new track - and a new track
          needs BYTETRACK_NEW_TRACK_THRESH confidence, which is stricter
          than the detector's own floor.

A track that goes unmatched is not deleted; it is marked LOST and kept
for TRACK_BUFFER frames, still predicting, so it can be re-associated
when the person reappears.

Body size as a tie-breaker only
-------------------------------
The optional body term follows the rule already established in
core/tracker.py: GEOMETRY decides what is possible, SIZE only decides
which of the possible pairings wins. It is never a veto. A person half
behind a door frame measures short and narrow for a few frames, and a
veto on that throws away the only correct match and invents a new,
unnamed person.

This used to compare the COLOUR of people's clothes, which is what let
one person's name follow the wrong body out of an overlap. It now
compares stature and build (core/body.py), which two people who pass
close together do not exchange.

This file is self-contained numpy and scipy. It deliberately does not
depend on lap/lapx/cython_bbox, which are the usual reason a ByteTrack
integration will not install on Windows.
"""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:                                   # pragma: no cover
    _HAVE_SCIPY = False

from config.settings import (TRACK_BUFFER, BYTETRACK_HIGH_THRESH,
                             BYTETRACK_LOW_THRESH, BYTETRACK_NEW_TRACK_THRESH,
                             BYTETRACK_MATCH_THRESH,
                             BYTETRACK_SECOND_MATCH_THRESH)

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


# ---------------------------------------------------------------------
#                            Kalman filter
# ---------------------------------------------------------------------
class KalmanFilter:
    """Constant-velocity filter on (centre x, centre y, aspect, height).

    Height rather than area, and aspect separately, because a person's
    box height is the stable quantity in a fixed camera - it changes
    smoothly with distance - while the width jumps every time an arm
    moves or a shoulder is occluded. Tracking height and letting aspect
    absorb the width noise is what keeps the prediction usable through a
    partial occlusion.

    The noise terms are scaled by the box height, so one set of constants
    works for somebody at the door and somebody at the far desk.
    """

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = np.asarray(measurement, dtype=np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        h = measurement[3]
        std = [2 * self._std_weight_position * h,
               2 * self._std_weight_position * h,
               1e-2,
               2 * self._std_weight_position * h,
               10 * self._std_weight_velocity * h,
               10 * self._std_weight_velocity * h,
               1e-5,
               10 * self._std_weight_velocity * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        h = mean[3]
        std_pos = [self._std_weight_position * h,
                   self._std_weight_position * h,
                   1e-2,
                   self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * h,
                   self._std_weight_velocity * h,
                   1e-5,
                   self._std_weight_velocity * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = (self._motion_mat @ covariance @ self._motion_mat.T
                      + motion_cov)
        return mean, covariance

    def project(self, mean, covariance):
        h = mean[3]
        std = [self._std_weight_position * h,
               self._std_weight_position * h,
               1e-1,
               self._std_weight_position * h]
        innovation_cov = np.diag(np.square(std))
        mean_out = self._update_mat @ mean
        covariance_out = self._update_mat @ covariance @ self._update_mat.T
        return mean_out, covariance_out + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        # Solving is both faster and better conditioned than forming the
        # inverse; a near-singular covariance on a stationary seated
        # person is common enough that this matters in practice.
        try:
            kalman_gain = np.linalg.solve(
                projected_cov.T,
                (covariance @ self._update_mat.T).T).T
        except np.linalg.LinAlgError:
            return mean, covariance
        innovation = np.asarray(measurement, dtype=np.float64) - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


# ---------------------------------------------------------------------
#                          matching helpers
# ---------------------------------------------------------------------
def iou_matrix(a_boxes, b_boxes):
    """IoU between two sets of (x1, y1, x2, y2) boxes -> [len(a), len(b)]."""
    a = np.asarray(a_boxes, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b_boxes, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def assign(cost, threshold):
    """Minimum-cost assignment. Returns (matches, unmatched_a, unmatched_b).

    Hungarian when scipy is available, greedy best-first otherwise. The
    greedy fallback is not as good - it can commit to a pairing that
    blocks two better ones - but it is a great deal better than
    refusing to run, and this project should not fail to start because
    an optional wheel is missing.
    """
    cost = np.asarray(cost, dtype=np.float32)
    if cost.ndim != 2:
        cost = cost.reshape(0, 0)
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        # An empty matrix still has to report which side had entries -
        # returning empty lists for both is how every detection on the
        # very first frame was silently discarded and no track was ever
        # created.
        return [], list(range(rows)), list(range(cols))

    matches = []
    if _HAVE_SCIPY:
        row_index, col_index = linear_sum_assignment(cost)
        for r, c in zip(row_index, col_index):
            if cost[r, c] <= threshold:
                matches.append((int(r), int(c)))
    else:                                             # pragma: no cover
        order = np.dstack(np.unravel_index(np.argsort(cost, axis=None),
                                           cost.shape))[0]
        used_rows, used_cols = set(), set()
        for r, c in order:
            if cost[r, c] > threshold:
                break
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(int(r))
            used_cols.add(int(c))
            matches.append((int(r), int(c)))

    matched_rows = {m[0] for m in matches}
    matched_cols = {m[1] for m in matches}
    unmatched_a = [i for i in range(rows) if i not in matched_rows]
    unmatched_b = [i for i in range(cols) if i not in matched_cols]
    return matches, unmatched_a, unmatched_b


# ---------------------------------------------------------------------
#                              tracks
# ---------------------------------------------------------------------
class STrack:
    """One ByteTrack track: a Kalman state, an id, and a small history.

    Deliberately thin. It knows where somebody IS; it knows nothing about
    who they are, how long they have been present, or which database row
    belongs to them. All of that stays on core/tracker.Track, which this
    feeds - keeping identity out of the motion model is what allowed
    ByteTrack to be added without touching visits, presence or the
    dashboard.
    """

    _shared_kalman = KalmanFilter()

    __slots__ = ("track_id", "mean", "covariance", "state", "is_activated",
                 "score", "frame_id", "start_frame", "tracklet_len",
                 "_tlwh", "descriptor")

    def __init__(self, box, score, descriptor=None):
        self._tlwh = self._to_tlwh(box)
        self.score = float(score)
        self.descriptor = descriptor
        self.track_id = 0
        self.mean = None
        self.covariance = None
        self.state = NEW
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    # ------------------------------------------------------ geometry
    @staticmethod
    def _to_tlwh(box):
        x1, y1, x2, y2 = [float(v) for v in box]
        return np.array([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)],
                        dtype=np.float32)

    @staticmethod
    def _tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret.astype(np.float32)

    @property
    def box(self):
        """(x1, y1, x2, y2), which is what the rest of the project uses."""
        t = self.tlwh
        return (float(t[0]), float(t[1]), float(t[0] + t[2]),
                float(t[1] + t[3]))

    # ------------------------------------------------------ lifecycle
    def activate(self, kalman, frame_id, track_id):
        self.track_id = track_id
        self.mean, self.covariance = kalman.initiate(
            self._tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TRACKED
        # A track is only "activated" (reported to the caller) on its
        # second frame, so a one-frame detector blip never becomes a
        # person. The very first frame of the whole run is exempt,
        # otherwise nothing could ever start.
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, kalman, detection, frame_id, new_id=None):
        self.mean, self.covariance = kalman.update(
            self.mean, self.covariance,
            self._tlwh_to_xyah(detection.tlwh))
        self.tracklet_len = 0
        self.state = TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = detection.score
        if detection.descriptor is not None:
            self.descriptor = detection.descriptor
        if new_id is not None:
            self.track_id = new_id

    def update(self, kalman, detection, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = kalman.update(
            self.mean, self.covariance,
            self._tlwh_to_xyah(detection.tlwh))
        self.state = TRACKED
        self.is_activated = True
        self.score = detection.score
        if detection.descriptor is not None:
            self.descriptor = detection.descriptor

    def predict(self, kalman):
        if self.mean is None:
            return
        mean = self.mean.copy()
        if self.state != TRACKED:
            # A lost track is not walking anywhere we know about, so its
            # height velocity is zeroed. Letting it keep growing or
            # shrinking during an occlusion is how a lost track drifts
            # into a box that then matches the wrong person on return.
            mean[7] = 0
        self.mean, self.covariance = kalman.predict(mean, self.covariance)

    def mark_lost(self):
        self.state = LOST

    def mark_removed(self):
        self.state = REMOVED


# ---------------------------------------------------------------------
#                            the tracker
# ---------------------------------------------------------------------
class ByteTracker:
    """ByteTrack over one camera's person detections.

    update() takes the detections for one frame and returns the live
    tracks. Ids are stable for as long as a person is followed, which is
    the property the whole identity layer is built on.
    """

    def __init__(self, high_thresh=None, low_thresh=None,
                 new_track_thresh=None, match_thresh=None,
                 second_match_thresh=None, track_buffer=None,
                 body_weight=0.0):
        self.high_thresh = (BYTETRACK_HIGH_THRESH if high_thresh is None
                            else float(high_thresh))
        self.low_thresh = (BYTETRACK_LOW_THRESH if low_thresh is None
                           else float(low_thresh))
        self.new_track_thresh = (BYTETRACK_NEW_TRACK_THRESH
                                 if new_track_thresh is None
                                 else float(new_track_thresh))
        self.match_thresh = (BYTETRACK_MATCH_THRESH if match_thresh is None
                             else float(match_thresh))
        self.second_match_thresh = (BYTETRACK_SECOND_MATCH_THRESH
                                    if second_match_thresh is None
                                    else float(second_match_thresh))
        self.buffer_frames = (TRACK_BUFFER if track_buffer is None
                              else int(track_buffer))
        self.body_weight = float(body_weight)

        self.kalman = KalmanFilter()
        self.tracked = []          # currently followed
        self.lost = []             # missing, still within the buffer
        self.frame_id = 0
        self._next_id = 1

        if not _HAVE_SCIPY:
            print("[BYTETRACK] scipy not installed - falling back to greedy "
                  "assignment. Tracking still works; install scipy for the "
                  "optimal one.")

    def _new_id(self):
        self._next_id += 1
        return self._next_id - 1

    # --------------------------------------------------------- costs
    def _cost(self, tracks, detections, use_body):
        """1 - IoU, optionally nudged by how well the sizes agree.

        Being the right size can only ever REDUCE the cost of a pairing
        geometry already considers possible, never raise it above the
        threshold, so it decides ties without being able to veto a match.
        """
        cost = 1.0 - iou_matrix([t.box for t in tracks],
                                [d.box for d in detections])
        if not use_body or self.body_weight <= 0 or cost.size == 0:
            return cost

        from core.body import similarity as body_similarity
        bonus = np.zeros_like(cost)
        for i, track in enumerate(tracks):
            if track.descriptor is None:
                continue
            for j, detection in enumerate(detections):
                if detection.descriptor is None:
                    continue
                fits = body_similarity(track.descriptor, detection.descriptor)
                if fits is not None:
                    bonus[i, j] = self.body_weight * float(fits)
        return np.maximum(0.0, cost - bonus)

    # -------------------------------------------------------- update
    def update(self, boxes, scores=None, descriptors=None):
        """One frame of detections -> the live tracks.

        `boxes`       [(x1, y1, x2, y2), ...] in frame coordinates
        `scores`      detector confidences, or None to treat all as high
        `descriptors` optional body measurements (core/body.py),
                      aligned with boxes
        """
        self.frame_id += 1
        boxes = list(boxes or [])
        if scores is None:
            scores = [1.0] * len(boxes)
        scores = [float(s) for s in scores]
        if descriptors is None:
            descriptors = [None] * len(boxes)

        high, low = [], []
        for box, score, descriptor in zip(boxes, scores, descriptors):
            detection = STrack(box, score, descriptor)
            if score >= self.high_thresh:
                high.append(detection)
            elif score >= self.low_thresh:
                low.append(detection)

        # ---- predict every existing track forward one frame ----------
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = confirmed + self.lost
        for track in pool + unconfirmed:
            track.predict(self.kalman)

        activated, refound, newly_lost, removed = [], [], [], []

        # ---- PASS 1: confident detections against everything ---------
        cost = self._cost(pool, high, use_body=True)
        matches, unmatched_tracks, unmatched_high = assign(
            cost, self.match_thresh)
        for track_index, detection_index in matches:
            track = pool[track_index]
            detection = high[detection_index]
            if track.state == TRACKED:
                track.update(self.kalman, detection, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(self.kalman, detection, self.frame_id)
                refound.append(track)

        # ---- PASS 2: the low band, only for what pass 1 missed -------
        # This is the pass that keeps a half-occluded person alive. It
        # cannot start anything new, so a low-confidence false positive
        # can never become a person - it can only, at worst, briefly
        # steer an existing track.
        still_tracked = [pool[i] for i in unmatched_tracks
                         if pool[i].state == TRACKED]
        cost = self._cost(still_tracked, low, use_body=False)
        matches, unmatched_second, _unmatched_low = assign(
            cost, self.second_match_thresh)
        for track_index, detection_index in matches:
            track = still_tracked[track_index]
            detection = low[detection_index]
            if track.state == TRACKED:
                track.update(self.kalman, detection, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(self.kalman, detection, self.frame_id)
                refound.append(track)

        # Everything the two passes could not account for is now LOST -
        # not deleted. It keeps predicting for TRACK_BUFFER frames so the
        # person can be picked up again when they come back into view.
        for index in unmatched_second:
            track = still_tracked[index]
            if track.state != LOST:
                track.mark_lost()
                newly_lost.append(track)

        # ---- PASS 3: unconfirmed tracks, then genuinely new people ---
        remaining = [high[i] for i in unmatched_high]
        cost = self._cost(unconfirmed, remaining, use_body=True)
        matches, unmatched_unconfirmed, unmatched_remaining = assign(cost, 0.7)
        for track_index, detection_index in matches:
            track = unconfirmed[track_index]
            track.update(self.kalman, remaining[detection_index], self.frame_id)
            activated.append(track)
        for index in unmatched_unconfirmed:
            track = unconfirmed[index]
            track.mark_removed()
            removed.append(track)

        for index in unmatched_remaining:
            detection = remaining[index]
            if detection.score < self.new_track_thresh:
                continue
            detection.activate(self.kalman, self.frame_id, self._new_id())
            activated.append(detection)

        # ---- retire tracks that have been lost for too long ----------
        for track in self.lost:
            if self.frame_id - track.frame_id > self.buffer_frames:
                track.mark_removed()
                removed.append(track)

        # ---- rebuild the pools --------------------------------------
        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _join(self.tracked, activated)
        self.tracked = _join(self.tracked, refound)
        self.lost = _without(self.lost, self.tracked)
        self.lost.extend(newly_lost)
        self.lost = _without(self.lost, removed)
        self.tracked, self.lost = _drop_duplicates(self.tracked, self.lost)

        return [t for t in self.tracked if t.is_activated], removed

    def reset(self):
        self.tracked, self.lost = [], []
        self.frame_id = 0
        self._next_id = 1


def _join(primary, extra):
    """Union of two track lists, keeping order and dropping duplicates."""
    seen = {id(t) for t in primary}
    out = list(primary)
    for track in extra:
        if id(track) not in seen:
            seen.add(id(track))
            out.append(track)
    return out


def _without(tracks, remove):
    drop = {id(t) for t in remove}
    return [t for t in tracks if id(t) not in drop]


def _drop_duplicates(tracked, lost, overlap=0.85):
    """Resolve a tracked and a lost track sitting on the same person.

    This happens after a long occlusion: the lost track is re-associated
    at almost the same moment a new one was started for the same body.
    Keeping both means two ids on one person - and, downstream, two
    visits and two presence sessions. The one that has existed longer
    wins, because it is the one carrying the identity and the database
    row.
    """
    if not tracked or not lost:
        return tracked, lost
    overlaps = iou_matrix([t.box for t in tracked], [t.box for t in lost])
    pairs = np.where(overlaps > overlap)
    drop_tracked, drop_lost = set(), set()
    for i, j in zip(*pairs):
        age_tracked = tracked[i].frame_id - tracked[i].start_frame
        age_lost = lost[j].frame_id - lost[j].start_frame
        if age_tracked > age_lost:
            drop_lost.add(int(j))
        else:
            drop_tracked.add(int(i))
    return ([t for i, t in enumerate(tracked) if i not in drop_tracked],
            [t for j, t in enumerate(lost) if j not in drop_lost])
