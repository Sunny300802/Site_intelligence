"""
core/seats.py
=============
Identify people by WHERE THEY SIT.

Why this exists
---------------
Face recognition cannot work on a work-area camera, and no amount of
tuning changes that. Two reasons, both geometric:

  * heads are 30-55 px wide at this distance, while the recognition
    model needs 112x112 - below about 60 px the embedding is unreliable
  * far more importantly, people face their MONITORS, so the camera sees
    the back or side of most heads. No face is visible at all.

But an office has something a lobby does not: **assigned desks**. If a
person is tracked sitting at desk 12, they are whoever sits at desk 12.
That is close to 100% reliable while they are at their own desk, needs no
face, and costs nothing to compute.

How it is applied
-----------------
A track must SETTLE in a seat before it is credited: it has to stay
inside the zone for a few seconds, so somebody walking past a desk is
never mistaken for its occupant. Face recognition still takes precedence
if it ever does produce a confident match, so the two methods cooperate
rather than compete.
"""
import time

from core.geometry import to_pixels, point_inside


class Seat:
    """One desk position mapped to the person who works there."""

    __slots__ = ("employee", "code", "label", "polygon", "_px")

    def __init__(self, cfg):
        self.employee = cfg.get("employee", "Unknown")
        self.code = str(cfg.get("code", ""))
        self.label = cfg.get("label") or self.employee
        self.polygon = cfg.get("zone") or []
        self._px = None

    def pixels(self, width, height):
        if self._px is None and self.polygon:
            self._px = to_pixels(self.polygon, width, height)
        return self._px

    def contains(self, x, y, width, height):
        px = self.pixels(width, height)
        if px is None:
            return False
        return point_inside(px, x, y)


class SeatMap:
    """All the seats on one camera, plus the dwell logic."""

    def __init__(self, seat_configs, dwell_seconds=6.0):
        self.seats = [Seat(c) for c in (seat_configs or [])]
        self.dwell_seconds = dwell_seconds
        self._settling = {}     # track id -> (seat, since_when)

    def __bool__(self):
        return bool(self.seats)

    @staticmethod
    def anchor(box):
        """Where a person 'is'. Their lower-middle - roughly the chair -
        is far more stable than the box centre when they are half hidden
        behind a desk or monitor."""
        x1, y1, x2, y2 = box
        return (x1 + x2) // 2, int(y1 + 0.85 * (y2 - y1))

    def seat_for(self, box, width, height):
        x, y = self.anchor(box)
        for seat in self.seats:
            if seat.contains(x, y, width, height):
                return seat
        return None

    def resolve(self, track, width, height, now=None):
        """Return the Seat this track has settled into, or None.

        Requires the person to stay put: a passer-by crossing the zone
        resets the timer and is never credited with the desk.
        """
        now = now or time.time()
        seat = self.seat_for(track.box, width, height)
        if seat is None:
            self._settling.pop(track.id, None)
            return None

        current = self._settling.get(track.id)
        if current is None or current[0] is not seat:
            self._settling[track.id] = (seat, now)
            return None

        if now - current[1] >= self.dwell_seconds:
            return seat
        return None

    def forget(self, track_id):
        self._settling.pop(track_id, None)

    def all_pixels(self, width, height):
        return [(s, s.pixels(width, height)) for s in self.seats
                if s.pixels(width, height) is not None]
