"""
core/identity.py
================
Make sure one name belongs to one person at a time.

The problem this solves
-----------------------
Two people were both labelled "Shivajyothi" on screen at the same moment.
That happens because each track decides its own identity independently -
nothing was stopping the same name being handed to two different bodies.

A person cannot be in two places at once, so a name is treated as an
exclusive claim on this camera. When a second track claims a name that is
already taken, the better-scoring claim wins and the loser goes back to
being unidentified. That is honest: we would rather show one correct name
and one "Unknown" than two confident labels where one must be wrong.
"""


class NameRegistry:
    """Tracks which track currently holds each name, on one camera."""

    def __init__(self, steal_margin=0.0):
        self._holder = {}     # name -> (track_id, score)
        # How much better a rival claim must be before it takes a name off
        # the track already holding it.
        #
        # Any improvement at all used to be enough, and that is what made
        # names flicker: two bodies scoring 0.74 and 0.75 traded the name
        # back and forth every few frames, and the loser was blanked to
        # "Unknown" each time. A person watching the dashboard sees a
        # correct name appear and then vanish for no visible reason.
        #
        # Requiring a clear margin means a name only moves when the
        # evidence genuinely says it was on the wrong body.
        self._steal_margin = float(steal_margin)

    def claim(self, name, track_id, score):
        """Try to take a name. Returns (granted, evicted_track_id)."""
        if not name or name == "Unknown":
            return False, None

        current = self._holder.get(name)
        if current is None:
            self._holder[name] = (track_id, score)
            return True, None

        holder_id, holder_score = current
        if holder_id == track_id:
            # same track refreshing its own claim; keep the best score
            if score > holder_score:
                self._holder[name] = (track_id, score)
            return True, None

        if score >= holder_score + self._steal_margin:
            # clearly stronger - take the name from the other track
            self._holder[name] = (track_id, score)
            return True, holder_id

        return False, None      # somebody else has a better claim

    def holder(self, name):
        entry = self._holder.get(name)
        return entry[0] if entry else None

    def release(self, track_id):
        """Give up every name held by a track that has gone."""
        for name, (holder_id, _score) in list(self._holder.items()):
            if holder_id == track_id:
                del self._holder[name]

    def release_name(self, name, track_id):
        entry = self._holder.get(name)
        if entry and entry[0] == track_id:
            del self._holder[name]

    def names_in_use(self):
        return dict(self._holder)
