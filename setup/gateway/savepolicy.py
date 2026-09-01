#!/usr/bin/env python3
"""savepolicy — WHEN a prefix is written to disk, as a decision and not a reflex.

    p = Policy(min_sightings=2, debounce_s=10, max_defers=3)
    p.saw(t, "abc123")            # a request for this prefix arrived
    p.idle_since(t)               # nothing is in flight since t
    p.due(now)                    # -> ids that should be written now

Pure: no clock, no disk, no network. Every time is passed in. That is what
lets bench/suites/save-policy-sim.py replay seven days of real traffic through
it in milliseconds, and what lets the tests state a rule instead of a timing.

WHY THERE IS A POLICY AT ALL
----------------------------
The gateway wrote a prefix on its FIRST cold appearance, in a background task,
while the machine was at its busiest — the answer had just gone out and the
next turn was, measured on this stack, a median of 1.0 s away. With one slot,
saving means putting the prefix INTO that slot, so the save and the next turn
fight over it. Both defects of 28.08.2026 come out of that fight:

    autosave-evicts-the-working-slot     the turn loses: 0.7 s -> 13.6 s
    saved-prefix-holds-a-foreign-state   the save loses: the file gets the
                                         turn's prefix under its own name

THREE RULES, AND EACH ONE ANSWERS A MEASUREMENT

`min_sightings` — write what has PROVEN itself, not what has been seen. Of the
four real files in the store on 28.08., two had been used exactly once: 1.4 GB
written, a collision window opened, and nothing gained, because the benefit of
a saved prefix only ever materialises after a server restart and a one-off
project is not there to collect it.

`debounce_s` — write in a gap, not inside a tool loop. Claude Code does not
wait for a human between turns: 63 % of this consumer's follow-ups arrive
within 2 seconds. Waiting for a quiet stretch moves the write out of the
burst, and the same measurement says a quiet stretch does come: a quarter of
the gaps are longer than 30 s.

`max_defers` — but a rule that only writes when quiet writes nothing on a busy
day, and a prefix that is never written is a guaranteed cold start later. So
deferral is counted, and after `max_defers` the write happens anyway.

The ledger of sightings is PERSISTENT on purpose (the caller owns the file;
this module only takes a dict). A counter that lives in the gateway process
forgets everything on restart — and then a prefix used once per session, every
day, is never written at all: it is always on its first sighting. That was the
hole in the first version of this rule.
"""

# What a caller should hand `saw()` for a prefix worth remembering at all. Here
# so the number has one home; the gateway's AUTO_MIN_CHARS decides what reaches
# this module.
DEFAULT_MIN_SIGHTINGS = 2
DEFAULT_DEBOUNCE_S = 10.0
DEFAULT_MAX_DEFERS = 3


def stale_rivals(now, activity, grace_s):
    """Split same-head rivals into (evict, keep) for a replacement decision.

    `activity` maps rival id -> when a request for that saved prefix last
    arrived (epoch seconds), or None where nothing is recorded. None reads as
    idle: the worst a wrong eviction costs is one cold start before the
    prefix earns its file back, while a wrong protection costs a cold start
    on EVERY session of the successor — the 80,721-token starts of
    01.09.2026 were that side of the asymmetry.

    A rival asked for within `grace_s` proves the incumbent is still in
    service; the newcomer is then the churn and nothing should be written.
    Without one, the incumbent is the leftover of a drifted tool set and
    gives way. The boundary counts as expired. Both lists come back sorted
    so logs and traces read stably.
    """
    evict, keep = [], []
    for id_, t in activity.items():
        (keep if t is not None and (now - t) < grace_s else evict).append(id_)
    return sorted(evict), sorted(keep)


class Policy:
    """The decision, held as state that a caller can persist and restore.

    `ledger` maps prefix id -> how often it has been seen, ACROSS restarts. It
    is passed in rather than built, so the caller can load it from disk and
    write it back; this module never touches a file.
    """

    def __init__(self, min_sightings=DEFAULT_MIN_SIGHTINGS,
                 debounce_s=DEFAULT_DEBOUNCE_S,
                 max_defers=DEFAULT_MAX_DEFERS,
                 ledger=None, saved=None):
        self.min_sightings = min_sightings
        self.debounce_s = debounce_s
        self.max_defers = max_defers
        self.ledger = dict(ledger or {})
        self.saved = set(saved or ())
        self.defers = {}
        self._quiet_since = None
        self._busy = False

    # --- what the world tells it ------------------------------------------
    def saw(self, t, id_):
        """A request for this prefix arrived. Returns the new sighting count."""
        self.ledger[id_] = self.ledger.get(id_, 0) + 1
        self._busy = True
        self._quiet_since = None
        return self.ledger[id_]

    def busy(self, t):
        """Something is in flight. A save must not start now."""
        self._busy = True
        self._quiet_since = None

    def idle_since(self, t):
        """Nothing is in flight, and has not been since `t`."""
        if not self._busy and self._quiet_since is not None:
            return                      # already idle; keep the earlier mark
        self._busy = False
        self._quiet_since = t

    def note_saved(self, id_):
        self.saved.add(id_)
        self.defers.pop(id_, None)

    def note_deferred(self, id_):
        """Counted, because deferring forever is the same as never saving."""
        self.defers[id_] = self.defers.get(id_, 0) + 1
        return self.defers[id_]

    # --- what it decides ---------------------------------------------------
    def candidates(self):
        """Prefixes that have earned a place on disk but do not have one."""
        return [i for i, n in self.ledger.items()
                if i not in self.saved and n >= self.min_sightings]

    def quiet_for(self, now):
        if self._busy or self._quiet_since is None:
            return 0.0
        return max(0.0, now - self._quiet_since)

    def due(self, now):
        """Which prefixes to write RIGHT NOW, most-seen first.

        Two ways to become due, and the second is the important one: quiet
        long enough, or deferred often enough that waiting for quiet has
        stopped being a plan.
        """
        quiet = self.quiet_for(now)
        out = []
        for i in self.candidates():
            if quiet >= self.debounce_s or self.defers.get(i, 0) >= self.max_defers:
                out.append(i)
        out.sort(key=lambda i: (-self.ledger[i], i))
        return out

    def forced(self, id_):
        """Was this one written because it ran out of patience rather than
        because the machine went quiet? The caller logs the difference: a
        forced save is the one that can make a turn wait."""
        return self.defers.get(id_, 0) >= self.max_defers
