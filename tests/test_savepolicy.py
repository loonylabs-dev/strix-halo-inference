"""savepolicy — the rule that decides WHEN a prefix is written to disk.

Pure logic, so these tests state rules rather than timings. What they protect
is not the numbers — those come from bench/suites/save-policy-sim.py against
real traffic — but the properties the numbers are meaningless without: a
deferral that is counted, a candidate that has to earn its place, and a ledger
that survives the process it was built in.
"""
import unittest

import common

P = common.load("setup/gateway/savepolicy.py", "savepolicy")


class TestWhatEarnsAPlaceOnDisk(unittest.TestCase):
    def test_one_sighting_is_not_enough_when_two_are_asked_for(self):
        p = P.Policy(min_sightings=2)
        p.saw(0, "a")
        self.assertEqual(p.candidates(), [])
        p.saw(10, "a")
        self.assertEqual(p.candidates(), ["a"])

    def test_a_prefix_already_on_disk_is_never_a_candidate(self):
        p = P.Policy(min_sightings=1, saved={"a"})
        p.saw(0, "a")
        self.assertEqual(p.candidates(), [])

    def test_the_ledger_is_handed_in_so_it_can_outlive_the_process(self):
        """The hole in the first version of this rule: a counter that lives in
        the gateway forgets everything on restart, and a prefix used once per
        session — every day — is then always on its first sighting and never
        written at all."""
        p = P.Policy(min_sightings=2, ledger={"a": 1})
        p.saw(0, "a")
        self.assertEqual(p.candidates(), ["a"],
                         "a sighting from a previous life must still count")

    def test_the_most_seen_goes_first(self):
        p = P.Policy(min_sightings=1)
        for _ in range(3):
            p.saw(0, "often")
        p.saw(0, "once")
        p.idle_since(1)          # busy is busy: nothing is due until it is not
        self.assertEqual(p.due(1e6)[0], "often")


class TestWhenItWrites(unittest.TestCase):
    def test_nothing_is_due_while_the_machine_is_busy(self):
        """Writing means putting the prefix into the one slot. Doing that
        while a request is in flight is the collision both defects of
        28.08.2026 come out of."""
        p = P.Policy(min_sightings=1, debounce_s=10)
        p.saw(0, "a")
        self.assertEqual(p.due(0), [])
        p.busy(5)
        self.assertEqual(p.due(100), [], "busy must not become due by waiting")

    def test_a_gap_shorter_than_the_debounce_is_not_a_gap(self):
        p = P.Policy(min_sightings=1, debounce_s=10)
        p.saw(0, "a")
        p.idle_since(1)
        self.assertEqual(p.due(6), [], "5 s of quiet is a pause, not a gap")
        self.assertEqual(p.due(11), ["a"])

    def test_the_first_idle_mark_is_the_one_that_counts(self):
        """Two idle notices in a row must not restart the clock — otherwise a
        caller that reports idleness often can never accumulate quiet."""
        p = P.Policy(min_sightings=1, debounce_s=10)
        p.saw(0, "a")
        p.idle_since(1)
        p.idle_since(5)
        self.assertEqual(p.due(11), ["a"])

    def test_a_request_resets_the_quiet(self):
        p = P.Policy(min_sightings=1, debounce_s=10)
        p.saw(0, "a")
        p.idle_since(1)
        p.saw(9, "a")
        self.assertEqual(p.due(12), [], "the burst started again")


class TestPatienceRunsOut(unittest.TestCase):
    """A rule that only writes when quiet writes nothing on a busy day — and a
    prefix that is never written is a guaranteed cold start later."""

    def test_enough_deferrals_make_it_due_without_a_gap(self):
        p = P.Policy(min_sightings=1, debounce_s=10, max_defers=3)
        p.saw(0, "a")
        p.idle_since(1)
        for _ in range(3):
            self.assertEqual(p.due(2), [], "not yet: 1 s of quiet")
            p.note_deferred("a")
        self.assertEqual(p.due(2), ["a"], "after three deferrals it happens anyway")

    def test_a_forced_write_says_so(self):
        """The caller logs the difference, because a forced write is the one
        that can make a turn wait."""
        p = P.Policy(min_sightings=1, debounce_s=10, max_defers=2)
        p.saw(0, "a")
        self.assertFalse(p.forced("a"))
        p.note_deferred("a"); p.note_deferred("a")
        self.assertTrue(p.forced("a"))

    def test_saving_clears_the_deferrals(self):
        p = P.Policy(min_sightings=1, max_defers=2)
        p.saw(0, "a"); p.note_deferred("a")
        p.note_saved("a")
        self.assertEqual(p.candidates(), [])
        self.assertFalse(p.forced("a"))


class TestAStaleRivalGivesWayToItsSuccessor(unittest.TestCase):
    """The 31.08. rule refused to save a prefix whose head was already on
    disk, written against one session whose tool list churned three times.
    Four days of traces (29.08.-01.09.) say the real pattern is DRIFT: tool
    sets went 6 -> 13 -> 21 -> 15 -> 19 -> 20 -> 52 -> 64 -> 87 across
    sessions and a superseded set never returned. Under drift the refusal
    keeps the dead file and makes every new session start cold — 80,721
    tokens, 577 s, measured 01.09. 15:39. So the rule turns around: a rival
    nobody asked for within the grace gives way; one recently in service
    still protects its place, which is what stops thrash if a client ever
    does oscillate between two sets."""

    def test_an_idle_rival_is_evicted(self):
        evict, keep = P.stale_rivals(
            now=200_000.0, activity={"old": 10_000.0}, grace_s=86_400.0)
        self.assertEqual(evict, ["old"])
        self.assertEqual(keep, [])

    def test_a_rival_in_service_protects_its_place(self):
        evict, keep = P.stale_rivals(
            now=200_000.0, activity={"live": 199_000.0}, grace_s=86_400.0)
        self.assertEqual(evict, [])
        self.assertEqual(keep, ["live"])

    def test_unknown_activity_reads_as_idle(self):
        """A sidecar without `last_used` (one measured absence, 01.09.2026)
        belongs to a prefix nobody has asked for since it was written. The
        worst a wrong eviction costs is one cold start before the prefix
        earns its file back; the wrong protection costs a cold start on
        every session of its successor."""
        evict, keep = P.stale_rivals(
            now=200_000.0, activity={"ghost": None}, grace_s=86_400.0)
        self.assertEqual(evict, ["ghost"])
        self.assertEqual(keep, [])

    def test_a_mixed_field_splits_and_sorts(self):
        evict, keep = P.stale_rivals(
            now=200_000.0,
            activity={"b": None, "a": 10_000.0, "c": 199_999.0},
            grace_s=86_400.0)
        self.assertEqual(evict, ["a", "b"])
        self.assertEqual(keep, ["c"])

    def test_exactly_at_the_grace_is_idle(self):
        """The boundary reads as expired, not as in service — a rule, stated
        so the comparison operator cannot drift silently."""
        evict, keep = P.stale_rivals(
            now=100_000.0, activity={"edge": 13_600.0}, grace_s=86_400.0)
        self.assertEqual(evict, ["edge"])
        self.assertEqual(keep, [])


class TestItCannotDecideByItself(unittest.TestCase):
    """No clock, no disk, no network — every time comes from the caller. That
    is what lets seven days of real traffic run through it in milliseconds,
    and what keeps these tests from being about timing."""

    def test_the_module_touches_nothing(self):
        src = (common.REPO / "setup" / "gateway" / "savepolicy.py").read_text(
            encoding="utf-8")
        for forbidden in ("import os", "import time", "open(", "requests",
                          "urllib", "subprocess"):
            self.assertNotIn(forbidden, src,
                             "a pure decision must not reach out to %s" % forbidden)


if __name__ == "__main__":
    unittest.main()
