"""probe — the watchdog that turns a silent fault into a loud one.

The two dangerous defects here end the same way: the server keeps answering
and every answer degenerates to '////' until it restarts. Nothing raises. The
probe exists to notice that, and it is only worth having if it is wrong in the
safe direction — a watchdog that cries wolf is one that gets switched off, and
then the real fault arrives to an empty room.

So the tests are mostly about what must NOT trip it.
"""
import os, sys, unittest

import urllib.error

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "setup" / "scripts"))
import probe                                                  # noqa: E402


class TestDegeneracy(unittest.TestCase):
    def test_the_actual_signature(self):
        bad, why = probe.looks_degenerate("/" * 200)
        self.assertTrue(bad)
        self.assertIn("100%", why)

    def test_the_signature_mixed_with_a_little_text(self):
        self.assertTrue(probe.looks_degenerate("The answer is " + "/" * 300)[0])

    def test_a_correct_short_answer_is_not_degenerate(self):
        """391 is three characters. Below min_len nothing is judged, because a
        short answer cannot carry the evidence."""
        self.assertFalse(probe.looks_degenerate("391")[0])

    def test_prose_with_a_markdown_rule_is_not_degenerate(self):
        """A run test would fire here. The ratio test must not: the rule is
        long but the answer around it is ordinary."""
        text = ("Here is the summary of the change.\n"
                + "-" * 72 + "\n"
                "It rewrites the loader and keeps the old path for one release.")
        self.assertFalse(probe.looks_degenerate(text)[0])

    def test_an_answer_made_of_whitespace_is_not_this_fault(self):
        self.assertFalse(probe.looks_degenerate(" \n" * 200)[0])

    def test_ordinary_english_is_not_degenerate(self):
        self.assertFalse(probe.looks_degenerate(
            "The product of seventeen and twenty-three is three hundred "
            "and ninety-one, which you can check by expanding.")[0])


class TestJudge(unittest.TestCase):
    def test_a_good_answer_passes(self):
        ok, verdict, _ = probe.judge("391")
        self.assertTrue(ok)
        self.assertEqual(verdict, "ok")

    def test_degenerate_outranks_wrong(self):
        """Both are failures, but only one names a cause. Reporting '////' as
        WRONG would file a known hardware fault under 'the model erred'."""
        ok, verdict, _ = probe.judge("/" * 200)
        self.assertFalse(ok)
        self.assertEqual(verdict, "DEGENERATE")

    def test_a_merely_wrong_answer_is_wrong(self):
        ok, verdict, _ = probe.judge("The answer is 392, I believe, roughly.")
        self.assertFalse(ok)
        self.assertEqual(verdict, "WRONG")


class TestPatienceWithARestartingServer(unittest.TestCase):
    """A refused connection is DOWN, and that is not this watchdog's fault.

    It exists for the SILENT one: a server that answers, and answers wrongly.
    Downtime already has three detectors — systemd, check.sh and the gateway —
    and conflating them made the probe cry wolf every time production was
    restarted. Measured 27.08.: sideserver restarted production and this timer
    in the same second at 10:55:46, the timer's interval had elapsed while it
    was stopped so it fired at 10:55:47, and the model finished loading at
    10:55:55. Two runs, two false alarms, two red lines in check.sh, nothing
    wrong either time. A detector that cries wolf is one people stop reading.
    """

    def setUp(self):
        self._ask = probe.ask
        self.slept = []

    def tearDown(self):
        probe.ask = self._ask

    def test_a_server_that_comes_up_late_is_not_a_failure(self):
        calls = []

        def flaky(url, timeout=180):
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionRefusedError(111, "Connection refused")
            return "the answer"
        probe.ask = flaky
        text, err = probe.ask_with_patience("x", grace=90, sleep=self.slept.append)
        self.assertEqual(text, "the answer")
        self.assertIsNone(err)
        self.assertEqual(len(calls), 3)

    def test_a_server_that_never_comes_up_still_fails(self):
        """The patience must not turn a dead server into a green light."""
        probe.ask = lambda url, timeout=180: (_ for _ in ()).throw(
            ConnectionRefusedError(111, "Connection refused"))
        text, err = probe.ask_with_patience("x", grace=0, sleep=self.slept.append)
        self.assertIsNone(text)
        # "not ready" rather than "refusing" since 27.08.: the window now
        # covers a 503 as well, and one message has to describe both. Since
        # 29.08. it names the ELAPSED time instead — "still not ready after
        # 90s" was printed after 180 s had passed, and the phrase also claimed
        # a server that was merely busy had never come up. What the test is
        # for is unchanged: the reason has to survive into the message.
        self.assertIn("gave up after", err)
        self.assertIn("Connection refused", err,
                      "the reason must survive into the message, or a dead "
                      "server and a slow one read the same")

    def test_a_server_that_ANSWERS_is_judged_at_once(self):
        """The whole point. A poisoned server answers — retrying it would let
        it look healthy for another minute, which is the one thing this
        watchdog may not do."""
        calls = []

        def http_500(url, timeout=180):
            calls.append(1)
            raise urllib.error.HTTPError("u", 500, "boom", None, None)
        probe.ask = http_500
        text, err = probe.ask_with_patience("x", grace=90, sleep=self.slept.append)
        self.assertIsNone(text)
        self.assertEqual(len(calls), 1, "an answered request must not be retried")
        self.assertEqual(self.slept, [])

    def test_a_503_is_not_ready_rather_than_a_verdict(self):
        """llama-server returns 503 while the weights load and the gateway
        passes it through. Treating it as an answer made the patience cover
        only half the case it exists for: measured 27.08. at 23:19:06 and
        23:42:36, two red lines in check.sh from a probe that fired into a
        model still coming up after a measurement restored production."""
        calls = []

        def slow_then_up(url, timeout=180):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.HTTPError("u", 503, "Service Unavailable",
                                             None, None)
            return "391"
        probe.ask = slow_then_up
        text, err = probe.ask_with_patience("x", grace=90,
                                            sleep=self.slept.append)
        self.assertEqual(text, "391")
        self.assertIsNone(err)
        self.assertEqual(len(calls), 3)

    def test_a_503_that_persists_still_fails(self):
        """The patience must not turn a server that never becomes ready into
        a green light — the same property the connection case already has."""
        probe.ask = lambda url, timeout=180: (_ for _ in ()).throw(
            urllib.error.HTTPError("u", 503, "Service Unavailable", None, None))
        text, err = probe.ask_with_patience("x", grace=0,
                                            sleep=self.slept.append)
        self.assertIsNone(text)
        self.assertIn("503", err)
        self.assertIn("gave up after", err)

    def test_every_other_status_is_still_judged_at_once(self):
        """The positive control for the exemption: 503 must be the only one.
        A 500 from a server that is up is a real fault and retrying it would
        hide it for another minute."""
        for code in (400, 404, 500, 502, 504):
            calls = []

            def answered(url, timeout=180, code=code, calls=calls):
                calls.append(1)
                raise urllib.error.HTTPError("u", code, "x", None, None)
            probe.ask = answered
            text, err = probe.ask_with_patience("x", grace=90,
                                                sleep=self.slept.append)
            self.assertIsNone(text)
            self.assertEqual(len(calls), 1, "status %d was retried" % code)

    def test_the_grace_covers_a_big_model_coming_up(self):
        """16.7 GiB took nine seconds on 27.08.; the window has to hold more
        than that without holding up the ten-minute cadence."""
        self.assertGreaterEqual(probe.GRACE_S, 60)
        self.assertLess(probe.GRACE_S, 300)


class TestSideserverRestoresTheWatchdogLast(unittest.TestCase):
    def src(self):
        return (common.REPO / "bench" / "sideserver.py").read_text(encoding="utf-8")

    def test_production_is_waited_for_before_the_timer_returns(self):
        src = self.src()
        i = src.index('systemctl("start", a.stop)')
        j = src.index('systemctl("start", PROBE_TIMER)')
        self.assertLess(i, j, "the timer must go back after production")
        self.assertIn("wait_for_slots(PRODUCTION_URL", src[i:j],
                      "starting them together is what caused the false alarm")


class TestBusyIsNotDown(unittest.TestCase):
    """A server working on somebody else's request is not an outage.

    Production runs ONE slot -- the mitigation for gfx1151-two-slots -- so
    every request queues behind the one in flight, the probe included.
    Measured 29.08. over 277 runs: median 1 s, but 12 runs took over a
    minute and the longest that still answered took 160 s. The read timeout
    in ask() is 180 s. The three UNREACHABLE verdicts on record are not a
    different kind of event from the 160 s success -- they are the same
    queue, one request longer.

    Measured the same day against production with the slot deliberately
    occupied for 112 s: /health answered 200 in 0.001 s on 11 of 11 samples,
    /slots reported is_processing=True throughout, and a probe-shaped request
    queued behind it came back correct after 109 s. So `busy` and `down` are
    distinguishable without waiting the timeout out at all.
    """

    def setUp(self):
        import tempfile
        self.path = os.path.join(tempfile.mkdtemp(), "streak")

    def state(self, alive, processing, detail="1 of 1 slots busy"):
        return lambda url, timeout=5: (alive, processing, detail)

    def classify(self, alive, processing, err="TimeoutError('timed out')"):
        return probe.classify_stall(
            "u", err, streak_path=self.path,
            state=self.state(alive, processing))

    def test_a_busy_server_is_not_unreachable(self):
        verdict, ok, _ = self.classify(True, True)
        self.assertEqual(verdict, "BUSY")
        self.assertTrue(ok, "a working server must not be a systemd failure")

    def test_a_server_that_is_gone_is_still_unreachable(self):
        """The patience must keep catching a real outage."""
        verdict, ok, _ = self.classify(False, None)
        self.assertEqual(verdict, "UNREACHABLE")
        self.assertFalse(ok)

    def test_a_server_that_answers_health_but_computes_nothing_is_a_stall(self):
        """The case BUSY must not swallow: /health fine, no slot working, and
        a chat request that still timed out. Nothing explains that, so it
        stays a failure rather than being filed as ordinary load."""
        verdict, ok, _ = self.classify(True, False)
        self.assertEqual(verdict, "STALLED")
        self.assertFalse(ok)

    def test_a_status_code_is_an_answer_and_keeps_its_old_verdict(self):
        """Only a request that never came back can be explained by the queue.
        A 500 IS an answer, and must not be re-filed as load."""
        verdict, ok, _ = self.classify(
            True, True, err="HTTPError('u', 500, 'boom', None, None)")
        self.assertEqual(verdict, "UNREACHABLE")
        self.assertFalse(ok)

    def test_the_reason_survives_into_the_detail(self):
        _, _, detail = self.classify(True, True)
        self.assertIn("timed out", detail)
        self.assertIn("slot", detail)

    def test_slots_being_unreadable_is_not_a_stall(self):
        """/health answered, so the server is there. Whether it computes is
        then UNKNOWN — and unknown must not be reported as STALLED, which is
        a finding. --no-slots alone would otherwise manufacture one on every
        busy minute."""
        verdict, ok, _ = self.classify(True, None)
        self.assertEqual(verdict, "UNKNOWN")
        self.assertTrue(ok)

    def test_unknown_still_counts_as_a_round_without_a_look(self):
        """It is not a failure, but it is not a look at the model either —
        so it must reach BLIND just as BUSY does, or --no-slots would make
        the watchdog quietly permanent."""
        for _ in range(probe.BUSY_LIMIT):
            probe.busy_streak(self.path, True)
        verdict, ok, _ = self.classify(True, None)
        self.assertEqual(verdict, "BLIND")
        self.assertFalse(ok)


class TestBlindStreak(unittest.TestCase):
    """BUSY is not a failure -- but an unbroken run of it is.

    A watchdog that never gets a turn reports nothing and looks exactly like
    a watchdog that keeps finding everything in order. That is the silent
    failure mode of the silent-failure detector, so the streak is counted.
    """

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "busy-streak")

    def test_one_busy_run_is_fine(self):
        self.assertEqual(probe.busy_streak(self.path, True), 1)

    def test_the_streak_accumulates(self):
        for expected in (1, 2, 3):
            self.assertEqual(probe.busy_streak(self.path, True), expected)

    def test_any_real_verdict_clears_it(self):
        probe.busy_streak(self.path, True)
        probe.busy_streak(self.path, True)
        self.assertEqual(probe.busy_streak(self.path, False), 0)
        self.assertEqual(probe.busy_streak(self.path, True), 1)

    def test_a_long_streak_becomes_a_failure(self):
        for _ in range(probe.BUSY_LIMIT):
            probe.busy_streak(self.path, True)
        verdict, ok, detail = probe.classify_stall(
            "u", "TimeoutError('timed out')", streak_path=self.path,
            state=lambda url, timeout=5: (True, True, "1 of 1 slots busy"))
        self.assertEqual(verdict, "BLIND")
        self.assertFalse(ok, "not seeing the server for %d runs in a row is "
                             "not a green light" % probe.BUSY_LIMIT)
        self.assertIn(str(probe.BUSY_LIMIT + 1), detail)

    def test_an_unreadable_counter_does_not_take_the_probe_down(self):
        """State on disk is a convenience; the watchdog must survive losing
        it."""
        self.assertEqual(probe.busy_streak("/proc/nonexistent/x", True), 1)


class TestTheMessageNamesTheTimeActuallyWaited(unittest.TestCase):
    """It said `still not ready after 90s` while 180 s had passed.

    90 is the connect-patience; the read timeout in ask() is 180. Reading
    that line sends whoever investigates looking at the wrong number --
    measured 29.08., it did exactly that.
    """

    def setUp(self):
        self._ask = probe.ask

    def tearDown(self):
        probe.ask = self._ask

    def test_it_reports_the_elapsed_time(self):
        clock = [0.0]

        def slow_fail(url, timeout=180):
            clock[0] += 180.0
            raise TimeoutError("timed out")
        probe.ask = slow_fail
        _, err = probe.ask_with_patience("x", grace=90, sleep=lambda s: None,
                                         now=lambda: clock[0])
        self.assertIn("180", err, "the message must name the time that passed")
        self.assertNotIn("after 90s", err,
                         "90 is the connect window, not what was waited")


class TestTheAnswerTimeoutIsAKnob(unittest.TestCase):
    """180 s lived as a default argument on ask() and nowhere else.

    It is the number that actually decides when a queued probe gives up --
    the 90 s grace never gets a say once a connection is accepted -- so it
    belongs beside GRACE_S, and a watchdog whose deadline cannot be set
    cannot be exercised against a real busy server either.
    """

    def setUp(self):
        self._ask = probe.ask

    def tearDown(self):
        probe.ask = self._ask

    def test_the_constant_exists_and_is_the_measured_one(self):
        self.assertEqual(probe.ANSWER_TIMEOUT_S, 180)

    def test_it_reaches_ask(self):
        seen = []

        def spy(url, timeout=probe.ANSWER_TIMEOUT_S):
            seen.append(timeout)
            return "391"
        probe.ask = spy
        probe.ask_with_patience("u", grace=0, sleep=lambda s: None, timeout=7)
        self.assertEqual(seen, [7])


class TestCheckShDoesNotCallASkippedRoundAPass(unittest.TestCase):
    """`ok` and `BUSY` both exit 0, and they mean opposite things.

    BUSY is not a failure — the server was working on somebody else's
    request — but it is not a look at the model either. check.sh read
    ExecMainStatus alone, so a run that checked NOTHING printed `last probe
    passed`. That is the silent-failure detector failing silently, which is
    the one shape this whole file exists to prevent.
    """

    def src(self):
        return (common.REPO / "setup" / "check.sh").read_text(encoding="utf-8")

    def test_it_reads_the_verdict_and_not_only_the_exit_code(self):
        src = self.src()
        i = src.index("llama-probe.timer enabled")
        j = src.index("Memory budget", i)
        section = src[i:j]
        self.assertIn("BUSY", section,
                      "check.sh must tell a skipped round from a passed one")
        self.assertIn("journalctl", section,
                      "the verdict lives in the journal; the exit code cannot "
                      "carry it")

    def test_the_verdicts_that_mean_no_look_are_all_covered(self):
        src = self.src()
        i = src.index("llama-probe.timer enabled")
        section = src[i:src.index("Memory budget", i)]
        for verdict in ("BUSY", "UNKNOWN"):
            self.assertIn(verdict, section,
                          "%s exits 0 and checked nothing" % verdict)


if __name__ == "__main__":
    unittest.main()
