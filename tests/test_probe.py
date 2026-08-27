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
        self.assertIn("still refusing", err)

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


if __name__ == "__main__":
    unittest.main()
