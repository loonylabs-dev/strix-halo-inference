"""bench/sideserver.py, workload path — the trust boundaries of the fence.

Born from the 01.09.2026 quality review, whose sharpest finding was this:
`systemctl show` of a unit that no longer exists answers Result=success,
ExecMainStatus=0, ActiveState=inactive — measured on this systemd (259).
A fence that reads those without checking LoadState fabricates a verdict
and prints ready-to-paste "measured" declarations of 0.0. Every test here
pins a boundary where the fence must say "unknown" or refuse, instead of
inventing the convenient answer.
"""
import signal
import sys
import unittest

import common

sys.path.insert(0, str(common.REPO / "bench"))
sys.path.insert(0, str(common.REPO / "setup" / "lib"))

import sideserver                                             # noqa: E402


def fake_show(answers):
    """A systemctl-show stand-in: returns `key=value` lines for the
    requested properties, the way `systemctl show -p a -p b` does."""
    def run(unit, names):
        return "\n".join("%s=%s" % (n, answers.get(n, "")) for n in names)
    return run


class TestAVanishedUnitIsNotASuccess(unittest.TestCase):
    """systemd's not-found answers, verbatim from the review's measurement."""
    NOT_FOUND = {"LoadState": "not-found", "ActiveState": "inactive",
                 "SubState": "dead", "Result": "success",
                 "ExecMainStatus": "0"}

    def test_props_carry_the_load_state(self):
        props = sideserver.unit_props(
            "ghost", ["LoadState", "Result", "ExecMainStatus"],
            run=fake_show(self.NOT_FOUND))
        self.assertEqual(props["LoadState"], "not-found")

    def test_the_outcome_of_a_vanished_unit_is_unknown_never_ok(self):
        ok, line = sideserver.job_outcome(self.NOT_FOUND)
        self.assertFalse(ok)
        self.assertIn("unknown", line.lower())

    def test_a_loaded_successful_unit_is_ok(self):
        ok, line = sideserver.job_outcome(
            {"LoadState": "loaded", "ActiveState": "active",
             "SubState": "exited", "Result": "success",
             "ExecMainStatus": "0"})
        self.assertTrue(ok, line)

    def test_a_loaded_failed_unit_is_not_ok(self):
        ok, line = sideserver.job_outcome(
            {"LoadState": "loaded", "ActiveState": "failed",
             "SubState": "failed", "Result": "exit-code",
             "ExecMainStatus": "2"})
        self.assertFalse(ok)


class TestTheReleaseBaseline(unittest.TestCase):
    """The teardown waited for GTT to fall below the PRE-STOP reading —
    ~36 GiB with production up — which is true the moment the workload
    starts tearing down. Production then restarts onto a teardown in
    flight: the exact step the 26.08. incidents skipped, on the way back."""

    def test_settled_wins_when_production_was_stopped(self):
        self.assertEqual(sideserver.release_baseline(36.2, 0.7), 0.7)

    def test_the_prestop_reading_serves_when_nothing_was_stopped(self):
        self.assertEqual(sideserver.release_baseline(36.2, None), 36.2)

    def test_no_reading_at_all_falls_to_zero(self):
        self.assertEqual(sideserver.release_baseline(None, None), 0.0)


class TestTheDeadlineCoversTheJob(unittest.TestCase):
    """--job-timeout 3600 with the default --deadline 45 lets the dead
    man's switch start production INTO the running measurement: the peak
    metering then swallows qwen38's ~36 GiB and offers the contaminated
    number as a declaration."""

    def test_a_covered_job_passes(self):
        self.assertTrue(sideserver.deadline_covers(1800, 45))

    def test_an_uncovered_job_is_refused(self):
        self.assertFalse(sideserver.deadline_covers(3600, 45))

    def test_the_settle_and_release_waits_are_part_of_the_arithmetic(self):
        """Review finding (01.09.2026): the deadline clock starts at
        arming, so it must also cover the pre-job settle wait (up to
        180 s) and the teardown's GTT-release wait (up to 180 s) — both
        coded in this same file. 2340+300 fit inside 45 min and the check
        passed while the release wait was still running at fire time."""
        self.assertFalse(sideserver.deadline_covers(2340, 45))

    def test_the_boundary_belongs_to_the_refusal_side(self):
        # job + slack + settle + release == the deadline exactly — equality
        # is not coverage.
        boundary = (45 * 60 - sideserver.DEADLINE_SLACK_S
                    - sideserver.GTT_SETTLE_TIMEOUT_S
                    - sideserver.GTT_RELEASE_TIMEOUT_S)
        self.assertFalse(sideserver.deadline_covers(boundary, 45))
        self.assertTrue(sideserver.deadline_covers(boundary - 1, 45))


class TestMeterBlindnessIsNamed(unittest.TestCase):
    """A metered 0.0 and a blind instrument printed identically. The meter
    now says which quantities it actually SAW; a declaration is only
    offered for a seen one."""

    def _run(self, gtt_values, rss_value, props_answers=None):
        answers = props_answers or {"LoadState": "loaded",
                                    "ActiveState": "inactive",
                                    "SubState": "dead",
                                    "MainPID": "0", "ControlGroup": ""}
        gtt_iter = iter(gtt_values)

        def props_of(unit, names):
            return {n: answers.get(n, "") for n in names}

        return sideserver.meter_until_exit(
            "u", 0.5, timeout=5,
            props_of=props_of,
            gtt=lambda: next(gtt_iter, None),
            rss_of=lambda cg: rss_value,
            sleep=lambda s: None)

    def test_no_sample_at_all_reads_as_unseen_not_zero(self):
        m = self._run(gtt_values=[None], rss_value=None)
        self.assertFalse(m.gtt_seen)
        self.assertFalse(m.rss_seen)

    def test_a_real_zero_is_a_seen_zero(self):
        """chatterbox's GTT 0.0 is a MEASURED zero (CPU job, amdgpu
        readable) — the flag is what keeps that distinct from blindness."""
        m = self._run(gtt_values=[0.5], rss_value=1.4)
        self.assertTrue(m.gtt_seen)
        self.assertAlmostEqual(m.peak_gtt, 0.0)
        self.assertTrue(m.rss_seen)

    def test_a_vanished_unit_ends_the_meter_and_says_so(self):
        m = self._run(gtt_values=[0.5, 0.5],
                      rss_value=None,
                      props_answers={"LoadState": "not-found",
                                     "ActiveState": "inactive",
                                     "SubState": "dead",
                                     "MainPID": "0", "ControlGroup": ""})
        self.assertTrue(m.vanished)
        self.assertFalse(m.timed_out)

    def test_one_dbus_hiccup_does_not_kill_a_healthy_job(self):
        """Re-review finding (01.09.2026): a single transiently empty
        `systemctl show` answer read as vanished and tore down a RUNNING
        job — loud, but a full fenced cycle lost. One hiccup is tolerated;
        two consecutive non-loaded reads are a verdict."""
        answers = iter([
            {},                                              # the hiccup
            {"LoadState": "loaded", "ActiveState": "active",
             "SubState": "running", "MainPID": "7", "ControlGroup": "cg"},
            {"LoadState": "loaded", "ActiveState": "inactive",
             "SubState": "dead", "MainPID": "0", "ControlGroup": "cg"},
        ])
        m = sideserver.meter_until_exit(
            "u", 0.5, timeout=10,
            props_of=lambda u, names: next(answers),
            gtt=lambda: 0.5, rss_of=lambda cg: None,
            sleep=lambda s: None)
        self.assertFalse(m.vanished,
                         "one empty show answer must not end a healthy job")
        self.assertFalse(m.timed_out)


class TestSigtermRunsTheTeardown(unittest.TestCase):
    """Review finding (01.09.2026): the teardown is finally-only, and
    SIGTERM ends Python WITHOUT unwinding — only SIGINT becomes an
    exception by default. A SIGTERM mid-metering left the workload unit
    pinning GTT and production down until the dead man's switch started
    qwen38 INTO the still-running job: the co-residency the fence exists
    to prevent."""

    def test_sigterm_becomes_a_systemexit_so_finally_runs(self):
        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)
        sideserver.install_sigterm_handler()
        ran_teardown = False
        with self.assertRaises(SystemExit) as ctx:
            try:
                signal.raise_signal(signal.SIGTERM)
            finally:
                ran_teardown = True
        self.assertTrue(ran_teardown, "the finally block must run")
        self.assertEqual(ctx.exception.code, 128 + signal.SIGTERM)

    def test_main_installs_the_handler_before_any_dance(self):
        src = (common.REPO / "bench" / "sideserver.py").read_text(
            encoding="utf-8")
        body = src[src.index("def main("):]
        self.assertIn("install_sigterm_handler()", body,
                      "main() must arm the handler for BOTH paths — the "
                      "llama finally has the same hole")
        self.assertLess(body.index("install_sigterm_handler()"),
                        body.index("workload_main(a)"),
                        "the handler must be armed before dispatch")


class TestArmingIsVerified(unittest.TestCase):
    """Review finding (01.09.2026): arm_deadman discarded the systemd-run
    result and announced 'armed' unconditionally. A stale deadman unit
    from a SIGKILLed fence makes systemd-run fail on the name collision —
    the run then proceeded with no working switch, or with a stale timer
    firing mid-measurement on its older clock."""

    def test_a_failed_arming_says_no(self):
        def failing_run(cmd, **kw):
            class R:
                returncode = 1
                stderr = "Failed to start transient timer unit: exists"
                stdout = ""
            return R()
        self.assertFalse(sideserver.arm_deadman(
            "deadman-x", "llama-user@qwen38", 45, run=failing_run))

    def test_a_successful_arming_says_yes(self):
        def ok_run(cmd, **kw):
            class R:
                returncode = 0
                stderr = ""
                stdout = ""
            return R()
        self.assertTrue(sideserver.arm_deadman(
            "deadman-x", "llama-user@qwen38", 45, run=ok_run))

    def test_both_call_sites_refuse_on_a_failed_arming(self):
        src = (common.REPO / "bench" / "sideserver.py").read_text(
            encoding="utf-8")
        self.assertEqual(src.count("if not arm_deadman("), 2,
                         "both paths must check the arming before they "
                         "stop production — an unarmed fence is not a fence")


class TestCeilingKeepsAMeasuredZero(unittest.TestCase):
    """chatterbox declares WORKLOAD_GTT_GIB=0.0 — a MEASURED zero (CPU
    job). `if not gtt:` conflated it with 'no figure' and substituted the
    live GTT reading: a fabricated rationale and a tighter MemoryMax, the
    exact seen-zero-vs-blind distinction the meter's gtt_seen flag was
    built to keep (review, 01.09.2026)."""

    def test_a_measured_zero_is_not_replaced_by_the_live_reading(self):
        mem_max, _, why = sideserver.ceiling(
            total_gib=124.9, live_gtt_gib=36.0, gtt_override=0.0)
        self.assertIn("0 the model will pin", why)
        self.assertEqual(mem_max, "%dG" % int(
            124.9 - 0.0 - sideserver.HOST_RESERVE_GIB))

    def test_no_figure_still_falls_back_to_the_live_reading(self):
        _, _, why = sideserver.ceiling(
            total_gib=124.9, live_gtt_gib=36.0, gtt_override=None,
            argv=[])
        self.assertIn("36 the model will pin", why)


class TestWorkloadRefusesLlamaOnlyFlags(unittest.TestCase):
    """Review finding (01.09.2026): --bin/--extra/--port/--slots-timeout
    parsed cleanly beside --workload and were silently ignored — an
    operator passing --extra "--steps 5" got the profile's full run and
    read the timing as a 5-step figure."""

    def _expect_refusal(self, argv):
        with self.assertRaises(SystemExit) as ctx:
            sideserver.main(argv)
        self.assertEqual(ctx.exception.code, 2)

    def test_extra_is_refused(self):
        self._expect_refusal(["--workload", "x.env", "--extra", "--steps 5"])

    def test_bin_is_refused(self):
        self._expect_refusal(["--workload", "x.env", "--bin", "/x/y"])

    def test_port_is_refused(self):
        self._expect_refusal(["--workload", "x.env", "--port", "9999"])

    def test_slots_timeout_is_refused(self):
        self._expect_refusal(["--workload", "x.env", "--slots-timeout", "5"])


class TestTheDefaultPropsReaderCannotHang(unittest.TestCase):
    """Ultrareview finding (01.09.2026): the default systemctl-show call
    carried no timeout, and meter_until_exit polls it at 1 Hz inside the
    fenced window. A dbus stall then blocks the loop IN the call — the
    job-timeout is never re-checked, the teardown queues behind the same
    block, and the dead man's switch eventually starts production onto
    the still-pinning workload. A timeout turns the hang into the empty
    answer the two-reads hiccup path already handles."""

    def _patch_run(self, fake):
        import subprocess as sp
        real = sp.run
        sp.run = fake
        self.addCleanup(setattr, sp, "run", real)

    def test_the_systemctl_call_carries_a_timeout(self):
        seen = {}

        def fake(cmd, **kw):
            seen.update(kw)

            class R:
                stdout = "LoadState=loaded\n"
            return R()
        self._patch_run(fake)
        props = sideserver.unit_props("u", ["LoadState"])
        self.assertEqual(props.get("LoadState"), "loaded")
        self.assertGreater(seen.get("timeout") or 0, 0,
                           "the 1 Hz poll must not be able to hang forever")

    def test_a_hung_systemctl_reads_as_an_empty_answer(self):
        import subprocess as sp

        def fake(cmd, **kw):
            raise sp.TimeoutExpired(cmd, kw.get("timeout") or 5)
        self._patch_run(fake)
        self.assertEqual(sideserver.unit_props("u", ["LoadState"]), {},
                         "a timeout is the dbus-hiccup shape, not a crash")


class TestNoDirectStopOutsideTheHelper(unittest.TestCase):
    """The pre-refactor test forbade any raw stop of production; the
    call-site rewrite (01.09.) lost that property — a rogue direct
    `systemctl("stop", a.stop)` with a matching arm count would have
    passed. Restored here at the literal level."""

    def test_the_raw_stop_literal_does_not_exist(self):
        src = (common.REPO / "bench" / "sideserver.py").read_text(
            encoding="utf-8")
        self.assertNotIn('systemctl("stop", a.stop)', src,
                         "production may only be stopped through "
                         "stop_production_and_settle()")
        self.assertEqual(src.count('systemctl("stop", production_unit)'), 1,
                         "exactly one place stops production — the helper")


if __name__ == "__main__":
    unittest.main()
