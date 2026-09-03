"""Tests for bench/measure.py — the place where measurements come into being.

Why tests here of all places: the numbers from bench/ are the basis of every
statement in docs/. A measurement that silently yields a wrong rate is worse
than one that aborts — it travels into the documentation as a finding and
afterwards cannot be told apart from a real measurement.
"""
import unittest

import common

MH = common.load("bench/measure.py", "measure")


class TestEvaluate(unittest.TestCase):
    def test_computes_the_documented_rate(self):
        # The numbers behind --swa-full: setup/README.md, "The server switch
        # that decides everything", and docs/measurements/cache-hunt-finding.md.
        m = MH.evaluate({"usage": {"input_tokens": 1637,
                                    "cache_read_input_tokens": 17734}}, 10.4)
        self.assertEqual(m["new"], 1637)
        self.assertEqual(m["cached"], 17734)
        self.assertEqual(m["rate"], 91.5)
        self.assertEqual(m["seconds"], 10.4)

    def test_a_cold_run_is_zero_percent(self):
        m = MH.evaluate({"usage": {"input_tokens": 19371,
                                    "cache_read_input_tokens": 0}})
        self.assertEqual(m["rate"], 0.0)
        self.assertNotIn("seconds", m)

    def test_missing_usage_aborts_instead_of_guessing(self):
        """This used to produce -0.0 % — which reads like a finding."""
        for antwort in ({}, {"usage": {}}, {"usage": None},
                        {"usage": {"cache_read_input_tokens": 5}}, None, "broken"):
            with self.subTest(antwort=antwort):
                with self.assertRaises(MH.NoMeasurement):
                    MH.evaluate(antwort)

    def test_the_servers_error_is_passed_along(self):
        with self.assertRaises(MH.NoMeasurement) as k:
            MH.evaluate({"error": {"message": "context window exceeded"}})
        self.assertIn("context window exceeded", str(k.exception))

    def test_zero_tokens_is_no_measurement(self):
        with self.assertRaises(MH.NoMeasurement):
            MH.evaluate({"usage": {"input_tokens": 0,
                                    "cache_read_input_tokens": 0}})


class TestRequired(unittest.TestCase):
    def test_returns_the_value(self):
        self.assertEqual(MH.required({"input_tokens": 42}), 42)

    def test_raises_instead_of_minus_one(self):
        for u in ({}, None, {"cache_read_input_tokens": 3}):
            with self.subTest(u=u):
                with self.assertRaises(MH.NoMeasurement):
                    MH.required(u)


class TestGtt(unittest.TestCase):
    def test_returns_a_number_or_none_but_never_raises(self):
        # On a machine without amdgpu the result must be None, not an error.
        w = MH.gtt_gib()
        self.assertTrue(w is None or isinstance(w, float))


class TestTheBuildHelpersAreSharedNotCopied(unittest.TestCase):
    """Which binary a measurement is about, and what produced its numbers.

    Two suites need this now — restore-safety.py, which compares builds for
    the restore corruption, and np2-candidates.py, which asks the same of the
    OTHER gfx1151 defect. The second copy of anything is where this
    repository's bugs live: three parsers for LLAMA_ARGS that disagreed, three
    copies of the memory arithmetic of which the one that ran checked the
    wrong quantity, a convention list that existed in three places within
    three hours.

    So both go through bench/run.py, and neither carries its own.
    """

    SUITES = ("bench/suites/restore-safety.py",
              "bench/suites/np2-candidates.py")

    def src(self, path):
        return (common.REPO / path).read_text(encoding="utf-8")

    def test_neither_suite_defines_its_own(self):
        for suite in self.SUITES:
            src = self.src(suite)
            for name in ("def resolve_binary", "def provenance"):
                self.assertNotIn(name, src,
                                 "%s defines %s — it belongs in bench/run.py"
                                 % (suite, name))

    def test_both_suites_reach_the_shared_one(self):
        """And the count is asserted, so this cannot pass by finding
        nothing."""
        found = 0
        for suite in self.SUITES:
            src = self.src(suite)
            if "resolve_binary" in src and "provenance" in src:
                found += 1
        self.assertEqual(found, len(self.SUITES),
                         "a suite that measures a build must say which one")

    def test_the_one_implementation_is_where_it_says(self):
        src = self.src("bench/run.py")
        self.assertIn("def resolve_binary(", src)
        self.assertIn("def provenance(", src)


class TestProvenanceEnvironment(unittest.TestCase):
    """The environment a measurement ran in, as the report records it.

    The filter used to be `GGML_`/`LLAMA_` only, and on 29.08. that was one
    prefix short of the experiment being run: llama.cpp #27579 carries an
    outside reproduction on this very hardware in which
    `HIP_LAUNCH_BLOCKING=1` restores correct output. Measuring that means
    the variable IS the independent variable — and a result.json that does
    not name it is indistinguishable from a run without it. Exactly the
    failure the comment in provenance() already describes for
    GGML_SCHED_UMA_RING, one vendor prefix later.
    """

    def setUp(self):
        self.run = common.load("bench/run.py", "benchrun")

    def env_of(self, **vars_):
        import os
        old = dict(os.environ)
        try:
            os.environ.update(vars_)
            return self.run.provenance("/nonexistent/bin/llama-server")["env"]
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_a_runtime_variable_of_the_backend_is_recorded(self):
        for var in ("HIP_LAUNCH_BLOCKING", "HSA_OVERRIDE_GFX_VERSION",
                    "AMD_SERIALIZE_KERNEL", "ROCR_VISIBLE_DEVICES"):
            with self.subTest(var=var):
                self.assertIn(var, self.env_of(**{var: "1"}),
                              "%s changes what the GPU does and the report "
                              "must say whether it was set" % var)

    def test_the_variables_it_already_carried_are_still_there(self):
        env = self.env_of(GGML_SCHED_UMA_RING="1", LLAMA_SET_ROWS="1")
        self.assertIn("GGML_SCHED_UMA_RING", env)
        self.assertIn("LLAMA_SET_ROWS", env)

    def test_the_path_variable_stays_out(self):
        """LLAMA_SRC names a directory on one machine, not a configuration."""
        self.assertNotIn("LLAMA_SRC", self.env_of(LLAMA_SRC="/home/x/llama.cpp"))

    def test_unrelated_variables_stay_out(self):
        env = self.env_of(HOME="/home/x", EDITOR="vi")
        self.assertNotIn("HOME", env)
        self.assertNotIn("EDITOR", env)


class TestNoSuiteReachesIntoThePreRenameHome(unittest.TestCase):
    """The stack left ~/.claude in 09/2026; a suite that still points there
    finds nothing.

    The cost is not the missing file — it is WHEN it is missed:
    save-eviction.py stops production, restarts the gateway and then spawns
    prewarm.py, so a dead path aborts the run after the machine has already
    been taken apart. Found 01.09.2026, three weeks of it being wrong and
    unnoticed because nothing runs these suites unattended.
    """

    def test_no_suite_names_the_old_home(self):
        stale = []
        for f in sorted((common.REPO / "bench" / "suites").glob("*.py")):
            src = f.read_text(encoding="utf-8")
            for i, line in enumerate(src.splitlines(), 1):
                if "~/.claude" in line and not line.lstrip().startswith("#"):
                    stale.append("%s:%d" % (f.name, i))
        self.assertEqual(stale, [],
                         "these lines still resolve into ~/.claude, which the "
                         "09/2026 move emptied: %s" % ", ".join(stale))


class TestASweepReportSaysWhetherItsConditionsHeld(unittest.TestCase):
    """A sweep runs for tens of minutes; the power profile is not a constant
    over that. Measured 03.09.2026: this machine's platform_profile went from
    'performance' to 'quiet' unnoticed and the GPU served eight hours at 35 W
    instead of 70, at 99 % busy the whole time. A sweep straddling that moment
    would have recorded 'performance' from its first second and produced a
    table where nothing looked wrong.
    """

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()
        self.sweep = common.load("bench/sweep.py", "sweep")
        self.compare = common.load("bench/compare.py", "compare")

    def ctx(self, **kw):
        import json, os
        with open(os.path.join(self.dir, "context.json"), "w",
                  encoding="utf-8") as f:
            json.dump(kw, f)
        return self.dir

    # --- close_conditions -------------------------------------------------

    def test_an_unchanged_profile_is_recorded_as_held(self):
        c = {"platform_profile": "performance"}
        self.sweep.platform_profile = lambda: "performance"
        self.assertTrue(self.sweep.close_conditions(c))
        self.assertTrue(c["conditions_held"])
        self.assertEqual(c["platform_profile_end"], "performance")

    def test_a_changed_profile_is_recorded_and_shouted_about(self):
        import contextlib, io
        c = {"platform_profile": "performance"}
        self.sweep.platform_profile = lambda: "quiet"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(self.sweep.close_conditions(c))
        self.assertFalse(c["conditions_held"])
        said = out.getvalue()
        self.assertIn("performance -> quiet", said)
        self.assertIn("WARNING", said)

    def test_a_machine_without_the_interface_is_held_not_broken(self):
        """None at the start and None at the end is not a change. A desktop
        must not have every sweep flagged as contaminated."""
        c = {"platform_profile": None}
        self.sweep.platform_profile = lambda: None
        self.assertTrue(self.sweep.close_conditions(c))

    # --- what the report shows -------------------------------------------

    def test_a_held_run_says_so(self):
        note = self.compare.conditions_note(
            [self.ctx(platform_profile="performance", conditions_held=True)])
        self.assertIn("performance", note)
        self.assertIn("unchanged", note)

    def test_a_contaminated_run_is_labelled_not_comparable(self):
        note = self.compare.conditions_note(
            [self.ctx(platform_profile="performance",
                      platform_profile_end="quiet", conditions_held=False)])
        self.assertIn("WARNING", note)
        self.assertIn("not comparable", note)
        self.assertIn("quiet", note)

    def test_an_older_report_is_unknown_and_must_not_read_as_fine(self):
        """The gap has to survive into the output. Rendering a report that
        never recorded the answer as if it had held turns a hole in the record
        into a claim about it — which is the one thing a measurement log must
        not do."""
        note = self.compare.conditions_note(
            [self.ctx(platform_profile="balanced")])
        self.assertIn("NOT recorded", note)
        self.assertNotIn("unchanged", note)

    def test_no_context_and_no_profile_say_nothing_at_all(self):
        import tempfile
        self.assertEqual(self.compare.conditions_note([tempfile.mkdtemp()]), "")
        self.assertEqual(self.compare.conditions_note([self.ctx(model="x")]), "")

    def test_the_note_rides_above_the_table(self):
        """render() must carry it — the note living only in context.json is
        the state this fixes."""
        import json, os
        d = self.ctx(platform_profile="performance",
                     platform_profile_end="quiet", conditions_held=False)
        os.makedirs(os.path.join(d, "v1"), exist_ok=True)
        with open(os.path.join(d, "v1", "summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"label": "v1", "ctx": 65536}, f)
        out = self.compare.render(d)
        self.assertIn("WARNING", out)
        self.assertLess(out.index("WARNING"), out.index("| variant |"))


if __name__ == "__main__":
    unittest.main()
