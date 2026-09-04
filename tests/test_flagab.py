"""flag-ab — the parts that decide WHAT gets compared.

Same reasoning as test_speedab.py: a suite that runs the wrong comparison
does not fail, it prints a table and attributes the difference to the wrong
thing. The arm parser and the one-difference rule are what stand between a
label and a claim, so they are what gets tested — the plumbing around
llama-bench is speed-ab's, and tested there.
"""
import json, os, shutil, subprocess, sys, tempfile, unittest

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "bench"))
SUITE = str(REPO / "bench" / "suites" / "flag-ab.py")
flag_ab = common.load(SUITE, "flag_ab")


class TestTheArmParser(unittest.TestCase):
    def test_flags_and_env_are_told_apart(self):
        label, flags, env = flag_ab.parse_arm(
            "mixed:-ub 512 ROCBLAS_USE_HIPBLASLT=1")
        self.assertEqual(label, "mixed")
        self.assertEqual(flags, {"-ub": "512"})
        self.assertEqual(env, {"ROCBLAS_USE_HIPBLASLT": "1"})

    def test_a_flag_without_a_value_is_refused(self):
        """llama-bench would not fail on it — repeated or dangling flags feed
        its own sweep logic and quietly multiply the run matrix."""
        with self.assertRaises(SystemExit):
            flag_ab.parse_arm("bad:-ub")
        with self.assertRaises(SystemExit):
            flag_ab.parse_arm("bad:-ub -fa on")

    def test_free_text_is_refused_not_forwarded(self):
        with self.assertRaises(SystemExit):
            flag_ab.parse_arm("bad:512")

    def test_an_empty_spec_is_refused(self):
        with self.assertRaises(SystemExit):
            flag_ab.parse_arm("bad:")
        with self.assertRaises(SystemExit):
            flag_ab.parse_arm("no-colon")


class TestTheOneDifferenceRule(unittest.TestCase):
    """The point of the suite, as it is of speed-ab: zero differences measure
    nothing, two measure neither."""

    def arm(self, label, spec):
        return flag_ab.parse_arm("%s:%s" % (label, spec))

    def test_one_varying_flag_is_the_axis(self):
        arms = [self.arm("a", "-ub 512"), self.arm("b", "-ub 1024"),
                self.arm("c", "-ub 2048")]
        self.assertEqual(flag_ab.the_one_axis(arms), ("flag", "-ub"))

    def test_one_varying_env_is_the_axis(self):
        arms = [self.arm("off", "ROCBLAS_USE_HIPBLASLT=0"),
                self.arm("on", "ROCBLAS_USE_HIPBLASLT=1")]
        self.assertEqual(flag_ab.the_one_axis(arms),
                         ("env", "ROCBLAS_USE_HIPBLASLT"))

    def test_identical_arms_are_refused(self):
        arms = [self.arm("a", "-ub 512"), self.arm("b", "-ub 512")]
        with self.assertRaises(SystemExit) as cm:
            flag_ab.the_one_axis(arms)
        self.assertIn("nothing to compare", str(cm.exception))

    def test_two_differences_at_once_are_refused(self):
        arms = [self.arm("a", "-ub 512 -fa on"),
                self.arm("b", "-ub 1024 -fa off")]
        with self.assertRaises(SystemExit) as cm:
            flag_ab.the_one_axis(arms)
        self.assertIn("MORE THAN ONE", str(cm.exception))

    def test_a_variable_missing_from_one_arm_is_refused(self):
        """An omitted variable would mean llama-bench's default — a value
        nobody wrote down, so the arms would differ in something unstated."""
        arms = [self.arm("a", "-ub 512 -fa on"), self.arm("b", "-ub 1024")]
        with self.assertRaises(SystemExit) as cm:
            flag_ab.the_one_axis(arms)
        self.assertIn("same variables", str(cm.exception))


class TestTheInvocationCarriesTheProfile(unittest.TestCase):
    def test_base_matches_the_serving_profile(self):
        """An arm that does not vary a knob must measure production's value,
        not llama-bench's default. BASE is a declared copy of qwen38.env, and
        this test is what keeps the copy honest: it reads the profile's own
        LLAMA_ARGS rather than repeating the numbers here — repeating them
        would be a THIRD reader of one file, which is how the three
        LLAMA_ARGS parsers began. The day this went in, BASE still said
        -ub 2048 while the profile had moved to 512."""
        env = (REPO / "setup" / "env" / "qwen38.env").read_text()
        args = env.replace("\\\n", " ")
        line = next(l for l in args.splitlines()
                    if l.startswith("LLAMA_ARGS="))
        toks = line.split()
        for flag in ("-fa", "-ub", "-b"):
            self.assertIn(flag, toks, "profile no longer carries %s" % flag)
            self.assertEqual(flag_ab.BASE[flag], toks[toks.index(flag) + 1],
                             "BASE[%s] drifted from qwen38.env" % flag)

    def test_an_arm_overrides_rather_than_repeats(self):
        """llama-bench accumulates repeated flags into a sweep; a base -ub
        AND an arm's -ub would run BOTH. The dry argv must carry -ub exactly
        once, with the arm's value. The arm value is chosen to differ from
        BASE's, whatever BASE currently says — an override test that
        overrides with the base value tests string formatting."""
        arm_ub = "1024"
        self.assertNotEqual(flag_ab.BASE["-ub"], arm_ub,
                            "pick a different arm value for this test")
        rows = []
        real = flag_ab.sab.say
        flag_ab.sab.say = rows.append
        try:
            flag_ab.bench("/nonexistent/llama-bench", "/m.gguf", [0], 2048,
                          64, {"-ub": arm_ub}, {}, dry=True)
        finally:
            flag_ab.sab.say = real
        argv = rows[0]
        self.assertEqual(argv.count(" -ub "), 1, argv)
        self.assertIn("-ub " + arm_ub, argv)
        self.assertNotIn("-ub " + flag_ab.BASE["-ub"], argv)


class TestAnEmptyTableIsNeverSilent(unittest.TestCase):
    """THE defect this suite met on 04.09.2026, end to end.

    Three arms over two interleaved rounds; all six llama-bench invocations
    failed with `failed to load model`. The suite exited 0 and wrote
    bench/reports/2026-09-04_1220_qwen36-ub/ — a RESULT.md whose table had a
    header row and no data rows, and a rounds.json with three empty objects.
    No failure count, no error text, in either file. A reader opening that
    directory sees an empty table, which is what a NULL RESULT looks like.

    Driven as a subprocess against a stub llama-bench, because the exit code
    is half the subject and `sys.exit()` is what carries it. Nothing is
    stopped and nothing is started: `--keep-production` and `--no-warmup`, so
    this says the same thing on a machine that is serving.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    FAILS = ("#!/bin/sh\n"
             "echo 'ggml_backend_load_all: loading' >&2\n"
             "echo \"llama_bench: error: failed to load model\" >&2\n"
             "exit 1\n")

    # llama-bench's own JSON shape, cut to the four fields key_of/label_of and
    # the table read. The stub fails only for the arm carrying -ub 1024.
    ROWS = ('[{"n_prompt":512,"n_gen":0,"n_depth":0,"avg_ts":100.0},'
            '{"n_prompt":0,"n_gen":64,"n_depth":0,"avg_ts":20.0}]')
    HALF = ("#!/bin/sh\n"
            "for a in \"$@\"; do case \"$a\" in 1024)\n"
            "  echo 'llama_bench: error: failed to load model' >&2; exit 1;;\n"
            "esac; done\n"
            "echo '%s'\n" % ROWS)

    def build(self, body):
        d = os.path.join(self.tmp, "build-stub")
        os.makedirs(os.path.join(d, "bin"), exist_ok=True)
        for exe in ("llama-bench", "llama-server"):
            p = os.path.join(d, "bin", exe)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, 0o755)
        with open(os.path.join(d, ".build-stamp"), "w") as f:
            f.write("build_id=stub-1\nbuilt_at=2026-09-04T10:00:00+02:00\n"
                    "upstream_commit=%s\n" % ("c" * 40))
        return d

    def run_suite(self, out):
        return subprocess.run(
            [sys.executable, SUITE, "--build", "build-stub",
             "--model", "/nonexistent/m.gguf", "--name", "stub",
             "--arm", "a:-ub 512", "--arm", "b:-ub 1024",
             "--depths", "0", "--prompt", "512", "--reps", "1",
             "--keep-production", "--no-warmup", "--out", out],
            capture_output=True, text=True, timeout=300,
            env=dict(os.environ, LLAMA_SRC=self.tmp))

    def read(self, path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_every_arm_failing_does_not_exit_0(self):
        """`exited 0` is the part that let this be read as a measurement — a
        caller chaining on `&&` had nothing to go on."""
        self.build(self.FAILS)
        out = os.path.join(self.tmp, "report")
        r = self.run_suite(out)
        self.assertNotEqual(r.returncode, 0,
                            "a run that measured NOTHING exited 0:\n" + r.stdout)

    def test_the_report_says_how_many_failed_and_quotes_the_error(self):
        self.build(self.FAILS)
        out = os.path.join(self.tmp, "report")
        self.run_suite(out)
        md = self.read(os.path.join(out, "RESULT.md"))
        self.assertIn("2 of 2 llama-bench invocations FAILED", md)
        self.assertIn("llama_bench: error: failed to load model", md)
        self.assertIn("NOTHING was measured", md)

    def test_the_stderr_lands_in_rounds_json(self):
        """The empty rounds.json of 04.09. carried no error string at all —
        so the one artefact that survives a run could not say what happened."""
        self.build(self.FAILS)
        out = os.path.join(self.tmp, "report")
        self.run_suite(out)
        d = json.loads(self.read(os.path.join(out, "rounds.json")))
        self.assertEqual(d["_failures"]["attempted"], 2)
        self.assertEqual(d["_failures"]["failed"], 2)
        arms = sorted(f["arm"] for f in d["_failures"]["detail"])
        self.assertEqual(arms, ["a", "b"], "a failure must name its arm")
        self.assertIn("failed to load model",
                      d["_failures"]["detail"][0]["stderr_tail"])

    def test_one_arm_failing_still_completes_and_exits_0(self):
        """The repo's rule, and it is not softness: a cell that fails is
        recorded rather than fatal (bench/README.md, `Comparing two builds`),
        because ending the run on the first failure is what cost
        `prefill-nospec` three times over. Only ALL of them failing is not a
        measurement."""
        self.build(self.HALF)
        out = os.path.join(self.tmp, "report")
        r = self.run_suite(out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        md = self.read(os.path.join(out, "RESULT.md"))
        self.assertIn("1 of 2 llama-bench invocations FAILED", md)
        self.assertIn("failed in: b (1)", md,
                      "the short table must name the arm that emptied it")
        self.assertNotIn("NOTHING was measured", md)

    def test_a_clean_run_states_the_tally_too(self):
        """Otherwise a report that mentions failures only when there are some
        cannot be told from one written before it could mention them — which
        is every report in bench/reports/ before 04.09.2026."""
        self.build("#!/bin/sh\necho '%s'\n" % self.ROWS)
        out = os.path.join(self.tmp, "report")
        r = self.run_suite(out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        md = self.read(os.path.join(out, "RESULT.md"))
        self.assertIn("all 2 llama-bench invocations succeeded", md)
        self.assertIn("pp512 @ d0", md, "the table itself must still be there")


if __name__ == "__main__":
    unittest.main()
