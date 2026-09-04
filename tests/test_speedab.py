"""unroll-flag — the parts that decide WHICH binaries get compared.

A measurement suite that picks the wrong pair does not fail. It runs, prints
a table, and attributes the difference to the flag — which is the failure mode
this repository keeps meeting, and the reason the picking is tested rather
than the plumbing.

One of these tests exists because the bug had already happened: newest_of_family()
returned a full PATH, runlib.resolve_binary() takes a path OR a directory name
OR a build id, and resolved the path against $LLAMA_SRC — producing
`~/llama.cpp/llama-bench`, a file that does not exist. That one announced
itself. The version where it silently finds the WRONG build would not.
"""
import json, os, shutil, subprocess, sys, tempfile, unittest

import common

REPO = common.REPO
SUITE = str(REPO / "bench" / "suites" / "speed-ab.py")
# The suite imports bench/run.py as `run`, so that has to be importable
# before it is loaded — the hyphen in its own filename is why it cannot
# simply be imported by name.
sys.path.insert(0, str(REPO / "bench"))
speed_ab = common.load(SUITE, "speed_ab")


def make_build(root, name, built_at, cmake="-DCMAKE_BUILD_TYPE=Release",
               commit="c799f10147916ad58f00648b5ef0b87425f554c0"):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    for exe in ("llama-server", "llama-bench"):
        p = os.path.join(d, "bin", exe)
        with open(p, "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(p, 0o755)
    with open(os.path.join(d, ".build-stamp"), "w") as f:
        f.write("build_id=%s\nfamily=rocm-unroll\npatched=yes\n"
                "upstream_commit=%s\nbuilt_at=%s\ncmake=%s\n"
                % (name.split("-", 3)[-1], commit, built_at, cmake))
    return d


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestItPicksTheRightUnrollBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._old = os.environ.get("LLAMA_SRC")
        os.environ["LLAMA_SRC"] = self.tmp
        self.addCleanup(self._restore)

    def _restore(self):
        if self._old is None:
            os.environ.pop("LLAMA_SRC", None)
        else:
            os.environ["LLAMA_SRC"] = self._old

    def test_it_returns_a_directory_name_not_a_path(self):
        """The bug that already happened. resolve_binary() resolves a bare
        name against $LLAMA_SRC and a path as itself; handing it a path it
        then joins produced ~/llama.cpp/llama-bench."""
        make_build(self.tmp, "build-rocm-unroll-b1", "2026-08-31T00:23:15+02:00")
        got = speed_ab.newest_of_family()
        self.assertEqual(got, "build-rocm-unroll-b1")
        self.assertNotIn(os.sep, got)

    def test_it_takes_the_newest_by_stamp_not_by_name(self):
        """Directory order is arbitrary and build ids do not sort by age."""
        make_build(self.tmp, "build-rocm-unroll-zzz", "2026-08-01T10:00:00+02:00")
        make_build(self.tmp, "build-rocm-unroll-aaa", "2026-08-30T10:00:00+02:00")
        self.assertEqual(speed_ab.newest_of_family(), "build-rocm-unroll-aaa")

    def test_it_ignores_the_other_families(self):
        make_build(self.tmp, "build-rocm-patched-b1", "2026-08-31T00:00:00+02:00")
        make_build(self.tmp, "build-rocm-unpatched-b1", "2026-08-31T00:00:00+02:00")
        self.assertIsNone(speed_ab.newest_of_family())

    def test_nothing_there_is_none_rather_than_a_guess(self):
        self.assertIsNone(speed_ab.newest_of_family())


class TestTheStampIsReadFromBesideTheBinary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_it_finds_the_stamp_two_levels_up_from_bin(self):
        d = make_build(self.tmp, "build-rocm-unroll-b1", "2026-08-31T00:00:00+02:00",
                       cmake="-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600")
        st = speed_ab.stamp_beside(os.path.join(d, "bin", "llama-bench"))
        self.assertIn("amdgpu-unroll", st["cmake"])
        self.assertEqual(st["patched"], "yes")

    def test_a_missing_stamp_is_empty_rather_than_an_exception(self):
        """A build without a stamp predates them. The suite must be able to
        SAY that, which it cannot do from a traceback."""
        self.assertEqual(
            speed_ab.stamp_beside("/nonexistent/bin/llama-bench"), {})


class TestItRefusesAPairThatCannotAnswerTheQuestion(unittest.TestCase):
    """The comparison is only worth running if exactly one arm carries the
    flag. Both or neither is a run that produces a table meaning nothing."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_suite(self, ref, unroll):
        return subprocess.run(
            [sys.executable, SUITE, "--reference", ref, "--variant", unroll,
             "--dry-run"],
            capture_output=True, text=True, timeout=120,
            env=dict(os.environ, LLAMA_SRC=self.tmp))

    def test_it_refuses_when_the_reference_already_carries_the_flag(self):
        flag = "-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600"
        make_build(self.tmp, "build-rocm-patched-b1", "2026-08-30T10:00:00+02:00",
                   cmake=flag)
        make_build(self.tmp, "build-rocm-unroll-b1", "2026-08-30T11:00:00+02:00",
                   cmake=flag)
        r = self.run_suite("build-rocm-patched-b1", "build-rocm-unroll-b1")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("nothing to compare", (r.stdout + r.stderr).lower())

    def test_it_refuses_a_pair_that_differs_in_nothing(self):
        """Neither arm carries the flag, same ROCm, same commit. A table from
        this pair would be two runs of the same binary wearing two names."""
        make_build(self.tmp, "build-rocm-patched-b1", "2026-08-30T10:00:00+02:00")
        make_build(self.tmp, "build-rocm-unroll-b1", "2026-08-30T11:00:00+02:00")
        r = self.run_suite("build-rocm-patched-b1", "build-rocm-unroll-b1")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("do not differ in anything", r.stdout + r.stderr)

    def test_two_differences_at_once_are_refused(self):
        """The flag AND a different commit. Whatever such a table showed
        could not be attributed to either, which is precisely the mistake
        llama.cpp#19984 made — so it is refused rather than footnoted."""
        flag = "-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600"
        make_build(self.tmp, "build-rocm-patched-b1", "2026-08-30T10:00:00+02:00",
                   commit="a" * 40)
        make_build(self.tmp, "build-rocm-unroll-b1", "2026-08-30T11:00:00+02:00",
                   cmake=flag, commit="b" * 40)
        r = self.run_suite("build-rocm-patched-b1", "build-rocm-unroll-b1")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("MORE THAN ONE difference", r.stdout + r.stderr)
        self.assertIn("the llama.cpp commit", r.stdout + r.stderr)
        self.assertIn("the unroll flag", r.stdout + r.stderr)


class TestTheBatchGeometryFollowsTheProfile(unittest.TestCase):
    def test_ub_and_batch_match_qwen38_env(self):
        """The suite measures builds AT THE OPERATING POINT. Its -ub/-b were
        hardcoded 2048 and survived the profile's move to 512 by a few hours
        — a comparison run there would have ranked builds at a batch size
        production no longer serves. Same rule as flag-ab's BASE, same
        anti-drift check: read the profile's own LLAMA_ARGS, do not repeat
        numbers here."""
        env = (REPO / "setup" / "env" / "qwen38.env").read_text()
        line = next(l for l in env.replace("\\\n", " ").splitlines()
                    if l.startswith("LLAMA_ARGS="))
        toks = line.split()
        self.assertEqual(speed_ab.UB, toks[toks.index("-ub") + 1],
                         "speed-ab UB drifted from qwen38.env")
        self.assertEqual(speed_ab.BATCH, toks[toks.index("-b") + 1],
                         "speed-ab BATCH drifted from qwen38.env")


class TestTheSummaryArithmetic(unittest.TestCase):
    def test_the_median_is_the_middle_not_the_mean(self):
        """One slow round — a background job, a thermal dip — must not move
        the answer the way a mean would."""
        self.assertEqual(speed_ab.median([10.0, 11.0, 100.0]), 11.0)
        self.assertEqual(speed_ab.median([10.0, 20.0]), 15.0)
        self.assertEqual(speed_ab.median([]), 0.0)

    def test_rows_are_matched_across_arms_by_shape_and_depth(self):
        """pp512@d0 must never be compared against pp512@d65536."""
        a = {"n_prompt": 512, "n_gen": 0, "n_depth": 32768}
        b = {"n_prompt": 512, "n_gen": 0, "n_depth": 0}
        self.assertNotEqual(speed_ab.key_of(a), speed_ab.key_of(b))
        self.assertEqual(speed_ab.label_of(speed_ab.key_of(a)),
                         "pp512 @ d32768")
        self.assertEqual(
            speed_ab.label_of(speed_ab.key_of(
                {"n_prompt": 0, "n_gen": 128, "n_depth": 0})), "tg128 @ d0")


class TestTheArmsTakeTurnsGoingFirst(unittest.TestCase):
    """Straight A,B,A,B leaves B second in EVERY round, on a machine one pass
    warmer. Measured on a pair that differed in nothing, that is worth -0.5 to
    -1.2 % to whichever arm goes second — small, but the same order as the
    prefill difference this suite was asked to resolve. Alternating cancels it
    rather than making it small."""

    ARMS = [("reference", "/a"), ("variant", "/b")]

    def test_it_alternates(self):
        first = [speed_ab.order_for(i, self.ARMS)[0][0] for i in range(4)]
        self.assertEqual(first, ["reference", "variant",
                                 "reference", "variant"])

    def test_each_arm_leads_half_the_rounds(self):
        for n in (2, 4, 6):
            leads = [speed_ab.order_for(i, self.ARMS)[0][0] for i in range(n)]
            self.assertEqual(leads.count("reference"), n // 2, leads)
            self.assertEqual(leads.count("variant"), n // 2, leads)

    def test_both_arms_run_in_every_round(self):
        """Alternating must reorder, never drop — a round with one arm would
        silently halve that arm's sample count."""
        for i in range(4):
            names = sorted(n for n, _ in speed_ab.order_for(i, self.ARMS))
            self.assertEqual(names, ["reference", "variant"])

    def test_it_does_not_mutate_the_caller_s_list(self):
        """reversed() on the shared list would flip the arms permanently, and
        every later round would then read the wrong labels."""
        arms = list(self.ARMS)
        for i in range(4):
            speed_ab.order_for(i, arms)
        self.assertEqual(arms, self.ARMS)


class TestReportsDoNotNameThisMachine(unittest.TestCase):
    """Reports are PUBLISHED. A home directory in one names the machine it was
    measured on, and tests/test_localenv.py does not read bench/reports/ — so
    this has to be got right at the point of writing rather than caught later.

    systemdfile.unexpand()'s own docstring records the previous occurrence:
    three reports written with a home directory in them."""

    def test_paths_are_folded(self):
        home = os.path.expanduser("~")
        self.assertNotIn(home, speed_ab.rec(os.path.join(home, "llama.cpp")))
        self.assertIn("@HOME@", speed_ab.rec(os.path.join(home, "llama.cpp")))

    def test_a_whole_cmake_line_is_folded(self):
        """The stamp's cmake line carries several absolute paths at once, and
        it is copied into the report verbatim."""
        home = os.path.expanduser("~")
        line = ("-DCMAKE_HIP_COMPILER=%s/sdk/llvm/bin/clang "
                "-DROCM_PATH=%s/sdk" % (home, home))
        self.assertNotIn(home, speed_ab.rec(line))

    def test_the_written_reports_are_clean(self):
        """The two that exist. Belt and braces: the helper could be right and
        a caller still bypass it."""
        home = os.path.expanduser("~")
        import glob
        found = []
        for f in glob.glob(str(REPO / "bench" / "reports" / "*" / "RESULT.md")):
            with open(f, encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if home in line:
                        found.append("%s:%d" % (os.path.basename(
                            os.path.dirname(f)), n))
        self.assertEqual(found, [], "reports naming this machine")


class TestTheProductionUnitIsAskedAndNotAssumed(unittest.TestCase):
    """The most expensive kind of bug this suite can have: it succeeds.

    `UNIT = "llama-user@qwen38"` was a module constant until 04.09.2026, and
    on that day a flag-ab run measuring a completely different model stopped
    `llama-user@flashnext` at the start and started `llama-user@qwen38` at the
    end. Nothing failed. The table printed, the suite exited 0, and the
    machine served a model nobody had switched to — `is-enabled` still said
    flashnext, and only the process holding port 8080 disagreed. The dead
    man's switch had armed the same wrong start, so a crash would have done it
    too.

    CLAUDE.md carries the rule and records the same defect being fixed in the
    determinism lane on 01.09.2026. This copy survived that review because it
    hard-wired the UNIT rather than the profile — a second spelling of one
    mistake, which is exactly what a test is for and a review is not.

    These tests drive `unit()` against a stubbed `models.sh serving` rather
    than against this machine, so they say the same thing on a machine with
    nothing running.
    """

    def setUp(self):
        speed_ab._UNIT = None
        self.addCleanup(setattr, speed_ab, "_UNIT", None)
        self.calls = []

    def stub_serving(self, stdout):
        """Replace the ONE reader with a stub and record that it was asked."""
        real = speed_ab.subprocess.run
        calls = self.calls

        def fake(cmd, *a, **k):
            if len(cmd) >= 2 and cmd[0] == "bash" and cmd[1].endswith("models.sh"):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout, "")
            return real(cmd, *a, **k)

        speed_ab.subprocess.run = fake
        self.addCleanup(setattr, speed_ab.subprocess, "run", real)

    def test_it_reports_whatever_is_serving_and_not_a_constant(self):
        self.stub_serving("flashnext\n")
        self.assertEqual(speed_ab.unit(), "llama-user@flashnext")
        self.assertTrue(self.calls, "models.sh serving was never asked")
        self.assertEqual(self.calls[0][2], "serving",
                         "asked models.sh the wrong question — `active` cannot "
                         "say which instance won the race for port 8080")

    def test_a_different_model_gives_a_different_unit(self):
        self.stub_serving("qwen36\n")
        self.assertEqual(speed_ab.unit(), "llama-user@qwen36")

    def test_nothing_serving_gives_None_rather_than_a_guess(self):
        """Inventing a unit to start is how the 04.09. defect did its damage:
        the run did not merely fail to restore production, it STARTED
        something."""
        self.stub_serving("")
        self.assertIsNone(speed_ab.unit())

    def test_two_servers_refuse_rather_than_pick_one(self):
        self.stub_serving("flashnext\nqwen36\n")
        self.assertIsNone(speed_ab.unit())

    def test_the_answer_is_cached_because_it_is_asked_again_after_the_stop(self):
        """The restart happens in a `finally` by which time nothing is
        serving. A second live lookup there would return None and silently
        leave production down."""
        self.stub_serving("flashnext\n")
        first = speed_ab.unit()
        self.stub_serving("")           # as it looks once production is stopped
        self.assertEqual(speed_ab.unit(), first)

    def test_no_llama_user_instance_is_named_in_either_suite(self):
        """The constant is gone; a new one must not grow back — in this file
        or in flag-ab.py, which drives the same machinery.

        Checked on the SYNTAX TREE and not on the text. A grep for
        `llama-user@` also hits the docstring above, which exists to explain
        why the constant is gone — a guard that goes red at its own
        explanation teaches the next person to delete the explanation.
        Docstrings are excluded by identity, not by pattern: every string
        literal that is the first statement of a module, class or function is
        prose by definition, and every other one is code."""
        import ast
        offenders = []
        for name in ("speed-ab.py", "flag-ab.py"):
            path = REPO / "bench" / "suites" / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            prose = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = getattr(node, "body", None) or []
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        prose.add(id(body[0].value))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and "llama-user@" in node.value
                        and "%s" not in node.value
                        and id(node) not in prose):
                    offenders.append("%s:%d  %r"
                                     % (name, node.lineno, node.value[:60]))
        self.assertEqual(offenders, [],
                         "a production unit is named in code again — derive it "
                         "from models.sh serving:\n  " + "\n  ".join(offenders))


class TestAFailedInvocationIsRecordedAndNotDropped(unittest.TestCase):
    """The defect of 04.09.2026, in the machinery both suites share.

    A flag-ab run whose six llama-bench invocations ALL failed to load the
    model wrote a RESULT.md with a header row and no data rows, a rounds.json
    with three empty objects, and exited 0. Everything the failure knew — the
    return code, the stderr saying `failed to load model` — was printed to a
    console nobody keeps and then dropped. The report read as a null result.

    bench/README.md's first rule is that a check which cannot fail is not a
    check; a REPORT that cannot say it failed is the same shape.
    """

    def setUp(self):
        speed_ab.reset_ledger()
        self.addCleanup(speed_ab.reset_ledger)
        self.said = []
        real = speed_ab.say
        speed_ab.say = self.said.append
        self.addCleanup(setattr, speed_ab, "say", real)

    def test_the_stderr_survives_into_the_ledger(self):
        speed_ab.record_attempt()
        speed_ab.record_failure(
            "reference", ["/bin/llama-bench", "-m", "/m.gguf"], 1,
            "ggml_backend_load: ok\nllama_bench: error: failed to load model\n")
        self.assertEqual(speed_ab.LEDGER["attempted"], 1)
        f, = speed_ab.LEDGER["failures"]
        self.assertEqual(f["arm"], "reference")
        self.assertEqual(f["returncode"], 1)
        self.assertIn("failed to load model", f["stderr_tail"])

    def test_the_quoted_line_is_the_error_and_not_the_first_line(self):
        """llama-bench's first stderr line is a backend load message. Quoting
        line 1 would quote noise in exactly the case this exists for."""
        self.assertEqual(
            speed_ab.first_error_line(
                "ggml_backend_load_all: loading\n"
                "llama_bench: error: failed to load model '/m.gguf'\n"),
            "llama_bench: error: failed to load model '/m.gguf'")

    def test_stderr_with_no_error_word_falls_back_to_the_last_line(self):
        """A crash ends at the last line. Better a real line than nothing —
        the full tail is in rounds.json either way."""
        self.assertEqual(speed_ab.first_error_line("one\ntwo\nthree\n"),
                         "three")
        self.assertEqual(speed_ab.first_error_line(""), "(no stderr)")

    def test_exit_0_with_unparseable_output_counts_as_a_failure(self):
        """The shape most likely to be read as `this arm simply had no rows`:
        llama-bench exits 0 and prints something that is not JSON."""
        d = make_build(tempfile.mkdtemp(), "build-rocm-unroll-b1",
                       "2026-09-04T10:00:00+02:00")
        self.addCleanup(shutil.rmtree, os.path.dirname(d), ignore_errors=True)
        b = os.path.join(d, "bin", "llama-bench")      # the stub exits 0, mute
        rows = speed_ab.bench(b, "/m.gguf", [0], 512, 64, [], arm="reference")
        self.assertEqual(rows, [])
        self.assertEqual(len(speed_ab.LEDGER["failures"]), 1)
        self.assertIn("unparseable",
                      speed_ab.LEDGER["failures"][0]["stderr_tail"])

    def test_the_recorded_paths_do_not_name_this_machine(self):
        """rounds.json is published beside RESULT.md. A model path in a
        failure's stderr is a home directory like any other."""
        home = os.path.expanduser("~")
        speed_ab.record_attempt()
        speed_ab.record_failure(
            "reference", [os.path.join(home, "llama.cpp", "llama-bench")], 1,
            "error: failed to load model '%s/models/x.gguf'" % home)
        f, = speed_ab.LEDGER["failures"]
        self.assertNotIn(home, f["stderr_tail"])
        self.assertNotIn(home, f["first_error_line"])
        self.assertNotIn(home, " ".join(f["argv"]))
        self.assertIn("@HOME@", f["stderr_tail"])


class TestTheTallyIsStatedWhetherOrNotAnythingFailed(unittest.TestCase):
    """A report that mentions failures ONLY when there are some cannot be
    told apart from a report written before it could mention them at all —
    which is what every report in bench/reports/ before 04.09.2026 is."""

    def setUp(self):
        speed_ab.reset_ledger()
        self.addCleanup(speed_ab.reset_ledger)
        real = speed_ab.say                 # record_failure also SAYS it
        speed_ab.say = lambda msg: None
        self.addCleanup(setattr, speed_ab, "say", real)

    def fail(self, arm, n=1):
        for _ in range(n):
            speed_ab.record_attempt()
            speed_ab.record_failure(arm, ["/x"], 1, "error: failed to load")

    def test_a_clean_run_says_so(self):
        for _ in range(4):
            speed_ab.record_attempt()
        self.assertEqual(speed_ab.failure_lines(),
                         ["all 4 llama-bench invocations succeeded."])

    def test_it_says_how_many_of_how_many(self):
        self.fail("variant", 2)
        speed_ab.record_attempt()
        text = " ".join(speed_ab.failure_lines())
        self.assertIn("2 of 3 llama-bench invocations FAILED", text)
        self.assertIn("error: failed to load", text)

    def test_it_names_which_arm_emptied_the_table(self):
        """The table is the INTERSECTION of the arms' rows: one arm producing
        nothing takes every row with it, and a reader of the short table
        otherwise cannot tell which column did it."""
        self.fail("variant", 2)
        speed_ab.record_attempt()
        self.assertIn("failed in: variant (2)",
                      " ".join(speed_ab.failure_lines()))

    def test_an_all_failed_run_says_the_table_is_empty_for_that_reason(self):
        self.fail("reference")
        self.fail("variant")
        self.assertIn("NOTHING was measured",
                      " ".join(speed_ab.failure_lines()))

    def test_the_markdown_form_quotes_the_error_line(self):
        """A compiler message with an asterisk or an underscore in it would
        otherwise be read as formatting by the markdown renderer."""
        self.fail("variant")
        md = " ".join(speed_ab.failure_lines(md=True))
        self.assertIn("`error: failed to load`", md)
        self.assertTrue(speed_ab.failure_blockquote().startswith("> **"))

    def test_a_clean_run_writes_no_blockquote(self):
        speed_ab.record_attempt()
        self.assertEqual(speed_ab.failure_blockquote(), "")


class TestAllFailedIsNotSuccess(unittest.TestCase):
    """Some cells failing is recorded and NOT fatal — bench/README.md, `a cell
    that fails is recorded rather than fatal`, which was paid for by three
    reports that lost `prefill-nospec` to a restore timeout. Every cell
    failing is not a measurement at all, and the exit code has to tell those
    two apart: on 04.09.2026 the all-failed run exited 0."""

    def setUp(self):
        speed_ab.reset_ledger()
        self.addCleanup(speed_ab.reset_ledger)
        real = speed_ab.say                 # record_failure also SAYS it
        speed_ab.say = lambda msg: None
        self.addCleanup(setattr, speed_ab, "say", real)

    def fail(self, n):
        for _ in range(n):
            speed_ab.record_attempt()
            speed_ab.record_failure("variant", ["/x"], 1, "error: boom")

    def test_all_failed(self):
        self.fail(3)
        self.assertTrue(speed_ab.every_invocation_failed())

    def test_some_failed_is_still_a_measurement(self):
        self.fail(2)
        speed_ab.record_attempt()
        self.assertFalse(speed_ab.every_invocation_failed())

    def test_nothing_attempted_is_not_all_failed(self):
        """A --dry-run attempts nothing. Reading that as `everything failed`
        would make the dry run exit non-zero."""
        self.assertFalse(speed_ab.every_invocation_failed())


class TestARunThatMeasuredNothingExitsNonZero(unittest.TestCase):
    """End to end, through the real command line, against a stub llama-bench
    that fails the way the real one did: `failed to load model`, exit 1.

    Driven as a subprocess because the exit code IS the subject — a caller
    reading `$?` is the reader this protects, and an in-process check of
    main()'s return value would not exercise the `sys.exit()` that carries it.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def build(self, name, body, cmake="-DCMAKE_BUILD_TYPE=Release"):
        d = make_build(self.tmp, name, "2026-09-04T10:00:00+02:00", cmake=cmake)
        for exe in ("llama-bench", "llama-server"):
            p = os.path.join(d, "bin", exe)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, 0o755)
        return d

    FAILS = ("#!/bin/sh\n"
             "echo 'ggml_backend_load_all: loading' >&2\n"
             "echo \"llama_bench: error: failed to load model\" >&2\n"
             "exit 1\n")

    UNROLL = "-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600"

    def run_suite(self, out):
        return subprocess.run(
            [sys.executable, SUITE,
             "--reference", "build-rocm-patched-r1",
             "--variant", "build-rocm-unroll-v1",
             "--model", "/nonexistent/m.gguf", "--depths", "0",
             "--prompt", "512", "--reps", "1",
             # nothing is stopped and nothing is started: this test must say
             # the same thing on a machine that is serving.
             "--keep-production", "--no-warmup", "--out", out],
            capture_output=True, text=True, timeout=300,
            env=dict(os.environ, LLAMA_SRC=self.tmp))

    def test_every_invocation_failing_is_not_exit_0(self):
        self.build("build-rocm-patched-r1", self.FAILS)
        self.build("build-rocm-unroll-v1", self.FAILS, cmake=self.UNROLL)
        out = os.path.join(self.tmp, "report")
        r = self.run_suite(out)
        self.assertNotEqual(r.returncode, 0,
                            "a run that measured nothing exited 0:\n" + r.stdout)
        md = read(os.path.join(out, "RESULT.md"))
        self.assertIn("2 of 2 llama-bench invocations FAILED", md)
        self.assertIn("failed to load model", md)
        rounds = json.loads(read(os.path.join(out, "rounds.json")))
        self.assertEqual(rounds["_failures"]["failed"], 2)
        self.assertIn("failed to load model",
                      rounds["_failures"]["detail"][0]["stderr_tail"])

    def test_a_run_that_measured_something_still_exits_0(self):
        """The other half of the rule: a failing cell is recorded, not fatal.
        Here the reference measures and the variant does not."""
        rows = ('[{"n_prompt":512,"n_gen":0,"n_depth":0,"avg_ts":100.0}]')
        self.build("build-rocm-patched-r1",
                   "#!/bin/sh\necho '%s'\n" % rows)
        self.build("build-rocm-unroll-v1", self.FAILS, cmake=self.UNROLL)
        out = os.path.join(self.tmp, "report")
        r = self.run_suite(out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        md = read(os.path.join(out, "RESULT.md"))
        self.assertIn("1 of 2 llama-bench invocations FAILED", md)
        self.assertIn("failed in: variant (1)", md)

    def test_a_clean_run_reports_the_tally_too(self):
        rows = ('[{"n_prompt":512,"n_gen":0,"n_depth":0,"avg_ts":100.0}]')
        body = "#!/bin/sh\necho '%s'\n" % rows
        self.build("build-rocm-patched-r1", body)
        self.build("build-rocm-unroll-v1", body, cmake=self.UNROLL)
        out = os.path.join(self.tmp, "report")
        r = self.run_suite(out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        md = read(os.path.join(out, "RESULT.md"))
        self.assertIn("all 2 llama-bench invocations succeeded", md)
        self.assertNotIn("> **", md)


if __name__ == "__main__":
    unittest.main()
