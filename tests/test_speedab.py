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
import os, shutil, subprocess, sys, tempfile, unittest

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


if __name__ == "__main__":
    unittest.main()
