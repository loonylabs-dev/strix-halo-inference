"""unroll-flag — the parts that decide WHICH binaries get compared.

A measurement suite that picks the wrong pair does not fail. It runs, prints
a table, and attributes the difference to the flag — which is the failure mode
this repository keeps meeting, and the reason the picking is tested rather
than the plumbing.

One of these tests exists because the bug had already happened: newest_unroll()
returned a full PATH, runlib.resolve_binary() takes a path OR a directory name
OR a build id, and resolved the path against $LLAMA_SRC — producing
`~/llama.cpp/llama-bench`, a file that does not exist. That one announced
itself. The version where it silently finds the WRONG build would not.
"""
import os, shutil, subprocess, sys, tempfile, unittest

import common

REPO = common.REPO
SUITE = str(REPO / "bench" / "suites" / "unroll-flag.py")
# The suite imports bench/run.py as `run`, so that has to be importable
# before it is loaded — the hyphen in its own filename is why it cannot
# simply be imported by name.
sys.path.insert(0, str(REPO / "bench"))
unroll_flag = common.load(SUITE, "unroll_flag")


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
        got = unroll_flag.newest_unroll()
        self.assertEqual(got, "build-rocm-unroll-b1")
        self.assertNotIn(os.sep, got)

    def test_it_takes_the_newest_by_stamp_not_by_name(self):
        """Directory order is arbitrary and build ids do not sort by age."""
        make_build(self.tmp, "build-rocm-unroll-zzz", "2026-08-01T10:00:00+02:00")
        make_build(self.tmp, "build-rocm-unroll-aaa", "2026-08-30T10:00:00+02:00")
        self.assertEqual(unroll_flag.newest_unroll(), "build-rocm-unroll-aaa")

    def test_it_ignores_the_other_families(self):
        make_build(self.tmp, "build-rocm-patched-b1", "2026-08-31T00:00:00+02:00")
        make_build(self.tmp, "build-rocm-unpatched-b1", "2026-08-31T00:00:00+02:00")
        self.assertIsNone(unroll_flag.newest_unroll())

    def test_nothing_there_is_none_rather_than_a_guess(self):
        self.assertIsNone(unroll_flag.newest_unroll())


class TestTheStampIsReadFromBesideTheBinary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_it_finds_the_stamp_two_levels_up_from_bin(self):
        d = make_build(self.tmp, "build-rocm-unroll-b1", "2026-08-31T00:00:00+02:00",
                       cmake="-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600")
        st = unroll_flag.stamp_beside(os.path.join(d, "bin", "llama-bench"))
        self.assertIn("amdgpu-unroll", st["cmake"])
        self.assertEqual(st["patched"], "yes")

    def test_a_missing_stamp_is_empty_rather_than_an_exception(self):
        """A build without a stamp predates them. The suite must be able to
        SAY that, which it cannot do from a traceback."""
        self.assertEqual(
            unroll_flag.stamp_beside("/nonexistent/bin/llama-bench"), {})


class TestItRefusesAPairThatCannotAnswerTheQuestion(unittest.TestCase):
    """The comparison is only worth running if exactly one arm carries the
    flag. Both or neither is a run that produces a table meaning nothing."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_suite(self, ref, unroll):
        return subprocess.run(
            [sys.executable, SUITE, "--reference", ref, "--unroll", unroll,
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

    def test_it_refuses_when_the_unroll_arm_does_not_carry_it(self):
        make_build(self.tmp, "build-rocm-patched-b1", "2026-08-30T10:00:00+02:00")
        make_build(self.tmp, "build-rocm-unroll-b1", "2026-08-30T11:00:00+02:00")
        r = self.run_suite("build-rocm-patched-b1", "build-rocm-unroll-b1")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("does NOT carry the flag", r.stdout + r.stderr)

    def test_a_differing_commit_is_called_out(self):
        """Not fatal — sometimes the reference IS an older build — but it must
        not pass in silence, because the difference then carries more than
        the flag."""
        flag = "-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600"
        make_build(self.tmp, "build-rocm-patched-b1", "2026-08-30T10:00:00+02:00",
                   commit="a" * 40)
        make_build(self.tmp, "build-rocm-unroll-b1", "2026-08-30T11:00:00+02:00",
                   cmake=flag, commit="b" * 40)
        r = self.run_suite("build-rocm-patched-b1", "build-rocm-unroll-b1")
        self.assertIn("NOT the same commit", r.stdout + r.stderr)


class TestTheSummaryArithmetic(unittest.TestCase):
    def test_the_median_is_the_middle_not_the_mean(self):
        """One slow round — a background job, a thermal dip — must not move
        the answer the way a mean would."""
        self.assertEqual(unroll_flag.median([10.0, 11.0, 100.0]), 11.0)
        self.assertEqual(unroll_flag.median([10.0, 20.0]), 15.0)
        self.assertEqual(unroll_flag.median([]), 0.0)

    def test_rows_are_matched_across_arms_by_shape_and_depth(self):
        """pp512@d0 must never be compared against pp512@d65536."""
        a = {"n_prompt": 512, "n_gen": 0, "n_depth": 32768}
        b = {"n_prompt": 512, "n_gen": 0, "n_depth": 0}
        self.assertNotEqual(unroll_flag.key_of(a), unroll_flag.key_of(b))
        self.assertEqual(unroll_flag.label_of(unroll_flag.key_of(a)),
                         "pp512 @ d32768")
        self.assertEqual(
            unroll_flag.label_of(unroll_flag.key_of(
                {"n_prompt": 0, "n_gen": 128, "n_depth": 0})), "tg128 @ d0")


if __name__ == "__main__":
    unittest.main()
