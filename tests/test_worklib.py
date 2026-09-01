"""bench/worklib.py — the shared bench core, proven on stub jobs.

The refactor's contract: what the three siblings shared now has ONE
implementation, and its behavior is pinned here without a GPU — the
incremental write after every rep, the partial flag's disappearance, the
distinct-output count, the medians, the exit code, the fence refusal, and
the one build_stamp_of. The end-to-end proof is a fenced qwen3-tts re-run
whose determinism hash must match the pre-refactor baseline (c7778614…,
report 2026-09-01_0527) — recorded in the migration commit.
"""
import contextlib
import io
import json
import os
import sys
import unittest

import common

sys.path.insert(0, str(common.REPO / "bench"))
sys.path.insert(0, str(common.REPO / "setup" / "lib"))

import worklib                                                # noqa: E402
import budget                                                 # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.profile = os.path.join(self.tmp, "stub.env")
        with open(self.profile, "w") as fh:
            fh.write("WORKLOAD_TITLE=stub\nWORKLOAD_KIND=image\n"
                     "WORKLOAD_MODE=batch\n"
                     "WORKLOAD_CMD=/bin/true --flag\n"
                     "WORKLOAD_PROMPT=a stub prompt\n"
                     "WORKLOAD_FILES=hf-cache\n")

    def report(self, reps=3):
        r = worklib.BenchReport("stub", self.profile, reps, note="t")
        # Reports belong under bench/reports in production; a test points
        # the same object at scratch space instead — and swallows the
        # narration, so the gate output stays a test log, not a bench log.
        r.dest = os.path.join(self.tmp, "out")
        os.makedirs(r.dest, exist_ok=True)
        self._silence = contextlib.redirect_stdout(io.StringIO())
        self._silence.__enter__()
        self.addCleanup(lambda: self._silence.__exit__(None, None, None))
        return r


class TestTheRepRunner(Base):
    def test_partial_is_on_disk_after_every_rep_and_gone_at_the_end(self):
        report = self.report(reps=2)
        seen_partial = []

        def do_rep(r):
            if r == 2:
                with open(os.path.join(report.dest, "result.json")) as fh:
                    seen_partial.append(json.load(fh).get("partial"))
            return {"ok": True, "sha256": "h%d" % r}

        report.run_reps(do_rep, describe=lambda rep: "x")
        rc = report.finalize()
        self.assertEqual(seen_partial, [True],
                         "rep 2 must find rep 1's partial record on disk — "
                         "a timeout between reps costs the remaining reps, "
                         "never the measured ones")
        with open(os.path.join(report.dest, "result.json")) as fh:
            final = json.load(fh)
        self.assertNotIn("partial", final)
        self.assertEqual(rc, 0)

    def test_distinct_outputs_counts_hashes_not_reps(self):
        report = self.report(reps=3)
        report.run_reps(lambda r: {"ok": True, "sha256": "same"},
                        describe=lambda rep: "x")
        report.finalize()
        with open(os.path.join(report.dest, "result.json")) as fh:
            self.assertEqual(json.load(fh)["distinct_outputs"], 1)

    def test_a_broken_rep_costs_the_exit_code_not_the_record(self):
        report = self.report(reps=2)
        report.run_reps(lambda r: {"ok": r == 1, "sha256": "h%d" % r},
                        describe=lambda rep: "x")
        rc = report.finalize()
        self.assertEqual(rc, 1)
        with open(os.path.join(report.dest, "result.json")) as fh:
            final = json.load(fh)
        self.assertEqual(len(final["reps"]), 2)
        self.assertIn("median_seconds", final,
                      "the sound rep still earns its summary")

    def test_post_runs_after_timing_so_it_can_derive_rates(self):
        report = self.report(reps=1)
        got = []
        report.run_reps(lambda r: {"ok": True},
                        describe=lambda rep: "x",
                        post=lambda rep: got.append("seconds" in rep))
        self.assertEqual(got, [True])

    def test_the_clock_stops_at_the_subprocess_not_at_the_judge(self):
        """Both re-reviews converged on this independently (01.09.2026):
        run_reps timing the whole do_rep silently moved the measurement
        boundary — `seconds` then included hashing and the checker verdict,
        ~0.3-1 % one-sided inflation on video clips, invisible to the hash
        verification by construction. The bench stamps its own seconds via
        timed_run (clock around the subprocess ALONE); the runner's stamp
        is only a fallback for stubs and may never overwrite it."""
        import time as _time
        report = self.report(reps=1)

        def do_rep(r):
            rc, secs = worklib.timed_run(
                ["/bin/true"], os.path.join(report.dest, "rep%d.log" % r))
            rep = {"ok": True, "exit": rc, "seconds": secs}
            _time.sleep(0.15)          # the "judge" — outside the clock
            return rep

        report.run_reps(do_rep, describe=lambda rep: "x")
        self.assertLess(report.reps[0]["seconds"], 0.1,
                        "a slow judge inflated the wall time — the "
                        "measurement boundary moved again")

    def test_a_stub_without_its_own_clock_still_gets_one(self):
        report = self.report(reps=1)
        report.run_reps(lambda r: {"ok": True}, describe=lambda rep: "x")
        self.assertIn("seconds", report.reps[0])

    def test_metadata_names_machine_build_and_unexpanded_argv(self):
        report = self.report(reps=1)
        report.run_reps(lambda r: {"ok": True, "sha256": "h"},
                        describe=lambda rep: "x")
        report.finalize()
        with open(os.path.join(report.dest, "result.json")) as fh:
            final = json.load(fh)
        for key in ("machine", "build_stamp", "argv", "prompt", "binary"):
            self.assertIn(key, final)
        self.assertIn("gfx", final["machine"])


class TestTheFence(Base):
    def test_refuses_beside_a_serving_llama(self):
        old = budget.server_pid
        try:
            budget.server_pid = lambda: "1234"
            self.assertTrue(worklib.fence_refusal("x"))
            budget.server_pid = lambda: None
            self.assertFalse(worklib.fence_refusal("x"))
        finally:
            budget.server_pid = old


class TestBuildStampOf(Base):
    def test_finds_a_stamp_beside_and_above_the_binary(self):
        build = os.path.join(self.tmp, "build-vulkan-abc")
        os.makedirs(os.path.join(build, "bin"))
        with open(os.path.join(build, ".build-stamp"), "w") as fh:
            fh.write("build_id=vulkan-abc\n")
        deep = os.path.join(build, "bin", "tool")
        flat = os.path.join(build, "tool")
        for p in (deep, flat):
            with open(p, "w") as fh:
                fh.write("x")
        # Both layouts read the SAME stamp — the divergence the two copies
        # had at birth (two vs three search levels).
        self.assertEqual(worklib.build_stamp_of(deep)["build_id"],
                         "vulkan-abc")
        self.assertEqual(worklib.build_stamp_of(flat)["build_id"],
                         "vulkan-abc")

    def test_no_stamp_is_an_empty_dict_not_an_error(self):
        p = os.path.join(self.tmp, "tool")
        with open(p, "w") as fh:
            fh.write("x")
        self.assertEqual(worklib.build_stamp_of(p), {})


if __name__ == "__main__":
    unittest.main()
