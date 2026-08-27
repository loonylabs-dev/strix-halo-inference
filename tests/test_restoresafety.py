"""restore-safety — a cell that fails is a result, not the end of the run.

The suite measures six cells and, three times running, reported four. The
restore in `busy-nospec` hangs — that is a registered defect,
`slot-restore-hangs-busy` — and its 300 s timeout propagated out of the cell,
out of main(), and into the finally that restores production. Everything
after it was never measured: `prefill-nospec` has no reading on the patched
build at all, on 26.08. twice and on 27.08. once.

Nothing about those runs looked wrong. The verdict prints `?` for a cell that
was never reached and `?` for a cell nobody asked for, so a truncated run and
a deliberately narrow one are the same report. That is this repository's
recurring shape one level up: it ran, it exited, and the thing it was for did
not happen.

What is pinned here is therefore not "the timeout is handled" — it is that a
cell CANNOT end the run:

  * a body that raises is written into its own cell and the next cell runs;
  * the one call known to hang records that it did not come back, with how
    long it took, instead of raising past everything;
  * a probe that does not answer is recorded VERBATIM — '<TimeoutError…>' is
    a different finding from '////' and a boolean cannot tell them apart;
  * every server start goes through the runner, so no future cell can be
    written that starts a server outside the thing guaranteeing its teardown.

Nothing here talks to a server or to a GPU.
"""
import ast, contextlib, io, unittest

import common


@contextlib.contextmanager
def quiet():
    """The suite prints what it recorded, which is right when it runs and
    noise when a test drives it. A CI log is what a stranger reads at the
    moment they already have a problem — 566 warnings deep was this
    repository's own lesson on 27.08."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield

RS = common.load("bench/suites/restore-safety.py", "restore_safety")
SOURCE = (common.REPO / "bench/suites/restore-safety.py").read_text(
    encoding="utf-8")
TREE = ast.parse(SOURCE)


def _func(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("no function %r in restore-safety.py" % name)


def _calls_to(name, inside=None):
    """Every call to `name`, as (call node, enclosing function name)."""
    out = []
    for fn in ast.walk(TREE):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            got = (f.id if isinstance(f, ast.Name) else
                   ("%s.%s" % (f.value.id, f.attr)
                    if isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Name) else None))
            if got == name:
                out.append((node, fn.name))
    return out


class TestACellCannotEndTheRun(unittest.TestCase):
    """The property the three truncated reports were missing."""

    def setUp(self):
        self.stopped = []
        self.started = []
        # A fake server object: run_cell only ever hands it to stop_server.
        RS.start = lambda argv, log: self.started.append(log) or "proc"
        RS.sweep.stop_server = self.stopped.append

    def test_a_body_that_raises_is_recorded_and_the_next_cell_still_runs(self):
        results, saves = {}, []
        with quiet():
            RS.run_cell("busy-nospec", results, lambda: saves.append(1),
                        [], "/dev/null",
                        lambda r: (_ for _ in ()).throw(TimeoutError("300 s")))
            RS.run_cell("prefill-nospec", results, lambda: saves.append(1),
                        [], "/dev/null",
                        lambda r: r.update(clean=True))

        self.assertFalse(results["busy-nospec"]["clean"])
        self.assertTrue(results["busy-nospec"]["aborted"])
        self.assertIn("TimeoutError", results["busy-nospec"]["error"])
        self.assertTrue(results["prefill-nospec"]["clean"],
                        "the cell AFTER the failing one is the one three "
                        "reports are missing")
        self.assertEqual(len(saves), 2,
                         "result.json is written after every cell, so a run "
                         "that is killed keeps what it measured")

    def test_the_server_is_torn_down_even_when_the_body_raises(self):
        """A cell that leaves its server behind holds the GPU and the port,
        and the next cell then fails for a reason that is not the build."""
        with quiet():
            RS.run_cell("c", {}, lambda: None, [], "/dev/null",
                        lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertEqual(self.stopped, ["proc"])

    def test_a_server_that_never_comes_up_is_a_cell_not_a_traceback(self):
        """runlib.start_server raises SystemExit — for a memory guard that
        refuses as much as for a server that dies. SystemExit is not an
        Exception, so catching only Exception would have left this one
        ending the run."""
        def boom(argv, log):
            raise SystemExit("server exits immediately")
        RS.start = boom
        results = {}
        with quiet():
            RS.run_cell("idle-spec", results, lambda: None, [], "/dev/null",
                        lambda r: r.update(clean=True))
        self.assertTrue(results["idle-spec"]["aborted"])
        self.assertEqual(results["idle-spec"]["phase"], "start")
        self.assertIn("SystemExit", results["idle-spec"]["error"])


class TestTheHangIsAMeasurement(unittest.TestCase):
    """`slot-restore-hangs-busy` is the finding, not the accident."""

    def test_a_restore_that_does_not_return_is_recorded_with_its_duration(self):
        RS.slot_action = lambda *a, **k: (_ for _ in ()).throw(
            TimeoutError("timed out"))
        r = {}
        with quiet():
            self.assertFalse(RS.timed_restore(r, 0, "exp-nospec.bin"))
        self.assertIs(r["restore_returned"], False)
        self.assertIn("TimeoutError", r["restore_error"])
        self.assertIsInstance(r["restore_seconds"], float)
        self.assertEqual(r["phase"], "restore",
                         "the phase is where it died, which is the difference "
                         "between the restore and the probes after it")

    def test_a_restore_that_returns_says_so(self):
        RS.slot_action = lambda *a, **k: {"ok": True}
        r = {}
        self.assertTrue(RS.timed_restore(r, 0, "exp-spec.bin"))
        self.assertIs(r["restore_returned"], True)
        self.assertNotIn("restore_error", r)

    def test_the_probe_budget_shrinks_only_after_the_cell_is_already_lost(self):
        """The healthy bound must not change: a probe queues behind two
        2,500-token generations, and without speculation those are minutes.
        Shortening that would turn a SLOW server into a dirty one."""
        self.assertEqual(RS.probe_timeout_for({"restore_returned": True}),
                         RS.PROBE_TIMEOUT_OK)
        self.assertEqual(RS.probe_timeout_for({"restore_returned": False}),
                         RS.PROBE_TIMEOUT_AFTER_FAILURE)
        self.assertLess(RS.PROBE_TIMEOUT_AFTER_FAILURE, RS.PROBE_TIMEOUT_OK)


class TestProbesAreRecordedVerbatim(unittest.TestCase):
    def test_a_probe_that_raises_is_written_down_rather_than_thrown(self):
        RS.probe = lambda timeout=0: (_ for _ in ()).throw(
            TimeoutError("read timed out"))
        ok, texts = RS.probes_clean()
        self.assertFalse(ok)
        self.assertEqual(len(texts), 1,
                         "two more waits of the same length only add to how "
                         "long production is down")
        self.assertIn("TimeoutError", texts[0])

    def test_a_dirty_answer_and_a_missing_one_do_not_read_the_same(self):
        RS.probe = lambda timeout=0: "////////"
        ok, dirty = RS.probes_clean()
        self.assertFalse(ok)
        self.assertEqual(len(dirty), 3)
        RS.probe = lambda timeout=0: (_ for _ in ()).throw(OSError("closed"))
        _, gone = RS.probes_clean()
        self.assertNotEqual(dirty, gone)

    def test_three_good_answers_are_clean(self):
        RS.probe = lambda timeout=0: "391"
        ok, texts = RS.probes_clean()
        self.assertTrue(ok)
        self.assertEqual(texts, ["391", "391", "391"])


class TestTheVerdictHasThreeStates(unittest.TestCase):
    """A cell that was not measured must read like neither a pass nor a
    failure. The old verdict had two states for three situations."""

    def test_not_run_pass_fail_and_skipped_are_all_distinguishable(self):
        words = {
            "not run": RS.verdict_line(None),
            "clean": RS.verdict_line({"clean": True}),
            "skipped": RS.verdict_line({"clean": None, "skipped": True,
                                        "error": "no state to restore"}),
            "hung": RS.verdict_line({"clean": False, "restore_returned": False,
                                     "restore_seconds": 300.0}),
        }
        self.assertEqual(len(set(words.values())), 4, words)
        self.assertIn("SKIPPED", words["skipped"])
        self.assertIn("CLEAN", words["clean"])
        self.assertIn("300", words["hung"])

    def test_a_dirty_cell_says_why_without_opening_result_json(self):
        line = RS.verdict_line({"clean": False, "probes": ["////", "x", "y"]})
        self.assertIn("DIRTY", line)
        self.assertIn("////", line)


class TestABoundMustNotBeMistakenForAFinding(unittest.TestCase):
    """The cell that failed four times was measuring its own timeout.

    A restore QUEUES BEHIND THE SLOT IT TARGETS. `busy-nospec` fills slot 0
    with a 2,500-token generation and then asks for a restore into it with a
    300 s bound. Measured 27.08. on b10631: that generation runs at 7.45 t/s
    and takes 335.5 s, so the client gave up 35 s early — three runs running,
    filed as `slot-restore-hangs-busy`. The same cell WITH speculation runs
    the same generation in 73.7 s and the restore returns at 68.8. The
    asymmetry the defect entry recorded as unexplained is the drafter.

    So the suite has to be able to say which of the two was shorter. It is
    not that the bound is wrong — it is that a bound below the work it waits
    for measures the bound, and a report that cannot say so reads as a
    server that wedged.
    """

    def test_a_timeout_below_the_work_says_so(self):
        r = {"restore_returned": False, "restore_timeout": 300.0}
        with quiet():
            # the longest one SECOND on purpose: with it first, held[0] and
            # max(held) are the same value and the assertion below cannot
            # tell them apart. It could not, and said so under mutation.
            RS.note_what_blocked(r, [{"seconds": 334.4}, {"seconds": 335.5}])
        self.assertEqual(r["blocking_seconds"], 335.5,
                         "the LONGEST holder, not the first")
        self.assertTrue(r["timeout_was_shorter_than_the_work"])
        line = RS.verdict_line(dict(r, clean=False, restore_seconds=300.1))
        self.assertIn(str(round(r["blocking_seconds"])), line)
        self.assertIn("shorter of the two", line)

    def test_a_timeout_above_the_work_is_a_real_finding_and_stays_silent(self):
        """If the slot freed and the restore still did not come back, the
        bound is not the explanation and must not be offered as one."""
        r = {"restore_returned": False, "restore_timeout": 900.0}
        with quiet():
            RS.note_what_blocked(r, [{"seconds": 335.5}])
        self.assertNotIn("timeout_was_shorter_than_the_work", r)
        self.assertNotIn("shorter of the two",
                         RS.verdict_line(dict(r, clean=False,
                                              restore_seconds=900.2)))

    def test_a_restore_that_returned_is_never_explained_away(self):
        r = {"restore_returned": True, "restore_timeout": 300.0}
        with quiet():
            RS.note_what_blocked(r, [{"seconds": 999.0}])
        self.assertNotIn("timeout_was_shorter_than_the_work", r)

    def test_the_workers_record_how_long_they_held_the_server(self):
        """Without this the comparison has nothing to compare against, and
        the field would be silently absent rather than wrong."""
        import inspect
        for fn in (RS.long_generation, RS.big_prefill):
            src = inspect.getsource(fn)
            self.assertIn('box["seconds"]', src, fn.__name__)
            self.assertIn("finally:", src,
                          "%s must record its duration even when the request "
                          "fails — a worker that raised still held the slot"
                          % fn.__name__)

    def test_the_bound_is_recorded_with_the_numbers_it_produced(self):
        """A report whose bound is not in it cannot be compared with one
        taken under a different bound."""
        self.assertEqual(RS.RESTORE_TIMEOUT_DEFAULT, 300,
                         "changing the default silently rewrites what every "
                         "earlier report means")
        r = {}
        RS.slot_action = lambda *a, **k: (_ for _ in ()).throw(OSError("x"))
        with quiet():
            RS.timed_restore(r, 0, "exp.bin")
        self.assertEqual(r["restore_timeout"], RS.RESTORE_TIMEOUT)


class TestNoCellStartsAServerOnItsOwn(unittest.TestCase):
    """The structural half, and the durable one.

    A cell is a server start plus a body. If a future cell starts one
    OUTSIDE run_cell, everything above is bypassed for that cell and nothing
    would say so — which is exactly how the suite acquired six inline cells
    that each had to remember their own teardown.
    """

    def test_every_server_start_goes_through_the_runner(self):
        calls = _calls_to("start")
        self.assertTrue(calls, "no call to start() found — this test would "
                               "pass by finding nothing")
        for _, where in calls:
            self.assertEqual(where, "run_cell",
                             "start() is called from %r" % where)

    def test_every_teardown_goes_through_the_runner(self):
        calls = _calls_to("sweep.stop_server")
        self.assertTrue(calls, "no call to sweep.stop_server() found")
        for _, where in calls:
            self.assertEqual(where, "run_cell",
                             "sweep.stop_server() is called from %r" % where)

    def test_the_restore_is_only_ever_made_through_timed_restore(self):
        """The one call known to hang. Reached directly, it raises past the
        cell again and the three truncated reports come back."""
        found = 0
        for node, where in _calls_to("slot_action"):
            args = [a.value for a in node.args
                    if isinstance(a, ast.Constant)]
            if "restore" in args:
                found += 1
                self.assertEqual(where, "timed_restore",
                                 "a restore is made in %r" % where)
        self.assertEqual(found, 1,
                         "expected exactly one restore call site, found %d"
                         % found)


class TestTheReportSaysWhichBuildProducedIt(unittest.TestCase):
    """--backend names a ROLE. Two runs of two builds under one label are two
    different binaries, and until 27.08. result.json said so nowhere."""

    def test_a_build_can_be_named_without_moving_the_production_symlink(self):
        import os
        src = RS.LLAMA_SRC
        default = RS.resolve_binary("rocm-patched", None)
        self.assertEqual(default, RS.BINARIES["rocm-patched"])
        self.assertEqual(RS.resolve_binary("rocm-patched", "/bin/sh"), "/bin/sh")
        self.assertNotEqual(
            os.path.realpath(default),
            os.path.realpath(RS.resolve_binary("rocm-patched", "/bin/sh")))
        self.assertTrue(src)

    def test_a_bare_build_directory_name_resolves(self):
        """`--binary rocm` — the stock build, and the obvious thing to type.
        The first version tried `<spec>` and `build-rocm-patched-<spec>` and
        not `build-<spec>`, so it resolved to nothing and the run died before
        production was even stopped. The help text had promised it."""
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            for name in ("build-rocm", "build-rocm-patched-b1", "loose"):
                os.makedirs(os.path.join(d, name, "bin"))
                b = os.path.join(d, name, "bin", "llama-server")
                open(b, "w").close()
                os.chmod(b, 0o755)
            old = RS.LLAMA_SRC
            self.addCleanup(setattr, RS, "LLAMA_SRC", old)
            RS.LLAMA_SRC = d
            for spec, want in (("rocm", "build-rocm"),
                               ("b1", "build-rocm-patched-b1"),
                               ("loose", "loose")):
                self.assertEqual(RS.resolve_binary("rocm-patched", spec),
                                 os.path.join(d, want, "bin", "llama-server"),
                                 spec)

    def test_an_unknown_build_fails_loudly_and_names_what_it_tried(self):
        with self.assertRaises(SystemExit) as e:
            RS.resolve_binary("rocm-patched", "no-such-build-id")
        self.assertIn("no-such-build-id", str(e.exception))

    def test_a_stamp_is_believed_only_if_it_matches_the_binary(self):
        """A .build-stamp is a file BESIDE the binary, not a property of it.
        Believing a stale one attributes a measurement to the wrong commit,
        which is the single error a build comparison must not make."""
        import os, tempfile, textwrap
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "bin"))
            binary = os.path.join(d, "bin", "llama-server")
            with open(binary, "w") as f:
                f.write("#!/bin/sh\necho 'version: 0.3.0-dev "
                        "(build 270, commit 545de45e5)'\n")
            os.chmod(binary, 0o755)
            with open(os.path.join(d, ".build-stamp"), "w") as f:
                f.write(textwrap.dedent("""\
                    build_id=someone-elses-build
                    patch_commit=deadbeef123456789
                    """))
            with quiet():
                meta = RS.provenance(binary)
            self.assertFalse(meta["stamp_matches_binary"])
            self.assertEqual(meta["build_id"], meta["build_from_binary"],
                             "a stamp that does not match the binary must "
                             "not name the report")

            with open(os.path.join(d, ".build-stamp"), "w") as f:
                f.write("build_id=b10631-18-gc1dcd9825\n"
                        "patch_commit=545de45e5d25c4839a55b860727b24a81bdee089\n")
            meta = RS.provenance(binary)
            self.assertTrue(meta["stamp_matches_binary"])
            self.assertEqual(meta["build_id"], "b10631-18-gc1dcd9825")


if __name__ == "__main__":
    unittest.main()
