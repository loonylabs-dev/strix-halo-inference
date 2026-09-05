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
        src = RS.runlib.LLAMA_SRC
        default = RS.resolve_binary(None, RS.BINARIES["rocm-patched"])
        self.assertEqual(default, RS.BINARIES["rocm-patched"])
        self.assertEqual(RS.resolve_binary("/bin/sh"), "/bin/sh")
        self.assertNotEqual(
            os.path.realpath(default),
            os.path.realpath(RS.resolve_binary("/bin/sh")))
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
            old = RS.runlib.LLAMA_SRC
            self.addCleanup(setattr, RS.runlib, "LLAMA_SRC", old)
            RS.runlib.LLAMA_SRC = d
            for spec, want in (("rocm", "build-rocm"),
                               ("b1", "build-rocm-patched-b1"),
                               ("loose", "loose")):
                self.assertEqual(RS.resolve_binary(spec),
                                 os.path.join(d, want, "bin", "llama-server"),
                                 spec)

    def test_an_unknown_build_fails_loudly_and_names_what_it_tried(self):
        with self.assertRaises(SystemExit) as e:
            RS.resolve_binary("no-such-build-id")
        self.assertIn("no-such-build-id", str(e.exception))

    def test_the_family_and_not_the_role_names_the_report(self):
        """--backend is a ROLE and --binary overrides it, so with both in play
        the directory name said `rocm-patched` for a build stamped
        `patched=no`. The file was honest and its name was not, which is the
        worse half: a name is what a reader sees first and what a report is
        cited by."""
        import ast, inspect
        src = inspect.getsource(RS.main)
        self.assertIn('meta["stamp"]["family"]', src)
        self.assertIn("stamp_matches_binary", src)
        # and the label, not a.backend, is what reaches the path
        tree = ast.parse(src.lstrip())
        joins = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "join"]
        names = {a.id for j in joins for a in ast.walk(j)
                 if isinstance(a, ast.Name)}
        self.assertIn("label", names,
                      "the report path must be built from the label")

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

    def test_an_unpatched_stamp_is_checked_against_the_right_field(self):
        """A patched build is built from the patch branch's tip; an unpatched
        one from the upstream commit, and carries patch_commit=none. Comparing
        patch_commit in both cases made every unpatched build fail against the
        literal string "none" — a false negative that then named three report
        directories `rocm-patched` for builds stamped `patched=no`.

        It failed SAFE, which is the right direction and not an excuse: a
        check that refuses a correct stamp teaches its reader to ignore it."""
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "bin"))
            binary = os.path.join(d, "bin", "llama-server")
            with open(binary, "w") as f:
                f.write("#!/bin/sh\necho 'version: 0.3.0-dev "
                        "(build 269, commit c1dcd9825)'\n")
            os.chmod(binary, 0o755)
            with open(os.path.join(d, ".build-stamp"), "w") as f:
                f.write("build_id=b10631-18-gc1dcd9825\n"
                        "family=rocm-unpatched\npatched=no\n"
                        "upstream_commit=c1dcd98252a44f1712b1a887ed8085e87a1"
                        "ae435\npatch_commit=none\n")
            meta = RS.provenance(binary)
            self.assertTrue(meta["stamp_matches_binary"],
                            "an unpatched stamp is identified by its UPSTREAM "
                            "commit, not by patch_commit=none")
            self.assertEqual(meta["build_id"], "b10631-18-gc1dcd9825")
            self.assertEqual(meta["stamp"]["family"], "rocm-unpatched")

    def test_an_unpatched_stamp_from_a_different_build_is_still_refused(self):
        """The positive control for the fix: reading the other field must not
        mean reading no field at all."""
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "bin"))
            binary = os.path.join(d, "bin", "llama-server")
            with open(binary, "w") as f:
                f.write("#!/bin/sh\necho 'version: 0.3.0-dev "
                        "(build 269, commit c1dcd9825)'\n")
            os.chmod(binary, 0o755)
            with open(os.path.join(d, ".build-stamp"), "w") as f:
                f.write("build_id=somebody-elses\npatched=no\n"
                        "upstream_commit=deadbeef1234567890\n"
                        "patch_commit=none\n")
            with quiet():
                meta = RS.provenance(binary)
            self.assertFalse(meta["stamp_matches_binary"])
            self.assertEqual(meta["build_id"], meta["build_from_binary"])


def _value_of(argv, flag):
    """The value following `flag`, or None if the flag is absent."""
    for i, tok in enumerate(argv):
        if tok == flag:
            return argv[i + 1] if i + 1 < len(argv) else ""
    return None


class TwoSlotsAreWhatThisSuiteIs(unittest.TestCase):
    """A profile's `-np 1` must not reach the server this suite starts.

    The built-in qwen38 argv carries `-np 2` because every cell here needs a
    second slot: `parallel` is two concurrent prefills and nothing else — the
    bare trigger of llama.cpp #27579 — and the restore cells were measured
    against a second slot when slot-restore-poison was found.

    `--env` arrived on 02.09.2026 so flashnext could be asked the same
    question, and it read the profile's argv through unchanged. Every
    production profile in this repo carries `-np 1`, because that is the
    mitigation the defect entries prescribe. So the flag that exists to point
    the instrument at another model also disarmed it: the suite would start
    ONE slot, run two prefills through it sequentially, find nothing, and
    write `clean` — for a configuration in which the defect cannot appear.

    That is the same shape slot-corruption.py carries in its own docstring as
    a paid-for lesson: a check that cannot fail, in the instrument for the
    defect the mitigation exists for. It could not be found by reading either
    file alone; it lives in what the two disagree about.
    """

    def _profile(self, d, args):
        import os
        p = os.path.join(d, "sample.env")
        with open(p, "w", encoding="utf-8") as f:
            f.write("LLAMA_ARGS=%s\n" % args)
        return p

    def test_a_profile_that_says_one_slot_does_not_get_one_slot(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._profile(
                d, "--alias qwen36 -m /models/x.gguf -ngl 999 -fa on "
                   "-c 262144 -np 1 -cram 32768")
            with quiet():
                base, _ = RS.base_from_profile(p)
            self.assertEqual(
                _value_of(base, "-np"), "2",
                "a one-slot server cannot produce the defect this suite "
                "measures, so `clean` from it would be a reading of nothing")

    def test_a_profile_with_no_slot_count_at_all_still_gets_two(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._profile(d, "--alias m -m /models/x.gguf -c 65536")
            with quiet():
                base, _ = RS.base_from_profile(p)
            self.assertEqual(_value_of(base, "-np"), "2")

    def test_the_long_spelling_is_caught_too(self):
        """`--parallel 1` is the same instruction as `-np 1`, and a fix that
        greps for one spelling and not the other is how this repository lost
        an afternoon on 04.09.2026 — the flag exists twice, so the test does
        too."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._profile(
                d, "--alias m -m /models/x.gguf -c 65536 --parallel 1")
            with quiet():
                base, _ = RS.base_from_profile(p)
            self.assertNotIn("--parallel", base)
            self.assertEqual(_value_of(base, "-np"), "2")

    def test_the_override_is_reported_rather_than_done_in_silence(self):
        """A measurement that quietly runs a different configuration than the
        profile it names is the failure one level up from the one being
        fixed. The suite has to say which flag it replaced."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._profile(
                d, "--alias m -m /models/x.gguf -c 65536 -np 1")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                RS.base_from_profile(p)
            said = out.getvalue()
            self.assertIn("-np", said)
            self.assertIn("2", said)

    def test_a_profile_that_already_says_two_is_left_alone(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._profile(
                d, "--alias m -m /models/x.gguf -c 65536 -np 2")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                base, _ = RS.base_from_profile(p)
            self.assertEqual(_value_of(base, "-np"), "2")
            self.assertEqual(out.getvalue(), "",
                             "nothing was overridden, so nothing is reported")

    def test_the_parallel_cell_documents_the_slot_count_it_needs(self):
        """Pins the reason rather than the mechanics: if somebody ever makes
        the slot count configurable again, this is the sentence that has to
        stay true."""
        src = ast.get_source_segment(SOURCE, _func("cell_parallel")) or ""
        self.assertIn("-np", src)


class TheVerdictNamesWhatWasMeasured(unittest.TestCase):
    """A run that measured something must not print a verdict of six blanks.

    Found 05.09.2026 by reading the output of a real run: `--cells parallel`
    measured two cells, wrote both into result.json as clean, and printed a
    VERDICT listing six OTHER cells as `not run` and neither of the two that
    had actually happened. The loop iterated a hard-coded tuple that predates
    the parallel cell.

    Read on its own, that verdict says a run measured nothing. It is the same
    failure this file was opened for in the first place, one level further
    out: the suite ran, it exited, and what it was for is not in what it says.

    The second half is older and is quoted in verdict_line's own docstring as
    though it were fixed: a cell nobody asked for printed the same `?` as a
    cell an aborted run never reached. Three states were promised, two were
    implemented. A comment that claims more than the code does is worse than
    no comment, so either the code grows the third state or the docstring
    stops claiming it. It grows the state.
    """

    def test_a_cell_that_ran_is_in_the_verdict(self):
        lines = RS.verdict_lines({"parallel-spec": {"clean": True}},
                                 {"parallel"})
        joined = "\n".join(lines)
        self.assertIn("parallel-spec", joined)
        self.assertIn("CLEAN", joined)

    def test_a_cell_nobody_asked_for_reads_differently_from_one_that_was_lost(self):
        """The distinction the whole verdict turns on: a narrow run and a
        truncated one must not produce the same page."""
        narrow = RS.verdict_lines({"parallel-spec": {"clean": True}},
                                  {"parallel"})
        idle = [l for l in narrow if l.strip().startswith("idle-spec")]
        self.assertEqual(len(idle), 1)
        self.assertNotIn("not run", idle[0])

        truncated = RS.verdict_lines({}, {"idle"})
        idle2 = [l for l in truncated if l.strip().startswith("idle-spec")]
        self.assertEqual(len(idle2), 1)
        self.assertIn("not run", idle2[0])

        self.assertNotEqual(idle[0], idle2[0],
                            "a cell nobody asked for and a cell the run never "
                            "reached are different findings")

    def test_every_known_cell_appears_even_when_nothing_ran(self):
        joined = "\n".join(RS.verdict_lines({}, set()))
        for cell in ("idle", "busy", "prefill", "parallel"):
            for half in ("spec", "nospec"):
                self.assertIn("%s-%s" % (cell, half), joined)

    def test_a_cell_in_the_results_is_never_dropped_for_being_unknown(self):
        """The hard-coded tuple is what lost the parallel cell. A cell that
        exists in the results but in no list must still be printed — the next
        cell somebody adds must not be able to vanish the same way."""
        joined = "\n".join(RS.verdict_lines({"invented-spec": {"clean": True}},
                                            set()))
        self.assertIn("invented-spec", joined)


class TheReportCanSayHowItWasMeasured(unittest.TestCase):
    """`-np` decides whether this suite measures anything at all, and the
    report of 05.09.2026 could not say what it had used.

    The run was correct — the override had printed itself to the terminal —
    but a terminal is not a record. result.json carried the binary, the build
    stamp, its cmake line and the restore timeout, and not one word about the
    slot count, the context size, or which profile the flags came from. On a
    suite whose entire subject is a defect that only exists at -np >= 2, that
    is the one field that cannot be missing.

    This repo's rule is that a figure carries its source. A report that cannot
    say its conditions held has to say so; this one could not say what its
    conditions WERE.
    """

    def _main_source(self):
        return ast.get_source_segment(SOURCE, _func("main")) or ""

    def test_the_effective_server_argv_is_recorded(self):
        self.assertIn('meta["argv"]', self._main_source(),
                      "without the argv nothing downstream can tell a "
                      "two-slot run from a one-slot one")

    def test_the_profile_the_flags_came_from_is_recorded(self):
        self.assertIn('meta["profile"]', self._main_source())

    def test_the_cells_that_were_asked_for_are_recorded(self):
        """Otherwise `not run` in a stored verdict cannot be told from a cell
        the caller deliberately left out — the same distinction the verdict
        makes on screen, made durable."""
        self.assertIn('meta["cells"]', self._main_source())


if __name__ == "__main__":
    unittest.main()
