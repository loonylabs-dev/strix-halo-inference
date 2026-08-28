"""defects — the registry, and the two ways it could lie.

A defect registry is only worth having if it is wrong in the safe direction.
The two failure modes that matter:

  * **Silence read as safety.** With no server running, an argument check
    cannot be answered. It must say so and never say "guarded" — an operator
    who reads a green line and starts an unguarded server is worse off than
    one who read nothing at all.
  * **Crying wolf.** An architecture-specific defect reported while a
    different model serves trains the reader to skip the section, which is
    how a real warning gets missed. That is exactly what happened to
    check.sh's MAX_INFLIGHT line, which had been reporting a real defect for
    days before anybody read it.

The registry's CONTENT is pinned too: a suite that has been renamed away, or
an entry without a mitigation, makes the file worse than the six documents it
replaced.
"""
import json, os, re, sys, unittest

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "setup" / "lib"))
import defects                                                # noqa: E402

QWEN38 = ["llama-server", "--alias", "qwen38",
          "-m", "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
          "-ngl", "999", "-fa", "on", "-c", "204800", "-np", "1",
          "--slot-save-path", "/home/x/.cache/llama-slots"]

FLASHNEXT = ["llama-server", "--alias", "flashnext",
             "-m", "/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
             "-ngl", "999", "-fa", "on", "-fit", "off", "-c", "65536", "-np", "1"]

STAMP_GOOD = {"patch_commit": "6b39dd5d5b98", "cmake": "-DGGML_HIP=ON -DGPU_TARGETS=gfx1151"}


def by_id(id_):
    for d in defects.load():
        if d["id"] == id_:
            return d
    raise AssertionError("no defect %r in the registry" % id_)


class TestTheRegistryItself(unittest.TestCase):
    """A half-written entry is a lie with a green tick next to it."""

    def setUp(self):
        self.defects = defects.load()

    def test_every_entry_carries_what_a_reader_needs(self):
        for d in self.defects:
            with self.subTest(id=d.get("id")):
                for field in ("id", "title", "shows_as", "symptom",
                              "measured", "mitigation", "detect", "status"):
                    self.assertTrue(d.get(field), "%s: %s is empty" % (d.get("id"), field))

    def test_shows_as_is_one_of_the_four(self):
        for d in self.defects:
            self.assertIn(d["shows_as"], defects.SEVERITY, d["id"])

    def test_ids_are_unique(self):
        ids = [d["id"] for d in self.defects]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_named_suite_exists(self):
        """A registry that points at a deleted suite is worse than none: it
        sends the reader to run a measurement that cannot be run."""
        for d in self.defects:
            suite = d.get("suite")
            if suite:
                self.assertTrue((REPO / suite).exists(),
                                "%s names a suite that is gone: %s" % (d["id"], suite))

    def test_upstream_entries_are_urls(self):
        for d in self.defects:
            for u in d.get("upstream", []):
                self.assertTrue(u.startswith("https://"), "%s: %s" % (d["id"], u))

    def test_a_manual_defect_says_so_rather_than_guessing(self):
        """WITHDRAWN entries are excluded, and the exclusion is the finding.

        `slot-restore-hangs-busy` still carries `detect.kind: manual` because
        that is what it WAS, and the entry is kept as a correction rather
        than deleted. It must not be reported as a question any more, so the
        two states have to be separable here — see
        TestAWithdrawnEntryStopsAsking. The count is asserted so that this
        loop cannot pass by finding nothing to check.
        """
        checked = 0
        for d in self.defects:
            if str(d.get("status", "")).startswith("withdrawn"):
                continue
            if d["detect"].get("kind") != "manual":
                continue
            checked += 1
            # Against a cmdline the defect APPLIES to. Judging every entry by
            # qwen38 alone conflated two different answers: on 28.08. the
            # first manual defect scoped to one model — the PLE table not
            # being demand-paged, which is a qwen4exp defect — answered "n/a"
            # here and failed a test whose subject is guessing. Staying quiet
            # about a model that is not running is the behaviour this file's
            # own docstring asks for, so the fixture has to match the entry.
            verdicts = [defects.evaluate(d, cmd, STAMP_GOOD)[0]
                        for cmd in (QWEN38, FLASHNEXT)]
            self.assertIn(defects.MANUAL, verdicts,
                          "%s never says 'only a measurement answers this', "
                          "for either model: %s" % (d["id"], verdicts))
            self.assertNotIn(defects.GUARDED, verdicts,
                             "%s claims to be guarded without a measurement" % d["id"])
        self.assertGreater(checked, 0, "no manual defect left to check")


class TestSilenceIsNotSafety(unittest.TestCase):
    def test_an_argument_check_is_unknown_with_nothing_running(self):
        """Only ARGUMENT checks. A build defect is a property of the build and
        is answered by the .build-stamp whether or not a server runs — the
        first version of this test asserted it of every kind and was wrong
        about exactly that, which is the distinction worth pinning."""
        for d in defects.load():
            if d["detect"].get("kind") != "cmdline":
                continue
            verdict, _ = defects.evaluate(d, None, STAMP_GOOD)
            with self.subTest(id=d["id"]):
                self.assertEqual(verdict, defects.UNKNOWN,
                                 "%s answered an argument question with no argv" % d["id"])

    def test_a_build_defect_is_still_answerable_without_a_server(self):
        self.assertEqual(defects.evaluate(by_id("gfx1151-hip-integrated"),
                                          None, STAMP_GOOD)[0], defects.GUARDED)

    def test_no_build_stamp_is_unknown_not_guarded(self):
        d = by_id("gfx1151-hip-integrated")
        self.assertEqual(defects.evaluate(d, QWEN38, None)[0], defects.UNKNOWN)


class TestCommandLineDetection(unittest.TestCase):
    def test_two_slots_are_reported(self):
        d = by_id("gfx1151-two-slots")
        argv = [a if a != "1" else "4" for a in QWEN38]
        self.assertEqual(defects.evaluate(d, argv, STAMP_GOOD)[0], defects.EXPOSED)

    def test_one_slot_is_guarded(self):
        self.assertEqual(defects.evaluate(by_id("gfx1151-two-slots"), QWEN38,
                                          STAMP_GOOD)[0], defects.GUARDED)

    def test_an_unpassed_flag_is_judged_by_its_DEFAULT(self):
        """-fit defaults to ON. Not passing it is not the same as passing off,
        and a registry that treated absence as absence would have missed the
        one defect that made this rule worth writing down."""
        d = by_id("qwen4exp-fit-crash")
        without = [a for a in FLASHNEXT if a not in ("-fit", "off")]
        verdict, detail = defects.evaluate(d, without, STAMP_GOOD)
        self.assertEqual(verdict, defects.EXPOSED)
        self.assertIn("default is on", detail)
        self.assertEqual(defects.evaluate(d, FLASHNEXT, STAMP_GOOD)[0], defects.GUARDED)

    def test_a_flag_that_must_be_ABSENT(self):
        d = by_id("qwen4exp-slot-restore")
        self.assertEqual(defects.evaluate(d, FLASHNEXT, STAMP_GOOD)[0], defects.GUARDED)
        with_it = FLASHNEXT + ["--slot-save-path", "/home/x/.cache/llama-slots"]
        self.assertEqual(defects.evaluate(d, with_it, STAMP_GOOD)[0], defects.EXPOSED)


class TestScope(unittest.TestCase):
    def test_an_architecture_defect_is_silent_about_another_model(self):
        for id_ in ("qwen4exp-fit-crash", "qwen4exp-quantized-kv",
                    "qwen4exp-slot-restore"):
            with self.subTest(id=id_):
                self.assertEqual(defects.evaluate(by_id(id_), QWEN38, STAMP_GOOD)[0],
                                 defects.NA)

    def test_and_speaks_up_about_the_right_one(self):
        d = by_id("qwen4exp-quantized-kv")
        q8 = FLASHNEXT + ["-ctk", "q8_0"]
        self.assertEqual(defects.evaluate(d, q8, STAMP_GOOD)[0], defects.EXPOSED)


class TestBuildFlags(unittest.TestCase):
    def test_a_forbidden_cmake_flag_is_found(self):
        stamp = dict(STAMP_GOOD, cmake=STAMP_GOOD["cmake"] + " -DGGML_HIP_ROCWMMA_FATTN=ON")
        self.assertEqual(defects.evaluate(by_id("rocwmma-fattn-prefill"), QWEN38, stamp)[0],
                         defects.EXPOSED)

    def test_an_unpatched_build_is_exposed(self):
        stamp = dict(STAMP_GOOD, patch_commit="")
        self.assertEqual(defects.evaluate(by_id("gfx1151-hip-integrated"), QWEN38, stamp)[0],
                         defects.EXPOSED)


class TestOrdering(unittest.TestCase):
    def test_exposed_comes_first_and_silent_before_loud(self):
        """The ordering IS the message: on this hardware the dangerous
        defects do not raise."""
        argv = [a if a != "1" else "4" for a in QWEN38]      # -np 4: silent, exposed
        stamp = dict(STAMP_GOOD, cmake=STAMP_GOOD["cmake"] + " -DGGML_HIP_ROCWMMA_FATTN=ON")
        rows = defects.report(defects.load(), argv, stamp)
        verdicts = [v for _, v, _ in rows]
        self.assertEqual(verdicts[0], defects.EXPOSED)
        exposed = [d["shows_as"] for d, v, _ in rows if v == defects.EXPOSED]
        self.assertEqual(exposed[0], "silent",
                         "a slow defect was listed above a silent one")


class TestAWithdrawnEntryStopsAsking(unittest.TestCase):
    """The third way the registry could lie, found 27.08.

    `slot-restore-hangs-busy` was withdrawn that evening: it was not a
    defect, it was bench/suites/restore-safety.py giving a slot restore 300 s
    while filling that slot with 325-341 s of work. The entry stays, because
    it was the stated reason `-np 2` remained closed through three sessions
    and deleting a correction deletes the record of the mistake.

    But `detect.kind` is `manual`, so without this the registry kept printing
    "only a measurement answers this" for a question that had been measured
    and answered — sending every future reader to re-run a suite to
    rediscover an artefact. That is crying wolf, which this file's own
    docstring already names as a way to make a registry worse than nothing.
    """

    def setUp(self):
        self.defects = defects.load()

    def test_the_withdrawn_entry_is_not_reported_as_an_open_question(self):
        d = by_id("slot-restore-hangs-busy")
        verdict, detail = defects.evaluate(d, QWEN38, STAMP_GOOD, "gfx1151")
        self.assertEqual(verdict, defects.WITHDRAWN, detail)
        self.assertNotEqual(verdict, defects.MANUAL)
        self.assertNotEqual(verdict, defects.EXPOSED,
                            "a withdrawn entry must never turn check.sh red")

    def test_it_still_says_what_it_was_and_why_it_is_not(self):
        """A withdrawn entry whose text was not rewritten is worse than a
        deleted one: it reads as an open defect with a quiet status field."""
        d = by_id("slot-restore-hangs-busy")
        self.assertTrue(d["status"].startswith("withdrawn"), d["status"])
        for field in ("symptom", "measured", "mitigation"):
            self.assertIn("bound", d[field].lower() + " " + d["title"].lower(),
                          "%s does not say what the bound had to do with it"
                          % field)
        self.assertIn("319.8", d["measured"],
                      "the confirming measurement has to be IN the entry")

    def test_withdrawn_sorts_below_everything_that_is_still_a_question(self):
        rows = defects.report(self.defects, QWEN38, STAMP_GOOD, "gfx1151")
        ids = [d["id"] for d, _, _ in rows]
        self.assertEqual(ids[-1], "slot-restore-hangs-busy", ids)

    def test_an_ordinary_entry_is_untouched_by_the_new_verdict(self):
        """The positive control: if `status` decided everything, every entry
        would be withdrawn the moment somebody wrote prose into the field."""
        d = by_id("slot-restore-poison")
        verdict, _ = defects.evaluate(d, QWEN38, STAMP_GOOD, "gfx1151")
        self.assertEqual(verdict, defects.MANUAL)


class TestTheRetirementProbeCanReachBothAnswers(unittest.TestCase):
    """A probe that can only ever say "keep" is not a probe.

    `gfx1151-hip-integrated` watched the line
    `info.devices[id].integrated = prop.integrated`, on the reasonable
    assumption that a fix would delete it. llama.cpp PR #27311 does not: it
    leaves the line and makes the buffer it leads to safe, in a different
    file. Measured 28.08. — a build containing that PR has the line at
    ggml-cuda.cu:306 and is 0 of 10 corrupt where master is 10 of 10.

    So on the day the fix lands the old probe would still have said "keep
    shipping it", for ever, and the only thing standing between this stack and
    carrying a patch it no longer needs would be somebody remembering. That is
    the shape this repository keeps finding, in the one check whose whole job
    is to say when you may stop.
    """

    CAUSE = "info.devices[id].integrated = prop.integrated;"
    FIX = 'int n_copies_uma = is_uma ? 2 : 1;\ngetenv("GGML_SCHED_UMA_RING");'

    def probe(self, source, id_="gfx1151-hip-integrated"):
        d = [x for x in defects.load() if x["id"] == id_]
        self.assertEqual(len(d), 1, id_)
        return defects.check_upstream(d, fetch=lambda p: source)[0][1]

    def test_it_says_keep_while_the_fix_is_not_in_master(self):
        self.assertEqual(self.probe(self.CAUSE), "keep")

    def test_it_says_retire_once_the_fix_lands(self):
        """The answer the old one could not reach — and note the cause line is
        STILL in the source here, because that is what actually happens."""
        self.assertEqual(self.probe(self.CAUSE + "\n" + self.FIX), "RETIRE?")

    def test_the_old_condition_would_have_been_stuck(self):
        """Why the `present_means` field exists at all. Same source as the
        test above; a probe watching the CAUSE line still says keep."""
        old = {"id": "x", "upstream_check": {
            "kind": "source-contains", "path": "p",
            "pattern": r"info\.devices\[id\]\.integrated",
            "while_present": "keep shipping it", "when_gone": "retire"}}
        state = defects.check_upstream(
            [old], fetch=lambda p: self.CAUSE + "\n" + self.FIX)[0][1]
        self.assertEqual(state, "keep",
                         "if this ever reports RETIRE?, the reason this field "
                         "exists has gone away and the field can go with it")

    def test_present_means_defaults_to_keep(self):
        """Every other entry watches a CAUSE and must be untouched by the new
        field. Asserted on the registry itself, not on a fixture."""
        checked = 0
        for d in defects.load():
            pr = d.get("upstream_check")
            if not pr or pr.get("present_means") == "retire":
                continue
            checked += 1
            self.assertEqual(
                defects.check_upstream([d], fetch=lambda p: pr["pattern"]
                                       .replace("\\", ""))[0][1],
                "keep", d["id"])
        self.assertGreater(checked, 0, "no cause-watching probe left to check")

    def test_the_registry_entry_names_the_fix_and_not_the_cause(self):
        """The narrow thing, pinned so a future edit cannot quietly put it
        back: the probe must not be watching ggml-cuda.cu any more."""
        d = [x for x in defects.load()
             if x["id"] == "gfx1151-hip-integrated"][0]["upstream_check"]
        self.assertEqual(d["present_means"], "retire")
        self.assertIn("ggml-backend", d["path"])
        self.assertIn("UMA_RING", d["pattern"])


if __name__ == "__main__":
    unittest.main()
