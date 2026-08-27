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
        for d in self.defects:
            if d["detect"].get("kind") == "manual":
                self.assertEqual(defects.evaluate(d, QWEN38, STAMP_GOOD)[0],
                                 defects.MANUAL, d["id"])


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


if __name__ == "__main__":
    unittest.main()
