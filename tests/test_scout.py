"""scout — the reasoning it does before a download, without a network.

The point of the tool is to answer three questions in the minutes before
committing to 60 GB, and two of its answers are easy to get subtly wrong:

  * a sharded GGUF is ONE model and only its total matters, while a .gguf that
    is an mmproj or an imatrix is not a candidate at all;
  * "it does not fit" has two completely different causes — the GTT cap, which
    a kernel parameter fixes, and total RAM, which nothing fixes. Reporting
    the wrong one sends the reader to reboot for no reason, or worse, tells
    them a reboot will help when it will not.

Both are pure functions here, so both are tested here. The network paths are
not tested; they are thin and their failure is loud.
"""
import os, sys, unittest

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "setup" / "scripts"))
import scout                                                  # noqa: E402

GIB = 1024 ** 3


class TestGrouping(unittest.TestCase):
    def files(self, *paths_and_sizes):
        return [{"path": p, "size": s} for p, s in paths_and_sizes]

    def test_shards_are_one_model_with_one_total(self):
        """Laguna is three parts and its profile names part one, which finds
        the rest. Counting them separately would show three models that each
        look like they fit."""
        g = scout.group_files(self.files(
            ("Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf", 4 * GIB),
            ("Laguna-S-2.1-UD-Q4_K_XL-00002-of-00003.gguf", 47 * GIB),
            ("Laguna-S-2.1-UD-Q4_K_XL-00003-of-00003.gguf", 22 * GIB)))
        self.assertEqual(list(g), ["Laguna-S-2.1-UD-Q4_K_XL"])
        self.assertEqual(g["Laguna-S-2.1-UD-Q4_K_XL"]["bytes"], 73 * GIB)
        self.assertEqual(g["Laguna-S-2.1-UD-Q4_K_XL"]["parts"], 3)

    def test_companions_are_marked_and_models_are_not(self):
        g = scout.group_files(self.files(
            ("Qwen3.8-27B-UD-Q4_K_XL.gguf", 17 * GIB),
            ("mmproj-F16.gguf", 1 * GIB),
            ("imatrix_unsloth.gguf", 1),
            ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 1 * GIB)))
        self.assertFalse(g["Qwen3.8-27B-UD-Q4_K_XL"]["companion"])
        for name in ("mmproj-F16", "imatrix_unsloth", "MTP/mtp-Qwen3.8-27B-Q4_0"):
            with self.subTest(name=name):
                self.assertTrue(g[name]["companion"],
                                "%s would be offered as a model to load" % name)

    def test_non_gguf_files_are_ignored(self):
        g = scout.group_files(self.files(("README.md", 100),
                                         ("model.safetensors", 50 * GIB)))
        self.assertEqual(g, {})


class TestFit(unittest.TestCase):
    """The distinction that decides what the reader does next."""

    RAM = 124.9

    def test_a_model_that_fits_says_how_much_is_left(self):
        verdict, why = scout.fit_verdict(60 * GIB, gtt_gib=116.0, ram_gib=self.RAM)
        self.assertEqual(verdict, "yes")
        self.assertIn("left for KV", why)

    def test_too_big_for_the_cap_points_at_the_cap(self):
        """96 GiB was the cap until 26.08., and a 96 GiB model is exactly the
        case that made the whole question urgent."""
        verdict, why = scout.fit_verdict(96 * GIB, gtt_gib=96.0, ram_gib=self.RAM)
        self.assertEqual(verdict, "cap")
        # It used to assert that the message names `gtt.sh --set <number>`.
        # It does not any more, and that is the point: a one-line "raise the
        # cap" is advice that cost this machine two hangs on 26.08. The cap is
        # a ceiling as well as a budget. What the verdict must still do is say
        # WHICH limit was hit, so the reader knows a smaller quant is one of
        # the answers.
        self.assertIn("cap", why)
        self.assertNotIn("--set", why, "a one-line raise is not advice")

    def test_too_big_for_the_machine_does_not_suggest_a_reboot(self):
        """The failure mode this guards: telling someone a kernel parameter
        will fix a model that is simply larger than the RAM."""
        verdict, why = scout.fit_verdict(120 * GIB, gtt_gib=116.0, ram_gib=self.RAM)
        self.assertEqual(verdict, "no")
        self.assertNotIn("gtt.sh", why)
        self.assertIn("no kernel parameter", why)

    def test_raising_the_cap_turns_cap_into_yes(self):
        """The estimate for a realistic Q4 of Flash-Next is ~96 GiB of
        weights. At the cap this machine booted with until 26.08. that is a
        'cap' verdict; at 116 GiB it fits — 96 + 10 KV = 106 of 116, and
        118 of 124.9 GiB in total. Which is what the raise was for."""
        size = 96 * GIB
        self.assertEqual(scout.fit_verdict(size, 96.0, self.RAM)[0], "cap")
        self.assertEqual(scout.fit_verdict(size, 116.0, self.RAM)[0], "yes")

    def test_no_cap_saves_a_model_that_is_larger_than_the_machine(self):
        """Raising the cap to the maximum still cannot help here — and the
        message must not pretend otherwise."""
        for cap in (96.0, 116.0, 120.0, self.RAM):
            with self.subTest(cap=cap):
                verdict, why = scout.fit_verdict(115 * GIB, cap, self.RAM)
                self.assertEqual(verdict, "no")
                self.assertNotIn("gtt.sh", why)


class TestSupportLookup(unittest.TestCase):
    """Whether OUR checkout can handle an architecture — not llama.cpp in
    general. The two halves can disagree: convertible but not loadable, or a
    GGUF from someone else that loads without us being able to convert it."""

    SRC = os.environ.get("LLAMA_SRC", os.path.expanduser("~/llama.cpp"))

    def setUp(self):
        if not os.path.isdir(os.path.join(self.SRC, "conversion")):
            self.skipTest("no llama.cpp checkout at %s" % self.SRC)

    def test_a_known_architecture_is_found(self):
        hits = scout.converter_supports("Qwen3NextForCausalLM", self.SRC)
        self.assertIn("qwen.py", hits)

    def test_an_invented_architecture_is_not(self):
        self.assertEqual(scout.converter_supports("NotARealArch9000", self.SRC), [])

    def test_the_runtime_list_is_read_and_not_empty(self):
        archs = scout.runtime_architectures(self.SRC)
        self.assertGreater(len(archs), 100, "LLM_ARCH_NAMES was not parsed")
        for known in ("llama", "qwen3next", "qwen35moe"):
            with self.subTest(arch=known):
                self.assertIn(known, archs)

    def test_a_missing_checkout_is_empty_rather_than_an_exception(self):
        """scout runs before anything is set up; it must degrade, not crash."""
        self.assertEqual(scout.runtime_architectures("/does/not/exist"), [])
        self.assertEqual(scout.converter_supports("X", "/does/not/exist"), [])


class TestArchitecturesInText(unittest.TestCase):
    """The parser is split out because a caller may ask the same question
    of a llama-arch.cpp it FETCHED from upstream master. One parser, two
    sources — a second one would be a second thing to get wrong, and getting
    an architecture name wrong is silent: the answer is 'not supported',
    which is also the answer when it is genuinely not supported."""

    TEXT = '''
        static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
            { LLM_ARCH_LLAMA,    "llama"    },
            { LLM_ARCH_QWEN4EXP, "qwen4exp" },
        };
    '''

    def test_it_reads_names_out_of_a_text(self):
        self.assertEqual(scout.architectures_in_text(self.TEXT),
                         ["llama", "qwen4exp"])

    def test_the_underscore_spelling_is_not_the_gguf_name(self):
        """config.json's model_type is qwen4_exp; general.architecture and
        LLM_ARCH_NAMES say qwen4exp. Confusing them cost a working watch."""
        self.assertNotIn("qwen4_exp", scout.architectures_in_text(self.TEXT))

    def test_a_file_without_the_table_is_empty_rather_than_wrong(self):
        self.assertEqual(scout.architectures_in_text("nothing here"), [])
        self.assertEqual(scout.architectures_in_text(""), [])


class TestMachineLimits(unittest.TestCase):
    def test_it_reads_this_machine_or_says_zero(self):
        gtt, ram = scout.machine_limits()
        self.assertGreater(ram, 1.0, "MemTotal was not read")
        if gtt:
            self.assertLessEqual(gtt, ram,
                                 "the GTT cap cannot exceed total RAM — if it "
                                 "does, one of the two is being misread")


if __name__ == "__main__":
    unittest.main()
