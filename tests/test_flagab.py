"""flag-ab — the parts that decide WHAT gets compared.

Same reasoning as test_speedab.py: a suite that runs the wrong comparison
does not fail, it prints a table and attributes the difference to the wrong
thing. The arm parser and the one-difference rule are what stand between a
label and a claim, so they are what gets tested — the plumbing around
llama-bench is speed-ab's, and tested there.
"""
import os, sys, unittest

import common

REPO = common.REPO
sys.path.insert(0, str(REPO / "bench"))
flag_ab = common.load(str(REPO / "bench" / "suites" / "flag-ab.py"), "flag_ab")


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


if __name__ == "__main__":
    unittest.main()
