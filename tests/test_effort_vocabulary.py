"""effort-vocabulary — the map derived from a template, and when it must not be.

The suite renders each model's chat template with every level Anthropic can
ask for and derives the profile lines from what comes back. The derivation is
where it can be wrong quietly, so it is pinned here: both bugs below were in
the first version and both produced output that looked entirely reasonable.

  * A Python precedence slip — `[a] if cond else [] + [b]` binds as
    `[a] if cond else ([] + [b])` — dropped every level mode for exactly the
    models that HAVE levels, leaving `MODES=nothink:off` for qwen38.
  * "Renders without crashing" was read as "the model supports this". For a
    template that validates nothing, that proposed `max:max` for gpt-oss —
    the untrained word the clamp exists to keep out of the prompt.

The second one is the reason this file exists. A measurement that cannot
distinguish "supported" from "not rejected" has to say so instead of
answering.
"""
import unittest

import common

EV = common.load("bench/suites/effort-vocabulary.py", "effort_vocabulary")
M = common.load("setup/gateway/modes.py", "modes")


def measured(levels, thinking=True, plain="DEFAULT", measurable=True):
    """A measure() result, spelled out. `levels` maps a level to the sha of its
    rendering (probed with thinking ON), or None where the template raised."""
    accepted = {lv: h for lv, h in levels.items() if h}
    ignored = bool(accepted) and all(h == plain for h in accepted.values()) \
        and len(accepted) > 1
    lv_set = set() if ignored else set(accepted)
    canon = {lv: next(x for x in EV.LEVELS if levels.get(x) == h)
             for lv, h in levels.items() if h}
    words = {}
    for w in EV.LEVELS:
        v = EV.value_for(w, lv_set, thinking, canon)
        words[w] = None if v is None else {
            "value": v,
            # A word's rendering is its canonical level's, and `none`/`on` are
            # their own thing — enough for suggest(), which only compares
            # against the baseline.
            "render": plain if v in (M.OFF,) else "R" + str(v)}
    return {"default_render": plain,
            "levels": levels,
            "crashes": sorted(lv for lv, h in levels.items() if h is None),
            "reads_effort": not ignored,
            "reads_enable_thinking": thinking,
            "measurable": measurable,
            "probe_render": plain,
            "words": words}


QWEN = measured({"low": "A", "medium": "B", "high": "C", "xhigh": "C",
                 "max": None})
GPTOSS = measured({"low": "A", "medium": "DEFAULT", "high": "C", "xhigh": "D",
                   "max": "E"}, thinking=False, measurable=False)
GEMMA = measured({lv: "DEFAULT" for lv in EV.LEVELS if lv != "none"})


class TestWhatItSuggestsIsWhatTheProfileParsesBack(unittest.TestCase):
    """The test that was missing, and the defect it would have caught.

    Every profile names this suite as the source of its MODES and
    TEMPLATE_LEVELS lines. Until 28.08.2026 it emitted `nothink:off` — which
    modes.parse_modes REFUSES — and an `EFFORT_MAP=` line that nothing reads.
    The profile lines had in fact been typed by hand while claiming to be
    measured, which is the state this whole repository argues against, wearing
    a measurement's clothes.

    Two green test files asserted opposite things about the same vocabulary and
    nobody noticed, because no test ever fed one to the other. This is that
    test, and it is four lines. Same shape as TestConflicts and
    test_systemunit, which the repo already holds up as the answer for exactly
    this class.
    """

    def crossing(self, m):
        modes_line, levels_line = EV.suggest(m)
        if modes_line.lstrip().startswith("#"):
            return None, M.parse_levels(levels_line.split("=", 1)[1])
        return (M.parse_modes(modes_line.split("=", 1)[1]),
                M.parse_levels(levels_line.split("=", 1)[1]))

    def test_a_validating_template_round_trips(self):
        modes, levels = self.crossing(QWEN)
        self.assertTrue(modes, "suggested nothing to parse")
        M.check_modes(modes, levels)

    def test_a_template_without_levels_round_trips(self):
        modes, levels = self.crossing(GEMMA)
        self.assertTrue(modes)
        M.check_modes(modes, levels)

    def test_an_unmeasurable_template_round_trips(self):
        """It suggests no MODES at all — deliberately — but its
        TEMPLATE_LEVELS must still parse, and must say `unmeasurable` rather
        than leaving the field empty."""
        modes, levels = self.crossing(GPTOSS)
        self.assertIsNone(modes)
        self.assertIsNone(levels, "an empty field would read as 'not stated'")

    def test_no_suggested_name_is_outside_the_vocabulary(self):
        for label, m in (("qwen", QWEN), ("gemma", GEMMA)):
            modes, _ = self.crossing(m)
            for name in modes:
                self.assertIn(name, M.VOCABULARY, "%s: %s" % (label, name))


class TestTheClampIsWhatTheTemplateSurvives(unittest.TestCase):
    def test_a_level_that_raises_is_clamped_to_the_ceiling(self):
        """`max` raises on the Qwen templates, so it must send this template's
        highest real level instead of a 500."""
        modes, _ = M.parse_modes(EV.suggest(QWEN)[0].split("=", 1)[1]), None
        self.assertEqual(modes["max"], modes["high"],
                         "max must land on the same value as the ceiling")

    def test_levels_that_render_alike_send_one_value(self):
        """high and xhigh render identically. Sending both words verbatim
        would be two chat_template_kwargs for one prompt, and since 28.08. two
        prefix-cache keys."""
        modes = M.parse_modes(EV.suggest(QWEN)[0].split("=", 1)[1])
        self.assertEqual(modes["high"], modes["xhigh"])

    def test_a_template_without_levels_only_offers_the_knob(self):
        modes = M.parse_modes(EV.suggest(GEMMA)[0].split("=", 1)[1])
        self.assertEqual(set(modes.values()), {M.ON})

    def test_the_measured_levels_are_reported_as_a_list(self):
        self.assertEqual(EV.suggest(QWEN)[1],
                         "TEMPLATE_LEVELS=low  medium  high  xhigh")


class TestTheSuiteNeedsNoServer(unittest.TestCase):
    """The property that makes it usable at all: a 180 B model is as cheap to
    measure as a 13 GiB one, because the template is metadata and llama.cpp
    ships the engine that renders it."""

    def test_it_reads_the_template_from_the_file_not_from_a_server(self):
        src = (common.REPO / "bench" / "suites" /
               "effort-vocabulary.py").read_text(encoding="utf-8")
        self.assertIn("GGUFReader", src)
        self.assertNotIn("urllib", src, "a server would defeat the point")

    def test_it_renders_through_the_output_file_not_stdout(self):
        """The false start: stdout is trace output. Hashing it reported `high`
        and `xhigh` as different renderings, contradicting the template's own
        source."""
        src = (common.REPO / "bench" / "suites" /
               "effort-vocabulary.py").read_text(encoding="utf-8")
        self.assertIn('"--output"', src)


if __name__ == "__main__":
    unittest.main()
