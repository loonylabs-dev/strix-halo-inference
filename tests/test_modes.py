"""modes — the standard reasoning vocabulary, translated per model.

Two decisions are pinned here and they are not the same one.

**The names are the standard vocabulary and nothing else.** `none`, `low`,
`medium`, `high`, `xhigh`, `max` — what Anthropic's API speaks and what a
consumer already knows. An earlier version of this file invented `think`,
`deep` and `full`; that was a third register for an idea that already had two
and it is gone. A name outside the vocabulary is refused rather than accepted
quietly, because the whole value of a fixed vocabulary is that it stays fixed.

**The translation downward is measured per model.** The vocabulary is the
consumer's; what a chat template accepts is decided by whoever exported the
GGUF, and the two do not match. Measured 28.08.2026 by
bench/suites/effort-vocabulary.py:

    qwen38 · flashnext   renders low, medium, xhigh. `high` is aliased to
                         xhigh BY THE TEMPLATE, and `max`/`none` RAISE — an
                         HTTP 500 at the server
    gptoss               validates nothing; any string reaches the prompt
    gemma · laguna       no levels at all, only enable_thinking

So `qwen38-max` must exist as a name and must not send `max`.

Why several names may share one value
-------------------------------------
`high`, `xhigh` and `max` all send `xhigh` on qwen38, because that is what the
template renders for all three (measured: sha 546e0307 for high and for
xhigh). Sending the consumer's own word instead would produce three different
`chat_template_kwargs` for one identical prompt — and since 28.08. the kwargs
are part of the prefix id, that is three cache keys and two wasted prefills.
Canonicalising to what the template actually does collapses them to one.
"""
import unittest

import common

M = common.load("setup/gateway/modes.py", "modes")

# Exactly what the profiles declare, after measurement. No `none` on either:
# both Qwen profiles switch thinking off on the command line and gemma's
# template defaults to off, so a `none` mode would render what the bare alias
# already renders — the same prompt under a second cache key. laguna measures
# the other way round and declares `none` and nothing else.
QWEN = "low:on+low  medium:on+medium  high:on+xhigh  xhigh:on+xhigh  max:on+xhigh"
QWEN_LEVELS = "low medium xhigh"
# gemma reads no levels: every level word a consumer may ask for is one thing.
GEMMA = "low:on  medium:on  high:on  xhigh:on  max:on"


class TestTheModuleDocstringSurvivesItsOwnParser(unittest.TestCase):
    """The example at the top of modes.py showed `MODES=nothink:off …` and an
    `EFFORT_MAP=` line: one refused by parse_modes, the other a field nothing
    reads. That is the first thing a reader of the module sees, and it was
    wrong in both halves.
    """

    def test_the_example_lines_parse(self):
        import re
        src = (common.REPO / "setup" / "gateway" / "modes.py").read_text(encoding="utf-8")
        head = src.split('"""')[1]
        modes = re.search(r"^\s*MODES=(.*)$", head, re.M)
        levels = re.search(r"^\s*TEMPLATE_LEVELS=(.*)$", head, re.M)
        self.assertTrue(modes and levels, "the docstring shows no example")
        M.check_modes(M.parse_modes(modes.group(1)),
                      M.parse_levels(levels.group(1)))

    def test_it_does_not_advertise_a_field_that_was_removed(self):
        src = (common.REPO / "setup" / "gateway" / "modes.py").read_text(encoding="utf-8")
        self.assertNotIn("EFFORT_MAP", src)


class TestTheVocabularyIsFixed(unittest.TestCase):
    def test_the_six_words_and_only_those(self):
        self.assertEqual(list(M.VOCABULARY),
                         ["none", "low", "medium", "high", "xhigh", "max"])

    def test_an_invented_name_is_refused(self):
        """`think`, `deep` and `full` were in these profiles until 28.08. A
        fixed vocabulary that admits one more word is not fixed."""
        with self.assertRaises(SystemExit) as cm:
            M.parse_modes("think:on+low")
        self.assertIn("think", str(cm.exception))
        for word in M.VOCABULARY:
            self.assertIn(word, str(cm.exception),
                          "the refusal has to say what IS allowed")

    def test_the_order_is_the_vocabulary_order_not_the_file_order(self):
        """A picker reads top to bottom, and `none` before `max` is the only
        order that means anything. Declaration order would make it depend on
        how somebody happened to type the line."""
        modes = M.parse_modes("max:on+xhigh  none:off  low:on+low")
        self.assertEqual(list(modes), ["none", "low", "max"])

    def test_a_profile_may_offer_a_subset(self):
        """gpt-oss has no off switch at all; gemma has no levels. Offering
        every word everywhere would advertise modes a model cannot tell
        apart."""
        self.assertEqual(list(M.parse_modes("low:on+low  high:on+xhigh")),
                         ["low", "high"])


class TestTheValueIsWhatTheTemplateGets(unittest.TestCase):
    def test_off_is_the_thinking_knob(self):
        self.assertEqual(M.kwargs_for("off"), {"enable_thinking": False})

    def test_on_is_the_thinking_knob(self):
        self.assertEqual(M.kwargs_for("on"), {"enable_thinking": True})

    def test_both_knobs_together(self):
        """MEASURED: qwen38.env's command line sets enable_thinking:false, and
        the template gates the whole effort block on it. A value sending only
        the level renders identically to sending nothing (sha 1ad7792b either
        way) while carrying its own prefix id — worse than doing nothing."""
        self.assertEqual(M.kwargs_for("on+low"),
                         {"enable_thinking": True, "reasoning_effort": "low"})

    def test_off_with_a_level_is_refused(self):
        with self.assertRaises(SystemExit):
            M.kwargs_for("off+low")

    def test_the_bare_alias_asks_for_nothing(self):
        self.assertEqual(M.kwargs_for(None), {})


class TestSeveralNamesMayShareOneRendering(unittest.TestCase):
    def setUp(self):
        self.modes = M.parse_modes(QWEN)

    def test_high_xhigh_and_max_all_send_xhigh(self):
        """The template aliases high to xhigh itself and raises on max, so all
        three ARE one rendering. The profile says so once."""
        for name in ("high", "xhigh", "max"):
            kw, _ = M.resolve("qwen38-" + name, "qwen38", self.modes)
            self.assertEqual(kw["reasoning_effort"], "xhigh", name)

    def test_they_therefore_share_a_prefix_id(self):
        """Three names, one cache key. Sending the consumer's own word would
        have made three."""
        got = {tuple(sorted(M.resolve("qwen38-%s" % n, "qwen38", self.modes)[0].items()))
               for n in ("high", "xhigh", "max")}
        self.assertEqual(len(got), 1)

    def test_the_ones_that_differ_still_differ(self):
        got = {tuple(sorted(M.resolve("qwen38-%s" % n, "qwen38", self.modes)[0].items()))
               for n in ("low", "medium", "xhigh")}
        self.assertEqual(len(got), 3)


class TestOfferingAndAcceptingAreNotTheSameList(unittest.TestCase):
    """A picker should show CHOICES, not synonyms.

    On qwen38 `high`, `xhigh` and `max` all send xhigh — the template aliases
    high itself and raises on max — so a listing that names all three offers
    one choice three times. Six entries, four behaviours.

    But all three still have to RESOLVE. A client configured
    `ANTHROPIC_MODEL=qwen38-max` means "as much as this model has"; if the name
    matched nothing it would fall through to the bare alias, and on qwen38 the
    bare alias does not think at all — the opposite of what was asked. So the
    two lists differ on purpose:

        names()    one entry per distinct behaviour   — what a picker shows
        resolve()  every declared word                — what a client may send

    The invariant that matters is preserved and is the safe direction:
    everything OFFERED resolves. The bug this design was written against was a
    listing offering what the injection then refused.
    """

    def setUp(self):
        self.modes = M.parse_modes(QWEN)

    def test_only_the_distinct_behaviours_are_offered(self):
        self.assertEqual(M.names("qwen38", self.modes),
                         ["qwen38", "qwen38-low", "qwen38-medium", "qwen38-high"])

    def test_the_synonyms_still_resolve(self):
        for name in ("qwen38-xhigh", "qwen38-max"):
            kw, hit = M.resolve(name, "qwen38", self.modes)
            self.assertTrue(hit, name)
            self.assertEqual(kw["reasoning_effort"], "xhigh", name)

    def test_everything_offered_resolves(self):
        """The invariant. A listing must never advertise what injection
        refuses — that was the original defect."""
        offered = M.names("qwen38", self.modes)
        self.assertEqual(len(offered), 4, "nothing was offered to check")
        for name in offered:
            self.assertTrue(M.resolve(name, "qwen38", self.modes)[1], name)

    def test_the_representative_is_the_lowest_word_of_its_group(self):
        """Deterministic, and it errs low: never advertise a level above what
        the model can actually do. `qwen38-max` in a picker would promise
        something this template raises on."""
        offered = M.names("qwen38", self.modes)
        self.assertIn("qwen38-high", offered)
        self.assertNotIn("qwen38-max", offered)

    def test_a_model_with_no_levels_offers_one_thinking_entry(self):
        """gemma reads no levels at all, so all five level words are one
        behaviour. Six names for two behaviours is exactly the noise this
        avoids."""
        modes = M.parse_modes(GEMMA)
        self.assertEqual(M.names("gemma26", modes), ["gemma26", "gemma26-low"])
        for word in ("medium", "high", "xhigh", "max"):
            self.assertTrue(M.resolve("gemma26-" + word, "gemma26", modes)[1], word)


class TestTheNamesAreDerivedFromWhatServes(unittest.TestCase):
    def setUp(self):
        self.modes = M.parse_modes(QWEN)

    def test_the_bare_alias_comes_first(self):
        self.assertEqual(M.names("qwen38", self.modes)[0], "qwen38")

    def test_one_name_per_distinct_behaviour(self):
        """Not one per declared word — see
        TestOfferingAndAcceptingAreNotTheSameList."""
        self.assertEqual(M.names("qwen38", self.modes),
                         ["qwen38", "qwen38-low", "qwen38-medium", "qwen38-high"])

    def test_another_model_yields_other_names(self):
        names = M.names("flashnext", self.modes)
        self.assertNotIn("qwen38-low", names)
        self.assertIn("flashnext-low", names)

    def test_a_model_without_modes_offers_only_itself(self):
        self.assertEqual(M.names("laguna", {}), ["laguna"])


class TestResolving(unittest.TestCase):
    def setUp(self):
        self.modes = M.parse_modes(QWEN)

    def test_a_matching_name(self):
        kw, hit = M.resolve("qwen38-medium", "qwen38", self.modes)
        self.assertTrue(hit)
        self.assertEqual(kw, {"enable_thinking": True, "reasoning_effort": "medium"})

    def test_the_bare_alias(self):
        self.assertEqual(M.resolve("qwen38", "qwen38", self.modes), ({}, True))

    def test_a_name_from_another_model(self):
        self.assertFalse(M.resolve("qwen38-low", "flashnext", self.modes)[1])

    def test_a_word_this_profile_does_not_offer(self):
        modes = M.parse_modes("low:on+low")
        self.assertFalse(M.resolve("qwen38-max", "qwen38", modes)[1])


class TestMeasuredNothingAndCouldNotMeasureAreDifferentAnswers(unittest.TestCase):
    """`TEMPLATE_LEVELS=` and a missing TEMPLATE_LEVELS line were the same
    thing, and they are not the same answer.

    systemdfile.variable returns None for both — verified against all seven
    profiles, which write the line explicitly and read back as absent. So a
    profile whose author simply forgot it got the same free pass as gpt-oss,
    whose levels genuinely cannot be measured because its template validates
    nothing. The guard that turns "an HTTP 500 for whoever selects it" into a
    load-time refusal was therefore off wherever it mattered most, silently.

    Three answers, spelled out:

        low medium xhigh   measured: these render
        no-levels          measured: this template reads no levels at all
        unmeasurable       gpt-oss — it renders any string, so nothing here
                           can tell a trained level from a typo
    """

    def test_measured_levels_are_a_set(self):
        self.assertEqual(M.parse_levels("low medium xhigh"),
                         {"low", "medium", "xhigh"})

    def test_no_levels_is_an_empty_set_not_a_missing_answer(self):
        self.assertEqual(M.parse_levels("no-levels"), set())

    def test_unmeasurable_is_its_own_answer(self):
        self.assertIsNone(M.parse_levels("unmeasurable"))

    def test_a_missing_line_is_refused_when_there_are_modes(self):
        """The profile has to say which of the three it means. Falling back to
        'check nothing' is the budget._num rule not applied one directory
        over."""
        with self.assertRaises(SystemExit) as cm:
            M.check_modes(M.parse_modes(QWEN), M.parse_levels(None))
        self.assertIn("TEMPLATE_LEVELS", str(cm.exception))

    def test_a_missing_line_is_fine_when_there_are_no_modes(self):
        """Nothing to check. Most profiles will never declare a mode."""
        M.check_modes({}, M.parse_levels(None))

    def test_no_levels_still_refuses_a_level(self):
        """The case the old empty-set could not express: a template that reads
        NO levels must reject a mode that names one."""
        with self.assertRaises(SystemExit) as cm:
            M.check_modes(M.parse_modes("low:on+low"), M.parse_levels("no-levels"))
        self.assertIn("low", str(cm.exception))

    def test_no_levels_allows_the_thinking_knob(self):
        M.check_modes(M.parse_modes(GEMMA), M.parse_levels("no-levels"))

    def test_unmeasurable_checks_nothing(self):
        M.check_modes(M.parse_modes("max:on+max"), M.parse_levels("unmeasurable"))


class TestAProfileCannotDeclareWhatItsTemplateRejects(unittest.TestCase):
    """TEMPLATE_LEVELS is the measurement — what the template actually
    renders. It exists so that a mode naming something else is caught at load,
    once, instead of being a 500 for whoever selects it."""

    def setUp(self):
        self.levels = M.parse_levels(QWEN_LEVELS)

    def test_the_measured_levels_pass(self):
        M.check_modes(M.parse_modes(QWEN), self.levels)

    def test_a_level_the_template_raises_on_is_refused(self):
        """The trap the vocabulary creates: `max` is a legal NAME and an
        illegal VALUE. `max:on+max` looks symmetrical and is a 500."""
        with self.assertRaises(SystemExit) as cm:
            M.check_modes(M.parse_modes("max:on+max"), self.levels)
        self.assertIn("max", str(cm.exception))

    def test_on_and_off_are_not_levels(self):
        M.check_modes(M.parse_modes(GEMMA), self.levels)

    def test_without_a_measurement_nothing_is_checked(self):
        """gpt-oss validates nothing, so its accepted set cannot be measured
        and its profile declares none. Refusing its modes for want of a
        measurement would refuse the one model that cannot supply one."""
        M.check_modes(M.parse_modes(QWEN), None)


if __name__ == "__main__":
    unittest.main()
