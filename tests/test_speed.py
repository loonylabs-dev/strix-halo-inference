"""speed — the copy workload, and why a second one had to exist.

A decode rate is not a property of a configuration. It is a property of the
configuration AND the workload, and the two disagree by a factor whenever a
drafter is involved. This repo learned that the expensive way: `ngram-mod` was
measured at 8.5 t/s against 8.6 without it and written down as "gives
nothing", using a probe that asks the model to COUNT — while an n-gram drafter
drafts from the PROMPT, where the digits of a counting task never appear. The
drafter was not broken; the probe could not have seen it working.

So the pieces that make the copy probe honest are tested here, and the one
that matters most is the self-check: a copy probe the model answered in prose
is not a copy-heavy number, and it has to say so rather than be averaged in.

Nothing here talks to a server. The network paths are thin and fail loudly.
"""
import sys, unittest

import common

SPD = common.load("bench/speed.py", "bench_speed")
CMP = common.load("bench/compare.py", "bench_compare")


class TestCopySource(unittest.TestCase):
    def test_it_is_deterministic(self):
        """Two runs that differ in their prompt are not two measurements."""
        self.assertEqual(SPD.copy_source(), SPD.copy_source())

    def test_the_values_are_not_guessable_from_the_shape(self):
        """If a model could produce the block WITHOUT reading it, the probe
        would measure prediction rather than copying — which is the failure it
        exists to avoid."""
        lines = SPD.copy_source(12).split("\n")
        budgets = [l.split('"budget_ms": ')[1].split(",")[0] for l in lines]
        self.assertEqual(len(set(budgets)), len(budgets),
                         "every line must carry a distinct value")

    def test_it_contains_what_the_instruction_asks_to_change(self):
        """The probe says 'change every "retries": 0 into 1'. If the block has
        no zeros, the instruction is a no-op and the answer is a pure copy —
        still useful, but not what the docstring claims is being measured."""
        self.assertIn('"retries": 0', SPD.copy_source())

    def test_it_is_long_enough_to_fill_the_generation_cap(self):
        """128 output tokens is roughly 5 lines here. Fewer lines than that
        and the model runs out of block before it runs out of budget."""
        self.assertGreaterEqual(len(SPD.copy_source().split("\n")), 10)


class TestCopiedFraction(unittest.TestCase):
    def test_an_exact_copy_is_one(self):
        src = SPD.copy_source()
        self.assertEqual(SPD.copied_fraction(src, src), 1.0)

    def test_the_requested_edit_still_counts_as_copied(self):
        """The probe ASKS for a substitution. A check that punished it would
        mark every correct answer as a failure to copy."""
        src = SPD.copy_source()
        edited = src.replace('"retries": 0', '"retries": 1')
        self.assertGreater(SPD.copied_fraction(edited, src), SPD.COPIED_MIN)

    def test_a_truncated_copy_still_counts(self):
        """max_tokens cuts the answer off mid-block. That is the normal case,
        not a defect."""
        src = SPD.copy_source()
        half = "\n".join(src.split("\n")[:5])
        self.assertGreater(SPD.copied_fraction(half, src), SPD.COPIED_MIN)

    def test_prose_and_counting_are_not_copying(self):
        src = SPD.copy_source()
        for answer in ("I cannot reproduce that block, but here is a summary.",
                       "\n".join(str(i) for i in range(1, 61))):
            self.assertLess(SPD.copied_fraction(answer, src), SPD.COPIED_MIN)

    def test_an_empty_answer_is_zero_and_not_an_exception(self):
        self.assertEqual(SPD.copied_fraction("", SPD.copy_source()), 0.0)
        self.assertEqual(SPD.copied_fraction(None, SPD.copy_source()), 0.0)


class TestAnswerText(unittest.TestCase):
    def test_the_thinking_channel_is_returned_separately(self):
        """Concatenating them would let thinking text dilute the copy check;
        ignoring it entirely is the defect that voided a battery run on
        26.08., where a model answering in `reasoning_content` scored zero."""
        r = {"choices": [{"message": {"content": "visible",
                                      "reasoning_content": "hidden"}}]}
        self.assertEqual(SPD.answer_text(r), ("visible", "hidden"))

    def test_a_missing_answer_is_two_empty_strings(self):
        self.assertEqual(SPD.answer_text({}), ("", ""))
        self.assertEqual(SPD.answer_text({"choices": [{}]}), ("", ""))


class TestProseIsTheFloor(unittest.TestCase):
    """The third workload, added 27.08. `count` and `copy` are both CEILINGS —
    one for a trained draft head, one for an n-gram drafter — so two numbers
    could only ever say how high, never how low. `prose` is novel text that
    neither drafter can predict, and it is what the hardware does with no
    speculation to hide behind."""

    def test_it_is_the_first_workload_measured(self):
        """It establishes the filler prefix, so the cells after it measure
        decode rather than a cache miss."""
        self.assertEqual(SPD.WORKLOADS[0], "prose")

    def test_the_prompt_asks_for_something_the_filler_does_not_contain(self):
        ask = SPD.payload_for("prose", 512, 128, "m")[0]["messages"][0]["content"]
        tail = ask[len(SPD.FILLER) * 2:]
        self.assertNotIn("quick brown fox", tail.split("Ignore")[-1])
        self.assertIn("Ignore the text above", ask)

    def test_a_parroted_answer_is_flagged_rather_than_reported_as_the_floor(self):
        """The mirror of the copy check, and it exists for the same reason:
        an answer that echoes its own prompt is a COPY rate wearing the
        floor's label, and a probe that cannot fail visibly is exactly the
        mistake this file is recovering from."""
        ask = SPD.payload_for("prose", 512, 128, "m")[0]["messages"][0]["content"]
        parrot = SPD.copied_fraction(SPD.FILLER * 20, ask)
        self.assertGreater(parrot, SPD.PARROT_MAX,
                           "echoing the filler has to read as parroting")
        novel = SPD.copied_fraction(
            "A bridge carries load through geometry that cannot be revised "
            "after it is poured, whereas a program is revised continuously "
            "and its structure is a claim about future change.", ask)
        self.assertLess(novel, SPD.PARROT_MAX,
                        "genuine prose must not read as parroting")

    def test_the_two_thresholds_do_not_overlap(self):
        """A single answer must not be able to fail both checks, or the two
        workloads would be judging the same text in contradictory ways."""
        self.assertLess(SPD.PARROT_MAX, SPD.COPIED_MIN)


class TestPayloads(unittest.TestCase):
    def test_every_workload_lands_at_the_same_depth(self):
        """The copy block is worth a few hundred tokens. If it were simply
        added, the copy cell would sit deeper than the count cell it is
        compared against — and decode moves with depth, which would put a
        second workload-dependent effect inside the number."""
        for depth in (8192, 32768):
            sizes = [len(SPD.payload_for(w, depth, 128, "m")[0]
                         ["messages"][0]["content"])
                     for w in SPD.WORKLOADS]
            self.assertLess((max(sizes) - min(sizes)) / max(sizes), 0.01,
                            "depths differ by more than 1 %% at %d: %r"
                            % (depth, sizes))

    def test_only_the_copy_workload_returns_a_source_block(self):
        for w in ("count", "prose"):
            self.assertEqual(SPD.payload_for(w, 512, 128, "m")[1], "", w)
        self.assertTrue(SPD.payload_for("copy", 512, 128, "m")[1])

    def test_the_copy_prompt_carries_the_block_and_the_instruction(self):
        payload, src = SPD.payload_for("copy", 512, 128, "m")
        ask = payload["messages"][0]["content"]
        self.assertIn(src, ask)
        self.assertIn("nothing else", ask)

    def test_an_unknown_workload_raises_instead_of_measuring_something_else(self):
        with self.assertRaises(ValueError):
            SPD.payload_for("rewrite", 512, 128, "m")

    def test_the_three_workloads_ask_three_different_things(self):
        """A workload that accidentally duplicated another would report a
        spread of zero and look like a finding."""
        asks = {w: SPD.payload_for(w, 512, 128, "m")[0]["messages"][0]["content"]
                for w in SPD.WORKLOADS}
        self.assertEqual(len(set(asks.values())), len(SPD.WORKLOADS))

    def test_a_shallow_depth_never_produces_an_empty_prompt(self):
        for w in SPD.WORKLOADS:
            payload, _ = SPD.payload_for(w, 1, 128, "m")
            self.assertIn(SPD.FILLER.strip(), payload["messages"][0]["content"])


class TestRepetitions(unittest.TestCase):
    """One run of this file decides nothing at shallow depth — measured.

    27.08., live qwen38, the same count cell at depth 586 in two runs six
    minutes apart: 44.6 t/s and 135.1 t/s, with draft acceptance moving
    79.9 % -> 100 %. That is the registered defect `spec-decoding-unrepeatable`
    appearing inside the instrument meant to measure it. The deep cells
    reproduced to 0.1 %; the shallow ones did not reproduce at all.
    """

    def test_the_median_is_the_middle_and_not_the_mean(self):
        """A mean lets one wild run drag the number; a median does not."""
        self.assertEqual(SPD.median([44.6, 135.1, 90.0]), 90.0)
        self.assertEqual(SPD.median([1.0, 2.0]), 1.5)

    def test_an_empty_or_all_none_input_is_none_and_not_zero(self):
        self.assertIsNone(SPD.median([]))
        self.assertIsNone(SPD.median([None, None]))

    def test_a_wild_spread_is_reported_next_to_the_median_that_hides_it(self):
        """The real pair. A median of 90 with no spread beside it would read
        as a measurement, and it is not one."""
        m = SPD.merge_reps([{"tg_tps": 44.6}, {"tg_tps": 135.1},
                            {"tg_tps": 90.0}])
        self.assertEqual(m["tg_tps"], 90.0)
        self.assertEqual((m["tg_min"], m["tg_max"]), (44.6, 135.1))
        self.assertIn("44.6-135.1", m["spread_warning"])

    def test_a_reproducible_cell_carries_no_warning(self):
        """The deep cells did reproduce. Warning on everything is warning on
        nothing."""
        m = SPD.merge_reps([{"tg_tps": 91.7}, {"tg_tps": 91.8},
                            {"tg_tps": 91.6}])
        self.assertNotIn("spread_warning", m)

    def test_one_failed_run_out_of_three_is_not_a_dead_cell(self):
        m = SPD.merge_reps([{"error": "timeout"}, {"tg_tps": 50.0},
                            {"tg_tps": 52.0}])
        self.assertEqual(m["tg_tps"], 51.0)
        self.assertEqual(m["reps_ok"], 2)
        self.assertNotIn("error", m)

    def test_all_runs_failing_stays_an_error(self):
        m = SPD.merge_reps([{"error": "a"}, {"error": "b"}])
        self.assertEqual(m["error"], "a")

    def test_prefill_is_not_medianed_across_repetitions(self):
        """Repeating a prefill does not measure the prefill again: the prefix
        is cached by run 2, so the later runs process almost nothing. A median
        over them is a cache measurement wearing a prefill label — it printed
        a row reading "0 % cached, pp 19.8 t/s", which cannot both be true."""
        m = SPD.merge_reps([
            {"pp_tps": 211.7, "cached_pct": 0.0, "tg_tps": 112.9},
            {"pp_tps": 17.9, "cached_pct": 99.6, "tg_tps": 113.1},
            {"pp_tps": 17.6, "cached_pct": 99.6, "tg_tps": 112.4}])
        self.assertEqual(m["pp_tps"], 211.7, "pp must be the cold run")
        self.assertEqual(m["cached_pct"], 0.0,
                         "the cache share must match the pp beside it")
        self.assertEqual(m["tg_tps"], 112.9, "decode IS medianed")

    def test_non_rate_fields_come_from_a_successful_run(self):
        """Depth and cache share do not vary in a way a median describes."""
        m = SPD.merge_reps([{"error": "x"},
                            {"tg_tps": 10.0, "depth_n": 9358, "cached_pct": 75.6}])
        self.assertEqual(m["depth_n"], 9358)
        self.assertEqual(m["cached_pct"], 75.6)


class TestCompareReadsTheCurrentShape(unittest.TestCase):
    """render() read summary["probes"] until 27.08., which speed.run() had
    stopped writing — so every sweep produced a table of dashes and the only
    reports it could render were the discredited wall-clock ones."""

    def summary(self, **kw):
        base = {"label": "v", "ctx": 65536, "gtt_gib": 80.4, "depths": [
            {"asked": 512, "workload": "prose", "depth_n": 624,
             "cached_pct": 0.0, "pp_tps": 184.0, "tg_tps": 9.9,
             "draft_accept_pct": 1.0, "copied_pct": 3.0},
            {"asked": 512, "workload": "count", "depth_n": 626,
             "cached_pct": 0.0, "pp_tps": 183.2, "tg_tps": 13.7,
             "draft_accept_pct": 4.0},
            {"asked": 512, "workload": "copy", "depth_n": 631,
             "cached_pct": 0.0, "pp_tps": 180.0, "tg_tps": 31.4,
             "draft_accept_pct": 79.0, "copied_pct": 96.7},
        ]}
        base.update(kw)
        return base

    def test_both_decode_columns_appear(self):
        md = CMP.render_summaries([self.summary()])
        self.assertIn("9.9", md, "the prose floor is missing from the table")
        self.assertIn("13.7", md)
        self.assertIn("31.4", md, "the copy rate is missing from the table")
        self.assertIn("1.0/4.0/79.0", md,
                      "draft acceptance per workload is missing")

    def test_one_row_per_depth(self):
        s = self.summary()
        s["depths"] += [
            {"asked": 8192, "workload": "count", "depth_n": 9124, "tg_tps": 11.8},
            {"asked": 8192, "workload": "copy", "depth_n": 9130, "tg_tps": 28.0},
        ]
        md = CMP.render_summaries([s])
        self.assertIn("9124", md)
        self.assertIn("28.0", md)

    def test_a_copy_probe_the_model_refused_is_flagged_not_averaged(self):
        s = self.summary()
        s["depths"][1]["warning"] = "only 3.0 % of the answer was copied"
        md = CMP.render_summaries([s])
        self.assertIn("do not mean what the column says", md)
        self.assertIn("only 3.0 %", md)
        self.assertIn("copy", md)

    def test_an_old_wall_clock_report_is_rendered_but_marked(self):
        """Refusing to show it helps nobody; showing it unmarked next to
        server-clock numbers is how the two get compared."""
        old = {"label": "legacy", "gtt_gib": 35.8,
               "probes": {"prefill_cold": {"tps": 205.7},
                          "decode_warm": {"tps": 20.0}}}
        md = CMP.render_summaries([old])
        self.assertIn("205.7", md)
        self.assertIn("legacy *", md)
        self.assertIn("wall clock", md)

    def test_a_failed_cell_is_a_gap_and_not_a_zero(self):
        s = self.summary()
        # By WORKLOAD, not by index: the list gained `prose` at position 0 on
        # 27.08. and an index-based fixture silently started replacing a
        # different cell than it named.
        s["depths"] = [c for c in s["depths"] if c["workload"] != "copy"]
        s["depths"].append({"asked": 512, "workload": "copy", "error": "timeout"})
        md = CMP.render_summaries([s])
        self.assertIn("no measurement", md)
        self.assertNotIn("| 0.0 |", md)


if __name__ == "__main__":
    unittest.main()
