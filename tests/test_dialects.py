"""Tests for dialects.py — the shared reading of request bodies.

This module decides two things that fail silently when they are wrong:

  * the prefix id. Too eager and two different projects share a slot and
    destroy each other's cache; too jumpy and every turn runs cold. Nobody
    gets an error either way — you only notice by the clock.
  * the shape a body keeps. A rewrite that hands Anthropic blocks to an
    OpenAI endpoint (or a bare string to Anthropic) produces a 400 at best
    and a silently truncated prompt at worst.

Both dialects are held against the same expectations here, and
test_gateway.py::TestIdContract additionally holds the gateway and prewarm
against each other.
"""
import re
import unittest

import common

D = common.load("setup/gateway/dialects.py", "dialects")
SYN = common.load("tools/synthetic.py", "synthetic")

VOLATILE = [re.compile(r"<total_tokens>\s*\d+\s*tokens left\s*</total_tokens>")]

TOOLS_OAI = [{"type": "function",
              "function": {"name": "Read", "description": "read a file",
                           "parameters": {"type": "object"}}}]


def oai_body(system="You are an agent.", question="hi", tools=None,
             extra_messages=()):
    """A dsh-shaped OpenAI request body."""
    return {"model": "qwen38",
            "messages": ([{"role": "system", "content": system},
                          {"role": "user", "content": question}]
                         + list(extra_messages)),
            "tools": list(tools if tools is not None else TOOLS_OAI)}


class TestDetection(unittest.TestCase):
    def test_the_path_decides_not_the_body(self):
        self.assertEqual(D.detect("/v1/chat/completions"), D.OPENAI)
        self.assertEqual(D.detect("/v1/chat/completions?x=1"), D.OPENAI)
        self.assertEqual(D.detect("/v1/messages"), D.ANTHROPIC)
        self.assertEqual(D.detect("/v1/messages/count_tokens"), D.ANTHROPIC)
        self.assertEqual(D.detect(""), D.ANTHROPIC)

    def test_inference_paths_but_not_token_counting(self):
        self.assertTrue(D.is_inference("/v1/messages"))
        self.assertTrue(D.is_inference("/v1/chat/completions"))
        self.assertFalse(D.is_inference("/v1/messages/count_tokens"))
        self.assertFalse(D.is_inference("/v1/models"))
        self.assertFalse(D.is_inference("/completion"))


class TestSystemHead(unittest.TestCase):
    def test_anthropic_string_and_blocks(self):
        self.assertEqual(D.system_head({"system": "abc"}, D.ANTHROPIC), "abc")
        self.assertEqual(
            D.system_head({"system": [{"type": "text", "text": "a"},
                                      {"type": "text", "text": "b"}]},
                          D.ANTHROPIC), "ab")

    def test_openai_reads_the_leading_system_message(self):
        self.assertEqual(D.system_head(oai_body(system="abc"), D.OPENAI), "abc")

    def test_openai_content_parts_count_too(self):
        b = oai_body()
        b["messages"][0]["content"] = [{"type": "text", "text": "xy"}]
        self.assertEqual(D.system_head(b, D.OPENAI), "xy")

    def test_a_body_without_a_system_prompt_is_empty_not_an_error(self):
        self.assertEqual(D.system_head({"messages": [
            {"role": "user", "content": "hi"}]}, D.OPENAI), "")
        self.assertEqual(D.system_head({}, D.ANTHROPIC), "")


class TestPrefixId(unittest.TestCase):
    def test_the_question_does_not_change_the_id(self):
        a = D.prefix_id(oai_body(question="alpha"), D.OPENAI)
        b = D.prefix_id(oai_body(question="beta"), D.OPENAI)
        self.assertEqual(a, b)

    def test_system_and_tools_do_change_it(self):
        base = D.prefix_id(oai_body(), D.OPENAI)
        self.assertNotEqual(base, D.prefix_id(oai_body(system="other"),
                                              D.OPENAI))
        self.assertNotEqual(base, D.prefix_id(oai_body(tools=[]), D.OPENAI))

    def test_the_thinking_mode_changes_the_id(self):
        """MEASURED 28.08.2026 against the running qwen38 server: the three
        modes the gateway offers under one loaded model render three different
        prompts, and they diverge at CHARACTER 19 — the template puts
        `Reasoning effort is set to low.` at the very front, before the tools.

            off (server default)  sha fe8d7ee8   1108 chars
            think  (low)          sha 677f3ace   1235 chars
            deep   (medium)       sha aa6c7b7d   1097 chars

        The id ignored chat_template_kwargs, so all three shared one key. What
        that costs is not theoretical: prewarm renders and saves ONE of them,
        `ANTHROPIC_MODEL=qwen38-think` asks for another, the slot is restored
        from a state that diverges at token ~5, almost everything is
        re-prefilled — and the gateway logs RESTORED and counts it warm. The
        warm percentages this repo reasons from were measuring the wrong
        thing, and setup/env/qwen38.env's "a mode switch keeps the prompt
        cache 100 % warm" can only have been measured between the two modes
        that happen to render almost identically.
        """
        base = oai_body()
        off = dict(base, chat_template_kwargs={"enable_thinking": False})
        low = dict(base, chat_template_kwargs={"enable_thinking": True,
                                               "reasoning_effort": "low"})
        med = dict(base, chat_template_kwargs={"enable_thinking": True,
                                               "reasoning_effort": "medium"})
        ids = {D.prefix_id(b, D.OPENAI)[0] for b in (base, off, low, med)}
        self.assertEqual(len(ids), 4,
                         "modes that render differently must not share a slot")

    def test_the_kwargs_are_read_in_a_stable_order(self):
        """Two dicts with the same pairs in a different order are the same
        request. A key that depends on dict order would run cold at random."""
        a = dict(oai_body(), chat_template_kwargs={"enable_thinking": True,
                                                   "reasoning_effort": "low"})
        b = dict(oai_body(), chat_template_kwargs={"reasoning_effort": "low",
                                                   "enable_thinking": True})
        self.assertEqual(D.prefix_id(a, D.OPENAI), D.prefix_id(b, D.OPENAI))

    def test_a_body_without_kwargs_is_not_the_same_as_an_empty_map(self):
        """Absent means "the server's command line decides"; {} means the
        same thing. They must not be two keys for one rendering."""
        a = oai_body()
        b = dict(oai_body(), chat_template_kwargs={})
        self.assertEqual(D.prefix_id(a, D.OPENAI), D.prefix_id(b, D.OPENAI))

    def test_the_two_dialects_do_not_share_an_id(self):
        """Same logical prompt, different rendering — they must not land in
        the same slot, or they would evict each other every turn."""
        ant = {"system": "You are an agent.",
               "tools": [{"name": "Read", "description": "read a file",
                          "input_schema": {"type": "object"}}]}
        self.assertNotEqual(D.prefix_id(ant, D.ANTHROPIC)[0],
                            D.prefix_id(oai_body(), D.OPENAI)[0])

    def test_head_id_survives_a_changed_tail(self):
        long_head = "x" * (D.HEAD_BYTES + 100)
        a = D.prefix_id(oai_body(system=long_head + "A"), D.OPENAI)
        b = D.prefix_id(oai_body(system=long_head + "B"), D.OPENAI)
        self.assertNotEqual(a[0], b[0], "full ids must differ")
        self.assertEqual(a[1], b[1], "head ids must match — that is the point")


class TestHoisting(unittest.TestCase):
    def test_openai_keeps_the_leading_system_message_in_place(self):
        b = oai_body(extra_messages=[
            {"role": "system", "content": "AGENTS\n<total_tokens>5 tokens left</total_tokens>"}])
        out, n_vol = D.hoist_system_messages(b, D.OPENAI, VOLATILE)
        self.assertEqual(out["messages"][0]["role"], "system")
        self.assertIn("AGENTS", D.system_head(out, D.OPENAI))
        self.assertEqual(n_vol, 1)
        tail = out["messages"][-1]
        self.assertIn("tokens left", D.blocks_to_text(tail["content"]))
        self.assertNotIn("AGENTS", D.blocks_to_text(tail["content"]))

    def test_the_prefix_stays_equal_across_turns_in_both_dialects(self):
        """The whole point of hoisting: a changed counter must not move the
        id, or every turn runs cold."""
        for dialect, mk in ((D.OPENAI, lambda n: oai_body(extra_messages=[
                {"role": "system",
                 "content": "AGENTS\n<total_tokens>%d tokens left</total_tokens>" % n}])),
                            (D.ANTHROPIC, lambda n: SYN.body(budget_left=n))):
            with self.subTest(dialect=dialect):
                a, _ = D.hoist_system_messages(mk(15000000), dialect, VOLATILE)
                b, _ = D.hoist_system_messages(mk(14211873), dialect, VOLATILE)
                self.assertEqual(D.prefix_id(a, dialect),
                                 D.prefix_id(b, dialect))

    def test_a_body_without_mid_system_messages_is_untouched(self):
        before = oai_body()
        after, n = D.hoist_system_messages(oai_body(), D.OPENAI, VOLATILE)
        self.assertEqual(after, before)
        self.assertEqual(n, 0)

    def test_anthropic_hoisting_keeps_cache_control_blocks(self):
        """Claude Code marks its system blocks with cache_control; rewriting
        them into one string would drop those markers."""
        b = {"system": [{"type": "text", "text": "head",
                         "cache_control": {"type": "ephemeral"}}],
             "messages": [{"role": "user", "content": "q"},
                          {"role": "system", "content": "AGENTS"}]}
        out, _ = D.hoist_system_messages(b, D.ANTHROPIC, VOLATILE)
        self.assertEqual(out["system"][0]["cache_control"],
                         {"type": "ephemeral"})
        self.assertIn("AGENTS", D.system_head(out, D.ANTHROPIC))


class TestMidSystemToUser(unittest.TestCase):
    def test_openai_content_stays_a_string(self):
        b = oai_body(extra_messages=[{"role": "system", "content": "later"}])
        out, n = D.mid_system_to_user(b, D.OPENAI)
        self.assertEqual(n, 1)
        self.assertEqual(out["messages"][-1],
                         {"role": "user", "content": "later"})
        self.assertEqual(out["messages"][0]["role"], "system")

    def test_anthropic_content_becomes_blocks(self):
        b = {"messages": [{"role": "user", "content": "q"},
                          {"role": "system", "content": "later"}]}
        out, n = D.mid_system_to_user(b, D.ANTHROPIC)
        self.assertEqual(n, 1)
        self.assertEqual(out["messages"][-1]["content"],
                         [{"type": "text", "text": "later"}])

    def test_the_prefix_id_does_not_move(self):
        for dialect, b in ((D.OPENAI, oai_body(extra_messages=[
                                {"role": "system", "content": "later"}])),
                           (D.ANTHROPIC, SYN.body(turns=2))):
            with self.subTest(dialect=dialect):
                before = D.prefix_id(b, dialect)
                out, _ = D.mid_system_to_user(b, dialect)
                self.assertEqual(D.prefix_id(out, dialect), before)


class TestTemplatePayloadCarriesTheMode(unittest.TestCase):
    """prewarm renders through this to decide what to save. If it renders
    without the mode, it saves the server-default prompt whatever mode the
    session that triggered the save was in — and the file can then never match
    the request that caused it."""

    def test_the_kwargs_reach_the_render(self):
        body = dict(oai_body(), chat_template_kwargs={"reasoning_effort": "low"})
        pay = D.template_payload(body, D.OPENAI)
        self.assertEqual(pay.get("chat_template_kwargs"),
                         {"reasoning_effort": "low"})

    def test_a_body_without_them_stays_as_it_was(self):
        self.assertNotIn("chat_template_kwargs",
                         D.template_payload(oai_body(), D.OPENAI))


class TestTemplatePayload(unittest.TestCase):
    def test_openai_tools_pass_through_unchanged(self):
        p = D.template_payload(oai_body(), D.OPENAI)
        self.assertEqual(p["tools"], TOOLS_OAI)
        self.assertEqual(p["messages"][0]["role"], "system")
        self.assertEqual(p["messages"][1]["content"], "X")

    def test_anthropic_tools_are_converted(self):
        p = D.template_payload(
            {"system": "s",
             "tools": [{"name": "Read", "description": "d",
                        "input_schema": {"type": "object"}}]}, D.ANTHROPIC)
        self.assertEqual(p["tools"], [{
            "type": "function",
            "function": {"name": "Read", "description": "d",
                         "parameters": {"type": "object"}}}])

    def test_a_real_claude_code_body_renders_without_losing_tools(self):
        p = D.template_payload(SYN.body(n_tools=7), D.ANTHROPIC)
        self.assertEqual(len(p["tools"]), 7)
        self.assertTrue(all(t["type"] == "function" for t in p["tools"]))


class TestWhatEachMessageWeighs(unittest.TestCase):
    """`message_shape` says THAT a message changed. On 29.08.2026 that was not
    enough: an 18,450-token re-prefill could have been one re-rendered line or
    a third of the conversation compacted away, and the trace could not tell
    them apart. The size is the difference."""

    def body(self, *texts):
        return {"messages": [{"role": "user", "content": t} for t in texts]}

    def test_it_carries_a_hash_and_a_length_per_message(self):
        fp = D.message_fingerprints(self.body("hello", "a longer message"))
        self.assertEqual(len(fp), 2)
        for h, n in fp:
            self.assertRegex(h, r"^[0-9a-f]{8}$")
            self.assertIsInstance(n, int)
        self.assertLess(fp[0][1], fp[1][1])

    def test_the_shape_is_exactly_its_hashes(self):
        """Two functions that disagree would put a wrong index in the log."""
        b = self.body("one", "two", "three")
        self.assertEqual(D.message_shape(b),
                         [h for h, _ in D.message_fingerprints(b)])

    def test_a_same_length_edit_changes_the_hash_but_not_the_size(self):
        """The re-render case — and the one a length alone would miss."""
        a = D.message_fingerprints(self.body("counter: 41"))[0]
        b = D.message_fingerprints(self.body("counter: 42"))[0]
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(a[1], b[1], "same length, so only the hash may differ")

    def test_no_messages_is_an_empty_list_not_an_error(self):
        self.assertEqual(D.message_fingerprints({}), [])


class TestACacheHintIsNotARewrittenHistory(unittest.TestCase):
    """`cache_control` is Anthropic's own cache breakpoint, and Claude Code
    MOVES it forward as a conversation grows. It never reaches the rendered
    prompt — but `message_shape` hashed the whole message JSON, so a message
    whose text had not changed by one character read as a rewrite.

    Measured 30.08.2026, two consecutive live turns: message 1 identical in all
    5,100 characters of its text and differing ONLY by the presence of
    {"cache_control": {"type": "ephemeral"}}. `msgs_kept` was logged as 4 of 5;
    recomputed without the hint it is 5 of 5, a pure append. The instrument
    whose entire purpose is telling "the state was lost" from "the client
    rewrote its history" was answering the second when the truth was neither.
    """

    def msg(self, text, hint=False):
        block = {"type": "text", "text": text}
        if hint:
            block["cache_control"] = {"type": "ephemeral"}
        return {"role": "user", "content": [block]}

    def test_a_moved_breakpoint_is_not_a_change(self):
        a = D.message_shape({"messages": [self.msg("gleich", hint=True)]})
        b = D.message_shape({"messages": [self.msg("gleich")]})
        self.assertEqual(a, b)

    def test_a_changed_text_still_is(self):
        a = D.message_shape({"messages": [self.msg("eins", hint=True)]})
        b = D.message_shape({"messages": [self.msg("zwei", hint=True)]})
        self.assertNotEqual(a, b)

    def test_it_is_stripped_at_the_top_level_too(self):
        a = D.renderable({"role": "user", "content": "x",
                          "cache_control": {"type": "ephemeral"}})
        self.assertNotIn("cache_control", a)

    def test_the_original_is_not_mutated(self):
        """The body is on its way to llama-server; dropping the hint from it
        would change what Claude Code asked for."""
        m = self.msg("x", hint=True)
        D.renderable(m)
        self.assertIn("cache_control", m["content"][0])

    def test_only_fields_known_not_to_render_are_dropped(self):
        """Conservative on purpose: a field wrongly kept costs a false
        `rewritten`, a field wrongly dropped HIDES a real one."""
        self.assertEqual(D.IGNORED_IN_SHAPE, ("cache_control",))

    def test_an_unserialisable_message_still_does_not_raise(self):
        shape = D.message_shape({"messages": [{"role": "user", "content": object()}]})
        self.assertEqual(len(shape), 1)


class TestSpokenTextOutOfARawStream(unittest.TestCase):
    """The gateway's sniff holds raw SSE. Everything below depends on getting
    the model's words back out of it first."""

    def test_an_anthropic_stream(self):
        sse = ('data: {"type":"content_block_delta","delta":'
               '{"type":"text_delta","text":"Hello "}}\n\n'
               'data: {"type":"content_block_delta","delta":'
               '{"type":"text_delta","text":"world"}}\n\n')
        self.assertEqual(D.spoken_text(sse), "Hello world")

    def test_an_openai_stream(self):
        sse = ('data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
               'data: {"choices":[{"delta":{"content":"world"}}]}\n\n')
        self.assertEqual(D.spoken_text(sse), "Hello world")

    def test_a_non_streamed_body_of_either_dialect(self):
        self.assertEqual(
            D.spoken_text('{"content":[{"type":"text","text":"391"}]}'), "391")
        self.assertEqual(
            D.spoken_text('{"choices":[{"message":{"content":"391"}}]}'), "391")

    def test_truncated_json_costs_that_event_and_nothing_else(self):
        """It is fed the head of a stream by design, so the last event is
        usually cut in half. A gateway that raised over its own bookkeeping
        would be worse than one that reads a little less."""
        sse = ('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
               'data: {"choices":[{"delta":{"cont')
        self.assertEqual(D.spoken_text(sse), "ok")

    def test_nothing_readable_is_empty_not_an_exception(self):
        for junk in ("", ": keep-alive\n\n", "data: [DONE]\n\n", None):
            self.assertEqual(D.spoken_text(junk), "")


class TestTheHistogramMustNotRunOnRawFrames(unittest.TestCase):
    """The trap this design exists around, pinned.

    A degenerate answer arrives as hundreds of one-character deltas, each
    wrapped in JSON. Over the RAW frames the dominant character is the quote
    or the brace, not the slash — so a check applied straight to the sniff
    reads every corrupted answer as healthy and never says why. Silent, and in
    the detector for silent faults.
    """

    def raw(self, n=200):
        return "".join('data: {"choices":[{"delta":{"content":"/"}}]}\n\n'
                       for _ in range(n))

    def test_the_raw_frames_look_healthy_which_is_the_trap(self):
        self.assertFalse(D.looks_degenerate(self.raw())[0])

    def test_the_extracted_answer_does_not(self):
        bad, why = D.looks_degenerate(D.spoken_text(self.raw()))
        self.assertTrue(bad)
        self.assertIn("100%", why)


class TestDegeneracyAgainstRecordedTraffic(unittest.TestCase):
    """Not invented examples: every answer this machine recorded during the
    slot-corruption runs, with the verdict it was given at the time."""

    @classmethod
    def setUpClass(cls):
        import glob, json
        cls.dirty, cls.clean = [], []
        for f in sorted(glob.glob(
                str(common.REPO / "bench/reports/*slot-corruption*/*.json"))):
            for run in json.load(open(f, encoding="utf-8")).get("runs", []):
                for a in run.get("answers") or []:
                    o = a if isinstance(a, dict) else (
                        json.loads(a) if isinstance(a, str) else None)
                    if not isinstance(o, dict):
                        continue
                    tgt = (cls.dirty if str(o.get("verdict")).upper() == "CORRUPT"
                           else cls.clean)
                    tgt.append(str(o.get("text", "")))

    def test_the_corpus_is_there(self):
        """If the reports move or are pruned, the two tests below would pass
        by having nothing to judge."""
        self.assertGreaterEqual(len(self.dirty), 300)
        self.assertGreaterEqual(len(self.clean), 300)

    def test_every_recorded_corruption_is_found(self):
        missed = [t for t in self.dirty if not D.looks_degenerate(t)[0]]
        self.assertEqual(missed, [], "%d of %d corrupted answers slipped "
                                     "through" % (len(missed), len(self.dirty)))

    def test_no_healthy_answer_is_accused(self):
        """The expensive direction. A watchdog that cries wolf gets switched
        off, and then the real fault arrives to an empty room."""
        wrong = [t for t in self.clean if D.looks_degenerate(t)[0]]
        self.assertEqual(wrong, [], "%d of %d healthy answers were called "
                                    "degenerate" % (len(wrong), len(self.clean)))


class TestTheShapesRealAnswersContain(unittest.TestCase):
    """Hand-built, because the recorded corpus cannot prove the absence of a
    shape it happens not to contain. These are what a histogram trips over."""

    def ok(self, name, text):
        self.assertFalse(D.looks_degenerate(text)[0], name)

    def test_a_markdown_rule_inside_prose(self):
        self.ok("rule", "Here is the answer.\n\n" + "-" * 80 + "\n\nAnd more.")

    def test_an_ascii_table(self):
        self.ok("table", "| a | b |\n" + "|---|---|\n" * 12)

    def test_base64_and_hex(self):
        """Dominated by one character, but alphanumeric — which the signature
        of this fault never is."""
        self.ok("base64", "A" * 64)
        self.ok("hex", "0" * 64)

    def test_a_path_listing_and_long_prose(self):
        self.ok("paths", "\n".join("/usr/lib/x%d/y/z" % i for i in range(12)))
        self.ok("prose", "The cache holds the prefix in the slot. " * 12)

    def test_a_short_answer_is_never_judged(self):
        """391 is three characters. Below the floor there is no evidence
        either way — and the floor cannot be raised, because the corrupted
        answers have a median length of 120."""
        self.assertFalse(D.looks_degenerate("391")[0])
        self.assertFalse(D.looks_degenerate("////")[0])

    def test_the_bare_signature_at_full_length_is_caught(self):
        self.assertTrue(D.looks_degenerate("/" * 200)[0])
        self.assertTrue(D.looks_degenerate("The answer is " + "/" * 300)[0])


if __name__ == "__main__":
    unittest.main()
