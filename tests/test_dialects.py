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

D = common.load("setup/claude/dialects.py", "dialects")
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


if __name__ == "__main__":
    unittest.main()
