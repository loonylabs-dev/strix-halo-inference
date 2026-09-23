"""test_colonstop — the stall replay's pure parts: what counts as a stall, what
the replayed request carries, and that private prompts cannot land in the repo."""
import subprocess
import sys
import unittest
import unittest.mock

import common

REPO = common.REPO
CS = common.load("bench/colonstop.py", "colonstop")


def conv(*msgs):
    return {"model": "local-low", "system": [{"type": "text", "text": "sys"}],
            "messages": list(msgs), "enable_thinking": True,
            "reasoning_effort": "low", "stop": ["<|im_end|>"],
            "tools": [{"name": "Edit", "input_schema": {"type": "object"}}]}


USER = {"role": "user", "content": [{"type": "text", "text": "do it"}]}
STALL = {"role": "assistant", "content": [
    {"type": "thinking", "thinking": "Let me edit."},
    {"type": "text", "text": "Jetzt die Änderungen:"}]}
CALL = {"role": "assistant", "content": [
    {"type": "text", "text": "Jetzt:\n\n"},
    {"type": "tool_use", "id": "t1", "name": "Edit", "input": {}}]}


class TestClassify(unittest.TestCase):
    def test_the_three_outcomes(self):
        self.assertEqual(CS.classify(""), "stop")
        self.assertEqual(CS.classify("\n\n<tool_call>\n<function=Edit>"), "tool")
        self.assertEqual(CS.classify("Ein Punkt noch:"), "other")

    def test_whitespace_alone_is_still_a_stop(self):
        # the model emitting '\n' then <|im_end|> ended the turn all the same
        self.assertEqual(CS.classify(" \n"), "stop")


class TestStall(unittest.TestCase):
    def test_returns_its_own_thinking_and_text(self):
        self.assertEqual(CS.stall(conv(USER, STALL), 1),
                         ("Let me edit.", "Jetzt die Änderungen:"))

    def test_a_turn_that_called_a_tool_is_refused(self):
        with self.assertRaises(SystemExit):
            CS.stall(conv(USER, CALL), 1)

    def test_a_user_turn_is_refused(self):
        with self.assertRaises(SystemExit):
            CS.stall(conv(USER, STALL), 0)

    def test_an_index_past_the_trace_is_refused_by_name(self):
        with self.assertRaises(SystemExit) as cm:
            CS.stall(conv(USER, STALL), 5)
        self.assertIn("only 2 messages", str(cm.exception))


class TestOpenaiBody(unittest.TestCase):
    def test_the_history_ends_before_the_stall(self):
        o = CS.openai_body(conv(USER, STALL), 1)
        roles = [m["role"] for m in o["messages"]]
        self.assertEqual(roles, ["system", "user"])

    def test_it_carries_what_the_gateway_copies_after_translating(self):
        o = CS.openai_body(conv(USER, STALL), 1)
        self.assertIs(o["enable_thinking"], True)
        self.assertEqual(o["reasoning_effort"], "low")
        self.assertEqual(o["stop"], ["<|im_end|>"])
        self.assertTrue(o["tools"])


class TestRenderKeepsTheBytes(unittest.TestCase):
    """A Windows client's CLAUDE.md arrives with \\r\\n, and the prompt has to
    reach the server as those bytes. Text-mode subprocess output folds \\r\\n
    into \\n — found 23.09.2026 as a 1,276-char difference from the prompt the
    measurement actually used; every token after the first CR would have
    differed, silently."""

    def test_crlf_survives_the_container_round_trip(self):
        import os
        import tempfile
        d = tempfile.mkdtemp()
        fake = os.path.join(d, "podman")
        with open(fake, "w") as f:
            f.write("#!/bin/sh\ncat >/dev/null\nprintf 'a\\r\\nb%s' "
                    "'<|im_start|>assistant\n<think>\n'\n")
        os.chmod(fake, 0o755)
        with unittest.mock.patch.dict(os.environ,
                                      {"PATH": d + os.pathsep + os.environ["PATH"]}):
            out = CS.render("any", {"messages": []})
        self.assertTrue(out.startswith("a\r\nb"), repr(out[:10]))

    def test_crlf_survives_the_prompt_cache(self):
        import pathlib
        import tempfile
        p = pathlib.Path(tempfile.mkdtemp()) / "p.txt"
        self.assertEqual(CS.cached(p, lambda: "a\r\nb"), "a\r\nb")   # written
        self.assertEqual(CS.cached(p, lambda: "unused"), "a\r\nb")   # read back


class TestPrivacy(unittest.TestCase):
    def test_a_cache_inside_the_repo_is_refused(self):
        r = subprocess.run(
            [sys.executable, str(REPO / "bench" / "colonstop.py"),
             "--trace", "/nonexistent", "--prefix", "x", "--at", "1",
             "--label", "l", "--out", "/tmp/unused-colonstop-out",
             "--cache", str(REPO / "bench" / "reports")],
            capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("inside the repo", r.stderr)


if __name__ == "__main__":
    unittest.main()
