"""Tests for bench/suites/slot-corruption.py — the instrument for defect 1.

The suite that decides whether a build corrupts had no tests at all until
29.08.2026, which is a strange place for the gap to be: every "0 of N
CORRUPT" in setup/defects.json was read off it, and a verdict function that
mislabels an answer turns a measurement into a wrong sentence in the
registry.

The gap that prompted them: an answer carrying `tool_calls` and no text
reaches `verdict()` as the empty string, indistinguishable from a server
that returned nothing at all. The bodies define ten tools, so this is not
an exotic case — it is the model doing what the tools invite. Five of the
twelve answers of the 29.08. HIP_LAUNCH_BLOCKING run landed in that bucket.
"""
import unittest

import common

SC = common.load("bench/suites/slot-corruption.py", "slotcorruption")


class TestVerdict(unittest.TestCase):
    def test_the_nonce_coming_back_is_ok(self):
        self.assertEqual(SC.verdict({"content": "A-1234-0"}, "A-1234-0"), "ok")

    def test_slashes_are_corrupt(self):
        self.assertEqual(SC.verdict({"content": "//////////"}, "A-1"), "CORRUPT")
        self.assertEqual(
            SC.verdict({"content": "x " + "/" * 9}, "A-1"), "CORRUPT")

    def test_a_wrong_answer_is_other(self):
        self.assertEqual(SC.verdict({"content": "Hello there"}, "A-1"), "other")

    def test_nothing_at_all_is_empty(self):
        self.assertEqual(SC.verdict({"content": ""}, "A-1"), "empty")
        self.assertEqual(SC.verdict({"content": None}, "A-1"), "empty")

    def test_a_tool_call_is_not_an_empty_answer(self):
        """The blind spot. A model that answers with a tool call has
        answered — wrongly for this prompt, but it produced tokens, and the
        server logs show them. Counting that as `empty` reads as a server
        that returned nothing, which is a different and much more alarming
        thing."""
        m = {"content": None,
             "tool_calls": [{"type": "function",
                             "function": {"name": "T00_A", "arguments": "{}"}}]}
        self.assertEqual(SC.verdict(m, "A-1"), "tool-call")

    def test_a_tool_call_beside_slashes_is_still_corruption(self):
        """Corruption wins over every other label — it is what the suite
        exists to see."""
        m = {"content": "////////////",
             "tool_calls": [{"type": "function",
                             "function": {"name": "T00_A", "arguments": "{}"}}]}
        self.assertEqual(SC.verdict(m, "A-1"), "CORRUPT")


if __name__ == "__main__":
    unittest.main()
