"""stopstring_ab's verdict — what counts as the stop string ending a turn.

The tool detected an emptied turn by its content AND by zeroed timings, the
signature of the stop-string path up to Halogen 0.13.8 (public issue #106).
0.14.0 fixed #106, so the same failure now reports real timings — and the
summary read 0 empty turns on 26.09.2026 while 6 of 6 stop-string turns had
ended with no content (bench/reports/2026-09-26_halogen-0.14.0/). A fix
upstream made a detector here blind, and it printed a clean result.
"""
import unittest

import common

stopstring_ab = common.load(str(common.REPO / "bench" / "stopstring_ab.py"),
                            "stopstring_ab")


def row(arm, content, zeroed, tools=0, end_token=False):
    return {"arm": arm, "content_chars": content, "tool_calls": tools,
            "timings_zeroed": zeroed, "reasoning_has_end_token": end_token,
            "took_s": 1.0}


class TestEmptyTurnsAreCountedWhateverTheTimingsSay(unittest.TestCase):

    def test_an_empty_turn_with_real_timings_is_empty(self):
        """The 0.14.0 shape: no content, no tool call, timings not zeroed."""
        s = stopstring_ab.summarize([row("gateway-stops", 0, False)] * 3)
        self.assertEqual(s["gateway-stops"]["empty"], 3)
        self.assertEqual(s["gateway-stops"]["zeroed"], 0)

    def test_the_old_signature_still_counts_and_says_so(self):
        s = stopstring_ab.summarize([row("gateway-stops", 0, True)] * 2)
        self.assertEqual(s["gateway-stops"]["empty"], 2)
        self.assertEqual(s["gateway-stops"]["zeroed"], 2)

    def test_a_tool_call_or_content_is_not_empty(self):
        s = stopstring_ab.summarize([row("no-stops", 0, False, tools=1),
                                     row("no-stops", 12, False)])
        self.assertEqual(s["no-stops"]["empty"], 0)

    def test_errors_are_not_counted_as_turns(self):
        s = stopstring_ab.summarize([dict(row("no-stops", 0, False), error="x")])
        self.assertEqual(s["no-stops"]["runs"], 1)
        self.assertEqual(s["no-stops"]["errors"], 1)
        self.assertEqual(s["no-stops"]["empty"], 0)


if __name__ == "__main__":
    unittest.main()
