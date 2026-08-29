"""The trace viewer — what it must not serve, and what it must not need.

Two properties carry this file. The first is that a row's TEXT is not sent
until the row is clicked: the table ships sizes, the click ships content. A
viewer that hands out whole prompts by default is one careless port-forward
away from being the /slots exposure again, and docs/SECURITY.md calls that the
worst finding this project has had.

The second is that the page needs nothing from the internet. No CDN, no font,
no script — partly because this machine's whole point is running without one,
and partly because a debugging tool that breaks when the network does is not a
debugging tool.
"""
import json, os, re, shutil, tempfile, unittest

import common

UI = common.load("tools/tracelog.py", "tracelog_cli")
PAGE = common.REPO / "tools" / "traceui.html"


class TestTheServerDoesNotHandOutText(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="traceui-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = os.path.join(self.dir, "trace-2026-08-29.jsonl")
        with open(self.path, "w", encoding="utf-8") as f:
            for i in range(3):
                f.write(json.dumps({
                    "t": 1000 + i, "kind": "request", "prefix": "p%d" % i,
                    "system_head": "SECRET PROMPT %d" % i,
                    "answer_tail": "SECRET ANSWER %d" % i}) + "\n")

    def test_events_replace_text_with_its_size(self):
        recs, _ = UI.read_since(self.path, 0)
        self.assertEqual(len(recs), 3,
                         "nothing was read, so nothing was checked")
        for r in recs:
            self.assertNotIn("SECRET", json.dumps(r))
            self.assertIn("characters", r["system_head"])
            self.assertIn("click", r["system_head"])

    def test_one_record_can_be_fetched_in_full(self):
        """That is what the click does — and it has to be a DIFFERENT call, so
        the table can never carry the text by accident."""
        recs, _ = UI.read_since(self.path, 0)
        full = UI.read_one(self.path, recs[1]["_at"])
        self.assertEqual(full["system_head"], "SECRET PROMPT 1")

    def test_every_record_carries_where_it_is(self):
        recs, off = UI.read_since(self.path, 0)
        self.assertEqual(len(recs), 3)
        self.assertEqual(off, os.path.getsize(self.path))
        self.assertTrue(all(isinstance(r["_at"], int) for r in recs))

    def test_reading_continues_where_it_stopped(self):
        first, off = UI.read_since(self.path, 0)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": 2000, "kind": "save", "prefix": "x"}) + "\n")
        more, _ = UI.read_since(self.path, off)
        self.assertEqual([r["kind"] for r in more], ["save"])

    def test_a_half_written_line_is_left_for_next_time(self):
        """The gateway appends while this reads. A record that is not finished
        yet must not be parsed, and must not be skipped either."""
        _, off = UI.read_since(self.path, 0)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"t": 3000, "kind": "req')
        recs, new_off = UI.read_since(self.path, off)
        self.assertEqual(recs, [])
        self.assertEqual(new_off, off, "the offset must not move past a fragment")

    def test_a_truncated_file_restarts_rather_than_seeking_past_the_end(self):
        _, off = UI.read_since(self.path, 0)
        open(self.path, "w").close()
        recs, new_off = UI.read_since(self.path, off)
        self.assertEqual(recs, [])
        self.assertEqual(new_off, 0)


class TestItServesNothingElse(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="traceui-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        open(os.path.join(self.dir, "trace-2026-08-29.jsonl"), "w").close()
        open(os.path.join(self.dir, "level"), "w").write("{}")

    def test_only_a_day_file_is_reachable(self):
        self.assertIsNotNone(UI._safe_file("trace-2026-08-29.jsonl", self.dir))
        for bad in ("level", "../../etc/passwd", "trace-2026-08-29.jsonl.bak",
                    "/etc/passwd", "", None, "trace-x.jsonl"):
            with self.subTest(name=bad):
                self.assertIsNone(UI._safe_file(bad, self.dir))

    def test_it_binds_to_localhost_only(self):
        src = (common.REPO / "tools" / "tracelog.py").read_text(encoding="utf-8")
        self.assertIn('("127.0.0.1", a.port)', src)
        self.assertNotIn('("0.0.0.0"', src)


class TestThePageStandsAlone(unittest.TestCase):
    def setUp(self):
        self.html = PAGE.read_text(encoding="utf-8")

    def test_it_loads_nothing_from_the_network(self):
        for pattern in (r'src\s*=\s*["\']https?:', r'href\s*=\s*["\']https?:',
                        r'@import\s+url\(https?:', r'fonts\.googleapis'):
            with self.subTest(pattern=pattern):
                self.assertNotRegex(self.html, pattern)

    def test_the_text_is_fetched_only_for_one_record(self):
        """The table calls /events, the click calls /record. If the table ever
        called /record — or /events with a text flag — the redaction would be
        decoration."""
        self.assertIn("/record?file=", self.html)
        self.assertIn("/events?since=", self.html)
        self.assertNotRegex(self.html, r"/events\?[^\"']*text=")

    def test_polling_stops_when_nobody_is_looking(self):
        self.assertIn("visibilityState", self.html)

    def test_the_live_view_can_be_stopped(self):
        self.assertIn('id=live', self.html)
        self.assertIn("addEventListener(\"scroll\"", self.html)

    def test_the_dom_is_capped(self):
        self.assertRegex(self.html, r"MAX_ROWS\s*=\s*\d+")


if __name__ == "__main__":
    unittest.main()


class TestTheChartsShowTheSameRowsAsTheTable(unittest.TestCase):
    """A picture that shows something other than the list beneath it is how a
    wrong conclusion gets drawn. All three read the filtered rows."""

    def setUp(self):
        self.html = PAGE.read_text(encoding="utf-8")

    def test_they_are_drawn_from_the_filtered_set(self):
        self.assertRegex(self.html, r"drawCharts\(shown\)")
        for fn in ("drawDurations", "drawTokens", "drawPrefixes"):
            self.assertIn("%s(rows)" % fn, self.html)

    def test_durations_use_a_log_scale(self):
        """One morning held 0.66 s and 836 s. On a linear axis everything
        below a minute is a line on the floor."""
        self.assertIn("Math.log10", self.html)

    def test_a_point_opens_the_same_detail_as_its_row(self):
        """Otherwise the chart shows an outlier and leaves you hunting for it
        in the table."""
        self.assertIn('closest("[data-at]")', self.html)
        self.assertIn('"data-at": r._at', self.html)

    def test_the_svg_is_built_by_hand(self):
        """No library, and none is coming: this repo has no npm and three
        pictures are not a reason to acquire one."""
        self.assertIn("createElementNS", self.html)
        self.assertNotRegex(self.html, r"(d3|chart\.js|plotly|echarts)")

    def test_the_charts_can_be_switched_off(self):
        self.assertIn("charts-btn", self.html)


class TestWhichProgramAsked(unittest.TestCase):
    """`who` is the ACCESS — it says martin-pc2 for Claude Code and for a
    harness alike. Telling them apart needs what the request itself carries:
    the dialect the path implies, the path, and what the client calls itself.
    """

    def setUp(self):
        self.html = PAGE.read_text(encoding="utf-8")
        self.gw = (common.REPO / "setup" / "claude" / "cc-gateway.py").read_text(
            encoding="utf-8")

    def test_the_gateway_records_all_three(self):
        # to the end of the call, not a guessed window: `ua` sits in the
        # detail group, which is further down than the summary one
        start = self.gw.index('TRACE.record(\n                "request"')
        block = self.gw[start:self.gw.index("# A restore that carried nothing", start)]
        for field in ('"dialect"', '"path"', '"ua"'):
            with self.subTest(field=field):
                self.assertIn(field, block)

    def test_the_user_agent_is_truncated(self):
        """A label, not a document."""
        self.assertRegex(self.gw, r'user-agent"\) or ""\)\[:\d+\]')

    def test_the_guess_keeps_the_raw_string_within_reach(self):
        """The column is a guess; a label that pretends to be certain is worse
        than one that does not. The exact user-agent stays on hover."""
        self.assertIn("function client(r)", self.html)
        self.assertIn('title="${esc(r.ua || "")}"', self.html)

    def test_it_falls_back_to_the_dialect_when_nothing_names_itself(self):
        self.assertIn('r.dialect === "openai"', self.html)
        self.assertIn('r.dialect === "anthropic"', self.html)
