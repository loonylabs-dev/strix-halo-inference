"""tracelog — a switch that must be off, a level that must not leak, a cap.

The three things this file protects, in order of how much they would cost if
they were wrong:

  * `text` writes COMPLETE PROMPTS to disk. docs/SECURITY.md calls the /slots
    prompt exposure the worst finding of this project, and this is the same
    material, persistent. It must be unreachable by accident, expire by
    itself, and never be what a fresh install does.
  * a trace that raises takes the request with it. Every failure has to be
    swallowed.
  * a trace with no cap fills the disk of a machine whose whole memory budget
    is measured in this repo.
"""
import json, os, stat, tempfile, shutil, unittest

import common

TR = common.load("setup/claude/tracelog.py", "tracelog")


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="trace-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.now = [1_000_000.0]
        self.t = TR.Trace(directory=self.dir, now=lambda: self.now[0])

    def lines(self):
        out = []
        for f in sorted(os.listdir(self.dir)):
            if f.startswith("trace-"):
                for line in open(os.path.join(self.dir, f), encoding="utf-8"):
                    out.append(json.loads(line))
        return [r for r in out if r.get("kind") != "header"]


class TestItIsOffUntilSwitchedOn(Base):
    def test_a_fresh_directory_traces_nothing(self):
        self.assertEqual(self.t.level, "off")
        self.assertIsNone(self.t.record("request", summary={"prefix": "a"}))
        self.assertEqual(self.lines(), [])

    def test_the_switch_takes_effect_without_a_restart(self):
        """The moment something looks odd is the moment you want it on, and
        restarting the gateway clears the prefix bookkeeping you turned it on
        to look at."""
        other = TR.Trace(directory=self.dir, now=lambda: self.now[0])
        other.set_level("summary")
        self.assertIsNotNone(self.t.record("request", summary={"prefix": "a"}),
                             "the running instance has to see the new level")

    def test_an_unknown_level_is_refused(self):
        with self.assertRaises(ValueError):
            self.t.set_level("verbose")

    def test_a_corrupt_control_file_reads_as_off(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "level"), "w") as f:
            f.write("{not json")
        self.assertEqual(self.t.refresh(), "off")


class TestWhatEachLevelMayWrite(Base):
    FIELDS = dict(summary={"prefix": "abc"}, detail={"tools": 24},
                  text={"system_head": "SECRET PROMPT"})

    def written(self, level):
        self.t.set_level(level, minutes=60 if level == "text" else None)
        self.t.record("request", **self.FIELDS)
        return self.lines()[-1]

    def test_summary_carries_no_text_and_no_detail(self):
        r = self.written("summary")
        self.assertEqual(r["prefix"], "abc")
        self.assertNotIn("tools", r)
        self.assertNotIn("system_head", r)

    def test_detail_stops_short_of_the_prompt(self):
        r = self.written("detail")
        self.assertEqual(r["tools"], 24)
        self.assertNotIn("system_head", r,
                         "a prompt must never appear below the text level")

    def test_text_carries_everything_and_says_so_in_the_file(self):
        r = self.written("text")
        self.assertEqual(r["system_head"], "SECRET PROMPT")
        header = None
        for f in os.listdir(self.dir):
            if f.startswith("trace-"):
                with open(os.path.join(self.dir, f), encoding="utf-8") as fh:
                    header = json.loads(fh.readline())
        self.assertEqual(header["kind"], "header")
        self.assertIn("COMPLETE PROMPTS", header["note"])


class TestTextGivesItselfBack(Base):
    """An operator who forgets is the normal case, not the exception, and the
    cost of forgetting is a disk full of other people's conversations."""

    def test_it_expires_into_detail(self):
        self.t.set_level("text", minutes=30)
        self.assertEqual(self.t.level, "text")
        self.now[0] += 31 * 60
        self.assertEqual(self.t.refresh(), "detail")

    def test_it_gets_an_expiry_even_when_none_is_asked_for(self):
        self.t.set_level("text")
        self.now[0] += 61 * 60
        self.assertEqual(self.t.refresh(), "detail")

    def test_the_other_levels_do_not_expire(self):
        self.t.set_level("detail")
        self.now[0] += 30 * 24 * 3600
        self.assertEqual(self.t.refresh(), "detail")


class TestTheFilesThemselves(Base):
    def test_one_file_per_day(self):
        self.t.set_level("summary")
        self.t.record("request", summary={"n": 1})
        self.now[0] += 24 * 3600
        self.t.record("request", summary={"n": 2})
        files = [f for f in os.listdir(self.dir) if f.startswith("trace-")]
        self.assertEqual(len(files), 2, files)

    def test_nobody_else_may_read_them(self):
        """They hold at least prefix ids and, on the text level, whole
        conversations."""
        self.t.set_level("summary")
        self.t.record("request", summary={"n": 1})
        for f in os.listdir(self.dir):
            path = os.path.join(self.dir, f)
            mode = stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(mode & 0o077, 0, "%s is %o" % (f, mode))
        self.assertEqual(stat.S_IMODE(os.stat(self.dir).st_mode) & 0o077, 0)

    def test_the_cap_prunes_the_oldest_day_first(self):
        small = TR.Trace(directory=self.dir, cap_bytes=2000,
                         now=lambda: self.now[0])
        small.set_level("summary")
        for day in range(4):
            for i in range(40):
                small.record("request", summary={"pad": "x" * 40, "i": i})
            self.now[0] += 24 * 3600
        files = sorted(f for f in os.listdir(self.dir) if f.startswith("trace-"))
        self.assertLess(len(files), 4, "nothing was pruned: %s" % files)
        self.assertIn("trace-", files[-1])


class TestItCannotBreakTheGateway(Base):
    def test_an_unwritable_directory_is_survived(self):
        """Every call site in cc-gateway is unguarded on purpose — the
        swallowing happens here, once."""
        self.t.set_level("summary")
        shutil.rmtree(self.dir)
        open(self.dir, "w").close()          # a FILE where the directory was
        self.assertIsNone(self.t.record("request", summary={"prefix": "a"}))

    def test_unserialisable_values_do_not_raise(self):
        self.t.set_level("summary")
        self.assertIsNotNone(self.t.record("request", summary={"x": object()}))


class TestTheGatewayActuallyRecords(unittest.TestCase):
    def test_the_call_sites_are_there(self):
        src = (common.REPO / "setup" / "claude" / "cc-gateway.py").read_text(
            encoding="utf-8")
        import re
        for kind in ("request", "restore", "save", "quarantine", "mismatch"):
            with self.subTest(kind=kind):
                self.assertRegex(
                    src, r'TRACE\.record\(\s*"%s"' % kind,
                    "the gateway records no %s event" % kind)

    def test_a_trace_left_on_is_visible_at_startup(self):
        """Otherwise it runs for weeks and nobody knows."""
        src = (common.REPO / "setup" / "claude" / "cc-gateway.py").read_text(
            encoding="utf-8")
        self.assertIn("TRACE IS ON at level", src)
        self.assertIn("prompts are being written in the clear", src)


if __name__ == "__main__":
    unittest.main()


class TestTheSuiteCannotWriteIntoTheRealTrace(unittest.TestCase):
    """A test run must not appear in the operator's data.

    It did, on 29.08.2026: the fixtures `abc123` and `id1` and the access
    `tester` were in ~/.cache/cc-gateway-trace, because loading cc-gateway
    constructs a Trace at import and the level happened to be on. An analysis
    of that morning showed twelve quarantines, all of them this suite.
    """

    def test_the_gateway_under_test_traces_somewhere_harmless(self):
        import importlib
        gw = common.load("setup/claude/cc-gateway.py", "cc_gateway_trace_check",
                         {"MAX_INFLIGHT": "2", "TOKEN_FILE": "/nonexistent-token",
                          "SLOT_PATH": "/nonexistent-slots",
                          "TRACE_DIR": "/nonexistent-trace"})
        self.assertEqual(gw.TRACE.dir, "/nonexistent-trace")
        self.assertEqual(gw.TRACE.level, "off",
                         "a directory that does not exist has no level to read")

    def test_the_env_var_is_what_redirects_it(self):
        src = (common.REPO / "setup" / "claude" / "tracelog.py").read_text(
            encoding="utf-8")
        self.assertIn('os.environ.get("TRACE_DIR")', src)
        loader = (common.REPO / "tests" / "test_gateway.py").read_text(
            encoding="utf-8")
        self.assertIn('"TRACE_DIR"', loader,
                      "the suite has to point the trace away from the real one")


class TestBothNamesAreRecorded(unittest.TestCase):
    """What came in and what was used are two different things, and the gap
    between them is where a whole morning went wrong: a harness sent
    `qwen38-think` — a name from the retired vocabulary — and was served as the
    bare alias, i.e. with no thinking at all, silently."""

    GW = None

    @classmethod
    def setUpClass(cls):
        cls.GW = common.load("setup/claude/cc-gateway.py", "cc_gateway_names",
                             {"MAX_INFLIGHT": "2", "TOKEN_FILE": "/nonexistent-token",
                              "SLOT_PATH": "/nonexistent-slots",
                              "TRACE_DIR": "/nonexistent-trace"})

    def test_a_mode_slug_reads_as_its_mode(self):
        self.assertEqual(self.GW._mode_of("qwen38-low", "qwen38"), "low")
        self.assertEqual(self.GW._mode_of("qwen38-xhigh", "qwen38"), "xhigh")

    def test_the_bare_alias_is_called_bare(self):
        self.assertEqual(self.GW._mode_of("qwen38", "qwen38"), "bare")

    def test_a_slug_for_another_model_is_not_read_as_a_mode(self):
        """`gemma26-low` while qwen38 serves is not a mode of qwen38, and
        pretending otherwise would put a name in the column that nothing
        honours."""
        self.assertEqual(self.GW._mode_of("gemma26-low", "qwen38"), "bare")

    def test_nothing_known_still_answers(self):
        self.assertEqual(self.GW._mode_of(None, "qwen38"), "bare")
        self.assertEqual(self.GW._mode_of("qwen38-low", None), "bare")

    def test_the_record_carries_slug_served_and_mode(self):
        src = (common.REPO / "setup" / "claude" / "cc-gateway.py").read_text(
            encoding="utf-8")
        block = src[src.index('TRACE.record(\n                "request"'):][:1200]
        for field in ('"slug"', '"served"', '"mode"'):
            self.assertIn(field, block)


class TestWhatCameBack(unittest.TestCase):
    """Read and written are two different numbers, and until 29.08. only the
    reading was recorded."""

    DIA = common.load("setup/claude/dialects.py", "dialects_output")

    def test_llama_cpps_own_count(self):
        self.assertEqual(self.DIA.output_from_text('{"timings":{"predicted_n":42}}'), 42)

    def test_the_anthropic_stream_reports_it_at_the_end(self):
        """message_start says 1 and means nothing yet; message_delta at the
        end is the figure that counts."""
        sse = ('data: {"type":"message_start","message":{"usage":{"output_tokens":1}}}\n\n'
               'data: {"type":"message_delta","usage":{"output_tokens":137}}\n\n')
        self.assertEqual(self.DIA.output_from_text(sse), 137)

    def test_the_openai_shape(self):
        self.assertEqual(self.DIA.output_from_text('{"usage":{"completion_tokens":9}}'), 9)

    def test_rubbish_is_unknown_rather_than_zero(self):
        for text in ("", "not json", '{"usage": {"output_tokens": "many"}}'):
            with self.subTest(text=text[:12]):
                self.assertIsNone(self.DIA.output_from_text(text))


class TestOneNameMeansOneThing(unittest.TestCase):
    def test_output_is_written_tokens_and_nothing_else(self):
        """A save event used to carry prewarm's stdout under `output`, and the
        table — which reads `output` as the tokens the model wrote — printed a
        paragraph of console text in a number column. Found on screen, 29.08."""
        src = (common.REPO / "setup" / "claude" / "cc-gateway.py").read_text(
            encoding="utf-8")
        self.assertIn('"prewarm_stdout"', src)
        self.assertNotIn('detail={"output"', src)


class TestTheRatesAreMeasuredOrAbsent(unittest.TestCase):
    """Reading and writing are two different machines — ~200 t/s against
    ~10-15 on this stack — and a request's total duration cannot be split into
    them afterwards. So the rates come from llama.cpp's own `timings` or they
    do not come at all."""

    DIA = common.load("setup/claude/dialects.py", "dialects_rates")

    def test_both_rates_come_from_the_timings_object(self):
        got = self.DIA.rates_from_text(
            '{"timings":{"prompt_per_second":203.4,"predicted_per_second":13.27}}')
        self.assertEqual(got, (203.4, 13.3))

    def test_the_anthropic_shape_has_none_and_says_so(self):
        """llama.cpp's to_json_anthropic() builds id, type, role, content,
        model, stop_reason, stop_sequence and usage — no timings. Claude Code
        rows therefore have no rates, and inventing one would be worse than an
        empty column."""
        self.assertIsNone(self.DIA.rates_from_text(
            '{"usage":{"cache_read_input_tokens":5650,"input_tokens":98,'
            '"output_tokens":40}}'))

    def test_zero_is_not_a_rate(self):
        self.assertIsNone(self.DIA.rates_from_text(
            '{"timings":{"prompt_per_second":0,"predicted_per_second":0}}'))

    def test_one_of_the_two_is_still_worth_having(self):
        got = self.DIA.rates_from_text('{"timings":{"predicted_per_second":12.5}}')
        self.assertEqual(got, (None, 12.5))

    def test_nothing_is_derived_from_the_duration(self):
        src = (common.REPO / "setup" / "claude" / "dialects.py").read_text(
            encoding="utf-8")
        body = src[src.index("def rates_from_text"):]
        for forbidden in ("took", "elapsed", "/ duration", "time.time"):
            self.assertNotIn(forbidden, body)


class TestALostStateLooksNothingLikeARewrite(unittest.TestCase):
    """Both show up as a drop in `reused`, and on 29.08.2026 that cost two
    rounds of blaming the watchdog for what the client had done to its own
    history. One hash per message tells them apart."""

    DIA = common.load("setup/claude/dialects.py", "dialects_shape")

    def msgs(self, *texts):
        return {"messages": [{"role": "user", "content": t} for t in texts]}

    def test_appending_keeps_every_earlier_message(self):
        a = self.DIA.message_shape(self.msgs("eins", "zwei"))
        b = self.DIA.message_shape(self.msgs("eins", "zwei", "drei"))
        self.assertEqual(self.DIA.shapes_agree(a, b), 2)

    def test_a_changed_message_cuts_the_agreement_there(self):
        a = self.DIA.message_shape(self.msgs("eins", "zwei", "drei"))
        b = self.DIA.message_shape(self.msgs("eins", "GEAENDERT", "drei"))
        self.assertEqual(self.DIA.shapes_agree(a, b), 1)

    def test_the_order_matters(self):
        a = self.DIA.message_shape(self.msgs("eins", "zwei"))
        b = self.DIA.message_shape(self.msgs("zwei", "eins"))
        self.assertEqual(self.DIA.shapes_agree(a, b), 0)

    def test_it_is_per_message_and_says_so(self):
        """A change INSIDE one long message marks the whole message. Enough
        for the distinction, and cheap enough for the request path."""
        src = (common.REPO / "setup" / "claude" / "dialects.py").read_text(
            encoding="utf-8")
        self.assertIn("Per MESSAGE, not per token", src)

    def test_an_unserialisable_message_does_not_raise(self):
        shape = self.DIA.message_shape({"messages": [{"role": "user", "content": object()}]})
        self.assertEqual(len(shape), 1)

    def test_the_gateway_bounds_what_it_remembers(self):
        src = (common.REPO / "setup" / "claude" / "cc-gateway.py").read_text(
            encoding="utf-8")
        self.assertIn("LAST_SHAPE_MAX", src)
        self.assertIn("LAST_SHAPE.pop(next(iter(LAST_SHAPE)))", src)


class TestTheShapeSurvivesTheProcessThatComputedIt(unittest.TestCase):
    """`msgs_kept` is the comparison reduced to a count, and the count is made
    against LAST_SHAPE — memory a gateway restart wipes. On 29.08.2026 that
    was the whole gap: the incident at 21:23 had a shape, nobody had written
    it down, and by the time the question was asked the count for that pair
    could never be recomputed.

    So the list goes into the record at `detail`, and the text it would take
    to read the change stays behind `text`.
    """

    def setUp(self):
        self.src = (common.REPO / "setup" / "claude" / "cc-gateway.py").read_text(
            encoding="utf-8")
        start = self.src.index('TRACE.record(\n                "request"')
        self.block = self.src[start:self.src.index(
            "# A restore that carried nothing", start)]

    def _group(self, name):
        """The record call's summary=/detail=/text= group, as source."""
        at = self.block.index("%s={" % name)
        nxt = [self.block.index("%s={" % o) for o in ("summary", "detail", "text")
               if o != name and "%s={" % o in self.block
               and self.block.index("%s={" % o) > at]
        return self.block[at:min(nxt) if nxt else len(self.block)]

    def test_the_shape_and_the_sizes_are_written_at_detail(self):
        g = self._group("detail")
        self.assertIn('"shape": shape', g)
        self.assertIn('"msg_chars"', g)

    def test_the_whole_body_needs_the_text_level(self):
        """A record that carries every message is a conversation on disk. It
        belongs in the group that expires by itself, not the one that runs
        all day."""
        self.assertIn('"body_full": p', self._group("text"))
        self.assertNotIn("body_full", self._group("detail"))
        self.assertNotIn("body_full", self._group("summary"))

    def test_the_hashes_and_the_sizes_come_from_one_pass(self):
        """Two passes over the same messages would let the log carry a size
        that belongs to a different rendering than the hash beside it."""
        self.assertIn("fingers = DIA.message_fingerprints", self.src)
        self.assertIn("shape = [h for h, _ in fingers]", self.src)


class TestTheDiffNamesTheMessageThatChanged(unittest.TestCase):
    """The question of 29.08.2026, made answerable: 18,450 tokens were
    re-prefilled inside ONE session — was the state taken away, or did the
    client rewrite its own history, and if so, where?

    Two consecutive records now hold enough to say so without the texts, and
    with the texts they say what the change was.
    """

    CLI = common.load("tools/tracelog.py", "tracelog_cli_diff")

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="trace-diff-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.old = os.environ.get("TRACE_DIR")
        os.environ["TRACE_DIR"] = self.dir
        self.addCleanup(lambda: os.environ.__setitem__("TRACE_DIR", self.old)
                        if self.old else os.environ.pop("TRACE_DIR", None))

    def write(self, *recs):
        with open(os.path.join(self.dir, "trace-2026-08-29.jsonl"), "w",
                  encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(dict({"kind": "request", "prefix": "p"}, **r))
                        + "\n")

    def run_diff(self):
        import argparse, contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.CLI.cmd_diff(argparse.Namespace(day=None, n=40))
        return buf.getvalue()

    def test_a_clean_cheap_append_is_not_reported(self):
        self.write({"t": 1, "shape": ["a", "b"], "reused": 100, "computed": 5},
                   {"t": 2, "shape": ["a", "b", "c"], "reused": 200, "computed": 5})
        self.assertIn("nothing", self.run_diff())

    def test_a_rewrite_is_located_by_message_index(self):
        self.write({"t": 1, "shape": ["a", "b", "c"], "msg_chars": [10, 20, 30],
                    "reused": 100, "computed": 5},
                   {"t": 2, "shape": ["a", "X", "c"], "msg_chars": [10, 20, 30],
                    "reused": 10, "computed": 900})
        out = self.run_diff()
        self.assertIn("kept 1", out)
        self.assertIn("message 1", out)

    def test_it_says_which_kind_of_rewrite(self):
        for old_n, new_n, word in ((20, 20, "re-rendered"), (20, 5, "truncated"),
                                   (20, 99, "extended")):
            with self.subTest(word=word):
                self.write({"t": 1, "shape": ["a", "b"], "msg_chars": [10, old_n],
                            "reused": 100, "computed": 5},
                           {"t": 2, "shape": ["a", "X"], "msg_chars": [10, new_n],
                            "reused": 10, "computed": 900})
                self.assertIn(word, self.run_diff())

    def test_an_agreeing_shape_with_a_collapsed_reuse_blames_the_server(self):
        """The other half of the distinction, and the half that was blamed
        wrongly twice: the history is identical, so the state was lost."""
        self.write({"t": 1, "shape": ["a", "b"], "reused": 100, "computed": 5},
                   {"t": 2, "shape": ["a", "b", "c"], "reused": 10, "computed": 900})
        out = self.run_diff()
        self.assertIn("pure append", out)
        self.assertIn("SERVER's", out)

    def test_without_the_text_it_says_how_to_get_it(self):
        self.write({"t": 1, "shape": ["a", "b"], "msg_chars": [10, 20],
                    "reused": 100, "computed": 5},
                   {"t": 2, "shape": ["a", "X"], "msg_chars": [10, 20],
                    "reused": 10, "computed": 900})
        self.assertIn("on text", self.run_diff())

    def test_with_the_text_it_prints_both_versions(self):
        self.write({"t": 1, "shape": ["a", "b"], "msg_chars": [10, 20],
                    "reused": 100, "computed": 5,
                    "body_full": {"messages": [{"c": "keep"}, {"c": "BEFORE"}]}},
                   {"t": 2, "shape": ["a", "X"], "msg_chars": [10, 20],
                    "reused": 10, "computed": 900,
                    "body_full": {"messages": [{"c": "keep"}, {"c": "AFTER"}]}})
        out = self.run_diff()
        self.assertIn("BEFORE", out)
        self.assertIn("AFTER", out)
        self.assertNotIn("keep", out, "only the message that changed")

    def test_nothing_to_compare_is_not_nothing_found(self):
        """Records written before 29.08.2026 carry no shape. Printing "every
        append was clean" over them turns a gap into a clean bill of health —
        which is exactly the mistake this whole command exists to stop."""
        self.write({"t": 1, "reused": 10, "computed": 900},
                   {"t": 2, "reused": 10, "computed": 900})
        out = self.run_diff()
        self.assertIn("nothing to compare", out)
        self.assertIn("2 requests", out)
        self.assertNotIn("clean", out)

    def test_a_clean_result_says_how_much_it_looked_at(self):
        self.write({"t": 1, "shape": ["a"], "reused": 100, "computed": 5},
                   {"t": 2, "shape": ["a", "b"], "reused": 200, "computed": 5})
        self.assertIn("1 pairs", self.run_diff())


class TestAChangedHeadIsItsOwnFailure(unittest.TestCase):
    """Grouping by prefix — which the shape comparison must do — is blind to
    the most expensive failure there is, because the two requests land in
    different groups and are never compared.

    Measured 30.08.2026, 00:01: the tool list went 13 -> 21 mid-conversation,
    the prefix id changed with it, and 55,856 tokens of an UNTOUCHED
    conversation were recomputed. 655 seconds. The tools sit in front of the
    messages, so nothing behind them survives.
    """

    CLI = common.load("tools/tracelog.py", "tracelog_cli_head")

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="trace-head-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.old = os.environ.get("TRACE_DIR")
        os.environ["TRACE_DIR"] = self.dir
        self.addCleanup(lambda: os.environ.__setitem__("TRACE_DIR", self.old)
                        if self.old else os.environ.pop("TRACE_DIR", None))

    def run_on(self, *recs):
        import contextlib, io
        rows = [dict({"kind": "request", "who": "someone"}, **r) for r in recs]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.CLI._head_changes(rows)
        return buf.getvalue()

    def test_a_new_head_is_reported_with_what_grew(self):
        out = self.run_on(
            {"t": 1, "prefix": "aaa", "tools": 13, "prefix_chars": 60428},
            {"t": 2, "prefix": "bbb", "tools": 21, "prefix_chars": 73404,
             "reused": 17784, "computed": 55856, "took_s": 655.8})
        self.assertIn("NEVER SEEN", out)
        self.assertIn("13 -> 21", out)
        self.assertIn("60428 -> 73404", out)
        self.assertIn("55856", out)

    def test_a_return_to_a_known_prefix_is_counted_not_reported(self):
        """Claude Code runs two prompt types side by side and they flip the
        prefix back and forth all day. Reporting each flip buries the one
        change that cost eleven minutes."""
        out = self.run_on({"t": 1, "prefix": "aaa"}, {"t": 2, "prefix": "bbb"},
                          {"t": 3, "prefix": "aaa"}, {"t": 4, "prefix": "bbb"})
        self.assertIn("2 returns", out)
        self.assertEqual(out.count("NEVER SEEN"), 1, "only bbb was ever new")

    def test_it_says_whether_the_history_was_to_blame(self):
        same = ["h%d" % i for i in range(20)]
        out = self.run_on(
            {"t": 1, "prefix": "aaa", "shape": same},
            {"t": 2, "prefix": "bbb", "shape": same + ["new"], "computed": 9})
        self.assertIn("the history was not the cause", out)

    def test_the_expensive_one_comes_first(self):
        """A day holds a dozen head changes and one of them cost eleven
        minutes. Chronological order buries it."""
        out = self.run_on(
            {"t": 1, "prefix": "p0"},
            {"t": 2, "prefix": "p1", "computed": 10},
            {"t": 3, "prefix": "p2", "computed": 55856})
        first = [l for l in out.splitlines() if "NEVER SEEN" in l][0]
        self.assertIn("-> p2", first)

    def test_a_caller_that_never_changed_head_says_so(self):
        out = self.run_on({"t": 1, "prefix": "aaa"}, {"t": 2, "prefix": "aaa"})
        self.assertIn("never seen before", out)

    def test_two_callers_do_not_share_a_history(self):
        """`who` is the access. One caller's first sight of a prefix is not
        made stale by another caller having used it."""
        rows = [{"kind": "request", "who": "a", "t": 1, "prefix": "p1"},
                {"kind": "request", "who": "a", "t": 2, "prefix": "p2"},
                {"kind": "request", "who": "b", "t": 3, "prefix": "p2"},
                {"kind": "request", "who": "b", "t": 4, "prefix": "p1"}]
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.CLI._head_changes(rows)
        self.assertEqual(buf.getvalue().count("NEVER SEEN"), 2)
