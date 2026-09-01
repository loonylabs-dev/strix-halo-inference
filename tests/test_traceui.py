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
import json, os, re, shutil, subprocess, tempfile, unittest

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
        self.gw = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
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


class TestAColumnMeansOneThing(unittest.TestCase):
    """Seen on screen, 29.08.: a `save` row printed a paragraph of prewarm's
    console output in the token column, because that record carried it under
    `output` — the name the request rows use for written tokens. Renaming the
    field fixed new records; the TABLE had to stop asking non-requests for
    numbers, or every record written before the rename would keep doing it."""

    def setUp(self):
        self.html = PAGE.read_text(encoding="utf-8")

    def test_written_tokens_are_asked_only_of_requests(self):
        """The original defect: a `save` record carried prewarm's console
        output under `output`, the name a request uses for written tokens, and
        a paragraph of text landed in a number column."""
        self.assertIn('const req = r.kind === "request";', self.html)
        self.assertIn('${req ? (r.output ?? "") : ""}', self.html)

    def test_input_and_cache_are_asked_of_both(self):
        """A save IS a prefill, so it answers the same two questions: how many
        tokens went in, and how much of them the slot already had. Leaving
        those blank while the row beside it shows 99 % was the confusion of
        30.08."""
        self.assertIn('${inAny(r) ?? ""}', self.html)
        self.assertIn('${share(r)}', self.html)

    def test_everything_else_stays_in_the_detail_view(self):
        """Where it is labelled, instead of guessed at by column position."""
        self.assertIn("loadDetail", self.html)


class TestTheHeaderOffsetIsMeasured(unittest.TestCase):
    """The column titles stick under the header. Hard-wiring that offset works
    until the header wraps to two lines on a narrow window — and then the
    titles sit on top of the filters. Seen in a screenshot, 29.08."""

    def test_the_offset_comes_from_the_header_itself(self):
        html = PAGE.read_text(encoding="utf-8")
        self.assertIn("--hh", html)
        self.assertIn("offsetHeight", html)
        self.assertIn("ResizeObserver(measureHeader)", html,
                      "a resize listener misses a wrap caused by a zoom or a "
                      "longer filename — measured 51 px against a 93 px header")


class TestAWholeRequestIsNotATableCell(unittest.TestCase):
    """`body_full` is the entire conversation. The redaction was written for
    strings, and a dict slips straight through it — which would put every
    prompt of every row on the wire by default, the exact thing this file
    exists to prevent."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="traceui-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = os.path.join(self.dir, "trace-2026-08-29.jsonl")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "t": 1000, "kind": "request", "prefix": "p",
                "body_full": {"system": "SECRET SYSTEM",
                              "messages": [{"role": "user",
                                            "content": "SECRET MESSAGE"}]},
                "shape": ["aaaaaaaa"], "msg_chars": [42]}) + "\n")

    def test_the_body_never_reaches_the_table(self):
        recs, _ = UI.read_since(self.path, 0)
        self.assertEqual(len(recs), 1)
        self.assertNotIn("SECRET", json.dumps(recs[0]))
        self.assertIn("click", recs[0]["body_full"])

    def test_the_click_still_gets_it(self):
        recs, _ = UI.read_since(self.path, 0)
        full = UI.read_one(self.path, recs[0]["_at"])
        self.assertEqual(full["body_full"]["system"], "SECRET SYSTEM")

    def test_the_shape_is_not_redacted(self):
        """It is the diagnosis, it is small, and a row that hides it makes the
        table useless for the thing it was extended for."""
        recs, _ = UI.read_since(self.path, 0)
        self.assertEqual(recs[0]["shape"], ["aaaaaaaa"])
        self.assertEqual(recs[0]["msg_chars"], [42])


class TestADerivedRateIsMarkedAsOne(unittest.TestCase):
    """llama.cpp reports rates on the OpenAI route and nothing on the
    Anthropic one, so the column stood empty for Claude Code and a "12 tokens
    per second" had to be read out of the server journal by hand — which is
    exactly what happened on 30.08. and prompted the question "where does that
    number come from?".

    The gateway now times the first delta, which separates the two phases
    itself. That value is not llama.cpp's accounting and must never look like
    it."""

    def setUp(self):
        self.html = PAGE.read_text(encoding="utf-8")

    def test_the_measured_pair_still_wins(self):
        self.assertIn("if (r.read_tps != null || r.write_tps != null)", self.html)

    def test_the_derived_one_carries_a_tilde(self):
        self.assertIn('return `–·~${f(r.write_tps_derived)}`;', self.html)

    def test_nothing_at_all_stays_empty(self):
        """An empty column is the honest answer when neither exists. A save row
        falls through to saveRate, which returns "" unless it has both a token
        count and a duration — so the guarantee is there, one function later."""
        self.assertIn("return saveRate(r);", self.html)
        self.assertRegex(self.html,
                         r"function saveRate\(r\) \{[\s\S]{0,200}return \"\";")

    def test_the_column_title_says_which_is_which(self):
        self.assertIn("with ~ derived by the gateway", self.html)

    def test_a_save_row_says_where_its_seconds_went(self):
        """51.8 in the seconds column and nothing else reads as a slow disk.
        Asked 30.08.2026 in exactly those words; the write was 237 ms of it and
        the rest is a prefill."""
        self.assertIn("function saveHint(r)", self.html)
        self.assertIn("the rest is prefill", self.html)

    def test_a_save_shows_the_tokens_it_computed(self):
        self.assertIn('if (typeof r.computed === "number") return r.computed;',
                      self.html)


class TestTheseFunctionsActuallyRun(unittest.TestCase):
    """Every other test in this file greps the HTML for a string. That proves
    the source contains something, not that it computes anything — and on
    30.08.2026 a table cell was wrong for an hour while a test asserting its
    presence stayed green.

    So the script is extracted and its pure functions are CALLED. Node is used
    because the code is JavaScript; nothing is installed and no package manager
    is involved, which is the rule this repo keeps. Where node is absent the
    test skips and says why — a skip that names its reason is honest, a grep
    standing in for an execution is not.
    """

    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")

    def run_js(self, calls):
        if not self.node:
            self.skipTest("node is not on PATH — these functions are not "
                          "exercised here, only their presence elsewhere")
        html = PAGE.read_text(encoding="utf-8")
        js = html[html.index("<script>") + 8:html.rindex("</script>")]
        stub = """
const stub = new Proxy({}, {get: (t,k) => (k === 'style' || k === 'classList')
  ? new Proxy({}, {get: () => () => {}, set: () => true})
  : (k === 'value' || k === 'innerHTML' || k === 'textContent') ? "" : () => stub,
  set: () => true});
const document = {querySelector: () => stub, addEventListener: () => {},
                  documentElement: {style: {setProperty: () => {}}},
                  createElementNS: () => stub, title: ""};
const window = {scrollY: 0}; const addEventListener = () => {};
const ResizeObserver = class { observe() {} };
const fetch = () => Promise.resolve({json: () => ({days: [], records: [], offset: 0})});
const setInterval = () => {};
"""
        src = stub + js.replace("days(); poll(true); setInterval(poll, 2000);", "") + calls
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            r = subprocess.run([self.node, path], capture_output=True, text=True,
                               timeout=60)
            self.assertEqual(r.returncode, 0,
                             "the page's script did not run: %s" % r.stderr[-600:])
            return r.stdout.strip().splitlines()
        finally:
            os.unlink(path)

    def test_a_save_reports_the_tokens_it_computed(self):
        out = self.run_js("console.log(tokensOf({kind:'save', computed:11005}));")
        self.assertEqual(out[-1], "11005")

    def test_a_restore_still_reports_its_own_field(self):
        out = self.run_js("console.log(tokensOf({kind:'restore', tokens:8077}));")
        self.assertEqual(out[-1], "8077")

    def test_the_save_rate_is_the_prefill_rate(self):
        """11,005 tokens in 51.8 s minus a 237 ms write is ~213 a second, which
        is what prewarm's own line says it did."""
        out = self.run_js("console.log(saveRate({kind:'save', write_ms:237,"
                          " computed:11005, saved_s:51.8}));")
        self.assertEqual(out[-1], "~213·–")

    def test_a_request_without_rates_stays_empty(self):
        out = self.run_js("console.log(JSON.stringify(rates({kind:'request'})));")
        self.assertEqual(out[-1], '""')

    def test_the_hint_explains_the_seconds(self):
        out = self.run_js("console.log(saveHint({kind:'save', write_ms:237}));")
        self.assertIn("prefill", out[-1])
        self.assertIn("237", out[-1])

    def test_an_old_record_still_shows_its_numbers(self):
        """The gateway only started parsing prewarm's line at 30.08. 19:03.
        Every save before that carries the numbers ONLY inside the console
        output, and a reader that gives up on those shows an empty column for
        the whole history — which is what was asked about: "I still do not see
        11005"."""
        old = json.dumps({"kind": "save", "saved_s": 51.8, "prewarm_stdout":
                          "precomputing …\\n  done in 51.4 s\\nsaved: x.bin — "
                          "11005 tokens, 878 MB, 237 ms"})
        out = self.run_js("const o = %s;\n"
                          "console.log(tokensOf(o));\n"
                          "console.log(saveRate(o));\n"
                          "console.log(saveHint(o).includes('237'));" % old)
        self.assertEqual(out[-3], "11005")
        self.assertEqual(out[-2], "~213·–")
        self.assertEqual(out[-1], "true")

    def test_a_save_without_any_numbers_stays_empty(self):
        out = self.run_js("console.log(JSON.stringify(tokensOf("
                          "{kind:'save', saved_s:3})));\n"
                          "console.log(JSON.stringify(saveRate("
                          "{kind:'save', saved_s:3})));")
        self.assertEqual(out[-2], '""')
        self.assertEqual(out[-1], '""')

    def test_a_save_fills_both_columns_when_it_can(self):
        out = self.run_js(
            "const n={kind:'save',saved_s:51.8,reused:3000,computed:8005,"
            "tokens:11005,write_ms:237};\n"
            "console.log(inAny(n)); console.log(share(n));")
        self.assertEqual(out[-2], "11005")
        self.assertEqual(out[-1], "27%")

    def test_an_old_save_shows_the_total_and_no_share(self):
        """Records before 30.08. 19:03 carry no split. The honest half-answer
        is the total with an empty share, not a guessed 0 %."""
        out = self.run_js(
            "const o={kind:'save',saved_s:51.8,"
            "prewarm_stdout:'saved: x.bin — 11005 tokens, 878 MB, 237 ms'};\n"
            "console.log(inAny(o)); console.log(JSON.stringify(share(o)));")
        self.assertEqual(out[-2], "11005")
        self.assertEqual(out[-1], '""')

    def test_a_non_json_error_answer_still_reaches_the_operator(self):
        """A stale viewer process without the POST endpoint answers 501 with
        an HTML body (measured 2026-09-01). The page parsed that body as JSON
        before looking at the status: the parse threw past the alert, and the
        select snapped back on the next poll without a word to the operator."""
        out = self.run_js(
            'globalThis.alert = m => console.log("ALERT " + m);\n'
            'levelResponse({ok: false, status: 501,\n'
            '               json: () => Promise.reject(new SyntaxError())})\n'
            '  .then(d => console.log("RESULT " + d));')
        self.assertIn("ALERT HTTP 501", out)
        self.assertIn("RESULT null", out)

    def test_the_servers_own_error_text_still_wins(self):
        """The 403/415 answers carry a reason in their JSON body — the
        fallback must not replace it."""
        out = self.run_js(
            'globalThis.alert = m => console.log("ALERT " + m);\n'
            'levelResponse({ok: false, status: 403,\n'
            '               json: () => Promise.resolve({error: "not you"})})\n'
            '  .then(d => console.log("RESULT " + d));')
        self.assertIn("ALERT not you", out)
        self.assertIn("RESULT null", out)


class TestItCanBeRestarted(unittest.TestCase):
    """A viewer that cannot be restarted for a minute is a viewer nobody
    restarts.

    The server holds the port for as long as its last connection sits in
    TIME-WAIT, so `kill` followed straight away by a fresh `serve` dies with
    EADDRINUSE — which reads like "it is still running" and invites a second
    kill on the wrong PID. SO_REUSEADDR is what prevents that, and it is read
    inside server_bind(), which the constructor calls: setting the attribute
    on the finished instance is too late to have any effect at all.
    """

    def test_the_listening_socket_really_carries_so_reuseaddr(self):
        import http.server
        import socket

        srv = UI._Server(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
        self.addCleanup(srv.server_close)
        self.assertTrue(
            srv.socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR),
            "the bound socket has no SO_REUSEADDR — a restart within the "
            "TIME-WAIT window will fail with EADDRINUSE")


class TestThePageNamesWhatItShows(unittest.TestCase):
    """The gateway was renamed in 09/2026; the page kept the old name in its
    title bar for a day, because a rename greps for paths and unit names and
    nobody greps for headings. An operator reading `cc-gateway trace` above a
    table has every reason to believe they are looking at a stale viewer —
    which is exactly the thing this page is used to rule out.
    """

    def test_no_heading_carries_the_pre_rename_name(self):
        html = PAGE.read_text(encoding="utf-8")
        for tag in ("title", "h1"):
            m = re.search(r"<%s>(.*?)</%s>" % (tag, tag), html, re.S)
            self.assertIsNotNone(m, "no <%s> on the page" % tag)
            self.assertIn("llm-gateway", m.group(1))
            self.assertNotIn("cc-gateway", m.group(1))


class TestTheLevelSwitch(unittest.TestCase):
    """The one endpoint that WRITES — and the page is served to a browser.

    Every other tab that browser has open can send requests to 127.0.0.1, so
    the question is not whether the operator can flip the switch but whether a
    page from anywhere else can flip it for them. `text` writes complete
    prompts to disk in the clear; a switch reachable by a stray <img> would
    hand that to any site the operator happens to be reading.

    Three guards, and each is tested by performing the attack rather than by
    reading the source: the method (a GET switches nothing), the content type
    (application/json forces a CORS preflight this server never answers), and
    the Origin/Host pair (a rebound DNS name is not this viewer).
    """

    def setUp(self):
        import http.server                                  # noqa: F401
        import threading
        import urllib.request

        self.request = urllib.request
        self.dir = tempfile.mkdtemp(prefix="traceui-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.trace = UI.TR.Trace(directory=self.dir)
        self.trace.set_level("detail")

        self.srv = UI._Server(("127.0.0.1", 0), None)
        self.port = self.srv.server_address[1]
        self.srv.RequestHandlerClass = UI.make_handler(
            self.trace, str(PAGE), self.port)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.base = "http://127.0.0.1:%d" % self.port

    def post(self, body, ctype="application/json", origin=None, host=None):
        req = self.request.Request(
            self.base + "/level", data=json.dumps(body).encode(),
            method="POST", headers={"Content-Type": ctype})
        if origin:
            req.add_header("Origin", origin)
        if host:
            req.add_header("Host", host)
        try:
            with self.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read().decode())
        except Exception as e:
            return getattr(e, "code", 0), {}

    def level_on_disk(self):
        with open(os.path.join(self.dir, "level"), encoding="utf-8") as f:
            return json.load(f)

    def test_the_operators_own_page_can_switch(self):
        code, body = self.post({"level": "summary"},
                               origin=self.base)
        self.assertEqual(code, 200)
        self.assertEqual(body.get("level"), "summary")
        self.assertEqual(self.level_on_disk()["level"], "summary")

    def test_text_always_gets_an_expiry_even_when_none_is_asked_for(self):
        """The CLI forces one; a switch that did not would be the easier way
        to leave prompts running for a week."""
        code, body = self.post({"level": "text"}, origin=self.base)
        self.assertEqual(code, 200)
        self.assertGreater(self.level_on_disk()["expires"], 0)
        self.assertGreater(body.get("expires", 0), 0)

    def test_a_page_from_somewhere_else_cannot_switch(self):
        code, _ = self.post({"level": "text"}, origin="https://evil.example")
        self.assertEqual(code, 403)
        self.assertEqual(self.level_on_disk()["level"], "detail")

    def test_a_rebound_name_is_not_this_viewer(self):
        code, _ = self.post({"level": "text"}, host="evil.example:%d" % self.port)
        self.assertEqual(code, 403)
        self.assertEqual(self.level_on_disk()["level"], "detail")

    def test_a_form_post_is_refused_so_the_preflight_cannot_be_skipped(self):
        """A cross-origin POST only escapes the preflight while its content
        type is a form/plain one. Refusing those is what makes the browser
        ask first."""
        for ctype in ("text/plain", "application/x-www-form-urlencoded",
                      "multipart/form-data"):
            with self.subTest(ctype=ctype):
                code, _ = self.post({"level": "text"}, ctype=ctype)
                self.assertEqual(code, 415)
                self.assertEqual(self.level_on_disk()["level"], "detail")

    def test_a_get_switches_nothing(self):
        """An <img src> on a foreign page is a GET, and it carries no Origin
        the server could refuse."""
        try:
            with self.request.urlopen(
                    self.base + "/level?set=text", timeout=5) as r:
                code = r.status
        except Exception as e:
            code = getattr(e, "code", 0)
        self.assertEqual(code, 404)
        self.assertEqual(self.level_on_disk()["level"], "detail")

    def test_an_unknown_level_is_refused(self):
        code, _ = self.post({"level": "everything"}, origin=self.base)
        self.assertEqual(code, 400)
        self.assertEqual(self.level_on_disk()["level"], "detail")

    def test_only_text_gets_an_expiry_written(self):
        """An expiry means one thing: when `text` gives itself back. refresh()
        applies it to nothing else, so writing one beside `summary` — which
        the page did on its first outing, because it always sent the minutes
        field — records a deadline that will never be acted on."""
        code, body = self.post({"level": "summary", "minutes": 60},
                               origin=self.base)
        self.assertEqual(code, 200)
        self.assertEqual(self.level_on_disk()["expires"], 0)
        self.assertEqual(body.get("expires"), 0)


class TestThePageIsEnglish(unittest.TestCase):
    """This repo is English, and the viewer was the last German thing in it.

    It was written German-first and translated in pieces — the bar in one
    sitting, the table and the charts three exchanges later, each time because
    somebody read a word on screen and asked. A page half in each language is
    worse than either: `Historie` beside `served/mode` reads as a column that
    means something different.

    The check is a HEURISTIC, not a language detector: umlauts, plus function
    words that cannot appear in English prose. It will not catch a German
    sentence written without either, and it is not meant to — it catches the
    way this actually regresses, which is a hurried string added in the
    language the conversation happens to be in.
    """

    # Matched on WORD BOUNDARIES, which the first version was not: `und`
    # found itself inside "around" and "background" and reported two English
    # comments as German. A guard that cries wolf gets deleted by the next
    # person in a hurry, which costs more than the German it was watching for.
    WORDS = ("nicht", "und", "wurde", "kamen", "gerechnet", "geladen",
             "pausiert", "Anfrage[n]?", "Historie", "Anteil", "kalt", "davon",
             "Zustand", "Dauer", "vom", "Nachrichten", "Schreiben")
    UMLAUTS = "äöüßÄÖÜ"

    def test_nothing_on_the_page_is_german(self):
        pattern = re.compile(r"\b(%s)\b" % "|".join(self.WORDS))
        found = []
        for i, line in enumerate(PAGE.read_text(encoding="utf-8").splitlines(), 1):
            hit = pattern.search(line)
            if not hit:
                hit = next((c for c in self.UMLAUTS if c in line), None)
            if hit:
                found.append("%d: %s (%s)" % (
                    i, line.strip()[:60],
                    hit if isinstance(hit, str) else hit.group(0)))
        self.assertEqual(found, [], "German left on the page:\n  " +
                         "\n  ".join(found))
