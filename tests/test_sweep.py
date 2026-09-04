"""Tests for the sweep chain: env parsing, server start, variants, the cost
measurement, comparison.

The two bugs this file exists for were both of the silent kind:
- bench/run.py referenced an env_ that was never defined — every --env run
  died before the server came up, and only the --running path had ever run.
- shlex.split stripped the double quotes out of
  --chat-template-kwargs {"reasoning_effort":"medium"}, so the identical
  line meant something different under systemd than under bench/run.py.
"""
import http.server, json, os, re, tempfile, threading, unittest

import common

RUN = common.load("bench/run.py", "bench_run")
SPD = common.load("bench/speed.py", "bench_speed")
CMP = common.load("bench/compare.py", "bench_compare")
SWEEP = common.load("bench/sweep.py", "bench_sweep")

KWARG = '{"reasoning_effort":"medium"}'


class TestArgsFromEnv(unittest.TestCase):
    def _parse(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            return RUN.args_from_env(path)
        finally:
            os.unlink(path)

    def test_continuation_lines_are_joined(self):
        argv = self._parse("# comment\nLLAMA_ARGS=-a 1 \\\n  -b 2\n")
        self.assertEqual(argv, ["-a", "1", "-b", "2"])

    def test_json_kwargs_survive_like_under_systemd(self):
        """systemd splits $LLAMA_ARGS on whitespace and passes quotes through
        as data. The parser here has to agree, or the same env file means two
        different servers."""
        argv = self._parse("LLAMA_ARGS=--chat-template-kwargs %s --jinja\n"
                           % KWARG)
        self.assertIn(KWARG, argv)
        json.loads(argv[argv.index("--chat-template-kwargs") + 1])

    def test_the_shipped_qwen_profile_carries_valid_json(self):
        """The production default is thinking OFF at the server; the modes
        come per request via the gateway (KWARGS_BY_MODEL). What matters
        here is only that the JSON survives the split intact."""
        argv = RUN.args_from_env(str(common.REPO / "setup/env/qwen38.env"))
        i = argv.index("--chat-template-kwargs")
        self.assertEqual(json.loads(argv[i + 1]), {"enable_thinking": False})


class TestStartServer(unittest.TestCase):
    def test_env_reaches_the_child_and_the_log_gates_the_return(self):
        """A fake llama-server proves three things at once: start_server no
        longer dies on the undefined env_, SLOTS_DEBUG is passed through, and
        the call only returns once 'model loaded' shows in the log."""
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, "fake-server")
            with open(fake, "w") as f:
                f.write("#!/usr/bin/env bash\n"
                        "echo \"DEBUG=${LLAMA_SERVER_SLOTS_DEBUG:-unset}\"\n"
                        "echo 'model loaded'\nsleep 60\n")
            os.chmod(fake, 0o755)
            log = os.path.join(d, "server.log")
            os.environ["SLOTS_DEBUG"] = "1"
            try:
                proc = RUN.start_server([], log, fake)
            finally:
                os.environ.pop("SLOTS_DEBUG", None)
            try:
                with open(log) as f:
                    content = f.read()
                self.assertIn("DEBUG=1", content)
                self.assertIn("model loaded", content)
            finally:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                # Reaped, not just signalled. Without the wait the Popen is
                # collected while the child is still going and Python says so
                # on stderr — one more line of noise in a CI log that a
                # stranger reads only when something is already wrong.
                proc.wait(timeout=10)



class TestTheWatchdogGoesDownWithProduction(unittest.TestCase):
    """A detector that cries wolf on every measurement gets ignored.

    llama-probe.timer asks the production server one question every ten
    minutes and reports a failure when nothing answers. A measurement that
    stops production therefore has to take it down too — bench/sideserver.py
    has done that since 27.08.2026, with a long comment saying why.

    THE FIX COVERED ONE STOP PATH OF THREE. bench/sweep.py and
    bench/suites/restore-safety.py stop production themselves, and neither
    touched the timer. It bit the same evening: the first restore-safety run
    after the timer was armed left

        ! the last probe FAILED (status 1)

    in check.sh, from a probe that fired at 19:17:54 against a server the
    harness had deliberately stopped twelve minutes earlier. Exactly the false
    alarm sideserver's comment describes, arriving through the path the fix
    never reached.

    So the stopping lives in one place now, and this is what keeps it there.
    """

    ALLOWED = "bench/sweep.py"          # where stop_production() is defined

    def sources(self):
        import glob as _g
        return sorted(_g.glob(str(common.REPO / "bench/**/*.py"), recursive=True))

    def test_only_one_place_stops_the_model_unit(self):
        """Any other file calling systemctl stop on a unit has to go through
        stop_production(), or the watchdog is left armed against nothing."""
        bad = []
        for path in self.sources():
            rel = os.path.relpath(path, str(common.REPO))
            if rel == self.ALLOWED:
                continue
            with open(path, encoding="utf-8") as fh:
                for n, line in enumerate(fh.read().splitlines(), 1):
                    if line.lstrip().startswith("#"):
                        continue
                    if re.search(r'systemctl_user\(\s*["\']stop["\']', line):
                        bad.append("    %s:%d  %s" % (rel, n, line.strip()[:64]))
        self.assertFalse(
            bad,
            "these stop a unit directly instead of calling "
            "sweep.stop_production(), which takes llama-probe.timer down with "
            "it. Every run longer than ten minutes then leaves a failed unit "
            "and a red line in check.sh that means nothing:\n%s"
            % "\n".join(bad))

    def test_the_helper_takes_the_timer_with_it(self):
        """The property itself, not just where it lives."""
        import inspect
        src = inspect.getsource(SWEEP.stop_production)
        self.assertIn("PROBE_TIMER", src)
        self.assertIn("stop", src)
        back = inspect.getsource(SWEEP.start_production)
        self.assertIn("PROBE_TIMER", back,
                      "it goes down with production and must come back with it")

    def test_the_scan_reads_the_bench_tree(self):
        """Positive control: a glob that matches nothing would pass the test
        above by checking nothing."""
        self.assertGreater(len(self.sources()), 15,
                           "only %d files found under bench/ — the scan is not "
                           "reading the tree" % len(self.sources()))


def variants_files():
    """EVERY variants file, not one of them.

    This class read `bench/variants/qwen38.json` and nothing else until
    04.09.2026, so a new file was covered by no test at all — and the first
    one written after that, bench/variants/glm47flash.json, carried
    `--port 8081` where sweep.py polls 8080. The sweep ran two cells, reported
    "model loaded, but /slots never answered" for both, and the servers were
    up the whole time answering /slots in 0.7 ms on the port nobody asked.
    Twenty-five minutes, and the failure text accused the model.

    A check that cannot go red for the case in front of it is not a check
    (bench/README.md); pinning one file out of three was that.
    """
    return sorted((common.REPO / "bench" / "variants").glob("*.json"))


class TestVariantsFile(unittest.TestCase):
    def setUp(self):
        self.specs = {}
        for p in variants_files():
            with open(p, encoding="utf-8") as f:
                self.specs[p.name] = json.load(f)
        self.assertTrue(self.specs, "no variants files found at all")
        # Kept so the assertions below can stay written against one spec.
        self.spec = self.specs["qwen38.json"]

    def test_every_variants_file_is_covered_here(self):
        """The guard on the guard: this file used to pin one name."""
        self.assertGreaterEqual(len(self.specs), 2)
        for name, spec in self.specs.items():
            self.assertIn("base_args", spec, name)
            self.assertIn("variants", spec, name)

    def test_shape_and_uniqueness(self):
        """The binary must be unambiguous — but AFTER expansion.

        This asserted `isabs` on the raw value until 27.08., which was right
        while the file carried eight absolute
        `/home/<user>/llama.cpp/build-*/...` paths and wrong the moment they
        became @HOME@ placeholders. The property worth having is unchanged:
        a sweep must never resolve its binary against the current working
        directory, because which build ran is half of what a report means.
        """
        import sys
        sys.path.insert(0, str(common.REPO / "setup" / "lib"))
        import systemdfile
        names = [v["name"] for v in self.spec["variants"]]
        self.assertEqual(len(names), len(set(names)))
        for v in self.spec["variants"]:
            self.assertFalse(os.path.isabs(v["binary"]),
                             "%s: an absolute path ties the file to one "
                             "machine — use @HOME@" % v["name"])
            self.assertTrue(os.path.isabs(systemdfile.expand(v["binary"])),
                            "%s does not expand to an absolute path" % v["name"])
            self.assertTrue(v["binary"].endswith("llama-server"), v["name"])
            self.assertIsInstance(v["args"], list)

    def test_the_json_kwargs_are_valid_json_in_every_variant(self):
        """The template accepts ONLY reasoning_effort xhigh/medium/low —
        'none' raises a Jinja exception (measured 24.08., cost one variant).
        Thinking is switched off via enable_thinking:false instead."""
        for v in self.spec["variants"]:
            args = v["args"]
            if "--chat-template-kwargs" in args:
                kw = json.loads(args[args.index("--chat-template-kwargs") + 1])
                effort = kw.get("reasoning_effort")
                if effort is not None:
                    self.assertIn(effort, ("xhigh", "medium", "low"), v["name"])
                else:
                    self.assertEqual(kw.get("enable_thinking"), False,
                                     v["name"])

    def test_base_args_pin_port_and_single_slot(self):
        """The port is compared against the one sweep.py ACTUALLY POLLS.

        Asserting the literal "8080" would be a second spelling of the same
        fact, and this repo has already paid for that once — a production unit
        hard-wired in two places, where a grep for one spelling did not find
        the other (CLAUDE.md). sweep.py's URL is the single source here, so a
        change to it moves the requirement rather than silently parting from
        it.
        """
        want = SWEEP.URL.rsplit(":", 1)[-1]
        # The positive control, and test_vacuity.py caught its absence the
        # first time this was written: everything below asserts inside a loop,
        # so an empty self.specs would make this test pass while checking
        # nothing — which is the precise defect the test itself exists for.
        self.assertGreaterEqual(len(self.specs), 2,
                                "no variants files read — this test would "
                                "pass without checking anything")
        self.assertRegex(want, r"^\d+$", "sweep.py's URL carries no port")
        for name, spec in self.specs.items():
            base = spec["base_args"]
            self.assertIn("--port", base, name)
            self.assertEqual(
                base[base.index("--port") + 1], want,
                "%s starts its servers on a port sweep.py does not poll — it "
                "polls %s. The symptom is every cell reporting that /slots "
                "never answered, while the servers are fine." % (name, SWEEP.URL))
            self.assertEqual(base[base.index("-np") + 1], "1", name)


class _FakeChat(http.server.BaseHTTPRequestHandler):
    """A llama-server double for /v1/chat/completions with timings."""
    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        body = json.dumps({
            "choices": [{"finish_reason": "stop", "message": {
                "content": "<think>short</think>x = 1",
                "reasoning_content": "let me think"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "timings": {"prompt_n": 100, "prompt_per_second": 300.0,
                        "predicted_n": 50, "predicted_per_second": 25.0},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestTheShapeThatIsMeasured(unittest.TestCase):
    """What survived the removal of the model battery on 26.08.

    `chat` and `summarize` went with bench/quality.py — they scored answers,
    which is a question this repo no longer asks. The remap did not: it is
    about the request SHAPE, not about how good the reply is. cc-gateway
    applies it to every real request (MID_SYSTEM_TO_USER) because Qwen's
    template rejects a system block that is not first with HTTP 500, so a
    measurement that skipped it would be timing a request production never
    sends — or not timing anything at all.
    """

    def test_mid_conversation_system_is_remapped_but_the_leading_one_stays(self):
        """Qwen's template 500s on system messages after position 0; the
        Claude-Code body carries exactly one there. The remap has to catch
        it — and must NOT touch the leading system message, or the probe
        would measure a different prefix than production sends."""
        p = {"system": [{"type": "text", "text": "head"}],
             "messages": [
                 {"role": "user", "content": [{"type": "text", "text": "q"}]},
                 {"role": "system", "content": "agent block"},
             ]}
        out = SPD._system_mid_conversation_remap(p)
        self.assertEqual(out["messages"][1]["role"], "user")
        self.assertEqual(out["messages"][1]["content"],
                         [{"type": "text", "text": "agent block"}])
        first_system = {"messages": [{"role": "system", "content": "s"},
                                     {"role": "user", "content": "q"}]}
        self.assertEqual(
            SPD._system_mid_conversation_remap(first_system)["messages"][0],
            {"role": "system", "content": "s"})

    def test_the_window_is_recorded_next_to_the_rate(self):
        """A t/s number without its -c is not comparable to another one:
        prefill and decode both move as the context grows. That is the whole
        reason for measuring per window."""
        self.assertEqual(SPD.ctx_of(["-ngl", "999", "-c", "262144"]), 262144)
        self.assertEqual(SPD.ctx_of(["--ctx-size", "65536"]), 65536)
        self.assertIsNone(SPD.ctx_of(["-ngl", "999"]))


class TestASweepThatMeasuredNothingSaysSo(unittest.TestCase):
    """The 04.09.2026 shape: six cells, six failures, exit 0, empty table.

    An absent comparison is also what a sweep with nothing to compare looks
    like, so the output has to separate the two. The same defect was fixed for
    both A/B suites earlier the same day (2690121); this sibling still had it.
    """

    SIX_FAILED = {n: "failed: model loaded, but /slots never answered"
                  for n in ("rocm-nospec", "vulkan-nospec", "rocm-ngram",
                            "vulkan-ngram", "rocm-mtp", "vulkan-mtp")}

    def test_the_04_09_shape_exits_2_and_says_nothing_was_measured(self):
        lines, code = SWEEP.verdict(self.SIX_FAILED)
        self.assertEqual(code, 2)
        text = "\n".join(lines)
        self.assertIn("6 of 6", text)
        self.assertIn("EVERY cell failed", text)

    def test_a_clean_sweep_states_the_tally_anyway(self):
        """Stated ALWAYS. A report that mentions failures only when there are
        some cannot be told apart from one written before it could mention
        them at all — which is every sweep report before today."""
        lines, code = SWEEP.verdict({"a": "ok", "b": "ok"})
        self.assertEqual(code, 0)
        self.assertIn("all 2 cells measured", "\n".join(lines))

    def test_some_failing_still_completes(self):
        """bench/README.md: a cell that fails is recorded, not fatal. Paid for
        by three reports that lost prefill-nospec to a restore timeout."""
        lines, code = SWEEP.verdict({"a": "ok", "b": "failed: wall cap"})
        self.assertEqual(code, 0)
        text = "\n".join(lines)
        self.assertIn("1 of 2 cells FAILED", text)
        self.assertIn("wall cap", text)

    def test_no_cell_attempted_is_not_success_either(self):
        lines, code = SWEEP.verdict({})
        self.assertEqual(code, 2)


class TestCompare(unittest.TestCase):
    def test_a_sweep_dir_renders_in_variant_order(self):
        """Variant order comes from context.json, not from the filesystem —
        a table sorted by mtime tells you nothing about the sweep.

        The fixture is what speed.run() actually writes — `depths`, with one
        cell per workload. It said `probes` until 27.08., which is the shape
        speed.py had stopped producing, so this test went on passing against a
        renderer that produced nothing but dashes for every real report.
        """
        with tempfile.TemporaryDirectory() as d:
            for name, ctx, gtt, tg in (("v-slow", 262144, 87.4, 12.4),
                                       ("v-fast", 65536, 80.4, 30.0)):
                os.makedirs(os.path.join(d, name))
                with open(os.path.join(d, name, "summary.json"), "w") as f:
                    json.dump({"label": name, "ctx": ctx, "gtt_gib": gtt,
                               "depths": [
                                   {"asked": 512, "workload": "prose",
                                    "depth_n": 624, "cached_pct": 0.0,
                                    "pp_tps": 300.0, "tg_tps": tg / 2},
                                   {"asked": 512, "workload": "count",
                                    "depth_n": 626, "cached_pct": 0.0, "pp_tps": 300.0,
                                    "tg_tps": tg},
                                   {"asked": 512, "workload": "copy",
                                    "depth_n": 631, "cached_pct": 0.0, "pp_tps": 299.0,
                                    "tg_tps": tg * 2, "copied_pct": 96.7}]}, f)
            with open(os.path.join(d, "context.json"), "w") as f:
                json.dump({"order": ["v-slow", "v-fast"], "reference": None}, f)
            md = CMP.render(d)
        self.assertIn("| v-slow | 262144 | 87.4 | 624 | 0 | 300.0 | 6.2 | 12.4 | 24.8 |", md)
        self.assertLess(md.index("v-slow"), md.index("v-fast"))
        self.assertNotIn("wall clock", md, "a current report must not be marked legacy")

    def test_a_cell_that_never_measured_is_a_gap_not_a_zero(self):
        """The rule that already governs measure.py: an absent number must not
        become 0, because a 0 reads as a finding. A silently invented '-0.0 %'
        once travelled into the documentation as one."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "dead"))
            with open(os.path.join(d, "dead", "summary.json"), "w") as f:
                json.dump({"label": "dead", "ctx": 65536, "depths": [
                    {"asked": 512, "workload": "count",
                     "error": "never served /slots"}]}, f)
            md = CMP.render(d)
        self.assertIn("never served /slots", md)
        self.assertIn("no measurement", md)
        self.assertNotIn("| 0 |", md)
        self.assertNotIn("| 0.0 |", md)


    def test_an_empty_dir_says_so_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("no summary.json", CMP.render(d))


if __name__ == "__main__":
    unittest.main()
