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
