"""Tests for prewarm.py — without llama-server.

req() is the only place that needs the server; it is replaced. That makes it
possible to check everything that would otherwise only show during a real
save — and that shows late, because a wrongly stored prefix produces no error,
it is simply never restored again.
"""
import json, os, shutil, tempfile, time, types, unittest
from unittest import mock

import common

VW  = common.load("tools/prewarm.py", "prewarm",
                     {"SLOT_PATH": "/nonexistent-slots"})
SYN = common.load("tools/synthetic.py", "synthetic")


def days_ago(n):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - n * 86400))


class WithStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="vw-")
        self.old, VW.SLOT_PATH = VW.SLOT_PATH, self.dir

    def tearDown(self):
        VW.SLOT_PATH = self.old
        shutil.rmtree(self.dir, ignore_errors=True)

    def put(self, name, size=1000, last=None, gk=None, with_bin=True):
        d = {"name": name, "ident": "k" * 12,
             "gateway_id": gk if gk is not None else name,
             "token": 22000, "bytes": size, "saved_at": days_ago(1)}
        if last:
            d["last_used"] = last
        with open(os.path.join(self.dir, name + ".json"), "w") as f:
            json.dump(d, f)
        if with_bin:
            with open(os.path.join(self.dir, name + ".bin"), "wb") as f:
                f.write(b"x" * size)

    def present(self, name):
        return os.path.exists(os.path.join(self.dir, name + ".bin"))


class TestBuildPrefix(WithStore):
    def test_volatile_drops_out_and_the_cut_is_at_user(self):
        seen = {}
        def fake_req(path, nutzlast=None, method=None, t=1800):
            seen["oai"] = nutzlast
            return {"prompt": "KOPF" + nutzlast["messages"][0]["content"] + "<user>X"}
        with mock.patch.object(VW, "req", fake_req):
            prefix = VW.build_prefix(SYN.body(turns=1))
        self.assertNotIn("<user>", prefix)
        self.assertNotIn("total_tokens", prefix,
                         "the volatile counter does not belong in the prefix")
        self.assertIn("Available agent types", prefix,
                      "the stable agent block has to be hoisted")


class TestSave(WithStore):
    def _fake_req(self):
        def req(path, nutzlast=None, method=None, t=1800):
            if path == "/apply-template":
                return {"prompt": "PREFIX<user>X"}
            if path == "/tokenize":
                return {"tokens": list(range(22000))}
            if path == "/completion":
                return {"id_slot": 0}
            if path == "/slots":
                return [{"id": 0, "n_prompt_tokens": 22000}]
            if path.startswith("/slots/0?action=save"):
                return {"n_saved": 22000, "n_written": 628_000_000,
                        "timings": {"save_ms": 247}}
            raise AssertionError("unerwarteter Pfad %s" % path)
        return req

    def do_save(self, **kw):
        body = os.path.join(self.dir, "body.json")
        with open(body, "w") as f:
            json.dump(SYN.body(), f)
        a = types.SimpleNamespace(body=body, name="ziel", **kw)
        with mock.patch.object(VW, "req", self._fake_req()), \
             mock.patch.object(VW, "wait_until_ready", lambda t=900: True), \
             mock.patch("builtins.print"):
            VW.save(a)
        with open(os.path.join(self.dir, "ziel.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_passed_in_id_is_taken_over(self):
        """The heart of the bug: the gateway hands over an already corrected
        body. Recomputing the id from it puts a key into the store that no
        request ever produces."""
        d = self.do_save(gateway_id="abc123abc123")
        self.assertEqual(d["gateway_id"], "abc123abc123")

    def test_without_it_the_body_is_used(self):
        d = self.do_save(gateway_id=None)
        with open(os.path.join(self.dir, "body.json"), encoding="utf-8") as f:
            self.assertEqual(d["gateway_id"], VW.gateway_id(json.load(f)))


class TestCheck(WithStore):
    def test_finds_the_wrongly_stored_ones(self):
        self.put("abc123abc123", gk="97c89b475106")   # the real case
        self.put("projA", gk="b2205fae3e1c")          # saved by hand, fine
        with mock.patch("builtins.print"):
            self.assertEqual(VW.check(types.SimpleNamespace(repair=False)), 1)

    def test_repairs_and_is_found_afterwards(self):
        self.put("abc123abc123", gk="97c89b475106")
        with mock.patch("builtins.print"):
            VW.check(types.SimpleNamespace(repair=True))
            with open(os.path.join(self.dir, "abc123abc123.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["gateway_id"], "abc123abc123")
            self.assertEqual(VW.check(types.SimpleNamespace(repair=False)), 0)

    def test_reports_a_sidecar_without_content(self):
        self.put("abc123abc123", with_bin=False)
        with mock.patch("builtins.print") as p:
            VW.check(types.SimpleNamespace(repair=False))
        self.assertTrue(any(".bin missing" in str(c) for c in p.call_args_list))


class TestCleanup(WithStore):
    def clean(self, **kw):
        values = {"max_gb": None, "max_count": None, "ttl_days": None,
                  "dry_run": False, "purge": False}
        values.update(kw)
        with mock.patch("builtins.print"):
            VW.cleanup(types.SimpleNamespace(**values))

    def test_ttl_deletes_the_long_unused(self):
        self.put("old", last=days_ago(40))
        self.put("frisch", last=days_ago(2))
        self.clean(ttl_days=28)
        self.assertFalse(self.present("old"))
        self.assertTrue(self.present("frisch"))

    def test_last_used_beats_the_file_date(self):
        # Both saved equally long ago, but one is used daily.
        self.put("taeglich", last=days_ago(1))
        self.put("vergessen", last=days_ago(90))
        self.clean(max_count=1)
        self.assertTrue(self.present("taeglich"))
        self.assertFalse(self.present("vergessen"))

    def test_upper_limit_in_gb(self):
        for n, tage in (("a", 1), ("b", 2), ("c", 3)):
            self.put(n, size=1000, last=days_ago(tage))
        self.clean(max_gb=2e-6)          # 2000 Bytes
        self.assertTrue(self.present("a"))
        self.assertTrue(self.present("b"))
        self.assertFalse(self.present("c"))

    def test_dry_run_deletes_nothing(self):
        self.put("old", last=days_ago(90))
        self.clean(ttl_days=1, dry_run=True)
        self.assertTrue(self.present("old"))

    def test_zero_is_refused_rather_than_guessed(self):
        """`--max-gb 0` in a tool whose job is deleting reads as "keep
        nothing". The code read it as "no limit", because `if a.max_gb:` is
        false at zero — so the command ran, printed "nothing to delete", and
        did exactly nothing. Twice, on 28.08.2026, before anyone looked at the
        source.

        Both readings are defensible, which is the whole problem: a number
        that can mean either must not be interpreted silently. Same rule as
        budget._num, which refuses a typo rather than falling back — "silently
        falling back would hide a mistake in the one place where a number is
        being trusted".
        """
        self.put("a", last=days_ago(1))
        for flag, kw in (("--max-gb", {"max_gb": 0}),
                         ("--max-count", {"max_count": 0}),
                         ("--ttl-days", {"ttl_days": 0})):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit) as cm:
                    self.clean(**kw)
                msg = str(cm.exception)
                self.assertIn(flag, msg)
                self.assertIn("--purge", msg, "the message must name the way "
                              "to say 'keep nothing'")
                self.assertIn("omit", msg, "and the way to say 'no limit'")
        self.assertTrue(self.present("a"), "nothing may be deleted on refusal")

    def test_a_negative_limit_is_refused_too(self):
        self.put("a", last=days_ago(1))
        with self.assertRaises(SystemExit):
            self.clean(max_gb=-1)
        self.assertTrue(self.present("a"))

    def test_purge_deletes_everything(self):
        self.put("a", last=days_ago(1))
        self.put("b", last=days_ago(90))
        self.clean(purge=True)
        self.assertFalse(self.present("a"))
        self.assertFalse(self.present("b"))

    def test_purge_honours_dry_run(self):
        """The one flag that deletes without a rule has to be previewable."""
        self.put("a", last=days_ago(1))
        self.clean(purge=True, dry_run=True)
        self.assertTrue(self.present("a"))

    def test_purge_together_with_a_limit_is_refused(self):
        """"Keep nothing" and "keep this much" cannot both be meant."""
        self.put("a", last=days_ago(1))
        with self.assertRaises(SystemExit):
            self.clean(purge=True, max_gb=20)
        self.assertTrue(self.present("a"))

    def test_the_cli_offers_purge(self):
        """A rule reachable only by constructing a namespace is not a rule a
        user can reach."""
        src = (common.REPO / "tools" / "prewarm.py").read_text(encoding="utf-8")
        self.assertIn('"--purge"', src)

    def test_without_a_rule_everything_stays(self):
        self.put("a"); self.put("b")
        self.clean()
        self.assertTrue(self.present("a") and self.present("b"))


class TestRestore(WithStore):
    def _req(self, slots):
        def req(path, nutzlast=None, method=None, t=1800):
            if path == "/slots":
                return slots
            if "action=restore" in path:
                return {"n_restored": 22000, "timings": {"restore_ms": 97}}
            raise AssertionError(path)
        return req

    def run_one(self, slots):
        lines = []
        a = types.SimpleNamespace()
        with mock.patch.object(VW, "req", self._req(slots)), \
             mock.patch.object(VW, "wait_until_ready", lambda t=900: True), \
             mock.patch("builtins.print",
                        lambda *x, **k: lines.append(" ".join(map(str, x)))):
            restored = VW.restore(a)
        return restored, lines

    @staticmethod
    def restored_lines(lines):
        """Only the lines that report a restore."""
        return [z for z in lines if "<-" in z and "FEHLER" not in z]

    def test_all_slots_free(self):
        self.put("a"); self.put("b")
        restored, lines = self.run_one([{"id": 0}, {"id": 1}])
        self.assertEqual(restored, 2)
        self.assertNotIn("NOT restored", " ".join(lines))

    def test_busy_slots_are_reported(self):
        """The message used to stay away as soon as there were not MORE saved
        prefixes than slots — skipped ones went unnoticed, and the caller is
        ExecStartPost, where nobody reads along."""
        self.put("a"); self.put("b"); self.put("c"); self.put("d")
        restored, lines = self.run_one([{"id": 0}, {"id": 1, "n_prompt_tokens": 22000},
                                    {"id": 2, "n_prompt_tokens": 22000}, {"id": 3}])
        out = " ".join(lines)
        self.assertEqual(restored, 2)
        self.assertIn("NOT restored", out)
        self.assertIn("2 of 4 restored", out)

    def test_last_used_go_into_the_slots_first(self):
        """Sorted alphabetically, a prefix with seven tokens stood ahead of two
        real projects in production — with two slots it would have evicted one."""
        self.put("aaa-tiny", last=days_ago(30))
        self.put("projA",      last=days_ago(1))
        self.put("projB",      last=days_ago(2))
        restored, lines = self.run_one([{"id": 0}, {"id": 1}])
        restored_lines = " ".join(self.restored_lines(lines))
        self.assertEqual(restored, 2)
        self.assertIn("projA", restored_lines)
        self.assertIn("projB", restored_lines)
        self.assertNotIn("aaa-tiny", restored_lines)

    def test_a_broken_sidecar_does_not_stop_the_service(self):
        """restore runs as ExecStartPost — if it raises, the model server does
        not start."""
        self.put("good")
        with open(os.path.join(self.dir, "broken.json"), "w") as f:
            f.write("{this is not json")
        restored, lines = self.run_one([{"id": 0}, {"id": 1}])
        self.assertEqual(restored, 1)
        self.assertIn("unreadable", " ".join(lines))

    def test_sidecar_without_bin_is_not_even_tried(self):
        self.put("without", with_bin=False)
        self.put("with")
        restored, lines = self.run_one([{"id": 0}, {"id": 1}])
        self.assertEqual(restored, 1)
        self.assertNotIn("without", " ".join(self.restored_lines(lines)))


if __name__ == "__main__":
    unittest.main()
