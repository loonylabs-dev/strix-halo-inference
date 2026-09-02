"""Tests for cc-gateway — without a GPU, without llama-server, without a service.

llama-server is replaced by a small aiohttp application that records what
arrives at it. That makes it possible to check what the smoke test cannot do
deterministically against the real stack: the per-access throttle, cancelled
callers, and above all the contract between the gateway's id and the store on
disk.
"""
import asyncio, json, os, re, shutil, tempfile, unittest, urllib.error
from unittest import mock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

import common

# TRACE_DIR points the tracing away from the operator's real one. Without it
# the suite writes its fixtures — `abc123`, `id1`, who=`tester` — into
# ~/.cache/cc-gateway-trace whenever tracing happens to be switched on, and an
# analysis of that file then reports twelve quarantines that never happened.
# Found on 29.08.2026 by reading the trace of a real morning and nearly
# drawing a conclusion from test data.
GW  = common.load("setup/gateway/gateway.py", "gateway",
                     {"MAX_INFLIGHT": "2", "TOKEN_FILE": "/nonexistent-token",
                      "SLOT_PATH": "/nonexistent-slots",
                      "TRACE_DIR": "/nonexistent-trace"})
VW  = common.load("tools/prewarm.py", "prewarm",
                     {"SLOT_PATH": "/nonexistent-slots"})
SYN = common.load("tools/synthetic.py", "synthetic")
DIA = common.load("setup/gateway/dialects.py", "dialects")


# --------------------------------------------------------- Cache-Korrektur ---
class TestCorrection(unittest.TestCase):
    """The stable prefix has to stay equal character for character across
    turns. If VOLATILE one day no longer matches the counter, it wanders into
    the prefix, the id changes every turn, and EVERY request runs cold —
    without any error message. Nobody would otherwise notice."""

    def test_prefix_stays_equal_across_turns(self):
        a, _ = GW.correct(SYN.body(turns=1, budget_left=15000000))
        b, _ = GW.correct(SYN.body(turns=4, budget_left=14211873))
        self.assertEqual(a["system"], b["system"],
                         "the hoisted prefix must not differ between turns")


    def test_counter_is_found_and_stays_behind(self):
        p, n = GW.correct(SYN.body(turns=1))
        self.assertGreaterEqual(n, 1, "the <total_tokens> counter was not found")
        self.assertNotIn("total_tokens", json.dumps(p["system"]),
                         "a volatile counter has got into the prefix")
        self.assertIn("total_tokens", json.dumps(p["messages"]),
                      "the counter must not be lost, only moved to the back")

    def test_agent_block_moves_to_the_front(self):
        p, _ = GW.correct(SYN.body(turns=1))
        self.assertIn("Available agent types", json.dumps(p["system"]))
        self.assertNotIn("Available agent types", json.dumps(p["messages"]))

    def test_id_depends_on_system_and_tools(self):
        k1 = GW.prefix_id(SYN.body(n_tools=24))[0]
        k2 = GW.prefix_id(SYN.body(n_tools=25))[0]
        k3 = GW.prefix_id(SYN.body(project="/tmp/projB"))[0]
        self.assertNotEqual(k1, k2)
        self.assertNotEqual(k1, k3)

    def test_id_ignores_the_question(self):
        k1 = GW.prefix_id(SYN.body(question="Sag alpha."))[0]
        k2 = GW.prefix_id(SYN.body(question="Etwas ganz anderes."))[0]
        self.assertEqual(k1, k2)


# ------------------------------------------------------ Kennungs-Vertrag ---
class TestMidSystemToUser(unittest.TestCase):
    """Qwen 3.8's template rejects system messages after position 0 with a
    500 — and the Claude Code body carries exactly one (the volatile counter
    that correct() deliberately leaves in place). The opt-in rewrite has to
    clear every such message without touching the prefix identity, or
    switching models silently kills either the requests or the cache."""

    def test_no_system_role_survives_after_position_zero(self):
        p, _ = GW.correct(SYN.body(turns=3))
        p, n = GW.mid_system_to_user(p)
        self.assertGreater(n, 0)
        roles = [m["role"] for m in p["messages"][1:]]
        self.assertNotIn("system", roles)

    def test_the_counter_text_stays_in_place_as_a_user_block(self):
        p, _ = GW.correct(SYN.body(turns=1, budget_left=4242))
        idx = [i for i, m in enumerate(p["messages"])
               if m["role"] == "system"]
        p, _ = GW.mid_system_to_user(p)
        m = p["messages"][idx[0]]
        self.assertEqual(m["role"], "user")
        self.assertIn("4242 tokens left", m["content"][0]["text"])

    def test_a_leading_system_message_is_left_alone(self):
        p = {"messages": [{"role": "system", "content": "s"},
                          {"role": "user", "content": "q"}]}
        p, n = GW.mid_system_to_user(p)
        self.assertEqual(n, 0)
        self.assertEqual(p["messages"][0]["role"], "system")

    def test_the_prefix_id_does_not_change(self):
        """The id comes from system field and tools; if the rewrite ever
        leaked into it, every saved prefix would go cold on the switch."""
        p, _ = GW.correct(SYN.body(turns=2))
        before = GW.prefix_id(p)
        p, _ = GW.mid_system_to_user(p)
        self.assertEqual(GW.prefix_id(p), before)

    def test_the_rewrite_is_off_by_default(self):
        self.assertFalse(GW.MID_SYSTEM_TO_USER)


class TestModelKwargs(unittest.TestCase):
    """One loaded model, several thinking modes: the map fills
    chat_template_kwargs by model name. If it ever overwrote what a request
    already carries, an explicit caller could no longer opt out."""

    TABLE = {"qwen38": {"enable_thinking": False},
             "qwen38-think": {"reasoning_effort": "medium"}}

    def test_the_map_fills_by_model_name(self):
        p, hit = GW.inject_model_kwargs({"model": "qwen38"}, self.TABLE)
        self.assertTrue(hit)
        self.assertEqual(p["chat_template_kwargs"], {"enable_thinking": False})

    def test_request_kwargs_win_key_by_key(self):
        p, _ = GW.inject_model_kwargs(
            {"model": "qwen38-think",
             "chat_template_kwargs": {"reasoning_effort": "low"}}, self.TABLE)
        self.assertEqual(p["chat_template_kwargs"],
                         {"reasoning_effort": "low"})

    def test_an_unknown_model_passes_untouched(self):
        p, hit = GW.inject_model_kwargs({"model": "laguna"}, self.TABLE)
        self.assertFalse(hit)
        self.assertNotIn("chat_template_kwargs", p)

    def test_the_map_is_empty_by_default(self):
        self.assertEqual(GW.KWARGS_BY_MODEL, {})


class TestKwargsBelongToTheServedModel(unittest.TestCase):
    """The map matches a NAME. It has to belong to the model that is loaded.

    There is one llama-server, and which model it holds is decided by
    switch-model.sh, not by the request. The consumer's name is set somewhere
    else again — ANTHROPIC_MODEL in setup/claude/local.json, today
    `qwen38-think`. Nothing tied the two together, so a switch to another model
    left the client asking for a name that was no longer served, and the map
    answered it anyway: qwen38's thinking mode injected into a request bound
    for Flash-Next, over a command line that had set it otherwise, silently.

    KNOWN LIMIT, and it is why this is a stopgap rather than the fix: the rule
    is per-TABLE. The moment KWARGS_BY_MODEL names two profiles — the obvious
    thing to do on a machine with seven of them — the served model IS in the
    table, the guard switches off, and the fault is re-armed by a config change
    nobody would think twice about. A bare name cannot say which base model it
    belongs to; `served not in table` is a proxy for ownership that holds only
    while the table describes one model. The structural answer is to derive the
    names FROM the served alias instead of listing them in a third file.
    """

    TABLE = {"qwen38": {"enable_thinking": False},
             "qwen38-think": {"reasoning_effort": "medium"}}

    def test_the_served_model_still_gets_its_modes(self):
        p, hit = GW.inject_model_kwargs({"model": "qwen38-think"}, self.TABLE,
                                        served="qwen38")
        self.assertTrue(hit)
        self.assertEqual(p["chat_template_kwargs"], {"reasoning_effort": "medium"})

    def test_a_name_from_another_model_injects_nothing(self):
        """The switch happened, the client did not follow."""
        p, hit = GW.inject_model_kwargs({"model": "qwen38-think"}, self.TABLE,
                                        served="flashnext")
        self.assertFalse(hit)
        self.assertNotIn("chat_template_kwargs", p,
                         "qwen38's thinking mode reached Flash-Next")

    def test_a_map_written_for_the_served_model_applies(self):
        table = dict(self.TABLE, flashnext={"enable_thinking": False})
        p, hit = GW.inject_model_kwargs({"model": "flashnext"}, table,
                                        served="flashnext")
        self.assertTrue(hit)
        self.assertEqual(p["chat_template_kwargs"], {"enable_thinking": False})

    def test_the_known_limit_is_pinned_rather_than_believed_away(self):
        """A table naming two models turns the guard off. Asserted so that the
        docstring above cannot quietly become untrue, and so that whoever adds
        a second profile to KWARGS_BY_MODEL meets this test first."""
        table = dict(self.TABLE, flashnext={"enable_thinking": False})
        p, hit = GW.inject_model_kwargs({"model": "qwen38-think"}, table,
                                        served="flashnext")
        self.assertTrue(hit, "the guard is per-table; this is the known hole")

    def test_an_unknown_served_model_keeps_the_old_behaviour(self):
        """llama-server may not have answered yet. Refusing to inject would
        silently drop the daily thinking mode every time the lookup failed,
        which is a regression dressed as caution."""
        p, hit = GW.inject_model_kwargs({"model": "qwen38-think"}, self.TABLE,
                                        served=None)
        self.assertTrue(hit)


class TestTheServedModelIsAskedForOnce(unittest.TestCase):
    """Where the lookup lives is not a detail — it was the first attempt.

    Resolving it lazily in the request path worked and broke
    TestZoneRemote::test_valid_token_200, which asserts that one client request
    causes exactly ONE upstream request. That assertion caught a per-request
    round trip nobody would have noticed except as latency. So the lookup sits
    in main(), beside query_slots, and the module import stays free of network.
    """

    def listing(self, payload):
        import io, json as J
        class Resp(io.BytesIO):
            def __enter__(self_): return self_
            def __exit__(self_, *a): return False
        return mock.patch("urllib.request.urlopen",
                          lambda *a, **k: Resp(J.dumps(payload).encode()))

    def test_it_reads_the_name_the_server_reports(self):
        with self.listing({"data": [{"id": "qwen38"}]}):
            self.assertEqual(GW.query_served_model(), "qwen38")

    def test_a_server_that_does_not_answer_is_not_a_name(self):
        def boom(*a, **k):
            raise OSError("connection refused")
        with mock.patch("urllib.request.urlopen", boom):
            self.assertIsNone(GW.query_served_model())

    def test_an_unparseable_listing_is_not_a_name(self):
        """The fake llama-server in these tests answers {"ok": true} to every
        path. A lookup that read a name out of that would be inventing one."""
        with self.listing({"ok": True, "path": "/v1/models"}):
            self.assertIsNone(GW.query_served_model())

    def test_main_asks_at_startup(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(encoding="utf-8")
        body = src[src.index("def main("):]
        self.assertIn("query_served_model()", body,
                      "main() never learns which model is served")

    def test_both_staleness_detectors_call_the_refresh(self):
        """Not a grep for the word SERVED — that is what the first version of
        this test did, and it stayed green while the second of watch_server's
        two branches left the name stale. Both branches must reach the same
        call."""
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(encoding="utf-8")
        body = src[src.index("async def watch_server("):src.index("async def note_server_restart(")]
        self.assertEqual(body.count("restarted = True"), 2,
                         "one of the two detectors does not mark a restart")
        self.assertEqual(body.count("await note_server_restart()"), 1,
                         "the refresh must be reached from one place, after both")


class TestIdContract(unittest.TestCase):
    """The bug that made the automatic saving useless.

    The gateway forms the id from the RAW body but hands prewarm.py the
    CORRECTED one — only from that can the prefix be rendered that later
    actually arrives. Recomputing the id there yields a different value and
    writes a key into the store that no request ever produces.
    store that no request ever produces.
    """

    def setUp(self):
        self.raw_body = SYN.body(turns=1)
        self.ident = GW.prefix_id(json.loads(json.dumps(self.raw_body)))[0]

    def test_raw_body_yields_the_same_id(self):
        self.assertEqual(VW.gateway_id(self.raw_body), self.ident,
                         "in manual use prewarm.py has to arrive at the same"
                         " id as the gateway")

    def test_corrected_body_yields_a_different_one(self):
        corrected, _ = GW.correct(json.loads(json.dumps(self.raw_body)))
        self.assertNotEqual(
            VW.gateway_id(corrected), self.ident,
            "If this ever becomes equal, --gateway-id has become superfluous. "
            "As long as they differ, the id MUST NOT be recomputed in "
            "automatic operation.")

    def test_both_sides_agree_in_both_dialects(self):
        """Gateway and prewarm now share dialects.py — but they are still two
        processes, and only this holds them together. If it ever breaks, a
        saved prefix lands under a key nobody produces: SAVED in the log,
        file on disk, RESTORED never — the id contract, pinned here and in
        tests/test_dialects.py."""
        oai = {"model": "qwen38",
               "messages": [{"role": "system", "content": "You are an agent."},
                            {"role": "user", "content": "hi"}],
               "tools": [{"type": "function",
                          "function": {"name": "Read", "description": "d",
                                       "parameters": {"type": "object"}}}]}
        for dialect, body in ((GW.DIA.ANTHROPIC, self.raw_body),
                              (GW.DIA.OPENAI, oai)):
            with self.subTest(dialect=dialect):
                gw = GW.prefix_id(json.loads(json.dumps(body)), dialect)[0]
                pw = VW.gateway_id(json.loads(json.dumps(body)), dialect)
                self.assertEqual(gw, pw)

    def test_head_bytes_are_not_defined_twice(self):
        self.assertIs(GW.HEAD_BYTES, GW.DIA.HEAD_BYTES)


# ------------------------------------------------------------ store/disk ---
class TestStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="slots-")
        self.old = GW.SLOT_PATH
        GW.SLOT_PATH = self.dir
        GW.SAVED = {}
        GW._saved_state = object()
        self.log_patch = mock.patch.object(GW, "log", lambda *a: None)
        self.log_patch.start()

    def tearDown(self):
        self.log_patch.stop()
        GW.SLOT_PATH = self.old
        shutil.rmtree(self.dir, ignore_errors=True)

    def put(self, name, gk, content=b"x" * 32):
        with open(os.path.join(self.dir, name + ".json"), "w") as f:
            json.dump({"name": name, "gateway_id": gk, "token": 1,
                       "bytes": len(content), "saved_at": "2026-08-01 10:00"}, f)
        with open(os.path.join(self.dir, name + ".bin"), "wb") as f:
            f.write(content)

    def test_store_is_indexed_by_the_gateway_id(self):
        self.put("abc123abc123", "abc123abc123")
        self.assertEqual(GW.load_saved(), {"abc123abc123": "abc123abc123"})

    def test_sidecar_without_bin_does_not_count(self):
        with open(os.path.join(self.dir, "leer.json"), "w") as f:
            json.dump({"name": "leer", "gateway_id": "kkkkkkkkkkkk"}, f)
        self.assertEqual(GW.load_saved(), {})

    def test_refresh_notices_deleted_files(self):
        self.put("abc123abc123", "abc123abc123")
        self.assertIn("abc123abc123", GW.refresh_saved(force=True))
        os.remove(os.path.join(self.dir, "abc123abc123.bin"))
        # The cleanup service deletes in its own process. Without refreshing,
        # the gateway would run into a restore onto nothing while holding a
        # Leere.
        self.assertNotIn("abc123abc123", GW.refresh_saved())

    def test_restore_gives_up_when_the_file_is_gone(self):
        self.put("abc123abc123", "abc123abc123")
        GW.refresh_saved(force=True)
        os.remove(os.path.join(self.dir, "abc123abc123.bin"))
        self.assertFalse(asyncio.run(GW.restore_from_disk("abc123abc123")))
        self.assertEqual(GW.SAVED, {})


# ------------------------------------------------------ Automatik-Sichern ---
class TestAutoSave(unittest.IsolatedAsyncioTestCase):
    """Checks the call to prewarm.py without actually running it."""

    async def asyncSetUp(self):
        self.dir = tempfile.mkdtemp(prefix="slots-")
        self.old_path, GW.SLOT_PATH = GW.SLOT_PATH, self.dir
        self.old_max, GW.SAVE_QUEUE_MAX = GW.SAVE_QUEUE_MAX, 4
        GW.SAVED = {}
        GW._saved_state = object()
        GW._save_lock = None
        GW._save_pending.clear()
        self.calls = []
        self.log_lines = []
        self.log_patch = mock.patch.object(
            GW, "log", lambda *a: self.log_lines.append(" ".join(map(str, a))))
        self.log_patch.start()

    async def asyncTearDown(self):
        self.log_patch.stop()
        GW.SLOT_PATH = self.old_path
        GW.SAVE_QUEUE_MAX = self.old_max
        shutil.rmtree(self.dir, ignore_errors=True)

    def _fake_exec(self, delay=0.0, write_files=True):
        """Stand-in for prewarm.py: writes the sidecar file exactly the way
        the real script does with --gateway-id."""
        calls, verz = self.calls, delay
        dir_ = self.dir

        class Proc:
            returncode = 0
            async def communicate(self):
                if verz:
                    await asyncio.sleep(verz)
                return (b"", b"")

        async def exec_(*argv, **kw):
            calls.append(list(argv))
            if write_files:
                name = argv[argv.index("--name") + 1]
                gk = argv[argv.index("--gateway-id") + 1]
                with open(os.path.join(dir_, name + ".json"), "w") as f:
                    json.dump({"name": name, "gateway_id": gk}, f)
                with open(os.path.join(dir_, name + ".bin"), "wb") as b:
                    b.write(b"x")
            return Proc()
        return exec_

    async def test_id_is_passed_through_and_found_again(self):
        raw_body = SYN.body()
        ident = GW.prefix_id(json.loads(json.dumps(raw_body)))[0]
        corrected, _ = GW.correct(json.loads(json.dumps(raw_body)))
        with mock.patch("asyncio.create_subprocess_exec", self._fake_exec()):
            await GW.auto_save(ident, corrected)
        self.assertEqual(len(self.calls), 1)
        argv = self.calls[0]
        self.assertIn("--gateway-id", argv)
        self.assertEqual(argv[argv.index("--gateway-id") + 1], ident)
        # The actual point: the freshly written store has to contain the id
        # that the next request will arrive with.
        self.assertIn(ident, GW.SAVED,
                      "saved_ids, but under a key that no request ever "
                      "produces — exactly the old bug")

    async def test_second_save_is_not_dropped_silently(self):
        k1, k2 = "a" * 12, "b" * 12
        with mock.patch("asyncio.create_subprocess_exec",
                        self._fake_exec(delay=0.05)):
            await asyncio.gather(GW.auto_save(k1, {"system": "a"}),
                                 GW.auto_save(k2, {"system": "b"}))
        self.assertEqual(len(self.calls), 2,
                         "the second save was discarded — the prefix then "
                         "counts as warm and would never come up again")

    async def test_the_same_id_is_saved_only_once(self):
        k = "c" * 12
        with mock.patch("asyncio.create_subprocess_exec",
                        self._fake_exec(delay=0.05)):
            await asyncio.gather(GW.auto_save(k, {"system": "a"}),
                                 GW.auto_save(k, {"system": "a"}))
        self.assertEqual(len(self.calls), 1)

    async def test_queue_is_capped_but_not_concealed(self):
        GW.SAVE_QUEUE_MAX = 2
        messages = []
        with mock.patch("asyncio.create_subprocess_exec",
                        self._fake_exec(delay=0.05)), \
             mock.patch.object(GW, "log", lambda *a: messages.append(" ".join(map(str, a)))):
            await asyncio.gather(*[GW.auto_save("%012d" % i, {"system": str(i)})
                                   for i in range(5)])
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(any("not saved" in m for m in messages),
                        "a discarded save has to appear in the log")


# ------------------------------------------------------------- Schleuse ---
class TestSavePhaseLifesign(unittest.IsolatedAsyncioTestCase):
    """The save-before-serve phase must not be silent.

    A cold Claude-Code-sized prefix prefills for 100-145 s inside
    save_prefix_first, and until 31.08.2026 nothing was written to the caller
    for all of it — over Cloudflare's ~125 s window, so a remote streaming
    caller got a 524 while the save marched on. Same rule as the queue phase
    (test_a_queued_streaming_caller_gets_a_sign_of_life): a streaming caller
    gets ":\\n\\n" every KEEPALIVE seconds, a plain one cannot be kept alive
    without spending its status code and stays untouched.
    """

    class Resp:
        def __init__(self, status=200, headers=None):
            self.writes = []
            self.prepared = None

        async def prepare(self, req):
            self.prepared = req

        async def write(self, data):
            self.writes.append(data)

    class Req:
        remote = "unit-test"

    async def _run(self, streaming, resp, delay=0.3):
        async def slow_save(id_, body, dialect):
            await asyncio.sleep(delay)
        with mock.patch.object(GW, "auto_save", slow_save), \
             mock.patch.object(GW, "log", lambda *a: None):
            old_ka, GW.KEEPALIVE = GW.KEEPALIVE, 0.05
            try:
                return await GW.save_prefix_first(
                    "p1", {}, req=self.Req(), resp=resp, streaming=streaming)
            finally:
                GW.KEEPALIVE = old_ka

    async def test_a_streaming_caller_is_pinged_while_the_prefix_is_saved(self):
        resp = self.Resp()
        out = await self._run(streaming=True, resp=resp)
        self.assertIs(out, resp)
        self.assertGreaterEqual(len(resp.writes), 2,
                                "no sign of life during the save")
        self.assertTrue(all(w == b":\n\n" for w in resp.writes))

    async def test_the_stream_is_opened_if_none_is_open_yet(self):
        with mock.patch.object(GW.web, "StreamResponse", self.Resp):
            out = await self._run(streaming=True, resp=None)
        self.assertIsInstance(out, self.Resp)
        self.assertIsNotNone(out.prepared, "headers never went out")
        self.assertGreaterEqual(len(out.writes), 2)

    async def test_a_plain_caller_keeps_its_status_code(self):
        out = await self._run(streaming=False, resp=None)
        self.assertIsNone(
            out, "a non-streaming caller must not get a committed stream")


class TestPriorityGate(unittest.IsolatedAsyncioTestCase):
    async def test_priority_beats_arrival(self):
        s = GW.PriorityGate(1)
        await s.enter(0)                       # Platz belegt
        order = []
        async def waiter(prio, name):
            await s.enter(prio)
            order.append(name)
        late = asyncio.create_task(waiter(2, "extern"))
        await asyncio.sleep(0)
        early = asyncio.create_task(waiter(0, "lokal"))
        await asyncio.sleep(0)
        s.leave(); await asyncio.sleep(0)
        s.leave(); await asyncio.gather(late, early)
        self.assertEqual(order, ["lokal", "extern"])

    async def test_ageing_beats_priority_once_the_wait_is_long_enough(self):
        """Priority decides who goes first, not who goes at all."""
        old, GW.AGE_AFTER = GW.AGE_AFTER, 0
        try:
            g = GW.PriorityGate(1)
            await g.enter(0)
            order = []
            async def waiter(prio, name):
                await g.enter(prio)
                order.append(name)
            remote = asyncio.create_task(waiter(2, "remote"))
            await asyncio.sleep(0)
            local = asyncio.create_task(waiter(0, "local"))
            await asyncio.sleep(0)
            g.leave(); await asyncio.sleep(0)
            g.leave(); await asyncio.gather(remote, local)
            self.assertEqual(order[0], "remote",
                             "the one that waited longest has to go first once "
                             "it has aged in")
            self.assertEqual(g.overtaken, 2)
        finally:
            GW.AGE_AFTER = old

    async def test_a_waiter_is_not_starved_by_newcomers(self):
        """Measured on the real stack: with four local streams a LAN caller was
        still waiting after 200 s. Newer arrivals must not keep overtaking."""
        old, GW.AGE_AFTER = GW.AGE_AFTER, 0.05
        try:
            g = GW.PriorityGate(1)
            await g.enter(0)
            served = []
            async def waiter(prio, name):
                await g.enter(prio)
                served.append(name)
            remote = asyncio.create_task(waiter(2, "remote"))
            await asyncio.sleep(0.06)              # let it age in
            newer = [asyncio.create_task(waiter(0, "local%d" % i)) for i in range(3)]
            await asyncio.sleep(0)
            for _ in range(4):
                g.leave()
                await asyncio.sleep(0)
            await asyncio.gather(remote, *newer)
            self.assertEqual(served[0], "remote")
        finally:
            GW.AGE_AFTER = old

    async def test_priority_still_wins_while_everyone_is_fresh(self):
        old, GW.AGE_AFTER = GW.AGE_AFTER, 3600
        try:
            g = GW.PriorityGate(1)
            await g.enter(0)
            order = []
            async def waiter(prio, name):
                await g.enter(prio)
                order.append(name)
            late = asyncio.create_task(waiter(2, "remote"))
            await asyncio.sleep(0)
            early = asyncio.create_task(waiter(0, "local"))
            await asyncio.sleep(0)
            g.leave(); await asyncio.sleep(0)
            g.leave(); await asyncio.gather(late, early)
            self.assertEqual(order, ["local", "remote"])
            self.assertEqual(g.overtaken, 0)
        finally:
            GW.AGE_AFTER = old

    async def test_cancel_while_queued_loses_no_slot(self):
        s = GW.PriorityGate(1)
        await s.enter(0)
        t = asyncio.create_task(s.enter(1))
        await asyncio.sleep(0)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        s.leave()
        # The slot has to be grantable again.
        await asyncio.wait_for(s.enter(1), timeout=1)


# ------------------------------------------- Zonen, Zugang, Positivliste ---
class GatewayOnTheWire(unittest.IsolatedAsyncioTestCase):
    """Shared setup: the real gateway, a fake llama-server."""
    TUNNEL = True          # True = alles gilt als 'extern'

    async def asyncSetUp(self):
        self.seen = []
        self.hang = None
        # Per-path canned answers. The default {"ok": true} is fine for the
        # request path, but an endpoint the gateway RESHAPES — /v1/models —
        # needs something with the real shape, or the test measures the fake
        # instead of the gateway.
        self.llama_json = {}
        async def llama(request):
            self.seen.append((request.method, request.path_qs, await request.read()))
            if self.hang is not None:
                await self.hang.wait()
            if request.path in self.llama_json:
                return web.json_response(self.llama_json[request.path])
            if request.path == "/slots":
                return web.json_response([])
            return web.json_response({"ok": True, "path": request.path})
        lapp = web.Application()
        lapp.router.add_route("*", "/{tail:.*}", llama)
        self.lserver = TestServer(lapp)
        await self.lserver.start_server()

        port = await common.free_port()
        self.backup = {k: getattr(GW, k) for k in
                          ("LLAMA", "TUNNEL_PORT", "TOKENS", "PREFIXES",
                           "SAVED", "IN_FLIGHT_PER_TOKEN", "GATE",
                           "PER_TOKEN_MAX", "AUTO_SAVE")}
        GW.LLAMA = str(self.lserver.make_url("")).rstrip("/")
        GW.TUNNEL_PORT = port if self.TUNNEL else None
        GW.TOKENS = {"geheim": "tester"}
        GW.PREFIXES, GW.SAVED, GW.IN_FLIGHT_PER_TOKEN = {}, {}, {}
        GW.GATE = GW.PriorityGate(2)
        GW.PER_TOKEN_MAX = 2
        GW.AUTO_SAVE = False

        # Record the log instead of printing it — otherwise the test output
        # drowns in operational messages.
        self.log_lines = []
        self.log_patch = mock.patch.object(
            GW, "log", lambda *a: self.log_lines.append(" ".join(map(str, a))))
        self.log_patch.start()

        # Same runner regime as production (handler_cancellation) — the
        # client-abort contract below only exists under it.
        self.server = TestServer(GW.build_app(), port=port,
                                 **GW.RUNNER_KWARGS)
        await self.server.start_server()
        self.url = "http://127.0.0.1:%d" % port
        self.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        self.log_patch.stop()
        await self.session.close()
        await self.server.close()
        await self.lserver.close()
        for k, v in self.backup.items():
            setattr(GW, k, v)

    def payload(self, **kw):
        return {"model": "laguna", "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}], **kw}

    async def fetch(self, path, token="geheim", method="POST", headers=None, payload=None):
        k = dict(headers or {})
        if token:
            k["Authorization"] = "Bearer " + token
        return await self.session.request(
            method, self.url + path, headers=k,
            json=(self.payload() if payload is None and method == "POST" else payload))


class TestARestartChangesTheServedModel(GatewayOnTheWire):
    """The behaviour, not the spelling.

    A `systemctl restart llama-user@gemma26` that loads faster than the 15 s
    poll is never seen as "gone": watch_server's SECOND branch fires, clears
    the prefixes, and used to leave SERVED saying qwen38 — so the gateway went
    on advertising qwen38's names and injecting its thinking mode into a model
    whose template ignores the field entirely.
    """
    TUNNEL = False

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.saved = {k: getattr(GW, k) for k in
                      ("SERVED", "MODES", "PROFILE_DIR", "UNKNOWN_MODELS")}
        self.envdir = tempfile.mkdtemp(prefix="prof-")
        with open(os.path.join(self.envdir, "gemma26.env"), "w") as f:
            f.write("MODES=low:on\nTEMPLATE_LEVELS=no-levels\n")
        GW.PROFILE_DIR = self.envdir
        GW.SERVED = "qwen38"
        GW.MODES = {"low": "on+low"}
        GW.UNKNOWN_MODELS = set()
        self.addCleanup(shutil.rmtree, self.envdir, ignore_errors=True)
        self.addCleanup(lambda: [setattr(GW, k, v) for k, v in self.saved.items()])

    async def test_the_refresh_picks_up_the_new_model_and_its_modes(self):
        self.llama_json["/v1/models"] = {"data": [{"id": "gemma26"}]}
        await GW.note_server_restart()
        self.assertEqual(GW.SERVED, "gemma26")
        self.assertEqual(dict(GW.MODES), {"low": "on"},
                         "the new model's own profile must be read")

    async def test_a_server_that_cannot_be_asked_leaves_the_old_answer(self):
        """Better a name that is probably still right than none at all — the
        same trade query_slots makes one function above."""
        self.llama_json["/v1/models"] = {"ok": True}
        await GW.note_server_restart()
        self.assertEqual(GW.SERVED, "qwen38")

    async def test_the_unknown_name_notes_are_forgotten_with_the_model(self):
        """They were about the old model's names. Keeping them would silence
        the note for a name that is genuinely wrong under the new one."""
        GW.UNKNOWN_MODELS.add("qwen38-medium")
        self.llama_json["/v1/models"] = {"data": [{"id": "gemma26"}]}
        await GW.note_server_restart()
        self.assertEqual(GW.UNKNOWN_MODELS, set())


class TestAnUnmatchedNameIsSaidOnce(GatewayOnTheWire):
    """The old code was wrong loudly; the new code is right quietly.

    After any switch-model.sh the client's ANTHROPIC_MODEL still names the old
    model's mode — that file is not touched by the switch — and the only other
    signal a user gets is that thinking stopped happening.
    """
    TUNNEL = False

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.saved = {k: getattr(GW, k) for k in ("SERVED", "MODES", "UNKNOWN_MODELS")}
        GW.SERVED, GW.MODES, GW.UNKNOWN_MODELS = "qwen38", {"low": "on+low"}, set()
        self.addCleanup(lambda: [setattr(GW, k, v) for k, v in self.saved.items()])

    async def test_it_is_logged_and_only_once(self):
        for _ in range(3):
            GW.inject_model_kwargs({"model": "qwen38-think"}, served="qwen38")
        hits = [m for m in self.log_lines if "matches no mode" in m]
        self.assertEqual(len(hits), 1, self.log_lines)
        self.assertIn("qwen38-low", hits[0], "say what IS offered")

    async def test_a_name_that_matches_says_nothing(self):
        GW.inject_model_kwargs({"model": "qwen38-low"}, served="qwen38")
        self.assertFalse([m for m in self.log_lines if "matches no mode" in m])


class TestReasoningOnTheWire(GatewayOnTheWire):
    """A tested function nobody calls is the failure this repo keeps naming.

    TestKwargsBelongToTheServedModel pins the arithmetic. These pin that the
    request path runs it — what llama-server receives, not what a helper
    returns.
    """
    TUNNEL = False

    TABLE = {"qwen38": {"enable_thinking": False},
             "qwen38-think": {"reasoning_effort": "medium"}}

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.saved = {k: getattr(GW, k) for k in ("KWARGS_BY_MODEL", "SERVED")}
        GW.KWARGS_BY_MODEL = dict(self.TABLE)
        self.addCleanup(lambda: [setattr(GW, k, v) for k, v in self.saved.items()])

    async def sent(self, payload):
        """The body that reached llama-server for an inference request."""
        await self.session.post(self.url + "/v1/messages", json=payload)
        bodies = [json.loads(b) for m, path, b in self.seen
                  if path.startswith("/v1/messages") and b]
        self.assertTrue(bodies, "nothing reached llama-server")
        return bodies[-1]

    async def test_the_served_model_gets_its_mode(self):
        GW.SERVED = "qwen38"
        body = await self.sent({"model": "qwen38-think", "messages": []})
        self.assertEqual(body.get("chat_template_kwargs"),
                         {"reasoning_effort": "medium"})

    async def test_a_stale_client_name_reaches_another_model_clean(self):
        """switch-model.sh ran, ANTHROPIC_MODEL did not follow."""
        GW.SERVED = "flashnext"
        body = await self.sent({"model": "qwen38-think", "messages": []})
        self.assertNotIn("chat_template_kwargs", body,
                         "qwen38's thinking mode was injected into a request "
                         "bound for another model")

    async def test_two_modes_are_two_prefixes(self):
        """The id has to see the mode the name selects.

        It is computed from the body BEFORE injection — deliberately, so that
        the store is keyed by what arrives. But the mode arrives too, as the
        model name, and it changes the rendered prompt at character 19. So the
        name must be resolved to its kwargs before the id is taken, or two
        different prompts land on one key and the restore poisons the slot.
        """
        GW.SERVED = "qwen38"
        GW.PREFIXES.clear()
        await self.sent({"model": "qwen38", "messages": [],
                         "system": "same system", "tools": []})
        await self.sent({"model": "qwen38-think", "messages": [],
                         "system": "same system", "tools": []})
        self.assertEqual(len(GW.PREFIXES), 2,
                         "off and think shared one prefix id")

    async def test_the_native_reasoning_fields_do_not_reach_the_server(self):
        """They are dropped, not translated — see DROP for the four reasons a
        translator was tried on 28.08. and taken out again. Pinned because the
        next person to measure that llama-server answers 200 to these will be
        tempted to pass them through, and passing them through is not the same
        as making them work."""
        GW.SERVED = "qwen38"
        body = await self.sent({"model": "qwen38", "messages": [],
                                "output_config": {"effort": "medium"},
                                "thinking": {"type": "adaptive"}})
        self.assertNotIn("output_config", body)
        self.assertNotIn("thinking", body)
        self.assertNotIn("reasoning_effort", body.get("chat_template_kwargs", {}))


class TestModesComeFromTheProfile(GatewayOnTheWire):
    """The names the gateway answers for are DERIVED from the served alias.

    That is the whole of variant 1: `qwen38-think` cannot exist while flashnext
    serves, because the names are built from what serves. No guard is needed
    for a stale name, and the listing cannot advertise what the injection would
    refuse — both read the same two inputs.
    """
    TUNNEL = False

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.saved = {k: getattr(GW, k) for k in
                      ("KWARGS_BY_MODEL", "SERVED", "PROFILE_DIR", "MODES")}
        self.envdir = tempfile.mkdtemp(prefix="prof-")
        with open(os.path.join(self.envdir, "qwen38.env"), "w") as f:
            # A complete map, because check_modes() holds the two lines
            # against each other: a mode may only name a level the template
            # renders. The first version of this fixture declared `deep:medium`
            # under a map without `medium` and the check caught it — which is
            # what it is for.
            f.write("MODES=none:off  low:on+low  medium:on+medium\n"
                    "TEMPLATE_LEVELS=low medium xhigh\n")
        GW.PROFILE_DIR = self.envdir
        GW.KWARGS_BY_MODEL = {}
        self.addCleanup(shutil.rmtree, self.envdir, ignore_errors=True)
        self.addCleanup(lambda: [setattr(GW, k, v) for k, v in self.saved.items()])

    def serve(self, alias):
        GW.SERVED = alias
        GW.MODES = GW.load_profile_modes(alias)

    async def sent(self, payload):
        await self.session.post(self.url + "/v1/messages", json=payload)
        bodies = [json.loads(b) for m, path, b in self.seen
                  if path.startswith("/v1/messages") and b]
        self.assertTrue(bodies, "nothing reached llama-server")
        return bodies[-1]

    def test_the_profile_is_read_for_the_served_alias(self):
        self.serve("qwen38")
        self.assertEqual(list(GW.MODES), ["none", "low", "medium"])

    def test_a_model_without_a_profile_has_no_modes(self):
        """laguna.env does not exist in this fixture. Absent is a normal
        state, not an error — four of seven real profiles read no levels."""
        self.serve("laguna")
        self.assertEqual(GW.MODES, {})

    async def test_a_declared_mode_reaches_the_server(self):
        self.serve("qwen38")
        body = await self.sent({"model": "qwen38-medium", "messages": []})
        self.assertEqual(body.get("chat_template_kwargs"),
                         {"enable_thinking": True, "reasoning_effort": "medium"})

    async def test_the_bare_alias_asks_for_nothing(self):
        """It means "whatever the profile's command line says". An empty map
        must not be written into the body, or it buys a second prefix id for
        the same rendering."""
        self.serve("qwen38")
        body = await self.sent({"model": "qwen38", "messages": []})
        self.assertNotIn("chat_template_kwargs", body)

    async def test_a_name_from_another_model_reaches_it_clean(self):
        """switch-model.sh ran, ANTHROPIC_MODEL did not follow. Under the old
        blob this injected qwen38's mode into another model."""
        self.serve("flashnext")
        body = await self.sent({"model": "qwen38-low", "messages": []})
        self.assertNotIn("chat_template_kwargs", body)

    async def test_the_listing_offers_exactly_the_names_that_work(self):
        """The asymmetry that made dsh worse for one afternoon: injection was
        scoped and the listing was not, so the picker offered names the
        gateway then refused. Derived names cannot drift apart like that."""
        self.serve("qwen38")
        self.llama_json["/v1/models"] = {
            "models": [{"name": "qwen38", "model": "qwen38",
                        "capabilities": ["completion"],
                        "details": {"n_ctx": 204800}}]}
        r = await self.session.get(self.url + "/v1/models")
        listing = await r.json()
        offered = [e.get("name") or e.get("id")
                   for e in listing.get("models", listing.get("data", []))]
        self.assertEqual(offered, ["qwen38", "qwen38-none",
                                   "qwen38-low", "qwen38-medium"])

    async def test_the_old_blob_still_works_where_no_profile_declares_modes(self):
        """Migration: a profile that has not been given MODES yet must keep
        behaving exactly as before, or the switch becomes a flag day."""
        GW.KWARGS_BY_MODEL = {"laguna": {"enable_thinking": False},
                              "laguna-think": {"reasoning_effort": "low"}}
        self.serve("laguna")
        body = await self.sent({"model": "laguna-think", "messages": []})
        self.assertEqual(body.get("chat_template_kwargs"),
                         {"reasoning_effort": "low"})

    def test_main_loads_them_at_startup(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(encoding="utf-8")
        body = src[src.index("def main("):]
        self.assertIn("load_profile_modes", body,
                      "main() never reads the served profile's modes")


class TestRestoreGuard(unittest.IsolatedAsyncioTestCase):
    """A restore may only touch a FULLY idle server. The 25.08. incident:
    restoring into an idle slot while the other slot was computing left the
    server producing degenerate output until a fresh start. A skipped
    restore costs one cold prefill; a poisoned KV state ruins everything
    after it."""

    async def _run(self, slots):
        self.seen = []
        async def llama(request):
            self.seen.append(request.path_qs)
            if request.path == "/slots":
                return web.json_response(slots)
            return web.json_response({"n_restored": 5})
        lapp = web.Application()
        lapp.router.add_route("*", "/{tail:.*}", llama)
        server = TestServer(lapp)
        await server.start_server()
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "n1.bin"), "wb").close()
            with mock.patch.object(GW, "LLAMA",
                                   str(server.make_url("")).rstrip("/")), \
                 mock.patch.object(GW, "SLOT_PATH", d), \
                 mock.patch.object(GW, "SAVED", {"id1": "n1"}), \
                 mock.patch.object(GW, "log", lambda *a: None):
                try:
                    return await GW.restore_from_disk("id1")
                finally:
                    await server.close()

    async def test_no_restore_while_any_slot_computes(self):
        ok = await self._run([
            {"id": 0, "is_processing": True, "n_prompt_tokens": 9000},
            {"id": 1, "is_processing": False, "n_prompt_tokens": 0}])
        self.assertFalse(ok)
        self.assertFalse([p for p in self.seen if "action=restore" in p],
                         "restore was sent although a slot was computing")

    async def test_restore_proceeds_on_an_idle_server(self):
        ok = await self._run([
            {"id": 0, "is_processing": False, "n_prompt_tokens": 0},
            {"id": 1, "is_processing": False, "n_prompt_tokens": 7000}])
        self.assertTrue(ok)
        self.assertTrue([p for p in self.seen if "action=restore" in p])


class TestModelListing(unittest.TestCase):
    """A consumer that builds its model picker from /v1/models (dsh does)
    must see the names the GATEWAY serves, not only the server's own alias.
    Reported from a second machine on 25.08.: the listing advertised
    `qwen38` alone, so the thinking variants were invisible."""

    TABLE = {"qwen38": {"enable_thinking": False},
             "qwen38-think": {"enable_thinking": True,
                              "reasoning_effort": "low"},
             "qwen38-deep": {"enable_thinking": True,
                             "reasoning_effort": "medium"}}

    def _listing(self):
        return {"models": [{"name": "qwen38", "model": "qwen38",
                            "capabilities": ["completion", "multimodal"],
                            "details": {"n_ctx": 204800}}]}

    def test_an_entry_we_did_not_derive_from_survives(self):
        """REPLACE was the docstring's claim and it is only true of the model
        being served. llama-server sends exactly one entry today, so replace
        and extend are indistinguishable here — which is why the first test of
        this used a single-entry listing and could not tell them apart. A draft
        model or a projector would have been erased, and erasing what you did
        not understand is not replacing what you did."""
        listing = self._listing()
        listing["models"].append({"name": "draft-0.5b", "model": "draft-0.5b"})
        out = GW.add_derived_names(listing, ["qwen38", "qwen38-low"])
        names = [e["name"] for e in out["models"]]
        self.assertEqual(names, ["qwen38", "qwen38-low", "draft-0.5b"])

    def test_the_copied_entries_keep_what_the_server_said_about_itself(self):
        """Measured 28.08.: llama-server reports `capabilities` and
        `details.format` per entry and NOT n_ctx, which the docstring used to
        claim. Whatever it does report has to stay true for every derived
        name, because it IS the same loaded model."""
        out = GW.add_derived_names(self._listing(), ["qwen38", "qwen38-low"])
        self.assertEqual(out["models"][1]["capabilities"],
                         out["models"][0]["capabilities"])
        self.assertEqual(out["models"][1]["details"],
                         out["models"][0]["details"])

    def test_a_name_the_gateway_will_not_honour_is_not_advertised(self):
        """The listing and the injection have to agree, and until 28.08. they
        did — both were unscoped, so a stale name did the WRONG thing.
        Scoping only the injection made it worse, not better: the picker went
        on offering `qwen38-think` while another model served, the gateway
        refused to honour it, and dsh saw four models where one exists, three
        of them lies inheriting the served model's n_ctx and capabilities.

        One rule, both directions: a name the map will not honour must not be
        offered."""
        out = GW.add_aliases(self._listing(), self.TABLE, served="flashnext")
        names = [e["name"] for e in out["models"]]
        self.assertEqual(names, ["qwen38"],
                         "the picker offers names the gateway now refuses")

    def test_the_served_model_still_gets_its_variants(self):
        out = GW.add_aliases(self._listing(), self.TABLE, served="qwen38")
        self.assertEqual(len(out["models"]), 3)

    def test_an_unknown_served_model_keeps_the_old_listing(self):
        """Same trade as the injection: not knowing must not hide the modes."""
        out = GW.add_aliases(self._listing(), self.TABLE, served=None)
        self.assertEqual(len(out["models"]), 3)

    def test_the_endpoint_passes_the_served_model_in(self):
        """A rule the caller does not apply is a rule in a docstring."""
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(encoding="utf-8")
        self.assertIn("add_aliases(listing, KWARGS_BY_MODEL, SERVED)", src)

    def test_every_configured_name_appears_once(self):
        out = GW.add_aliases(self._listing(), self.TABLE)
        names = [e["name"] for e in out["models"]]
        self.assertEqual(sorted(names),
                         sorted(["qwen38", "qwen38-think", "qwen38-deep"]))
        self.assertEqual(len(names), len(set(names)), "no duplicates")

    def test_aliases_inherit_what_the_server_reports(self):
        """They ARE the same loaded model — capabilities and context size
        must not be invented differently for them."""
        out = GW.add_aliases(self._listing(), self.TABLE)
        for e in out["models"]:
            self.assertEqual(e["capabilities"], ["completion", "multimodal"])
            self.assertEqual(e["details"]["n_ctx"], 204800)

    def test_the_description_names_the_thinking_level(self):
        out = GW.add_aliases(self._listing(), self.TABLE)
        by = {e["name"]: e.get("description", "") for e in out["models"]}
        self.assertIn("low", by["qwen38-think"])
        self.assertIn("medium", by["qwen38-deep"])

    def test_an_openai_shaped_listing_works_too(self):
        listing = {"data": [{"id": "qwen38", "object": "model"}]}
        out = GW.add_aliases(listing, self.TABLE)
        self.assertEqual(sorted(e["id"] for e in out["data"]),
                         sorted(["qwen38", "qwen38-think", "qwen38-deep"]))

    def test_both_arrays_are_extended_when_both_are_present(self):
        """llama-server answers with `models` AND `data` in one body.
        Extending only the first left an OpenAI client — the dialect this
        was written for — seeing a single model. Reported 26.08."""
        listing = {"models": [{"name": "qwen38", "model": "qwen38"}],
                   "object": "list",
                   "data": [{"id": "qwen38", "object": "model"}]}
        out = GW.add_aliases(listing, self.TABLE)
        self.assertEqual(sorted(e["name"] for e in out["models"]),
                         sorted(self.TABLE))
        self.assertEqual(sorted(e["id"] for e in out["data"]),
                         sorted(self.TABLE))

    def test_an_empty_or_odd_listing_is_passed_through_untouched(self):
        for listing in ({"models": []}, {}, {"models": "nonsense"}):
            self.assertEqual(GW.add_aliases(dict(listing), self.TABLE),
                             dict(listing))


class TestClientAbort(GatewayOnTheWire):
    """A caller that vanishes mid-request must not keep occupying the
    gateway. Observed 25.08. in production, before RUNNER_KWARGS: a client
    timeout left the gate slot and the per-token counter taken for the full
    ~10-minute upstream generation — the consumer got nothing but 429s from
    a machine that was working for nobody."""
    TUNNEL = False

    async def test_a_dead_client_frees_the_gate_at_once(self):
        self.hang = asyncio.Event()          # the fake upstream hangs
        body = {"model": "x", "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}]}
        t = asyncio.create_task(
            self.session.post(self.url + "/v1/messages", json=body))
        entered = await common.wait_until(lambda: GW.GATE.free < 2)
        self.assertTrue(entered, "the request never reached the gate")
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        freed = await common.wait_until(lambda: GW.GATE.free == 2, limit=3.0)
        self.assertTrue(freed, "gate slot still taken after the client left")
        self.assertEqual(GW.IN_FLIGHT_PER_TOKEN.get("local", 0), 0)
        self.hang.set()


class TestSaveSurvivesTheClientLeaving(GatewayOnTheWire):
    """The saving of a cold prefix must not depend on how the CLIENT behaves.

    Found 26.08. by running tests/live_prefix.sh: it reported "cold request
    answered (72 s)" and then "not saved". The gateway log showed START and no
    DONE. Reproduced on the live stack with two identical requests that
    differed only in the client:

        curl (closes at once)          START, no DONE, nothing saved
        connection held open 5 s       START, DONE took=0.8s

    RUNNER_KWARGS carries handler_cancellation=True, which exists so a caller
    who vanishes frees the gate slot at once (TestClientAbort). With
    aiohttp 3.13 the cancellation also lands when the client closes NORMALLY,
    right after the answer — and it landed before the two lines that schedule
    the save and record the use.

    What that cost: the prefix store is the mechanism that turns a 100-180 s
    cold start into 1.4 s, and it only filled for clients that happened to
    keep the connection open. The accounting behind the LRU cleanup was
    skipped the same way, so the store also evicted by wrong information.
    """
    TUNNEL = False

    async def asyncSetUp(self):
        await super().asyncSetUp()
        GW.AUTO_SAVE = True
        self.saved = []
        async def fake_save(id_, body, dialect=None):
            self.saved.append(id_)
        self.save_patch = mock.patch.object(GW, "auto_save", fake_save)
        self.save_patch.start()
        self.min_backup, GW.AUTO_MIN_CHARS = GW.AUTO_MIN_CHARS, 0

    async def asyncTearDown(self):
        self.save_patch.stop()
        GW.AUTO_MIN_CHARS = self.min_backup
        await super().asyncTearDown()

    def cancel_after_the_answer(self):
        """Reproduce the production timing deterministically.

        Racing a real disconnect against a fake upstream that answers in
        microseconds is not reproducible — the handler simply finishes first.
        What production does is unambiguous though, and this models exactly
        that: the answer reaches the client in full (write_eof completed) and
        THEN the handler is cancelled. Everything after the forward() call is
        therefore skipped.
        """
        real = GW.forward
        async def forward_then_cancelled(*a, **kw):
            await real(*a, **kw)
            raise asyncio.CancelledError()
        return mock.patch.object(GW, "forward", forward_then_cancelled)

    async def test_a_cold_prefix_is_saved_even_if_the_client_is_gone(self):
        with self.cancel_after_the_answer():
            try:
                await self.fetch("/v1/messages", token=None)
            except Exception:
                pass                     # the client sees a broken connection
        ok = await common.wait_until(lambda: bool(self.saved), limit=3.0)
        self.assertTrue(ok,
                        "the cold prefix of a client that closed at once was "
                        "never scheduled for saving — every such consumer "
                        "pays a full cold start on its next request")

    async def test_the_completion_is_logged_even_then(self):
        """DONE is what the operator reads. A log that shows START without an
        end for every short-lived client cannot be used to find anything."""
        with self.cancel_after_the_answer():
            try:
                await self.fetch("/v1/messages", token=None)
            except Exception:
                pass
        ok = await common.wait_until(
            lambda: any(l.startswith("DONE") for l in self.log_lines), limit=3.0)
        self.assertTrue(ok, "no DONE line: %r" % self.log_lines)

    async def test_the_use_is_recorded_even_then(self):
        """PREFIXES[...]['last'] feeds the LRU that prefix-cleanup evicts by.
        Skipping it makes the store throw away the wrong prefixes."""
        with self.cancel_after_the_answer():
            try:
                await self.fetch("/v1/messages", token=None)
            except Exception:
                pass
        ok = await common.wait_until(
            lambda: any(e.get("last") for e in GW.PREFIXES.values()), limit=3.0)
        self.assertTrue(ok, "no prefix recorded a time of last use")

    async def test_an_abort_BEFORE_the_answer_no_longer_decides_anything(self):
        """THE CONTRACT CHANGED ON 29.08.2026, and the old one is worth
        keeping visible.

        It read: a caller who leaves during the prefill has no complete answer
        in the slot, so saving would write a PARTIAL prefix that later
        restores as a whole one. True while the save copied whatever the slot
        happened to hold, at the end of the request.

        The save now happens BEFORE the answer and does not read the slot's
        leftovers at all: prewarm renders the prefix, prefills that alone, and
        refuses to publish a file whose token count is not the prefix's. The
        file is complete by construction, so an abort no longer decides
        whether it is valid — only whether the work was worth doing, which is
        a cost question and not a correctness one.
        """
        async def cancelled_during(*a, **kw):
            raise asyncio.CancelledError()
        with mock.patch.object(GW, "forward", cancelled_during):
            try:
                await self.fetch("/v1/messages", token=None)
            except Exception:
                pass
        await asyncio.sleep(0.2)
        self.assertEqual(self.saved, [self.saved[0]] if self.saved else [],
                         "the save runs before the answer now")
        self.assertTrue(self.saved,
                        "and it is not cancelled by the caller leaving — the "
                        "write is shielded so no half file can be left behind")


class TestZoneRemote(GatewayOnTheWire):
    TUNNEL = True

    async def test_without_a_token_401(self):
        r = await self.fetch("/v1/messages", token=None)
        self.assertEqual(r.status, 401)
        self.assertEqual(self.seen, [], "llama-server darf nichts seen haben")

    async def test_wrong_token_401(self):
        r = await self.fetch("/v1/messages", token="erfunden")
        self.assertEqual(r.status, 401)

    async def test_valid_token_200(self):
        r = await self.fetch("/v1/messages")
        self.assertEqual(r.status, 200)
        self.assertEqual(len(self.seen), 1)

    async def test_x_api_key_is_accepted_too(self):
        r = await self.fetch("/v1/messages", token=None, headers={"x-api-key": "geheim"})
        self.assertEqual(r.status, 200)

    async def test_blocked_paths_404_despite_a_token(self):
        # /v1/chat/completions used to stand here. It is an allowed dialect
        # of the same inference since 25.08. (test below) — /completion is
        # NOT: it takes a raw prompt, bypasses the chat template and was the
        # free-inference hole in docs/SECURITY.md.
        for path, method in (("/slots", "GET"), ("/completion", "POST"),
                              ("/props", "GET"), ("/health", "GET")):
            with self.subTest(path=path):
                r = await self.fetch(path, method=method, payload={"prompt": "hi"})
                self.assertEqual(r.status, 404)
        self.assertEqual(self.seen, [],
                         "no blocked path may reach llama-server")

    async def test_openai_dialect_is_allowed_but_still_needs_a_token(self):
        """OpenAI-speaking agents (dsh) reach the model through the tunnel —
        under exactly the same zone rules as Claude Code."""
        r = await self.fetch("/v1/chat/completions", token="erfunden")
        self.assertEqual(r.status, 401)
        self.assertEqual(self.seen, [], "no token, no forwarding")
        r = await self.fetch("/v1/chat/completions")
        self.assertEqual(r.status, 200)
        self.assertEqual(len(self.seen), 1)

    async def test_allowed_paths(self):
        r = await self.fetch("/v1/models", method="GET")
        self.assertEqual(r.status, 200)
        r = await self.fetch("/v1/messages/count_tokens")
        self.assertEqual(r.status, 200)

    async def test_allow_list_cannot_be_bypassed(self):
        for path in ("/v1/messages/../slots", "/V1/MESSAGES", "/slots?x=1",
                     "//v1/messages", "/v1/messages/x"):
            with self.subTest(path=path):
                r = await self.fetch(path, method="GET")
                self.assertEqual(r.status, 404)

    async def test_status_is_blocked_from_remote(self):
        # The tunnel comes from 127.0.0.1 when cloudflared runs natively.
        # Looking only at the IP here hands out prefix and consumer names.
        r = await self.session.get(self.url + "/gateway/status",
                                   headers={"Authorization": "Bearer geheim"})
        self.assertEqual(r.status, 403)

    async def test_throttle_per_access(self):
        GW.PER_TOKEN_MAX = 1
        self.hang = asyncio.Event()
        erste = asyncio.create_task(self.fetch("/v1/messages"))
        while not self.seen:
            await asyncio.sleep(0.01)
        zweite = await self.fetch("/v1/messages")
        self.assertEqual(zweite.status, 429)
        self.hang.set()
        self.assertEqual((await erste).status, 200)
        # and free again afterwards
        await common.wait_until(lambda: GW.IN_FLIGHT_PER_TOKEN.get("tester", 0) == 0)
        self.assertEqual(GW.IN_FLIGHT_PER_TOKEN.get("tester", 0), 0)

    async def test_a_queued_streaming_caller_gets_a_sign_of_life(self):
        """Between GATE.enter() and forward() nothing used to be written. A
        queued remote caller therefore saw only silence, and Cloudflare drops a
        connection after 125 s of it — measured: with four local streams the
        caller was still waiting after 200 s."""
        old_ka, GW.KEEPALIVE = GW.KEEPALIVE, 0.05
        GW.GATE = GW.PriorityGate(1)
        try:
            self.hang = asyncio.Event()
            busy = asyncio.create_task(self.fetch("/v1/messages"))
            while not self.seen:
                await asyncio.sleep(0.01)
            r = await self.session.post(
                self.url + "/v1/messages",
                headers={"Authorization": "Bearer geheim"},
                json=self.payload(stream=True))
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers.get("content-type"), "text/event-stream")
            chunk = await asyncio.wait_for(r.content.read(3), timeout=5)
            self.assertEqual(chunk, b":\n\n", "no sign of life while queued")
            r.close()
            self.hang.set()
            await busy
        finally:
            GW.KEEPALIVE = old_ka

    async def test_a_queued_non_streaming_caller_keeps_its_status_code(self):
        """Only a stream can be kept alive — committing to a status code for
        everyone would throw away the upstream's answer."""
        old_ka, GW.KEEPALIVE = GW.KEEPALIVE, 0.05
        GW.GATE = GW.PriorityGate(1)
        try:
            self.hang = asyncio.Event()
            busy = asyncio.create_task(self.fetch("/v1/messages"))
            while not self.seen:
                await asyncio.sleep(0.01)
            queued = asyncio.create_task(self.fetch("/v1/messages"))
            await asyncio.sleep(0.2)             # long enough for two keep-alives
            self.assertFalse(queued.done(), "it should still be waiting")
            self.hang.set()
            r = await queued
            self.assertEqual(r.status, 200)
            self.assertNotEqual(r.headers.get("content-type"), "text/event-stream")
            await busy
        finally:
            GW.KEEPALIVE = old_ka

    async def test_counter_returns_when_waiting_is_cancelled(self):
        """The caller aborts while standing in the queue.

        The counter used to stay put: after PER_TOKEN_MAX such cases the
        access got nothing but 429 until the service was restarted.
        """
        async def boom(prio):
            raise asyncio.CancelledError()
        GW.GATE.enter = boom
        try:
            await self.fetch("/v1/messages")
        except Exception:
            pass
        await common.wait_until(lambda: GW.IN_FLIGHT_PER_TOKEN.get("tester", 0) == 0)
        self.assertEqual(GW.IN_FLIGHT_PER_TOKEN.get("tester", 0), 0)

    async def test_gate_slot_returns_when_the_reload_is_cancelled(self):
        """Cancelled during restore_from_disk — that catches only 'except
        Exception', and CancelledError is not one."""
        ident = GW.prefix_id(self.payload())[0]
        GW.SAVED = {ident: "irgendwas"}
        async def boom(k):
            raise asyncio.CancelledError()
        GW.restore_from_disk = boom
        free_before = GW.GATE.free
        try:
            await self.fetch("/v1/messages")
        except Exception:
            pass
        await common.wait_until(lambda: GW.GATE.free == free_before)
        self.assertEqual(GW.GATE.free, free_before,
                         "MAX_INFLIGHT has dropped permanently")
        self.assertEqual(GW.IN_FLIGHT_PER_TOKEN.get("tester", 0), 0)


class TestSaveThreshold(GatewayOnTheWire):
    """Not every cold prefix is worth 628 MB."""
    TUNNEL = False

    async def _watch_saving(self, payload, path="/v1/messages"):
        GW.AUTO_SAVE = True
        # Through the REAL path: since 29.08. the save happens INSIDE the
        # request, before it is forwarded, so there is no timing to arrange —
        # by the time the answer is here, the decision has been made.
        saved_ids = []
        async def fake(ident, body, dialect=GW.DIA.ANTHROPIC):
            saved_ids.append((ident, dialect))
        GW.auto_save = fake
        r = await self.fetch(path, token=None, payload=payload)
        self.assertEqual(r.status, 200)
        for _ in range(50):                # the save runs as a task
            if saved_ids:
                break
            await asyncio.sleep(0.02)
        return saved_ids

    async def test_minimal_body_is_not_saved(self):
        # Exactly the body from setup/smoketest.sh. In production it left a
        # file with seven tokens in the store.
        self.assertEqual(await self._watch_saving(self.payload()), [])

    async def test_real_prefix_is_saved(self):
        p = SYN.body()
        p["model"], p["max_tokens"] = "laguna", 1
        saved = await self._watch_saving(p)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][1], GW.DIA.ANTHROPIC,
                         "the dialect has to reach prewarm — it decides how "
                         "the prefix is rendered")

    async def test_an_openai_prefix_is_saved_under_its_own_dialect(self):
        """A dsh-shaped body must be saved too, and prewarm has to be told
        which shape it has: rendered as Anthropic it would lose the system
        prompt and the tools, and the saved state would fit no request."""
        p = {"model": "qwen38", "max_tokens": 1,
             "messages": [{"role": "system",
                           "content": SYN.system_text("/tmp/p", "/tmp/m")},
                          {"role": "user", "content": "hi"}],
             "tools": [{"type": "function",
                        "function": {"name": "T%02d" % i,
                                     "description": "d" * 200,
                                     "parameters": {"type": "object"}}}
                       for i in range(8)]}
        saved = await self._watch_saving(p, path="/v1/chat/completions")
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][1], GW.DIA.OPENAI)


class TestZoneLocal(GatewayOnTheWire):
    TUNNEL = False

    async def test_local_needs_no_token(self):
        r = await self.fetch("/v1/messages", token=None)
        self.assertEqual(r.status, 200)

    async def test_local_may_do_everything(self):
        r = await self.fetch("/slots", token=None, method="GET")
        self.assertEqual(r.status, 200)

    async def test_status_is_readable_locally(self):
        r = await self.session.get(self.url + "/gateway/status")
        self.assertEqual(r.status, 200)
        self.assertIn("prefixes", await r.json())


# ---------------------------------------------------------------- Token ---
class TestSseError(unittest.TestCase):
    """Once the keep-alives have gone out the HTTP status is spent — a later
    failure can only be delivered inside the stream."""

    def test_shape(self):
        raw = GW.sse_error(400, "context window exceeded").decode()
        self.assertTrue(raw.startswith("event: error\ndata: "))
        self.assertTrue(raw.endswith("\n\n"))
        payload = json.loads(raw.split("data: ", 1)[1])
        self.assertEqual(payload["type"], "error")
        self.assertIn("400", payload["error"]["message"])
        self.assertIn("context window exceeded", payload["error"]["message"])


class TestTokenFile(unittest.TestCase):
    def read_tokens(self, content):
        with tempfile.NamedTemporaryFile("w", suffix=".tokens", delete=False) as f:
            f.write(content)
            path = f.name
        old, GW.TOKEN_FILE = GW.TOKEN_FILE, path
        old_t, GW.TOKEN = GW.TOKEN, ""
        try:
            return GW.load_tokens()
        finally:
            GW.TOKEN_FILE, GW.TOKEN = old, old_t
            os.unlink(path)

    def test_name_and_secret(self):
        self.assertEqual(self.read_tokens("a eins\nb zwei\n"), {"eins": "a", "zwei": "b"})

    def test_comments_and_blank_lines(self):
        self.assertEqual(self.read_tokens("# nichts\n\n  a  eins  \n"), {"eins": "a"})

    def test_secret_may_contain_spaces(self):
        # split(None, 1): everything after the name is the secret. smoketest.sh
        # used to read only the first word here and then reported false 401s.
        self.assertEqual(self.read_tokens("a eins zwei\n"), {"eins zwei": "a"})

    def test_no_file_means_no_access(self):
        old, GW.TOKEN_FILE = GW.TOKEN_FILE, "/does/not/exist"
        old_t, GW.TOKEN = GW.TOKEN, ""
        try:
            self.assertEqual(GW.load_tokens(), {})
        finally:
            GW.TOKEN_FILE, GW.TOKEN = old, old_t


if __name__ == "__main__":
    unittest.main()


class TestWarmIsAMeasurement(unittest.TestCase):
    """`warm` was a claim: "I have seen this prefix id". Whether llama.cpp
    then reused anything was never asked — and on 28.08.2026 a restore from
    disk was logged warm, cost a full 14960-token prefill, and nothing
    anywhere disagreed. The numbers were in the answer the gateway had just
    proxied. See `saved-prefix-holds-a-foreign-state`.
    """

    def test_llama_cpps_own_timings_are_preferred(self):
        got = DIA.reuse_from_text('{"timings": {"cache_n": 14957, "prompt_n": 4}}')
        self.assertEqual(got, (14957, 4))

    def test_the_anthropic_shape_is_read_from_the_head_of_a_stream(self):
        """message_start is the FIRST event, not the last, which is why the
        sniffer keeps a head as well as a tail."""
        sse = ('data: {"type":"message_start","message":{"usage":'
               '{"cache_read_input_tokens": 30, "input_tokens": 15}}}\n\n'
               'data: {"type":"ping"}\n\n')
        self.assertEqual(DIA.reuse_from_text(sse), (30, 15))

    def test_the_openai_shape_subtracts_rather_than_guessing(self):
        got = DIA.reuse_from_text(
            '{"usage": {"prompt_tokens": 100,'
            ' "prompt_tokens_details": {"cached_tokens": 90}}}')
        self.assertEqual(got, (90, 10))

    def test_rubbish_answers_none_instead_of_raising(self):
        """It is fed the two ends of a proxied stream, so half events and
        truncated JSON are NORMAL input. A gateway that dies over its own
        bookkeeping is worse than one that cannot label a request."""
        for text in ("", "data: {\"broken\": ", ":\n\n", "not json at all",
                     '{"usage": {"input_tokens": "many"}}'):
            with self.subTest(text=text[:20]):
                self.assertIsNone(DIA.reuse_from_text(text))

    def test_a_restore_that_carried_nothing_is_a_verdict(self):
        ok, why = GW.restore_verdict(("f", 14957), (0, 14960))
        self.assertIs(ok, False)
        self.assertIn("14957", why)
        self.assertIn("does not hold what its name says", why)

    def test_a_restore_that_carried_is_left_alone(self):
        ok, _ = GW.restore_verdict(("f", 14957), (14957, 4))
        self.assertIs(ok, True)

    def test_a_prefix_that_grew_a_little_is_not_condemned(self):
        """The threshold is generous on purpose: a request that added a tool
        since the save still reuses most of the state, and that is the file
        working, not failing."""
        ok, _ = GW.restore_verdict(("f", 14957), (12000, 3000))
        self.assertIs(ok, True)

    def test_without_numbers_there_is_no_verdict(self):
        """Unknown must not read as guilty — an answer whose accounting could
        not be parsed says nothing about the file."""
        self.assertIsNone(GW.restore_verdict(("f", 14957), None)[0])
        self.assertIsNone(GW.restore_verdict(None, (0, 14960))[0])
        self.assertIsNone(GW.restore_verdict(("f", 0), (0, 14960))[0])


class TestQuarantine(unittest.TestCase):
    """A wrong file is not the defect. A wrong file that costs a cold prefill
    on every future request for that prefix is."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="slots-")
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        self.saved = {k: getattr(GW, k) for k in ("SLOT_PATH", "SAVED")}
        self.addCleanup(lambda: [setattr(GW, k, v) for k, v in self.saved.items()])
        GW.SLOT_PATH = self.d
        GW.SAVED = {"abc123": "abc123"}
        with open(os.path.join(self.d, "abc123.bin"), "w") as f:
            f.write("x")
        with open(os.path.join(self.d, "abc123.json"), "w") as f:
            json.dump({"name": "abc123", "render_id": "deadbeef"}, f)

    def test_the_file_is_set_aside_with_its_reason_and_not_deleted(self):
        """1.1 GB and a cold prefill bought it; it is worth looking at before
        it goes, and a deletion cannot be argued with afterwards."""
        self.assertTrue(GW.quarantine("abc123", "abc123", "reused 0 of 14957"))
        self.assertFalse(os.path.exists(os.path.join(self.d, "abc123.bin")))
        self.assertTrue(os.path.exists(os.path.join(self.d, "abc123.bin.unusable")))
        side = json.load(open(os.path.join(self.d, "abc123.json")))
        self.assertIn("reused 0 of 14957", side["unusable"]["reason"])
        self.assertEqual(side["render_id"], "deadbeef", "the sidecar is kept")

    def test_it_leaves_the_store_so_the_next_request_does_not_pay_again(self):
        GW.quarantine("abc123", "abc123", "why")
        self.assertNotIn("abc123", GW.SAVED)

    def test_a_missing_file_does_not_take_the_gateway_down(self):
        os.remove(os.path.join(self.d, "abc123.bin"))
        self.assertTrue(GW.quarantine("abc123", "abc123", "why"))
        self.assertNotIn("abc123", GW.SAVED)


class TestTheSaveIsBracketed(unittest.TestCase):
    """The WRITE side of the same defect.

    Saving a prefix means putting it into a slot, and with -np 1 there is one.
    The save is asynchronous and takes ~102 s here, so a request admitted
    meanwhile takes that slot — and what lands on disk is its prefix under our
    name. The window cannot be inspected afterwards without reading a
    gigabyte, so it is WATCHED instead.
    """

    def test_the_source_counts_what_was_served_and_compares_it(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertIn("served_before = SERVED_COUNT", src)
        self.assertIn("n > served_before", src,
                      "the window has to be compared, however it is spelled")

    def test_the_counter_is_raised_where_the_request_reaches_the_model(self):
        """Not at admission and not at the answer: what matters is that the
        model was asked, because that is what touches the slot."""
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        start = src.index("async def handler(")
        body = src[start:src.index("\n    finally:", start)]
        self.assertIn("SERVED_COUNT += 1", body)
        # The INFERENCE forward, not the pass-through one earlier in the
        # handler: only the former puts a prompt into a slot.
        self.assertLess(body.index("SERVED_COUNT += 1"),
                        body.index("return await forward(req, body, out, early,"),
                        "counted after the forward, a save could still race it")

    def test_a_dropped_save_names_the_defect_it_avoids(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertIn("saved-prefix-holds-a-foreign-state", src)


class TestTheWindowAVerdictNeeds(unittest.TestCase):
    """A measurement is about THIS file only if nothing else could have taken
    the slot during it. The write side got that on 28.08.; the read side did
    not, and would have quarantined a good 1.1 GB file for somebody else's
    traffic."""

    def setUp(self):
        self.saved = {k: getattr(GW, k) for k in ("SERVED_TRAIL", "MAX_INFLIGHT")}
        self.addCleanup(lambda: [setattr(GW, k, v) for k, v in self.saved.items()])
        GW.SERVED_TRAIL = []
        GW.MAX_INFLIGHT = 2

    def test_alone_with_the_slot_is_a_clean_window(self):
        GW.SERVED_TRAIL = [(5, "me")]
        self.assertTrue(GW._window_was_clean((4, GW.MAX_INFLIGHT - 1), "me"))

    def test_another_prefix_in_the_window_makes_it_unjudgeable(self):
        GW.SERVED_TRAIL = [(5, "me"), (6, "somebody-else")]
        self.assertFalse(GW._window_was_clean((4, GW.MAX_INFLIGHT - 1), "me"))

    def test_the_same_prefix_does_not_spoil_it(self):
        """Measured as harmless in autosave-evicts-the-working-slot: saving or
        serving the prefix that is already in the slot changes nothing. A rule
        that called this dirty would throw away good files for free."""
        GW.SERVED_TRAIL = [(5, "me"), (6, "me")]
        self.assertTrue(GW._window_was_clean((4, GW.MAX_INFLIGHT - 1), "me"))

    def test_somebody_already_in_flight_makes_it_unjudgeable(self):
        """MAX_INFLIGHT is 2 whenever the slot count could not be read, and
        the gateway says so in a WARNING at startup. Then a second request can
        be in the slot while this one is measured."""
        self.assertFalse(GW._window_was_clean((4, 0), "me"))

    def test_no_window_is_not_a_clean_window(self):
        self.assertFalse(GW._window_was_clean(None, "me"))


class TestTheSidecarHashIsFinallyRead(unittest.TestCase):
    """prewarm has written `render_id` into every sidecar since the store
    existed, and until 29.08.2026 no line of code read it back — which is how
    a file holding another prefix's state kept its name and cost a full
    prefill to everything that found it."""

    def test_the_recipe_matches_prewarms(self):
        """Both cut the rendered prompt at the user marker and take
        sha256[:12] of what is in front. If they drift apart, every restore
        looks like a mismatch — which is exactly why a mismatch must not
        condemn a file."""
        gw = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        pw = (common.REPO / "tools" / "prewarm.py").read_text(encoding="utf-8")
        for piece in ('hashlib.sha256(', '[:12]', 'find("<user>")',
                      'hoist_system_messages'):
            with self.subTest(piece=piece):
                self.assertIn(piece, gw)
                self.assertIn(piece, pw)

    def test_a_mismatch_skips_the_restore_and_does_not_condemn(self):
        """The hash is DERIVED. Being wrong here costs one restore; being
        wrong the other way would delete the store."""
        gw = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        start = gw.index("async def restore_from_disk(")
        body = gw[start:gw.index("def quarantine(", start)]
        self.assertIn("genuinely cold", body,
                      "the request has to be called what it is")
        self.assertNotIn("quarantine(", body,
                         "a derived hash must not be able to delete anything")

    def setUp(self):
        self.llama = GW.LLAMA
        self.addCleanup(lambda: setattr(GW, "LLAMA", self.llama))

    def test_an_uncomputable_hash_means_carry_on(self):
        """None is 'do not know'. Treating it as a mismatch would switch the
        store off the first time /apply-template hiccups."""
        GW.LLAMA = "http://127.0.0.1:1"          # nothing listens there
        with mock.patch.object(GW, "log"):
            self.assertIsNone(GW.render_id_of({"messages": []}, GW.DIA.ANTHROPIC))


class TestTheSaveHappensBeforeTheAnswer(GatewayOnTheWire):
    """The ordering that dissolved the whole save policy.

    Measured 29.08.2026: prefilling the prefix ALONE leaves the slot holding
    exactly the prefix — the state a saved file must contain — so the save is
    a 314 ms write. Doing it after the answer means the slot holds
    prefix+question+answer, and getting back costs ~11 s of recomputation and
    cuts the session's context out of the slot.
    """
    TUNNEL = False

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.order = []
        GW.AUTO_SAVE = True
        self.min_backup, GW.AUTO_MIN_CHARS = GW.AUTO_MIN_CHARS, 0
        async def fake_save(id_, body, dialect=None):
            self.order.append("save")
        real_forward = GW.forward
        async def watched_forward(*a, **kw):
            self.order.append("forward")
            return await real_forward(*a, **kw)
        self.patches = [mock.patch.object(GW, "auto_save", fake_save),
                        mock.patch.object(GW, "forward", watched_forward)]
        for pp in self.patches:
            pp.start()

    async def asyncTearDown(self):
        for pp in self.patches:
            pp.stop()
        GW.AUTO_MIN_CHARS = self.min_backup
        await super().asyncTearDown()

    async def test_the_save_runs_first(self):
        await self.fetch("/v1/messages", token=None)
        self.assertEqual(self.order, ["save", "forward"],
                         "the prefix has to be on disk before the request "
                         "fills the slot with a session")

    async def test_a_prefix_already_on_disk_is_not_saved_again(self):
        GW.SAVED = dict(GW.SAVED)
        r = await self.fetch("/v1/messages", token=None)
        self.assertEqual(r.status, 200)
        ident = self.order and "save" in self.order
        self.assertTrue(ident)
        self.order.clear()
        # second time round the id is in the store
        GW.SAVED[GW.DIA.prefix_id(self.payload(), GW.DIA.ANTHROPIC)[0]] = "x"
        await self.fetch("/v1/messages", token=None)
        self.assertEqual(self.order, ["forward"])

    async def test_a_failing_save_never_costs_the_answer(self):
        """A save that fails must cost a cold prefill, never a reply."""
        async def boom(id_, body, dialect=None):
            raise RuntimeError("prewarm exploded")
        with mock.patch.object(GW, "auto_save", boom), mock.patch.object(GW, "log"):
            r = await self.fetch("/v1/messages", token=None)
        self.assertEqual(r.status, 200)

    async def test_a_hanging_save_is_bounded(self):
        async def hang(id_, body, dialect=None):
            await asyncio.sleep(30)
        with mock.patch.object(GW, "auto_save", hang), \
             mock.patch.object(GW, "SAVE_TIMEOUT_S", 0.05), \
             mock.patch.object(GW, "log"):
            r = await self.fetch("/v1/messages", token=None)
        self.assertEqual(r.status, 200, "a stuck save must not hang the answer")


class TestTheRestoreOnlyEarnsItsKeepOnAColdServer(unittest.IsolatedAsyncioTestCase):
    """Putting the prefix into the slot makes the slot a PERFECT prefix of the
    incoming request, and llama.cpp consults its RAM prompt cache only when
    `f_keep < 0.5` — so the restore switches that lookup off. Measured
    30.08.2026 on the production server:

        the cache holds the conversation   restore 56.4 s   without it 1.0 s
        the cache holds only the prefix    redundant — the cache returned the
                                           prefix by itself, 5 of 5 takeovers
        the cache holds nothing            the restore saves one prefill

    So the file helps in exactly one situation, and the gateway's own `cold`
    flag does not identify it: `cold` means "I have not served this since I
    started", while the cache belongs to llama-server, which restarts
    separately. On 29.08. the gateway restarted at 23:38 beside a server up
    since 09:48, every prefix looked cold, and the first restore hid a
    69,939-token state for 506 s.

    The guard is OFF by default, so these tests arm it — and one of them holds
    the default inert, because switching this on by accident trades a rare tail
    risk for a 92 s prefill on every genuine cold start.
    """

    def setUp(self):
        GW.MAX_TASK_ID = None
        self.addCleanup(setattr, GW, "MAX_TASK_ID", None)

    def slots(self, id_task, busy=False):
        return [{"id": 0, "is_processing": busy, "n_prompt_tokens": 0,
                 "id_task": id_task}]

    # --- the reading itself ------------------------------------------------
    def test_a_fresh_server_is_cold(self):
        cold, seen, why = GW.server_is_cold(self.slots(2))
        self.assertTrue(cold)
        self.assertIn("fresh", why)

    def test_a_working_server_is_not(self):
        cold, seen, why = GW.server_is_cold(self.slots(12365))
        self.assertFalse(cold)
        self.assertEqual(seen, 12365)

    def test_a_counter_that_fell_means_it_restarted(self):
        GW.server_is_cold(self.slots(12365))
        cold, seen, why = GW.server_is_cold(self.slots(7))
        self.assertTrue(cold)
        self.assertIn("restarted", why)

    def test_the_high_water_mark_follows_the_restart_down(self):
        """Otherwise every reading after a restart looks like another one."""
        GW.server_is_cold(self.slots(12365))
        GW.server_is_cold(self.slots(7))
        cold, _, _ = GW.server_is_cold(self.slots(9))
        self.assertFalse(cold, "9 > 7 is the same server still working")

    def test_a_gateway_that_restarts_beside_a_warm_server_sees_it(self):
        """The 29.08. incident in one assertion: no memory at all, and the
        server's own counter still says it has been working."""
        self.assertIsNone(GW.MAX_TASK_ID)
        cold, _, _ = GW.server_is_cold(self.slots(9999))
        self.assertFalse(cold)

    def test_no_counter_decides_as_before(self):
        """An unknown must not silently become a new policy."""
        cold, seen, why = GW.server_is_cold([{"id": 0, "is_processing": False}])
        self.assertTrue(cold)
        self.assertIsNone(seen)
        self.assertIn("as before", why)

    # --- and what it does to the restore -----------------------------------
    async def _run(self, guard, id_task):
        self.seen = []
        async def llama(request):
            self.seen.append(request.path_qs)
            if request.path == "/slots":
                return web.json_response(self.slots(id_task))
            return web.json_response({"n_restored": 5})
        lapp = web.Application()
        lapp.router.add_route("*", "/{tail:.*}", llama)
        server = TestServer(lapp)
        await server.start_server()
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "n1.bin"), "wb").close()
            with mock.patch.object(GW, "LLAMA",
                                   str(server.make_url("")).rstrip("/")), \
                 mock.patch.object(GW, "SLOT_PATH", d), \
                 mock.patch.object(GW, "SAVED", {"id1": "n1"}), \
                 mock.patch.object(GW, "RESTORE_ONLY_WHEN_SERVER_COLD", guard), \
                 mock.patch.object(GW, "log", lambda *a: None):
                try:
                    return await GW.restore_from_disk("id1")
                finally:
                    await server.close()

    async def test_a_prefix_this_server_already_holds_is_left_alone(self):
        """A WARM SERVER IS NOT ENOUGH, and the first version of this test said
        it was. Live DeepSeek-harness traffic on 30.08. at 16:20:57: the server
        had run 41 tasks of benchmark, the conversation's prefix had not been
        near it, the restore was skipped and cost 74 s. The ledger entry is
        what makes the skip legitimate."""
        GW.SEEN["id1"] = 9000
        self.addCleanup(GW.SEEN.clear)
        ok = await self._run(True, 12365)
        self.assertFalse(ok)
        self.assertFalse([p for p in self.seen if "action=restore" in p])

    async def test_a_warm_server_alone_does_not_block_the_restore(self):
        ok = await self._run(True, 12365)          # no ledger entry
        self.assertTrue(ok, "41 tasks of somebody else's traffic prove nothing")

    async def test_a_cold_one_still_gets_the_file(self):
        ok = await self._run(True, 1)
        self.assertTrue(ok)
        self.assertTrue([p for p in self.seen if "action=restore" in p])

    async def test_off_is_what_ships(self):
        ok = await self._run(False, 12365)
        self.assertTrue(ok, "the guard must be inert until it is switched on")

    def test_the_default_is_off_and_the_number_says_it_is_untuned(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertIn('"RESTORE_NUR_WENN_SERVER_KALT"', src)
        self.assertRegex(src, r'RESTORE_NUR_WENN_SERVER_KALT",\s*\n\s*"0"\) == "1"')
        self.assertIn("INSENSITIVE BY DESIGN, NOT TUNED", src)


class TestTheIdIsTakenBeforeThePromptIsRewritten(unittest.TestCase):
    """`ident` is hashed from the body as it ARRIVES; `correct()` then hoists
    the stable part of mid-conversation system messages to the FRONT, which is
    exactly where llama.cpp measures reuse from. Nothing checked that the two
    agreed.

    Measured 30.08.2026, prefix 7ff6bcd1f1de, two turns of one Claude Code
    session: volatile_moved 25 -> 26, the hoisted prefix 73404 -> 73738
    characters, ONE unchanged id, reused 0, computed 73877, 668.9 s -- and the
    gateway logged it as `warm`.

    These tests do not claim the defect is fixed. They hold the gateway to
    NOTICING it, which is a different and smaller thing.
    """

    def setUp(self):
        self.src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")

    def test_the_prefix_is_hashed_again_after_the_correction(self):
        at_correct = self.src.index("p, n_vol = correct(p, dialect)")
        at_post = self.src.index("post_id = DIA.prefix_id(")
        self.assertLess(at_correct, at_post,
                        "hashing before the correction measures the wrong thing")

    def test_it_costs_no_round_trip(self):
        """render_id_of() would be more faithful and posts to /apply-template.
        A network call per request to improve a log label is not a trade worth
        making, and the defect showed up in the prefix TEXT anyway."""
        start = self.src.index("post_id = DIA.prefix_id(")
        self.assertNotIn("render_id_of", self.src[start - 1200:start + 200])

    def test_it_fires_when_the_rendering_moved(self):
        self.assertTrue(GW.rendering_changed("aaaaaaaa", "bbbbbbbb"))

    def test_it_stays_quiet_when_it_did_not(self):
        self.assertFalse(GW.rendering_changed("aaaaaaaa", "aaaaaaaa"))

    def test_a_first_sighting_is_not_a_rewrite(self):
        """The first request for a prefix — and the first after a gateway
        restart — has nothing to compare against. Reporting those would fire
        on every cold start and mean nothing."""
        for prev, now in ((None, "bbbbbbbb"), ("", "bbbbbbbb"),
                          ("aaaaaaaa", None), (None, None)):
            with self.subTest(prev=prev, now=now):
                self.assertFalse(GW.rendering_changed(prev, now))

    def test_a_rewritten_prefix_is_not_called_warm(self):
        self.assertIn('"COLD" if cold else ("REWRITTEN" if rewritten else "warm")',
                      self.src)

    def test_cold_itself_is_left_alone(self):
        """`cold` drives the automatic save. A relabelling that also flipped it
        would answer a 669 s turn by writing a gigabyte to disk."""
        start = self.src.index("prev_post = (LAST_SHAPE.get(ident)")
        block = self.src[start:self.src.index('log("START', start)]
        self.assertNotIn("cold =", block)

    def test_the_previous_rendering_is_remembered_per_prefix(self):
        self.assertIn('LAST_SHAPE[ident] = {"shape": shape, "post": post_id,',
                      self.src)
        self.assertIn('(LAST_SHAPE.get(ident) or {}).get("post")', self.src)

    def test_both_ids_reach_the_trace(self):
        """The log line is a label; the record is what a later session can
        compare. Two records sharing `prefix` and differing in `post_id` ARE
        the defect."""
        self.assertIn('"post_id": post_id,', self.src)
        self.assertIn('"rewritten": rewritten', self.src)


class TestTheBannerAnswersWhatIsOn(unittest.TestCase):
    """The first question asked of RESTORE_ONLY_WHEN_SERVER_COLD after it was
    switched on was "is it actually on?", and nothing in the startup banner
    could answer it. A switch that does not announce itself cannot be
    verified."""

    def setUp(self):
        self.src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")

    def test_the_restore_policy_is_printed(self):
        self.assertIn('log("  restore a saved prefix: %s"', self.src)

    def test_both_states_are_printed_not_only_the_interesting_one(self):
        """A line that appears only when a setting is active leaves its
        absence meaning either `off` or `old build`."""
        start = self.src.index('log("  restore a saved prefix: %s"')
        block = self.src[start:start + 700]
        self.assertIn("only when THIS prefix has not run", block)
        self.assertIn("whenever this process has not served it yet", block)
        self.assertNotIn("if RESTORE_ONLY_WHEN_SERVER_COLD:\n        log",
                         self.src[start - 200:start + 200],
                         "printed conditionally, so silence is ambiguous")

    def test_the_off_state_names_the_defect_it_carries(self):
        self.assertIn("restore-blinds-the-ram-cache", self.src)


class TestHoistingCostsWhatItWasBuiltToSave(unittest.TestCase):
    """The hoist moves the stable part of mid-conversation system messages to
    the FRONT. Its stated reason: without it the counter Claude Code glues to
    such a block would change the prefix id every turn.

    Measured 30.08.2026, and the reason does not hold. `system_head()` reads
    only `body["system"]` (Anthropic) or `messages[0]` (OpenAI), and a
    mid-conversation system message is in neither. What the hoist DOES do is
    move the front whenever a NEW block appears — and at the prompt level that
    cost 203.2 s against 20.4 s for leaving them alone
    (bench/suites/hoist-cost.py).

    These tests pin the ID arithmetic, which is what the reason rested on, and
    hold the switch to being opt-in.
    """

    D = common.load("setup/gateway/dialects.py", "dialects_hoist")
    VOL = [re.compile(r"COUNTER \d+")]

    def body(self, counter, extra=False):
        msgs = [{"role": "user", "content": "Frage"},
                {"role": "assistant", "content": "Antwort"},
                {"role": "system", "content": "REMINDER stabil\nCOUNTER %d" % counter}]
        if extra:
            msgs.append({"role": "system",
                         "content": "REMINDER neu\nCOUNTER %d" % counter})
        msgs.append({"role": "user", "content": "Weiter"})
        return {"system": "SYSTEM stabil", "tools": [], "messages": msgs}

    def ident(self, body, hoist):
        if hoist:
            body = self.D.hoist_system_messages(body, self.D.ANTHROPIC, self.VOL)[0]
        return self.D.prefix_id(body, self.D.ANTHROPIC)[0]

    def test_the_counter_never_touched_the_id_in_the_first_place(self):
        """The whole justification for the hoist, and it is not true: the id
        is built from body["system"], and the counter is in messages."""
        self.assertEqual(self.ident(self.body(1), hoist=False),
                         self.ident(self.body(2), hoist=False))

    def test_hoisting_does_not_change_that(self):
        self.assertEqual(self.ident(self.body(1), hoist=True),
                         self.ident(self.body(2), hoist=True))

    def test_but_hoisting_makes_a_NEW_block_change_the_id(self):
        """Which is the defect: a block that appears mid-conversation is
        carried to the front, and everything behind the front is lost."""
        self.assertNotEqual(self.ident(self.body(2), hoist=True),
                            self.ident(self.body(3, extra=True), hoist=True))

    def test_leaving_them_alone_keeps_the_id_still(self):
        self.assertEqual(self.ident(self.body(2), hoist=False),
                         self.ident(self.body(3, extra=True), hoist=False))

    def test_the_switch_defaults_to_todays_behaviour(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertIn('"SYSTEM_HOCHZIEHEN", "1") == "1"', src)

    def test_off_still_counts_the_volatile_fragments(self):
        """`volatile_moved` is in the log and the trace. A switch that made it
        stop counting would make the two settings incomparable in exactly the
        records used to compare them."""
        with mock.patch.object(GW, "HOIST_SYSTEM", False), \
             mock.patch.object(GW, "VOLATILE", self.VOL):
            body, n = GW.correct(self.body(3, extra=True))
        self.assertEqual(n, 2, "two counters were in the conversation")

    def test_off_leaves_the_messages_untouched(self):
        with mock.patch.object(GW, "HOIST_SYSTEM", False), \
             mock.patch.object(GW, "VOLATILE", self.VOL):
            before = json.dumps(self.body(3, extra=True), sort_keys=True)
            body, _ = GW.correct(self.body(3, extra=True))
        self.assertEqual(json.dumps(body, sort_keys=True), before)


class TestAServerThatRestartedUnderUsIsCold(unittest.TestCase):
    """`cold` means "this process has not served this prefix". The slots
    belong to llama-server, which restarts separately, and watch_server
    notices that every 15 seconds.

    Measured 30.08.2026 with bench/suites/restore-guard-cold.py: the restart
    took 5 s and the next request arrived 2 s later, so it fell ENTIRELY
    between two polls. The request was logged `warm`, the restore path was
    never entered, and 8,711 tokens were prefilled with the file ready on
    disk. The second detector ("all slots empty") missed it too, because by
    the time it looked the slot held that very request.
    """

    def setUp(self):
        GW.MAX_TASK_ID = None
        self.addCleanup(setattr, GW, "MAX_TASK_ID", None)

    def slots(self, id_task):
        return [{"id": 0, "is_processing": False, "id_task": id_task}]

    def test_a_counter_that_fell_says_restarted(self):
        GW.server_life(self.slots(14645))
        now, restarted, why = GW.server_life(self.slots(2))
        self.assertEqual((now, restarted), (2, True))
        self.assertIn("restarted", why)

    def test_minus_one_is_what_a_fresh_slot_reports(self):
        """llama.cpp puts -1 in a slot that has never run a task, which is
        what /slots answers in the seconds after a restart — and what the
        failing run of 30.08. actually showed."""
        GW.server_life(self.slots(14645))
        self.assertEqual(GW.server_life(self.slots(-1))[:2], (-1, True))

    def test_it_says_cold_even_with_no_earlier_reading(self):
        """The gateway had never read the counter when this happened, so a
        check that only works after a baseline would not have helped."""
        self.assertEqual(GW.server_life(self.slots(-1))[:2], (-1, True))

    def test_a_working_server_is_not_restarted(self):
        self.assertEqual(GW.server_life(self.slots(14645))[:2], (14645, False))

    def test_the_second_reading_does_not_report_it_again(self):
        """The request path reads the counter and restore_from_disk reads it
        again a moment later. Reporting the restart twice would reset the
        ledger a second time for no reason."""
        GW.server_life(self.slots(14645))
        GW.server_life(self.slots(900))
        self.assertEqual(GW.server_life(self.slots(900))[:2], (900, False))

    def test_an_unreadable_counter_changes_nothing(self):
        """Two callers want opposite defaults from the same reading: for the
        RESTORE decision unknown means "carry on as before", for FORGETTING
        the ledger it means "change nothing"."""
        self.assertEqual(GW.server_life([{"id": 0}])[:2], (None, False))
        self.assertTrue(GW.server_is_cold([{"id": 0}])[0])

    def test_the_check_runs_only_where_it_can_change_something(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertIn("if not cold and ident and ident in SAVED:", src)

    def test_it_cannot_break_the_request(self):
        """A gateway that dies over its own bookkeeping is worse than one that
        cannot label a request."""
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        start = src.index("if not cold and ident and ident in SAVED:")
        block = src[start:src.index("if cold and ident in SAVED:", start)]
        self.assertIn("except Exception", block)


class TestTheSaveSurvivesAThiefItCannotSee(unittest.TestCase):
    """The bracket around a save watches traffic THROUGH the gateway.
    llama-probe.service goes straight to llama-server, so SERVED_COUNT never
    moves and `foreign` stays empty — only prewarm's own token-count check
    catches it, correctly, by refusing to publish.

    Measured 30.08.2026, 16:07:47: the prefix prefill released 10,098 tokens,
    the probe took the slot 0.0 s later and released 34, and the save captured
    those 34. The file was refused, so there was nothing on disk, so the
    cold-server test of the restore guard failed for a reason that had nothing
    to do with the guard.
    """

    def test_a_refusal_is_recognised_as_a_theft(self):
        self.assertTrue(GW.slot_was_stolen(
            b"refusing to publish x.bin: 34 tokens were written where the "
            b"prefix is 10098."))

    def test_other_failures_are_not_retried(self):
        """A template that cannot be rendered or a full disk will fail again;
        retrying those only delays the log line that says so."""
        for other in (b"server did not become ready in time",
                      b"disk at 101.0 of 100 GB", b"", None):
            with self.subTest(out=other):
                self.assertFalse(GW.slot_was_stolen(other))

    def test_it_matches_the_wording_prewarm_actually_uses(self):
        """Matched on the message rather than an exit code, so the two must
        stay pinned together."""
        pw = (common.REPO / "tools" / "prewarm.py").read_text(encoding="utf-8")
        self.assertIn("refusing to publish", pw)

    def test_one_retry_and_the_reason_for_the_number(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertIn('SAVE_RETRIES = int(env("SAVE_RETRIES", "SICHERN_VERSUCHE", 1))',
                      src)
        self.assertIn("three would be a different defect", src)

    def test_the_failure_log_is_not_truncated_to_its_tail(self):
        """prewarm's refusal carries its numbers in the FIRST line. Cutting to
        the last 200 characters kept only the explanation, and it cost a
        measurement to notice the log had thrown away the one thing needed to
        read it."""
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        start = src.index('log("NOTE        automatic save of %s failed')
        self.assertNotIn("[-200:]", src[start:start + 300])


class TestPrewarmAndTheGatewayHoistTheSameWay(unittest.TestCase):
    """Three places render the prefix: cc-gateway's correct(), its
    render_id_of(), and prewarm's build_prefix(). HOIST_SYSTEM was added to
    the first only, and for a few minutes on 30.08.2026 the other two still
    hoisted — so a save would have written the hoisted rendering while the
    request sent the other one, every restore would have matched nothing, and
    the file would have been quarantined for a defect that was in the renderer.
    """

    def test_prewarm_reads_the_same_switch(self):
        pw = (common.REPO / "tools" / "prewarm.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("HOIST_SYSTEM", "1") == "1"', pw)

    def test_the_gateway_passes_it_explicitly_rather_than_inheriting(self):
        """Inheriting works until somebody runs prewarm from a shell."""
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertIn('"--hoist", "1" if HOIST_SYSTEM else "0",', src)

    def test_the_render_id_follows_it_too(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        start = src.index("def render_id_of(")
        block = src[start:src.index("def ", start + 10)]
        self.assertIn("if HOIST_SYSTEM:", block)

    def test_prewarm_can_be_told_either_way(self):
        pw = (common.REPO / "tools" / "prewarm.py").read_text(encoding="utf-8")
        self.assertIn('s1.add_argument("--hoist"', pw)


class TestWhetherThisPrefixRanOnThisServer(unittest.TestCase):
    """"Is the server cold" and "does the server's cache hold THIS prefix" are
    different questions, and the difference was measured on 30.08.2026 at
    16:20:57 in live DeepSeek-harness traffic: the server had run 41 tasks, so
    it was not cold, but all 41 were a benchmark and the conversation's own
    prefix had not been near it since the restart. The restore was skipped and
    8,278 tokens were prefilled where an 8,077-token file lay ready. 74 s.

    The counter only rises within one llama-server life, so `now >= seen`
    means "the same life that already had this prefix" — and nothing else has
    to be remembered, neither a boot id nor a timestamp, both of which the
    server withholds.
    """

    def setUp(self):
        GW.MAX_TASK_ID = None
        GW.SEEN.clear()
        self.addCleanup(GW.SEEN.clear)
        self.addCleanup(setattr, GW, "MAX_TASK_ID", None)

    def slots(self, id_task):
        return [{"id": 0, "is_processing": False, "id_task": id_task}]

    def test_a_prefix_this_server_never_saw_is_cold(self):
        cold, _, why = GW.prefix_is_cold_on_this_server("p", self.slots(41))
        self.assertTrue(cold, "41 tasks of somebody else's traffic prove nothing")
        self.assertIn("has not been served", why)

    def test_a_prefix_served_in_this_life_is_not(self):
        GW.SEEN["p"] = 8842
        cold, _, why = GW.prefix_is_cold_on_this_server("p", self.slots(9100))
        self.assertFalse(cold)
        self.assertIn("its own cache is the better source", why)

    def test_a_counter_below_the_mark_means_the_server_restarted(self):
        """The 29.08. incident inverted: the gateway restarted beside a warm
        server and every prefix looked new. Here the SERVER restarted and the
        ledger notices, because the counter went backwards."""
        GW.SEEN["p"] = 8842
        cold, _, why = GW.prefix_is_cold_on_this_server("p", self.slots(41))
        self.assertTrue(cold)
        self.assertIn("restarted", why)

    def test_an_unreadable_counter_decides_as_before(self):
        cold, seen, why = GW.prefix_is_cold_on_this_server("p", [{"id": 0}])
        self.assertTrue(cold)
        self.assertIsNone(seen)
        self.assertIn("as before", why)

    def test_the_ledger_survives_a_gateway_restart(self):
        """The 29.08. incident needs it to: the gateway had restarted at 23:38
        beside a server up since 09:48, so anything held only in memory was
        gone exactly when it was needed."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seen.json")
            GW.SEEN.update({"a": 12, "b": 34})
            GW.save_seen(path)
            self.assertEqual(GW.load_seen(path), {"a": 12, "b": 34})

    def test_a_missing_or_broken_ledger_is_not_an_error(self):
        """Nothing known means restore, which is what ran before this existed."""
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(GW.load_seen(os.path.join(d, "nope.json")), {})
            bad = os.path.join(d, "bad.json")
            open(bad, "w").write("{not json")
            self.assertEqual(GW.load_seen(bad), {})

    def test_non_integer_entries_are_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seen.json")
            json.dump({"a": 12, "b": "later", "c": None}, open(path, "w"))
            self.assertEqual(GW.load_seen(path), {"a": 12})

    def test_the_mark_is_set_by_the_measurement_not_the_intention(self):
        """Marking a prefix "the server has it" because we chose not to restore
        would make one wrong skip permanent. Zero reuse says otherwise, and
        forgetting the entry makes the next request restore."""
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertIn("if reuse[0] > 0:\n                    SEEN[ident] = MAX_TASK_ID",
                      src)
        self.assertIn("SEEN.pop(ident, None)", src)

    def test_the_ledger_is_read_at_startup(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        self.assertIn('globals()["SEEN"] = load_seen()', src)


class TestAChurningPrefixIsNotWorthTheWait(unittest.TestCase):
    """Claude Code's tool list changed three times in one session, and each
    new prefix was saved again — near-identical 16k files, 20.6 GB in the
    store, each made worthless by the next change. The gateway already WARNED
    that the heads collide, a few lines before it wrote the file anyway.

    WHAT THAT WASTES IS DISK, NOT TIME, and the first version of this test said
    otherwise. The save's log line reads "80.9 s", but that is the prefix being
    computed — work the first request has to do regardless, moved in front of
    it. Measured 30.08. 17:46: 80.9 + 454.6 with the save, and ~539 s for the
    same turn computing 53,638 tokens in one pass without it. What the save
    ADDS is the write: 237 ms for 878 MB.

    TURNED AROUND 01.09.2026: the blanket refusal assumed churn — a set that
    flips back and forth. Four days of traces say the sets DRIFT and never
    return, so the refusal kept dead files and charged every new session a
    full cold start (80,721 tokens, 577 s, measured 15:39). Now only a rival
    that was asked for within RIVAL_GRACE_S still stops the save; an idle one
    is deleted and its successor takes the disk. The split is
    savepolicy.stale_rivals, tested where it lives.
    """

    def setUp(self):
        self.src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")

    def test_an_active_rival_still_stops_the_save(self):
        self.assertIn("SP.stale_rivals", self.src)
        self.assertIn('"save-skipped"', self.src)

    def test_a_stale_rival_gives_way_to_its_successor(self):
        self.assertIn('"save-replaced"', self.src)
        self.assertIn("drop_saved(", self.src)

    def test_replacement_deletes_rather_than_quarantines(self):
        """Quarantine is for a file that answered WRONGLY and is evidence.
        A file whose prompt shape no client sends any more is not evidence
        of anything; keeping it as .unusable would spend the gigabytes the
        replacement exists to stop wasting."""
        self.assertIn("Deletion, not quarantine", self.src)

    def test_the_grace_has_one_home_and_is_settable(self):
        self.assertIn('RIVAL_GRACE_S = float(env("RIVAL_GRACE_S"', self.src)

    def test_the_activity_reading_survives_a_restart(self):
        """The stamp comes from the sidecar record_use writes, not from
        gateway memory — a counter that forgets on restart was the hole in
        the savepolicy ledger once already."""
        self.assertIn("def last_activity_epoch(", self.src)
        self.assertIn('d.get("last_used")', self.src)

    def test_only_rivals_ON_DISK_count(self):
        """A collision with a prefix merely SEEN is the normal state of two
        prompt types in one session; refusing there would stop the store from
        ever filling."""
        start = self.src.index("saved_rivals = [k for k in SAVED")
        self.assertIn("for k in SAVED", self.src[start:start + 120])

    def test_it_survives_a_body_that_did_not_parse(self):
        """`head` is assigned beside `ident` inside the try. Without an
        initialiser a malformed body would reach this line with the name
        unbound — a NameError in the request path, for a bookkeeping decision."""
        self.assertIn("head = None                # set with ident", self.src)
        start = self.src.index("saved_rivals = [k for k in SAVED")
        self.assertIn("if (ident and head) else []",
                      self.src[start:start + 300])

    def test_the_reason_does_not_bill_the_prefill_to_the_save(self):
        """The prefix has to be computed for the first request either way. A
        comment that charges those seconds to the save invites removing the
        save to get them back, and they do not come back."""
        self.assertIn("WHAT IT COSTS IS SPACE, NOT TIME", self.src)
        self.assertIn("237 ms", self.src)


class TestTheTwoPhasesAreSeparableWithoutTheServersHelp(unittest.TestCase):
    """The Anthropic route carries no `timings`, so `read_tps` and `write_tps`
    are empty for Claude Code and prefill and generation hide inside one
    `took_s`. Asked on 30.08.2026 where a "12 tokens/s" figure came from, the
    honest answer was "llama-server's journal, not the trace".

    The gateway sees every chunk go past, so it can time the changeover itself.
    """

    def setUp(self):
        self.src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")

    def test_it_times_the_first_DELTA_not_the_first_chunk(self):
        """A stream opens with headers and a message_start, and a queued one
        gets keep-alive pings before that. Timing any of those would put the
        prefill in the wrong column."""
        self.assertIn('b"content_block_delta" in ch or b\'"delta"\' in ch',
                      self.src)

    def test_it_stops_looking_once_it_has_one(self):
        self.assertIn('if sniff.get("first_token_at") is None and (', self.src)

    def test_the_derived_rate_says_it_is_derived(self):
        """It includes the network hop and this process's scheduling, and it
        must never be mistaken for llama.cpp's own accounting — which is what
        read_tps/write_tps carry when the route provides them."""
        self.assertIn('"write_tps_derived"', self.src)
        self.assertIn('"read_tps": rates[0] if rates else None', self.src)
        self.assertIn('"write_tps": rates[1] if rates else None', self.src)

    def test_it_refuses_to_divide_by_almost_nothing(self):
        """A turn that answers in under half a second of generation would
        produce a rate of several thousand tokens per second and put it in a
        column next to real ones."""
        self.assertIn("(took - ttft) > 0.5", self.src)

    def test_took_is_not_measured_a_second_time(self):
        """Re-measuring here would move `took_s` by however long the trace
        block takes — a number quietly worse than the one it replaced."""
        self.assertEqual(self.src.count("took = time.time() - t_start"), 1)

    def test_the_wait_is_named_rather_than_subtracted(self):
        """ttft is measured from admission, so it carries the queue wait.
        Subtracting it here would hide a queue; `waited_s` is in the same
        record for whoever wants the difference."""
        self.assertIn("carries the queue wait with it", self.src)
        self.assertIn('"waited_s": round(waited, 2)', self.src)


class TestARestartingServerIsNotAStackTrace(unittest.IsolatedAsyncioTestCase):
    """switch-model.sh stops and starts llama-server as a matter of course, and
    every one of those windows reached aiohttp as an unhandled
    ConnectionRefusedError — a bare 500 for the client and a stack trace in the
    log that tells a reader nothing to act on. Seen 30.08.2026 while the
    operator restarted the server to change reasoning mode."""

    async def _run(self, prepared=False):
        # A port nothing listens on: the connection is refused for real, which
        # is the case under test. Mocking the exception would test the mock.
        with mock.patch.object(GW, "LLAMA", "http://127.0.0.1:1"), \
             mock.patch.object(GW, "log", lambda *a: None):
            app = web.Application()
            async def handler(request):
                return await GW.forward(request, b"{}", b"{}")
            app.router.add_route("*", "/{tail:.*}", handler)
            server = TestServer(app)
            await server.start_server()
            try:
                async with aiohttp.ClientSession() as c:
                    async with c.post(str(server.make_url("/v1/messages")),
                                      json={}) as r:
                        return r.status, r.headers.get("Retry-After"), await r.json()
            finally:
                await server.close()

    async def test_it_answers_503_rather_than_raising(self):
        status, retry, body = await self._run()
        self.assertEqual(status, 503)
        self.assertEqual(retry, "5")
        self.assertEqual(body["error"]["type"], "service_unavailable")

    async def test_the_message_says_what_to_do(self):
        _, _, body = await self._run()
        self.assertIn("restarting", body["error"]["message"])

    def test_it_is_caught_where_it_happens_not_probed_for(self):
        """A /health call before every request would put a round trip in the
        hot path to describe a state that is rare, and the connection attempt
        answers the same question for free."""
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        start = src.index("async def forward(")
        block = src[start:src.index("\nasync def status(", start)]
        self.assertNotIn('"/health"', block,
                         "a probe per request describes a rare state at the "
                         "cost of every common one")
        self.assertIn("except (OSError, aiohttp.ClientConnectionError)", block)

    def test_a_cancelled_request_is_not_swallowed(self):
        src = (common.REPO / "setup" / "gateway" / "gateway.py").read_text(
            encoding="utf-8")
        start = src.index("async def forward(")
        block = src[start:src.index("\nasync def status(", start)]
        at = block.index("except (OSError, aiohttp.ClientConnectionError)")
        self.assertIn("except asyncio.CancelledError:\n            raise",
                      block[:at],
                      "CancelledError is not an error here — it is a client "
                      "that left, and swallowing it would turn an abort into "
                      "a 503 nobody asked for")


class TestSessionSave(unittest.IsolatedAsyncioTestCase):
    """The save that writes the slot AS IT STANDS — cooldown, floor, and the
    one failure mode that must stay loud.

    Not to be confused with TestAutoSave above: that one covers the prewarm
    path, which re-creates a bare prefix in a subprocess. This one is a single
    HTTP call after a turn and shares nothing with it but the word "save".
    """

    async def asyncSetUp(self):
        self.posts = []
        self.old = {k: getattr(GW, k) for k in
                    ("SESSION_SAVE", "SESSION_SAVE_COOLDOWN_S",
                     "SESSION_SAVE_MIN_TOKENS")}
        GW.SESSION_SAVE = True
        GW.SESSION_SAVE_COOLDOWN_S = 300.0
        GW.SESSION_SAVE_MIN_TOKENS = 20000
        GW.SESSION_SAVED_AT.clear()
        self.log_lines = []
        self.patches = [
            mock.patch.object(GW, "log",
                              lambda *a: self.log_lines.append(" ".join(map(str, a)))),
            mock.patch.object(GW, "_post_json", self._fake_post),
        ]
        for p in self.patches:
            p.start()

    async def asyncTearDown(self):
        for p in self.patches:
            p.stop()
        for k, v in self.old.items():
            setattr(GW, k, v)
        GW.SESSION_SAVED_AT.clear()

    def _fake_post(self, url, payload, timeout):
        self.posts.append((url, payload))
        return {"n_saved": 42000, "n_written": 1_500_000_000,
                "timings": {"save_ms": 1900}}

    async def test_saves_a_deep_turn(self):
        await GW.session_save("abc123", (40000, 2000))
        self.assertEqual(len(self.posts), 1)
        url, payload = self.posts[0]
        self.assertIn("action=save", url)
        self.assertEqual(payload["filename"], "session-abc123.bin")

    async def test_off_by_default_does_nothing(self):
        GW.SESSION_SAVE = False
        await GW.session_save("abc123", (40000, 2000))
        self.assertEqual(self.posts, [])

    async def test_shallow_state_is_not_worth_a_file(self):
        await GW.session_save("abc123", (1000, 500))
        self.assertEqual(self.posts, [])

    async def test_cooldown_holds_the_second_turn_back(self):
        await GW.session_save("abc123", (40000, 2000))
        await GW.session_save("abc123", (41000, 2000))
        self.assertEqual(len(self.posts), 1, "the cooldown did not hold")

    async def test_cooldown_expires(self):
        await GW.session_save("abc123", (40000, 2000))
        GW.SESSION_SAVED_AT["abc123"] -= GW.SESSION_SAVE_COOLDOWN_S + 1
        await GW.session_save("abc123", (41000, 2000))
        self.assertEqual(len(self.posts), 2)

    async def test_cooldown_is_per_prefix(self):
        await GW.session_save("aaa", (40000, 2000))
        await GW.session_save("bbb", (40000, 2000))
        self.assertEqual(len(self.posts), 2, "one prefix blocked another")

    async def test_a_server_without_the_flag_is_reported_once(self):
        def refuse(url, payload, timeout):
            raise urllib.error.HTTPError(url, 501, "not supported", {}, None)
        with mock.patch.object(GW, "_post_json", refuse):
            await GW.session_save("abc123", (40000, 2000))
            GW.SESSION_SAVED_AT["abc123"] -= GW.SESSION_SAVE_COOLDOWN_S + 1
            await GW.session_save("abc123", (40000, 2000))
        said = [l for l in self.log_lines if "session save unavailable" in l]
        self.assertEqual(len(said), 1, "should say so once, not every turn")

    async def test_a_programming_error_is_NOT_swallowed(self):
        """The clause is narrow on purpose. A NameError reported as "the server
        does not support this" is a lie that looks like a configuration — and
        it was the actual state of this function in review on 02.09.2026,
        where the constant was LLAMA_URL instead of LLAMA."""
        def boom(url, payload, timeout):
            raise NameError("LLAMA_URL")
        with mock.patch.object(GW, "_post_json", boom):
            with self.assertRaises(NameError):
                await GW.session_save("abc123", (40000, 2000))

    async def test_unknown_token_count_still_saves(self):
        """The Anthropic route carries no timings, so `reuse` is None there.
        Refusing to save would mean never saving for Claude Code, which is the
        one consumer this exists for."""
        await GW.session_save("abc123", None)
        self.assertEqual(len(self.posts), 1)
