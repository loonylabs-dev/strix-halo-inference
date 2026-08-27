#!/usr/bin/env python3
"""cc-gateway — the single entrance to the local llama-server.

Replaces cc-cachefix2.py and carries its logic in full. On top of that:

  * admission control     at most N requests in flight at once
  * priority              local requests before LAN before tunnel
  * token                 for everything that does not come from 127.0.0.1
  * time limit            a silent connection never blocks a slot forever
  * log                   who, which priority, how long waited, cold/warm

Why this is needed: this machine has exactly ONE llama-server — Laguna takes
68.35 of 96 GiB, a second instance does not fit. Every user shares the same
process and the same GPU. Measured:

    decode, one session alone       25.3 t/s
    decode, two sessions at once    18.7 t/s each, 37.3 t/s together
    one cold prefix                 66-110 s, and it sets the pace for everyone
    a warm request alongside it     56 s instead of 4.8 s (11.7x)

Measured 2026-08-24 with bench/suites/prefill-decode.py, directly against
llama-server. Two lessons: decode SHARES well (two sessions gain 47 % in
aggregate and each loses only 26 %), so admitting several is right. A prefill
does NOT share — it sets the pace, and a warm request beside it crawls. The
expensive thing to schedule is therefore the cold prefix, not the request.

Without a limit a single foreign cold start can freeze your own session. The
priority ordering makes sure the local request goes next instead of queueing
behind five foreign ones.

Environment variables
    BIND             addresses, comma separated (default 127.0.0.1)
    PORT             default 8090
    LLAMA_URL        default http://127.0.0.1:8080
    MAX_INFLIGHT     in flight at once (default 2 — match it to -np)
    TOKEN            access token for non-local requests (empty = no
                     non-local requests allowed)
    SILENCE_MAX      seconds without a single byte before aborting
                     (default 900; a cold start needs ~100 s)
"""
import asyncio, heapq, ipaddress, json, os, re, sys, time
from aiohttp import web, ClientSession, ClientTimeout

# dialects.py sits next to this file — both in the repo and as a symlink in
# ~/.claude/bin. Importing by directory instead of by package keeps that
# working without an installation step.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dialects as DIA                                   # noqa: E402

def env(name, old=None, default=None):
    """Read an environment variable, still accepting its former German name.

    These variables were renamed to English in August 2026. An installation
    that still carries the old spelling in ~/.config/cc-gateway.env would
    otherwise fall back to the default silently — a change in behaviour that
    nobody would see. So the old name keeps working and says so once.
    """
    if name in os.environ:
        return os.environ[name]
    if old and old in os.environ:
        print("NOTE  %s is deprecated, please use %s" % (old, name), flush=True)
        return os.environ[old]
    return default

LLAMA        = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
PORT         = int(os.environ.get("PORT", 8090))
BIND         = [a.strip() for a in os.environ.get("BIND", "127.0.0.1").split(",") if a.strip()]
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", 2))
# How long to wait for the model server at startup before giving up and
# guessing the slot count. Long enough for a cold load of a large model from
# the NTFS partition; see query_slots for why guessing is the bad outcome.
SLOTS_WAIT   = int(os.environ.get("SLOTS_WAIT", 240))
TOKEN        = os.environ.get("TOKEN", "")
SILENCE_MAX  = int(env("SILENCE_MAX", "STILLE_MAX", 900))
# A separate port for tunnel traffic. The reason: when a request arrives
# through a tunnel, the source IP is no longer the client's but cloudflared's
# — 127.0.0.1 when run natively, 172.17.0.x in a container. Classifying by IP
# would therefore treat internet traffic as "local" or "lan", i.e. higher than
# it deserves. A separate port solves that without trusting any header:
# whatever arrives here is always "remote".
TUNNEL_PORT  = int(os.environ.get("TUNNEL_PORT", 0)) or None
TUNNEL_BIND  = [a.strip() for a in os.environ.get("TUNNEL_BIND", "127.0.0.1").split(",") if a.strip()]
# File with named tokens, one consumer per line:  name<whitespace>secret
# A single consumer can be revoked that way, and the log says WHO did
# something. One shared token can do neither.
TOKEN_FILE   = env("TOKEN_FILE", "TOKENDATEI",
                   os.path.expanduser("~/.config/cc-gateway-tokens"))
# Concurrent requests per consumer. Claude Code sends up to two prompt types
# in parallel, hence 2 as the default.
PER_TOKEN_MAX = int(env("PER_TOKEN_MAX", "JE_TOKEN_MAX", 2))
# Some chat templates hard-reject system messages that are not at position 0
# (measured 24.08.2026 with Qwen 3.8: 'System message must be at the
# beginning', HTTP 500 for every Claude-Code-shaped request, because the
# volatile counter block sits BEHIND the user question). 1 = rewrite those
# messages into user text blocks after the cache correction. Off by default:
# it changes the rendered prompt, and Laguna neither needs nor knows it.
# See docs/MODELS.md.
MID_SYSTEM_TO_USER = env("MID_SYSTEM_TO_USER", default="0") == "1"
# Thinking mode per MODEL NAME, without a second server: llama-server merges a
# request-level chat_template_kwargs over its command-line default (verified
# in server-common.cpp, and the Anthropic route passes the field through).
# One loaded model therefore serves several modes, switchable per request —
# e.g. '{"qwen38": {"enable_thinking": false},
#        "qwen38-think": {"reasoning_effort": "medium"}}'
# The consumer picks the mode by model name (/model in Claude Code). Kwargs
# already present in the request always win — the map only fills the gap.
try:
    KWARGS_BY_MODEL = json.loads(env("KWARGS_BY_MODEL", default="{}"))
    if not isinstance(KWARGS_BY_MODEL, dict):
        raise ValueError("not an object")
except ValueError as e:
    print("KWARGS_BY_MODEL is not valid JSON (%s) — ignored" % e, flush=True)
    KWARGS_BY_MODEL = {}

HOP  = {"host", "content-length", "connection", "transfer-encoding",
        "keep-alive", "accept-encoding"}
DROP = ("thinking", "context_management", "output_config")

# Content that changes from turn to turn and must therefore NOT move to the front.
VOLATILE = [re.compile(r"<total_tokens>\s*\d+\s*tokens left\s*</total_tokens>")]

T0 = time.time()

def load_tokens():
    """Read the token file. Returns {secret: name}."""
    table = {}
    if TOKEN:                     # backwards compatible: TOKEN from the environment
        table[TOKEN] = "standard"
    try:
        with open(TOKEN_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    name, secret = parts[0], parts[1].strip()
                    if secret:
                        table[secret] = name
    except FileNotFoundError:
        pass
    except Exception as e:
        print("token file not readable: %r" % (e,), flush=True)
    return table

TOKENS = {}
IN_FLIGHT_PER_TOKEN = {}

# Saved prefixes on disk: {gateway_id: filename-without-extension}
# The disk therefore covers any number of projects, not just as many as there
# are slots. When a cold request arrives whose prefix lies there, it is pulled
# into a free slot before forwarding — ~0.15 s instead of 110 s.
SLOT_PATH = env("SLOT_PATH", "SLOTPFAD", os.path.expanduser("~/.cache/llama-slots"))
SAVED = {}

# Save automatically once a prefix has warmed up for the first time. 0 turns
# it off. The upper limit keeps the disk from filling up; cleanup goes by
# least-recently-used (see prewarm.py cleanup).
AUTO_SAVE = env("AUTO_SAVE", "AUTO_SICHERN", "1") == "1"
AUTO_MAX_GB  = float(os.environ.get("AUTO_MAX_GB", 20))
# Below this prefix size saving is not worth it: a few hundred tokens are
# computed in fractions of a second. Without the limit every smoke test with a
# minimal body drops a file into the store — in production there was a prefix
# of SEVEN tokens in there, which on the next start would have occupied one of
# the two slots that a real project belongs in.
AUTO_MIN_CHARS = int(env("AUTO_MIN_CHARS", "AUTO_MIN_ZEICHEN", 4000))
PREWARM   = env("PREWARM", "VORWAERMEN",
                os.path.expanduser("~/.claude/bin/prewarm.py"))
_save_lock = None          # asyncio.Lock, can only be created inside the loop

# When was a saved prefix last USED? That is the only usable criterion for
# cleanup: a prefix never changes through use (it holds only the system prompt
# and the tools), so the file timestamp only says when it was saved. Written
# at most hourly per prefix, so the disk does not work needlessly.
# at most hourly per prefix, so the disk does not work needlessly.
USE_INTERVAL = 3600
_last_written = {}

def record_use(id_):
    name = SAVED.get(id_)
    if not name:
        return
    now = time.time()
    if now - _last_written.get(id_, 0) < USE_INTERVAL:
        return
    _last_written[id_] = now
    path = os.path.join(SLOT_PATH, name + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d["last_used"] = time.strftime("%Y-%m-%d %H:%M")
        d["used_total"] = d.get("used_total", d.get("benutzt_gesamt", 0)) + 1
        d.pop("benutzt_gesamt", None)      # migrate the old German key away
        d.pop("zuletzt_benutzt", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log("use not recorded (%s): %r" % (name, e))

def load_saved():
    table = {}
    try:
        for f in os.listdir(SLOT_PATH):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(SLOT_PATH, f), encoding="utf-8") as fh:
                d = json.load(fh)
            # "gateway_kennung" is the pre-rename spelling. Sidecar files
            # written before August 2026 still carry it; reading both keeps
            # 1.9 GB of saved prefixes usable instead of orphaning them.
            gk = d.get("gateway_id") or d.get("gateway_kennung")
            if gk and os.path.exists(os.path.join(SLOT_PATH, d["name"] + ".bin")):
                table[gk] = d["name"]
    except FileNotFoundError:
        pass
    except Exception as e:
        log("saved prefixes not readable: %r" % (e,))
    return table

# State of the store at the last read (st_mtime_ns of the directory).
_saved_state = object()          # deliberately not a valid mtime value

def refresh_saved(force=False):
    """Re-read the store as soon as the directory has changed.

    Needed because prefix-cleanup.timer deletes in its OWN process. Without
    this a gateway running for weeks would still consider deleted files
    present, occupy a gate slot for them and run into a restore on nothing —
    and the caller pays the full cold start.
    """
    global SAVED, _saved_state
    try:
        state = os.stat(SLOT_PATH).st_mtime_ns
    except OSError:
        state = None
    if force or state != _saved_state:
        _saved_state = state
        SAVED = load_saved()
    return SAVED

def disk_used_gb():
    try:
        return sum(os.path.getsize(os.path.join(SLOT_PATH, f))
                   for f in os.listdir(SLOT_PATH) if f.endswith(".bin")) / 1e9
    except FileNotFoundError:
        return 0.0

# At most this many prefixes wait for their save at the same time. A second
# concurrent save used to be dropped without replacement. That was worse than
# it looks: the prefix then stood as "warm" in PREFIXES, was never recognised
# as cold again and never made it to disk. Now it waits instead of being
# discarded — and if too many really do pile up, one is dropped, but with a
# log line instead of silently.
SAVE_QUEUE_MAX = int(env("SAVE_QUEUE_MAX", "SICHERN_WARTEN_MAX", 4))
_save_pending = set()

async def auto_save(id_, body, dialect=DIA.ANTHROPIC):
    """Save the PREFIX of a request once it has warmed up.

    Important: do not save the slot as it stands after the request — that
    holds prefix PLUS question PLUS answer and is then good for exactly that
    one request. What belongs on disk is the part up to <user>.

    prewarm.py does that, the very same version that is also used by hand —
    no second code path that could drift apart. It precomputes the prefix via
    /completion; because the slot is already warm, that costs fractions of a
    second instead of a full cold start.

    `body` is the CORRECTED body — the way it is about to go to llama-server;
    only from it can the prefix be rendered that later actually arrives. The
    id must therefore NOT be derived from it: it comes from the raw body and
    is passed through with --gateway-id. That is exactly what went wrong
    before — prewarm.py recomputed it from the corrected body and so wrote a
    key into the sidecar file that no incoming request ever produces. Result:
    saved again on every cold start, restored never.
    """
    global SAVED, _save_lock
    if _save_lock is None:
        _save_lock = asyncio.Lock()
    if id_ in SAVED or id_ in _save_pending:
        return                       # long since there, or already queued
    if len(_save_pending) >= SAVE_QUEUE_MAX:
        log("NOTE        %s not saved: %d saves already queued"
            % (id_, len(_save_pending)))
        return
    _save_pending.add(id_)
    try:
        async with _save_lock:
            if id_ in SAVED:
                return
            used = disk_used_gb()
            if used >= AUTO_MAX_GB:
                log("NOTE        not saved: disk at %.1f of %.0f GB — "
                    "is 'prewarm.py cleanup --max-gb %g' running?"
                    % (used, AUTO_MAX_GB, AUTO_MAX_GB))
                return
            tmp = os.path.join("/tmp", "cc-auto-%s.json" % id_)
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(body, f, ensure_ascii=False)
                t0 = time.time()
                pr = await asyncio.create_subprocess_exec(
                    sys.executable, PREWARM, "save",
                    "--body", tmp, "--name", id_,
                    "--gateway-id", id_, "--dialect", dialect,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                proc_out, _ = await pr.communicate()
                if pr.returncode == 0:
                    SAVED = refresh_saved(force=True)
                    if id_ not in SAVED:
                        log("WARNING     %s saved but missing from the store — "
                            "the sidecar file does not match the id"
                            % id_)
                    log("SAVED       prefix %s automatically, %.1f s, disk now %.1f GB"
                        % (id_, time.time() - t0, disk_used_gb()))
                else:
                    log("NOTE        automatic save of %s failed: %s"
                        % (id_, (proc_out or b"").decode("utf-8", "replace")[-200:].strip()))
            except Exception as e:
                log("NOTE        automatic save: %r" % (e,))
            finally:
                try: os.remove(tmp)
                except OSError: pass
    finally:
        _save_pending.discard(id_)

async def restore_from_disk(id_):
    """Pull a saved prefix into a free slot. True on success."""
    name = SAVED.get(id_)
    # The file may be gone since the last read (cleanup service). Then re-read
    # instead of handing llama-server a restore onto nothing.
    if name and not os.path.exists(os.path.join(SLOT_PATH, name + ".bin")):
        log("NOTE        %s.bin has vanished — store re-read" % name)
        name = refresh_saved(force=True).get(id_)
    if not name:
        return False
    try:
        timeout = ClientTimeout(total=300)
        async with ClientSession(timeout=timeout) as s:
            async with s.get(LLAMA + "/slots") as r:
                slots = await r.json()
            # Restore ONLY into a fully idle server. The 25.08. incident:
            # a restore into an idle slot WHILE the other slot was computing
            # left the server producing degenerate output until a fresh
            # start. A skipped restore just means a cold prefill — annoying;
            # a poisoned KV state ruins every answer after it — fatal.
            if any(x.get("is_processing") for x in slots):
                log("NOTE        restore of %s deferred: server busy" % name)
                return False
    # Prefer EMPTY slots. Otherwise the reload overwrites a prefix that was
    # just fetched — in testing projB evicted projA from slot 0, and the next
    # request for projA ran cold.
            empty = [x["id"] for x in slots
                    if not x.get("is_processing") and not x.get("n_prompt_tokens")]
            used = [x["id"] for x in slots
                      if not x.get("is_processing") and x.get("n_prompt_tokens")]
            free = empty + used
            if not free:
                log("NOTE        no free slot for %s" % name)
                return False
            target = free[0]
            if not empty:
                log("NOTE        all slots busy — %s evicts slot %d" % (name, target))
            t0 = time.time()
            async with s.post(LLAMA + "/slots/%d?action=restore" % target,
                              json={"filename": name + ".bin"}) as r:
                if r.status != 200:
                    log("NOTE        restoring %s: HTTP %d" % (name, r.status))
                    return False
                d = await r.json()
        log("RESTORED    prefix %s from %s.bin -> slot %d, %d tokens, %.0f ms"
            % (id_, name, target, d.get("n_restored", -1),
               (time.time() - t0) * 1000))
        return True
    except Exception as e:
        log("NOTE        restore failed: %r" % (e,))
        return False

def log(*a):
    print("[%8.1fs]" % (time.time() - T0), *a, flush=True)

# --------------------------------------------------------------- priority ---
def priority(ip, port=None):
    """0 = local, 1 = private network (LAN), 2 = everything else (tunnel).

    The tunnel port decides before the IP — see the comment at TUNNEL_PORT.
    """
    if TUNNEL_PORT and port == TUNNEL_PORT:
        return 2
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return 2
    if a.is_loopback:
        return 0
    if a.is_private:
        return 1
    return 2

PRIORITY_NAME = {0: "local", 1: "lan", 2: "remote"}

def zone(req):
    """Priority of a request — from source IP AND destination port.

    Every path must answer this the same way. status() used to ignore the port
    and classify by IP alone: if cloudflared runs natively instead of in a
    container, the tunnel comes from 127.0.0.1, and /gateway/status would have
    been readable from the internet — with prefix names, consumer names and
    source IPs. The handler was protected, status() was not; preventing two
    versions of the same rule is exactly what this function is for.
    """
    port = None
    try:
        sock = req.transport.get_extra_info("sockname")
        port = sock[1] if sock else None
    except Exception:
        pass
    return priority(req.remote or "?", port)

# After this many seconds of waiting, a request is served next regardless of
# its zone. Strict priority alone starves the lower zones: measured, a LAN
# caller was still waiting after 200 s while four local streams kept working,
# and two Claude Code sessions are exactly four streams. Priority should decide
# who goes first, not who goes at all.
AGE_AFTER = int(env("QUEUE_AGE_AFTER", None, 30))

class PriorityGate:
    """Semaphore that serves waiters by priority — but not forever.

    Priority decides the order while everyone is fresh. Once a waiter has been
    in the queue longer than AGE_AFTER, it is served before any newer arrival,
    whatever its zone. That keeps the ordering the operator wants without
    letting a busy local user lock a remote caller out entirely.
    """
    def __init__(self, n):
        self.free = n
        self.queue = []      # heap of (priority, seq, future, queued_at)
        self.seq = 0
        self.overtaken = 0   # how often ageing beat priority, for /gateway/status

    def depth(self):
        return len(self.queue)

    async def enter(self, prio):
        if self.free > 0:
            self.free -= 1
            return 0.0
        fut = asyncio.get_running_loop().create_future()
        self.seq += 1
        heapq.heappush(self.queue, (prio, self.seq, fut, time.time()))
        t0 = time.time()
        try:
            await fut
        except asyncio.CancelledError:
            # If we did get the slot and were cancelled anyway, it has to go
            # back — otherwise it seeps away.
            if fut.done() and not fut.cancelled():
                self.leave()
            raise
        return time.time() - t0

    def _take_overdue(self, now):
        """The waiter that has been queued longest, if it waited too long."""
        overdue = [e for e in self.queue
                   if not e[2].done() and now - e[3] >= AGE_AFTER]
        if not overdue:
            return None
        oldest = min(overdue, key=lambda e: e[3])
        self.queue.remove(oldest)
        heapq.heapify(self.queue)
        return oldest

    def leave(self):
        now = time.time()
        overdue = self._take_overdue(now)
        if overdue is not None:
            self.overtaken += 1
            overdue[2].set_result(None)
            return
        while self.queue:
            _, _, fut, _ = heapq.heappop(self.queue)
            if not fut.done():
                fut.set_result(None)
                return
        self.free += 1

# How often to send a sign of life while a request is queued. llama-server
# sends the same thing every 30 s during a long prefill, and that is what makes
# the tunnel work at all: Cloudflare drops a connection after 125 s of silence.
# Between GATE.enter() and forward() nothing used to be written, so a queued
# remote caller was dropped without the gateway ever learning about it.
KEEPALIVE = int(env("QUEUE_KEEPALIVE", None, 30))

SSE_HEADERS = {"content-type": "text/event-stream",
               "cache-control": "no-cache",
               "connection": "keep-alive"}

def sse_error(status, text):
    """An error inside an already-running stream.

    Once the keep-alives have gone out, the HTTP status is spent — a later
    failure can only be delivered as an SSE event. That is what the Anthropic
    API does too, so Claude Code understands it.
    """
    payload = json.dumps({"type": "error",
                          "error": {"type": "api_error",
                                    "message": "upstream answered %d: %s"
                                               % (status, text[:200])}})
    return ("event: error\ndata: %s\n\n" % payload).encode()

async def enter_with_lifesign(prio, req, streaming):
    """Wait for a gate slot, and keep a streaming caller alive while waiting.

    Returns (seconds waited, prepared response or None). A prepared response
    means the headers are already out and forward() has to write into it.

    Non-streaming callers get nothing — there is no way to keep an ordinary
    response alive without committing to a status code, and they would have run
    into llama-server's own silence during a prefill anyway.
    """
    gate = asyncio.create_task(GATE.enter(prio))
    resp = None
    t0 = time.time()
    try:
        while True:
            try:
                await asyncio.wait_for(asyncio.shield(gate), timeout=KEEPALIVE)
                return time.time() - t0, resp
            except asyncio.TimeoutError:
                if not streaming:
                    continue
                if resp is None:
                    resp = web.StreamResponse(status=200, headers=SSE_HEADERS)
                    await resp.prepare(req)
                    log("WAITING     %-15s %-6s sign of life while queued"
                        % (req.remote or "?", PRIORITY_NAME[prio]))
                await resp.write(b":\n\n")
    except BaseException:
        # Same rule as in enter(): a slot we already hold has to go back.
        if gate.done() and not gate.cancelled() and gate.exception() is None:
            GATE.leave()
        else:
            gate.cancel()
        raise

def query_slots(wait=0):
    """Ask the server for the slot count, so it is not maintained twice.

    MAX_INFLIGHT should match the slot count: more concurrent requests than
    slots gains nothing, because llama.cpp serialises them anyway — they would
    only queue inside the server instead of in the gateway, where the priority
    ordering no longer applies.

    `wait` seconds of retrying, and it is not optional in practice. A single
    try with a 10 s timeout fails at EVERY BOOT: llama-server needs 30-100 s
    to load a model, systemd starts both at once, and the fallback is a
    default of 2. So after every reboot the gateway admitted a second request
    that llama.cpp then queued internally — where the priority ordering
    between local, LAN and remote no longer applies. Silent, and check.sh had
    been reporting it as "MAX_INFLIGHT is 2, the server has 1 slots" after
    every restart since the gateway existed.

    Waiting rather than guessing means the port opens late at boot. That is
    the better failure: during those seconds the model is not up either, so a
    request would fail anyway — and a refused connection is clearer than a
    queue with the wrong depth.
    """
    import urllib.request
    end = time.time() + wait
    while True:
        try:
            with urllib.request.urlopen(LLAMA + "/slots", timeout=10) as x:
                return len(json.loads(x.read().decode()))
        except Exception:
            if time.time() >= end:
                return None
            time.sleep(3)

# The slot count is asked for in main(), not at import time: an import must
# not reach out to the network, otherwise the tests under tests/ would depend
# on a llama-server that does not exist there.
GATE = PriorityGate(MAX_INFLIGHT)

# Prefix bookkeeping. The point is not statistics for their own sake but
# spotting the pathological case: two prefixes that share a long common start
# without being equal. Those fight over ONE slot permanently and destroy each
# other's cache — measured 88 % instead of 99 %, and with too few slots a
# total loss.
# See docs/measurements/measurements.md, section 8.14.
PREFIXES = {}            # full_hash -> dict(head_hash, requests, cold, warm, ...)
HEAD_BYTES = DIA.HEAD_BYTES   # part of the id contract, see dialects.py

# ----------------------------------------------------- cache correction ---
def split_volatile(text):
    """Split a system message into its stable and its volatile part.

    Claude Code appends the counter to the END of the otherwise stable
    agent-types block. Classifying the whole block as volatile means never
    hoisting it, which throws the effect away.
    """
    fund, stable = [], text
    for r in VOLATILE:
        fund.extend(r.findall(stable))
        stable = r.sub("", stable)
    return stable.rstrip(), fund

def blocks_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""

def correct(p, dialect=DIA.ANTHROPIC):
    """Hoist stable system messages to the front — see dialects.py."""
    return DIA.hoist_system_messages(p, dialect, VOLATILE)

def inject_model_kwargs(p, table=None):
    """Fill chat_template_kwargs from the model-name table.

    Existing request kwargs win key by key: whoever sets them explicitly has
    read the docs and gets what they asked for. The model name itself stays
    untouched — llama-server ignores it, and the log keeps showing which
    mode a consumer chose.
    """
    table = KWARGS_BY_MODEL if table is None else table
    wanted = table.get(p.get("model"))
    if not wanted:
        return p, False
    merged = dict(wanted)
    given = p.get("chat_template_kwargs")
    if isinstance(given, dict):
        merged.update(given)
    p["chat_template_kwargs"] = merged
    return p, True

def mid_system_to_user(p, dialect=DIA.ANTHROPIC):
    """Non-leading system messages become user messages — see dialects.py."""
    return DIA.mid_system_to_user(p, dialect)

def prefix_text(p, dialect=DIA.ANTHROPIC):
    """The stable start of a request as text: system head and tools."""
    return DIA.prefix_text(p, dialect)

def prefix_id(p, dialect=DIA.ANTHROPIC):
    """Rough id of the stable prompt start — for the cold/warm estimate.

    Delegated to dialects.py so that prewarm.py computes the SAME id from
    the same body; the two used to drift apart, and a saved prefix under a
    key nobody produces is never found again; tests/test_gateway.py and
    tests/test_dialects.py pin the contract that broke.
    """
    return DIA.prefix_id(p, dialect, HEAD_BYTES)

# ------------------------------------------------------------ forwarding ---
# What a remote caller may reach at all. Everything else gets a 404 — including
# what llama-server offers.
#
# The reason is measured: without this list the gateway passed every path
# through to llama-server unchecked. Over the tunnel this was reachable:
#   /completion, /v1/chat/completions  free inference, token bypassed entirely
#   /slots                             complete prompts of every slot
#   /props, /v1/models, /health        server configuration
# See docs/SECURITY.md.
REMOTE_ALLOWED = (
    "/v1/messages",
    "/v1/messages/count_tokens",
    # The OpenAI dialect of the same inference, for agents that speak it
    # (DeepSeek Harness and most others). It is on the list for the same
    # reason /v1/messages is — it costs a token and a slot, and the zone
    # rules apply to it identically. What must NOT be here is /completion:
    # that one takes a raw prompt, bypasses the chat template, and was the
    # free-inference hole described in docs/SECURITY.md.
    "/v1/chat/completions",
    "/v1/models",
)

def is_inference(path):
    return DIA.is_inference(path)

def remote_allowed(path):
    """Allow list, exact path comparison without the query."""
    rein = path.split("?", 1)[0].rstrip("/") or "/"
    return rein in REMOTE_ALLOWED

def token_owner(req, prio):
    """Return the consumer name, or None if not allowed."""
    if prio == 0:
        return "local"                  # local is always allowed
    if not TOKENS:
        return None                     # without tokens nothing from outside
    auth = req.headers.get("authorization", "")
    secret = auth[7:] if auth.startswith("Bearer ") else req.headers.get("x-api-key", "")
    return TOKENS.get(secret)

async def handler(req):
    ip = req.remote or "?"
    prio = zone(req)
    inference = is_inference(req.path)
    dialect = DIA.detect(req.path)

    if prio != 0:
        # Remote: allow list first, then token. The order is deliberate — a
        # 404 does not reveal whether the path exists.
        if not remote_allowed(req.path):
            log("BLOCKED     %-15s %-6s %s" % (ip, PRIORITY_NAME[prio], req.path[:60]))
            return web.json_response({"error": "not found"}, status=404)
        who = token_owner(req, prio)
        if not who:
            # Diagnostics that help while setting up, without ever logging an
            # access value: only which headers were there and how long.
            #
            # Important, because foreign credentials can arrive here: a signed-in
            # Claude Code sends its subscription OAuth token (sk-ant-oat01…,
            # 108 characters) instead of ANTHROPIC_AUTH_TOKEN. That is exactly
            # why no value is ever logged here, not even a prefix of one.
            present = [k for k in ("authorization", "x-api-key", "anthropic-auth-token")
                         if req.headers.get(k)]
            a = req.headers.get("authorization", "")
            shape = " (%d chars)" % len(a) if a else ""
            log("REJECTED    %-15s %-6s no valid token  %s  headers=%s%s"
                % (ip, PRIORITY_NAME[prio], req.path[:30], present or "none", shape))
            return web.json_response(
                {"type": "error", "error": {"type": "authentication_error",
                                            "message": "a valid token is required"}},
                status=401)

    body = await req.read()
    out = None
    ident = None
    cold = False
    streaming = False

    if inference and body:
        try:
            p = json.loads(body)
            for k in DROP:
                p.pop(k, None)
            streaming = bool(p.get("stream"))
            # The id comes from the RAW body, before any correction — the
            # store is keyed by what arrives, not by what is forwarded.
            ident, head = prefix_id(p, dialect)
            cold = ident not in PREFIXES
            p, n_vol = correct(p, dialect)
            if MID_SYSTEM_TO_USER:
                p, _ = mid_system_to_user(p, dialect)
            p, _ = inject_model_kwargs(p)
            out = json.dumps(p).encode()
        except Exception as e:
            log("passed through unchanged: %r" % (e,))
            out = None

    if not inference:
        if req.path.rstrip("/") == "/v1/models" and KWARGS_BY_MODEL:
            return await models_with_aliases(req, body)
        return await forward(req, body, out)

    depth = GATE.depth()
    if cold and depth > 0:
        log("NOTE        cold prefix %s with %d waiting — blocks the others"
            % (ident, depth))

    if cold and ident:
        # Collision warning: is there already a prefix with the same head?
        rivals = [k for k, v in PREFIXES.items()
                         if v["head"] == head and k != ident]
        if rivals:
            log("WARNING     prefix %s shares its head with %s — the two "
                "fight over one slot and destroy each other's cache "
                "(see docs/CONSUMERS.md)"
                % (ident, ", ".join(rivals)))

    if prio == 0:
        who = "local"
    # Limit per consumer: keeps one of them from occupying the GPU alone.
    if who != "local" and IN_FLIGHT_PER_TOKEN.get(who, 0) >= PER_TOKEN_MAX:
        log("THROTTLED   %-15s %-6s %s already has %d in flight"
            % (ip, PRIORITY_NAME[prio], who, PER_TOKEN_MAX))
        return web.json_response(
            {"type": "error", "error": {"type": "rate_limit_error",
                                        "message": "at most %d concurrent requests per access"
                                                   % PER_TOKEN_MAX}},
            status=429)
    IN_FLIGHT_PER_TOKEN[who] = IN_FLIGHT_PER_TOKEN.get(who, 0) + 1
    # From the increment on, the counter must come back down on EVERY path,
    # even if the caller aborts — aiohttp cancels the handler as soon as the
    # connection is gone. The increment used to sit before the try: a Ctrl-C
    # while queued (100-180 s during a cold start) left it standing, and after
    # PER_TOKEN_MAX such cases the access got nothing but 429.
    early = None
    try:
        waited, early = await enter_with_lifesign(prio, req, streaming)
    except BaseException:
        # A slot already obtained is given back by enter_with_lifesign itself,
        # only the counter is missing here.
        IN_FLIGHT_PER_TOKEN[who] = max(0, IN_FLIGHT_PER_TOKEN.get(who, 1) - 1)
        raise
    # Set BEFORE the try, because the finally reads them and an exception can
    # be raised at any point inside.
    answered = {"ok": False}
    was_cold = False
    t_start = time.time()
    try:
        # Only now reload: the slot is secured, so a llama slot is free too.
        # This belongs INSIDE the try: restore_from_disk catches only
        # Exception, and CancelledError is not one — an abort during the
        # restore (up to 300 s) used to make the gate slot seep away, and
        # MAX_INFLIGHT dropped by one for the rest of the service's life.
        if cold and ident in SAVED:
            if await restore_from_disk(ident):
                cold = False
        t_start = time.time()
        if ident:
            record_use(ident)
            e = PREFIXES.setdefault(ident, {
                "head": head, "requests": 0, "cold": 0, "warm": 0,
                "took_sum": 0.0, "last": 0.0, "sources": set()})
            e["requests"] += 1
            e["cold" if cold else "warm"] += 1
            e["sources"].add(ip)
            e.setdefault("consumers", set()).add(who)
        log("START       %-15s %-6s who=%-12s prefix=%s %s waited=%.1fs queue=%d%s"
            % (ip, PRIORITY_NAME[prio], who, ident, "COLD" if cold else "warm",
               waited, depth, "  kept-alive" if early is not None else ""))
        was_cold = cold
        return await forward(req, body, out, early, answered)
    finally:
        # EVERYTHING that belongs after the answer happens HERE, not after the
        # forward() call, and this is not style.
        #
        # RUNNER_KWARGS carries handler_cancellation=True so that a caller who
        # vanishes frees the gate slot at once (TestClientAbort). Under
        # aiohttp 3.13 that cancellation also arrives when the client closes
        # NORMALLY — right after it has read the answer. Measured on the live
        # stack 26.08. with two identical requests:
        #
        #     curl, closes at once          START, no DONE, nothing saved
        #     connection held open 5 s      START, DONE took=0.8s
        #
        # So every short-lived consumer silently lost the one thing that turns
        # a 100-180 s cold start into 1.4 s, and the LRU that prefix-cleanup
        # evicts by was never updated for them either. tests/live_prefix.sh had
        # been failing on exactly this.
        #
        # `answered` is the honest signal: the upstream answer arrived in full.
        # A caller who leaves DURING the prefill leaves it false, and then
        # there is nothing worth saving — a partial prefix would later restore
        # as if it were whole.
        if answered["ok"]:
            took = time.time() - t_start
            big_enough = (out is not None
                           and len(prefix_text(p, dialect)) >= AUTO_MIN_CHARS)
            if (AUTO_SAVE and was_cold and ident and ident not in SAVED
                    and big_enough):
                try:
                    # An independent task: it must outlive this handler, which
                    # may be in the middle of being cancelled right now.
                    asyncio.create_task(auto_save(ident, json.loads(out), dialect))
                except Exception as e:
                    log("NOTE        save not scheduled: %r" % (e,))
            if ident:
                PREFIXES[ident]["took_sum"] += took
                PREFIXES[ident]["last"] = time.time()
            log("DONE        %-15s %-6s who=%-12s prefix=%s took=%.1fs"
                % (ip, PRIORITY_NAME[prio], who, ident, took))
        else:
            log("ABORTED     %-15s %-6s who=%-12s prefix=%s after=%.1fs — no "
                "answer, nothing saved"
                % (ip, PRIORITY_NAME[prio], who, ident, time.time() - t_start))
        GATE.leave()
        IN_FLIGHT_PER_TOKEN[who] = max(0, IN_FLIGHT_PER_TOKEN.get(who, 1) - 1)

def add_aliases(listing, table):
    """Add one entry per configured model name to a /v1/models listing.

    llama-server only knows its own --alias, so a consumer listing the
    models sees `qwen38` and never learns that `qwen38-think` and
    `qwen38-deep` exist — they are the gateway's doing, not the server's.
    A client that builds its model picker from this listing (dsh does)
    would offer only one of the three.

    Each alias is the served entry copied, with the name replaced and a
    description that says which thinking level it selects. Copying rather
    than inventing keeps whatever the server reports about itself —
    n_ctx, capabilities, multimodality — true for the aliases too, because
    it IS the same loaded model.
    """
    for key in DIA.model_listing_arrays(listing):
        entries = listing[key]
        have = {e.get("name") or e.get("id")
                for e in entries if isinstance(e, dict)}
        base = entries[0]
        for name, kwargs in table.items():
            if name in have:
                continue
            e = json.loads(json.dumps(base))
            for field in ("name", "id", "model"):
                if field in e:
                    e[field] = name
            effort = kwargs.get("reasoning_effort")
            if kwargs.get("enable_thinking") is False:
                e["description"] = "thinking off"
            elif effort:
                e["description"] = "thinking: %s" % effort
            entries.append(e)
        listing[key] = entries
    return listing

async def models_with_aliases(req, body):
    """Pass /v1/models through, then add the names the gateway serves."""
    hdrs = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
    try:
        async with ClientSession(timeout=ClientTimeout(total=30)) as s:
            async with s.request(req.method, LLAMA + req.path_qs,
                                 data=body or None, headers=hdrs) as up:
                if up.status != 200:
                    return web.Response(status=up.status, body=await up.read(),
                                        content_type=up.content_type)
                listing = await up.json()
    except Exception as e:
        log("NOTE        /v1/models could not be extended: %r" % (e,))
        return await forward(req, body, None)
    return web.json_response(add_aliases(listing, KWARGS_BY_MODEL))

async def forward(req, body, out, resp=None, answered=None):
    """Pass the request through. `resp` is an already-prepared response.

    It is set when the caller was kept alive while queued: the headers went out
    before the upstream status was known, so a later failure can only be
    reported inside the stream.

    `answered` is a one-key dict the caller owns. It is set the moment the
    UPSTREAM answer has been received in full — before write_eof, because at
    that point the model is done and the slot holds the prefix, which is the
    only precondition that matters for saving it. The caller cannot infer this
    from a normal return: with handler_cancellation the return may never
    happen. See the note in the handler's finally.
    """
    hdrs = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
    timeout = ClientTimeout(total=None, sock_read=SILENCE_MAX, sock_connect=30)
    async with ClientSession(timeout=timeout, auto_decompress=False) as s:
        async with s.request(req.method, LLAMA + req.path_qs,
                             data=(out if out is not None else body),
                             headers=hdrs, allow_redirects=False) as up:
            if resp is None:
                rh = {k: v for k, v in up.headers.items()
                      if k.lower() not in HOP and k.lower() != "content-encoding"}
                resp = web.StreamResponse(status=up.status, headers=rh)
                await resp.prepare(req)
            elif up.status != 200:
                log("NOTE        upstream %d after the headers were already out"
                    % up.status)
                await resp.write(sse_error(up.status, (await up.text())))
                await resp.write_eof()
                return resp
            async for ch in up.content.iter_any():   # no buffering -> SSE stays intact
                await resp.write(ch)
            if answered is not None:
                answered["ok"] = True    # the model is done; the slot holds it
            await resp.write_eof()
            return resp

async def status(req):
    if zone(req) != 0:
        return web.json_response({"error": "local only"}, status=403)
    return web.json_response({
        "in_flight": MAX_INFLIGHT - GATE.free,
        "max_concurrent": MAX_INFLIGHT,
        # Published so a test asserts against the CONFIGURED limit instead
        # of a hard-wired 2 — raising it used to make live_concurrency.sh
        # report a deviation that was really just the new setting.
        "per_token_max": PER_TOKEN_MAX,
        "queue": GATE.depth(),
        "overtaken_by_age": GATE.overtaken,
        "prefixes": sorted(
            [{"id": k, "head": v["head"], "requests": v["requests"],
              "cold": v["cold"], "warm": v["warm"],
              "warm_pct": round(100.0 * v["warm"] / v["requests"], 1) if v["requests"] else 0.0,
              "avg_seconds": round(v["took_sum"] / v["requests"], 2) if v["requests"] else 0.0,
              "sources": sorted(v["sources"]),
              "consumers": sorted(v.get("consumers", []))}
             for k, v in PREFIXES.items()],
            key=lambda d: -d["requests"]),
        "collisions": [
            sorted(g) for g in
            ({tuple(sorted(k for k, v in PREFIXES.items() if v["head"] == head))
              for head in {v["head"] for v in PREFIXES.values()}})
            if len(g) > 1],
        "llama": LLAMA,
        "uptime_seconds": round(time.time() - T0, 1),
    })

def build_app():
    """A fresh application per call.

    An aiohttp application can only be bound to one event loop; the tests
    start several in a row, each in its own.
    """
    a = web.Application(client_max_size=1024**3)
    a.router.add_get("/gateway/status", status)
    a.router.add_route("*", "/{tail:.*}", handler)
    return a

app = build_app()

async def watch_server():
    """Notice when llama-server was restarted, and then forget everything.

    Needed because "cold" so far only meant: the gateway has never seen this
    prefix. After a server restart the slots are empty while the gateway still
    considers the prefix warm — it then does not load it from disk, and the
    request runs into a full cold start. Exactly that happened in testing:
    109.7 s although the file was ready.
    """
    global PREFIXES
    was_gone = False
    while True:
        await asyncio.sleep(15)
        refresh_saved()
        try:
            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as s_:
                async with s_.get(LLAMA + "/slots") as r:
                    slots = await r.json()
            if was_gone:
                if PREFIXES:
                    log("NOTE        llama-server was gone — prefix bookkeeping reset "
                        "so that disk loading kicks in again")
                    PREFIXES = {}
                was_gone = False
            elif PREFIXES and not any(x.get("n_prompt_tokens") for x in slots):
                # All slots empty although we know prefixes: restarted or
                # cleared from outside.
                log("NOTE        all slots empty — prefix bookkeeping reset")
                PREFIXES = {}
        except Exception:
            was_gone = True

# The accounting above relies on "aiohttp cancels the handler as soon as the
# connection is gone" — which stopped being the DEFAULT in aiohttp 3.9: it is
# opt-in now. Observed 25.08. without it: a client that timed out mid-request
# left the gateway silently waiting for the complete upstream answer, holding
# its gate slot and per-token counter for the whole ~10-minute generation —
# the consumer saw nothing but 429s from a machine that was working for
# nobody. The handlers are cancellation-safe by construction (every counter
# comes back in a finally, the queue path handles CancelledError explicitly,
# auto_save runs as its own task), so switching the old behaviour back on is
# exactly right. Shared with the tests via RUNNER_KWARGS so they run under
# the same regime. Residual limit: llama-server itself still finishes a
# NON-streaming generation for a vanished caller; only streaming requests
# stop at the next chunk.
RUNNER_KWARGS = {"handler_cancellation": True}

async def serve():
    asyncio.create_task(watch_server())
    runner = web.AppRunner(app, **RUNNER_KWARGS)
    await runner.setup()
    for address in BIND:
        await web.TCPSite(runner, address, PORT).start()
    if TUNNEL_PORT:
        for address in TUNNEL_BIND:
            await web.TCPSite(runner, address, TUNNEL_PORT).start()
    while True:
        await asyncio.sleep(3600)

def main():
    """Everything with a side effect — only on a direct call.

    A module import therefore stays free of consequences: no token file, no
    store, no network. That is exactly what the tests under tests/ need, which
    must run without a GPU and without a running service.
    """
    global TOKENS, SAVED, MAX_INFLIGHT, GATE
    if "MAX_INFLIGHT" not in os.environ:
        n = query_slots(wait=SLOTS_WAIT)
        if n:
            MAX_INFLIGHT = n
            GATE = PriorityGate(MAX_INFLIGHT)
        else:
            log("  WARNING: the server did not answer within %d s. Falling back "
                "to MAX_INFLIGHT=%d, which is a GUESS — if the slot count is "
                "lower, the admission control is bypassed for the difference. "
                "Restart this service once the model is up."
                % (SLOTS_WAIT, MAX_INFLIGHT))
    TOKENS = load_tokens()
    SAVED = refresh_saved(force=True)
    log("cc-gateway on %s:%d -> %s" % (",".join(BIND), PORT, LLAMA))
    log("  saved prefixes on disk: %d (%.1f GB)" % (len(SAVED), disk_used_gb()))
    log("  automatic saving: %s%s"
        % ("on" if AUTO_SAVE else "off",
           ", limit %g GB" % AUTO_MAX_GB if AUTO_SAVE else ""))
    log("  access: %s" % (", ".join(sorted(set(TOKENS.values()))) or "NONE — reachable locally only"))
    if TUNNEL_PORT:
        log("  tunnel port %s:%d — everything from there counts as 'remote'"
            % (",".join(TUNNEL_BIND), TUNNEL_PORT))
    log("  at most %d concurrent overall, %d per access · silence limit %d s"
        % (MAX_INFLIGHT, PER_TOKEN_MAX, SILENCE_MAX))
    log("  queue: served by priority, but nobody waits longer than %d s; "
        "a queued stream gets a sign of life every %d s"
        % (AGE_AFTER, KEEPALIVE))
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
