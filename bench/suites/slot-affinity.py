#!/usr/bin/env python3
"""slot-affinity — which slot does the second prefix take, and what does the
first one pay for it?

THE QUESTION, asked 05.09.2026 by a workload rather than by a defect. An
agent harness runs a main agent beside a sub-agent: two conversations, one
project, therefore two live prefixes that share a long head (system prompt
plus tool schemas — ~19k tokens for Claude Code, of which ~16.8k are the tool
block) and diverge after it. One slot is the worst possible shape for that,
and the day's earlier measurements established that a second slot is not
corrupting on this build. This suite asks the OTHER half: does a second slot
actually help.

WHY IT MIGHT NOT, and it is not a hypothesis about llama.cpp so much as a
reading of it. get_available_slot() (server-context.cpp) picks by longest
common prefix among slots that are not processing — and it SKIPS EMPTY ONES:

    // skip the slot if it does not contains cached tokens
    if (tokens.empty()) { ... continue; }

So with A resident in slot 0 and slot 1 empty, a request for B — which shares
the whole head with A — can score above the -sps threshold (default 0.10)
against slot 0 and take it, while the empty slot is never considered. Two
slots, one of them permanently idle, and the two prefixes destroying each
other exactly as they did with one. docs/CONSUMERS.md already measures that
collision for a different cause: 88 % hit rate instead of 99 %, 14 s per turn
instead of 1.6.

READING IS NOT MEASURING, which is why this file exists. The arms, same
conversation in each:

    np1             -np 1                      what is served today
    np2             -np 2                      a second slot, nothing else
    np2-sps         -np 2 -sps 0.95            a flag instead of gateway work
    np2-pin         -np 2, id_slot per request the gateway change, simulated
    np2-pin-cram    the same at --cram N       what the RAM cache is worth
                                              (0 = off; the arm is named
                                              after the value it ran at)

MORE THAN TWO AGENTS, added 05.09.2026 because the operator asked and none of
the first three runs could answer it. A main agent that drives SEVERAL
sub-agents cannot be extrapolated from two: the subs have to share the second
slot, so every sub-agent turn evicts the previous one. Whether that is
affordable is not a question about slots at all — it is a question about the
RAM prompt cache, because an evicted state either comes back out of -cram or
is recomputed from nothing.

    --pattern ABACAD    a main agent between every sub-agent
    --agents follows the pattern; the letters in it ARE the agents

The pin strategy generalises the way a gateway would have to: the FIRST agent
keeps slot 0 to itself and every other one shares slot 1. That protects the
most expensive context and lets the cheap ones rotate. `np2-pin-nocram` is
that arm with the cache switched off and nothing else changed, so the gap
between the two is attributable to the cache alone.

A WARNING THE MEASUREMENTS PAID FOR. `--pattern` defaults to AB, and AB
flatters -sps: when no slot clears the threshold llama.cpp falls back to the
least recently used one, which under strict alternation is always the
caller's own. Run …_1604 therefore showed -sps equal to pinning, and run
…_1615 (pattern AAB) showed it costing 3.3x. A schedule that alternates
perfectly is not one anybody runs.

The last arm also answers a question of its own: llama.cpp reads `id_slot`
out of the request body (server-context.cpp, task.id_slot = json_value(data,
"id_slot", -1)) at a point that looks like it precedes the OAI-compat branch.
Whether it therefore works through /v1/chat/completions is READ, not known,
and an arm that pins and lands somewhere else says so.

THE METRIC is not throughput. It is RECOMPUTED TOKENS PER TURN —
prompt_tokens minus cached_tokens, straight out of the server's own usage
block. That is what an agent pays when its partner takes the slot, and it is
the number the whole arrangement lives or dies by. Seconds are recorded
beside it because the RAM prompt cache (-cram) can absorb an eviction, and
absorbing it is not the same as avoiding it: the tokens come back cheaper,
not free.

EVERY ARM VERIFIES ITS OWN PREMISE. An arm named np2 that came up with one
slot is not a clean reading, it is a broken one — that is the lesson this
repository paid for twice in one day (restore-safety.py's --env, and
slot-corruption.py's own docstring). The slot count is read back off /slots
and recorded per arm; a mismatch marks the arm and never reads as a result.

    python3 bench/suites/slot-affinity.py --env setup/env/qwen36.env

Written 05.09.2026.
"""
import argparse, json, os, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(BENCH)
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(REPO, "setup", "lib"))
import run as runlib                                          # noqa: E402
import sweep                                                  # noqa: E402
import systemdfile                                            # noqa: E402

PORT = 8081
URL = "http://127.0.0.1:%d" % PORT
SLOT_DIR = os.path.expanduser("~/.cache/llama-slots")
# How long an answer may be. Set from --max-tokens in main(). It matters for
# the parallel mode and almost not at all for the sequential one: with a short
# answer the wall clock is nearly all prefill, and decode sharing — the thing
# two slots are supposed to buy — would be too small a term to see.
MAX_TOKENS = 48


def server_argv(env_path, np, sps=None, cram=None, port=PORT):
    """The profile's own flags, with the slot count and -sps under control.

    Deliberately NOT shared with restore-safety.py's base_from_profile(),
    which adds --slot-save-path because its cells are about restoring. This
    suite must not save or restore anything: a saved prefix landing in a slot
    is a second mechanism moving the very thing being measured.
    """
    argv, i, out = systemdfile.llama_args(env_path), 0, []
    dropped = {}
    while i < len(argv):
        tok, nxt = argv[i], argv[i + 1] if i + 1 < len(argv) else None
        takes_value = nxt is not None and not nxt.startswith("-")
        drop = ["-np", "--parallel", "-sps", "--slot-prompt-similarity",
                "--port", "--host"]
        if cram is not None:
            drop += ["-cram", "--cache-ram"]
        if tok in drop:
            if takes_value:
                dropped[tok] = nxt
                i += 1
        else:
            out.append(tok)
        i += 1
    out += ["-np", str(np)]
    if sps is not None:
        out += ["-sps", str(sps)]
    if cram is not None:
        out += ["-cram", str(cram)]
    if dropped:
        print("   profile flags overridden: %s"
              % ", ".join("%s %s" % kv for kv in sorted(dropped.items())))
    return out + ["--host", "127.0.0.1", "--port", str(port)]


def head_and_body(which, head_reps, body_reps):
    """A long shared head and a divergent body — the shape of two agents in
    one project.

    The head is IDENTICAL across A and B down to the character, because that
    is what a shared system prompt and a shared tool block are. Everything
    that differs sits behind it, which is exactly the condition under which
    the LCP selection has to choose.
    """
    head = ("You have access to the following tools. "
            "Tool: read_file(path) - reads a file from the repository. "
            "Tool: write_file(path, content) - writes a file. "
            "Tool: search(pattern) - searches the repository. ") * head_reps
    body = ("Context for agent %s: this agent owns task %s and must not "
            "touch the other one's files. " % (which, which)) * body_reps
    return head, body


def ask(messages, max_tokens=MAX_TOKENS, id_slot=None, timeout=900):
    payload = {"model": "affinity", "stream": False, "temperature": 0,
               "max_tokens": max_tokens, "messages": messages}
    if id_slot is not None:
        payload["id_slot"] = id_slot
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    return out, time.time() - t0


def usage_of(out):
    """prompt / cached / recomputed, from the server's own usage block.

    `cached_tokens` is what llama.cpp reports it did NOT have to compute.
    Absent rather than zero on a server that does not report it — and the
    difference matters, because a missing field read as 0 would turn every
    warm turn into a fabricated cold one.
    """
    u = out.get("usage") or {}
    prompt = u.get("prompt_tokens")
    details = u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    rec = None
    if prompt is not None and cached is not None:
        rec = prompt - cached
    return {"prompt_tokens": prompt, "cached_tokens": cached,
            "recomputed": rec, "id_slot": out.get("id_slot")}


def slots_snapshot():
    """What each slot holds right now. The server is idle between turns, so
    this is a reading rather than a race — and it is the witness for `id_slot`
    when the response does not carry it."""
    try:
        with urllib.request.urlopen(URL + "/slots", timeout=10) as x:
            slots = json.loads(x.read().decode())
    except Exception as e:                                    # noqa: BLE001
        return {"error": "%s: %s" % (type(e).__name__, e)}
    return {"count": len(slots),
            "holding": [s.get("n_prompt_tokens") for s in slots]}



def slot_action(slot, action, filename, timeout=900):
    """save / restore / erase on ONE slot, straight at the server.

    The gateway can only do this on slot 0 — the path is hard-wired in
    session_save() and session_restore() (gateway.py, 05.09.2026). The SERVER
    has no such limit, so this is the same trick the pin arm uses: perform
    what a fixed gateway would do, and measure it before writing the fix.
    """
    req = urllib.request.Request(
        URL + "/slots/%d?action=%s" % (slot, action),
        data=json.dumps({"filename": filename}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    return out, time.time() - t0


def run_arm(name, argv, binary, dest, turns, head_reps, body_reps, pin,
            growth_reps=0, pattern="AB", mode="sequential", persist=False):
    """One arm against one fresh server. An arm that fails is recorded whole,
    because the arms are only worth anything as a set."""
    rec = {"arm": name, "argv": [systemdfile.unexpand(x) for x in argv],
           "pinned": pin, "growth_reps": growth_reps,
           "pattern": pattern, "mode": mode, "persist": persist,
           "turns": []}
    log = os.path.join(dest, "%s.log" % name)
    before = runlib._gtt("used")
    proc = None
    try:
        proc = runlib.start_server(argv, log, binary)
        if not sweep.slots_ready(URL, 420):
            rec["error"] = "/slots never answered"
            return rec

        # THE PREMISE, read back rather than assumed. An arm named np2 that
        # came up with one slot must not be able to report a result.
        snap = slots_snapshot()
        rec["slots_seen"] = snap.get("count")
        want = int(argv[argv.index("-np") + 1])
        rec["premise_holds"] = (snap.get("count") == want)
        if not rec["premise_holds"]:
            rec["error"] = ("asked for %d slots, server came up with %s"
                            % (want, snap.get("count")))
            print("   PREMISE FAILED — %s" % rec["error"])
            return rec

        agents = sorted(set(pattern))
        head, _ = head_and_body(agents[0], head_reps, body_reps)
        convo = {}
        for which in agents:
            _, body = head_and_body(which, head_reps, body_reps)
            convo[which] = [{"role": "system", "content": head + body}]

        # The ORDER is a variable, and the run of …_1604 showed why. With a
        # strict A,B,A,B the LRU fallback lands on the right slot by itself:
        # whenever no slot clears the -sps threshold, the least recently used
        # one IS the caller's own. That made -sps look equal to pinning, and
        # it is an artefact of the alternation rather than a property of the
        # flag. A main agent that checks on its sub-agent, or prepares the
        # next one, does not alternate — and then the oldest slot is somebody
        # else's.
        nth = {c: 0 for c in agents}
        lock = threading.Lock()

        def one_turn(which, quiet=False):
            """One agent's turn. Safe to call from a thread: everything shared
            is either per-agent or taken under the lock."""
            with lock:
                nth[which] += 1
                t = nth[which]
            # The filler stands in for a tool result, and it is what makes
            # this a fair test of -sps rather than a flattering one. The
            # threshold is a FRACTION of the incoming prompt, so a turn
            # that adds little keeps the fraction near 1.0 and any
            # threshold passes. Real agent turns append tool output. It
            # must differ per turn AND per agent, or it would join the
            # common prefix and add nothing to measure.
            filler = ("Tool result %d for agent %s: the file defines a "
                      "handler and carries comments about it. "
                      % (t, which)) * growth_reps
            convo[which].append(
                {"role": "user",
                 "content": filler +
                            "Turn %d for agent %s. Reply with the single "
                            "word %s%d and nothing else." % (t, which, which, t)})
            # THE STRATEGY, and it is the one a gateway could
            # implement: the first agent keeps slot 0 to itself, every
            # other one shares slot 1. With two agents that is one each;
            # with four it protects the most expensive context — the main
            # agent's — and lets the cheap ones rotate. Whether rotating
            # is cheap is what the -cram arm answers.
            slot = (0 if which == agents[0] else 1) if pin else None
            # FILE PERSISTENCE, simulated for the agents that SHARE a slot.
            # The first agent owns slot 0 and never loses its state, so it
            # needs neither call; everyone else is evicted by the next
            # sub-agent and is exactly who a fixed gateway would save.
            # Restore first (nothing to restore on the first turn, which the
            # server answers with an error we record rather than raise), the
            # turn, then the save.
            rst = sav = None
            rst_reply = sav_reply = None
            held_after_restore = None
            share = persist and slot == 1
            if share:
                try:
                    rst_reply, rst = slot_action(
                        slot, "restore", "affinity-%s.bin" % which)
                    # WHAT THE CALL ACTUALLY DID, because a 200 is not an
                    # answer. The first version of this arm threw the reply
                    # away and read `cached=0` afterwards with no way to tell
                    # a restore that failed from one that worked and was
                    # ignored. llama-server reports n_restored/n_read here,
                    # and /slots says what the slot holds now — one of the two
                    # has to move, or the restore did nothing.
                    held_after_restore = slots_snapshot().get("holding")
                except Exception as e:                        # noqa: BLE001
                    rst = "none: %s" % type(e).__name__
            out, secs = ask(convo[which], id_slot=slot)
            msg = ((out.get("choices") or [{}])[0].get("message") or {})
            convo[which].append({"role": "assistant",
                                 "content": msg.get("content") or ""})
            if share:
                try:
                    sav_reply, sav = slot_action(
                        slot, "save", "affinity-%s.bin" % which)
                except Exception as e:                        # noqa: BLE001
                    sav = "failed: %s" % type(e).__name__
            u = usage_of(out)
            u.update(turn=t, agent=which, seconds=round(secs, 2),
                     restore_s=round(rst, 2) if isinstance(rst, float) else rst,
                     save_s=round(sav, 2) if isinstance(sav, float) else sav,
                     restore_reply=rst_reply, save_reply=sav_reply,
                     held_after_restore=held_after_restore,
                     text=(msg.get("content") or "")[:40], concurrent=not quiet)
            with lock:
                rec["turns"].append(u)
                print("   %s t%d  slot=%-4s prompt=%-6s cached=%-6s "
                      "recomputed=%-6s %5.1fs"
                      % (which, t, u["id_slot"], u["prompt_tokens"],
                         u["cached_tokens"], u["recomputed"], secs)
                      + ("  restore=%s save=%s" % (u["restore_s"], u["save_s"])
                         if share else ""))
            return u

        if mode == "parallel":
            # WHAT THIS MODE IS FOR, and it is the question the whole two-slot
            # discussion started from: not "does a prefix survive" but "do two
            # requests AT THE SAME TIME finish sooner". Every other mode here
            # is sequential and cannot answer it.
            #
            # Phase 1 warms every agent one at a time. With one slot only the
            # last one can still be resident afterwards — that is not a flaw
            # of the warm-up, it IS the one-slot condition, and the recorded
            # cached_tokens say which agents kept anything.
            print("   -- warm-up, one at a time")
            for which in agents:
                one_turn(which, quiet=True)
            for cycle in range(1, turns + 1):
                print("   -- round %d, all %d at once" % (cycle, len(agents)))
                t0 = time.time()
                threads = [threading.Thread(target=one_turn, args=(w,))
                           for w in agents]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
                wall = round(time.time() - t0, 2)
                rec.setdefault("rounds", []).append(
                    {"cycle": cycle, "agents": len(agents), "wall": wall})
                print("   -- round %d wall clock %.2f s" % (cycle, wall))
        else:
            for cycle in range(1, turns + 1):
                for which in pattern:
                    one_turn(which)
                    rec["turns"][-1]["slots"] = slots_snapshot()
    except (Exception, SystemExit) as e:                      # noqa: BLE001
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        print("   ABORTED %s" % rec["error"])
    finally:
        # The state files are large (597 MiB each at 25k tokens) and belong to
        # nobody after the run. Named with a prefix of their own so this can
        # never reach a production session file.
        for f in os.listdir(SLOT_DIR) if os.path.isdir(SLOT_DIR) else []:
            if f.startswith("affinity-"):
                try:
                    os.unlink(os.path.join(SLOT_DIR, f))
                except OSError:
                    pass
        if proc is not None:
            sweep.stop_server(proc)
            try:
                runlib.wait_for_gtt_release(before)
            except Exception as e:                            # noqa: BLE001
                print("   (gtt wait: %s)" % e)
    return rec


def steady_state(rec):
    """Recomputed tokens per turn once both agents are established.

    Turn 1 is excluded on purpose: both prefixes are cold there and every arm
    pays the same, so including it would flatter exactly the arms that go on
    to thrash. What an agent pays FOREVER is the number from turn 2 on.
    """
    later = [t for t in rec.get("turns", [])
             if t.get("turn", 0) > 1 and t.get("recomputed") is not None]
    if not later:
        return None
    return {"turns": len(later),
            "recomputed_total": sum(t["recomputed"] for t in later),
            "recomputed_median": sorted(t["recomputed"] for t in later)[len(later) // 2],
            "seconds_total": round(sum(t["seconds"] for t in later), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="a setup/env/*.env profile")
    ap.add_argument("--binary", help="override the profile's LLAMA_BIN")
    ap.add_argument("--turns", type=int, default=4)
    # The two defaults are MEASURED rather than estimated, on the smoke run of
    # 05.09.2026 (report …_1549): head 40 + body 10 rendered to 2,035 prompt
    # tokens, i.e. 4.26 characters per token on these strings. The head
    # repetition is 195 characters and the body one 86, which puts ~16k tokens
    # of shared head at 350 and ~8k of divergent body at 396. A first guess of
    # 1400/350 sat here and would have built a 70k head — four times the
    # intended size and slow enough to change what the run could cover.
    ap.add_argument("--head-reps", type=int, default=350,
                    help="repetitions in the SHARED head (~16k tokens, the "
                         "size of a real tool block)")
    ap.add_argument("--body-reps", type=int, default=396,
                    help="repetitions in the divergent body (~8k tokens). "
                         "Head/total decides the LCP fraction the selection "
                         "sees, so it is a variable rather than a constant")
    # ~21 tokens per repetition (89 characters at the 4.26 char/token rate
    # measured above), so 95 is ~2k tokens of tool output on every turn.
    # Zero is the run of …_1550, which grew by 30 tokens a turn and therefore
    # could not tell -sps apart from pinning.
    ap.add_argument("--persist", action="store_true",
                    help="simulate the gateway's session persistence for the "
                         "agents that SHARE slot 1: restore before the turn, "
                         "save after it. Answers whether the file layer "
                         "catches what a too-small -cram drops")
    ap.add_argument("--max-tokens", type=int, default=48,
                    help="answer length. Raise it for --mode parallel, where "
                         "decode is the term under test")
    ap.add_argument("--mode", choices=("sequential", "parallel"),
                    default="sequential",
                    help="sequential asks one agent at a time and measures "
                         "who kept their prefix; parallel fires every agent "
                         "AT ONCE and measures the wall clock until all are "
                         "done. Only the second one answers whether two "
                         "requests finish sooner together")
    ap.add_argument("--pattern", default="AB",
                    help="turn order within one cycle, e.g. AAB for a main "
                         "agent that takes two turns per sub-agent turn. "
                         "Anything but strict alternation is where the LRU "
                         "fallback stops helping by accident")
    ap.add_argument("--growth-reps", type=int, default=0,
                    help="filler repetitions added to EVERY turn, standing "
                         "in for a tool result (~21 tokens each)")
    ap.add_argument("--arms", default="np1,np2,np2-sps,np2-pin")
    ap.add_argument("--sps", type=float, default=0.95,
                    help="the -sps for the np2-sps arm")
    ap.add_argument("--cram", type=int, default=0,
                    help="the -cram for the *-nocram arm (0 disables the RAM "
                         "prompt cache). The difference between that arm and "
                         "its twin IS what the cache is worth")
    a = ap.parse_args()

    global MAX_TOKENS
    MAX_TOKENS = a.max_tokens
    if not os.path.exists(a.env):
        raise SystemExit("no such profile: %s" % a.env)
    binary = a.binary
    if not binary:
        pinned = systemdfile.variable(a.env, "LLAMA_BIN")
        if pinned:
            binary = os.path.expanduser("~/" + pinned)
    binary = runlib.resolve_binary(binary)

    sweep.reexec_with_inhibit()
    meta = runlib.provenance(binary)
    meta["profile"] = os.path.basename(a.env)
    meta["turns"] = a.turns
    meta["head_reps"] = a.head_reps
    meta["body_reps"] = a.body_reps
    meta["sps"] = a.sps
    meta["growth_reps"] = a.growth_reps
    meta["pattern"] = a.pattern
    meta["cram_arm"] = a.cram
    meta["mode"] = a.mode
    meta["max_tokens"] = a.max_tokens
    meta["persist"] = a.persist
    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = os.path.join(BENCH, "reports", "%s_slot-affinity_%s"
                        % (stamp, meta["build_id"]))
    os.makedirs(dest, exist_ok=True)
    print("binary:  %s" % meta["binary"])
    print("profile: %s   report: %s" % (meta["profile"], dest))

    ARMS = {
        "np1":     dict(np=1, sps=None, pin=False),
        "np2":     dict(np=2, sps=None, pin=False),
        "np2-sps": dict(np=2, sps=a.sps, pin=False),
        "np2-pin": dict(np=2, sps=None, pin=True),
        # Same as np2-pin with the RAM prompt cache switched off. A slot that
        # is taken away can either come back out of -cram or be recomputed
        # from nothing, and with several sub-agents sharing one slot that is
        # the difference between an arrangement that works and one that does
        # not. Nothing else about the arm differs, so the gap is attributable.
        # The name carries the value, because "nocram" stopped being true
        # the moment --cram became a number: an arm called nocram running at
        # 2048 is the kind of label this repository spent 05.09.2026 finding
        # in other people's files. Reports before 17:0x use the old name and
        # were all -cram 0.
        "np2-pin-cram": dict(np=2, sps=None, pin=True, cram=a.cram),
    }
    wanted = [x.strip() for x in a.arms.split(",") if x.strip()]
    for w in wanted:
        if w not in ARMS:
            raise SystemExit("unknown arm %r; known: %s"
                             % (w, ", ".join(ARMS)))

    was_active = sweep.active_llama_unit()
    results = {"_meta": meta, "arms": []}

    def save():
        with open(os.path.join(dest, "result.json"), "w",
                  encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)

    try:
        if was_active:
            sweep.stop_production(was_active)
            for _ in range(60):
                if sweep.port_free(URL):
                    break
                time.sleep(1)
        for name in wanted:
            cfg = ARMS[name]
            label = (name if cfg.get("cram") is None
                     else "%s%d" % (name, cfg["cram"]))
            print("\n=== arm %s" % label)
            argv = server_argv(a.env, cfg["np"], cfg["sps"],
                               cfg.get("cram"))
            rec = run_arm(label, argv, binary, dest, a.turns,
                          a.head_reps, a.body_reps, cfg["pin"],
                          a.growth_reps, a.pattern, a.mode, a.persist)
            rec["steady_state"] = steady_state(rec)
            results["arms"].append(rec)
            save()
    finally:
        if was_active:
            print("\nrestoring %s ..." % was_active)
            sweep.start_production(was_active)
            sweep.slots_ready(sweep.URL, 900)
        save()
        print("report: %s" % dest)

    if a.mode == "parallel":
        print("\nRESULT — wall clock until ALL agents of a round are done")
        print("  %-15s %-8s %-9s %s" % ("arm", "slots", "premise", "rounds (s)"))
        for rec in results["arms"]:
            if rec.get("error"):
                print("  %-15s %-8s %s" % (rec["arm"], rec.get("slots_seen"),
                                           "FAILED — " + rec["error"]))
                continue
            walls = [r["wall"] for r in rec.get("rounds", [])]
            print("  %-15s %-8s %-9s %s"
                  % (rec["arm"], rec.get("slots_seen"),
                     "ok" if rec.get("premise_holds") else "BROKEN",
                     "  ".join("%.2f" % w for w in walls) or "?"))
        print("\n  Lower is better. The warm-up turn is excluded; each round "
              "fires every\n  agent at the same moment and ends when the last "
              "one answers.")
        return

    print("\nRESULT — recomputed tokens per turn, from turn 2 on")
    print("  %-15s %-8s %-9s %-11s %-9s %s"
          % ("arm", "slots", "premise", "recomputed", "median", "seconds"))
    for rec in results["arms"]:
        s = rec.get("steady_state")
        if rec.get("error"):
            print("  %-15s %-8s %s" % (rec["arm"], rec.get("slots_seen"),
                                       "FAILED — " + rec["error"]))
            continue
        print("  %-15s %-8s %-9s %-11s %-9s %s"
              % (rec["arm"], rec.get("slots_seen"),
                 "ok" if rec.get("premise_holds") else "BROKEN",
                 s["recomputed_total"] if s else "?",
                 s["recomputed_median"] if s else "?",
                 s["seconds_total"] if s else "?"))
    print("\n  A lower number is a prefix that stayed where it was. The arms "
          "are only\n  comparable to each other — the absolute size is the "
          "head this run used.")


if __name__ == "__main__":
    main()
