#!/usr/bin/env python3
"""restore-reuse — a restored slot holds the tokens and the next request
recomputes them anyway.

FOUND 05.09.2026 while measuring whether file persistence can stand in for a
too-small RAM prompt cache. It cannot, and the reason is not the calls: a
save writes 589 MB and reports n_saved=25512, the restore reads it back and
reports n_restored=25512, /slots confirms the slot holds 25512 tokens — and
the very next request on that slot prefills all 27,341 from zero.
bench/reports/2026-09-05_1747_slot-affinity_*.

That measurement is too large to be a bug report. This file is the small
version: ~4k-token prompts, one slot pair, four cells, a few minutes.

THE SEQUENCE, per cell:

    1  ask A on slot 1                 warm, cached = 0
    2  save slot 1 to a file
    3  ask B on slot 1                 B takes the slot, A is gone
    4  restore the file into slot 1    n_restored says it landed
    5  ask A again                     <- THE MEASUREMENT

Step 5 comes in two shapes, because they fail for different reasons and only
one of them would be a plain reuse bug:

    same          byte-identical to step 1 — and therefore SHORTER than the
                  saved state, which also holds the generated answer. On a
                  hybrid/recurrent model that requires rewinding the slot and
                  cannot work; this cell measures the model, not the restore.
                  Kept because the distinction is the whole finding.
    continuation  A, the answer the server ACTUALLY generated, and one more
                  turn. Here the saved state is a genuine prefix of the new
                  request, so a working restore must be reused. This is the
                  cell that decides.

THE CELLS isolate the leading suspect and its neighbours:

    default            the profile's flags, -np 2
    small-cram         -cram 2048, where the effect was first seen
    no-idle-cache      --no-cache-idle-slots, on by default and documented as
                       "save idle slots to the prompt cache on new task, and
                       clear them when using unified KV" — if the restored
                       state is pushed into a full cache and evicted before
                       its own turn, the cache layer undoes the file layer
    no-cram            -cram 0, no RAM cache at all. Separates "the cache
                       evicts it" from "the cache is required for it"

WHAT A CLEAN RESULT LOOKS LIKE: cached > 0 on step 5. Anything else is the
defect, and the cell names which ingredient it needs.

    python3 bench/suites/restore-reuse.py --env setup/env/qwen36.env

Nothing here touches production beyond stopping it for the duration; the
server is a side server on 8081 and the state files carry their own prefix
and are deleted at the end.
"""
import argparse, json, os, sys, time, urllib.request

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
STATE = "restore-reuse-A.bin"
# The planted fact and its id. A six-digit value cannot be produced by
# accident, which is what makes the answer comparison mean something.
NEEDLE_ID = 137
NEEDLE_BASE = 424242


def server_argv(env_path, cram=None, extra=(), nospec=False, np=2, port=PORT):
    argv, i, out = systemdfile.llama_args(env_path), 0, []
    while i < len(argv):
        tok = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        takes = nxt is not None and not nxt.startswith("-")
        drop = ["-np", "--parallel", "--port", "--host"]
        if cram is not None:
            drop += ["-cram", "--cache-ram"]
        # THE HYPOTHESIS THIS SWITCH TESTS. SERVER_TASK_TYPE_SLOT_RESTORE
        # calls llama_state_seq_load_file(ctx_tgt, ...) and nothing else,
        # while server_prompt_cache::load() takes ctx_tgt AND ctx_dft and
        # restores both. With --spec-type there IS a draft context, so a
        # file restore would bring back half a state where the RAM cache
        # brings back all of it — which is exactly the asymmetry measured.
        if nospec and (tok.startswith("--spec-") or tok == "-md"):
            if takes:
                i += 1
        elif tok in drop:
            if takes:
                i += 1
        else:
            out.append(tok)
        i += 1
    out += ["-np", str(np)]
    if cram is not None:
        out += ["-cram", str(cram)]
    out += list(extra)
    return out + ["--host", "127.0.0.1", "--port", str(port)]


def body(which, reps, turn=1, answer=None):
    """A prompt of a few thousand tokens WITH A NEEDLE, and a question that can
    only be answered from it.

    The needle is why this is a correctness test and not just a cache-hit
    counter. "Answer with A1" was the first version, and two matching
    characters prove almost nothing: a damaged context can still produce them.
    A six-digit value planted in the middle of the context cannot be guessed —
    if the restored state is subtly wrong, the answer changes or goes vague,
    and that is the failure mode a wrongly reassembled checkpoint would have.

    `answer` MUST be the text the server actually generated, and getting that
    wrong is what the first version of this suite did. A saved state always
    holds prompt PLUS the generated tokens, so the only request that can use
    it is one whose rendered prompt begins with exactly those tokens. An
    invented assistant turn diverges at the first differing token, the slot
    would have to be rewound, and a hybrid/recurrent memory cannot rewind —
    which the server then reports as "forcing full prompt re-processing due
    to lack of cache data". That is the model's property, not a restore bug,
    and a test that supplies its own answer measures the property instead of
    the question.
    """
    needle = NEEDLE_BASE + (0 if which == "A" else 1)
    half = max(1, reps // 2)
    sys_text = (("Workspace %s. Rule %s about this repository. " % (which, which)) * half
                + "Fact %d for workspace %s is %d. " % (NEEDLE_ID, which, needle)
                + ("Workspace %s. Rule %s about this repository. " % (which, which)) * (reps - half))
    msgs = [{"role": "system", "content": sys_text},
            {"role": "user", "content": "What is the value of fact %d? "
                                        "Reply with the number and nothing else."
                                        % NEEDLE_ID}]
    if turn > 1:
        msgs += [{"role": "assistant", "content": answer if answer is not None
                                                  else str(needle)},
                 {"role": "user", "content": "Repeat the value of fact %d, "
                                             "digits only." % NEEDLE_ID}]
    return msgs


def ask(messages, id_slot=None, max_tokens=24, timeout=900):
    payload = {"model": "reuse", "stream": False, "temperature": 0,
               "max_tokens": max_tokens, "messages": messages}
    if id_slot is not None:
        payload["id_slot"] = id_slot
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    u = out.get("usage") or {}
    d = u.get("prompt_tokens_details") or {}
    msg = ((out.get("choices") or [{}])[0].get("message") or {})
    return {"prompt": u.get("prompt_tokens"), "cached": d.get("cached_tokens"),
            "text": msg.get("content") or "",
            "seconds": round(time.time() - t0, 2)}


def slot_action(slot, action, filename, timeout=900):
    req = urllib.request.Request(
        URL + "/slots/%d?action=%s" % (slot, action),
        data=json.dumps({"filename": filename}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def slots_now():
    with urllib.request.urlopen(URL + "/slots", timeout=10) as x:
        return [s.get("n_prompt_tokens") for s in json.loads(x.read().decode())]


def one_cell(name, argv, binary, dest, reps, shape, slot=1):
    rec = {"cell": name, "shape": shape,
           "argv": [systemdfile.unexpand(x) for x in argv]}
    before = runlib._gtt("used")
    proc = None
    try:
        proc = runlib.start_server(argv, os.path.join(dest, "%s-%s.log"
                                                      % (name, shape)), binary)
        if not sweep.slots_ready(URL, 420):
            rec["error"] = "/slots never answered"
            return rec
        rec["slots"] = len(slots_now())

        rec["slot"] = slot
        rec["step1_warm"] = ask(body("A", reps), id_slot=slot)
        rec["step2_save"] = slot_action(slot, "save", STATE)
        rec["step3_displace"] = ask(body("B", reps), id_slot=slot)
        rec["slots_after_displace"] = slots_now()
        rec["step4_restore"] = slot_action(slot, "restore", STATE)
        rec["slots_after_restore"] = slots_now()
        # THE MEASUREMENT. `same` repeats step 1 byte for byte; `continuation`
        # adds one turn, which is what a real session sends.
        again = body("A", reps, turn=1 if shape == "same" else 2,
                     answer=rec["step1_warm"]["text"])
        rec["step5_reuse"] = ask(again, id_slot=slot)
        c = rec["step5_reuse"]["cached"]
        rec["reused"] = bool(c)

        # STEP 6 — THE ANSWER, not the token count. Reusing a restored state is
        # worth nothing if the answer changes: this patch makes a rewind path
        # live that was previously dead, and a wrongly reassembled checkpoint
        # would not raise, it would answer differently. So the slot is erased
        # and the identical request asked again, which forces a full prefill
        # with nothing carried over. The two texts must match.
        try:
            slot_action(slot, "erase", STATE)
        except Exception as e:                                # noqa: BLE001
            rec["erase_error"] = "%s" % type(e).__name__
        rec["step6_reference"] = ask(again, id_slot=slot)
        a, b = rec["step5_reuse"]["text"], rec["step6_reference"]["text"]
        rec["answer_matches"] = (a == b)
        rec["answers"] = {"after_restore": a[:120], "recomputed": b[:120]}
        needle = str(NEEDLE_BASE)
        rec["needle_after_restore"] = needle in a
        rec["needle_recomputed"] = needle in b
        if not rec["needle_after_restore"]:
            print("   NEEDLE LOST after restore: %r" % a[:60])
        if not rec["answer_matches"]:
            print("   ANSWER DIFFERS after restore: %r vs %r" % (a[:60], b[:60]))
        print("   %-14s %-12s slots=%s  after_restore=%s  "
              "step5 prompt=%s cached=%s %ss  -> %s"
              % (name, shape, rec["slots"], rec["slots_after_restore"],
                 rec["step5_reuse"]["prompt"], c,
                 rec["step5_reuse"]["seconds"],
                 ("REUSED" if c else "RECOMPUTED")
                 + (" · answer OK" if rec.get("answer_matches")
                    else " · ANSWER DIFFERS")))
    except (Exception, SystemExit) as e:                      # noqa: BLE001
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        print("   ABORTED %s" % rec["error"])
    finally:
        for f in (os.listdir(SLOT_DIR) if os.path.isdir(SLOT_DIR) else []):
            if f.startswith("restore-reuse-"):
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


CELLS = {
    "default":       dict(cram=None, extra=()),
    "no-spec":       dict(cram=None, extra=(), nospec=True),
    "small-cram":    dict(cram=2048, extra=()),
    "no-idle-cache": dict(cram=2048, extra=("--no-cache-idle-slots",)),
    "no-cram":       dict(cram=0, extra=()),
    # Not a variable of the experiment — the same default cell with the
    # server's own reasoning switched on. SLT_DBG prints n_past and what the
    # common-prefix search decided, which is the one thing the API cannot say.
    # --verbose, NOT -lv 1: the flag is a THRESHOLD and messages above it are
    # dropped, so -lv 1 silenced the startup line bench/run.py waits for and
    # the server never came up (report …_1808, kept for that reason).
    "debug":         dict(cram=None, extra=("--verbose",)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--binary")
    ap.add_argument("--reps", type=int, default=250,
                    help="repetitions in the system text (~4k tokens at 250)")
    ap.add_argument("--cells", default=",".join(CELLS))
    # THE PRODUCTION SHAPE is --np 1 --slot 0: one slot, the same slot taken by
    # another prefix and then restored. That is exactly what the gateway's
    # SESSION_RESTORE=displaced does, so it is the configuration that decides
    # whether the persistence running in production today buys anything.
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--slot", type=int, default=1)
    ap.add_argument("--shapes", default="same,continuation")
    a = ap.parse_args()

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
    meta["reps"] = a.reps
    meta["np"] = a.np
    meta["slot"] = a.slot
    stamp = time.strftime("%Y-%m-%d_%H%M")
    dest = os.path.join(BENCH, "reports",
                        "%s_restore-reuse_%s" % (stamp, meta["build_id"]))
    os.makedirs(dest, exist_ok=True)
    print("binary:  %s" % meta["binary"])
    print("report:  %s" % dest)

    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    shapes = [s.strip() for s in a.shapes.split(",") if s.strip()]
    for c in cells:
        if c not in CELLS:
            raise SystemExit("unknown cell %r; known: %s" % (c, ", ".join(CELLS)))

    was_active = sweep.active_llama_unit()
    results = {"_meta": meta, "cells": []}

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
        for c in cells:
            for shape in shapes:
                cfg = CELLS[c]
                argv = server_argv(a.env, cfg["cram"], cfg["extra"],
                                   cfg.get("nospec", False), a.np)
                results["cells"].append(
                    one_cell(c, argv, binary, dest, a.reps, shape, a.slot))
                save()
    finally:
        if was_active:
            print("\nrestoring %s ..." % was_active)
            sweep.start_production(was_active)
            sweep.slots_ready(sweep.URL, 900)
        save()
        print("report: %s" % dest)

    print("\nRESULT — was the restored state reused?")
    print("  %-14s %-13s %-9s %s" % ("cell", "shape", "cached", "verdict"))
    for r in results["cells"]:
        if r.get("error"):
            print("  %-14s %-13s FAILED — %s" % (r["cell"], r["shape"],
                                                 r["error"]))
            continue
        print("  %-14s %-13s %-9s %s"
              % (r["cell"], r["shape"], r["step5_reuse"]["cached"],
                 ("reused" if r["reused"] else "RECOMPUTED")
                 + ("" if r.get("answer_matches") else "  ANSWER DIFFERS")))
    print("\n  A restored slot that reports n_restored and then recomputes "
          "everything is\n  the defect. The cell that reuses names the "
          "ingredient that was missing.")


if __name__ == "__main__":
    main()
