#!/usr/bin/env python3
"""save-under-load — does a save that collides with a turn cost the turn?

    D=~/.cache/slot-save-cost
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \\
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 60 \\
        --extra "--slot-save-path $D/ -cram 0" -- \\
        python3 bench/suites/save-under-load.py --url http://127.0.0.1:8081 \\
            --dir $D --from-file slot-save-cost-4.bin --seed 1788368012 \\
            --out $D/under-load.json

THE LAST OPEN QUESTION BEFORE A DESIGN.

`bench/reports/2026-09-02_1856_slot-save-cost/` measured a save at 1,889 ms
for a 180k state and a rewrite of that state at ~1800 s — a factor of 950, so
the write is cheap enough to do often. But every save in that run found an
IDLE slot. In service it would fire right after an answer, and this consumer's
median gap to the next turn is 1.0 s (save-policy, 28.08.) — shorter than the
write. So the two overlap, and the whole design rests on what happens then.

Two defects of 28.08.2026 are about exactly that overlap, both found on the
PREWARM path, where the gateway had to re-create a prefix before saving it:

    autosave-evicts-the-working-slot     the turn lost: 0.7 s -> 13.6 s, 19x
    saved-prefix-holds-a-foreign-state   the file lost: it carried somebody
                                         else's prefix under our name

Neither should reach a session save. `SLOT_SAVE` only READS the slot
(`slot->prompt.tokens.serialize()`, server-context.cpp:2529) and is deferred
whole while the slot is processing — it never puts anything back. But that is
an argument from source, and both defects above were found in a stack whose
source also looked fine. This suite makes the collision happen on purpose and
measures both ends of it.

    2  baseline           a warm turn, no save anywhere near it
    3  turn during save   the save starts, the turn follows 200 ms later
                          -> does the turn WAIT, or does it lose its state?
    4  save during turn   a long generation starts, the save follows 200 ms
                          later -> is it deferred, and how long does it wait?
    5  file integrity     restore what cells 3 and 4 wrote and continue from
                          it -> does the file hold OUR state or somebody's?

WHY IT DOES NOT PREFILL 180k AGAIN. The state written by the 18:56 run is on
disk and `--from-file` restores it in about 1.3 s, which turns a 30-minute
setup into a few seconds. The prompt is regenerated from the same `--seed`;
`/tokenize` is deterministic for one model, and cell 2 fails loudly if the
reconstruction does not match what the file holds. Without both flags the
suite prefills `--depth` itself.
"""
import argparse, json, os, sys, threading, time, urllib.error, urllib.request

SRV = None


def req(path, payload=None, t=7200):
    d = json.dumps(payload).encode("utf-8") if payload is not None else None
    r = urllib.request.Request(SRV + path, data=d,
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as x:
        return json.loads(x.read().decode("utf-8"))


def get(path, t=60):
    with urllib.request.urlopen(SRV + path, timeout=t) as x:
        return json.loads(x.read().decode("utf-8"))


def tokens_for(n, seed):
    """Same generator as bench/suites/slot-save-cost.py — the seed is what
    makes a state written by that run reusable here."""
    import random
    rnd = random.Random(seed)
    words = ["%s%d" % (rnd.choice("qxzjvkwfy"), rnd.randrange(10 ** 6))
             for _ in range(n * 2 + 40)]
    toks = req("/tokenize", {"content": " ".join(words)})["tokens"]
    if len(toks) < n:
        raise SystemExit("tokenizer returned %d < %d tokens" % (len(toks), n))
    return toks[:n]


def turn(toks, npredict=1, tag=None, sink=None):
    t0 = time.time()
    r = req("/completion", {"prompt": toks, "n_predict": npredict,
                            "cache_prompt": True, "stream": False,
                            "return_tokens": True})
    tm = r.get("timings") or {}
    gen = [t for t in (r.get("tokens") or []) if isinstance(t, int)]
    out = {"wall_s": round(time.time() - t0, 3), "t_start": t0,
           "t_end": time.time(), "cache_n": tm.get("cache_n"),
           "prompt_n": tm.get("prompt_n"), "n_gen": len(gen), "gen": gen}
    if tag:
        print("  %-30s %8.2f s   cache_n=%-8s prompt_n=%s"
              % (tag, out["wall_s"], out["cache_n"], out["prompt_n"]), flush=True)
    if sink is not None:
        sink.append(out)
    return out


def save(name, tag=None, sink=None):
    t0 = time.time()
    try:
        d = req("/slots/0?action=save", {"filename": name})
    except Exception as e:
        detail = e.read().decode("utf-8")[:200] if hasattr(e, "read") else str(e)
        out = {"failed": detail, "t_start": t0, "wall_s": time.time() - t0}
        if sink is not None:
            sink.append(out)
        print("  %-30s FAILED: %s" % (tag or "save", detail), flush=True)
        return out
    out = {"wall_s": round(time.time() - t0, 3), "t_start": t0,
           "t_end": time.time(), "n_saved": d.get("n_saved"),
           "save_ms": (d.get("timings") or {}).get("save_ms"),
           "n_written": d.get("n_written"), "file": name}
    if tag:
        print("  %-30s %8.2f s   n_saved=%-8s save_ms=%.0f"
              % (tag, out["wall_s"], out["n_saved"], out["save_ms"] or 0),
              flush=True)
    if sink is not None:
        sink.append(out)
    return out


def restore(name, tag=None):
    t0 = time.time()
    d = req("/slots/0?action=restore", {"filename": name})
    out = {"wall_s": round(time.time() - t0, 3), "n_restored": d.get("n_restored"),
           "restore_ms": (d.get("timings") or {}).get("restore_ms")}
    if tag:
        print("  %-30s %8.2f s   n_restored=%s"
              % (tag, out["wall_s"], out["n_restored"]), flush=True)
    return out


def main():
    global SRV
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/slot-save-cost"))
    ap.add_argument("--from-file", default=None,
                    help="restore this state instead of prefilling one")
    ap.add_argument("--seed", type=int, default=0,
                    help="the seed the state in --from-file was built with")
    ap.add_argument("--depth", type=int, default=20000,
                    help="the state to collide on. NOT 180000 by default: the "
                         "collision mechanism is depth-independent (defer, "
                         "read-only), while a probe that misses costs one full "
                         "prefill of this — 1831 s at 180k, 80 s at 20k. Depth "
                         "belongs to slot-save-cost.py, which measured it.")
    ap.add_argument("--lag-ms", type=int, default=200,
                    help="how long after the first request the second starts")
    ap.add_argument("--long-predict", type=int, default=250,
                    help="tokens the long generation of cell 4 writes")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SRV = a.url
    if "8080" in a.url:
        raise SystemExit("refusing: 8080 is production.")

    R = {"url": a.url, "depth": a.depth, "lag_ms": a.lag_ms, "cells": {}}
    lag = a.lag_ms / 1000.0

    print("=== 1. the state")
    if a.from_file:
        if not a.seed:
            raise SystemExit("--from-file needs the --seed it was built with, "
                             "or the prompt cannot be reconstructed")
        path = os.path.join(a.dir, a.from_file)
        if not os.path.exists(path):
            raise SystemExit("no such state: %s" % path)
        toks = tokens_for(a.depth, a.seed)
        r = restore(a.from_file, "restore from file")
        if (r["n_restored"] or 0) != a.depth:
            raise SystemExit("the file holds %s tokens, --depth says %d — "
                             "they must match or every later cell measures "
                             "a different state" % (r["n_restored"], a.depth))
    else:
        toks = tokens_for(a.depth, a.seed or int(time.time()))
        print("  prefilling %d tokens, this is the slow part" % a.depth)
        turn(toks, tag="prefill")

    print("\n=== 2. baseline: a warm turn, nothing else happening")
    base = turn(toks + toks[:8], tag="warm turn")
    base_gen = base["gen"]
    R["cells"]["baseline"] = base
    if (base["cache_n"] or 0) < a.depth * 0.5:
        raise SystemExit(
            "ABORTED — the baseline turn is not warm (cache_n=%s of %d). "
            "Either the reconstructed prompt does not match the restored "
            "state, or the state is not in the slot. Nothing after this "
            "would mean anything." % (base["cache_n"], a.depth))

    print("\n=== 3. THE COLLISION: turn %d ms into a save" % a.lag_ms)
    sv, tn = [], []
    th = threading.Thread(target=save, args=("under-load-3.bin", "  save (thread)", sv))
    th.start()
    time.sleep(lag)
    turn(toks + toks[:16], tag="turn during the save", sink=tn)
    th.join()
    s3, t3 = (sv[0] if sv else {}), (tn[0] if tn else {})
    R["cells"]["turn_during_save"] = {"save": s3, "turn": t3}
    extra = (t3.get("wall_s") or 0) - (base["wall_s"] or 0)
    kept = (t3.get("cache_n") or 0) >= a.depth * 0.9
    print("  turn paid %+.2f s against the baseline; state %s"
          % (extra, "KEPT" if kept else "LOST"))

    print("\n=== 4. the other order: save %d ms into a long generation" % a.lag_ms)
    sv4, tn4 = [], []
    th = threading.Thread(target=turn,
                          args=(toks + toks[:24], a.long_predict,
                                "  long generation", tn4))
    th.start()
    time.sleep(lag)
    save("under-load-4.bin", "save during the turn", sv4)
    th.join()
    s4, t4 = (sv4[0] if sv4 else {}), (tn4[0] if tn4 else {})
    R["cells"]["save_during_turn"] = {"save": s4, "turn": t4}
    if s4.get("wall_s") and s4.get("save_ms") is not None:
        waited = s4["wall_s"] - s4["save_ms"] / 1000.0
        print("  the save waited %.2f s before it got the slot (generation "
              "ran %.2f s)" % (waited, t4.get("wall_s") or 0))
        R["cells"]["save_during_turn"]["waited_s"] = round(waited, 3)

    print("\n=== 5. do those two files hold OUR state?")
    # ONE probe per file, RECONSTRUCTED from the turn that produced the state
    # and sized by `n_saved`. Two rules collide here, and the first two
    # versions of this cell each got one of them wrong:
    #
    #   in the SLOT, reuse is trimmed to the common prefix. Cell 3 shows it:
    #     the slot held toks+toks[:8]+gen, the turn sent toks+toks[:16], and
    #     cache_n came back 180008 — they diverge there and everything before
    #     it was kept.
    #   after a RESTORE, a state carrying anything past the prompt is
    #     discarded WHOLE (bench/reports/2026-08-29_restore-semantics/), so a
    #     probe that is not a true superset costs a full re-prefill of the
    #     depth — 1831 s measured at 180k, per wrong probe.
    #
    # Version 1 tried candidates in turn: an hour of wrong guesses.
    # Version 2 CONCATENATED the candidates and called the result a superset
    # of all of them. It is not: candidates branch, they do not nest —
    # toks[:8]+gen and toks[:24]+gen4 share only their first 8 tokens, so the
    # chain is a superset of the shortest branch and of nothing else. It
    # reported the deepest file as foreign after paying the 1831 s, which is
    # a false accusation of a registered defect. There is no single prompt
    # that supersets branching candidates; the probe must be built per file.
    #
    # `n_saved` makes that exact rather than guessed: the save captures the
    # prompt plus (n_gen - 1) generated tokens — measured on both files, 180008
    # against a 180008-token prompt with 1 generated, and 180096 against
    # 180024 with 73. So the state is reconstructed to the length the server
    # itself reported, and a length that matches no candidate is reported
    # WITHOUT paying for a probe.
    def expected(prompt, gen):
        return {"prompt": prompt, "gen": gen, "len": len(prompt) + max(0, len(gen) - 1)}

    cand = {
        "under-load-3.bin": [expected(toks + toks[:8], base_gen),
                             expected(toks + toks[:16], t3.get("gen") or [])],
        "under-load-4.bin": [expected(toks + toks[:24], t4.get("gen") or []),
                             expected(toks + toks[:16], t3.get("gen") or [])],
    }
    integrity = {}
    for cell, s in (("turn_during_save", s3), ("save_during_turn", s4)):
        name, n_saved = s.get("file"), s.get("n_saved")
        if not name:
            integrity[cell] = {"skipped": "no file written"}
            continue
        match = [c for c in cand.get(name, []) if c["len"] == n_saved]
        if not match:
            print("  %-22s n_saved=%-8s -> LENGTH MATCHES NO CANDIDATE (%s) — "
                  "not probing, that alone is the finding"
                  % (cell, n_saved, [c["len"] for c in cand.get(name, [])]))
            integrity[cell] = {"n_saved": n_saved, "holds_our_state": None,
                               "why": "length matches no candidate"}
            continue
        c = match[0]
        probe_prompt = c["prompt"] + c["gen"][:n_saved - len(c["prompt"])] + toks[:4]
        # Displace first: without it a warm slot answers and the file is never
        # read at all — the check would then pass on any file whatsoever.
        turn(tokens_for(24, (a.seed or 1) + 7))
        r = restore(name, "  restore %s" % name)
        probe = turn(probe_prompt, tag="  probe (rebuilt to n_saved)")
        ok = (probe["cache_n"] or 0) >= n_saved * 0.9
        integrity[cell] = {"n_saved": n_saved, "n_restored": r["n_restored"],
                           "cache_n": probe["cache_n"], "holds_our_state": ok}
        print("  %-22s n_saved=%-8s -> %s"
              % (cell, n_saved, "holds our state" if ok else
                 "NOT OURS — the 28.08. foreign-state shape"))
    R["cells"]["integrity"] = integrity

    print("\n=== verdict")
    print("  baseline turn                 %.2f s, cache_n %s"
          % (base["wall_s"], base["cache_n"]))
    print("  turn colliding with a save    %.2f s, cache_n %s  (%+.2f s)"
          % (t3.get("wall_s") or 0, t3.get("cache_n"), extra))
    print("  the turn kept its state     : %s" % ("YES" if kept else "NO"))
    ok3 = integrity.get("turn_during_save", {}).get("holds_our_state")
    ok4 = integrity.get("save_during_turn", {}).get("holds_our_state")
    print("  files hold our state        : %s / %s"
          % ({True: "YES", False: "NO"}.get(ok3, "?"),
             {True: "YES", False: "NO"}.get(ok4, "?")))
    if kept and ok3 and ok4:
        print("\n  The overlap costs WAITING and nothing else — neither defect "
              "of 28.08. reaches this path. A cooldown is then a wear "
              "decision, not a correctness one.")
    else:
        print("\n  Something was lost. Read the cells before designing "
              "anything on top of this path.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(R, f, indent=1)
        print("\n  rows -> %s" % a.out)
    for n in ("under-load-3.bin", "under-load-4.bin"):
        p = os.path.join(a.dir, n)
        if os.path.exists(p):
            print("  left on disk: %s (%.1f GB)" % (p, os.path.getsize(p) / 1e9))


if __name__ == "__main__":
    main()
