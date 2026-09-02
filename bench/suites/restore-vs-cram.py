#!/usr/bin/env python3
"""restore-vs-cram — does a restore HIDE a longer state the RAM cache holds?

    D=~/.cache/slot-save-cost && mkdir -p $D
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \\
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 45 \\
        --extra "--slot-save-path $D/" -- \\
        python3 bench/suites/restore-vs-cram.py --url http://127.0.0.1:8081 \\
            --dir $D --out $D/restore-vs-cram.json

NOTE the missing `-cram 0`: this suite needs the profile's own budget, and it
is the only one of the four that does.

THE QUESTION, and it can still sink the whole design.

Everything measured on 02.09. — that a restored state serves its own
continuation, what a save costs, that a collision costs only waiting — ran
with `-cram 0`, i.e. with llama.cpp's RAM prompt cache switched off so that
nothing but the file could answer. Production runs `-cram 14336`. And a
restore is known to interact badly with that cache:

  llama.cpp consults its RAM prompt cache only `if (f_keep < 0.5f)`
  (server-context.cpp). A restore fills the slot, which makes f_keep large,
  which switches that lookup OFF. Measured 30.08.2026
  (bench/reports/2026-08-30_restore-blinds-cache/): where the cache held the
  conversation, restoring took 56.4 s against 1.0 s for doing nothing.

That is why RESTORE_ONLY_WHEN_SERVER_COLD=1 runs today. The danger for a
session-persistence design is precise: the FILE holds turn N, the CACHE may
hold turn N+k, and a restore that puts turn N into the slot can stop the
longer state from ever being found. Then persistence makes production slower,
not faster, and every figure of 02.09. would be true and useless.

THE SHAPE THAT SEPARATES IT. A is a shallow state written to disk; B is the
same conversation grown deeper and left to the cache. Both arms then ask for
B's continuation:

    arm 1  no restore        the cache should hand B back        cache_n ~ B
    arm 2  restore A first   if the restore blinds the lookup    cache_n ~ A

Arm 1 is also the control that makes arm 2 admissible: if the cache does not
return B there, nothing in this run is about restores at all — the state was
simply gone — and the suite says so instead of reporting a difference.

A IS A TRUE PREFIX OF B by construction (`toks[:a]` against `toks[:b]`), and
the save captures the prompt plus (n_gen - 1) tokens, so with n_predict=1 the
file holds exactly `toks[:a]`. Both matter: a restored state that is not a
prefix of the incoming prompt is discarded whole
(bench/reports/2026-08-29_restore-semantics/), which would make arm 2 slow for
a reason that has nothing to do with the cache.

Written 02.09.2026.
"""
import argparse, json, os, sys, time, urllib.request

SRV = None
STATE = "restore-vs-cram-a.bin"


def req(path, payload=None, t=7200):
    d = json.dumps(payload).encode("utf-8") if payload is not None else None
    r = urllib.request.Request(SRV + path, data=d,
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as x:
        return json.loads(x.read().decode("utf-8"))


def tokens_for(n, seed):
    import random
    rnd = random.Random(seed)
    words = ["%s%d" % (rnd.choice("qxzjvkwfy"), rnd.randrange(10 ** 6))
             for _ in range(n * 2 + 40)]
    toks = req("/tokenize", {"content": " ".join(words)})["tokens"]
    if len(toks) < n:
        raise SystemExit("tokenizer returned %d < %d" % (len(toks), n))
    return toks[:n]


def turn(toks, tag=None, npredict=1):
    t0 = time.time()
    r = req("/completion", {"prompt": toks, "n_predict": npredict,
                            "cache_prompt": True, "stream": False})
    tm = r.get("timings") or {}
    out = {"wall_s": round(time.time() - t0, 2), "cache_n": tm.get("cache_n"),
           "prompt_n": tm.get("prompt_n")}
    if tag:
        print("  %-34s %8.2f s   cache_n=%-8s prompt_n=%s"
              % (tag, out["wall_s"], out["cache_n"], out["prompt_n"]), flush=True)
    return out


def save(tag=None):
    d = req("/slots/0?action=save", {"filename": STATE})
    if tag:
        print("  %-34s n_saved=%-8s %.0f MB"
              % (tag, d.get("n_saved"), (d.get("n_written") or 0) / 1e6), flush=True)
    return d


def restore(tag=None):
    t0 = time.time()
    d = req("/slots/0?action=restore", {"filename": STATE})
    if tag:
        print("  %-34s %8.2f s   n_restored=%s"
              % (tag, time.time() - t0, d.get("n_restored")), flush=True)
    return d


def main():
    global SRV
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/slot-save-cost"))
    ap.add_argument("--shallow", type=int, default=20000, help="A, goes to disk")
    ap.add_argument("--deep", type=int, default=40000, help="B, goes to the cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SRV = a.url
    if "8080" in a.url:
        raise SystemExit("refusing: 8080 is production.")
    if a.deep <= a.shallow:
        raise SystemExit("--deep must exceed --shallow, or there is nothing "
                         "for the cache to hold that the file does not")
    seed = a.seed or int(time.time())
    R = {"url": a.url, "shallow": a.shallow, "deep": a.deep, "seed": seed,
         "cells": {}}

    print("REQUIRES the profile's own -cram (14336 in production), NOT 0 —")
    print("this is the one suite that measures the cache rather than avoiding it.\n")
    toks = tokens_for(a.deep, seed)
    junk1, junk2 = tokens_for(24, seed + 1), tokens_for(24, seed + 2)

    print("=== 1. build A and write it to disk")
    turn(toks[:a.shallow], "prefill A")
    d = save("save A")
    n_a = d.get("n_saved") or 0
    if n_a != a.shallow:
        print("  NOTE the file holds %d tokens, not %d — A is still a prefix "
              "of B, so the run stands, but read the numbers against %d."
              % (n_a, a.shallow, n_a))

    print("\n=== 2. grow the conversation to B, then let the cache have it")
    turn(toks[:a.deep], "prefill B")
    turn(junk1, "displace (LRU path saves B)")

    print("\n=== 3. ARM 1 — continuation of B, no restore in front")
    arm1 = turn(toks[:a.deep] + toks[:4], "continuation, cache only")
    R["cells"]["arm1_cache_only"] = arm1
    if (arm1["cache_n"] or 0) < a.deep * 0.9:
        raise SystemExit(
            "\nABORTED — the cache did not hand B back (cache_n=%s of %d), so "
            "nothing here is about restores: the state was simply gone. Check "
            "that -cram is the profile's value and not 0, and that B's entry "
            "fits the budget." % (arm1["cache_n"], a.deep))

    print("\n=== 4. put B back in the cache, then ARM 2 — restore A first")
    turn(toks[:a.deep], "re-establish B")
    turn(junk2, "displace again")
    restore("restore A")
    arm2 = turn(toks[:a.deep] + toks[:8], "continuation, after restoring A")
    R["cells"]["arm2_after_restore"] = arm2

    print("\n=== verdict")
    c1, c2 = arm1["cache_n"] or 0, arm2["cache_n"] or 0
    print("  arm 1, cache only        cache_n=%-8s %.2f s" % (c1, arm1["wall_s"]))
    print("  arm 2, restore first     cache_n=%-8s %.2f s" % (c2, arm2["wall_s"]))
    blinds = c2 < c1 * 0.9
    if blinds:
        print("\n  THE RESTORE BLINDS THE CACHE. It put %d tokens in the slot "
              "and the %d the cache held were not found: %.2f s against "
              "%.2f s. A restore policy on a server WITH a cache has to know "
              "which of the two is longer BEFORE it restores — the 30.08. "
              "finding stands on this build."
              % (n_a, c1, arm2["wall_s"], arm1["wall_s"]))
    else:
        print("\n  The restore does NOT hide the longer state here: arm 2 came "
              "back with %d, the same order as arm 1's %d. Either the lookup "
              "is not switched off on this build, or f_keep stayed under the "
              "0.5 threshold. Read the server log for `f_keep` before "
              "generalising — this is the cell most likely to be true for a "
              "reason other than the one being tested." % (c2, c1))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(R, f, indent=1)
        print("\n  rows -> %s" % a.out)
    p = os.path.join(a.dir, STATE)
    if os.path.exists(p):
        print("  left on disk: %s (%.1f GB)" % (p, os.path.getsize(p) / 1e9))


if __name__ == "__main__":
    main()
