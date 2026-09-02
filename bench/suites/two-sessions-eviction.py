#!/usr/bin/env python3
"""two-sessions-eviction — the 02.09. incident, and whether a file heals it.

    D=~/.cache/slot-save-cost && mkdir -p $D
    python3 bench/sideserver.py --env setup/env/flashnext.env --port 8081 \\
        --stop "llama-user@$(bash setup/lib/models.sh serving)" --deadline 45 \\
        --extra "--slot-save-path $D/ -cram 2048" -- \\
        python3 bench/suites/two-sessions-eviction.py \\
            --url http://127.0.0.1:8081 --dir $D --depth 20000 \\
            --unit sideserver-8081 --out $D/two-sessions.json

THE SHAPE THIS REPRODUCES, from the morning of 02.09.2026:

    08:50:43  session A releases 80,507 tokens
    08:53:22  the health probe takes the slot by LRU — A goes to the RAM cache
    08:55:24  session B needs room for the probe's own 226 MiB entry; eviction
              is FIFO by age and blind to size, so it throws out A's 3,946 MiB
    then      A's next turn prefills 80,394 tokens in 599.4 s at 0 % cache

Everything measured on 02.09. after that had ONE conversation in it. This is
the two-client shape, and it carries the question the day ended on.

WHY IT IS THE ONE PATH LEFT FOR PERSISTENCE. 2026-09-02_2050_restore-vs-cram
found a restore hides a longer state the cache holds — 330x — so restoring in
a running server is a bet the gateway cannot price, and persistence was left
to the restart case. The eviction case is the exception: once the cache has
thrown a state OUT it holds nothing, there is nothing to hide, and a restore
can only help. If that holds, the gateway has a second legitimate moment to
restore, and it can recognise it — `making room for prompt cache entry` is a
journal line check.sh already counts.

    arm 1  A continues after being evicted, no restore   the incident
    arm 2  the same, restoring A's file first            the repair

WHY -cram 2048 AND depth 20000 RATHER THAN THE REAL NUMBERS. The mechanism is
a budget being too small for two entries plus a probe; which absolute numbers
express that changes nothing about it, and the entry sizes are already
measured (336.7 MiB + 39.12 KiB/token, bench/reports/2026-09-02_0938_cram-
state-size, plus a session surcharge that grows with depth). Two 20k entries
are ~2,200 MiB against a 2,048 MiB budget — the same arithmetic as two 148k
sessions against 14,336, at a twentieth of the prefill. The suite VERIFIES the
eviction happened rather than assuming it: it reads the server's own `making
room` line out of the journal, and refuses to report arms whose premise did
not occur.

Scaled to production (-cram 14336, computed not measured): two ~120k sessions
plus a probe fit; two ~148k do not. The 14:42 and 17:03 evictions of 02.09.,
both naming ~11,440 MiB, are one ~180k session leaving no room for anything
at all.

Written 02.09.2026.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

SRV = None
EVICT = re.compile(r"removing oldest entry \(size = ([0-9.]+) MiB\)")
STATE = "two-sessions-a.bin"


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


def turn(toks, tag=None):
    t0 = time.time()
    r = req("/completion", {"prompt": toks, "n_predict": 1,
                            "cache_prompt": True, "stream": False})
    tm = r.get("timings") or {}
    out = {"wall_s": round(time.time() - t0, 2), "cache_n": tm.get("cache_n"),
           "prompt_n": tm.get("prompt_n")}
    if tag:
        print("  %-36s %8.2f s   cache_n=%-8s prompt_n=%s"
              % (tag, out["wall_s"], out["cache_n"], out["prompt_n"]), flush=True)
    return out


def save(tag=None):
    d = req("/slots/0?action=save", {"filename": STATE})
    if tag:
        print("  %-36s n_saved=%-8s %.0f MB"
              % (tag, d.get("n_saved"), (d.get("n_written") or 0) / 1e6), flush=True)
    return d


def restore(tag=None):
    t0 = time.time()
    d = req("/slots/0?action=restore", {"filename": STATE})
    if tag:
        print("  %-36s %8.2f s   n_restored=%s"
              % (tag, time.time() - t0, d.get("n_restored")), flush=True)
    return d


def evictions_since(unit, since_iso):
    """The server's own `making room` lines, with the MiB they name.

    Read rather than inferred: an arm whose premise did not happen must not be
    reported as a result. `-cram` too generous for the chosen depth is exactly
    the silent way this suite could measure nothing and say something.
    """
    # BY IDENTIFIER, NOT BY UNIT — measured 02.09.2026, and the first two runs
    # of this suite were read through the broken version. A side server is a
    # TRANSIENT unit, and `journalctl --user -u sideserver-8081` returns
    # nothing for it: its lines carry the identifier `llamaexec` instead. The
    # call still exits 0, so an empty result read as "no eviction" when the
    # honest answer was "no data". Both runs happened to have no eviction
    # anyway, so the reading was accidentally right — which is the worst way
    # for a meter to be wrong, because nothing contradicts it.
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-t", "llamaexec", "--since", since_iso,
             "--no-pager"], capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        return None, "journal unreadable: %s" % e
    # "the server logged nothing in this window" and "the server logged no
    # eviction" are different facts, and only the second one is a measurement.
    if not out.strip():
        return None, ("no server lines at all in this window — the meter is "
                      "pointed at nothing, not at a healthy cache")
    # Anchored on the server's own wording, not on "any number in the line":
    # a journal line carries a timestamp and a pid, and a loose parse would
    # report those as eviction sizes. Same expression as
    # bench/suites/cram-state-size.py, which this figure has to stay
    # comparable with.
    return [float(m) for m in EVICT.findall(out)], None


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    global SRV
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/slot-save-cost"))
    ap.add_argument("--depth", type=int, default=20000)
    ap.add_argument("--unit", default="sideserver-8081",
                    help="the systemd unit whose journal names the evictions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SRV = a.url
    if "8080" in a.url:
        raise SystemExit("refusing: 8080 is production.")
    seed = a.seed or int(time.time())
    R = {"url": a.url, "depth": a.depth, "seed": seed, "cells": {}}

    # A and B share NOTHING: two different conversations, which is what makes
    # B's arrival a fresh entry rather than a continuation of A's.
    A = tokens_for(a.depth, seed)
    B = tokens_for(a.depth, seed + 1000)
    probe = tokens_for(24, seed + 2000)

    print("=== 1. session A, then its file")
    turn(A, "prefill A")
    d = save("save A")
    n_a = d.get("n_saved") or 0

    print("\n=== 2. the health probe takes the slot (A goes to the cache)")
    turn(probe, "probe")

    t0 = stamp()
    print("\n=== 3. session B arrives, then gives the slot up in its turn")
    turn(B, "prefill B")
    # THE SECOND PROBE IS NOT DECORATION, and leaving it out is how the first
    # run of this suite measured nothing (02.09., aborted by its own guard).
    # A state becomes a cache ENTRY when it is displaced, not when it is
    # prefilled: while B sits in the slot the cache holds only A and the first
    # probe — 1,327 MiB against a 2,048 budget, no reason to evict anything.
    # The eviction comes when room is needed for a NEW entry, which is exactly
    # how the 02.09. incident ran: not at B's prefill but at the next
    # takeover. So B has to be displaced too before the budget is under
    # pressure at all.
    turn(probe, "probe again (B becomes an entry)")
    sizes, err = evictions_since(a.unit, t0)
    if err:
        raise SystemExit("\nABORTED — %s" % err)
    print("  evictions during B: %s" % (sizes if sizes else "none"))
    R["cells"]["evictions_on_B"] = sizes
    if not sizes:
        raise SystemExit(
            "\nABORTED — the server logged no eviction, so there is no "
            "incident to measure and both arms below would describe a healthy "
            "cache.\n"
            "  MEASURED 02.09.2026: -cram 2048 at --depth 20000 does NOT "
            "evict, twice, though two 20k entries arithmetically exceed it "
            "(2 x 1100.694 + 226 against 2048). Why is unexplained — the\n"
            "  suspects are that a state becomes an entry later than assumed, "
            "or that an entry loaded back out of the cache is removed from it "
            "(server_prompt_cache::alloc deletes an entry fully contained in\n"
            "  the new prompt). Raise --depth or lower -cram further, and "
            "read `making room` in the journal rather than trusting the "
            "arithmetic — that is what this guard exists for.")

    print("\n=== 4. ARM 1 — A continues, no restore (this is the incident)")
    arm1 = turn(A + A[:4], "A after the eviction")
    R["cells"]["arm1_no_restore"] = arm1

    print("\n=== 5. evict A again, the same way")
    t1 = stamp()
    turn(B, "prefill B")
    turn(probe, "probe again")
    s2, _ = evictions_since(a.unit, t1)
    print("  evictions this time: %s" % (s2 if s2 else "none"))
    R["cells"]["evictions_second_round"] = s2
    if not s2:
        print("  NOTE arm 2 below therefore does NOT start from an evicted "
              "state — read it as a restore into a warm cache, which is the "
              "330x case of 2026-09-02_2050_restore-vs-cram and not this "
              "suite's question.")

    print("\n=== 6. ARM 2 — restore A's file first (this is the repair)")
    r = restore("restore A")
    arm2 = turn(A + A[:8], "A after restoring its file")
    R["cells"]["arm2_after_restore"] = dict(arm2, n_restored=r.get("n_restored"))

    print("\n=== verdict")
    c1, c2 = arm1["cache_n"] or 0, arm2["cache_n"] or 0
    print("  arm 1, evicted, no restore    cache_n=%-8s %.2f s" % (c1, arm1["wall_s"]))
    print("  arm 2, evicted, restored      cache_n=%-8s %.2f s" % (c2, arm2["wall_s"]))
    healed = c2 >= n_a * 0.9 and c2 > c1 * 1.5
    if healed:
        print("\n  THE FILE HEALS THE EVICTION: %d of %d tokens came back and "
              "the turn cost %.2f s instead of %.2f. A restore after an "
              "eviction cannot hide anything, because the cache holds nothing "
              "— so `making room` in the journal is a second legitimate "
              "moment to restore, beside a cold server."
              % (c2, n_a, arm2["wall_s"], arm1["wall_s"]))
    else:
        print("\n  The file did NOT heal it (arm 2 reused %d of a %d-token "
              "file against arm 1's %d). Persistence stays confined to the "
              "restart case; read the cells before concluding why."
              % (c2, n_a, c1))
    R["healed"] = bool(healed)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(R, f, indent=1)
        print("\n  rows -> %s" % a.out)
    p = os.path.join(a.dir, STATE)
    if os.path.exists(p):
        print("  left on disk: %s (%.1f GB)" % (p, os.path.getsize(p) / 1e9))


if __name__ == "__main__":
    main()
