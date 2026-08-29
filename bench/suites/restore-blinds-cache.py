#!/usr/bin/env python3
"""Does restoring a prefix hide the RAM cache that already held the answer?

The claim under test, from the live trace of 29.08.2026 23:41:50 and read out
of llama.cpp's source:

    llama.cpp consults its RAM prompt cache only `if (f_keep < 0.5f)`
    (server-context.cpp:1595), where f_keep is how much of the SLOT's current
    content the incoming prompt keeps. Restoring a prefix file makes the slot
    a PERFECT prefix of the next request — f_keep = 1.0 — so the lookup never
    runs, and a longer state of the same conversation sitting in the cache is
    never found.

If that holds, the restore is not a rescue but a cost: it replaces a
sub-second cache hit with a full re-prefill of everything behind the prefix.

TWO ROUNDS, identical except for one step.

    0  prefill PREFIX, save it to a .bin              (both rounds)
    1  prefill CONVERSATION (PREFIX + a long tail)    (both)
    2  send an unrelated tiny prompt                  (both)
         -> LRU takeover, which saves CONVERSATION into the RAM cache
    3  restore the .bin into the slot                 (ROUND A ONLY)
    4  send CONVERSATION + a short delta, and read `timings.cache_n`

    Round A predicts cache_n ~ |PREFIX|        — the cache was not consulted
    Round B predicts cache_n ~ |CONVERSATION|  — it was

The two rounds use DIFFERENT filler text, so their cache entries cannot stand
in for one another and a hit in B cannot be A's.

WHY THIS TALKS TO llama-server DIRECTLY. The gateway decides whether to
restore from its own bookkeeping, and there is no switch for "restore this
time but not next". Driving the slot API by hand is the only way to hold
everything else equal — and the mechanism under test is llama.cpp's, not the
gateway's.

    python3 bench/suites/restore-blinds-cache.py
    python3 bench/suites/restore-blinds-cache.py --rounds 2 --conv-tokens 3000

IT USES THE PRODUCTION SERVER, and it takes the one slot for a few minutes.
It writes two files into --slot-save-path and removes them again; both are
named `restore-blinds-<round>.bin` and nothing else in that directory is
touched.

IT ALSO EVICTS WHOEVER WAS USING THE MACHINE, and that is not a theoretical
cost. Measured 30.08.2026, 00:19, running these rounds against production:

    alloc: - making room for prompt cache entry, removing oldest entry
           (size = 6570.640 MiB)

That entry was the operator's live Claude Code session — 66,826 tokens. At
`-cram 32768` the RAM cache holds about five states of that size, and each
round of this suite adds two of its own. So the next turn of whatever was
running before is a cold start: on this machine, the better part of ten
minutes.

Run it when nobody is working, or on a side server via bench/sideserver.py.
The A/B itself is honest either way; the bystander is what pays.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

DEFAULT_URL = "http://127.0.0.1:8080"


def post(url, path, body, timeout=900):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def filler(tag, n_words):
    """Deterministic text, distinct per tag, ~1.3 tokens a word."""
    return " ".join("%s%d" % (tag, i) for i in range(n_words))


def send(url, prompt, label, quiet=False):
    t0 = time.time()
    d = post(url, "/completion",
             {"prompt": prompt, "n_predict": 1, "cache_prompt": True,
              "temperature": 0})
    tm = d.get("timings") or {}
    took = time.time() - t0
    if not quiet:
        print("      %-24s %6.1f s   cache_n=%-7s prompt_n=%-7s total=%s"
              % (label, took, tm.get("cache_n"), tm.get("prompt_n"),
                 d.get("tokens_evaluated")))
    return {"cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"),
            "total": d.get("tokens_evaluated"), "took_s": round(took, 1)}


def slot(url, action, filename, timeout=900):
    return post(url, "/slots/0?action=%s" % action, {"filename": filename},
                timeout=timeout)


def round_once(url, tag, restore, conv_words, prefix_words):
    """One round. `restore` is the ONLY difference between the two."""
    name = "restore-blinds-%s.bin" % tag
    prefix = filler(tag + "P", prefix_words)
    conv = prefix + " " + filler(tag + "C", conv_words)
    delta = conv + " " + filler(tag + "D", 20)
    print("    round %s — restore %s" % (tag, "ON" if restore else "OFF"))

    p = send(url, prefix, "0 prefill PREFIX")
    slot(url, "save", name)
    c = send(url, conv, "1 prefill CONVERSATION")
    send(url, "Ein voellig anderer Satz ohne Bezug.", "2 unrelated tiny")
    if restore:
        t0 = time.time()
        slot(url, "restore", name)
        print("      %-24s %6.1f s   (%s)"
              % ("3 restore the .bin", time.time() - t0, name))
    else:
        print("      %-24s          (skipped — this is the whole experiment)"
              % "3 restore the .bin")
    d = send(url, delta, "4 CONVERSATION + delta")
    return {"tag": tag, "restore": restore, "prefix_tokens": p["total"],
            "conv_tokens": c["total"], "final": d, "file": name}


def round_redundant(url, tag, conv_words, prefix_words):
    """Does the .bin add anything the RAM cache would not have given anyway?

    The other experiment shows the restore HURTS when the cache holds the
    conversation. This one asks the complementary question: when the cache
    holds only the PREFIX -- the state after a fresh project's first turn, and
    the case the disk store exists for -- does skipping the restore cost
    anything at all?

        0  prefill PREFIX                       -> slot
        1  unrelated tiny prompt                -> LRU takeover saves PREFIX
                                                   into the RAM cache
        2  no restore
        3  PREFIX + a long NEW conversation     -> read cache_n

    If cache_n comes back as the prefix, llama.cpp found it in RAM by itself
    and the file was never needed.
    """
    prefix = filler(tag + "P", prefix_words)
    conv = prefix + " " + filler(tag + "C", conv_words)
    print("    round %s — is the file redundant while the cache is warm?" % tag)
    p = send(url, prefix, "0 prefill PREFIX")
    send(url, "Noch ein ganz anderer Satz.", "1 unrelated tiny")
    d = send(url, conv, "3 PREFIX + new conversation")
    got = d["cache_n"] or 0
    print("      -> cache_n=%s of a %s-token prefix: %s"
          % (got, p["total"],
             "the RAM cache handed the prefix back WITHOUT any file"
             if abs(got - (p["total"] or 0)) <= 8 else
             "the prefix did NOT come back — the file would have been needed"))
    return {"tag": tag, "restore": None, "prefix_tokens": p["total"],
            "conv_tokens": d["total"], "final": d, "file": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("blinds", "redundant"), default="blinds",
                    help="blinds: A/B on the restore with the conversation "
                         "cached. redundant: does the file add anything when "
                         "only the prefix is cached?")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--rounds", type=int, default=1,
                    help="repetitions of the A/B pair")
    ap.add_argument("--conv-words", type=int, default=1800)
    ap.add_argument("--prefix-words", type=int, default=400)
    ap.add_argument("--salt", default="",
                    help="mixed into the filler text. A second run with the "
                         "same salt would find its OWN conversation still in "
                         "the RAM cache and measure nothing.")
    ap.add_argument("--restore-first", default="1",
                    help="1 = A then B, 0 = B then A. The order is a "
                         "confounder: everything B does happens with more in "
                         "the cache than A had.")
    ap.add_argument("--slot-path",
                    default=os.path.expanduser("~/.cache/llama-slots"),
                    help="only so the two files this writes can be removed again")
    a = ap.parse_args()

    print(__doc__.split("\n")[0])
    print("  server: %s" % a.url)
    results, written = [], []
    try:
        if a.mode == "redundant":
            for i in range(a.rounds):
                results.append(round_redundant(
                    a.url, "R%s%d" % (a.salt, i), a.conv_words, a.prefix_words))
            return 0
        order = (True, False) if a.restore_first == "1" else (False, True)
        for i in range(a.rounds):
            for restore in order:
                tag = "%s%s%d" % ("A" if restore else "B", a.salt, i)
                r = round_once(a.url, tag, restore, a.conv_words, a.prefix_words)
                written.append(r["file"])
                results.append(r)
    finally:
        for f in written:
            p = os.path.join(a.slot_path, f)
            try:
                os.remove(p)
            except OSError as e:
                print("  NOTE could not remove %s: %s" % (p, e))

    print("\n  %-8s %-9s %9s %9s %9s %9s"
          % ("round", "restore", "prefix", "conv", "cache_n", "took s"))
    for r in results:
        print("  %-8s %-9s %9s %9s %9s %9s"
              % (r["tag"], "ON" if r["restore"] else "OFF", r["prefix_tokens"],
                 r["conv_tokens"], r["final"]["cache_n"], r["final"]["took_s"]))

    a_rows = [r for r in results if r["restore"]]
    b_rows = [r for r in results if not r["restore"]]
    if not (a_rows and b_rows):
        return
    print("\n  What it says:")
    for rows, what in ((a_rows, "WITH the restore"), (b_rows, "WITHOUT it")):
        for r in rows:
            got, conv = r["final"]["cache_n"] or 0, r["conv_tokens"] or 0
            near_prefix = abs(got - (r["prefix_tokens"] or 0)) <= 8
            near_conv = got >= conv - 40
            verdict = ("the cache was NOT consulted — only the prefix survived"
                       if near_prefix else
                       "the cache WAS consulted — the conversation came back"
                       if near_conv else "neither prediction: read the rows")
            print("    %-8s %-17s cache_n=%-7s of %s  -> %s"
                  % (r["tag"], what, got, conv, verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
