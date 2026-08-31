#!/usr/bin/env python3
"""Does hoisting mid-conversation system messages help the cache or hurt it?

The gateway moves the STABLE part of every system message that sits inside the
conversation to the FRONT of the prompt, leaving only the volatile remainder
(the counter Claude Code glues to it) where it was. The reason on record is
that the counter would otherwise change the prefix every turn.

The objection, raised 30.08.2026: whatever is at the front is what every later
token depends on, so moving something there mid-conversation invalidates
everything behind it — and that is the opposite of what a prompt cache wants.
Measured that morning: one new system message, the front moved by 334
characters, and 73,877 tokens were recomputed for 668.9 s.

So: two renderings of the SAME three turns, driven straight at llama-server so
no gateway behaviour is in the way.

    A  hoisted      system = base + every stable block, joined
                    messages = conversation, with only the counters left inline
    B  left alone   system = base
                    messages = conversation with the system blocks where they
                    arrived, counters included

    turn 1   both cold — this is the baseline, not a result
    turn 2   the counter changes
    turn 3   a NEW system message appears, which is the case that hurt

What is read is `timings.cache_n`: how many tokens llama.cpp reused.

    python3 bench/suites/hoist-cost.py

IT USES THE PRODUCTION SERVER and takes the one slot for a few minutes. Each
variant leaves a few states in the RAM prompt cache; at the sizes used here
that is a few hundred MB each, not the gigabytes a real session holds.
"""
import argparse, json, sys, time, urllib.request

DEFAULT_URL = "http://127.0.0.1:8080"


def post(url, path, body, timeout=1800):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def words(tag, n):
    return " ".join("%s%d" % (tag, i) for i in range(n))


def send(url, prompt, label):
    t0 = time.time()
    d = post(url, "/completion", {"prompt": prompt, "n_predict": 1,
                                  "cache_prompt": True, "temperature": 0})
    tm = d.get("timings") or {}
    took = time.time() - t0
    print("      %-22s %7.1f s   cache_n=%-7s prompt_n=%-7s total=%s"
          % (label, took, tm.get("cache_n"), tm.get("prompt_n"),
             d.get("tokens_evaluated")))
    return {"cache_n": tm.get("cache_n") or 0, "total": d.get("tokens_evaluated"),
            "took_s": round(took, 1)}


def build(variant, salt, base_words, conv_words, stables, counter):
    """The same conversation, rendered the two ways.

    `stables` is how many distinct system blocks have appeared so far; turn 3
    passes one more than turn 2, which is the whole point of turn 3.

    The system blocks sit LATE in the conversation, where Claude Code puts
    them — behind the user question. That placement is what decides whether
    leaving them alone is cheap, so it is a parameter of the experiment and
    not an assumption buried in it.
    """
    base = "SYSTEM " + words(salt + "B", base_words)
    talk = "USER " + words(salt + "C", conv_words)
    blocks = ["REMINDER %d: %s" % (k, words("%sR%d" % (salt, k), 60))
              for k in range(stables)]
    tail = "COUNTER %d" % counter
    if variant == "hoisted":
        return (base + "\n\n" + "\n\n".join(blocks) + "\n\n"
                + talk + "\n\n" + tail)
    return base + "\n\n" + talk + "\n\n" + "\n\n".join(blocks) + "\n" + tail


def run(url, variant, salt, base_words, conv_words):
    print("    %s" % ("A  hoisted — the stable blocks live at the FRONT"
                      if variant == "hoisted" else
                      "B  left alone — they stay where they arrived"))
    out = []
    out.append(send(url, build(variant, salt, base_words, conv_words, 2, 1),
                    "turn 1 (cold)"))
    out.append(send(url, build(variant, salt, base_words, conv_words, 2, 2),
                    "turn 2 counter++"))
    out.append(send(url, build(variant, salt, base_words, conv_words, 3, 3),
                    "turn 3 NEW block"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--base-words", type=int, default=900)
    ap.add_argument("--conv-words", type=int, default=3500)
    ap.add_argument("--salt", default="H")
    a = ap.parse_args()

    print(__doc__.split("\n")[0])
    print("  server: %s" % a.url)
    res = {}
    for variant in ("hoisted", "plain"):
        res[variant] = run(a.url, variant, a.salt + variant[0],
                           a.base_words, a.conv_words)

    print("\n  %-12s %12s %12s %12s" % ("", "turn 1", "turn 2", "turn 3"))
    for variant in ("hoisted", "plain"):
        print("  %-12s %12s %12s %12s"
              % ("A hoisted" if variant == "hoisted" else "B left alone",
                 *["%d/%d" % (r["cache_n"], r["total"]) for r in res[variant]]))
    print("\n  reused tokens / prompt tokens. Turn 1 is cold in both and is "
          "the baseline.")

    print("\n  What it says:")
    for variant, name in (("hoisted", "A hoisted   "), ("plain", "B left alone")):
        t2, t3 = res[variant][1], res[variant][2]
        print("    %s turn 2 kept %.0f%%, turn 3 kept %.0f%%"
              % (name, 100.0 * t2["cache_n"] / max(t2["total"], 1),
                 100.0 * t3["cache_n"] / max(t3["total"], 1)))
    a3, b3 = res["hoisted"][2], res["plain"][2]
    if b3["cache_n"] > a3["cache_n"]:
        print("    -> on the turn where a NEW block appears, leaving them "
              "alone kept %d more tokens (%.1f s against %.1f s)."
              % (b3["cache_n"] - a3["cache_n"], b3["took_s"], a3["took_s"]))
    elif a3["cache_n"] > b3["cache_n"]:
        print("    -> hoisting kept %d MORE tokens on that turn. The objection "
              "does not hold at these sizes and placements."
              % (a3["cache_n"] - b3["cache_n"]))
    else:
        print("    -> no difference at these sizes. Read turn 2 as well "
              "before concluding anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
