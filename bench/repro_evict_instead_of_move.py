#!/usr/bin/env python3
"""Reproducer: the KV pool forgets a live 192k region where a small no-loss move would do.

    python3 repro_evict_instead_of_move.py [http://127.0.0.1:8080]

Needs a FRESHLY STARTED Halogen Flash Server with HALOGEN_KV_POOL_POSITIONS=524288
(the layout below depends on the pool starting empty). Standard library only.
Takes 22 minutes on 0.13.4 (measured 23.09.2026; six cold 192k prefills of about
193 s each are most of it) and less on 0.12.3, where round 5 stays warm.

WHAT IT DOES. Four conversations' worth of requests, strictly one at a time, in the
order recorded on 22.09.2026 (bench/sessionload.py --side-turns --live-fillers):

  F1, F2   two 192k conversations, touched every round with a one-word answer
           (max_tokens 64) — live neighbours, like a subagent beside a main loop
  MAIN     a 40k conversation that grows by ~4k per turn (max_tokens 12000)
  SIDE     the same conversation with message 2 rendered differently and a short
           question at the end (a harness's side query), sent after every MAIN turn

WHAT IT CHECKS. At SIDE turn 3 the pool holds F1 [0,192512) F2 [192512,385024)
SIDE [385024,443904) MAIN [443904,506624) and 17,664 free positions at the end
(reconstructed from the `pool N/524288` figures and the `moved` lines; the server
does not print the layout). SIDE needs 62,720, i.e. 3,840 more than its region
holds. Moving MAIN up by 3,840 would do it with no loss — the same kind of move
the server made one round earlier. Measured on 0.12.3 and 0.13.4 alike, the
server instead prints

    kv pool: no room for 62720 positions; forgot the region at 0 (2 cache entries,
             0 chunks; the least recently used)

i.e. F1's live 192k region, and F1's next touch is a cold ~193 s prefill.
That is FINDING 1.

FINDING 2 is round 5. SIDE's own region (at 0 by then) has to grow. 0.12.3 prints
`the region at 0 this request hit grows in place: forgot 2 longer entries
(cheapest)` — SIDE's own superseded entries — and every conversation stays warm.
0.13.4 prints `forgot the region at 72704 (... the least recently used)`, which is
F1's region again, and F1's next touch is cold.

The script reports each finding as REPRODUCED when F1's touch in that round comes
back with no cached tokens. Read the container log beside it:
    podman logs halogen 2>&1 | grep '^kv pool'
"""
import json
import random
import sys
import time
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
SALT = "ab2609"   # fixed: the recorded runs used it, and it makes every byte repeatable
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega").split()
QUESTIONS = [
    "How many record lines are in the text above? Answer with a number only.",
    "What is the value on the line whose id ends in 7? Answer with the number only.",
    "Name the three words that appear on the first record line, in order.",
    "Which record index is the largest in the text? Number only.",
    "What is the first word of the last record line?",
    "How many distinct words did you see across the first ten lines? Number only.",
]
SIDE_REMINDER = ("\n<system-reminder>\nThe user sent a new message while you "
                 "were working:\nplease also check the lines that end in 3\n"
                 "</system-reminder>")
SIDE_QUESTION = ("Describe your most recent action in 3-5 words using present "
                 "tense (-ing). Do not use tools.")


def document(salt, session, target_tokens):
    """Unique, non-compressible record lines; deterministic in (salt, session)."""
    rnd = random.Random(f"{salt}/{session}")
    lines, chars, i = [], 0, 0
    while chars < int(target_tokens * 2.05):
        words = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(8, 14)))
        line = (f"[record {i:07d} run {salt} id {rnd.randint(10**7, 10**8 - 1)}] "
                f"{words} value={rnd.random():.9f}")
        lines.append(line)
        chars += len(line) + 1
        i += 1
    return "\n".join(lines)


def ask(label, messages, max_tokens):
    body = json.dumps({"model": "halogen-qwen3.8-flash-next", "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0, "stream": False,
                       "enable_thinking": False}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.loads(r.read())
    wall = time.perf_counter() - t0
    u = resp.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    print(f"  {label:10s} prompt {u.get('prompt_tokens'):7d}  cached {cached:7d}  "
          f"{wall:7.2f} s", flush=True)
    return (resp["choices"][0]["message"].get("content") or ""), cached


def main():
    with urllib.request.urlopen(URL + "/health", timeout=10) as r:
        h = json.loads(r.read())
    with urllib.request.urlopen(URL + "/cache", timeout=10) as r:
        c = json.loads(r.read())
    print(f"server {h.get('version')}  pool {h.get('kv_pool_positions')}  "
          f"slots {h.get('slots')}  cache entries {c.get('entries')}")
    if h.get("kv_pool_positions") != 524288:
        sys.exit("needs HALOGEN_KV_POOL_POSITIONS=524288: the layout depends on it")
    if c.get("entries"):
        sys.exit("the prompt cache is not empty: restart the server first, the "
                 "layout depends on the pool starting empty")

    fillers = []
    for i in (0, 1):
        h_ = [{"role": "user", "content": document(SALT, 100 + i, 200000)
               + "\n\nAnswer with the word OK only."}]
        ans, _ = ask(f"F{i + 1} t0", h_, 64)
        h_.append({"role": "assistant", "content": ans})
        fillers.append(h_)

    main_h = [{"role": "user", "content": document(SALT, 0, 40000) + "\n\n" + QUESTIONS[0]}]
    found = {}
    for t in range(6):
        if t:
            main_h.append({"role": "user", "content":
                           document(f"{SALT}/pad{t}", 0, 4000) + "\n\n" + QUESTIONS[t]})
        ans, _ = ask(f"MAIN t{t}", main_h, 12000)
        main_h.append({"role": "assistant", "content": ans})
        if t >= 1:
            side = [dict(m) for m in main_h]
            side[2]["content"] += SIDE_REMINDER
            side.append({"role": "user", "content": SIDE_QUESTION})
            ask(f"SIDE t{t}", side, 12000)
        cold = []
        for i, h_ in enumerate(fillers):
            h_.append({"role": "user", "content": "Answer with OK only."})
            ans, cached = ask(f"F{i + 1} t{t}", h_, 64)
            h_.append({"role": "assistant", "content": ans})
            if cached == 0:
                cold.append(f"F{i + 1}")
        if t in (3, 5):
            found[t] = "F1" in cold

    print()
    print("FINDING 1 (round 3): " + (
        "REPRODUCED — F1's live 192k conversation came back cold; its region was "
        "forgotten for SIDE t3, which needed 3,840 positions more than it held."
        if found[3] else "not reproduced — F1 kept its cache at round 3."))
    print("FINDING 2 (round 5): " + (
        "REPRODUCED — F1 came back cold again; SIDE t5's region was not grown by "
        "dropping its own longer entries, a live neighbour was forgotten instead."
        if found[5] else "not reproduced — F1 kept its cache at round 5 "
        "(expected on 0.12.3, which drops SIDE's own longer entries there)."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
