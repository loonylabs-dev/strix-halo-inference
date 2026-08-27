#!/usr/bin/env python3
"""slot-corruption — which ingredient makes this build produce '////'?

A consumer on a second machine received another session's answer, and the
server here was found returning endless slashes to everyone. Four cases,
each against a freshly restarted server, one variable at a time:

    seq-one-prefix     three requests, one prefix, sequential
    seq-two-prefixes   two prefixes alternating, sequential
    par-two-prefixes   two prefixes, concurrent
    seq-no-tools       one prefix, no tool block

Measured 26.08.2026 on qwen38 / ROCm / gfx1151, build b10577:

    -np 2   one prefix           clean
    -np 2   two prefixes seq     CORRUPT on the 2nd request
    -np 2   two prefixes par     CORRUPT 8/8
    -np 1   both two-prefix runs clean

So it is the SECOND SLOT, not concurrency — serialising every request in
the gateway did not help, and neither did disabling the RAM prompt cache.
That is the gfx1151 HIP race (llama.cpp #27579, root cause #27572), and
the mitigation is -np 1 in the profile.

CAUTION: a corrupting run leaves the server poisoned until it restarts.
This suite restarts it before every case, and the LAST case leaves it in
whatever state it produced — run a probe afterwards.

    python3 bench/suites/slot-corruption.py
    python3 bench/suites/slot-corruption.py seq-two-prefixes
"""
import json, subprocess, sys, threading, time, urllib.request

GW = "http://127.0.0.1:8090"


def restart():
    subprocess.run(["systemctl", "--user", "restart", "llama-user@qwen38"],
                   check=True)
    for _ in range(150):
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/slots", timeout=3)
            time.sleep(2)
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("server did not come back")


def make_body(which, nonce, n_tools=10, bulk_reps=700, sys_reps=300):
    system = ("You are assistant %s. " % which) + ("Directive %s. " % which) * sys_reps
    bulk = ("Handbook for workspace %s.\n" % which) + \
           ("Rule %s about this repository. " % which) * bulk_reps
    b = {"model": "qwen38", "stream": False, "max_tokens": 120,
         "messages": [{"role": "system", "content": system},
                      {"role": "user",
                       "content": "Reply with exactly this word and nothing "
                                  "else: " + nonce},
                      {"role": "user", "content": bulk}]}
    if n_tools:
        b["tools"] = [{"type": "function",
                       "function": {"name": "T%02d_%s" % (i, which),
                                    "description": "d " * 50,
                                    "parameters": {"type": "object"}}}
                      for i in range(n_tools)]
    return b


def ask(body_, timeout=900):
    r = urllib.request.Request(GW + "/v1/chat/completions",
                               data=json.dumps(body_).encode(),
                               headers={"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        resp = json.loads(x.read().decode())
    return (resp["choices"][0]["message"].get("content") or "").strip()


def verdict(text, nonce):
    if "////" in text or text.count("/") > 8:
        return "CORRUPT"
    return "ok" if nonce in text else ("empty" if not text else "other")


def case(name, run):
    restart()
    print("== %s" % name)
    outs = run()
    for nonce, text in outs:
        print("   %-18s %-8s %r" % (nonce, verdict(text, nonce), text[:36]))
    return any(verdict(t, n) == "CORRUPT" for n, t in outs)


def seq_one_prefix():
    out = []
    for i in range(3):
        n = "SEQ-%d-%d" % (int(time.time()) % 10000, i)
        out.append((n, ask(make_body("A", n))))
    return out


def seq_two_prefixes():
    out = []
    for i in range(2):
        for w in ("A", "B"):
            n = "%s-%d-%d" % (w, int(time.time()) % 10000, i)
            out.append((n, ask(make_body(w, n))))
    return out


def par_two_prefixes():
    out, lock = [], threading.Lock()
    def run(w):
        for i in range(2):
            n = "%s-%d-%d" % (w, int(time.time()) % 10000, i)
            t = ask(make_body(w, n))
            with lock:
                out.append((n, t))
    ts = [threading.Thread(target=run, args=(w,)) for w in ("A", "B")]
    for t in ts: t.start()
    for t in ts: t.join()
    return out


def seq_no_tools():
    out = []
    for i in range(3):
        n = "NOTOOL-%d-%d" % (int(time.time()) % 10000, i)
        out.append((n, ask(make_body("A", n, n_tools=0))))
    return out


CASES = {"seq-one-prefix": seq_one_prefix,
         "seq-two-prefixes": seq_two_prefixes,
         "par-two-prefixes": par_two_prefixes,
         "seq-no-tools": seq_no_tools}

names = sys.argv[1:] or list(CASES)
bad = {}
for n in names:
    bad[n] = case(n, CASES[n])
print("\nRESULT")
for n in names:
    print("  %-18s %s" % (n, "CORRUPT" if bad[n] else "clean"))
