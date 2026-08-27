#!/usr/bin/env python3
"""four-sessions — the realistic case: 3 projects, 4 sessions, growing histories.

  S1  project A, standard tool set
  S2  project A, ONE tool fewer     -> different system prompt, own prefix
  S3  project B
  S4  project C

So four different prefixes with only two slots. Each session runs through
three turns of a tool conversation (the history grows, pure appending), and
at the end each gets a CHANGED question — that needs rolling back and is the
hard case after a restore from the RAM cache.
"""
import copy, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from measure import required, request_body

# Set CAPTURE to a captured body to measure against real data;
# without it the body is built synthetically.
CAPTURE = os.environ.get("CAPTURE")                          # noqa: E402


SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
W = os.path.dirname(os.path.abspath(__file__))
DROP = ("thinking", "context_management", "output_config")

def req(path, payload, timeout=1800):
    r = urllib.request.Request(SRV + path, data=json.dumps(payload).encode(),
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())


# S1..S4: (project name, how many tools to leave out)
SESSIONS = [
    ("S1 ProjA full",      "projA", 0),
    ("S2 ProjA -1 Tool",   "projA", 1),
    ("S3 ProjB",           "projB", 0),
    ("S4 ProjC",           "projC", 0),
]

BASE_TOOLS = 24          # what a captured Claude Code body carries

def build(_unused, project, tools_dropped, question=None):
    """One session's body: its own project, optionally one tool fewer.

    Dropping a tool is what makes S2 a different prefix from S1 while sharing
    a long start — the pathological case this suite exists for.
    """
    return request_body(project="/tmp/" + project,
                        n_tools=BASE_TOOLS - tools_dropped,
                        question=question, capture=CAPTURE)

def run_one(label, p):
    t0 = time.time()
    r = req("/v1/messages", p)
    dt = time.time() - t0
    u = r.get("usage", {})
    inp = required(u)
    cr = u.get("cache_read_input_tokens", 0)
    tot = inp + cr
    q = 100.0 * cr / tot if tot else 0.0
    marke = "" if q > 90 else ("  <== teilweise" if q > 20 else "  <== KALT")
    print("   %-26s new=%6d cache=%6d (%5.1f%%) %7.1fs%s"
          % (label, inp, cr, q, dt, marke))
    sys.stdout.flush()
    return q

print("=" * 92)
print("four sessions, three projects, two slots")
print("=" * 92)

print("\nTurn 1 · all four for the first time (each pays its own cold start)")
for label, proj, dropped in SESSIONS:
    run_one(label, build("tool-006-roh.json", proj, dropped))

print("\nturn 2 · in rotation, history grown (pure appending)")
for label, proj, dropped in SESSIONS:
    run_one(label, build("tool-007-roh.json", proj, dropped))

print("\nturn 3 · in rotation, history grown further")
for label, proj, dropped in SESSIONS:
    run_one(label, build("tool-008-roh.json", proj, dropped))

print("\nturn 4 · in rotation, history grown further still")
for label, proj, dropped in SESSIONS:
    run_one(label, build("tool-009-roh.json", proj, dropped))

print("\nhard case · new session in the same project, DIFFERENT question")
print("(needs rolling back — the test after a restore from the RAM cache)")
for label, proj, dropped in SESSIONS:
    run_one(label, build("tool-006-roh.json", proj, dropped, question="Say only the word delta."))
print()
