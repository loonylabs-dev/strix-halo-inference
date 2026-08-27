#!/usr/bin/env python3
"""concurrency — what happens when several Claude Code sessions run against
the same llama-server at once.

Until now only the SEQUENTIAL case was measured. This is about:
  phase 1  warm up two projects (as many as there are slots)
  phase 2  query both AT THE SAME TIME
  phase 3  a third project on top — more projects than slots
  phase 4  back to the first two: did they survive?
  phase 5  a short foreign request (like title generation) in between
"""
import copy, json, os, sys, threading, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from measure import required, request_body

# Set CAPTURE to a captured body to measure against real data;
# without it the body is built synthetically and no capture is needed.
CAPTURE = os.environ.get("CAPTURE")                          # noqa: E402


SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
W = os.path.dirname(os.path.abspath(__file__))
DROP = ("thinking", "context_management", "output_config")

def req(path, payload=None, timeout=1800):
    d = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(SRV + path, data=d,
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())


def project(n, question):
    """The body for project n with the given question.

    Comes from CAPTURE if that environment variable points at a captured body,
    otherwise from tools/synthetic.py. Either way the project path really
    differs between n — request_body() refuses to hand back a body where it
    did not, because four identical prefixes would look like a measurement and
    be none.
    """
    return request_body(project="/tmp/proj%d" % n, question=question,
                        capture=CAPTURE)

def title_request():
    """Imitates title generation: short prompt, no tools."""
    return {"model": "laguna", "stream": False, "max_tokens": 1,
            "system": "Analyze this conversation and generate a short title.",
            "messages": [{"role": "user", "content": "Hallo, ich brauche Hilfe beim Debuggen."}]}

results = {}
lock = threading.Lock()

def run_one(label, p, still=False):
    t0 = time.time()
    r = req("/v1/messages", p)
    dt = time.time() - t0
    u = r.get("usage", {})
    inp = required(u)
    cr = u.get("cache_read_input_tokens", 0)
    tot = inp + cr
    q = 100.0 * cr / tot if tot else 0.0
    row = "   %-32s new=%6d cache=%6d (%5.1f%%) %7.1fs" % (label, inp, cr, q, dt)
    with lock:
        if not still:
            print(row); sys.stdout.flush()
        results[label] = (inp, cr, q, dt)
    return q

def at_once(jobs):
    """jobs: (label, payload) pairs — every one started at the same moment."""
    threads = []
    start = threading.Barrier(len(jobs))
    def work(label, p):
        start.wait()
        run_one(label, p, still=True)
    for label, p in jobs:
        t = threading.Thread(target=work, args=(label, p))
        threads.append(t); t.start()
    for t in threads:
        t.join()
    for label, _ in jobs:
        inp, cr, q, dt = results[label]
        print("   %-32s new=%6d cache=%6d (%5.1f%%) %7.1fs" % (label, inp, cr, q, dt))
    sys.stdout.flush()

# Guarded: everything below RUNS, and one of the first things it does is
# call urlopen against the server. On import that is a network request
# nobody asked for — the second of the three things tests/common.py
# forbids a file loaded by path from doing.
if __name__ == "__main__":
    print("=" * 92)
    print("parallel use — several sessions against one server")
    print("=" * 92)
    slots = req("/slots") if False else None
    import urllib.request as _u
    with _u.urlopen(SRV + "/slots", timeout=30) as x:
        n_slots = len(json.loads(x.read().decode()))
    print("Slots auf diesem Server: %d" % n_slots)

    print("\nphase 1 · warm up two projects, one after another")
    run_one("P1 warm up", project(1, "Say alpha."))
    run_one("P2 warm up", project(2, "Say alpha."))

    print("\nphase 2 · both AT ONCE, same question")
    at_once([("P1 at once", project(1, "Say alpha.")),
                  ("P2 at once", project(2, "Say alpha."))])

    print("\nphase 3 · third project on top, all three AT ONCE")
    at_once([("P1 with a third", project(1, "Say alpha.")),
                  ("P2 with a third", project(2, "Say alpha.")),
                  ("P3 new",         project(3, "Say alpha."))])

    print("\nphase 4 · back to P1 and P2 — did they survive?")
    run_one("P1 afterwards", project(1, "Say alpha."))
    run_one("P2 afterwards", project(2, "Say alpha."))

    print("\nphase 5 · short foreign request (like title generation) in between")
    run_one("title request", title_request())
    run_one("P1 after the title request", project(1, "Say alpha."))
    run_one("P2 after the title request", project(2, "Say alpha."))
    print()
