#!/usr/bin/env python3
"""cold-prefill — is one cold prefill at a time really the right limit?

The claim in setup/README.md is that two concurrent cold starts gain nothing
and only make each other slower. It was never measured. If it is wrong, the
gateway should let several through; if it is right, it should hold them back.

Also measures how the cold start scales with the size of the prefix — 4, 12 and
24 tools — so that "100 to 180 seconds" gets a shape instead of a range.

Talks straight to llama-server (8080), no gateway.

    python3 bench/suites/cold-prefill.py
"""
import json, os, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
from synthetic import body                                # noqa: E402

SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
STAMP = int(time.time())

def cold(project, n_tools=24, max_tokens=8):
    p = body(project=project, n_tools=n_tools, question="Say alpha.")
    p["model"], p["max_tokens"], p["stream"] = "laguna", max_tokens, False
    r = urllib.request.Request(SRV + "/v1/messages", data=json.dumps(p).encode(),
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=900) as x:
        a = json.loads(x.read().decode())
    u = a.get("usage", {})
    return {"s": time.time() - t0, "new": u.get("input_tokens", 0),
            "cached": u.get("cache_read_input_tokens", 0)}

print("=" * 78)
print("cold prefills — one at a time, or several?")
print("=" * 78)

print("\n1 · how the cold start scales with the size of the prefix")
for n in (4, 12, 24):
    m = cold("/tmp/cp-size%d-%d" % (n, STAMP), n_tools=n)
    print("   %2d tools  %6d tokens new  %6.1f s   %5.0f tok/s prefill"
          % (n, m["new"], m["s"], m["new"] / m["s"] if m["s"] else 0))

print("\n2 · two cold prefills, one after the other")
seq = []
t0 = time.time()
for i in (1, 2):
    m = cold("/tmp/cp-seq%d-%d" % (i, STAMP))
    seq.append(m["s"])
    print("   run %d   %6.1f s" % (i, m["s"]))
seq_wall = time.time() - t0
print("   wall clock for both: %.1f s" % seq_wall)

print("\n3 · two cold prefills at the same time")
res = {}
def run(k):
    res[k] = cold("/tmp/cp-par%d-%d" % (k, STAMP))
ts = [threading.Thread(target=run, args=(i,)) for i in (1, 2)]
t0 = time.time()
for t in ts: t.start()
for t in ts: t.join()
par_wall = time.time() - t0
for k in (1, 2):
    print("   run %d   %6.1f s" % (k, res[k]["s"]))
print("   wall clock for both: %.1f s" % par_wall)

print("\n   sequential  %6.1f s" % seq_wall)
print("   parallel    %6.1f s   (factor %.2f)" % (par_wall, par_wall / seq_wall if seq_wall else 0))
if par_wall < seq_wall * 0.9:
    print("   -> running them together IS faster: do not serialise prefills")
elif par_wall > seq_wall * 1.1:
    print("   -> running them together is SLOWER: at most one prefill at a time")
else:
    print("   -> it makes no difference; serialise for the simpler latency story")
print("=" * 78)
