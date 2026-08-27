#!/usr/bin/env python3
"""multiproject — checks how many projects stay warm at the same time.

A "project" is a working directory to Claude Code. It appears in two places
in the system prompt (characters ~2,538 and ~4,670 of 6,081), so two projects
differ from character 2,538 on — the entire tool block sits behind that.
dahinter ist damit projektspezifisch.

Sequence:
  phase 1  warm up all N projects, one after another
  phase 2  the same question in rotation — has to hit 100 %
  phase 3  a changed question in rotation — has to hit partially
"""
import copy, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from measure import required, request_body

# Set CAPTURE to a captured body to measure against real data;
# without it the body is built synthetically and no capture is needed.
CAPTURE = os.environ.get("CAPTURE")                          # noqa: E402


SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
W = os.path.dirname(os.path.abspath(__file__))
DROP = ("thinking", "context_management", "output_config")
N = int(os.environ.get("N_PROJEKTE", 4))

def req(path, payload=None, method=None, timeout=1800):
    d = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(SRV + path, data=d, method=method,
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        b = x.read().decode()
        return json.loads(b) if b.strip().startswith(("{", "[")) else b


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

def run_one(label, p):
    t0 = time.time()
    r = req("/v1/messages", p)
    dt = time.time() - t0
    u = r.get("usage", {})
    inp = required(u)
    cr = u.get("cache_read_input_tokens", 0)
    tot = inp + cr
    q = 100.0 * cr / tot if tot else 0.0
    print("   %-34s new=%6d cache=%6d (%5.1f%%) %7.1fs" % (label, inp, cr, q, dt))
    sys.stdout.flush()
    return q

def gtt():
    # card1 used to be hard-coded here — on the next machine the GPU is
    # card0, and then this script silently measures nothing.
    from measure import gtt_gib
    return gtt_gib()

print("=" * 92)
print("keeping several projects warm at once — %d projects, --swa-full" % N)
print("=" * 92)
print("GTT before warming up: %.1f GiB" % gtt())

print("\nphase 1 · warm up every project, one after another")
for i in range(1, N + 1):
    run_one("P%d warm up" % i, project(i, "Say only the word alpha."))
print("GTT danach: %.1f GiB" % gtt())

print("\nphase 2 · in rotation, same question — has to hit fully")
treffer = []
for round_ in (1, 2):
    for i in range(1, N + 1):
        treffer.append(run_one("R%d P%d same question" % (round_, i),
                            project(i, "Say only the word alpha.")))

print("\nphase 3 · in rotation, changed question — has to hit partially")
part = []
for i in range(1, N + 1):
    part.append(run_one("P%d changed question" % i, project(i, "Say only the word beta.")))

print("\n" + "=" * 92)
print("result")
print("  same question:    %d of %d queries above 90 %% cache" %
      (sum(1 for t in treffer if t > 90), len(treffer)))
print("  changed question: %d of %d queries above 50 %% cache" %
      (sum(1 for t in part if t > 50), len(part)))
print("  GTT am Ende: %.1f GiB" % gtt())
