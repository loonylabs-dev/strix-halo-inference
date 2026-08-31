#!/usr/bin/env python3
"""prefill-decode — does one session's prefill overlap with another's decode?

Two questions this suite exists to answer:

  1. Does batching decode help on this MoE model, or does it only split the
     throughput? "~24 t/s for everyone together" does not say what a single
     session gets on its own. On a dense model batching is nearly free; on a
     117B MoE two tokens may route to different experts, so the GPU reads more
     weights rather than the same ones.

  2. Does a warm request get through while another session's cold prefill runs?
     The measured "98 s wait despite a 100 % cache" may have been the gateway's
     own admission limit (MAX_INFLIGHT=2) rather than the GPU.

Talks to llama-server DIRECTLY (8080), bypassing the gateway, so that the
gateway's admission control cannot confound the result. Bodies come from
tools/synthetic.py — no capture needed.

    python3 bench/suites/prefill-decode.py
"""
import json, os, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))       # .../bench/suites
REPO = os.path.dirname(os.path.dirname(HERE))          # .../inference-stack
sys.path.insert(0, os.path.join(REPO, "tools"))
from synthetic import body                                # noqa: E402

SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")

def post(payload, timeout=600):
    d = json.dumps(payload).encode()
    r = urllib.request.Request(SRV + "/v1/messages", data=d,
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=timeout) as x:
        answer = json.loads(x.read().decode())
    return answer, time.time() - t0

def ask(project, n_tools, max_tokens, question="Say alpha."):
    p = body(project=project, n_tools=n_tools, question=question)
    p["model"], p["max_tokens"], p["stream"] = "laguna", max_tokens, False
    answer, seconds = post(p)
    u = answer.get("usage", {})
    return {"new": u.get("input_tokens", -1),
            "cached": u.get("cache_read_input_tokens", 0),
            "out": u.get("output_tokens", 0),
            "seconds": seconds}

def rate(m):
    return m["out"] / m["seconds"] if m["seconds"] else 0.0

def cache_pct(m):
    total = m["new"] + m["cached"]
    return 100.0 * m["cached"] / total if total > 0 else 0.0

def line(label, m):
    print("   %-38s %5.1f t/s  %6.1f s  out=%3d  cache=%5.1f %%"
          % (label, rate(m), m["seconds"], m["out"], cache_pct(m)))
    sys.stdout.flush()

print("=" * 78)
print("prefill / decode — measured directly against llama-server, no gateway")
print("=" * 78)

# ---------------------------------------------------------------- part 1 ---
# Two small, distinct prefixes so that warming them costs seconds, not minutes.
# Decode throughput is dominated by reading weights, not by context length.
print("\n1 · warm two small prefixes")
for proj in ("/tmp/pd-a", "/tmp/pd-b"):
    line("warm up %s" % proj, ask(proj, 4, 1))

print("\n2 · decode alone")
alone = ask("/tmp/pd-a", 4, 200, "Count slowly from one to sixty, one number per line.")
line("one session", alone)

print("\n3 · decode with two sessions at once")
results = {}
def run(key, proj):
    results[key] = ask(proj, 4, 200, "Count slowly from one to sixty, one number per line.")
threads = [threading.Thread(target=run, args=("a", "/tmp/pd-a")),
           threading.Thread(target=run, args=("b", "/tmp/pd-b"))]
t0 = time.time()
for t in threads: t.start()
for t in threads: t.join()
wall = time.time() - t0
line("session A", results["a"])
line("session B", results["b"])
total_out = results["a"]["out"] + results["b"]["out"]
print("   %-38s %5.1f t/s  %6.1f s  (aggregate over the wall clock)"
      % ("both together", total_out / wall, wall))

print("\n   alone   %5.1f t/s" % rate(alone))
print("   together %4.1f t/s" % (total_out / wall))
if total_out / wall > rate(alone) * 1.25:
    print("   -> batching decode GAINS throughput: share, do not serialise")
elif total_out / wall < rate(alone) * 1.05:
    print("   -> batching decode gains nothing: serialising gives lower latency")
else:
    print("   -> batching gains little; latency and throughput are close to a wash")

# ---------------------------------------------------------------- part 2 ---
print("\n4 · baseline: warm request while the GPU is idle")
base = ask("/tmp/pd-a", 4, 8)
line("warm, idle", base)

print("\n5 · warm request injected into a running cold prefill")
cold_project = "/tmp/pd-cold-%d" % int(time.time())
cold = {}
def run_cold():
    cold["m"] = ask(cold_project, 24, 8)
th = threading.Thread(target=run_cold)
t0 = time.time()
th.start()
print("   cold prefill started (%s, 24 tools) — waiting 10 s" % cold_project)
time.sleep(10)
injected = ask("/tmp/pd-a", 4, 8)
inj_at = time.time() - t0
line("warm, injected after 10 s", injected)
th.join()
line("the cold prefill itself", cold["m"])

print()
if cache_pct(injected) < 90:
    print("   INCONCLUSIVE: the injected request came back cold (%.1f %% cache) —"
          % cache_pct(injected))
    print("   the cold prefill evicted its slot. Run again.")
else:
    factor = injected["seconds"] / base["seconds"] if base["seconds"] else 0
    print("   warm alone      %6.2f s" % base["seconds"])
    print("   warm alongside  %6.2f s   (%.1fx slower, finished %.0f s into the prefill)"
          % (injected["seconds"], factor, inj_at))
    if factor < 3:
        print("   -> prefill and decode DO overlap. The 98 s in the docs were the")
        print("      gateway's admission limit, not the GPU.")
    elif factor < 20:
        print("   -> they overlap, but the prefill sets the pace: decode crawls")
        print("      along at the rate of the prefill chunks.")
    else:
        print("   -> effectively no overlap: the prefill occupies the GPU.")
print("=" * 78)
