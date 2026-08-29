#!/usr/bin/env python3
"""What the watchdog costs the sessions it watches. From the log alone.

    python3 bench/suites/probe-cost.py 7        the last seven days

No GPU, no requests, no server needed — the numbers are already in
llama-server's journal. That matters here: the thing being measured is a
latency the OPERATOR feels, and measuring it by provoking it would add to it.

The signature, measured on 29.08.: the probe asks a fixed 34-token question,
so it leaves the slot holding ~34 tokens — and with -np 1 that is the whole
conversation gone. The NEXT request then re-prefills what it lost.

    task 8099  prompt eval 4217 tokens   release n_tokens = 62852
    task 8352  prompt eval    4 tokens   release n_tokens = 34      <- probe
    task 8389  prompt eval 18668 tokens  release n_tokens = 63141   <- the bill
"""
import re, subprocess, sys, datetime
from collections import defaultdict

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
out = subprocess.run(["journalctl", "--user", "-u", "llama-user@qwen38",
                      "--since", "-%dd" % DAYS, "--no-pager", "-o", "short-iso"],
                     capture_output=True, text=True).stdout

ev = re.compile(r"^(\S+) .*task (\d+) \| prompt eval time =\s+([0-9.]+) ms /\s+(\d+) tokens")
rel = re.compile(r"^(\S+) .*task (\d+) \| stop processing: n_tokens = (\d+)")
tasks, order = {}, []
for line in out.splitlines():
    m = ev.match(line)
    if m:
        t, tid, ms, n = m.groups()
        tasks[tid] = {"t": t, "eval_ms": float(ms), "eval_n": int(n)}
        order.append(tid)
        continue
    m = rel.match(line)
    if m and m.group(2) in tasks:
        tasks[m.group(2)]["after"] = int(m.group(3))

seq = [tasks[t] for t in order if "after" in tasks[t]]
probes, bills = [], []
for i, t in enumerate(seq):
    # the probe's own shape: it leaves the slot with about its own length
    if 20 <= t["after"] <= 60 and t["eval_n"] <= 40:
        before = seq[i-1]["after"] if i else 0
        nxt = seq[i+1] if i + 1 < len(seq) else None
        probes.append((t, before, nxt))

print("Zeitraum: %d Tage, %d Anfragen im Log\n" % (DAYS, len(seq)))
print("  %-20s %10s %12s %12s" % ("Zeitpunkt", "verdrängt", "danach neu", "Sekunden"))
tot_tok = tot_ms = 0
big = 0
for t, before, nxt in probes:
    if not nxt:
        continue
    cost_n, cost_ms = nxt["eval_n"], nxt["eval_ms"]
    if before >= 2000:                       # a real conversation was in there
        big += 1
        tot_tok += cost_n; tot_ms += cost_ms
        print("  %-20s %10d %12d %12.1f" % (t["t"][:19], before, cost_n, cost_ms/1000))
print("\n  Proben insgesamt: %d, davon auf ein laufendes Gespräch: %d" % (len(probes), big))
print("  danach neu gerechnet: %d Tokens in %.1f Minuten GPU-Zeit" % (tot_tok, tot_ms/60000))
