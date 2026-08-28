#!/usr/bin/env python3
"""What IS in 8774f83a80be.bin, if not the low rendering?

Restore it once, then ask each candidate rendering in turn and time it. The
one that comes back in seconds is what the file holds. Every candidate is the
same synthetic body, only the mode differs — and after each probe the file is
restored again, so every candidate meets the same slot.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools"); sys.path.insert(0, "setup/claude")
import synthetic as SYN
GW, LL = "http://127.0.0.1:8090", "http://127.0.0.1:8080"

def post(url, payload, timeout=900):
    r = urllib.request.Request(url, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())

def ask(model):
    b = SYN.body(project="/tmp/mode-cache", n_tools=24, question="Say alpha.")
    b["model"] = model; b["max_tokens"] = 8; b["stream"] = False
    t0 = time.time(); post(GW + "/v1/messages", b); return time.time() - t0

NAME = "8774f83a80be"
for model in ("qwen38-low", "qwen38", "qwen38-high", "qwen38-medium"):
    d = post(LL + "/slots/0?action=restore", {"filename": NAME + ".bin"})
    secs = ask(model)
    print("  restored %s (%s tokens) -> %-14s %6.1f s  %s"
          % (NAME, d.get("n_restored"), model, secs,
             "<<< THIS IS WHAT THE FILE HOLDS" if secs < 15 else ""), flush=True)
