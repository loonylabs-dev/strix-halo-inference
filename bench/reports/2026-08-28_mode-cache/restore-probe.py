#!/usr/bin/env python3
"""Does a restore from disk actually spare the server the prefill?

Three steps per file, and the third is the answer:
  1. put a DIFFERENT prefix in the slot, so nothing can be reused
  2. restore the file the way cc-gateway does it (POST /slots/N?action=restore)
  3. send the body that file was saved for, and read `prompt eval ... N tokens`
     out of llama-server's own log. ~4 tokens means the restore was used;
     ~15000 means it was not.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools"); sys.path.insert(0, "setup/claude")
import synthetic as SYN

GW = "http://127.0.0.1:8090"
LL = "http://127.0.0.1:8080"

def post(url, payload, timeout=900):
    r = urllib.request.Request(url, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())

def ask(model):
    b = SYN.body(project="/tmp/mode-cache", n_tools=24, question="Say alpha.")
    b["model"] = model; b["max_tokens"] = 8; b["stream"] = False
    t0 = time.time()
    post(GW + "/v1/messages", b)
    return time.time() - t0

def restore(name, slot=0):
    t0 = time.time()
    d = post(LL + "/slots/%d?action=restore" % slot, {"filename": name + ".bin"})
    return d.get("n_restored", -1), (time.time() - t0) * 1000

for name, model in (("8774f83a80be", "qwen38-low"),
                    ("b4124abc721a", "qwen38")):
    print("=== %s  (the file saved for %s)" % (name, model), flush=True)
    other = "qwen38-medium"
    print("  1. filling the slot with %s ... %.1f s" % (other, ask(other)), flush=True)
    n, ms = restore(name)
    print("  2. restored %s tokens in %.0f ms" % (n, ms), flush=True)
    secs = ask(model)
    print("  3. the request the file was saved for: %.1f s" % secs, flush=True)
    print("     -> %s" % ("RESTORE USED" if secs < 15 else "RESTORE IGNORED — full prefill"),
          flush=True)
