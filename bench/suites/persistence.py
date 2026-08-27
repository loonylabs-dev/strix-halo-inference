#!/usr/bin/env python3
"""persistence — can a saved slot state survive a server restart?

What used to be measured was only the roll-back case: restore, then a
CHANGED question. That one fails, because the slot file holds only the global
layers (28 KiB per token instead of 102).

This is about the case that matters in practice: restore, then APPEND.
Reines Anhaengen braucht kein Zurueckrollen — es koennte also trotzdem tragen.
"""
import json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from measure import required                          # noqa: E402


SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
# The repo, derived from this file rather than written down. It used to be
# the absolute path of one clone, which made every suite here unusable
# anywhere else — including from a second checkout on the same machine.
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(W, "tools"))
from synthetic import body                       # noqa: E402

STATE_FILE = "project-prefix.bin"

def req(path, payload=None, method=None, t=1800):
    d = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(SRV + path, data=d, method=method,
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=t) as x:
        b = x.read().decode()
        return json.loads(b) if b.strip().startswith(("{", "[")) else b

def wait_ready(t=600):
    for _ in range(t // 3):
        try:
            req("/props", t=5); return True
        except Exception:
            time.sleep(3)
    return False

def measure_one(label, p):
    p = dict(p); p["stream"] = False; p["max_tokens"] = 1
    t0 = time.time()
    r = req("/v1/messages", p)
    dt = time.time() - t0
    u = r.get("usage", {})
    new, cache = required(u), u.get("cache_read_input_tokens", 0)
    ges = new + cache
    q = 100.0 * cache / ges if ges else 0.0
    marke = "" if q > 90 else ("  <== teilweise" if q > 20 else "  <== KALT")
    print("   %-40s new=%6d cache=%6d (%5.1f%%) %7.1fs%s"
          % (label, new, cache, q, dt, marke))
    sys.stdout.flush()
    return q

def clear_slots():
    for i in (0, 1):
        try:
            req("/slots/%d?action=erase" % i, {}, "POST", 60)
        except Exception:
            pass

def busy_slot():
    for s in req("/slots"):
        if s.get("n_prompt_tokens"):
            return s["id"]
    return None

P1 = body(project="/tmp/persist", question="Say alpha.", turns=1)
P2 = body(project="/tmp/persist", question="Say alpha.", turns=2)   # reines Anhaengen
P3 = body(project="/tmp/persist", question="Say beta.",  turns=1)   # Rueckroll-Fall

print("=" * 96)
print("can a saved slot state survive a restart?")
print("=" * 96)

print("\nA · warm the slot up and save it")
clear_slots()
measure_one("turn1 cold", P1)
sid = busy_slot()
r = req("/slots/%d?action=save" % sid, {"filename": STATE_FILE}, "POST", 900)
print("   gesichert: Slot %s, %d Token, %d Bytes, %.0f ms"
      % (sid, r["n_saved"], r["n_written"], r["timings"]["save_ms"]))
print("   %.1f KiB je Token" % (r["n_written"] / r["n_saved"] / 1024))

print("\nB · erase the slot, restore, IDENTICAL request (control)")
clear_slots()
t0 = time.time()
r = req("/slots/0?action=restore", {"filename": STATE_FILE}, "POST", 900)
print("   restored: %d tokens, %.0f ms" % (r["n_restored"], r["timings"]["restore_ms"]))
measure_one("turn1 identisch", P1)

print("\nC · restore, then APPEND — the decisive case")
clear_slots()
req("/slots/0?action=restore", {"filename": STATE_FILE}, "POST", 900)
measure_one("turn2 (haengt an turn1 an)", P2)

print("\nD · restore, then a CHANGED question (known to fail)")
clear_slots()
req("/slots/0?action=restore", {"filename": STATE_FILE}, "POST", 900)
measure_one("turn1 with a different question", P3)

print("\nE · real service restart, then restore and append")
print("   restarting the service …")
subprocess.run(["systemctl", "--user", "restart", "llama-user@laguna"], check=False)
time.sleep(5)
if not wait_ready():
    print("   server did not come back"); raise SystemExit(1)
print("   server is back")
t0 = time.time()
r = req("/slots/0?action=restore", {"filename": STATE_FILE}, "POST", 900)
print("   restored: %d tokens, %.0f ms" % (r["n_restored"], r["timings"]["restore_ms"]))
measure_one("turn2 after the restart", P2)
measure_one("turn1 after the restart (identical)", P1)
print()
