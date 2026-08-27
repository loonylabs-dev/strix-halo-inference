#!/usr/bin/env python3
"""nachmessung — misst den korrigierten Proxy (Trennung stable/volatile).

The first version of fix=stable classified the whole agent-types block as
volatile and therefore never hoisted it.
This version splits the block up.
"""
import copy, json, os, re, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from measure import required, request_body

# Set CAPTURE to a captured body to measure against real data;
# without it the body is built synthetically.
CAPTURE = os.environ.get("CAPTURE")                          # noqa: E402


SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
W = os.path.dirname(os.path.abspath(__file__))
DROP = ("thinking", "context_management", "output_config")
FLUECHTIG = re.compile(r"<total_tokens>\s*\d+\s*tokens left\s*</total_tokens>")

def req(path, payload=None, method=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(SRV + path, data=data, method=method,
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        b = x.read().decode()
        return json.loads(b) if b.strip().startswith(("{", "[")) else b

def erase():
    for i in (0, 1):
        try:
            req("/slots/%d?action=erase" % i, payload={}, method="POST", timeout=60)
        except Exception:
            pass

def blocks_to_text(c):
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""

def split_volatile(text):
    fund = FLUECHTIG.findall(text)
    return FLUECHTIG.sub("", text).rstrip(), fund

def hoist_split(p):
    hoch, bleiben, seen = [], [], set()
    for m in p["messages"]:
        if isinstance(m, dict) and m.get("role") == "system":
            stable, fl = split_volatile(blocks_to_text(m.get("content")))
            if fl:
                bleiben.append({"role": "system", "content": "".join(fl)})
            if stable and stable not in seen:
                seen.add(stable); hoch.append(stable)
        else:
            bleiben.append(m)
    if not hoch:
        return p
    p["messages"] = bleiben
    schwanz = "\n\n".join(hoch)
    s = p.get("system")
    if isinstance(s, list):
        p["system"] = s + [{"type": "text", "text": "\n\n" + schwanz}]
    elif isinstance(s, str):
        p["system"] = s + "\n\n" + schwanz
    else:
        p["system"] = schwanz
    return p

def load(_unused, fix=False, question=None):
    """The base body, with the question replaced and the fix applied.

    The first argument used to be a capture file name. It is kept so the call
    sites read the same; the body now comes from tools/synthetic.py unless
    CAPTURE points at a real one.
    """
    p = request_body(question=question, capture=CAPTURE)
    if fix:
        p = hoist_split(p)
    return p

def run_one(label, p):
    t0 = time.time(); r = req("/v1/messages", p); dt = time.time() - t0
    u = r.get("usage", {})
    inp = required(u); cr = u.get("cache_read_input_tokens", 0)
    tot = inp + cr
    print("   %-40s new=%6d  cache=%6d  (%5.1f%%)  %6.1fs"
          % (label, inp, cr, 100.0 * cr / tot if tot else 0.0, dt))
    sys.stdout.flush()

print("=" * 96)
print("corrected proxy: SPLIT the stable and the volatile part")
print("=" * 96)

print("\nF · simple case, question changed")
erase()
run_one("F1 alpha (fills slot)", load("tool-001-roh.json", fix=True, question="Say only the word alpha."))
run_one("F2 beta  (changed question)", load("tool-001-roh.json", fix=True, question="Say only the word beta."))
run_one("F3 gamma (changed again)", load("tool-001-roh.json", fix=True, question="Say only the word gamma."))

print("\nG · tool conversation, 4 turns")
erase()
for i, n in enumerate(("tool-006-roh.json", "tool-007-roh.json",
                       "tool-008-roh.json", "tool-009-roh.json"), 1):
    run_one("G%d Turn %d" % (i, i), load(n, fix=True))
print()
