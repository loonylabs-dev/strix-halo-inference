#!/usr/bin/env python3
"""verification — measures the fix candidates against the running server.

Call after the server has been started with the configuration under test.
The server name is only used as a label, it is not checked.

  python3 verifikation.py "Konfigurationsname"
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

def hoist(p, mode):
    """mode: all = what cc-cachefix does; stable = leave the volatile ones"""
    msgs = p.get("messages")
    extra, keep, seen = [], [], set()
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            t = blocks_to_text(m.get("content"))
            if mode == "stable" and FLUECHTIG.search(t):
                keep.append(m); continue
            if t.strip() and t not in seen:
                seen.add(t); extra.append(t)
        else:
            keep.append(m)
    if not extra:
        return p
    p["messages"] = keep
    tail = "\n\n".join(extra)
    s = p.get("system")
    if isinstance(s, list):
        p["system"] = s + [{"type": "text", "text": "\n\n" + tail}]
    elif isinstance(s, str):
        p["system"] = s + "\n\n" + tail
    else:
        p["system"] = tail
    return p

def load(_unused, fix=None, question=None):
    """The base body, with the question replaced and the fix applied.

    The first argument used to be a capture file name. It is kept so the call
    sites read the same; the body now comes from tools/synthetic.py unless
    CAPTURE points at a real one.
    """
    p = request_body(question=question, capture=CAPTURE)
    if fix:
        p = hoist(p, fix)
    return p

def run_one(label, p):
    t0 = time.time(); r = req("/v1/messages", p); dt = time.time() - t0
    u = r.get("usage", {})
    inp = required(u); cr = u.get("cache_read_input_tokens", 0)
    tot = inp + cr
    q = 100.0 * cr / tot if tot else 0.0
    print("   %-40s new=%6d  cache=%6d  (%5.1f%%)  %6.1fs" % (label, inp, cr, q, dt))
    sys.stdout.flush()
    return q, dt

def main():
    kfg = sys.argv[1] if len(sys.argv) > 1 else "?"
    print("=" * 96)
    print("Konfiguration: %s" % kfg)
    print("=" * 96)

    print("\nA · simple case — same request, only the question changed (no proxy)")
    erase()
    run_one("A1 alpha (fills slot)", load("tool-001-roh.json", question="Say only the word alpha."))
    run_one("A2 beta  (changed question)", load("tool-001-roh.json", question="Say only the word beta."))

    print("\nB · tool conversation, 4 turns, WITHOUT a proxy")
    erase()
    for i, n in enumerate(("tool-006-roh.json", "tool-007-roh.json",
                           "tool-008-roh.json", "tool-009-roh.json"), 1):
        run_one("B%d Turn %d" % (i, i), load(n))

    print("\nC · tool conversation WITH cc-cachefix as today (fix=all)")
    erase()
    for i, n in enumerate(("tool-006-roh.json", "tool-007-roh.json",
                           "tool-008-roh.json", "tool-009-roh.json"), 1):
        run_one("C%d Turn %d" % (i, i), load(n, fix="all"))

    print("\nD · tool conversation WITH the corrected proxy (fix=stable)")
    erase()
    for i, n in enumerate(("tool-006-roh.json", "tool-007-roh.json",
                           "tool-008-roh.json", "tool-009-roh.json"), 1):
        run_one("D%d Turn %d" % (i, i), load(n, fix="stable"))

    print("\nE · Einfacher Fall MIT korrigiertem Proxy (fix=stable)")
    erase()
    run_one("E1 alpha (fills slot)", load("tool-001-roh.json", fix="stable",
                                        question="Say only the word alpha."))
    run_one("E2 beta  (changed question)", load("tool-001-roh.json", fix="stable",
                                             question="Say only the word beta."))
    run_one("E3 gamma (changed again)", load("tool-001-roh.json", fix="stable",
                                              question="Say only the word gamma."))

    print()

if __name__ == "__main__":
    main()
