#!/usr/bin/env python3
"""bisect — takes Claude Code's real request body apart and measures which
part destroys partial prefix reuse.

Two runs per variant: once with question A (fills the slot), then with B.
Measured is the SECOND run — that is where the cache has to bite.
"""
import copy, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from measure import required, request_body

# Optional first argument: a captured body. Without it, synthetic.                          # noqa: E402


SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
W = os.path.dirname(os.path.abspath(__file__))
DROP = ("thinking", "context_management", "output_config")

def req(path, payload=None, method=None, timeout=3600):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(SRV + path, data=data, method=method,
                               headers={"content-type": "application/json",
                                        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        b = x.read().decode()
        return json.loads(b) if b.strip().startswith(("{", "[")) else b

def clear_slots():
    for i in (0, 1):
        try:
            req("/slots/%d?action=erase" % i, payload={}, method="POST", timeout=60)
        except Exception as e:
            print("   (erase slot %d: %s)" % (i, e))

def set_question(p, text):
    for m in reversed(p["messages"]):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                for b in reversed(c):
                    if isinstance(b, dict) and b.get("type") == "text":
                        b["text"] = text
                        return
            elif isinstance(c, str):
                m["content"] = text
                return

def run_one(p, timeout=3600):
    p = copy.deepcopy(p)
    p["stream"] = False
    p["max_tokens"] = 1
    t0 = time.time()
    r = req("/v1/messages", p, timeout=timeout)
    dt = time.time() - t0
    u = r.get("usage", {})
    inp = required(u)
    cr = u.get("cache_read_input_tokens", 0)
    return inp, cr, dt

# ---------- Varianten ----------
def v_original(p):  return p

def v_no_tools(p):
    p.pop("tools", None); return p

def v_no_sysmsg(p):
    p["messages"] = [m for m in p["messages"] if m.get("role") != "system"]; return p

def v_sysfield_string(p):
    s = p.get("system")
    if isinstance(s, list):
        p["system"] = "".join(b.get("text", "") for b in s if b.get("type") == "text")
    return p

def v_usercontent_string(p):
    for m in p["messages"]:
        c = m.get("content")
        if isinstance(c, list) and all(isinstance(b, dict) and b.get("type") == "text" for b in c):
            m["content"] = "\n".join(b.get("text", "") for b in c)
    return p

def v_no_metadata(p):
    p.pop("metadata", None); return p

def v_no_cachecontrol(p):
    def strip(o):
        if isinstance(o, dict):
            o.pop("cache_control", None)
            for v in o.values(): strip(v)
        elif isinstance(o, list):
            for v in o: strip(v)
    strip(p); return p

def v_fewer_tools(p):
    """Keep only the first three tools — probes a size threshold."""
    if p.get("tools"): p["tools"] = p["tools"][:3]
    return p

def v_tools_to_system_text(p):
    """Tools out, system text lengthened by the same amount instead —
    probes whether it is the size or the tools field that matters."""
    tools = p.pop("tools", None)
    if tools:
        filler = json.dumps(tools)
        s = p.get("system")
        if isinstance(s, list):
            p["system"] = s + [{"type": "text", "text": "\n\n" + filler}]
        else:
            p["system"] = (s or "") + "\n\n" + filler
    return p

VARIANTS = [
    ("V0 unchanged (control)",              v_original),
    ("V1 without tools",                    v_no_tools),
    ("V2 without the system message",       v_no_sysmsg),
    ("V3 system field as a string",         v_sysfield_string),
    ("V4 user content as a string",         v_usercontent_string),
    ("V5 without metadata",                 v_no_metadata),
    ("V6 without cache_control",            v_no_cachecontrol),
    ("V7 only 3 tools",                     v_fewer_tools),
    ("V8 tools -> text in the system field", v_tools_to_system_text),
]

def main():
    # An optional capture as the first argument; without one the body is built
    # by tools/synthetic.py, so this runs on a fresh checkout.
    capture = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("V") else None
    only = sys.argv[2:] if capture else sys.argv[1:]
    raw = request_body(capture=capture)

    print("body: %s" % (os.path.basename(capture) if capture else "synthetic"))
    print("%-34s %-22s %-22s" % ("variant", "run 1 (fills slot)", "run 2 (changed question)"))
    print("-" * 92)
    for name, fn in VARIANTS:
        if only and not any(name.startswith(n) for n in only):
            continue
        clear_slots()
        p = fn(copy.deepcopy(raw))
        set_question(p, "Say only the word alpha.")
        i1, c1, t1 = run_one(p)
        p2 = fn(copy.deepcopy(raw))
        set_question(p2, "Say only the word beta.")
        i2, c2, t2 = run_one(p2)
        tot2 = i2 + c2
        q = 100.0 * c2 / tot2 if tot2 else 0.0
        marke = "  <== CACHE GREIFT" if q > 50 else ""
        print("%-34s new=%6d c=%6d %5.1fs | new=%6d c=%6d (%5.1f%%) %6.1fs%s"
              % (name, i1, c1, t1, i2, c2, q, t2, marke))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
