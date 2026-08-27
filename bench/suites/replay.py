#!/usr/bin/env python3
"""replay — plays a captured Claude Code body back against llama-server.

  --body      file with the raw request body
  --fix    none   unchanged, apart from the fields that would 400
           all    what cc-cachefix did: EVERY system message into the system field
           stable    hoist only non-volatile system messages,
                     volatile ones (counters etc.) stay where they are
           drop   remove volatile system messages entirely, hoist the rest
  --question  replaces the last text block of the last user message
  --dump   schreibt den gerenderten Prompt out /slots hierhin
  --label     label for the log line
"""
import argparse, json, os, re, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from measure import required                          # noqa: E402


SRV = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
DROP = ("thinking", "context_management", "output_config")

# What counts as "volatile": content that changes from turn to turn.
FLUECHTIG = re.compile(r"<total_tokens>\s*\d+\s*tokens left\s*</total_tokens>")

def blocks_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c
                       if isinstance(b, dict) and b.get("type") == "text")
    return ""

def hoist(p, mode):
    msgs = p.get("messages")
    if not isinstance(msgs, list):
        return p, 0, 0
    extra, keep, seen = [], [], set()
    n_fluechtig = 0
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            t = blocks_to_text(m.get("content"))
            volatil = bool(FLUECHTIG.search(t))
            if volatil:
                n_fluechtig += 1
            if mode == "all":
                pass                      # alles hochziehen
            elif mode == "stable" and volatil:
                keep.append(m); continue  # fluechtige stehen lassen
            elif mode == "drop" and volatil:
                continue                  # fluechtige verwerfen
            if t.strip() and t not in seen:
                seen.add(t); extra.append(t)
        else:
            keep.append(m)
    if not extra:
        return p, 0, n_fluechtig
    p["messages"] = keep
    tail = "\n\n".join(extra)
    s = p.get("system")
    if isinstance(s, list):
        p["system"] = s + [{"type": "text", "text": "\n\n" + tail}]
    elif isinstance(s, str):
        p["system"] = s + "\n\n" + tail
    else:
        p["system"] = tail
    return p, len(extra), n_fluechtig

def post(path, payload=None, timeout=3600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(SRV + path, data=data,
                                 headers={"content-type": "application/json",
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(path, timeout=60):
    with urllib.request.urlopen(SRV + path, timeout=timeout) as r:
        return json.loads(r.read().decode())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--fix", default="none", choices=["none", "all", "stable", "drop"])
    ap.add_argument("--question", default=None)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--maxtok", type=int, default=1)
    a = ap.parse_args()

    p = json.load(open(a.body))
    for k in DROP:
        p.pop(k, None)
    p["stream"] = False
    p["max_tokens"] = a.maxtok

    if a.question is not None:
        for m in reversed(p["messages"]):
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for b in reversed(m["content"]):
                    if isinstance(b, dict) and b.get("type") == "text":
                        b["text"] = a.question
                        break
                break

    n_hoist, n_volatile = 0, 0
    if a.fix != "none":
        p, n_hoist, n_volatile = hoist(p, a.fix)

    t0 = time.time()
    r = post("/v1/messages", p)
    dt = time.time() - t0
    u = r.get("usage", {})
    inp = required(u)
    cr  = u.get("cache_read_input_tokens", 0)
    tot = inp + cr
    rate = (100.0 * cr / tot) if tot else 0.0
    print("%-34s fix=%-6s new=%7d  cache=%7d  (%5.1f%%)  %7.1fs   hoisted=%d volatile=%d"
          % (a.label or os.path.basename(a.body), a.fix, inp, cr, rate, dt, n_hoist, n_volatile))

    if a.dump:
        for s in get("/slots"):
            pr = s.get("prompt")
            if pr:
                open(a.dump, "w").write(pr)
                print("   -> prompt (%d chars) written to %s" % (len(pr), a.dump))
                break
        else:
            print("   -> no prompt in /slots. This suite needs a server")
            print("      started with LLAMA_SERVER_SLOTS_DEBUG=1. That hands out")
            print("      the complete prompts of every slot — locally only and")
            print("      only for the measurement, see docs/SECURITY.md.")

if __name__ == "__main__":
    main()
