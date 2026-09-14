#!/usr/bin/env python3
"""Does reasoning_effort change how much this model actually thinks?

The template difference is proven (three distinct prompts). This asks the
other half: whether the MODEL responds to it. Through the gateway, so the
single slot is respected and nothing else can run beside it.
"""
import json, os, sys, time, urllib.request

GW = "http://127.0.0.1:8090/v1/chat/completions"
TOK = open(os.path.expanduser("~/.config/llm-gateway-tokens")).read().split()[1]
ALIAS = "halogen-qwen3.8-flash-next"
LEVELS = ["low", "medium", "high"]
REPEATS = 3
MAX_TOKENS = 1536
PROMPTS = [
    ("arith", "A shop sells pens at 3 for 2 euros and notebooks at 5 euros "
              "each. I buy 12 pens and 3 notebooks. What do I pay?"),
    ("logic", "Three people each tell one truth and one lie. A says: 'B is a "
              "knight, C is a knave.' B says: 'A is a knave, C is a knight.' "
              "C says: 'A is a knight, B is a knave.' Who is what?"),
    ("code",  "A Python function reads a file, parses each line as JSON and "
              "returns the list. It works locally and fails in CI with a "
              "UnicodeDecodeError on line 1. Name the two most likely causes "
              "and how to tell them apart."),
]

def ask(model, prompt):
    body = json.dumps({"model": model, "max_tokens": MAX_TOKENS,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    r = urllib.request.Request(GW, data=body, headers={
        "content-type": "application/json", "authorization": "Bearer " + TOK})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=1800) as x:
        d = json.loads(x.read().decode())
    took = time.time() - t0
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = d.get("usage") or {}
    return {"took_s": round(took, 2),
            "finish": ch.get("finish_reason"),
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "content_chars": len(msg.get("content") or ""),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}

rows = []
print("level   prompt  rep  compl_tok  reas_chars  cont_chars  finish      s",
      flush=True)
for lvl in LEVELS:
    for name, prompt in PROMPTS:
        for rep in range(1, REPEATS + 1):
            try:
                r = ask("%s-%s" % (ALIAS, lvl), prompt)
            except Exception as e:
                print("%-7s %-7s %-4d FAILED %r" % (lvl, name, rep, e), flush=True)
                continue
            r.update(level=lvl, prompt=name, rep=rep)
            rows.append(r)
            print("%-7s %-7s %-4d %-10s %-11s %-11s %-11s %s"
                  % (lvl, name, rep, r["completion_tokens"], r["reasoning_chars"],
                     r["content_chars"], r["finish"], r["took_s"]), flush=True)

json.dump(rows, open(sys.argv[1], "w"), indent=1)
print("\n=== MEDIAN completion_tokens per level ===", flush=True)
import statistics
for lvl in LEVELS:
    per = [r["completion_tokens"] for r in rows if r["level"] == lvl]
    reas = [r["reasoning_chars"] for r in rows if r["level"] == lvl]
    if per:
        print("%-7s n=%d  completion median=%d  min=%d max=%d | reasoning chars median=%d"
              % (lvl, len(per), statistics.median(per), min(per), max(per),
                 statistics.median(reas)), flush=True)
print("DONE-MARKER", flush=True)
