#!/usr/bin/env python3
"""Two controls that pin down WHY the post-answer restore carried nothing.

  A  post-answer state + the SAME question it was saved with
     (state = prefix+Q1+A1, prompt = prefix+Q1 — the state is longer)
  B  prefix-only state + a DIFFERENT question
     (state = prefix, prompt = prefix+Qx — the state is a true prefix)

If B reuses and A does not, the rule is: a restored state helps only where it
is a PREFIX of the incoming prompt, and llama.cpp does not trim a longer one
back to the common part.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools")
import synthetic as SYN
GW, LL = "http://127.0.0.1:8090", "http://127.0.0.1:8080"

def post(url, payload, timeout=900):
    r = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())

def ask(model, project, question, tag):
    b = SYN.body(project=project, n_tools=24, question=question)
    b["model"] = model; b["max_tokens"] = 16; b["stream"] = False
    t0 = time.time(); d = post(GW + "/v1/messages", b); secs = time.time() - t0
    u = d.get("usage", {})
    print("  %-46s %6.1f s   reused=%-6s computed=%-6s"
          % (tag, secs, u.get("cache_read_input_tokens"), u.get("input_tokens")),
          flush=True)

def restore(name):
    d = post(LL + "/slots/0?action=restore", {"filename": name + ".bin"})
    print("  restored %-16s %s tokens" % (name, d.get("n_restored")), flush=True)

print("=== A  post-answer state + the SAME question it was saved with")
restore("exp-postanswer")
ask("qwen38-low", "/tmp/postanswer",
    "Nenne die Hauptstadt von Frankreich, nur das Wort.", "A: same question as saved")

print("=== B  prefix-only state + a DIFFERENT question")
restore("b4124abc721a")
ask("qwen38", "/tmp/mode-cache", "Nenne die Hauptstadt von Spanien, nur das Wort.",
    "B: prefix-only, new question")
