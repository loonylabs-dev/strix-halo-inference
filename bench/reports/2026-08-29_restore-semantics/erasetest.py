#!/usr/bin/env python3
"""Does /completion with the prefix truncate a slot that holds prefix+Q+A —
or does it have to be erased first?

prewarm erases whenever the slot holds MORE than the prefix, and then pays a
full prefill. If llama.cpp keeps the common prefix instead, the erase is
unnecessary and the save is a write rather than a re-computation.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools"); sys.path.insert(0, "setup/claude")
import synthetic as SYN, dialects as DIA, prewarm as PW
LL, GW = "http://127.0.0.1:8080", "http://127.0.0.1:8090"

def post(url, payload, timeout=900):
    r = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())

def slots():
    with urllib.request.urlopen(LL + "/slots", timeout=30) as x:
        return [(s.get("id"), s.get("n_prompt_tokens")) for s in json.load(x)]

PROJECT = "/tmp/erasetest-%d" % int(time.time())
body = SYN.body(project=PROJECT, n_tools=24, question="Sag nur: eins.")

# 1. a turn, so the slot holds prefix + question + answer
b = json.loads(json.dumps(body)); b["model"]="qwen38-low"; b["max_tokens"]=8; b["stream"]=False
t0=time.time(); post(GW + "/v1/messages", b)
print("turn done in %.1f s, slots %s" % (time.time()-t0, slots()), flush=True)

# 2. render the prefix exactly as prewarm does
b2 = json.loads(json.dumps(body))
b2["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": "low"}
b2, _ = DIA.hoist_system_messages(b2, DIA.ANTHROPIC, PW.VOLATILE)
full = post(LL + "/apply-template", DIA.template_payload(b2, DIA.ANTHROPIC))["prompt"]
cut = full.find("<user>")
prefix = full[:cut if cut >= 0 else full.rfind("X")]
n_tok = len(post(LL + "/tokenize", {"content": prefix})["tokens"])
print("prefix: %d tokens" % n_tok, flush=True)

# 3. the question: send the prefix WITHOUT erasing first
t0 = time.time()
d = post(LL + "/completion", {"prompt": prefix, "n_predict": 1, "cache_prompt": True})
tm = d.get("timings", {})
print("  /completion without erasing: %.1f s   cache_n=%s prompt_n=%s"
      % (time.time()-t0, tm.get("cache_n"), tm.get("prompt_n")), flush=True)
print("  slots now %s (prefix is %d)" % (slots(), n_tok), flush=True)
