#!/usr/bin/env python3
"""The prefix-first sequence, end to end, with the loop closed.

    A  prefill the prefix ALONE      the work the first request must do anyway
    B  save it                       a pure write: the state IS the prefix
    C  the first real request        must reuse the whole prefix
    D  displace the slot             something else takes it
    E  restore the file              what B wrote
    F  a question it has NEVER seen  must reuse the whole prefix again

F is the point: a file written this way has to serve a LATER session, not just
the request that happened to create it.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools"); sys.path.insert(0, "setup/claude")
import synthetic as SYN, dialects as DIA, prewarm as PW
LL, GW = "http://127.0.0.1:8080", "http://127.0.0.1:8090"
KW = {"enable_thinking": True, "reasoning_effort": "low"}
P = "/tmp/prefixfirst-fixed"
NAME = "exp-prefixfirst2"

def post(url, payload, timeout=900):
    r = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())

def slots():
    with urllib.request.urlopen(LL + "/slots", timeout=30) as x:
        return [(s.get("id"), s.get("n_prompt_tokens")) for s in json.load(x)]

def ask(project, question, tag, tools=24):
    b = SYN.body(project=project, n_tools=tools, question=question)
    b["model"] = "qwen38-low"; b["max_tokens"] = 8; b["stream"] = False
    t0 = time.time(); d = post(GW + "/v1/messages", b)
    u = d.get("usage", {})
    print("  %-40s %6.1f s   reused=%-6s computed=%-6s"
          % (tag, time.time() - t0, u.get("cache_read_input_tokens"),
             u.get("input_tokens")), flush=True)

b = SYN.body(project=P, n_tools=24, question="Sag nur: eins.")
b2 = json.loads(json.dumps(b)); b2["chat_template_kwargs"] = KW
b2, _ = DIA.hoist_system_messages(b2, DIA.ANTHROPIC, PW.VOLATILE)
full = post(LL + "/apply-template", DIA.template_payload(b2, DIA.ANTHROPIC))["prompt"]
cut = full.find("<user>")
prefix = full[:cut if cut >= 0 else full.rfind("X")]
n_tok = len(post(LL + "/tokenize", {"content": prefix})["tokens"])
print("prefix: %d tokens" % n_tok, flush=True)

t0 = time.time()
d = post(LL + "/completion", {"prompt": prefix, "n_predict": 1, "cache_prompt": True})
tm = d.get("timings", {})
print("A  prefix alone      %6.1f s   cache_n=%s prompt_n=%s   slot %s"
      % (time.time() - t0, tm.get("cache_n"), tm.get("prompt_n"), slots()), flush=True)

t0 = time.time()
r = post(LL + "/slots/0?action=save", {"filename": NAME + ".bin"})
print("B  save              %6.0f ms   n_saved=%s   %s"
      % ((time.time() - t0) * 1000, r.get("n_saved"),
         "MATCHES the prefix" if abs(r.get("n_saved", -1) - n_tok) <= 2 else "MISMATCH"),
      flush=True)

ask(P, "Sag nur: eins.", "C  the first real request")
ask("/tmp/prefixfirst-other", "Sag nur: ok.", "D  displacing request", tools=4)
d = post(LL + "/slots/0?action=restore", {"filename": NAME + ".bin"})
print("E  restore           %s tokens   slot %s" % (d.get("n_restored"), slots()), flush=True)
ask(P, "Wie viel ist 7 mal 6? Nur die Zahl.", "F  a question it has never seen")
