#!/usr/bin/env python3
"""Can a slot be saved AS IT STANDS, after an answer — and still serve the
next question from its prefix?

If yes, the whole save policy is unnecessary: no re-rendering, no prefill, no
residency question, no race for the one slot. `prewarm` only re-creates the
prefix because it wants to save the prefix ALONE.

Two things have to hold, and the second is the one prewarm.py:158-179 warns
about in as many words ("puts a foreign user turn in front of the model, which
then answers the old question"):

    PERFORMANCE   the next question reuses the prefix (cache_n ~ prefix tokens)
    CORRECTNESS   the answer is to the NEW question, not the saved one
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools")
import synthetic as SYN

GW, LL = "http://127.0.0.1:8090", "http://127.0.0.1:8080"
NAME = "exp-postanswer"

def post(url, payload, timeout=900):
    r = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as x:
        return json.loads(x.read().decode())

def get(url):
    with urllib.request.urlopen(url, timeout=60) as x:
        return json.loads(x.read().decode())

def ask(model, question, tag):
    b = SYN.body(project="/tmp/postanswer", n_tools=24, question=question)
    b["model"] = model; b["max_tokens"] = 32; b["stream"] = False
    t0 = time.time()
    d = post(GW + "/v1/messages", b)
    secs = time.time() - t0
    u = d.get("usage", {})
    txt = " ".join(c.get("text", c.get("thinking", ""))
                   for c in d.get("content", []) if isinstance(c, dict))
    print("  %-34s %6.1f s   reused=%-6s computed=%-6s  %r"
          % (tag, secs, u.get("cache_read_input_tokens"), u.get("input_tokens"),
             txt[:90]), flush=True)
    return secs, u, txt

def slots():
    s = get(LL + "/slots")
    return [(x.get("id"), x.get("n_prompt_tokens")) for x in s]

print("=== 1. a first answer, so the slot holds prefix + question + answer")
ask("qwen38-low", "Nenne die Hauptstadt von Frankreich, nur das Wort.", "Q1 (fills the slot)")
print("     slots:", slots(), flush=True)

print("=== 2. save the slot AS IT STANDS")
t0 = time.time()
d = post(LL + "/slots/0?action=save", {"filename": NAME + ".bin"})
print("     saved %s tokens, %.0f MB, %.0f ms wall"
      % (d.get("n_saved"), d.get("n_written", 0) / 1e6, (time.time() - t0) * 1000),
      flush=True)

print("=== 3. displace the slot with a different prefix")
ask("qwen38-medium", "Sag nur: ok.", "displacing request")
print("     slots:", slots(), flush=True)

print("=== 4. restore the post-answer state")
t0 = time.time()
d = post(LL + "/slots/0?action=restore", {"filename": NAME + ".bin"})
print("     restored %s tokens, %.0f ms" % (d.get("n_restored"),
                                            (time.time() - t0) * 1000), flush=True)
print("     slots:", slots(), flush=True)

print("=== 5. THE QUESTION: same prefix, DIFFERENT question")
ask("qwen38-low", "Wie viel ist 7 mal 6? Nur die Zahl.", "Q2 after the restore")

print("=== 6. control: another question, now warm from the slot")
ask("qwen38-low", "Nenne die Hauptstadt von Italien, nur das Wort.", "Q3 (control, warm)")

print("=== 7. correctness: restore again, then ask something unmistakable")
post(LL + "/slots/0?action=restore", {"filename": NAME + ".bin"})
ask("qwen38-low", "Antworte nur mit dem Wort Banane.", "Q4 after a fresh restore")
