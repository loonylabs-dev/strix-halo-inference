#!/usr/bin/env python3
"""The user-visible property, measured: does the follow-up turn stay fast?

Until 28.08. a cold request scheduled the save immediately, in the background,
while the next turn was a measured median of 1.0 s away — and that turn was
evicted from the slot: 0.7 s -> 13.6 s, a factor of 19.

  turn 1  a fresh prefix, cold. Schedules a save.
  turn 2  one second later, same prefix. Must be WARM and fast.
  turn 3  after the dust settles, same prefix again.

Read `reused` in each line: a turn that was evicted shows reused=0.
"""
import json, sys, time, urllib.request
sys.path.insert(0, "tools")
import synthetic as SYN
GW = "http://127.0.0.1:8090"

def ask(project, question, tag):
    b = SYN.body(project=project, n_tools=24, question=question)
    b["model"] = "qwen38-low"; b["max_tokens"] = 8; b["stream"] = False
    r = urllib.request.Request(GW + "/v1/messages", data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json",
                 "anthropic-version": "2023-06-01"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=900) as x:
        d = json.loads(x.read().decode())
    u = d.get("usage", {})
    print("  %-28s %6.1f s   reused=%-6s computed=%-6s"
          % (tag, time.time() - t0, u.get("cache_read_input_tokens"),
             u.get("input_tokens")), flush=True)

P = "/tmp/turnproof-%d" % int(time.time())
ask(P, "Sag nur: eins.",  "turn 1 (cold, schedules a save)")
time.sleep(1.0)
ask(P, "Sag nur: zwei.",  "turn 2 (one second later)")
time.sleep(8.0)
ask(P, "Sag nur: drei.",  "turn 3 (after the save had its chance)")
