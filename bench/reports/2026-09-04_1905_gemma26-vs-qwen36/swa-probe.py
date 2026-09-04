#!/usr/bin/env python3
"""What is --swa-full worth on gemma26, on the real Claude Code body?

setup/README.md's 100.2 s -> 10.4 s was measured on LAGUNA. The MECHANISM
transfers to any model whose window is shorter than the block Claude Code
appends behind the question (gemma4's is 1024, the block is ~1624), but the
MAGNITUDE does not — and on this model --swa-full costs 24.96 GiB at
-c 131072, which is far too much to spend on another model's number.

Four requests against one server, thinking OFF so the wall time is about the
prefix rather than about 4096 tokens of reasoning (part A of gemma-probe.py):

    1  cold                — nothing to reuse
    2  same question again — the ceiling: everything reusable
    3  CHANGED question    — the case that decides, divergence ~1624 tokens
                             from the end, i.e. outside a 1024 window
    4  changed AGAIN       — confirms 3 was not a one-off

Reuse is read from `usage.input_tokens` / `usage.cache_read_input_tokens`;
/v1/messages does NOT carry it in `timings`, and reading it there returns 0/0,
which reads as "nothing cached" rather than as a wrong field name.
"""
import json, os, sys, time, urllib.request

sys.path.insert(0, "@REPO@/tools")
from synthetic import body                                       # noqa: E402

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8081")
LABEL = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
H = {"content-type": "application/json", "anthropic-version": "2023-06-01"}


def turn(label, question):
    p = body(project="/tmp/projGemma", n_tools=24, question=question)
    p["max_tokens"] = 16
    p["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(URL + "/v1/messages",
                                 data=json.dumps(p).encode(), headers=H)
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    wall = time.time() - t0
    u = r.get("usage", {})
    pn = u.get("input_tokens", 0)
    cn = u.get("cache_read_input_tokens", 0)
    print("  %-24s new=%6d  cache=%6d  (%5.1f %% reused)  %6.1f s"
          % (label, pn, cn, 100.0 * cn / (pn + cn) if (pn + cn) else 0.0, wall))


print("=== %s ===" % LABEL)
turn("1 cold", "Explain what the auth module does.")
turn("2 same question", "Explain what the auth module does.")
turn("3 CHANGED question", "Explain what the billing module does.")
turn("4 changed again", "Explain what the reporting module does.")
