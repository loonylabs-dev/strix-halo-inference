#!/usr/bin/env python3
"""checkpoint-grid — what does a changed start prompt cost, per checkpoint grid?

Context (setup/defects.json, hybrid-checkpoints-round-partial-reuse-down):
two prompts share system+tools (8,081 tokens here) and differ only in the
short user text behind them. A hybrid slot cannot be trimmed to the
divergence; it falls back to the last context checkpoint, and the default
`--checkpoint-min-step 8192` guarantees that checkpoint sits up to one ubatch
too early — measured as 2,050 recomputed tokens for a ~170-token difference.
Agent setups pay that on every call with a fresh task text.

This suite drives one fixed sequence of three bodies (a, b, c — same head,
three different start prompts) against one server and reports the wall time
and the server-reported usage per request. Run it once per server
configuration under bench/sideserver.py, varying only --checkpoint-min-step:

    python3 bench/sideserver.py --env setup/env/qwen38.env --port 8082 \
        --stop llama-user@qwen38 --extra "-c 32768 --checkpoint-min-step 512" \
        -- python3 bench/suites/checkpoint-grid.py --bodies <dir> --label cms512

The hard reuse numbers (reused/computed per request) live in the transient
unit's journal (`prompt processing`, `selected slot`, `release` lines); the
report pairs them with this suite's client-side rows.
"""

import argparse
import json
import time
import urllib.request


def post(base, body, path, timeout=240.0):
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return time.monotonic() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8082")
    ap.add_argument("--bodies", required=True,
                    help="directory holding dsh-a.json, dsh-b.json, dsh-c.json")
    ap.add_argument("--label", default="run")
    ap.add_argument("--path", default="/v1/chat/completions",
                    help="endpoint path; MUST match the client being modelled "
                         "— an OpenAI-format body sent to /v1/messages loses "
                         "its tools silently (measured 31.08.2026: 2,309 "
                         "tokens rendered instead of 8,251)")
    ap.add_argument("--seq", default="a,j,a,j,b,j,c",
                    help="request sequence over the bodies dsh-<x>.json. "
                         "The default is the shape production traffic had: "
                         "main prompt, judge call, an identical repeat, judge, "
                         "then a CHANGED start prompt. The order matters: "
                         "b straight after a reuses to the exact divergence "
                         "(measured reused=8074, computed=177), while the "
                         "same b after a repeat-and-judge history landed on a "
                         "checkpoint 1,880 tokens earlier in production. "
                         "Every reprocessing lays new end checkpoints and the "
                         "min-step eviction prunes old ones, so the history "
                         "IS the variable.")
    args = ap.parse_args()

    rows = []
    for step in args.seq.split(","):
        step = step.strip()
        body = json.load(open("%s/dsh-%s.json" % (args.bodies.rstrip("/"), step)))
        took, out = post(args.base, body, args.path)
        usage = out.get("usage", {})
        rows.append((step, took))
        print("[%s] %s  took=%.1fs  usage=%s"
              % (args.label, step, took, json.dumps(usage, sort_keys=True)))

    print("[%s] SUMMARY %s" % (args.label,
          "  ".join("%s=%.1fs" % (s, t) for s, t in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
