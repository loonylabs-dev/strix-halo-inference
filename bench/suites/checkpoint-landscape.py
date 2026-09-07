#!/usr/bin/env python3
"""checkpoint-landscape — what the checkpoint grid does to a GROWING chat.

The question this answers (operator, 07.09.2026): production qwen36 shows warm
requests whose history is provably unchanged falling back thousands of tokens
and re-prefilling them. The trace names the symptom `state lost`; the cause is
llama.cpp's context checkpoints, and `--checkpoint-min-step` (default 8192) is
the only knob over them.

The measured fact this suite is built on: on 07.09. every one of that day's
`reused` plateaus was a checkpoint position in the slot's sidecar. So the
checkpoint LANDSCAPE — how many checkpoints survive and how far the newest one
sits behind the end — is the quantity to compare, not the wall time of one
lucky request. Latency is an effect; the landscape is the cause, and it is
stable enough to A/B.

Why a growing chat rather than checkpoint-grid.py's fixed triplet: that suite
models an agent re-sending ONE prompt with a changed task text, and it settled
the qwen38 case at ~8k tokens. What hurts here is different — a conversation
that grows past the min-step several times over, which is what a coding agent
does and what the 07.09. trace contains.

    python3 bench/sideserver.py --env setup/env/qwen36.env --port 8082 \
        --stop llama-user@qwen36 \
        --extra "-c 65536 --checkpoint-min-step 512 --slot-save-path DIR" \
        -- python3 bench/suites/checkpoint-landscape.py \
             --label cms512 --save-dir DIR --out REPORT/cms512.json

One cell per grid value, same sequence, same seed. The suite writes its rows as
JSON so a report carries the numbers verbatim rather than a retyped summary.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ckptsidecar                                            # noqa: E402

# Deterministic filler. Not lorem ipsum: a repeating vocabulary tokenises
# predictably, so a turn's length in TOKENS tracks its length in words across
# runs, which is what makes two cells comparable. The seed is the turn index —
# there is no RNG here at all, deliberately (bench scripts must reproduce).
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
         "kilo lima mike november oscar papa quebec romeo sierra tango "
         "uniform victor whiskey xray yankee zulu").split()


def filler(seed, n_words):
    return " ".join(WORDS[(seed * 7 + i * 13) % len(WORDS)] for i in range(n_words))


def post(base, path, body, timeout):
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return time.monotonic() - t0, json.loads(resp.read())


def reuse_of(obj):
    """(reused, computed) from llama.cpp's own accounting, or (None, None).

    `timings.cache_n` / `prompt_n` is the server measuring itself. Nothing here
    derives a number from a duration — a derived reuse figure would be the one
    thing this suite must not report.
    """
    t = obj.get("timings")
    if isinstance(t, dict) and isinstance(t.get("cache_n"), int):
        return t["cache_n"], t.get("prompt_n")
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8082")
    ap.add_argument("--label", default="run")
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--head-words", type=int, default=4200,
                    help="opening user message; ~1.35 tokens per word here")
    ap.add_argument("--turn-words", type=int, default=700,
                    help="each follow-up user message")
    ap.add_argument("--short-tokens", type=int, default=60)
    ap.add_argument("--long-tokens", type=int, default=400,
                    help="every LONG-EVERY turn answers this long, so the slot "
                         "carries a real generated tail — the tail is what a "
                         "checkpoint cannot cover")
    ap.add_argument("--long-every", type=int, default=4)
    ap.add_argument("--intruder-every", type=int, default=0,
                    help="every Nth turn, send a small UNRELATED prompt first. "
                         "This models what actually shares the slot in "
                         "production: llama-probe fires every 10 minutes "
                         "straight at port 8080, past the gateway, and at "
                         "`-np 1` there is no second slot for it to land in "
                         "(measured 07.09.2026: a session-save right after a "
                         "probe recorded n_saved=30). 0 disables it. Without "
                         "an intruder a plain append NEVER falls back — "
                         "measured, cell cms8192, 24 of 24 turns at reused = "
                         "prev - 4 — so a grid comparison without this option "
                         "compares three healthy runs and proves nothing.")
    ap.add_argument("--pause-every", type=int, default=0,
                    help="idle for --pause-s before every Nth turn. MEASURED "
                         "07.09.2026 and the reason this option exists: in "
                         "production every one of the six state-lost rows had "
                         "a gap in front of it (20, 31, 38, 45, 178, 468 s) "
                         "while healthy turns came 1-3 s apart. Only 2 of the "
                         "6 gaps contained a probe run, so the idleness "
                         "itself is the candidate, not what fills it.")
    ap.add_argument("--pause-s", type=float, default=45.0)
    ap.add_argument("--save-dir", default=None,
                    help="the server's --slot-save-path; the sidecar is read "
                         "from here after the run")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default=None, help="write the rows as JSON here")
    a = ap.parse_args(argv)

    msgs = [{"role": "system",
             "content": "You are a terse assistant. Answer in one short sentence."},
            {"role": "user",
             "content": "Here is the project log.\n" + filler(0, a.head_words)
                        + "\nAcknowledge in one sentence."}]

    rows = []
    for turn in range(a.turns):
        if a.pause_every and turn and turn % a.pause_every == 0:
            print("[%s]  idle %.0fs before turn %d" % (a.label, a.pause_s, turn),
                  flush=True)
            time.sleep(a.pause_s)

        # The intruder goes FIRST, so the turn that follows it is the one whose
        # reuse we read — that is the shape production has: the probe fires,
        # and the next real request pays for it.
        if a.intruder_every and turn and turn % a.intruder_every == 0:
            try:
                itook, iout = post(a.base, "/v1/chat/completions",
                                   {"messages": [{"role": "user", "content":
                                                  "What is the capital of France? "
                                                  "Answer with one word."}],
                                    "max_tokens": 8, "temperature": 0.0,
                                    "seed": 7, "stream": False}, a.timeout)
                ir, ic = reuse_of(iout)
                rows.append({"turn": turn, "intruder": True,
                             "took_s": round(itook, 2),
                             "reused": ir, "computed": ic})
                print("[%s]  intruder before turn %d  took=%.1fs computed=%s"
                      % (a.label, turn, itook, ic), flush=True)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print("[%s] intruder FAILED: %s" % (a.label, e), flush=True)

        want = (a.long_tokens if a.long_every and turn % a.long_every == a.long_every - 1
                else a.short_tokens)
        body = {"messages": msgs, "max_tokens": want, "temperature": 0.0,
                "seed": 1234, "stream": False}
        try:
            took, out = post(a.base, "/v1/chat/completions", body, a.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print("[%s] turn %d FAILED: %s" % (a.label, turn, e), flush=True)
            break
        reused, computed = reuse_of(out)
        answer = out["choices"][0]["message"]["content"]
        usage = out.get("usage") or {}
        rows.append({"turn": turn, "took_s": round(took, 2),
                     "reused": reused, "computed": computed,
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"),
                     "want_tokens": want})
        print("[%s] turn %2d  took=%6.1fs  reused=%-8s computed=%-7s in=%s"
              % (a.label, turn, took, reused, computed,
                 usage.get("prompt_tokens")), flush=True)

        msgs.append({"role": "assistant", "content": answer})
        msgs.append({"role": "user",
                     "content": "Next note.\n" + filler(turn + 1, a.turn_words)
                                + "\nAcknowledge in one sentence."})

    result = {"label": a.label, "rows": rows}

    # The landscape itself. A save is the only way to see it from outside the
    # server — /slots reports the prompt, never the checkpoints.
    if a.save_dir and rows:
        name = "landscape-%s.bin" % a.label
        try:
            took, _ = post(a.base, "/slots/0?action=save", {"filename": name}, a.timeout)
            path = os.path.join(a.save_dir, name)
            cps = ckptsidecar.read(path + ".ckpt")
            result["checkpoints"] = cps
            result["summary"] = ckptsidecar.summarise(cps)
            result["state_bytes"] = os.path.getsize(path)
            s = result["summary"]
            print("[%s] SAVED in %.1fs — %d checkpoint(s), %.2f GiB sidecar "
                  "beside %.2f GiB state, newest gap %s, largest gap %s"
                  % (a.label, took, s["count"], s["bytes"] / 1073741824,
                     result["state_bytes"] / 1073741824,
                     s["tail_gap"], s["max_gap"]), flush=True)
            for c in cps:
                print("    n_tokens=%-8d %7.1f MiB" % (c["n_tokens"], c["size"] / 1048576),
                      flush=True)
        except (urllib.error.URLError, TimeoutError, OSError,
                ckptsidecar.SidecarError) as e:
            # NOT fatal and NOT silently dropped: the per-turn rows are still
            # the measurement, and a missing landscape has to be visible in the
            # report rather than read as "no checkpoints".
            result["checkpoints_error"] = str(e)
            print("[%s] sidecar unavailable: %s" % (a.label, e), flush=True)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)
        print("[%s] wrote %s" % (a.label, a.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
