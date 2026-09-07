#!/usr/bin/env python3
"""Compare checkpoint-landscape cells — one table, no retyped numbers.

    python3 bench/suites/checkpoint-landscape-compare.py REPORT/*.json

Reads what checkpoint-landscape.py wrote and answers the two questions a grid
change turns on:

  * DOES IT HELP — how often a warm turn fell back further than the four
    tokens a healthy `end-4` checkpoint costs, and how many tokens that cost
    in total. `computed` is llama.cpp's own count of tokens it had to process;
    the excess over the turn's genuinely new tokens is the waste.
  * WHAT IT COSTS — how many checkpoints survive and how many bytes they are.
    At production depth one qwen36 checkpoint is 62.8 MiB of recurrent target
    state plus a draft state that grows with position (measured 07.09.2026:
    15.6 MiB at 7.9k tokens, 371.0 MiB at 188k), so the count alone understates
    it badly and the bytes are the number to compare.

A cell whose sidecar could not be read prints its error rather than a zero: a
missing landscape is not an empty one.
"""
import argparse
import json

# Below this a "fallback" is not one. `prompt_tokens + completion_tokens` of the
# previous turn undercounts what the slot holds, because the chat template adds
# role markers between the messages that neither figure carries. MEASURED on
# cell cms8192 (24 healthy turns, every one of them at reused = prev_in - 4 in
# llama.cpp's own accounting): the residual is 435 tokens over 23 turns, i.e.
# ~19 per turn. 64 sits well above that and far below the thousands a real
# checkpoint fallback costs. It is a floor for THIS template and turn shape,
# not a constant of the stack.
FLOOR = 64


def fallbacks(rows):
    """(count, wasted_tokens) — turns that re-prefilled more than they added.

    A healthy warm turn re-processes the four tokens behind its end checkpoint
    plus whatever is genuinely new (the previous answer and the new message).
    We cannot see "genuinely new" directly, but the PREVIOUS turn's prompt plus
    its completion is exactly the boundary: anything the server recomputed
    below that boundary is state it already had and threw away.
    """
    count, wasted = 0, 0
    prev = None
    for r in rows:
        # An intruder row is not a turn of the conversation: it must never
        # become the baseline the next turn is measured against, or the very
        # turn this suite exists to catch would be silently excluded.
        if r.get("intruder"):
            continue
        if (prev and r.get("reused") is not None
                and prev.get("prompt_tokens") is not None):
            # what the slot held after the previous turn
            held = prev["prompt_tokens"] + (prev.get("completion_tokens") or 0)
            behind = held - r["reused"]
            if behind > FLOOR:
                count += 1
                wasted += behind
        prev = r
    return count, wasted


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+", help="the per-cell JSON files")
    a = ap.parse_args(argv)

    cells = []
    for path in a.cells:
        with open(path) as f:
            cells.append(json.load(f))
    cells.sort(key=lambda c: -int("".join(ch for ch in c["label"] if ch.isdigit()) or 0))

    print("%-10s %6s %8s %10s %12s %8s %10s %10s" %
          ("cell", "turns", "falls", "wasted tok", "prefill s", "ckpts",
           "sidecar", "newest gap"))
    for c in cells:
        rows = c["rows"]
        n_fall, wasted = fallbacks(rows)
        s = c.get("summary") or {}
        secs = sum(r.get("took_s") or 0 for r in rows)
        sidecar = ("%.2f GiB" % (s["bytes"] / 1073741824) if s.get("bytes")
                   else (c.get("checkpoints_error", "-")[:20]))
        print("%-10s %6d %8d %10d %12.1f %8s %10s %10s" %
              (c["label"], len(rows), n_fall, wasted, secs,
               s.get("count", "-"), sidecar, s.get("tail_gap", "-")))

    print("\nfalls  = warm turns that re-prefilled state the slot already held")
    print("wasted = how many tokens those turns re-processed")
    print("newest gap = tokens between the two newest checkpoints; the distance")
    print("             a turn falls back by when its own end checkpoint is pruned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
