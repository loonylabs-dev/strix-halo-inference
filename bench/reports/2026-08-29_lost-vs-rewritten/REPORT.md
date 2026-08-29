# Telling a lost state from a rewritten history

29.08.2026, late. Both show up as one thing — `reused` drops — and that
ambiguity cost two rounds of blaming the watchdog for what a client had done
to its own conversation (see 2026-08-29_probe-cost/REPORT.md, section
CORRECTION).

The gateway now records three numbers per request: `prev_in`, `msgs_prev`,
`msgs_kept`. Both halves of the distinction are proven against the running
stack, because a rule that has only been reasoned about is exactly what
produced the wrong answer in the first place.

## Half one: the client rewrites its history

Three turns; the third edits message 1 and appends.

    22:35:47  msgs 1  —       —       —          reused 4719   first turn
    22:35:48  msgs 3  prev 1  kept 1  in 4732    reused 4728   appended
    22:35:54  msgs 3  prev 3  kept 0  in 4754    reused 3602   rewritten

`kept 0 of 3` — the conversation diverges at its first message, so the drop in
reuse is the client's doing. Grey in the table: normal behaviour, nothing to
fix here.

## Half two: the history stands and the state is gone

Same shape, but instead of editing anything, the slot is ERASED between two
turns. Erasing is not a takeover, so llama.cpp's `prompt_save` never runs and
the RAM cache keeps nothing — which is precisely the situation the red verdict
is for.

    23:06:40  msgs 1  —       —       —          reused 4741   first turn
    23:06:42  msgs 3  prev 1  kept 1  in 4754    reused 4750   appended
    23:07:04  msgs 5  prev 3  kept 3  in 4776    reused    0   ZUSTAND VERLOREN

`kept 3 of 3` — every message came back unchanged — and reuse still collapsed
from 4776 to zero. 22.1 s against the 1.4 s of the untouched turn.

## What this settles, and what it does not

SETTLED: the two causes are now distinguishable from one record, without
reading three logs side by side. Both verdicts have fired against the real
server, not only in a test.

NOT SETTLED, and stated because the same gap misled me earlier:

* the comparison is per MESSAGE. A client that edits inside one long message
  marks that message whole, and a token-level split would need the rendered
  prompt, which is not worth keeping in the request path.
* `msgs_kept` says nothing about WHY a state was lost. It separates "not the
  cache's fault" from "the cache lost it" — which of the several possible
  evictors did it still needs the surrounding log.

## Reproduce

    python3 bench/reports/2026-08-29_lost-vs-rewritten/rewrite.py
    python3 bench/reports/2026-08-29_lost-vs-rewritten/lost.py
