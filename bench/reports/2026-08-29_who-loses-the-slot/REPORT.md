# Who actually takes the slot, and what it costs

29.08.2026, 23:12. Two experiments, four turns each, on the running stack.
They invert what I wrote into the defect register two hours earlier.

## The probe: absorbed, costs nothing

    A1 cold                              11.0 s   reused 4743
    A2 appended                           1.3 s   reused 4752
    PROBE (tiny, straight to llama-server) 1.2 s  -> selected slot by LRU
    A3 continues the conversation         1.1 s   reused 4774  computed 26

The watchdog took the slot and the next turn did not notice. 4774 of 4778
tokens came back.

## A second project: 2050 tokens gone

Same shape, but the interloper is another synthetic project — a prefix that
shares most of its tokens with the first.

    A3 after the similar prefix          10.2 s   reused 2750  computed 2050

The trace flagged it red: `msgs_kept 3 of 3` — the history was untouched — and
`reused 2750` against `prev_in 4778`. The state was lost.

## Why the harmless one is the one that looks dangerous

llama.cpp decides in two places (server-context.cpp):

    LRU branch     no slot resembles the new prompt
                   -> update_cache = true, the outgoing state is SAVED

    LCP branch     a slot resembles it
                   f_keep = kept / slot length
                   if (f_keep < 0.5f) update_cache = true;
                   -> above half, NOTHING is saved: the tail is trimmed away

So the rule is the opposite of the intuition:

    a completely unrelated request   is safe   — it triggers the save
    a HALF-similar request           is costly — it keeps just enough of the
                                                 slot that llama.cpp does not
                                                 bother to save the rest

The probe is unrelated by construction: 26 tokens, its own question, nothing
in common. It always lands in the LRU branch, and the conversation it displaces
goes into the 32 GiB RAM cache on the way out.

## What this means for this stack

Two Claude Code projects share the system prompt and the tool block, so
switching between them keeps well over half of the slot — and everything past
the common part is dropped without a cache entry and without a log line. That
is the shape that costs, and it is invisible in every log this repo had before
today: `reused` drops, and nothing says whether the client or the machine did
it.

The trace's new columns name it now: history intact, state lost, red.

## What is NOT settled

* The 50 % threshold is llama.cpp's, not ours, and `--cache-ram` does not
  change it. Whether a larger cache would help is therefore the wrong
  question — the state is never offered to the cache in that path.
* How often this happens in real use is unmeasured. It needs the new columns
  and a few days of normal work, not another experiment.

## Reproduce

    python3 bench/reports/2026-08-29_who-loses-the-slot/probe-shaped.py
    python3 bench/reports/2026-08-29_who-loses-the-slot/similar-prefix.py
