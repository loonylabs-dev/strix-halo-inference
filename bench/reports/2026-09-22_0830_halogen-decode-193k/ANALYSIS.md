# Decode at 193k: 0.12.3 is 10.6% faster than 0.8.1

**Measured 22.09.2026** on this machine (Ryzen AI MAX+ 395, 128 GiB, gfx1151),
`bench/sessionload.py --long-answer`, `platform_profile=performance` verified at
both ends of both runs. Identical documents (fixed salt), thinking off, one
session, four turns, `max_tokens=1200`, `HALOGEN_KV_POOL_POSITIONS=524288`,
`slot_ctx=262144`. Reservation 38% of the pool, so nothing here is about
eviction.

## Why this was run separately

The 21.09. comparison
(`../2026-09-21_1630_halogen-0.8.1-vs-0.12.3/ANALYSIS.md`) measured prefill and
cache placement and said so — its answers were 2 to 12 tokens long, chosen so
decode time would not bury what it was after. That left DECODE unmeasured, and
decode is what the operator's own sessions spend their time on: they run between
60k and 230k of context with answers of hundreds to thousands of tokens.
Upstream's 0.12.0 claims the gain there (serial 30.0 -> 35.2 tok/s at 262k,
43.5 -> 49.4 with the draft head), and a claim is not a measurement on this box.

## Result

| | 0.8.1 | 0.12.3 |
|---|---|---|
| turn 1 (cold prefill) | 28.69 tok/s | 33.84 tok/s |
| turn 2 | 30.53 | 32.91 |
| turn 3 | 29.42 | 32.43 |
| turn 4 | 31.38 | 33.48 |
| **median** | **30.0 tok/s** | **33.2 tok/s** |
| cold prefill (same prompt) | 198.6 s | 194.7 s |
| wall, four turns | 354.0 s | 333.2 s |

**+10.6% decode.** On a 1,000-token answer at this depth that is 33.3 s against
30.1 s. Prefill is 2% apart, which is within what a single pair of runs can
distinguish and is consistent with upstream saying 32k prefill is unchanged.

## The first attempt measured the wrong thing, and it is worth writing down

The first version of `--long-answer` asked the SAME question every turn. From
turn two the previous answer is in the history, so Prompt Lookup Decoding quotes
it back:

| turn | decode | PLD rounds | commit/round |
|---|---|---|---|
| 1 | 33.47 tok/s | 18 | 1.49 |
| 2 | 56.69 | 243 | 3.18 |
| 3 | 59.57 | 251 | 3.44 |

Those 56-59 tok/s are real in the sense that the server produced them, and on
real coding work — where an answer legitimately repeats context — they are the
rate an operator gets. They are not a decode rate. Reporting them as one would
have claimed a 70% improvement from an upgrade that delivers 10%.

The fix: four different questions, each about a subject the document cannot
supply (the document is random words and numbers, so prose about canal locks
cannot be n-gram-matched out of it). PLD rounds fell from 243-251 to 9-26 and
the rates became flat. **The served log is where the real mixture shows up**
(`bench/logstats.py`); this mode deliberately isolates the floor.

## A label that would have been read as a broken measurement

The 0.8.1 request lines say `235 tok beside other streams` on a run with ONE
session, which reads like contention that would have invalidated the comparison.
It is not: 0.12.3 renamed exactly that field to `with the head off`, because it
counts tokens produced with the drafter's head off — which includes the adaptive
policy's rest stretches on a lone stream. Same state, honest name. Without the
0.12.3 wording this run would have looked contaminated.

## What does NOT follow

* **n=1 per cell**, four turns each. A 10.6% difference from one pair of runs is
  suggestive, not settled; the four turns within each run agree to about 3%,
  which is why it is reported at all.
* **Nothing about 262k or above**, where upstream locates the larger part of the
  gain. These prompts sit at 193k.
* **Nothing about the flat band.** 90% of served turns are under 2,000 tokens
  and this says nothing about them.
* **Not a claim about PLD.** The 56-59 tok/s figure above is a real served rate
  under self-quotation and it was measured on 0.12.3 only; whether 0.8.1 reaches
  the same under the same self-quotation is unmeasured.
