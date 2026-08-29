# What the watchdog costs the sessions it watches

29.08.2026, from llama-server's own journal — no GPU, no requests. The thing
being measured is a latency the operator feels, and provoking it to measure it
would add to it.

## How it was found

A live trace row showed a turn at 70 % cache while its neighbours ran at
97–99 %:

    21:19:44  warm  167.0 s  in 60184  reused 55967  93 %
    21:23:23  warm  218.8 s  in 63069  reused 44401  70 %   <-
    21:24:54  warm   90.1 s  in 64821  reused 63141  97 %

The server's log says what happened in between:

    task 8099   prompt eval  4217 tokens   release n_tokens = 62852
    task 8352   prompt eval     4 tokens   release n_tokens = 34     <- the probe
    task 8389   prompt eval 18668 tokens   release n_tokens = 63141  <- the bill

`llama-probe.service` started at 21:19:28 and reported `ok 391` at 21:19:46.
It asks the model one question whose answer is known, every ten minutes, and
it goes DIRECTLY to llama-server — past the gateway and therefore past its
admission control. With `-np 1` its 34 tokens replace whatever is in the slot.

## What it costs, over seven days

    485   probes fired
    106   found a conversation of 2000+ tokens in the slot          (22 %)
     33   of those caused work that would not otherwise have happened
    209 587 tokens re-prefilled  ≈  24.8 minutes of GPU

The other 73 landed on a conversation and cost nothing: `-cram 32768` handed
the state back from RAM. That is the prompt cache earning its 32 GiB, and it
is also why this was never obvious — three quarters of the collisions are
invisible.

The ten most expensive, measured against the median re-prefill of an
unaffected request (206 tokens):

    2026-08-25 22:19  +31819 tokens   ≈ 321 s
    2026-08-29 21:23  +18461          ≈ 210 s
    2026-08-26 12:08  +17778          ≈ 176 s
    2026-08-29 10:20  +14845          ≈ 119 s
    2026-08-28 22:39  +14765          ≈  73 s
    2026-08-29 00:06  +14762          ≈  74 s
    2026-08-29 07:57  +14276          ≈  68 s
    2026-08-26 12:38   +7957          ≈  37 s
    2026-08-29 07:38   +7776          ≈  42 s
    2026-08-29 07:05   +6467          ≈  29 s

So: about five incidents a day, and the worst of them made someone wait five
minutes for a turn that should have taken thirty seconds.

## What the probe's own docstring says, and why it no longer holds

    "The obvious place to catch it is the gateway … It is the wrong place.
     forward() streams chunks through with no buffering …, and a detector
     there would have to parse SSE frames in the hot path of every request."

True when it was written. Since 28.08. the gateway already keeps the first and
last 8 KiB of every answer — it reads the token accounting out of them. A
degeneracy check is a character histogram over a string that is already in
hand: no parsing pass, no slot, no request.

That turns the watchdog passive for the failure it was built for. What it
cannot do passively is the arithmetic check ("the answer must contain 391"),
because foreign traffic asks its own questions — but degeneracy is the
measured, known failure mode, and "wrong but not degenerate" was always the
speculative half.

## CORRECTION, same evening: the attribution above is an upper bound

Read llama.cpp's source before trusting the number, and it does not hold as
stated. When a slot is taken over BY LRU — which is what the probe does —
server-context.cpp:1631 runs `prompt_save()` on the outgoing prompt before
`prompt_load()` brings in the new one. The evicted conversation goes into the
RAM prompt cache, and the next turn gets it back from there.

So the eviction is normally absorbed, and the arithmetic of the flagship
incident says so too:

    21:19:44  previous turn total          60184 tokens
    21:23:23  next turn total              63069, of which 44401 reused

If the probe had never run, the slot would have held those 60184 tokens — and
the longest common prefix with the new prompt is 44401 whatever holds it. The
18668 recomputed tokens are therefore NOT the probe's doing: the client sent a
conversation that diverges from its own previous one at token 44401.

That happens regularly and has nothing to do with the watchdog. Today alone,
in this trace:

    08:14:44   in 22688, reused 12435, previous total 21419
    21:12:15   in 44979, reused 14568, previous total 44405
    21:23:23   in 63069, reused 44401, previous total 60184

A client that rewrites its own history — compaction, a truncated tool result,
a re-ordered turn — costs exactly this, and the probe was standing next to it.

WHAT SURVIVES THE CORRECTION. The probe does take the one slot; that part is
directly observed. Its cost is the restore from RAM (sub-second) rather than a
re-prefill, EXCEPT when the RAM cache no longer holds the state. That happens
under pressure, and the server says so: `making room for prompt cache entry,
removing oldest entry` appears 18 times in seven days — all of them on 26.08.,
none since.

So the honest figure is not "24.8 minutes of GPU a week". It is: the probe
takes the slot about five times a day, is normally absorbed by a 32 GiB RAM
cache that was under pressure on exactly one day of seven, and the 209,587
tokens counted above are an upper bound that mixes it with client-side
divergence. Separating the two needs one more number per request — the common
prefix against the PREVIOUS prompt — which the gateway can record and does
not yet.

## What is NOT settled

* Whether a passive check finds the `////` signature reliably, and how often
  it would fire on healthy output. Both are measurable against recorded
  traffic before anything is built.
* Whether an active probe is still wanted for idle machines. When the machine
  is idle the slot is empty, so probing then costs nothing — the two are not
  in conflict.

## Reproduce

    python3 bench/suites/probe-cost.py 7
