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

## What is NOT settled

* Whether a passive check finds the `////` signature reliably, and how often
  it would fire on healthy output. Both are measurable against recorded
  traffic before anything is built.
* Whether an active probe is still wanted for idle machines. When the machine
  is idle the slot is empty, so probing then costs nothing — the two are not
  in conflict.

## Reproduce

    python3 bench/suites/probe-cost.py 7
