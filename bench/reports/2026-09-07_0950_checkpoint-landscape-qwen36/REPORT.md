# state lost on qwen36 — the plateaus are checkpoints, and three explanations are dead

Question (operator, 07.09.2026): the trace showed `state lost` twice that
morning on long-running requests. What is it, where does it come from, and how
do we fix it for qwen36?

- production: qwen36, `-np 1`, `-c 262144`, `-ub 512`, `-cram 32768`,
  `--spec-type draft-mtp,ngram-mod`, build `b10750-11-g1559ab34f` (the pinned
  checkpoint-fix build)
- instruments written for this: `bench/ckptsidecar.py` (reads the checkpoint
  sidecar beside a saved slot state), `bench/suites/checkpoint-landscape.py`,
  `bench/suites/checkpoint-landscape-compare.py`
- gate green before the work (1305 in 18.3 s, uncontaminated) and after

## What `state lost` means

`tools/traceui.html` paints a row red when the client's history came back
UNCHANGED (`msgs_kept == msgs_prev`) and the server still reused less than 90 %
of the previous request's input. It separates a cache the machine dropped from
a history the client rewrote — the distinction that cost two rounds of blame on
29.08.

## The cause is not inferred — the plateaus ARE checkpoint positions

That morning's `reused` values fall back to a small set of repeating numbers:
7902, 29730, 40919, 52977, 70563, 109992, 161192. The checkpoint sidecar of the
live session (`session-c36336bd1680.bin.ckpt`, read with `bench/ckptsidecar.py`)
holds ten checkpoints, at

    7902  29730  40919  52977  70563  109992  183389  187242  187712  187754

Every plateau is a checkpoint position. The slot rolls back to a context
checkpoint; that much is settled and needs no experiment.

The mechanism is `create_checkpoint` (server-context.cpp:2467): on each new
task, checkpoints closer than `--checkpoint-min-step` (default 8192) to an
earlier SURVIVING one are erased, unless they belong to the current task. The
sidecar shows both halves at once — old survivors spaced 11k-73k apart, and the
current task's still-spared tail at +3853, +470, +42.

## What one checkpoint costs on this model

From the same sidecar, and this is the number the qwen38 precedent does not
carry over:

    data_tgt   62.8 MiB   constant — the recurrent state of a linear-attention
                          model does not grow with position
    data_dft   15.6 MiB at 7.9k tokens ... 371.0 MiB at 188k — the MTP draft
                          model's KV cache, and it grows linearly
    data_spec   0.0 MiB

So a checkpoint is 78 MiB early and 434 MiB deep into a session. On qwen38
(8k contexts) it was a flat 149.8 MiB, which is why `--checkpoint-min-step 512`
was affordable there and is not obviously affordable here.

Measured sidecar cost against the state it accompanies:

| depth | checkpoints | sidecar | state | sidecar / state |
|---|---|---|---|---|
| 30k | 7 | 0.70 GiB | 0.64 GiB | **+110 %** |
| 104k | 16 | 2.86 GiB | 2.04 GiB | **+40 %** |
| 190k (live session) | 10 | 2.66 GiB | 3.71 GiB | **+72 %** |

This answers an open item in the baton ("the sidecar costs +53 % … unmeasured
at production depth"): at production depth it is between +40 % and +110 %, and
at 104k the checkpoints are LARGER than the slot state they belong to.

Two readings from the source that bound the RAM worry:

* the checkpoints travel INTO the RAM prompt cache — `server_prompt` carries
  them (server-task.h:569), so a displaced slot keeps them;
* `server_prompt_cache::alloc` adds `checkpoints_size` to the entry before
  testing it against the limit (server-task.cpp:1723), so `-cram` is not
  silently overshot. The cost lands as FEWER entries fitting, not as a breach.

## Four explanations, measured dead

Each cell is one `bench/sideserver.py` run with production stopped and
restored, same profile, same sequence, `--checkpoint-min-step 8192` (i.e.
today's production setting).

| cell | turns | final depth | fallbacks | checkpoints | sidecar / state |
|---|---|---|---|---|---|
| `cms8192` plain append | 24 | 30,253 | **0** | 7 | 0.70 / 0.64 GiB |
| `cms8192-intr` foreign request every 3rd turn | 24 | 30,253 | **0** | 7 | 0.70 / 0.64 GiB |
| `cms8192-deep` | 90 | 103,656 | **0** | 16 | 2.86 / 2.04 GiB |
| `cms8192-pause` 60 s idle every 2nd turn | 12 | 32,281+ | **0** | 6 | 0.69 / 0.78 GiB |

"fallbacks" counts turns that re-prefilled state the slot already held, above a
floor of 64 tokens (the chat template's per-turn residue, measured at ~19).

**A clean append never falls back.** In all 24 turns `reused` was exactly
`prev_in - 4` — llama.cpp's own accounting, the healthy `end-4` checkpoint.

**The watchdog probe is not the cause.** `llama-probe` fires every 10 minutes
straight at port 8080, past the gateway, and at `-np 1` there is no second slot
for it to land in — a session-save right after one recorded `n_saved=30`, so it
demonstrably takes the slot. It still costs nothing: the intruder itself
prefills 4 tokens and the next real turn is unaffected, because the RAM prompt
cache absorbs the displacement (`--cache-idle-slots`, on by default). The
statistical view agrees and is worth stating with its bias: across all trace
days 17 % of red rows and 9 % of healthy rows overlap a probe run (n=8), and
red rows run far longer, which by itself raises their chance of overlapping a
10-minute tick. No signal.

**Depth alone is not the cause.** 90 turns to 103,656 tokens, past the ~30k
threshold where production first shows the effect, 0 fallbacks.

**Idleness is not the cause either.** The one regularity that survives in the
production data is a GAP before the request: all six red rows on 07.09. had one
(20, 31, 38, 45, 178, 468 s) while healthy turns arrive 1-3 s apart, and across
all trace days the median gap before a red row is 5.6 s against 0.6 s for
healthy ones (at gaps ≥ 60 s: 25 % of red rows against 8 % of healthy ones).
Worth saying plainly: that is a signal, not a trigger — 75 % of red rows had
shorter gaps. And the cell settles it anyway: six 60-second idle periods,
turns 6 and 7 above the ~30k threshold, every turn at `reused = prev_in - 4`.

## What is left

Nothing in the SHAPE of the traffic reproduces it. What the side-server cells
do not carry is the production PATH:

* the gateway sits in front (these cells talk to llama-server directly),
  with its own queue, priority ordering and session save/restore;
* the requests stream; these do not;
* 25 tool definitions ride in the head of every production request.

Rather than guess at which of those matters, the next step measures where the
effect demonstrably happens: `LLAMA_ARG_LOG_VERBOSITY=4` as a drop-in on
`llama-user@qwen36`, which logs the checkpoint decisions themselves —

    checking checkpoint with [a, b] against N...
    restored context checkpoint (pos_min = .., n_tokens = ..)
    erasing context checkpoint too close to an earlier one

Verbosity 4 and NOT 5: `LOG_DBG` at 5 writes `converted request: <body>` — the
whole prompt — into the journal permanently. The drop-in says in its own first
line that it is temporary.

## ANSWERED, 10:31 the same morning

It took 5 minutes of the operator's normal work. The server's own account:

    selected slot by LCP similarity, f_sim_best = 0.219, f_keep = 0.029
    find_best:  - prompt with length      30, lcp =     1
                - prompt with length  128391, lcp =  3691
    checking checkpoint with [124840, 124840] against 3691...   (x12)
    erased invalidated context checkpoint (... pos_next = 0)     (x12)

The slot held 128,391 tokens of ANOTHER CONVERSATION. The arriving request
shares 3,691 tokens with it — the system prompt and the 25 tools — so every
checkpoint sits behind the common prefix and all twelve are erased as
`invalidated`. Not evicted for being too close: **no `--checkpoint-min-step`
value would have changed a single one of them.** The grid was the wrong lever
and this measurement is what says so.

The gateway trace shows the two conversations trading the one slot:

    10:28:28  c36336bd1680  reused=120256  ->  124,592
    10:29:13  086817bdc0ce  reused=0       ->   13,055   (cold)
    10:30:47  c36336bd1680  reused=124810  ->  124,845   (returns cleanly)
    10:31:19  086817bdc0ce  reused=0       ->   16,851   31.3 s, state lost
    10:33:18  086817bdc0ce  reused=0       ->   12,334   msgs_kept 1 of 6

That last row is why the cells never reproduced it: they drove ONE
conversation, and the trigger is the switch. `msgs_kept 1 of 6` under an
unchanged prefix id is not an edited history — it is a THIRD conversation
wearing the same identity, because `prefix_id` hashes only the system head and
the tool block. The cache was not under pressure at the time (4,946 of 32,768
MiB; 253k of an effective 850,757 tokens), which removes the last alternative.

Filed as `session-identity-ignores-the-conversation` in `setup/defects.json`.
The fix is not in llama.cpp.

## Reproduce

    python3 bench/sideserver.py --env setup/env/qwen36.env --port 8082 \
        --stop llama-user@qwen36 \
        --extra "-c 65536 --checkpoint-min-step 8192 --slot-save-path DIR" \
        -- python3 bench/suites/checkpoint-landscape.py \
             --label cms8192 --save-dir DIR --out cms8192.json

    python3 bench/suites/checkpoint-landscape-compare.py *.json
