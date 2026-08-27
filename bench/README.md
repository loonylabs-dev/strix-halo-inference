# bench — how a model is judged here

New models arrive faster than anyone can form an opinion about them. This
directory exists so the opinion is not needed: the same battery, the same
probes, the same report layout for every candidate, so two of them can be
put next to each other without arguing about anecdotes.

## What one run can and cannot decide (26.08.2026)

> **Record, not instruction.** This section is about the task battery, removed
> on 26.08. together with the coding battery — "pass rates" and
> `hard-count-numbers` no longer exist here. The REASONING is why
> `bench/speed.py --reps` defaults to 3 today, and it was confirmed on 27.08.
> in the new instrument: the same decode cell measured 44.6 and 135.1 t/s six
> minutes apart.

`--reps` defaulted to 1 back then, and the model decision of 24./25.08. was
made that way. The chain of measurements that says why that is not enough, in
order:

**1 · The battery is not repeatable.** `hard-count-numbers`, same build, same
configuration, same `seed: 7`, four times: FAIL at 228 s, then PASS at 243,
161 and 284 s, writing 4386 to 6578 output tokens.

**2 · It is not the sampler.** Three reps at `temperature: 0` — where greedy
decoding is supposed to give byte-identical output — produced 5700 / 8192 /
7212 tokens. (`extra_body` really does reach every task request; that was
checked, not assumed.)

**3 · The server is not generally non-deterministic.**
`bench/suites/determinism.py` asks a fixed prompt N times and compares byte for
byte. On the production server, with speculation on:

    140-token answer, warm          5 of 5 identical
    140-token answer, slots erased  5 of 5 identical
    1449-token answer, warm         5 of 5 identical
    4759-token transcription, warm  3 of 3 identical

So the KV/cache state alone changes nothing, and length alone does not either.

**4 · It bites at near-ties.** The three reasoning answers from step 2 are
byte-identical for their first 1107 characters and then split — at a
semantically neutral phrasing choice:

    rep 1   "the digit 0 can be part of the SET, but it cannot be the first digit"
    rep 2   "the digit 0 can be part of the NUMBER, but it cannot be the first digit"

rep 1 and rep 3 stay together for 2112 characters before splitting elsewhere.
That is the whole phenomenon: a transcription's next token wins by a mile and
stays identical for thousands of tokens; a reasoning step is full of two-way
coin flips, a difference in the logits' last bits decides one of them, and
from there the two runs write different texts. Which also means the effect is
NOT visible in throughput or in a short answer — only in tasks that reason.

**What follows for reading a report.** A difference of ONE task between two
single runs is not evidence. It was treated as evidence once — qwen38's 9/9
against Laguna's 8/9. The rest of that decision stands on the timings, which
were not close; that particular margin does not. For a decision, use
`--reps 3` and compare pass RATES.

**5 · It is speculative decoding.** `bench/suites/spec-determinism.py` starts
two side servers differing in exactly one flag — the rest of their arguments
are read out of `setup/env/qwen38.env` rather than restated — and runs the same
reasoning task three times against each:

    spec-on    3 distinct answers, split at character 1107   150 / 206 / 177 s
    spec-off   3 of 3 byte-identical, 7164 tokens each       826 / 827 / 827 s

Character 1107 is the SAME divergence point the production server produced, so
it is one specific near-tie in this prompt and the draft acceptance is what
decides it. Speculation is also worth 4-5x on this task, which is the whole
reason production runs it.

> **4-5x here and 13.7-17.2x in bench/speed.py is not a contradiction — it is
> the whole point.** This one is a REASONING task: the model composes novel
> text, the drafter is often wrong, and 4-5x is near the floor. speed.py's
> `copy` workload is the ceiling, where the answer is already in the prompt.
> Any single multiplier for "what speculation is worth" is a statement about a
> workload, never about a configuration.

### What to do with that

There is a real tension and it should not be resolved by picking one number:

| what you want to know | how to run it |
|---|---|
| is the answer RIGHT | speculation **off**. Then one run is one answer, and a difference between two models is a difference. |
| how FAST is it in production | speculation **on**, `--reps 3`, compare medians. This is the number a user feels. |

Doing both is two runs, and the second one is the cheap one. What must not
happen is a single spec-on run being read as a statement about correctness —
that is exactly what produced qwen38's 9/9 against Laguna's 8/9.


## Is the backend still RIGHT at depth? (26.08.2026)

llama.cpp #27579 — open — reports gfx1151/ROCm producing corrupted output
where Vulkan with identical weights and flags does not, and names Qwen3.8
specifically: "correct at shallow depths but fails past ~29k tokens,
confabulating tool definitions". qwen38 runs here with `-c 204800` and Claude
Code's tool head alone is ~43k, so that would not be a corner case, it would be
every deep session.

`suites/depth-curve.py` could not answer it — it measures what depth COSTS.
`suites/depth-correctness.py` measures whether the answer is still right:
anchors with verifiable values planted every 4k tokens, queried at three
relative positions per depth, plus a block of invented tool names, each
question asked twice at temperature 0.

Measured on qwen38, ROCm, patched b10631, `-np 1`, f16 KV:

    depth    first    middle   last     tools
     4062     ok       ok       ok       ok
    19950     ok       ok       ok       ok
    35835     ok       ok       ok       ok
    51720     ok       ok       ok       ok
    67609     ok       ok       ok       ok
    99390     ok       ok       ok       ok

Clean to 99k, every repeat stable. The upstream report does not reproduce in
this configuration — which is consistent with the report itself: it notes that
a 512-expert MoE passed all its tests and suspects `mul_mat` on dense models,
and Qwen3.8-27B is a hybrid MoE.

**What this does NOT say.** It is a retrieval probe. It would catch a backend
that garbles output or confabulates a tool list; it would not catch a quiet
loss of reasoning quality at depth. And it was run in one configuration — the
Vulkan comparison in the docstring stays the instrument for the day something
here does go wrong.

## Can we have two slots back? (26.08.2026, partly)

The production profile runs `-np 1` because of two measured defects on
gfx1151 — both in `setup/defects.json`, with the measurement that found them.
Both upstream issues are still open, so
the build update from b10577 to b10631 was not expected to change anything.
It did:

    bench/suites/np2-candidates.py rocm-patched+cram+mmproj
    four fresh server starts, 24 of 24 answers clean

    b10577, same cell:  clean on one start, CORRUPT 3/6 on the next

Defect 1 no longer reproduces. `-np 1` stays regardless, and the reasons are
worth keeping straight: the cell runs `-c 32768` against production's 204800
and the race is timing-dependent; it covers nothing of defect 2, the slot
RESTORE with two slots, which was CORRUPT 4/4 with a populated store; and four
clean starts against an intermittent SILENT failure raise the odds without
making an argument. The remaining sequence is in setup/patches/README.md.

## What this measures — and what it deliberately does not

**Not model intelligence.** A coding battery lived here until 26.08. and was
removed: other people benchmark models at more scale and with more scrutiny
than a home-grown battery can survive, such a battery ages the week a new
model lands, and every number it produces is an invitation to an argument this
project has no stake in.

What is measured here is the **stack on this hardware**, which nobody else
measures:

    what fits            weights, KV per token, at a given window
    how fast             prefill and decode, cold and warm, per setup
    still correct?       whether the answers stay right as the window fills,
                         across slots, and after a restore — properties of the
                         BUILD and the FLAGS, not of the model

The third one is not a quality question in disguise. The failures it catches —
degenerate output from the gfx1151 HIP race, a poisoned slot restore — are
faults of this machine and this build, and they present as worse answers with
no error anywhere. `setup/defects.json` lists them; the suites here are how
they were found.

## The one metric that decides

**Seconds until a verified-correct answer.** Pass rate first — a fast wrong
answer is worth nothing — then the median time to get there. Everything
else (t/s, token counts, memory) explains *why* a configuration lands where
it lands, but decides nothing on its own.

That is why every task carries a machine checker: generated code is
executed against tests, SQL against a reference result, extraction against
expected fields. No human reads an answer to score it, so a rerun three
weeks from now means the same thing.

## A new model arrived — the path

    # 1 · get it served. A profile per model, one line per decision:
    cp setup/env/qwen38.env setup/env/<model>.env      # then edit and explain
    ln -sfn "$PWD/setup/env/<model>.env" ~/.claude/env/<model>.env
    systemctl --user start llama-user@<model>

    # 2 · what does it COST here? Prefill and decode per KV depth, read off
    #     the server's own clock — not wall time, which puts prefill inside
    #     the decode number (llama.cpp #27623 made exactly that mistake).
    #     THREE decode workloads, and together they BRACKET the range:
    #       prose  novel text — the FLOOR, no drafter of any kind can help
    #       count  predictable output — ceiling for a trained draft head
    #       copy   a block reproduced with one edit — ceiling for an n-gram
    #              drafter, and the shape of most of what an agent emits
    #     Measured 27.08. on qwen38: 9.3 / 124.3 / 160.0 t/s at ~620 tokens.
    #     Speculation is worth 13.7-17.2x here — and no single one of those
    #     three numbers is "the decode rate".
    python3 bench/speed.py --label <model> --depths 512,8192,32768
    #     --reps defaults to 3, and that is not caution: the same cell
    #     measured 44.6 and 135.1 t/s six minutes apart on 27.08. The
    #     median is the number, tg_min/tg_max is whether to believe it.
    #
    #     A configuration that wins one column and loses the other is a
    #     result, not a contradiction. Reading only `count` is how this repo
    #     recorded ngram-mod at 8.5 t/s against 8.6 and concluded "the ngram
    #     drafters give nothing", from a probe that could not have shown one
    #     working. The copy cell also checks that the model ACTUALLY copied
    #     and marks the number when it did not.
    #
    #     "What is it GOOD at" is deliberately not asked here any more:
    #     the coding battery was removed on 26.08. Other people benchmark
    #     model quality with more scale than a home-grown battery survives.
    #     See docs/MODELS.md.

    # 3 · which configuration is the best one for THIS machine?
    #     (backend x speculation x thinking level — the axes that moved
    #      the numbers by 2x here; write the reasoning into the _warum field)
    cp bench/variants/qwen38.json bench/variants/<model>.json
    # paths in it use @HOME@ and @MODELS@ — the same two placeholders the
    # profiles use, expanded by setup/lib/systemdfile.py. Do not write an
    # absolute one: it ties the sweep to one machine, and to one checkout.
    python3 bench/sweep.py --variants bench/variants/<model>.json \
                          --restore <currently-serving-model>

    # 4 · how does it behave as the window fills? (GPU must be idle)
    python3 bench/suites/depth-curve.py

Steps 2 and 4 need nothing but a running server. Step 3 stops and restores
the service by itself and is the expensive one (hours, depending on how
many cells the variant file holds).

## What each tool answers

| Tool | Question |
|---|---|
| `sweep.py` | Which server configuration wins — runs a whole variant matrix, restores the service afterwards, tolerates a failing cell |
| `suites/depth-curve.py` | Prefill and decode as the context fills. These diverge far more than any flat number suggests |
| `suites/depth-correctness.py` | Whether the answers stay RIGHT as the window fills — a property of the build, not of the model |
| `suites/stock-vs-patched.py` | Whether the patched build is measurably better than an official binary at the setting production runs |
| `suites/slot-corruption.py` | Which ingredient makes this build emit `////` — one variable per case, fresh server each |
| `suites/np2-candidates.py` | Backend and flag combinations on a SIDE server (port 8081), so production keeps running while the question "can we have two slots yet?" gets measured |
| `suites/restore-safety.py` | Whether a slot restore is safe in a given state |
| `compare.py` | The decision table across the variants of one sweep |
| `run.py` | The cache suites (cold/warm, tool turns, multi-project) |

`quality.py` and `tasklib.py` are still here, and only as FIXTURES: two of the
correctness suites need a request body shaped like Claude Code's and a question
whose answer a machine can check. They are no longer a model battery and are no
longer the thing any decision rests on — see the note at the top.

## Never start a model beside production by hand

    python3 bench/sideserver.py --env setup/env/<model>.env --port 8081 \
        --stop llama-user@qwen38 -- <command to run while it is up>

It stops the unit, waits for GTT to actually fall, refuses if the weights do
not fit, and tears everything down in a `finally` that waits again before
restarting production.

This machine was taken down twice on 26.08.2026 by measurements. The second
time the guard already existed — `check_room_for()` and
`wait_for_gtt_release()` had been written that morning after the first — and a
throwaway script simply did not call them: `kill; sleep 5`, then an 87 GiB
model on top of an allocation still being released. `user.slice` peaked at
114.8 GiB, swap was fully consumed, and the box had to be power-cycled.

GTT is not swappable. An over-large start does not page, it hangs the machine.

## Rules that keep the numbers worth something

- **Never compare across power profiles.** Everything in `docs/` was
  measured on `balanced`; `sweep.py` writes the active profile into every
  `context.json` so a mismatch is visible later.
- **The GPU must be idle** for speed measurements. A second session sharing
  the slots halves the rates and nothing in the output says so.
- **Flags belong with the number.** Every report carries the full argv and
  the build id — a t/s value without them cannot be reproduced.
- **A missing measurement is a gap, not a zero.** `measure.py` aborts
  rather than computing a rate from an answer that carries no accounting. That rule exists because a silently invented
  "-0.0 %" once travelled into the documentation as a finding.
- **Speculation makes decode workload-dependent.** Always look at both ends
  of the range: free prose is the floor (drafts get discarded, it can even
  cost a little), repetitive output the ceiling. One number in the middle
  describes nothing.

## One edit to the records, on 27.08.2026, written down here

Reports are not edited. This was, once, and the exception is documented rather
than quiet.

`variant.json` and `context.json` recorded the absolute path of the binary a
sweep used — `/home/<a person>/llama.cpp/build-rocm/bin/llama-server`. In
fourteen files it was folded to `@HOME@`, which is the token
`setup/lib/systemdfile.py` expands everywhere else in this repository.

**No measurement was touched.** Not a number, not a timestamp, not a result.
The reason it was safe is that the field is redundant: what identifies a
binary is the build stamp, and the report records that separately —

    "build":  "b200-54ee5ee"      <- what it WAS
    "binary": "@HOME@/llama.cpp/build-rocm/bin/llama-server"   <- where it sat

The path reproduces nothing. Its only effect on a published repository was to
name whoever ran the sweep.

`bench/sweep.py` was fixed at the same time, and that was the actual bug: it
expanded `@HOME@` before recording, so every FUTURE report would have carried
the same path. It now expands to RUN and records unexpanded, which also makes
a report readable on a machine that is not the one it was taken on.

## Where the results live

    bench/reports/<stamp>_<kind>_<model>/     raw data, one directory per run
    docs/MODELS.md                            the decision that was made from them

Reports are committed. They are the evidence behind every claim in `docs/`,
and a claim whose evidence is gone is just an assertion.
