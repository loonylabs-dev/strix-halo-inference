# checkpoint-grid — a changed agent prompt costs one ubatch, and one flag fixes it

Question (operator, 31.08.): agent setups send many calls that share the head
and differ in the task text — does every one of them pay the ~2k-token
checkpoint rounding measured on live DSH traffic, and can it be avoided?

- model: qwen38 (production profile), side server at `-c 32768`, `-np 1`
- build: `b10702-11-gc799f1014`
- suite: `bench/suites/checkpoint-grid.py` under `bench/sideserver.py`, with a
  second gateway (production env, `AUTO_SAVE=0`, `LLAMA_URL` at the side
  server) in front — the direct Anthropic endpoint drops OpenAI-format tools
  silently (2,309 of 8,251 tokens rendered), so the gateway path is the only
  faithful one
- bodies: the operator's real DSH requests from the 14:26 trace (head
  8,074 tokens = system + 25 tools; start prompts differing by a few words;
  the harness's small judge call in between), sequence `a,j,a,j,b,j,c` —
  the shape production traffic actually had

```
                                 b: changed start prompt   c: changed again
default (checkpoint-min-step 8192)  reused 6201  computed 2050   6203 / 2052
--checkpoint-min-step 512           reused 8074  computed  177   8074 /  181
```

**The control reproduces production exactly** (live 14:30: reused=6201,
computed=2050) — deterministically, on a fresh server, twice in one run.

**The history is the trigger, not the change alone.** A changed prompt sent
STRAIGHT after its predecessor reuses to the exact divergence (reused=8074,
computed=177) even on the default grid — measured twice while building the
sequence. The rounding appears only after the production-shaped history of an
identical repeat plus judge calls: every reprocessing lays fresh end
checkpoints, and the min-step eviction (default 8,192 tokens) then prunes the
checkpoint at the user-message boundary as "too close". After that prune the
best rollback point left sits one ubatch (2,048) before the end of the
previous prompt — which is exactly where reused lands.

**`--checkpoint-min-step 512` removes the effect completely**: the boundary
checkpoint survives, both changed prompts reuse to the divergence, 11.6x less
prefill work per task switch (2,050 -> 177 tokens; the prefill share of the
wall time drops from ~10 s to under a second — total `took` is dominated by
generation and noisy).

**What it costs, and this is NOT measured:** more surviving checkpoints. One
checkpoint of this model measures 149.8 MiB (defect registry,
the-previous-answer entry), capacity is 32 per slot, and the RAM prompt cache
stores checkpoints with each prompt. Before the flag goes into
`setup/env/qwen38.env`, watch the `-cram` budget under real traffic — the
flag buys latency with memory.

**Applied to production the same afternoon** (operator's go, ~15:30):
`--checkpoint-min-step 512` in `setup/env/qwen38.env`, restart, flag verified
in the process argv, and the same sequence run against the live gateway:
`a` cold 8,249 → repeat `computed=4` → changed prompt **reused=8074,
computed=177**. Memory watch baseline: RssAnon 0.45 GiB right after the
restart, cache empty — compare after a real working day.

Raw gateway START/DONE lines for both cells: `gateway-done-lines.log` beside
this file; the numbers above are copied from them verbatim.
