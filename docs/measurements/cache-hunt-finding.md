# Cache hunt, 23.08.2026 — measurement record

A dated record, not living documentation. Nothing in it is edited when the
world moves on; where a finding here has been superseded, the newer document
says so. For the current state see [MODELS.md](../MODELS.md) and
[setup/README.md](../../setup/README.md).

> **Translated from German on 27.08.2026.** Every measured value is unchanged.
> Number formatting was converted to English convention — `1.637` became
> `1,637` and `100,2 s` became `100.2 s` — because leaving German separators in
> an English document does not preserve the number, it changes what it reads
> as. The original is kept out of this repository as
> `cache-hunt-finding.de.md`, so any figure here can be checked against it.

Continues a handover document from that session which is not part of this
repository. Every value measured on this machine, llama.cpp b10577 (Vulkan
build), Laguna S 2.1 UD-Q4_K_XL.

---

## Tools this session built

| File | Purpose |
|---|---|
| `cc-tap.py` | Recording proxy in front of `llama-server`. Stores every `/v1/messages` body raw, logs `usage` and duration. |
| `replay.py` | Replays a captured body, measures `input_tokens`/`cache_read_input_tokens`, can save the rendered prompt from `/slots`. |
| `bisekt.py` | Splits the real body into variants and measures which component breaks reuse. |
| `verifikation.py` | Measures the fix candidates against a running server configuration. |
| `analyse.py` | Structural report of a request body. |

### The decisive trick: `LLAMA_SERVER_SLOTS_DEBUG=1`

The handover proposed patching `server-context.cpp` and rebuilding in order to
see the rendered prompt. **That is not necessary.** The environment variable
`LLAMA_SERVER_SLOTS_DEBUG=1` unlocks two built-in diagnostics:

1. `GET /slots` returns a `prompt` field — the complete, actually rendered
   prompt from the `/v1/messages` path.
2. On a prefix miss the server prints the tokens **around the divergence
   point**, with `|` as the marker:

       old: ...  nur das Wort |  alpha.</user>
       new: ...  nur das Wort |  beta.</user>

That costs no build time and does not depend on `-lv 4`.

---

## The rendering is fine

Two real Claude Code requests, byte-identical except for the question,
compared through `/slots`:

    length A               80,600 characters
    common prefix          73,522 characters   =  91.2 %
    remainder               7,078 characters

    structure:  <system> at 7 · ### Tools at 6,098 · </available_tools> at 72,890
                </system> at 72,908 · <user> at 72,918 · </user> at 73,528

The server confirms it itself with `f_sim_best = 0.915`. The divergence sits
exactly at the user question. **The suspicion from the handover — that the
adapter renders differently from `/apply-template` — is refuted.**

---

## The cause: Laguna has sliding window attention

From the GGUF metadata of `Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf`:

    laguna.attention.sliding_window        512
    laguna.rope.freq_base_swa              10000.0
    laguna.rope.dimension_count_swa        128

**The correction of 22.08. in the runbook is wrong.** It states that Laguna's
GGUF header contains neither `sliding_window` nor `sliding_window_pattern`. It
contains `laguna.attention.sliding_window = 512`.

The consequence in `server-context.cpp`: if the divergence point is further
than `n_swa` tokens from the end of the prompt, the KV state before it can no
longer be reconstructed. The server then falls back to a context checkpoint —
and finds none, because ordinary mid-prompt checkpoints are skipped (they are
created only at user message boundaries and shortly before the prompt ends).
The result:

    forcing full prompt re-processing due to lack of cache data
    (likely due to SWA or hybrid/recurrent memory)

Claude Code appends a `system` message of ~1,640 tokens behind the user
question ("Available agent types for the Agent tool: …"). That puts every
changed question around 1,640 tokens from the end — far outside the 512-token
window.

---

## The bisection

Real Claude Code body, variants built, two runs per variant, the second is
measured (only the question changed).

    variant                            run 2
    ------------------------------------------------------------------
    V0 unchanged (control)             new=19,371 cache=0       100.2 s
    V1 without tools                   new= 3,173 cache=0        15.7 s
    V2 without the system message      new=     7 cache=17,734    0.3 s  <== works
    V3 system field as a string        new=19,371 cache=0       100.1 s
    V4 user content as a string        new=19,227 cache=0        99.3 s
    V5 without metadata                new=19,371 cache=0        99.8 s
    V6 without cache_control           new=19,371 cache=0       100.2 s
    V7, V8                             not measured — GPU device loss

Ruled out by this: the size of the tool block (V1 fails at only 3,173 tokens),
the array-shaped `system` field (V3), the array-shaped `user` content (V4),
`metadata` (V5) and the `cache_control` markers (V6 — which llama.cpp ignores
entirely; not a single hit in the source tree outside HTTP headers). What
decides is solely whether anything comes AFTER the user question (V2).

### The numbers that close the circle

Determined exactly through `/tokenize`:

    system field                               6,081 characters =  1,378 tokens
    tool block, 24 schemas                    66,764 bytes      = 16,789 tokens
    system message AFTER the user question     7,028 characters =  1,624 tokens
    counter message                               49 characters =     18 tokens

So the divergence sits 1,624 tokens from the end of the prompt while the SWA
window is 512 tokens wide — a factor of 3.2 too far.

With `cc-cachefix.py` in its current form the changing counter moves to
position 1,378 + 1,624 = 3,002. That explains the `f_sim_best = 0.112` from
the previous session's logs (0.112 × 26,421 ≈ 2,959) to within 1.5 %.

### Synthetic counter-check

Same order of magnitude, but the question is at the end of the prompt:

    S1 synthetic 26,507 tokens, cold    new=26,507 cache=0         114.7 s
    S2 repeated identically             new=     1 cache=26,506      0.1 s
    S3 changed question                 new=     7 cache=26,500      0.2 s
    S4 back to the first question       new=     7 cache=26,500      0.2 s

Partial prefix reuse works perfectly on this build, this model and these flags
— as long as the divergence is inside the 512-token window.

---

## Why `cc-cachefix.py` works — and why not for tool conversations

The proxy pulls all `system` messages out of `messages` into the `system`
field. Side effect: the user question slides to the end of the prompt, that is
**into** the 512-token window. That is the real mechanism — not the "a second
system message throws the rendering off" suspected in § 14.

For tool conversations it reverses. From turn 2 onward Claude Code appends
**another** `system` message per turn, and it contains a counter:

    tool-006 (turn 1)  1 system message
    tool-007 (turn 2)  + <total_tokens>14981262 tokens left</total_tokens>
    tool-008 (turn 3)  + <total_tokens>14981135 tokens left</total_tokens>
    tool-009 (turn 4)  + <total_tokens>14981007 tokens left</total_tokens>

The proxy pulls those forward too. The counter changes on every turn, is
therefore **in front of** the 66 KB tool block — and invalidates it entirely.
Deduplication in the proxy does not help, because the text differs every time.

This matches the `f_sim_best = 0.112` from the previous session's logs:
0.112 × 26,421 ≈ 2,960 tokens of common prefix — exactly the length of the
system text plus the hoisted agent-types block, which is where the changing
counter starts.

---

## The fix: `--swa-full`

`llama-server` has a `--swa-full` switch — it allocates the KV cache of the SWA
layers at full context length instead of only window-wide. The state is then
reconstructible at any position and partial reuse works again.

Measured against the **unchanged** Claude Code body, with no proxy at all:

    without --swa-full   A2 changed question   new=19,371  cache=     0  ( 0.0 %)  100.2 s
    with    --swa-full   A2 changed question   new= 1,637  cache=17,734  (91.5 %)   10.4 s

A factor of 9.6. The 1,637 newly processed tokens are exactly the 1,624-token
block behind the question plus the question itself — that is, precisely what
lies behind the divergence point. That confirms the explanation quantitatively.

**Memory:** GTT 73.2 → 82.8 GiB of 96. About 9.6 GiB extra for 2 × 65,536
tokens. The cold-start prefill does not change (101.1 s against 99.8 s), and
neither does model behaviour — SWA layers still only look into the window,
there is simply more of it kept.

### Context checkpoints are NOT enough

Gemma 4 26B already runs with `-ctxcp 64 -cms 4096` per `gemma26.env`. In the
same experiment (real Claude Code body, changed question) the cache still fails
completely:

    gemma26 with -ctxcp 64 -cms 4096   f_sim_best = 0.916   -> full re-run

The reason is in the source: ordinary checkpoints in the middle of the prompt
are skipped. They are created only at user message boundaries and shortly
before the end of the prompt — and the ones near the end are discarded again
during the roll-back (`erased invalidated context checkpoint`).

### The SWA situation across the house

From the GGUF metadata of every installed model:

    model         architecture   sliding_window   note
    ------------------------------------------------------------------
    gemma-4-26B   gemma4         1024             5 of 6 layers SWA
    gemma-4-31B   gemma4         1024             5 of 6 layers SWA
    Laguna S 2.1  laguna          512
    gpt-oss-120b  gpt-oss         128             smallest window
    Qwen3.8-27B   qwen35          none            the only one unaffected

Four of five models are affected. Qwen3.8-27B is the only one that caches
cleanly with no special handling — which makes it interesting as a fallback.

### Upstream

This is not a local problem. Open issues in the llama.cpp project with the same
message, as of August 2026:

    #21831  Server forces full prompt re-processing (SWA/recurrent memory)
    #22746  Qwen 3.6 27B forcing full prompt re-processing
    #20225  Qwen 3.5 full prompt re-processing on every conversation turn
    #20153  Qwen3.5 27B, lack of cache data
    #19794  Qwen3-Coder-Next hybrid, despite --swa-full
    #19394  Qwen3-Coder-Next

The reference in the source is PR #13194. For hybrid/recurrent models (Mamba)
`--swa-full` does not help per #19794; Laguna is pure SWA MoE, where it does —
measured here.

---

## The full verification matrix

Laguna S 2.1 with `--swa-full`, real captured request bodies, replayed directly
against `llama-server`. `new` = newly processed tokens, `cache` = reused.

    A · Simple case, same request, only the question changed, NO proxy
       A1 alpha (fills the slot)     new=19371  cache=    0  ( 0.0 %)  101.1 s
       A2 beta  (changed)            new= 1637  cache=17734  (91.5 %)   10.4 s

    B · Tool conversation, 4 turns, NO proxy
       B1 turn 1 (cold)              new=19443  cache=    0  ( 0.0 %)  101.5 s
       B2 turn 2                     new=  207  cache=19443  (98.9 %)    2.0 s
       B3 turn 3                     new=  111  cache=19650  (99.4 %)    1.5 s
       B4 turn 4                     new=  112  cache=19761  (99.4 %)    1.5 s

    C · Tool conversation WITH cc-cachefix.py in its current form
       C1 turn 1 (cold)              new=19438  cache=    0  ( 0.0 %)  101.3 s
       C2 turn 2                     new=16634  cache= 3007  (15.3 %)   89.2 s
       C3 turn 3                     new=16722  cache= 3026  (15.3 %)   90.0 s
       C4 turn 4                     new=16811  cache= 3045  (15.3 %)   90.4 s

    D · Tool conversation WITH the first version of the corrected proxy
       identical to B — 207 / 111 / 112 new, 2.0 / 1.6 / 1.5 s

    E · Simple case WITH the first version of the corrected proxy
       identical to A — 1637 new, 10.4 s

**Block C is the evidence** that `cc-cachefix.py` in its current form CAUSES
the problem that § 14 attributed to it as "unsolved": 90 seconds per turn
instead of 1.5. Four turns come to 371 s — the documented "~300 s".

The 3,007 cached tokens correspond exactly to the system field (1,378) plus the
hoisted agent-types block (1,624) = 3,002.

### A bug in the first version of the replacement

Blocks D and E fell back with the first version of `cc-cachefix2.py` to exactly
the values without a proxy. The reason: Claude Code appends the counter **to
the end of the otherwise stable agent-types block** —

    block length 7,028 characters, counter at characters 6,979 to 7,028

— and the first version therefore classified the *whole* block as volatile and
did not hoist it at all. The second version separates them: volatile matches
are cut out and stay in place as their own message, the stable remainder
(6,977 characters) moves to the front.

---

## Several projects, and the question of persistence

### Two projects stay warm at the same time

Real Claude Code, `--swa-full`, `cc-cachefix2.py`, `-np 2`. Two different
working directories, `lanewise` with its own `CLAUDE.md`:

    1. /tmp/cc-jagd, slot warm            1.4 s
    2. lanewise, first time             107.1 s   <- cold start of the new prefix
    3. lanewise, second question          1.4 s
    4. back to /tmp/cc-jagd               1.4 s   <- the first slot is still alive

Every project pays its cold start exactly once. The reason for the cold start
is already in section 13 of the state document: Claude Code writes the working
directory into the system prompt (from character 2,782 of 6,336), and a
`CLAUDE.md` comes on top. A different project is therefore a different prefix.

**The number of simultaneously warm projects is `-np`.** With
`--no-kv-unified`, `-c` is divided evenly across the slots, so the memory stays
constant — only the context per project drops (`-c 131072 -np 2` = 65,536 per
slot).

### Slot persistence: works, but is no good as a project cache

The finding from section 11.4 of the state document ("`--slot-save-path` does
not save the cache") is **partly refuted**. Measured with `--swa-full`:

    save        19,371 tokens, 546,240,884 bytes     152 ms
    restore                                           72 ms
    identical request afterwards   new=    1 cache=19,370 (100.0 %)    0.1 s
    changed question afterwards    new=19,371 cache=     0 (  0.0 %)  101.1 s

So restoring carries — but only for a byte-identical prompt. As soon as a
roll-back would be needed, everything fails (`f_sim_best = 0.915`, and a full
re-run regardless).

**The cause is in the file size.** Laguna has 48 layers, 12 of them with
`head_count = 48` and 36 with `head_count = 72`. The arithmetic for q8_0:

    KV per token, all 48 layers         102 KiB
    only the 12 global layers            26 KiB
    measured in the slot file            28 KiB

So the slot file contains only the global layers plus the window, not the full
SWA cache — not even with `--swa-full`. The restored state is therefore usable
exactly as far as an SWA state without `--swa-full` is: good for a direct hit,
useless for partial reuse.

**Practical consequence:** per-project persistence across a server restart buys
nothing, because the next real request always contains a different question.
What carries is the running server with enough slots.

### What would still be worth measuring

- **`-np 4`**: whether four projects stay warm at once. The finding in section
  12 ("from three agents onward the cache fails completely") predates
  `--swa-full` and is very probably obsolete — the failure there is the same
  roll-back problem.
- **`-cram`**: the RAM cache (32 GiB configured) holds evicted slot states.
  Whether it carries more projects than there are slots is unmeasured.

---

## Addendum: four projects at once — the finding in section 12 is obsolete

Laguna with `--swa-full`, `-np 4`, `-c 131072 --no-kv-unified` (so 32,768
tokens per slot). Four project prefixes built from the real Claude Code body by
replacing the working directory at its two positions in the system prompt
(characters 2,538 and 4,670 of 6,081).

    Phase 1 · warm them up one after another
       P1   new=19368  cache=0   99.7 s
       P2   new=19368  cache=0  100.5 s
       P3   new=19368  cache=0  100.3 s
       P4   new=19368  cache=0  100.3 s

    Phase 2 · round robin, same question, two rounds
       R1 P1..P4   each  new=1  cache=19,367  (100.0 %)  0.1–0.3 s
       R2 P1..P4   each  new=1  cache=19,367  (100.0 %)  0.1 s

    Phase 3 · round robin, changed question
       P1..P4      each  new=1637  cache=17,731  (91.5 %)  10.3–10.5 s

    GTT before 82.6 GiB · after 82.7 GiB · at the end 82.6 GiB

**Eight of eight requests at 100 %, four of four at 91.5 %** — the same values
as a single project with `-np 2`. The finding in section 12 of the state
document ("three and four agents — does not work, the cache fails completely")
was the same roll-back problem and is settled by `--swa-full`.

### The KV supply is one fixed pot

`--no-kv-unified` divides `-c` evenly across the slots. The memory therefore
depends on `-c`, not on `-np`:

    -c 131072  ->  about 14 GiB of KV  (112 KiB per token, measured)
    + 68.35 GiB of weights             =  82.7 GiB of 96 GiB GTT

A second pot does not fit: `-c 262144` would need about 96.8 GiB. So `-np` only
decides how the pot is cut:

    -np 2   2 projects warm at once   65,536 tokens per project
    -np 4   4 projects warm at once   32,768 tokens per project

At 32,768 only about 13k remains for the actual work after the 19.4k system
prompt — Claude Code then compacts correspondingly often.
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` must be set to the same value, or a longer
session overruns the slot.
