# GLM-4.7-Flash-30B-A3B against the incumbent, and against the candidate

**Measured 04.09.2026**, all of it on this machine, all of it the same day,
`platform_profile=performance` verified at both ends of the sweep.

> **The recommendation is at the bottom and it is: do not switch to this
> model.** Not because anything failed — it works, and it is faster than the
> incumbent on the floor — but because a faster and better-scoring candidate
> was measured on this same machine seven hours earlier, and this one loses to
> it on nine of nine comparable cells but one.

## What was asked, and what changed while it was being answered

The commission was "integrate GLM-4.7-Flash and measure it, maximum
throughput at sufficient quality". Mid-run the purpose was restated: the model
is wanted as a **sparring partner from a different family**, roughly as fast
as qwen36 or faster. That does not change any number below; it changes which
comparison decides. Against flashnext this model is a clear win on the floor.
Against qwen36 — the standard the operator actually named — it is not.

## 1 · Does it load? Yes, and it cost no build at all

GLM-4.7-Flash carries MLA attention and DeepSeek-style MoE routing, so
llama.cpp maps it onto the existing graph: `conversion/glm.py:249` registers
`Glm4MoeLiteForCausalLM` with `model_arch = DEEPSEEK2`. Checked three ways
**before a byte was downloaded**, over range requests against the published
GGUF headers:

* the file declares `general.architecture = deepseek2`;
* `strings bin/libllama.so | grep -c '^deepseek2$'` is 1 in the pinned build,
  the house build and build-vulkan alike;
* `src/models/deepseek2.cpp` builds a full nextn graph with MLA, so the
  model's own MTP head is servable with no sidecar.

**The MTP head is the qwen36 trap again, and worse.** The checkpoint has one
(`num_nextn_predict_layers: 1`) and BOTH obvious publishers drop it — ggml-org
and unsloth each ship 844 tensors, `block_count 47`, no `nextn` tensors. Two
third-party files keep it (meshllm's Q4_K_M, jacek2024's Q8_0: 868 tensors,
`block_count 48`, all six `blk.47.nextn.*`, and `token_embd.weight`, which is
the seconds-long check `defects.json` exists for).

## 2 · Memory: two pairs, and the head doubles the cache

Two-point method, each pair agreeing with itself on the base:

|                          | -c 65536 | -c 202752 | KiB/token | base    | KV at full window |
|--------------------------|---------:|----------:|----------:|--------:|------------------:|
| no speculation           |    33.00 |     40.20 |  **55.0** |   29.56 |             10.64 |
| `draft-mtp,ngram-mod`    |    37.77 |     52.16 | **110.0** |   30.90 |             21.26 |

**Exactly double.** llama.cpp's MTP draft context allocates a second,
full-size KV cache. For scale, all measured here: qwen36 21.0 KiB/token,
flashnext 37.3, qwen38 74.3. Even at 55.0 this is the second most expensive
cache in the repo and 2.6x qwen36's — and that difference is what the decode
numbers at depth are made of.

The derivation from the config (52.9 KiB/token, from MLA's single K buffer
over 47 layers) is **4 % under** the no-MTP measurement — the closest a
derivation has come on this machine. It read as 52 % under for one hour,
against the wrong pair.

## 3 · The MTP head drafts well and costs throughput

`prose` decode, t/s, six cells at -c 65536 — the workload no drafter can help,
and the only column where every cell came back without a spread warning:

| variant        | d512 | d8192 | d36k |
|----------------|-----:|------:|-----:|
| rocm-nospec    | 29.6 |  25.3 | 17.3 |
| vulkan-nospec  | 45.7 |  34.3 | 19.2 |
| rocm-ngram     | 29.9 |  23.5 | 17.0 |
| vulkan-ngram   | 45.1 |  29.8 | 18.7 |
| rocm-mtp       | 29.7 |  24.3 | 15.9 |
| vulkan-mtp     | 39.7 |  25.0 | 11.2 |

The head **accepts 82.2 / 59.4 / 78.1 %** of its drafts under ROCm and is
worth nothing there and **-13 / -27 / -42 %** under Vulkan. On qwen36 the same
mechanism at LOWER acceptance (45-49 %) was worth 15-29 % and was the largest
lever of that day. A drafter that accepts three quarters of what it writes and
still loses is one whose VERIFICATION costs what it saves; this model's KV is
2.6x qwen36's per token, which is the obvious candidate and is **not
established** — nothing here isolates it.

**Vulkan wins the floor by 1.54 / 1.36 / 1.11x** and, unlike qwen36, ROCm does
not take it back anywhere reliable. The Vulkan arm is handicapped twice (an
eight-day-older build, no RADV APU heap option on this machine), so that
margin is a floor.

**19 of the sweep's 54 cells carry the spread warning, and all 19 are
speculated count or copy cells** — `rocm-ngram` ranged 30.2-264.6 t/s over
three runs on one of them. On this model the speculated ceiling is not a
quantity this instrument can resolve. The floor is.

## 4 · The quant, after a false start worth recording

Q4_K_M first measured 1.26-1.60x the Q8_0 on prose. It was not a quant result:
`copy` reported 3.1 / 7.9 / 0.0 % of the answer actually copied against the
Q8's 96.2 / 100.0 / 91.7, the n-gram drafter accepted 72.0 % on NOVEL text
against the Q8's 0.0 %, and identical prompts tokenised 16 tokens longer.

**meshllm's Q4 has no `tokenizer.chat_template` key at all** — not empty,
absent — and llama-server does not refuse: it substitutes a built-in default
and serves. Supplying the Q8's template through `--chat-template-file` returns
everything: token counts exactly the Q8's, copy 88.9 / 96.2 / 88.9 %, n-gram
acceptance 0.0 %. Filed as `glm47flash-q4-gguf-has-no-chat-template`,
`shows_as: silent`.

The real comparison, one variable, on the warning-free column:

| prose | Q8_0 | Q4_K_M | Q4/Q8 |
|-------|-----:|-------:|------:|
| d512  | 45.6 |   63.3 |  1.39 |
| d8192 | 33.8 |   36.1 |  1.07 |
| d36k  | 14.0 |   21.7 |  1.55 |

plus 28.9 GiB of GTT against 41.7. **This does not repeat qwen36's finding**,
where Q8 cost 5-8 % and fidelity was nearly free. Same machine, same day, same
instrument — so "doubling the weights is nearly free on this box" is a
property of that model, not of the box.

## 5 · The three models, all measured today

`glm47flash` = Q4_K_M, Vulkan, n-gram only, -c 202752, GTT 28.9 GiB.
`flashnext` = production, measured live at 17:12 against port 8080.
`qwen36` = MTP-Q4 ROCm, its own serving shape, measured 11:52.
`!` marks speed.py's spread warning — that cell's median compares to nothing.

| cell        | flashnext | qwen36  | glm47flash | GLM/FN | GLM/q36 |
|-------------|----------:|--------:|-----------:|-------:|--------:|
| prose d512  |     20.4  |   44.9  |   **63.3** |  3.10  |   1.41  |
| prose d8192 |     19.6  |   37.8  |     36.1   |  1.84  |   0.96  |
| prose d36k  |     15.0  |   37.3  |     21.7   |  1.45  |   0.58  |
| count d512  |    110.3! |  265.0! |    237.0!  |   —    |    —    |
| count d8192 |    104.1  |  261.8  |     36.9!  |   —    |    —    |
| count d36k  |     81.4  |  212.7  |     25.1!  |   —    |    —    |
| copy d512   |    107.7! |  171.7! |     80.1!  |   —    |    —    |
| copy d8192  |    101.9  |  263.8  |     80.0!  |   —    |    —    |
| copy d36k   |     83.6  |  215.1  |   **30.4** |  0.36  |   0.14  |
| GTT GiB     |     91.6  |   29.0  |     28.9   |        |         |

Prefill, prose: GLM 617.0 / 441.8 / **140.4** against flashnext's 193.7 /
285.1 / **207.2** — it wins shallow and **loses at d36k**.

**Read the two warning-free speculated cells together with the prose row, not
instead of it.** `copy d36k` is the shape of most of what an agent emits, and
there GLM is 0.36x the incumbent and 0.14x qwen36. The prose advantage is
real; it is also the only column where this model leads at depth.

## 6 · depth-correctness: green, 24 of 24

`bench/reports/2026-09-04_1646_depth-correctness_glm47flash-q4-vulkan-ngram`.
Every anchor (first, middle, most recent) plus the tool definitions, at six
depths to **99,113 tokens**, on the exact serving shape. The upstream failure
pattern this suite exists for — correct when shallow, confabulated tool
definitions past ~29k — does not appear.

**What that settles and what it does not:** it rules out a BROKEN quant, not a
WORSE one. A file whose calibration is subtly off still finds a needle it can
see. meshllm publishes no imatrix, so the provenance point stands
undiminished.

## 7 · What is NOT measured here

Quality. This repo removed its model battery on 26.08. and that decision
stands. The figures below are **published vendor numbers, read 04.09.2026, not
measured here, and not strictly comparable across sources**:

| (published) | params | GPQA-D | LiveCodeBench v6 | MMLU-Pro | AIME | SWE-bench V |
|---|---|---:|---:|---:|---:|---:|
| flashnext | 125B / 6B | 91.7 | 91.9 | — | — | — |
| qwen36 | 35B / 3B | 86.0 | 80.4 | 85.6* | 92.7 | 73.4 |
| **GLM-4.7-Flash** | 30B / 3B | **75.2** | **64.0** | — | 91.6 | **59.2** |

\* from NVIDIA's own harness, which measured Nemotron and Qwen 3.6 side by
side — the only like-for-like row in the table.

## The recommendation

**Do not switch to GLM-4.7-Flash, and do not adopt it as the sparring
partner.** Three reasons, in order of weight:

1. **It is slower than the candidate already on this machine**, at the depths
   that matter. `prose` at d36k: 21.7 against qwen36's 37.3. `copy` at d36k —
   the only warning-free speculated cell — 30.4 against 215.1. The cause is
   measured and structural: 55.0 KiB/token of KV against 21.0, on a machine
   where decode waits for bandwidth.
2. **Its biggest lever does not work here.** The MTP head drafts well and
   costs throughput, so the 15-29 % that made qwen36 fast is not available.
3. **The published figures put it below qwen36 on every shared axis.** For a
   critic, that is not a detail — a sparring partner that judges worse than
   the model it is reviewing produces noise, not disagreement.

**What it IS good for, and it is not nothing:** against the incumbent it is
1.45-3.10x on prose at **28.9 GiB of GTT against 91.6**. If the question were
"replace flashnext with something small", this would be a candidate. It is not
the question that was asked.

**Where to look next:** `gemma26` — Gemma 4 26B-A4B — is already on this disk,
already a registered profile, and has **never been measured with this
instrument**. It is the strongest non-Qwen model of this class on published
numbers (GPQA-D 82.3, LCB v6 77.1, MMLU-Pro 82.6, AIME26 88.3), it is
QAT-q4_0 rather than a third-party quant, and Google publishes a purpose-built
draft model for it — `gemma-4-26B-A4B-it-assistant`, 0.30 GiB at Q4_0, whose
`embedding_length_out` of 2816 matches the installed main model's
`gemma4.embedding_length` exactly, and whose architecture (`gemma4-assistant`)
is present in both this machine's artifacts. That is the same lever the MTP
head was for qwen36, for 0.3 GiB instead of 17. Nothing about it has been
measured; the file is downloaded and parked.

---

## 8 · The verdict above was measured against the wrong standard — revised 17:50

**Written after the operator corrected a premise this report was built on.**
Everything above the line stands as measured; what changes is which comparison
decides, and the recommendation reverses.

**The premise that was wrong.** This report treated flashnext as the incumbent
to displace and qwen36 as the candidate competing for the same seat, so
GLM-4.7-Flash was judged on "is it better than qwen36 at qwen36's job". That
is not the arrangement the operator wants. **qwen36 IS the production model**
— switched at 17:44, see below — and the second model is wanted BESIDE it as a
sparring partner from a different family, not instead of it.

**What that changes.** "Slower than qwen36" stops being disqualifying and
becomes expected: a critic does not carry the interactive load. The questions
that decide become different ones, and the measurements above answer most of
them:

| the question the role actually asks | the measurement | verdict |
|---|---|---|
| genuinely different family? | Z.ai, MLA + DeepSeek routing, own tokenizer and template | **yes** — nothing shared with Qwen but the GGUF format |
| fast enough to review with? | prose 63.3 / 36.1 / 21.7 t/s | **yes** — a review is prose over a deep prompt, and d8192 is 36.1 |
| does it fit beside production? | 28.9 GiB against qwen36's 27.8, GTT cap 108 | **yes, 51.4 GiB spare** — but see the open question below |
| correct at depth? | depth-correctness 24 of 24 to 99,113 tokens | **yes** |
| does it judge well enough? | published only, NOT measured here | **unknown, and the weakest point** |

**The revised recommendation: GLM-4.7-Flash is a viable sparring partner, and
it is not the first one to try.** Two reasons, and only the second is about
GLM:

1. **`gemma26` is the stronger candidate for this role and costs nothing to
   find out.** Gemma 4 26B-A4B is already installed, is QAT-q4_0 rather than a
   third-party quant with no imatrix, and on published numbers it beats
   GLM-4.7-Flash on every shared axis — GPQA-D 82.3 against 75.2,
   LiveCodeBench v6 77.1 against 64.0. For a critic that gap is the whole
   point: a sparring partner that judges worse than the model it reviews
   produces noise rather than disagreement, and GLM sits below BOTH qwen36 and
   Gemma 4 on every published axis they share. It also has an unpulled lever —
   Google's own 0.30 GiB draft model, already downloaded and verified
   compatible.
2. **GLM's own weak point is the one this repo cannot measure.** Its speed is
   adequate for the role and its correctness at depth is green; what is in
   question is judgement quality, and the only evidence is vendor numbers that
   place it last of the three.

So: keep `glm47flash` registered and measured — it is a working, characterised
option and the profile carries every number beside the flag it justifies — and
measure `gemma26` before choosing between them.

**THE OPEN QUESTION THIS REPORT CANNOT ANSWER, and it is an architectural one
rather than a measurement.** "Beside" has two readings and they cost very
different things:

* **switchable** — two registered profiles, `switch-model.sh` between them.
  Works today, costs nothing, and only one model answers at a time.
* **simultaneous** — both served at once, on two ports, so a session can ask
  either. The memory is there (56.6 GiB of 108, 51.4 spare) and the host RAM
  is tight but workable (two `-cram 32768` ceilings against 124.9 GiB). What
  is NOT there is the arrangement: `llama-user@.service` carries a
  `Conflicts=` line naming every model precisely so a second server cannot
  start, because both want port 8080 and the loser is invisible —
  `setup/lib/models.sh`'s header is about that exact failure. Two models means
  a second port, a gateway that routes to it, and a memory guard that knows
  about both. None of it exists.

That decision belongs to the operator and nothing here presumes it.

## 9 · Production changed at 17:44 — qwen36 serves

On the operator's explicit instruction, `bash setup/switch-model.sh qwen36`.
The seven-step preflight passed, including the memory budget and the
`Conflicts=` check. Verified from three independent angles rather than from
`is-active`, which cannot tell which of two started servers won the port:

* `models.sh serving` reads the process command line: **qwen36**
* `models.sh enabled`: **qwen36**; `llama-user@flashnext` inactive
* the running server's own argv names
  `Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf` at `-c 262144`, GTT 27.75 GiB against
  the profile's declared 27.71

flashnext's saved prefixes were parked in `~/.cache/llama-slots.flashnext` by
step 3/7. `bash setup/switch-model.sh flashnext` is the way back.

**One discrepancy, recorded rather than smoothed:** RssAnon of the production
server reads **0.38 GiB** where `setup/env/qwen36.env` declares 0.29. The
declared figure is the largest of four SIDE-SERVER loads, which run without
the gateway and without a warm `-cram`; the production shape is more. The
direction is the harmless one for the guard's purpose only because the number
is small — 0.09 GiB — but the profile's claim that 0.29 is "the largest" is no
longer true, and the note beside it says so now.
